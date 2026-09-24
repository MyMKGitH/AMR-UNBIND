from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from rdkit import Chem

from .errors import DockingError
from .progress import DOCKING_FRACTION, Heartbeat

# Where "docking" sits in the overall pipeline's progress budget -- imported
# directly from amr_unbind/progress.py's single source of truth (see that
# module for the full budget and why it exists as one shared table rather
# than each file hardcoding its own numbers). Local alias kept so the rest
# of this file reads naturally.
DOCKING_STAGE_FRACTION = DOCKING_FRACTION


@dataclass(frozen=True)
class DockingResult:
    pose_sdf: Path
    pose_pdbqt: Path
    score_kcal_mol: float | None
    method: str
    executable: str | None = None
    receptor_prep_log: Path | None = None


def _meeko_command(module: str, *args: str) -> list[str]:
    """Build a Meeko CLI invocation in the active Python environment."""
    return [sys.executable, "-m", module, *args]


_MEEKO_BAD_RESIDUE_FLAG_CACHE: str | None = None


def _meeko_bad_residue_flag() -> str:
    """Return whichever flag this installed Meeko version uses to skip --
    rather than hard-fail on -- a residue whose local bonding pattern
    doesn't match any known chemical template.

    This is the single most common real-world receptor-preparation failure
    for an experimental structure: an unresolved (missing-residue) loop
    means two residues that are not actually bonded end up adjacent in the
    file, and Meeko's own residue-chain builder can geometrically infer a
    spurious extra bond across that gap -- surfacing as "No template
    matched for residue_key=... matched with excess inter-residue
    bond(s)". prepare_protein() already inserts a TER record at every such
    gap it detects (see prepare.py's _insert_ter_at_gap_boundaries), which
    is enough for tools that determine chain breaks from TER records the
    way OpenMM's PDB parser does -- but Meeko's own PDB/mmCIF reader does
    not appear to treat every TER as reliably, in practice, so a real
    residue this exact shape (confirmed against PDB 1STP, a well-known
    structure with a partially-disordered binding loop) can still reach
    this hard failure even with the TER record present. This flag is
    Meeko's own documented, intended way to handle that: the affected
    residue is dropped from the receptor model rather than the whole run
    failing -- see _prepare_pdbqt_inputs' docstring for why that's a
    reasonable default here.

    Meeko renamed this flag from `--allow_bad_res` (`-a`) to
    `--delete_bad_res` at some point after 0.8.0 (see
    https://github.com/forlilab/Meeko/pull/414); rather than hardcode a
    guess that could silently go stale the next time requirements-docking.txt's
    meeko pin changes, which name the *installed* version actually accepts
    is checked once (from that version's own --help output) and cached.
    """
    global _MEEKO_BAD_RESIDUE_FLAG_CACHE
    if _MEEKO_BAD_RESIDUE_FLAG_CACHE is not None:
        return _MEEKO_BAD_RESIDUE_FLAG_CACHE
    try:
        result = subprocess.run(
            _meeko_command("meeko.cli.mk_prepare_receptor", "--help"),
            capture_output=True,
            text=True,
            timeout=30,
        )
        help_text = f"{result.stdout}\n{result.stderr}"
    except Exception:
        help_text = ""
    flag = "--delete_bad_res" if "--delete_bad_res" in help_text else "--allow_bad_res"
    _MEEKO_BAD_RESIDUE_FLAG_CACHE = flag
    return flag


def _run_command(
    command: list[str],
    description: str,
    *,
    cwd: Path | None = None,
    log_output_to: Path | None = None,
    progress_callback=None,
    heartbeat_stage: str | None = None,
    heartbeat_fraction: float = 0.0,
    heartbeat_note: str = "",
) -> subprocess.CompletedProcess[str]:
    """Run `command`, optionally reporting a Heartbeat (see progress.py) for
    its whole duration. Pass heartbeat_stage to enable it -- without it,
    this behaves exactly as before, silently, for callers (like the Vina
    docking subprocess paths) that already have their own heartbeat
    wrapping the whole attempt rather than this one command.
    """
    heartbeat = (
        Heartbeat(progress_callback, heartbeat_stage, heartbeat_fraction, note=heartbeat_note)
        if heartbeat_stage is not None
        else None
    )
    try:
        if heartbeat is not None:
            heartbeat.__enter__()
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                cwd=str(cwd) if cwd is not None else None,
            )
        except FileNotFoundError as exc:
            raise DockingError(f"{description} could not be launched: executable or module not found.") from exc
        except PermissionError as exc:
            raise DockingError(f"{description} could not be launched: permission denied.") from exc
        except subprocess.CalledProcessError as exc:
            if log_output_to is not None:
                _write_command_log(log_output_to, command, exc.stdout, exc.stderr)
            stdout = (exc.stdout or "").strip()
            stderr = (exc.stderr or "").strip()
            details = stderr[-1500:] or stdout[-1500:] or f"exit code {exc.returncode}"
            raise DockingError(f"{description} failed: {details}") from exc
    finally:
        if heartbeat is not None:
            heartbeat.__exit__(None, None, None)
    if log_output_to is not None:
        _write_command_log(log_output_to, command, result.stdout, result.stderr)
    return result


def _write_command_log(path: Path, command: list[str], stdout: str | None, stderr: str | None) -> None:
    """Best-effort: write a subprocess's full stdout/stderr to disk for later
    inspection, e.g. to see exactly which residues (if any) Meeko's
    --delete_bad_res/--allow_bad_res flag silently excluded from a
    receptor. Never raises -- a failure to write this log should not turn
    into a run failure of its own.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "$ " + " ".join(command) + "\n\n--- stdout ---\n" + (stdout or "") + "\n--- stderr ---\n" + (stderr or ""),
            encoding="utf-8",
        )
    except Exception:
        pass


def _validate_docking_inputs(
    center_angstrom: tuple[float, float, float],
    box_size_angstrom: tuple[float, float, float],
    exhaustiveness: int,
) -> None:
    import math

    if len(center_angstrom) != 3 or not all(math.isfinite(float(x)) for x in center_angstrom):
        raise DockingError("Docking box center must contain three finite coordinates in Angstrom.")
    if len(box_size_angstrom) != 3 or not all(math.isfinite(float(x)) and float(x) > 0 for x in box_size_angstrom):
        raise DockingError("Docking box size must contain three positive finite dimensions in Angstrom.")
    if int(exhaustiveness) < 1:
        raise DockingError("Vina exhaustiveness must be at least 1.")


def _prepare_pdbqt_inputs(
    receptor_pdb: Path, ligand_sdf: Path, output_dir: Path, progress_callback=None
) -> tuple[Path, Path, Path]:
    receptor_pdb = Path(receptor_pdb)
    ligand_sdf = Path(ligand_sdf)
    if not receptor_pdb.exists():
        raise DockingError(f"Receptor PDB not found: {receptor_pdb}")
    if not ligand_sdf.exists():
        raise DockingError(f"Ligand SDF not found: {ligand_sdf}")

    output_dir.mkdir(parents=True, exist_ok=True)
    receptor_prefix = output_dir / "receptor"
    receptor_pdbqt = output_dir / "receptor.pdbqt"
    ligand_pdbqt = output_dir / "ligand.pdbqt"
    receptor_prep_log = output_dir / "meeko_receptor_prep.log"

    # Meeko is the chemistry-aware preparation layer. Do not attempt to create
    # PDBQT by text filtering or by hand-assigned atom types.
    #
    # --delete_bad_res/--allow_bad_res (see _meeko_bad_residue_flag) tells
    # Meeko to drop a residue it cannot confidently type rather than fail
    # the whole receptor -- the common case for an experimental structure
    # with an unresolved loop near the binding site. Meeko's own stdout/
    # stderr (which names exactly which residues, if any, were dropped)
    # is always written to receptor_prep_log, whether or not the run
    # otherwise succeeds, so that exclusion is never silent even though
    # it no longer stops the run.
    receptor_prep_result = _run_command(
        _meeko_command(
            "meeko.cli.mk_prepare_receptor",
            "--read_pdb",
            str(receptor_pdb),
            "-o",
            str(receptor_prefix),
            "-p",
            _meeko_bad_residue_flag(),
        ),
        "Meeko receptor preparation",
        log_output_to=receptor_prep_log,
        progress_callback=progress_callback,
        heartbeat_stage="docking",
        heartbeat_fraction=DOCKING_STAGE_FRACTION,
        heartbeat_note="preparing receptor (Meeko)",
    )
    _run_command(
        _meeko_command(
            "meeko.cli.mk_prepare_ligand",
            "-i",
            str(ligand_sdf),
            "-o",
            str(ligand_pdbqt),
        ),
        "Meeko ligand preparation",
        progress_callback=progress_callback,
        heartbeat_stage="docking",
        heartbeat_fraction=DOCKING_STAGE_FRACTION,
        heartbeat_note="preparing ligand (Meeko)",
    )

    if not receptor_pdbqt.exists() or receptor_pdbqt.stat().st_size == 0:
        raise DockingError(f"Meeko did not produce a valid receptor PDBQT: {receptor_pdbqt}")
    if not ligand_pdbqt.exists() or ligand_pdbqt.stat().st_size == 0:
        raise DockingError(f"Meeko did not produce a valid ligand PDBQT: {ligand_pdbqt}")
    del receptor_prep_result  # captured only so log_output_to always runs; nothing else needed from it here
    return receptor_pdbqt, ligand_pdbqt, receptor_prep_log


def _export_pose_to_sdf(pose_pdbqt: Path, pose_sdf: Path) -> None:
    try:
        from meeko import PDBQTMolecule, RDKitMolCreate
    except ImportError as exc:
        raise DockingError(
            "Meeko is required to reconstruct Vina PDBQT poses as SDF. "
            "Install the optional docking dependencies."
        ) from exc

    try:
        pdbqt_mol = PDBQTMolecule.from_file(str(pose_pdbqt), skip_typing=True)
        mols = RDKitMolCreate.from_pdbqt_mol(
            pdbqt_mol,
            only_cluster_leads=False,
            keep_flexres=False,
        )
        if not mols or mols[0] is None:
            raise DockingError("Meeko could not reconstruct an RDKit molecule from the Vina pose.")
        mol = Chem.Mol(mols[0])
        if mol.GetNumConformers() < 1:
            raise DockingError("Reconstructed Vina pose has no 3D coordinates.")
        Chem.SanitizeMol(mol)
        writer = Chem.SDWriter(str(pose_sdf))
        if writer is None:
            raise DockingError(f"Could not open SDF output for writing: {pose_sdf}")
        writer.write(mol)
        writer.close()
    except DockingError:
        raise
    except Exception as exc:
        raise DockingError(f"Could not convert Vina PDBQT pose to SDF with Meeko: {exc}") from exc

    if not pose_sdf.exists() or pose_sdf.stat().st_size == 0:
        raise DockingError(f"Pose conversion completed without a usable SDF: {pose_sdf}")


def _parse_vina_score(pose_pdbqt: Path) -> float | None:
    pattern = re.compile(r"^REMARK VINA RESULT:\s*([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)")
    try:
        for line in pose_pdbqt.read_text(encoding="utf-8", errors="replace").splitlines():
            match = pattern.match(line.strip())
            if match:
                return float(match.group(1))
    except Exception:
        return None
    return None


def _executable_candidates(project_root: Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    configured = os.environ.get("AMR_UNBIND_VINA", "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())

    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    for relative in (
        Path("bin") / "vina",
        Path("bin") / "vina.exe",
        Path("vina"),
        Path("vina.exe"),
    ):
        candidates.append(root / relative)

    for name in ("vina", "vina.exe"):
        resolved = shutil.which(name)
        if resolved:
            candidates.append(Path(resolved))

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = os.path.normcase(os.path.abspath(os.fspath(path)))
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def find_vina_executable(project_root: Path | None = None) -> Path | None:
    """Return the first usable Vina executable candidate, if any."""
    for candidate in _executable_candidates(project_root):
        try:
            result = subprocess.run(
                [str(candidate), "--version"],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (FileNotFoundError, PermissionError, subprocess.SubprocessError, OSError):
            continue
        version_text = f"{result.stdout or ''}\n{result.stderr or ''}".lower()
        if "vina" in version_text:
            return candidate
    return None


def _dock_with_python(
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    pose_pdbqt: Path,
    center_angstrom: tuple[float, float, float],
    box_size_angstrom: tuple[float, float, float],
    exhaustiveness: int,
    seed: int,
    progress_callback=None,
) -> float | None:
    try:
        from vina import Vina
    except ImportError as exc:
        raise DockingError("Python Vina bindings are not installed.") from exc

    try:
        v = Vina(sf_name="vina", cpu=0, seed=int(seed), verbosity=0)
        v.set_receptor(rigid_pdbqt_filename=str(receptor_pdbqt))
        v.set_ligand_from_file(str(ligand_pdbqt))
        v.compute_vina_maps(center=list(center_angstrom), box_size=list(box_size_angstrom))
        # The exhaustiveness/box-size combination below is what actually
        # determines how long this call takes -- from seconds to, on a
        # heavily-loaded free-tier CPU, a very long time -- and Vina's
        # Python bindings give no intermediate callback of their own, so
        # this is one single blocking call from AMR-UNBIND's point of
        # view. A Heartbeat here is the only way anything watching
        # progress can tell "still docking" from "hung" while it runs.
        with Heartbeat(
            progress_callback,
            "docking",
            DOCKING_STAGE_FRACTION,
            note=f"AutoDock Vina (Python), exhaustiveness={int(exhaustiveness)}",
        ):
            v.dock(exhaustiveness=int(exhaustiveness), n_poses=1)
        v.write_poses(pdbqt_filename=str(pose_pdbqt), n_poses=1, overwrite=True)
        try:
            energies = v.energies(n_poses=1)
            if len(energies):
                return float(energies[0][0])
        except Exception:
            pass
        return _parse_vina_score(pose_pdbqt)
    except Exception as exc:
        raise DockingError(f"Python AutoDock Vina docking failed: {exc}") from exc


def _dock_with_executable(
    executable: Path,
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    pose_pdbqt: Path,
    center_angstrom: tuple[float, float, float],
    box_size_angstrom: tuple[float, float, float],
    exhaustiveness: int,
    seed: int,
    progress_callback=None,
) -> float | None:
    command = [
        str(executable),
        "--receptor",
        str(receptor_pdbqt),
        "--ligand",
        str(ligand_pdbqt),
        "--center_x",
        f"{float(center_angstrom[0]):.6f}",
        "--center_y",
        f"{float(center_angstrom[1]):.6f}",
        "--center_z",
        f"{float(center_angstrom[2]):.6f}",
        "--size_x",
        f"{float(box_size_angstrom[0]):.6f}",
        "--size_y",
        f"{float(box_size_angstrom[1]):.6f}",
        "--size_z",
        f"{float(box_size_angstrom[2]):.6f}",
        "--exhaustiveness",
        str(int(exhaustiveness)),
        "--num_modes",
        "1",
        "--seed",
        str(int(seed)),
        "--out",
        str(pose_pdbqt),
    ]
    _run_command(
        command,
        f"AutoDock Vina executable ({executable})",
        progress_callback=progress_callback,
        heartbeat_stage="docking",
        heartbeat_fraction=DOCKING_STAGE_FRACTION,
        heartbeat_note=f"AutoDock Vina (executable), exhaustiveness={int(exhaustiveness)}",
    )
    if not pose_pdbqt.exists() or pose_pdbqt.stat().st_size == 0:
        raise DockingError(f"Vina executable completed but produced no pose file: {pose_pdbqt}")
    return _parse_vina_score(pose_pdbqt)


def automatic_vina_docking(
    receptor_pdb: Path,
    ligand_sdf: Path,
    output_dir: Path,
    center_angstrom: tuple[float, float, float],
    box_size_angstrom: tuple[float, float, float] = (22.0, 22.0, 22.0),
    exhaustiveness: int = 8,
    seed: int = 20260907,
    project_root: Path | None = None,
    progress_callback=None,
) -> DockingResult:
    """Dock one ligand with Vina using Python bindings with executable fallback.

    Meeko prepares the receptor and ligand because PDBQT is the required Vina
    input format. If the Python ``vina`` backend is unavailable or fails, the
    function tries a real Vina executable (configured path, project-local,
    then PATH). If both backends fail, both diagnostics are reported.
    """
    _validate_docking_inputs(center_angstrom, box_size_angstrom, exhaustiveness)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    receptor_pdbqt, ligand_pdbqt, receptor_prep_log = _prepare_pdbqt_inputs(
        Path(receptor_pdb), Path(ligand_sdf), output_dir, progress_callback=progress_callback
    )
    pose_pdbqt = output_dir / "docked_pose.pdbqt"
    pose_sdf = output_dir / "docked_pose.sdf"

    attempts: list[str] = []
    python_available = True
    try:
        import vina  # noqa: F401
    except ImportError:
        python_available = False
        attempts.append("Python Vina bindings unavailable")

    if python_available:
        try:
            score = _dock_with_python(
                receptor_pdbqt,
                ligand_pdbqt,
                pose_pdbqt,
                center_angstrom,
                box_size_angstrom,
                exhaustiveness,
                seed,
                progress_callback=progress_callback,
            )
        except DockingError as exc:
            attempts.append(str(exc))
        else:
            _export_pose_to_sdf(pose_pdbqt, pose_sdf)
            return DockingResult(pose_sdf, pose_pdbqt, score, "AutoDock Vina (Python)", receptor_prep_log=receptor_prep_log)

    executable = find_vina_executable(project_root)
    if executable is not None:
        try:
            score = _dock_with_executable(
                executable,
                receptor_pdbqt,
                ligand_pdbqt,
                pose_pdbqt,
                center_angstrom,
                box_size_angstrom,
                exhaustiveness,
                seed,
                progress_callback=progress_callback,
            )
        except DockingError as exc:
            attempts.append(str(exc))
        else:
            _export_pose_to_sdf(pose_pdbqt, pose_sdf)
            return DockingResult(
                pose_sdf,
                pose_pdbqt,
                score,
                "AutoDock Vina (executable)",
                executable=str(executable),
                receptor_prep_log=receptor_prep_log,
            )
    else:
        attempts.append("No usable Vina executable found (AMR_UNBIND_VINA, project bin/root, or PATH).")

    joined = "\n".join(f"- {message}" for message in attempts)
    raise DockingError(
        "Automatic docking could not be completed. Vina backends were attempted as follows:\n"
        f"{joined}\n"
        "Install Python Vina and/or place a compatible Vina executable at the configured/project path."
    )
