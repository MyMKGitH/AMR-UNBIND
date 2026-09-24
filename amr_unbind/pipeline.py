from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import math
import shutil

from .config import SimulationConfig
from .docking import automatic_vina_docking, DockingResult
from .errors import AMRUnbindError, InputValidationError
from .geometry import calculate_pull_geometry, count_protein_heavy_atoms_near_point
from .io import create_run_dir, write_json
from .prepare import download_pdb, prepare_ligand, prepare_protein, validate_ligand_pose, validate_residue
from .progress import (
    COMPLETE_FRACTION,
    Heartbeat,
    LIGAND_PREPARED_FRACTION,
    PROTEIN_PREPARED_FRACTION,
    SYSTEM_BUILT_FRACTION,
    safe_call,
)
from .simulation import build_system, run_simulation


def _resolve_docking_center_angstrom(
    override,
    computed_core_point_angstrom,
) -> tuple[float, float, float]:
    """Return the Vina search-box center to use for automatic docking.

    ``override`` is whatever a caller (Streamlit app, CLI, or a direct
    pipeline user) supplied as a custom docking-box center. When present, it
    takes precedence over ``computed_core_point_angstrom`` -- the geometric
    point ``calculate_pull_geometry()`` derives from ``core_chain``/
    ``core_residue`` -- so a person targeting a specific, already-known
    binding pocket is not limited to that residue-based estimate. When
    ``override`` is ``None`` (the default), behavior is unchanged from
    before this option existed: the computed core point is used.
    """
    if override is not None:
        if len(override) != 3 or not all(math.isfinite(float(x)) for x in override):
            raise InputValidationError(
                "Docking center override must contain exactly three finite "
                "Angstrom coordinates."
            )
        return (float(override[0]), float(override[1]), float(override[2]))
    return tuple(float(x) for x in computed_core_point_angstrom)


# How far a resolved docking center may sit from the protein's own
# geometry-derived core point before the cheap, no-PDB-parsing pre-filter
# below rejects it outright. This alone is not the real check -- see
# _validate_docking_center_is_near_protein's docstring for why a pure
# distance-from-one-point comparison cannot reliably tell "a legitimately
# distant pocket on this protein" from "a coordinate that happens to be a
# similar distance away but is meaningless for this structure." It exists
# only to catch egregious cases (hundreds of Angstroms off) cheaply,
# before paying for a full structure parse.
_MAX_DOCKING_CENTER_DISTANCE_FROM_CORE_ANGSTROM = 300.0

# The real check: how many real protein heavy atoms must exist within
# half the docking box's edge length of the resolved center. Reported
# failure: a docking-center override left at the app's own example
# defaults (PDB 3F2R's biotin pocket) while running a *different*
# structure (PDB 1STP) produced a center that was only ~36 A from that
# structure's own core point -- comfortably inside what an earlier,
# purely distance-based version of this check treated as acceptable (100
# A), yet nowhere near the real binding pocket (confirmed directly
# against 1STP's actual deposited coordinates: the real pocket sits ~8 A
# from the core point, not ~36). No fixed "too far" distance can catch
# that reliably, because 36 A is not an implausible distance for a real,
# somewhat-distant pocket in the abstract -- the only way to tell is to
# check whether real protein atoms actually exist there. 10 heavy atoms
# is a conservative floor: even a shallow, solvent-exposed pocket is
# lined by several residues' worth of heavy atoms within half a
# reasonably-sized search box, while open solvent has close to none.
_MIN_PROTEIN_HEAVY_ATOMS_NEAR_DOCKING_CENTER = 10


def _validate_docking_center_is_near_protein(
    docking_center_angstrom: tuple[float, float, float],
    core_point_angstrom,
    *,
    is_override: bool,
    prepared_pdb_path: Path,
    docking_box_angstrom: float,
) -> None:
    distance = math.dist(docking_center_angstrom, tuple(float(x) for x in core_point_angstrom))
    source = (
        "the docking center override you supplied"
        if is_override
        else "the auto-computed core/exit residue point"
    )
    suggestion = (
        "Docking-center coordinates are specific to one PDB structure's own coordinate frame -- "
        "a center that made sense for a different structure (including this app's own example "
        "defaults, from PDB 3F2R) is not meaningful here. Uncheck the override to use the "
        "auto-computed core/exit residue point instead, or supply coordinates you know are "
        "correct for this specific structure."
        if is_override
        else "This is unexpected for an auto-computed center and may indicate the core/exit "
        "residues themselves are not where you expect."
    )

    if distance > _MAX_DOCKING_CENTER_DISTANCE_FROM_CORE_ANGSTROM:
        raise InputValidationError(
            f"The resolved docking center {tuple(round(float(x), 2) for x in docking_center_angstrom)} "
            f"is {distance:.0f} \u00c5 from this structure's own core point "
            f"{tuple(round(float(x), 2) for x in core_point_angstrom)} -- {source} is almost "
            f"certainly not on this protein at all. {suggestion}"
        )

    radius = max(float(docking_box_angstrom) / 2.0, 5.0)
    nearby_atoms = count_protein_heavy_atoms_near_point(
        prepared_pdb_path, docking_center_angstrom, radius
    )
    if nearby_atoms < _MIN_PROTEIN_HEAVY_ATOMS_NEAR_DOCKING_CENTER:
        raise InputValidationError(
            f"The resolved docking center {tuple(round(float(x), 2) for x in docking_center_angstrom)} "
            f"has only {nearby_atoms} real protein heavy atom(s) within {radius:.1f} \u00c5 of it "
            f"(this structure's core point is {distance:.0f} \u00c5 away) -- {source} does not "
            f"appear to sit on or near this protein's actual structure. This is refused before "
            f"docking/simulation runs, not after, since a nonsensical center produces a severely "
            f"strained pose that can survive minimization only to fail later in equilibration or "
            f"SMD, wasting a long run. {suggestion}"
        )


class AMRUnbindPipeline:
    def __init__(self, work_dir: Path = Path("./amr_workspace")):
        self.work_dir = Path(work_dir)

    def run(
        self,
        *,
        pdb_path: Path | None = None,
        pdb_id: str | None = None,
        smiles: str,
        core_chain: str,
        core_residue: int,
        exit_chain: str,
        exit_residue: int,
        config: SimulationConfig,
        ligand_pose_sdf: Path | None = None,
        automatic_docking: bool = False,
        docking_box_angstrom: float = 22.0,
        docking_exhaustiveness: int = 8,
        docking_center_angstrom: tuple[float, float, float] | None = None,
        progress_callback=None,
    ) -> Path:
        try:
            config.validate()
        except ValueError as exc:
            # SimulationConfig.validate() raises a plain ValueError so it can
            # be used standalone (e.g. config.save()) without importing this
            # package's error hierarchy. Re-raise it as an AMRUnbindError here
            # so every caller of the pipeline (CLI, Streamlit app, scripts)
            # can rely on a single exception type for expected failures.
            raise InputValidationError(str(exc)) from exc
        if not smiles or not smiles.strip():
            raise InputValidationError("Ligand SMILES is required.")
        if docking_box_angstrom <= 0:
            raise InputValidationError("Docking box size must be positive.")
        if docking_exhaustiveness < 1:
            raise InputValidationError("Docking exhaustiveness must be at least 1.")
        if pdb_path is None and not pdb_id:
            raise InputValidationError("Provide either a local PDB file or a PDB ID.")
        if ligand_pose_sdf is not None and automatic_docking:
            raise InputValidationError("Choose either a supplied ligand pose or automatic docking, not both.")

        target_label = pdb_id.upper() if pdb_id else Path(pdb_path).stem
        run_dir = create_run_dir(self.work_dir, target_label)
        write_json(run_dir / "input_config.json", {
            "pdb_id": pdb_id,
            "pdb_path": str(pdb_path) if pdb_path else None,
            "smiles": smiles,
            "core_chain": core_chain,
            "core_residue": core_residue,
            "exit_chain": exit_chain,
            "exit_residue": exit_residue,
            "automatic_docking": automatic_docking,
            "docking_box_angstrom": docking_box_angstrom,
            "docking_exhaustiveness": docking_exhaustiveness,
            "docking_center_angstrom_override": (
                list(docking_center_angstrom) if docking_center_angstrom is not None else None
            ),
            "simulation_config": config.to_dict(),
        })

        try:
            if pdb_path is None:
                with Heartbeat(progress_callback, "downloading", 0.0, note="fetching PDB structure"):
                    source = download_pdb(pdb_id or "", run_dir / f"{target_label.lower()}.pdb")
            else:
                source = Path(pdb_path)
                if not source.exists():
                    raise InputValidationError(f"PDB file not found: {source}")

            prepared = run_dir / "prepared_protein.pdb"
            with Heartbeat(progress_callback, "protein-prepared", 0.0, note="preparing protein (PDBFixer)"):
                protein_result = prepare_protein(source, prepared, config.pH)
            validate_residue(prepared, core_chain, int(core_residue))
            validate_residue(prepared, exit_chain, int(exit_residue))
            if progress_callback:
                safe_call(progress_callback, "protein-prepared", PROTEIN_PREPARED_FRACTION, "complete")

            ligand_initial = run_dir / "ligand_initial.sdf"
            ligand_result = prepare_ligand(smiles, ligand_initial, config.seed)
            if progress_callback:
                safe_call(progress_callback, "ligand-prepared", LIGAND_PREPARED_FRACTION, "complete")

            geometry = calculate_pull_geometry(
                prepared,
                core_chain,
                int(core_residue),
                exit_chain,
                int(exit_residue),
            )

            pose_sdf = ligand_pose_sdf
            docking_result: DockingResult | None = None
            docking_center_used: tuple[float, float, float] | None = None
            if automatic_docking:
                docking_center_used = _resolve_docking_center_angstrom(
                    docking_center_angstrom, geometry.core_point_angstrom
                )
                _validate_docking_center_is_near_protein(
                    docking_center_used,
                    geometry.core_point_angstrom,
                    is_override=docking_center_angstrom is not None,
                    prepared_pdb_path=prepared,
                    docking_box_angstrom=docking_box_angstrom,
                )
                docking_result = automatic_vina_docking(
                    receptor_pdb=prepared,
                    ligand_sdf=ligand_initial,
                    output_dir=run_dir / "docking",
                    center_angstrom=docking_center_used,
                    box_size_angstrom=(float(docking_box_angstrom),) * 3,
                    exhaustiveness=int(docking_exhaustiveness),
                    seed=config.seed,
                    project_root=Path(__file__).resolve().parents[1],
                    progress_callback=progress_callback,
                )
                pose_sdf = docking_result.pose_sdf
            elif pose_sdf is not None:
                pose_sdf = Path(pose_sdf)
                if not pose_sdf.exists():
                    raise InputValidationError(f"Supplied ligand pose SDF not found: {pose_sdf}")
                validate_ligand_pose(pose_sdf, ligand_result.canonical_smiles)
                # Make the run self-contained and reproducible even when the
                # original upload lives outside the workspace.
                copied_pose = run_dir / "supplied_ligand_pose.sdf"
                shutil.copy2(pose_sdf, copied_pose)
                pose_sdf = copied_pose
            else:
                raise InputValidationError(
                    "A docked ligand SDF is required for an unbinding calculation. "
                    "Enable automatic docking or supply a pre-docked pose."
                )

            build = build_system(prepared, Path(pose_sdf), config)
            if progress_callback:
                safe_call(progress_callback, "system-built", SYSTEM_BUILT_FRACTION, "complete")

            result = run_simulation(
                build,
                geometry.vector,
                config,
                run_dir,
                progress_callback=progress_callback,
            )

            metadata = {
                "protein": asdict(protein_result),
                "ligand": asdict(ligand_result),
                "geometry": {
                    "pull_vector": geometry.vector.tolist(),
                    "core_point_angstrom": geometry.core_point_angstrom.tolist(),
                    "exit_point_angstrom": geometry.exit_point_angstrom.tolist(),
                    "core_exit_distance_nm": geometry.distance_nm,
                },
                "docking_center_angstrom_used": (
                    list(docking_center_used) if docking_center_used is not None else None
                ),
                "docking_center_source": (
                    "user_override"
                    if (automatic_docking and docking_center_angstrom is not None)
                    else ("auto_geometric_core" if automatic_docking else None)
                ),
                "simulation": config.to_dict(),
                "docking": asdict(docking_result) if docking_result else None,
                "build": {
                    "n_protein_atoms": build.n_protein_atoms,
                    "n_ligand_atoms": build.n_ligand_atoms,
                    "n_total_atoms": build.n_total_atoms,
                    "n_protein_backbone_restraint_atoms": len(build.protein_backbone_indices),
                },
            }
            write_json(run_dir / "run_metadata.json", metadata)
            if progress_callback:
                safe_call(progress_callback, "complete", COMPLETE_FRACTION, "")
            return run_dir
        except AMRUnbindError as exc:
            write_json(run_dir / "run_failure.json", {"error_type": type(exc).__name__, "error": str(exc)})
            raise
        except Exception as exc:
            failure = AMRUnbindError(f"AMR-UNBIND run failed: {exc}")
            write_json(run_dir / "run_failure.json", {"error_type": type(exc).__name__, "error": str(exc)})
            raise failure from exc
