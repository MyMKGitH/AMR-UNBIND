from __future__ import annotations

"""Optional cloud execution backends for AMR-UNBIND.

Local explicit-solvent OpenMM SMD is slow, and on a CPU-only or older
machine it can be unusably slow -- hours with nothing to show for it. This
module offers two independent ways to run the *same* pipeline somewhere
faster, chosen because they solve that problem for two different people:

- Modal (``submit_modal_run``): a serverless GPU backend for anyone who
  wants a "click run, get results" experience without leaving the app.
  Requires a (free-tier-eligible) Modal account, ``pip install modal``, and
  a one-time ``modal deploy amr_unbind/cloud.py`` to publish the remote
  function this module defines. After that, every run from the Streamlit
  app or the CLI is a single ``.remote()`` call.
- Google Colab (``generate_colab_notebook``): no account beyond a Google
  login and no local install at all. Produces a real, runnable ``.ipynb``
  file -- not a copy-paste snippet -- with this run's inputs already
  embedded, that installs AMR-UNBIND and a free Colab GPU runtime.

Both backends call ``amr_unbind.pipeline.AMRUnbindPipeline.run()`` -- the
exact same code a local run uses -- rather than re-implementing any
docking/simulation logic. A cloud run and a local run of the same inputs
are therefore the same computation, not two independently maintained ones,
and every failure mode the pipeline already knows how to report
(``AMRUnbindError`` and its subclasses) is reported the same way.

Nothing in ``amr_unbind/pipeline.py`` or any other core module imports this
file: cloud execution is strictly additive. Every function here degrades to
a clear, actionable ``CloudExecutionError`` -- never a silent no-op and
never a fabricated result -- when its prerequisites aren't met.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import json

from .config import SimulationConfig
from .errors import AMRUnbindError
from .progress import Heartbeat, safe_call

try:
    import modal

    MODAL_INSTALLED = True
except ImportError:
    MODAL_INSTALLED = False


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]

MODAL_APP_NAME = "amr-unbind-gpu-engine"
MODAL_FUNCTION_NAME = "run_pipeline_remote"
MODAL_VOLUME_NAME = "amr-unbind-workspace"
MODAL_REMOTE_WORK_DIR = "/vol/amr_workspace"
MODAL_VALID_GPU_TYPES = ("T4", "L4", "A10G", "A100")
MODAL_DEFAULT_GPU = "T4"
MODAL_DEFAULT_TIMEOUT_HOURS = 6.0

MODAL_DEPLOY_HINT = (
    "Run `pip install modal` (or `pip install -r requirements-cloud.txt`), "
    "then `modal token new` to authenticate, then `modal deploy "
    "amr_unbind/cloud.py` once. After that, cloud runs from the app or CLI "
    "just work."
)


class CloudExecutionError(AMRUnbindError):
    """A cloud backend (Modal or Colab notebook generation) failed."""


def modal_available() -> bool:
    """Whether the `modal` package is importable in *this* process.

    Only tells you the thin client SDK is installed here -- not whether a
    Modal account is configured, or whether `amr_unbind/cloud.py` has been
    deployed. `submit_modal_run` reports those failures separately, with an
    actionable message, rather than this function guessing at them.
    """
    return MODAL_INSTALLED


# ---------------------------------------------------------------------------
# Request/response data model
# ---------------------------------------------------------------------------


@dataclass
class CloudRunRequest:
    """Everything ``AMRUnbindPipeline.run()`` needs, serialized for a worker
    that isn't running on this filesystem -- a remote GPU container, or a
    browser-hosted Colab kernel.

    Mirrors ``AMRUnbindPipeline.run()``'s keyword arguments, but every local
    ``Path`` input becomes raw bytes plus a filename, since a local
    filesystem path is meaningless once it crosses a machine boundary.
    """

    smiles: str
    core_chain: str
    core_residue: int
    exit_chain: str
    exit_residue: int
    config: SimulationConfig
    pdb_id: str | None = None
    pdb_bytes: bytes | None = None
    pdb_filename: str | None = None
    pose_sdf_bytes: bytes | None = None
    pose_sdf_filename: str | None = None
    automatic_docking: bool = False
    docking_box_angstrom: float = 22.0
    docking_exhaustiveness: int = 8
    docking_center_angstrom: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        # Mirrors AMRUnbindPipeline.run()'s own input validation as closely
        # as possible, so a request that would be rejected locally is
        # rejected here too -- before anything is dispatched to a GPU that
        # bills by the second, or a notebook someone has to run by hand.
        if not self.smiles or not self.smiles.strip():
            raise CloudExecutionError("Ligand SMILES is required.")
        if self.pdb_id is None and self.pdb_bytes is None:
            raise CloudExecutionError("Provide either a PDB ID or an uploaded PDB file.")
        if self.pdb_id is not None and self.pdb_bytes is not None:
            raise CloudExecutionError("Provide either a PDB ID or an uploaded PDB file, not both.")
        if self.pose_sdf_bytes is not None and self.automatic_docking:
            raise CloudExecutionError("Choose either a supplied ligand pose or automatic docking, not both.")
        if self.pose_sdf_bytes is None and not self.automatic_docking:
            raise CloudExecutionError(
                "A pre-docked ligand SDF or automatic docking is required for a "
                "cloud run, exactly as for a local one."
            )
        if self.docking_box_angstrom <= 0:
            raise CloudExecutionError("Docking box size must be positive.")
        if self.docking_exhaustiveness < 1:
            raise CloudExecutionError("Docking exhaustiveness must be at least 1.")
        if self.docking_center_angstrom is not None and len(self.docking_center_angstrom) != 3:
            raise CloudExecutionError("Docking center override must contain exactly three Angstrom coordinates.")

    def to_job_dict(self) -> dict[str, Any]:
        """A plain-dict, JSON/cloudpickle-friendly form of this request."""
        return {
            "smiles": self.smiles,
            "core_chain": self.core_chain,
            "core_residue": int(self.core_residue),
            "exit_chain": self.exit_chain,
            "exit_residue": int(self.exit_residue),
            "config": self.config.to_dict(),
            "pdb_id": self.pdb_id,
            "pdb_bytes": self.pdb_bytes,
            "pdb_filename": self.pdb_filename,
            "pose_sdf_bytes": self.pose_sdf_bytes,
            "pose_sdf_filename": self.pose_sdf_filename,
            "automatic_docking": bool(self.automatic_docking),
            "docking_box_angstrom": float(self.docking_box_angstrom),
            "docking_exhaustiveness": int(self.docking_exhaustiveness),
            "docking_center_angstrom": (
                list(self.docking_center_angstrom) if self.docking_center_angstrom is not None else None
            ),
        }


# ---------------------------------------------------------------------------
# Modal backend
# ---------------------------------------------------------------------------
#
# Design notes (for anyone changing this section):
#
# - The image is built `from_dockerfile(Dockerfile.docking)` -- the same
#   Dockerfile this project already ships and tests locally -- rather than a
#   hand-picked package list. A separately maintained dependency list for
#   "the cloud version" is exactly the kind of thing that quietly drifts
#   from what actually runs locally; reusing the real Dockerfile makes that
#   impossible by construction.
# - Results are written to a `modal.Volume`, not returned directly from the
#   function call. Explicit-solvent SMD trajectories (`.dcd` files) can be
#   large, and Modal function return values are not the right transport for
#   large payloads. The remote function returns only lightweight JSON
#   metadata; `submit_modal_run` pulls the actual files from the Volume
#   afterward.
# - Calling code uses `modal.Function.from_name(...)` against an
#   already-*deployed* app, not an ephemeral `app.run()` context. Streamlit
#   request handlers are not a natural place to spin up (and tear down) an
#   ephemeral Modal app on every click; deploying once and calling a stable
#   deployed function is the supported pattern for calling Modal from
#   another long-running application.

if MODAL_INSTALLED:
    _modal_image = modal.Image.from_dockerfile(
        str(_PACKAGE_ROOT / "Dockerfile.docking"),
        context_dir=str(_PACKAGE_ROOT),
    )
    _modal_volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)
    modal_app = modal.App(MODAL_APP_NAME)

    @modal_app.function(
        image=_modal_image,
        gpu=MODAL_DEFAULT_GPU,
        timeout=int(MODAL_DEFAULT_TIMEOUT_HOURS * 3600),
        volumes={"/vol": _modal_volume},
    )
    def run_pipeline_remote(job: dict) -> dict:
        """Runs the real AMR-UNBIND pipeline on this container's GPU.

        Deliberately thin: it deserializes `job`, writes any uploaded bytes
        to local files inside the container, and calls
        `AMRUnbindPipeline.run()` -- the exact function a local run calls --
        so a cloud run cannot silently diverge from what this package's own
        tests and audits cover. Returns a small JSON-safe summary rather
        than the run's files; see the module docstring for why.
        """
        from pathlib import Path
        import sys
        import tempfile
        import traceback

        # Belt-and-suspenders in case the image's working directory or
        # PYTHONPATH doesn't already put the copied package on sys.path --
        # see Dockerfile.docking, which COPYs the repo to /opt/amr-unbind.
        if "/opt/amr-unbind" not in sys.path:
            sys.path.insert(0, "/opt/amr-unbind")

        from amr_unbind.config import SimulationConfig
        from amr_unbind.errors import AMRUnbindError
        from amr_unbind.pipeline import AMRUnbindPipeline

        tmp = Path(tempfile.mkdtemp(prefix="amr_unbind_job_"))
        pdb_path = None
        pose_path = None
        if job.get("pdb_bytes") is not None:
            pdb_path = tmp / (job.get("pdb_filename") or "input.pdb")
            pdb_path.write_bytes(job["pdb_bytes"])
        if job.get("pose_sdf_bytes") is not None:
            pose_path = tmp / (job.get("pose_sdf_filename") or "pose.sdf")
            pose_path.write_bytes(job["pose_sdf_bytes"])

        try:
            config = SimulationConfig.from_dict(job["config"])
            pipeline = AMRUnbindPipeline(work_dir=Path(MODAL_REMOTE_WORK_DIR))
            docking_center = job.get("docking_center_angstrom")
            run_dir = pipeline.run(
                pdb_path=pdb_path,
                pdb_id=job.get("pdb_id"),
                smiles=job["smiles"],
                core_chain=job["core_chain"],
                core_residue=int(job["core_residue"]),
                exit_chain=job["exit_chain"],
                exit_residue=int(job["exit_residue"]),
                config=config,
                ligand_pose_sdf=pose_path,
                automatic_docking=bool(job.get("automatic_docking", False)),
                docking_box_angstrom=float(job.get("docking_box_angstrom", 22.0)),
                docking_exhaustiveness=int(job.get("docking_exhaustiveness", 8)),
                docking_center_angstrom=tuple(docking_center) if docking_center else None,
            )
            _modal_volume.commit()
            metadata = {}
            metadata_path = run_dir / "run_metadata.json"
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            return {
                "status": "ok",
                "run_id": run_dir.name,
                "relative_run_dir": str(run_dir.relative_to(MODAL_REMOTE_WORK_DIR)),
                "metadata": metadata,
            }
        except AMRUnbindError as exc:
            return {"status": "error", "error_type": type(exc).__name__, "error_message": str(exc)}
        except Exception as exc:  # noqa: BLE001 -- this result crosses a machine boundary;
            # an unhandled exception here would otherwise surface to the
            # caller as an opaque RPC/serialization failure instead of the
            # real error.
            return {
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": f"{exc}\n\n{traceback.format_exc()}",
            }

else:
    modal_app = None
    run_pipeline_remote = None


def _lookup_modal_function():
    try:
        return modal.Function.from_name(MODAL_APP_NAME, MODAL_FUNCTION_NAME)
    except AttributeError:
        # Older `modal` SDKs exposed this as Function.lookup(...) instead of
        # the newer Function.from_name(...). Try the legacy name before
        # giving up, rather than forcing an upgrade for a rename.
        return modal.Function.lookup(MODAL_APP_NAME, MODAL_FUNCTION_NAME)


def _with_resource_overrides(fn, *, gpu: str, timeout_hours: float):
    """Apply a per-call GPU/timeout override if the installed `modal`
    version supports `Function.with_options(...)`; otherwise fall back to
    the function's deploy-time defaults (T4, 6h) and say so, rather than
    silently ignoring the values the caller asked for."""
    try:
        return fn.with_options(gpu=gpu, timeout=int(timeout_hours * 3600)), True
    except AttributeError:
        return fn, False


def _volume_entry_is_file(entry) -> bool:
    try:
        return entry.type == modal.volume.FileEntryType.FILE
    except AttributeError:
        # Fallback for SDK versions where the enum lives elsewhere: a
        # directory entry's path conventionally ends with "/".
        return not str(entry.path).endswith("/")


def submit_modal_run(
    request: CloudRunRequest,
    *,
    work_dir: Path,
    gpu: str = MODAL_DEFAULT_GPU,
    timeout_hours: float = MODAL_DEFAULT_TIMEOUT_HOURS,
    progress_callback: Callable[..., None] | None = None,
) -> Path:
    """Dispatches `request` to the deployed Modal function and syncs the
    results back into `work_dir/runs/<run_id>`, in exactly the layout a
    local `AMRUnbindPipeline.run()` produces -- so every downstream
    consumer (the Streamlit results view, the CLI, analysis scripts) works
    identically whether a run happened locally or on Modal.

    Requires `amr_unbind/cloud.py` to already have been published with
    `modal deploy amr_unbind/cloud.py` (see README.md). This function does
    not attempt to deploy it automatically: redeploying on every single run
    from inside a live Streamlit session would be slow, and would mask real
    deployment problems behind something that looks like a successful run.
    """
    if not MODAL_INSTALLED:
        raise CloudExecutionError(f"The `modal` package is not installed. {MODAL_DEPLOY_HINT}")
    if gpu not in MODAL_VALID_GPU_TYPES:
        raise CloudExecutionError(f"Unknown GPU type {gpu!r}; choose one of {MODAL_VALID_GPU_TYPES}.")
    if timeout_hours <= 0:
        raise CloudExecutionError("timeout_hours must be positive.")

    if progress_callback:
        safe_call(progress_callback, "connecting-to-modal", 0.0, "")
    try:
        fn = _lookup_modal_function()
    except Exception as exc:
        raise CloudExecutionError(
            f"Could not find a deployed Modal function named "
            f"'{MODAL_FUNCTION_NAME}' in app '{MODAL_APP_NAME}'. {MODAL_DEPLOY_HINT} "
            f"(Underlying error: {exc})"
        ) from exc

    fn, overrides_applied = _with_resource_overrides(fn, gpu=gpu, timeout_hours=timeout_hours)
    dispatch_note = "" if overrides_applied else "(using the deployed function's default GPU/timeout)"
    if progress_callback:
        safe_call(progress_callback, "dispatching-to-gpu", 0.05, dispatch_note)

    # fn.remote(...) is a single blocking call for however long the whole
    # remote run takes -- everything from image cold-start, through the
    # actual pipeline (which can itself run for a long time on a real
    # target), to the response coming back. Without a Heartbeat here, the
    # UI shows "dispatching-to-gpu (5%)" for that entire duration with no
    # way to tell a normal multi-hour run apart from a hung one -- exactly
    # the failure mode reported against an earlier version of this
    # function. The Heartbeat can only report elapsed time and local
    # resource usage (it has no visibility into what's actually happening
    # inside the remote container -- that's a real limitation, not
    # something this fixes), but "still waiting, Ns elapsed" is a real
    # improvement over total silence.
    try:
        with Heartbeat(
            progress_callback,
            "dispatching-to-gpu",
            0.05,
            interval_s=20.0,
            note="waiting on Modal (no visibility into remote progress until it returns)",
        ):
            result = fn.remote(request.to_job_dict())
    except Exception as exc:
        raise CloudExecutionError(
            f"Modal run failed to complete: {exc}. If this is an "
            f"authentication error, run `modal token new`; if it mentions "
            f"the app or function isn't found, {MODAL_DEPLOY_HINT.lower()}"
        ) from exc

    if result.get("status") != "ok":
        raise CloudExecutionError(
            f"The remote run failed with {result.get('error_type', 'UnknownError')}: "
            f"{result.get('error_message', 'no details returned')}"
        )

    if progress_callback:
        safe_call(progress_callback, "downloading-results", 0.9, "")

    run_id = result["run_id"]
    local_run_dir = Path(work_dir) / "runs" / run_id
    local_run_dir.mkdir(parents=True, exist_ok=True)
    volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=False)
    remote_prefix = f"amr_workspace/{result['relative_run_dir']}"
    for entry in volume.iterdir(remote_prefix, recursive=True):
        if not _volume_entry_is_file(entry):
            continue
        relative = Path(entry.path).relative_to(remote_prefix)
        dest = local_run_dir / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            for chunk in volume.read_file(entry.path):
                fh.write(chunk)

    if progress_callback:
        safe_call(progress_callback, "complete", 1.0, "")
    return local_run_dir


# ---------------------------------------------------------------------------
# Google Colab backend
# ---------------------------------------------------------------------------
#
# Generates a real, runnable .ipynb (plain JSON -- no nbformat dependency
# needed) rather than a static link plus a "paste these into Block 2" text
# block: this run's inputs are embedded directly into the notebook's cells,
# and every cell actually executes AMR-UNBIND's real pipeline code, so
# opening the file in Colab and running all cells is the entire workflow.


def _code_cell(lines: list[str]) -> dict:
    source = "\n".join(lines)
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _markdown_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def generate_colab_notebook(request: CloudRunRequest, *, repo_zip_url: str | None = None) -> bytes:
    """Builds a self-contained Google Colab notebook (as .ipynb bytes) that
    installs AMR-UNBIND and runs `request` on a free Colab GPU runtime.

    If `repo_zip_url` is given, the notebook downloads the package from
    there (e.g. a GitHub release asset you host). Otherwise it prompts for
    a manual upload of the same project zip you already have locally --
    since this project has no guaranteed public URL, that default is the
    one guaranteed to work for anyone, not just people who've set one up.
    """
    config_json = json.dumps(request.config.to_dict(), indent=2)
    smiles_literal = json.dumps(request.smiles)
    core_chain_literal = json.dumps(request.core_chain)
    exit_chain_literal = json.dumps(request.exit_chain)
    center_literal = (
        json.dumps(list(request.docking_center_angstrom)) if request.docking_center_angstrom is not None else "None"
    )

    cells = [
        _markdown_cell(
            "# AMR-UNBIND -- cloud run (Google Colab)\n\n"
            "This notebook installs AMR-UNBIND and runs **the same pipeline "
            "the desktop app uses**, on a free Colab GPU, with this run's "
            "inputs already filled in below.\n\n"
            "**Before running anything:** go to **Runtime -> Change runtime "
            "type** and select a **T4 GPU** (or better), then run the cells "
            "below in order (Runtime -> Run all).\n\n"
            "The last cell downloads a zip of the results back to your "
            "browser when the run finishes."
        ),
        _code_cell(["!nvidia-smi"]),
    ]

    if repo_zip_url:
        cells.append(
            _code_cell(
                [
                    "import pathlib, urllib.request, zipfile",
                    f"urllib.request.urlretrieve({json.dumps(repo_zip_url)}, 'amr_unbind_src.zip')",
                    "with zipfile.ZipFile('amr_unbind_src.zip') as zf:",
                    "    zf.extractall('.')",
                    "project_dirs = [p for p in pathlib.Path('.').iterdir() if p.is_dir() and p.name.lower().startswith('amr-unbind')]",
                    "assert project_dirs, 'Could not find an extracted AMR-UNBIND-* directory.'",
                    "%cd {project_dirs[0]}",
                ]
            )
        )
    else:
        cells.append(
            _markdown_cell(
                "## Upload the AMR-UNBIND project\n\n"
                "Upload the same `AMR-UNBIND-*.zip` you already have on your "
                "computer (the one this notebook came from)."
            )
        )
        cells.append(
            _code_cell(
                [
                    "import pathlib, zipfile",
                    "from google.colab import files",
                    "uploaded = files.upload()",
                    "zip_name = next(iter(uploaded))",
                    "with zipfile.ZipFile(zip_name) as zf:",
                    "    zf.extractall('.')",
                    "project_dirs = [p for p in pathlib.Path('.').iterdir() if p.is_dir() and p.name.lower().startswith('amr-unbind')]",
                    "assert project_dirs, 'Could not find an extracted AMR-UNBIND-* directory in the uploaded zip.'",
                    "%cd {project_dirs[0]}",
                ]
            )
        )

    cells.append(
        _code_cell(
            [
                "# Installs the exact dependency set requirements-docking.txt pins",
                "# for this version, so this notebook can't silently drift from what",
                "# the desktop app actually runs.",
                "!pip install -q -r requirements-docking.txt",
                "!pip install -q -e .",
            ]
        )
    )

    pdb_setup_lines = ["from pathlib import Path", ""]
    if request.pdb_bytes is not None:
        pdb_text_literal = json.dumps(request.pdb_bytes.decode("utf-8", errors="replace"))
        pdb_filename = request.pdb_filename or "input.pdb"
        pdb_setup_lines += [
            f"PDB_PATH = Path({json.dumps(pdb_filename)})",
            f"PDB_PATH.write_text({pdb_text_literal}, encoding='utf-8')",
            "PDB_ID = None",
        ]
    else:
        pdb_setup_lines += [
            "PDB_PATH = None",
            f"PDB_ID = {json.dumps(request.pdb_id)}",
        ]
    cells.append(_markdown_cell("## This run's inputs (already filled in)"))
    cells.append(_code_cell(pdb_setup_lines))

    pose_setup_lines: list[str] = []
    if request.pose_sdf_bytes is not None:
        pose_text_literal = json.dumps(request.pose_sdf_bytes.decode("utf-8", errors="replace"))
        pose_filename = request.pose_sdf_filename or "pose.sdf"
        pose_setup_lines = [
            f"POSE_SDF_PATH = Path({json.dumps(pose_filename)})",
            f"POSE_SDF_PATH.write_text({pose_text_literal}, encoding='utf-8')",
        ]
    else:
        pose_setup_lines = ["POSE_SDF_PATH = None"]
    cells.append(_code_cell(pose_setup_lines))

    cells.append(
        _code_cell(
            [
                f"SMILES = {smiles_literal}",
                f"CORE_CHAIN = {core_chain_literal}",
                f"CORE_RESIDUE = {int(request.core_residue)}",
                f"EXIT_CHAIN = {exit_chain_literal}",
                f"EXIT_RESIDUE = {int(request.exit_residue)}",
                f"AUTOMATIC_DOCKING = {bool(request.automatic_docking)}",
                f"DOCKING_BOX_ANGSTROM = {float(request.docking_box_angstrom)}",
                f"DOCKING_EXHAUSTIVENESS = {int(request.docking_exhaustiveness)}",
                f"DOCKING_CENTER_ANGSTROM = {center_literal}",
                "CONFIG_DICT = " + config_json,
            ]
        )
    )

    cells.append(_markdown_cell("## Run AMR-UNBIND\n\nThis is the same `AMRUnbindPipeline` the desktop app calls."))
    cells.append(
        _code_cell(
            [
                "from amr_unbind.config import SimulationConfig",
                "from amr_unbind.pipeline import AMRUnbindPipeline",
                "",
                "import time as _time",
                "_last_stage = {\"name\": None, \"started\": _time.time()}",
                "",
                "def _progress(stage, fraction, detail=\"\"):",
                "    now = _time.time()",
                "    if _last_stage[\"name\"] != stage:",
                "        _last_stage[\"name\"] = stage",
                "        _last_stage[\"started\"] = now",
                "    elapsed = now - _last_stage[\"started\"]",
                "    line = f'[{fraction:6.1%}] {stage} ({elapsed:.0f}s in this stage)'",
                "    if detail:",
                "        line += f' -- {detail}'",
                "    print(line, flush=True)",
                "",
                "config = SimulationConfig.from_dict(CONFIG_DICT)",
                "pipeline = AMRUnbindPipeline(work_dir=Path('./amr_workspace'))",
                "run_dir = pipeline.run(",
                "    pdb_path=PDB_PATH,",
                "    pdb_id=PDB_ID,",
                "    smiles=SMILES,",
                "    core_chain=CORE_CHAIN,",
                "    core_residue=CORE_RESIDUE,",
                "    exit_chain=EXIT_CHAIN,",
                "    exit_residue=EXIT_RESIDUE,",
                "    config=config,",
                "    ligand_pose_sdf=POSE_SDF_PATH,",
                "    automatic_docking=AUTOMATIC_DOCKING,",
                "    docking_box_angstrom=DOCKING_BOX_ANGSTROM,",
                "    docking_exhaustiveness=DOCKING_EXHAUSTIVENESS,",
                "    docking_center_angstrom=tuple(DOCKING_CENTER_ANGSTROM) if DOCKING_CENTER_ANGSTROM else None,",
                "    progress_callback=_progress,",
                ")",
                "print('Run complete:', run_dir)",
            ]
        )
    )

    cells.append(_markdown_cell("## Download the results"))
    cells.append(
        _code_cell(
            [
                "import shutil",
                "from google.colab import files",
                "",
                "archive_path = shutil.make_archive(run_dir.name, 'zip', run_dir)",
                "files.download(archive_path)",
            ]
        )
    )

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "accelerator": "GPU",
            "colab": {"name": "amr_unbind_run.ipynb", "provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=1).encode("utf-8")
