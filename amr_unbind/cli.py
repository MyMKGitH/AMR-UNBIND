from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

from .cloud import CloudRunRequest, MODAL_VALID_GPU_TYPES, submit_modal_run
from .config import SimulationConfig
from .errors import AMRUnbindError
from .pipeline import AMRUnbindPipeline
from . import __version__


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"AMR-UNBIND {__version__}")
    p.add_argument("--pdb-id")
    p.add_argument("--pdb-file")
    p.add_argument("--smiles", required=True)
    p.add_argument("--ligand-pose-sdf")
    p.add_argument("--automatic-docking", action="store_true")
    p.add_argument("--docking-box-angstrom", type=float, default=22.0)
    p.add_argument("--docking-exhaustiveness", type=int, default=8)
    p.add_argument(
        "--docking-center-angstrom",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "Custom Vina search-box center in Angstrom (three floats: X Y Z). "
            "Only used with --automatic-docking. Defaults to the geometric "
            "point between --core-residue and --exit-residue."
        ),
    )
    p.add_argument("--core-chain", required=True)
    p.add_argument("--core-residue", type=int, required=True)
    p.add_argument("--exit-chain", required=True)
    p.add_argument("--exit-residue", type=int, required=True)
    p.add_argument("--work-dir", default="./amr_workspace")
    p.add_argument("--config")
    p.add_argument(
        "--platform",
        default=None,
        choices=["auto", "cpu", "cuda", "opencl", "hip"],
        help="OpenMM platform. Defaults to 'auto' unless --config already sets preferred_platform.",
    )
    p.add_argument("--pull-distance-nm", type=float)
    p.add_argument("--pull-velocity-nm-ps", type=float)
    p.add_argument("--force-constant", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument(
        "--compute",
        default="local",
        choices=["local", "modal"],
        help=(
            "Where to run the pipeline. 'local' (default) runs on this "
            "machine. 'modal' dispatches to a deployed Modal GPU function "
            "(run `modal deploy amr_unbind/cloud.py` once first; see "
            "README.md's Cloud Compute section) and downloads the results "
            "here when it finishes."
        ),
    )
    p.add_argument(
        "--modal-gpu",
        default="T4",
        choices=list(MODAL_VALID_GPU_TYPES),
        help="GPU type for --compute modal. Default: T4.",
    )
    p.add_argument(
        "--modal-timeout-hours",
        type=float,
        default=6.0,
        help="Wall-clock timeout for --compute modal, in hours. Default: 6.0.",
    )
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.pdb_id and not args.pdb_file:
        raise SystemExit("Provide --pdb-id or --pdb-file")

    def progress(stage, fraction, detail=""):
        line = f"[{fraction:6.1%}] {stage}"
        if detail:
            line += f" -- {detail}"
        print(line, flush=True)

    try:
        # SimulationConfig.from_json()/from_dict() call SimulationConfig.validate()
        # internally and raise a plain ValueError on bad values (a bad --config
        # file, or an out-of-range --seed/--pull-distance-nm/etc. override).
        # Catch that here alongside AMRUnbindError from the pipeline itself so
        # every expected failure path gets one clean stderr message and a
        # non-zero exit code instead of a raw Python traceback.
        if args.config:
            config = SimulationConfig.from_json(Path(args.config))
        else:
            config = SimulationConfig()
        updates = {}
        # Only override preferred_platform when --platform was explicitly
        # passed. A truthy hardcoded default here would silently discard
        # whatever preferred_platform a --config file specified on every run.
        if args.platform is not None: updates["preferred_platform"] = args.platform
        if args.pull_distance_nm is not None: updates["pull_distance_nm"] = args.pull_distance_nm
        if args.pull_velocity_nm_ps is not None: updates["pull_velocity_nm_per_ps"] = args.pull_velocity_nm_ps
        if args.force_constant is not None: updates["steering_k_kj_mol_nm2"] = args.force_constant
        if args.seed is not None: updates["seed"] = args.seed
        if updates:
            config = SimulationConfig.from_dict({**config.to_dict(), **updates})

        docking_center = tuple(args.docking_center_angstrom) if args.docking_center_angstrom else None

        if args.compute == "modal":
            pdb_bytes = Path(args.pdb_file).read_bytes() if args.pdb_file else None
            pose_bytes = Path(args.ligand_pose_sdf).read_bytes() if args.ligand_pose_sdf else None
            request = CloudRunRequest(
                smiles=args.smiles,
                core_chain=args.core_chain,
                core_residue=args.core_residue,
                exit_chain=args.exit_chain,
                exit_residue=args.exit_residue,
                config=config,
                pdb_id=args.pdb_id,
                pdb_bytes=pdb_bytes,
                pdb_filename=Path(args.pdb_file).name if args.pdb_file else None,
                pose_sdf_bytes=pose_bytes,
                pose_sdf_filename=Path(args.ligand_pose_sdf).name if args.ligand_pose_sdf else None,
                automatic_docking=args.automatic_docking,
                docking_box_angstrom=args.docking_box_angstrom,
                docking_exhaustiveness=args.docking_exhaustiveness,
                docking_center_angstrom=docking_center,
            )
            run_dir = submit_modal_run(
                request,
                work_dir=Path(args.work_dir),
                gpu=args.modal_gpu,
                timeout_hours=args.modal_timeout_hours,
                progress_callback=progress,
            )
        else:
            pipeline = AMRUnbindPipeline(Path(args.work_dir))
            run_dir = pipeline.run(
                pdb_path=Path(args.pdb_file) if args.pdb_file else None,
                pdb_id=args.pdb_id,
                smiles=args.smiles,
                core_chain=args.core_chain,
                core_residue=args.core_residue,
                exit_chain=args.exit_chain,
                exit_residue=args.exit_residue,
                config=config,
                ligand_pose_sdf=Path(args.ligand_pose_sdf) if args.ligand_pose_sdf else None,
                automatic_docking=args.automatic_docking,
                docking_box_angstrom=args.docking_box_angstrom,
                docking_exhaustiveness=args.docking_exhaustiveness,
                docking_center_angstrom=docking_center,
                progress_callback=progress,
            )
    except (AMRUnbindError, ValueError) as exc:
        print(f"AMR-UNBIND run failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"run_dir": str(run_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
