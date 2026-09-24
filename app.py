from __future__ import annotations

import os

# Belt-and-suspenders: amr_unbind/__init__.py already sets this before any
# of its own submodules (OpenMM/OpenFF/RDKit) are imported, but this file
# imports Streamlit first, which can itself pull in NumPy/pyarrow before
# amr_unbind is ever touched. Setting it here too, before that import,
# closes that gap. See amr_unbind/__init__.py for the full explanation.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
import tempfile
import json
import threading
import time

import streamlit as st

from amr_unbind import __version__
from amr_unbind.analysis import make_force_plot, make_work_plot, load_analysis, qc_metrics
from amr_unbind.chimerax import ChimeraXBridge
from amr_unbind.cloud import (
    CloudRunRequest,
    MODAL_VALID_GPU_TYPES,
    generate_colab_notebook,
    modal_available,
    submit_modal_run,
)
from amr_unbind.config import SimulationConfig
from amr_unbind.errors import AMRUnbindError
from amr_unbind.pipeline import AMRUnbindPipeline
from amr_unbind.ml import save_ml_features


st.set_page_config(page_title=f"AMR-UNBIND v{__version__}", layout="wide")
st.title(f"🧪 AMR-UNBIND v{__version__}")
st.caption("AMR ligand unbinding workflow: docking, explicit-solvent OpenMM SMD, and trajectory analysis")

with st.sidebar:
    st.header("Target")
    source_mode = st.radio("Protein source", ["RCSB PDB ID", "Local PDB file"], horizontal=True)
    pdb_id = st.text_input("PDB ID", "1STP") if source_mode == "RCSB PDB ID" else None
    pdb_upload = st.file_uploader("Upload protein PDB", type=["pdb", "ent"]) if source_mode == "Local PDB file" else None

    st.header("Ligand")
    smiles = st.text_input("Ligand SMILES", "O=C1N[C@@H]2[C@@H](CS[C@H]2CCCCC(=O)O)N1")
    pose_upload = st.file_uploader("Pre-docked ligand SDF (recommended)", type=["sdf"])
    auto_dock = st.checkbox("Automatically dock ligand with AutoDock Vina", value=False)
    docking_box = st.slider("Docking box size (Å)", 5.0, 60.0, 15.0, 0.5)
    docking_exhaustiveness = st.slider("Vina exhaustiveness", 1, 32, 8, 1)

    st.header("Pull direction")
    core_chain = st.text_input("Core chain", "A")
    core_residue = st.number_input("Core residue", value=45, step=1)
    exit_chain = st.text_input("Exit chain", "A")
    exit_residue = st.number_input("Exit residue", value=84, step=1)

    st.header("SMD protocol")
    pull_distance = st.slider("Pull distance (nm)", 0.05, 1.50, 0.25, 0.05)
    pull_velocity = st.slider("Pull velocity (nm/ps)", 0.0005, 0.0200, 0.0050, 0.0005)
    force_constant = st.slider("Steering k (kJ/mol/nm²)", 50.0, 2000.0, 500.0, 50.0)
    solvation_padding = st.slider(
        "Solvation box padding (nm)",
        0.6,
        2.5,
        1.0,
        0.1,
        help=(
            "Minimum solvent buffer beyond the solute in every direction. "
            "Must stay comfortably larger than the pull distance so the "
            "ligand cannot approach its own periodic image while it is "
            "steered; the app will reject a run where it does not."
        ),
    )
    nvt_ps = st.number_input("NVT equilibration (ps)", value=10.0, min_value=0.0, step=5.0)
    npt_ps = st.number_input("NPT equilibration (ps)", value=10.0, min_value=0.0, step=5.0)
    seed = st.number_input("Random seed", value=20260907, step=1)
    platform = st.selectbox("OpenMM platform", ["auto", "cpu", "cuda", "opencl", "hip"])

    st.header("Compute")
    st.caption(
        "Explicit-solvent SMD is slow on CPU (hours, sometimes with nothing "
        "to show for it if it doesn't finish). Run it here anyway, or send "
        "it to a GPU instead."
    )
    compute_mode = st.radio(
        "Where to run",
        [
            "Local (this machine)",
            "Cloud \u2014 Modal GPU (pay-per-second; free credits available on a new account)",
            "Cloud \u2014 Google Colab notebook (free GPU in your browser; you run it, not this app)",
        ],
    )
    modal_gpu = "T4"
    modal_timeout_hours = 6.0
    colab_repo_zip_url = None
    if compute_mode.startswith("Cloud \u2014 Modal"):
        if not modal_available():
            st.warning(
                "The `modal` package isn't installed here. Run `pip install "
                "modal` (or `pip install -r requirements-cloud.txt`), then "
                "`modal token new`, then `modal deploy amr_unbind/cloud.py` "
                "once -- see README.md's Cloud Compute section."
            )
        modal_gpu = st.selectbox("Modal GPU type", list(MODAL_VALID_GPU_TYPES), index=0)
        modal_timeout_hours = st.number_input(
            "Modal timeout (hours)",
            value=6.0,
            min_value=0.25,
            max_value=24.0,
            step=0.25,
            help=(
                "Wall-clock limit for the remote run. Applied per-run if "
                "your installed `modal` version supports it; otherwise the "
                "deployed function's own timeout (set at `modal deploy` "
                "time) is used instead, and this app will say so."
            ),
        )
    elif compute_mode.startswith("Cloud \u2014 Google Colab"):
        st.caption(
            "This generates a notebook file for you to download and run "
            "yourself in Colab -- it does not run anything from here, and "
            "results come back to your browser, not this app."
        )
        colab_repo_zip_url = st.text_input(
            "Optional: URL to a hosted copy of this project (e.g. a GitHub "
            "release zip)",
            value="",
            help=(
                "Leave blank to have the generated notebook prompt for a "
                "manual upload of this project's zip instead -- works for "
                "anyone, with no setup."
            ),
        ) or None

st.write("**Docking Center Coordinates (Å)**")
st.caption(
    "⚠️ These coordinates are specific to one PDB structure's own coordinate "
    "frame. The X/Y/Z values below are only an example (a hypothetical biotin "
    "pocket in PDB 1STP) — they are not a sensible default for any structure, "
    "and using them for one will very likely produce a nonsensical docking "
    "box (this is refused before it wastes a run, but it's better to avoid "
    "in the first place). Leave the override unchecked unless you know the "
    "correct coordinates for the specific structure you're running."
)
use_custom_docking_center = st.checkbox(
    "Override auto-computed pocket center with these coordinates",
    value=False,
    help=(
        "Only applies when automatic docking is enabled above. When left "
        "unchecked, the Vina search-box center is the geometric point "
        "between the core and exit residues (computed automatically). "
        "When checked, these X/Y/Z coordinates are used instead -- useful "
        "when the binding pocket is already known and shouldn't depend on "
        "the core/exit residue choice, and known correct for the specific "
        "structure entered above."
    ),
)
col1, col2, col3 = st.columns(3)
with col1:
    center_x = st.number_input("Center X (example)", value=27.23, step=0.1, format="%.2f")
with col2:
    center_y = st.number_input("Center Y (example)", value=11.23, step=0.1, format="%.2f")
with col3:
    center_z = st.number_input("Center Z (example)", value=17.52, step=0.1, format="%.2f")

run_button_label = (
    "Generate Colab Notebook" if compute_mode.startswith("Cloud \u2014 Google Colab") else "Run AMR-UNBIND"
)
run = st.button(run_button_label, type="primary", use_container_width=True)


if run:
    tmpdir = Path(tempfile.mkdtemp(prefix="amr_unbind_"))
    local_pdb = None
    local_pose = None
    pdb_bytes = pdb_upload.getbuffer().tobytes() if pdb_upload is not None else None
    pose_bytes = pose_upload.getbuffer().tobytes() if pose_upload is not None else None
    if pdb_upload is not None:
        local_pdb = tmpdir / pdb_upload.name
        local_pdb.write_bytes(pdb_bytes)
    if pose_upload is not None:
        local_pose = tmpdir / pose_upload.name
        local_pose.write_bytes(pose_bytes)

    if auto_dock and local_pose is not None:
        st.error("Choose automatic docking or a supplied pose, not both.")
        st.stop()
    if not auto_dock and local_pose is None:
        st.error("A pre-docked ligand SDF or automatic docking is required. A free ligand conformer is not a bound complex.")
        st.stop()

    config = SimulationConfig(
        pull_distance_nm=float(pull_distance),
        pull_velocity_nm_per_ps=float(pull_velocity),
        steering_k_kj_mol_nm2=float(force_constant),
        solvation_padding_nm=float(solvation_padding),
        nvt_equilibration_ps=float(nvt_ps),
        npt_equilibration_ps=float(npt_ps),
        seed=int(seed),
        preferred_platform=platform,
    )
    docking_center = (
        (float(center_x), float(center_y), float(center_z))
        if (auto_dock and use_custom_docking_center)
        else None
    )

    # Shared by both cloud paths (Modal and Colab); the local path below
    # calls AMRUnbindPipeline.run() directly with these same values instead,
    # since it needs no serialization.
    def _build_cloud_request() -> CloudRunRequest:
        return CloudRunRequest(
            smiles=smiles,
            core_chain=core_chain.strip(),
            core_residue=int(core_residue),
            exit_chain=exit_chain.strip(),
            exit_residue=int(exit_residue),
            config=config,
            pdb_id=pdb_id if pdb_upload is None else None,
            pdb_bytes=pdb_bytes,
            pdb_filename=pdb_upload.name if pdb_upload is not None else None,
            pose_sdf_bytes=pose_bytes,
            pose_sdf_filename=pose_upload.name if pose_upload is not None else None,
            automatic_docking=auto_dock,
            docking_box_angstrom=float(docking_box),
            docking_exhaustiveness=int(docking_exhaustiveness),
            docking_center_angstrom=docking_center,
        )

    if compute_mode.startswith("Cloud \u2014 Google Colab"):
        try:
            notebook_bytes = generate_colab_notebook(_build_cloud_request(), repo_zip_url=colab_repo_zip_url)
        except AMRUnbindError as exc:
            st.error(str(exc))
            st.stop()
        st.success("Notebook generated -- this app did not run anything. Download it, open it in Colab, and run all cells there.")
        st.download_button(
            "Download Colab notebook (.ipynb)",
            notebook_bytes,
            file_name="amr_unbind_run.ipynb",
            mime="application/x-ipynb+json",
        )
        st.info(
            "In Colab: **Runtime > Change runtime type > T4 GPU**, then "
            "**Runtime > Run all**. The notebook's last cell downloads a "
            "results zip back to your browser when the run finishes; "
            "results do not appear in this app."
        )
    else:
        is_modal = compute_mode.startswith("Cloud \u2014 Modal")
        status = st.status("Running on Modal GPU" if is_modal else "Running AMR-UNBIND", expanded=True)
        bar = st.progress(0.0)
        detail_line = st.empty()

        # Heartbeat (see amr_unbind/progress.py) ticks from a background
        # thread during long blocking calls (Vina docking, Meeko prep,
        # minimization) so this callback gets invoked from a thread
        # Streamlit doesn't automatically know about. Attaching the
        # capturing script's context the first time each thread calls in
        # is Streamlit's own documented pattern for updating widgets from
        # a background thread; if it's unavailable (an older/newer
        # Streamlit than expected) this is wrapped defensively -- worst
        # case, a heartbeat tick just doesn't render, it never crashes the
        # run (every call into this function already goes through
        # amr_unbind.progress.safe_call's own try/except too).
        try:
            from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

            _script_ctx = get_script_run_ctx()
        except Exception:
            add_script_run_ctx = None
            _script_ctx = None
        _context_attached_threads: set[int] = set()

        _progress_state = {"stage": None, "stage_started": time.time(), "updates": 0}

        def progress(stage, fraction, detail=""):
            if add_script_run_ctx is not None and _script_ctx is not None:
                current = threading.current_thread()
                if current.ident not in _context_attached_threads:
                    try:
                        add_script_run_ctx(current, _script_ctx)
                    except Exception:
                        pass
                    _context_attached_threads.add(current.ident)

            now = time.time()
            if _progress_state["stage"] != stage:
                _progress_state["stage"] = stage
                _progress_state["stage_started"] = now
            _progress_state["updates"] += 1
            stage_elapsed = now - _progress_state["stage_started"]

            status.update(label=f"{stage} ({fraction:.0%})", state="running")
            bar.progress(min(1.0, max(0.0, float(fraction))))
            detail_text = (
                f"**Current stage:** `{stage}` \u2014 {stage_elapsed:.0f}s in this stage "
                f"\u2014 last update {time.strftime('%H:%M:%S')}"
            )
            if detail:
                detail_text += f"  \n{detail}"
            detail_line.markdown(detail_text)

        try:
            if is_modal:
                run_dir = submit_modal_run(
                    _build_cloud_request(),
                    work_dir=Path("./amr_workspace"),
                    gpu=modal_gpu,
                    timeout_hours=float(modal_timeout_hours),
                    progress_callback=progress,
                )
            else:
                pipeline = AMRUnbindPipeline(Path("./amr_workspace"))
                run_dir = pipeline.run(
                    pdb_path=local_pdb,
                    pdb_id=pdb_id,
                    smiles=smiles,
                    core_chain=core_chain.strip(),
                    core_residue=int(core_residue),
                    exit_chain=exit_chain.strip(),
                    exit_residue=int(exit_residue),
                    config=config,
                    ligand_pose_sdf=local_pose,
                    automatic_docking=auto_dock,
                    docking_box_angstrom=float(docking_box),
                    docking_exhaustiveness=int(docking_exhaustiveness),
                    docking_center_angstrom=docking_center,
                    progress_callback=progress,
                )
            status.update(label="Run complete", state="complete")
            st.session_state["last_run"] = str(run_dir)
        except AMRUnbindError as exc:
            status.update(label="Run failed", state="error")
            st.exception(exc)
            st.stop()


run_dir_str = st.session_state.get("last_run")
if run_dir_str:
    run_dir = Path(run_dir_str)
    st.success(f"Run directory: {run_dir}")
    analysis_path = run_dir / "smd_analysis.csv"
    if analysis_path.exists():
        df = load_analysis(analysis_path)
        metrics = qc_metrics(df)
        c1, c2, c3 = st.columns(3)
        c1.metric("Final displacement", f"{metrics['final_displacement_nm']:.3f} nm")
        c2.metric("Peak |force|", f"{metrics['peak_abs_force_kj_mol_nm']:.1f} kJ/mol/nm")
        c3.metric("Cumulative work", f"{metrics['final_work_kj_mol']:.1f} kJ/mol")

        st.subheader("Force analysis")
        st.plotly_chart(make_force_plot(df), use_container_width=True)
        st.subheader("Protocol work")
        st.plotly_chart(make_work_plot(df), use_container_width=True)

        if not metrics["all_numeric_values_finite"] or not metrics["target_coordinate_monotonic"]:
            st.warning("Automatic trajectory QC detected a numerical issue. Treat the run as invalid until inspected.")

        st.download_button("Download analysis CSV", analysis_path.read_bytes(), file_name="smd_analysis.csv")
        st.download_button(
            "Download run metadata",
            (run_dir / "run_metadata.json").read_bytes(),
            file_name="run_metadata.json",
        )

        ml_path = run_dir / "ml_features.csv"
        save_ml_features(analysis_path, ml_path)
        st.download_button("Download ML-ready feature vector", ml_path.read_bytes(), file_name="ml_features.csv")

        st.subheader("Prepared / final structure")
        final_pdb = run_dir / "final_structure.pdb"
        if final_pdb.exists():
            st.code(final_pdb.read_text(encoding="utf-8")[:12000], language="text")
            try:
                import py3Dmol
                from stmol import showmol
                view = py3Dmol.view(width=900, height=500)
                view.addModel(final_pdb.read_text(encoding="utf-8"), "pdb")
                view.setStyle({"cartoon": {"color": "spectrum"}})
                view.addStyle({"resn": "LIG"}, {"stick": {"colorscheme": "greenCarbon"}})
                view.zoomTo()
                showmol(view, height=500, width=900)
            except Exception as exc:
                st.info(f"3D viewer unavailable: {exc}")

        st.subheader("Optional ChimeraX rendering")
        cx_port = st.number_input("ChimeraX REST port", value=3000, min_value=1, step=1)
        if st.button("Ask ChimeraX to render movie"):
            try:
                out_movie = run_dir / "unbinding_movie.mp4"
                bridge = ChimeraXBridge(port=int(cx_port))
                n_frames = None
                summary_path = run_dir / "simulation_summary.json"
                if summary_path.exists():
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    n_frames = int(summary.get("trajectory_frames", 0)) or None
                bridge.render_movie(
                    run_dir / "equilibrated_start.pdb",
                    run_dir / "smd_trajectory.dcd",
                    out_movie,
                    n_frames=n_frames,
                )
                if out_movie.exists():
                    st.video(str(out_movie))
                else:
                    st.info("ChimeraX accepted the command, but no movie file is visible to the application. Check the ChimeraX log.")
            except Exception as exc:
                st.error(str(exc))
