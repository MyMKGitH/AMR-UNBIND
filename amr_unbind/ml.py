from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def trajectory_features(analysis_csv: Path) -> dict[str, float]:
    """Return a compact, ML-ready feature vector from one completed SMD run.

    This is feature extraction only. No model is trained from a single trajectory.
    Multiple independent runs/ligands are needed before supervised prediction is meaningful.
    """
    df = pd.read_csv(analysis_csv)
    force = df["Steering Force (kJ/mol/nm)"].to_numpy(dtype=float)
    work = df["Cumulative Work (kJ/mol)"].to_numpy(dtype=float)
    disp = df["Displacement (nm)"].to_numpy(dtype=float)
    if len(df) == 0:
        raise ValueError("Analysis file contains no rows.")
    if not np.isfinite(np.concatenate([force, work, disp])).all():
        raise ValueError("Analysis file contains non-finite force, work, or displacement values.")

    force_slope = 0.0
    if len(disp) >= 2 and float(np.ptp(disp)) > 1e-6:
        force_slope = float(np.polyfit(disp, force, 1)[0])

    return {
        "final_displacement_nm": float(disp[-1]),
        "max_abs_force_kj_mol_nm": float(np.max(np.abs(force))),
        "mean_abs_force_kj_mol_nm": float(np.mean(np.abs(force))),
        "force_std_kj_mol_nm": float(np.std(force)),
        "final_work_kj_mol": float(work[-1]),
        "work_per_nm_kj_mol_per_nm": float(work[-1] / max(abs(disp[-1]), 1e-12)),
        "force_slope_kj_mol_nm2": force_slope,
    }


def save_ml_features(analysis_csv: Path, output_csv: Path) -> None:
    features = trajectory_features(analysis_csv)
    pd.DataFrame([features]).to_csv(output_csv, index=False)
