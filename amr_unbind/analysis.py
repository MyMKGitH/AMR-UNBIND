from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px


def load_analysis(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {
        "Time (ps)",
        "Reaction Coordinate (nm)",
        "Target Coordinate (nm)",
        "Displacement (nm)",
        "Steering Force (kJ/mol/nm)",
        "Cumulative Work (kJ/mol)",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Analysis file missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Analysis file contains no trajectory samples.")
    for column in required:
        try:
            df[column] = pd.to_numeric(df[column], errors="raise")
        except Exception as exc:
            raise ValueError(f"Analysis column is not numeric: {column}") from exc
    return df


def make_force_plot(df: pd.DataFrame):
    return px.line(
        df,
        x="Displacement (nm)",
        y="Steering Force (kJ/mol/nm)",
        title="Steering Force vs. Ligand COM Displacement",
        markers=False,
    )


def make_work_plot(df: pd.DataFrame):
    return px.line(
        df,
        x="Displacement (nm)",
        y="Cumulative Work (kJ/mol)",
        title="Cumulative Protocol Work vs. Ligand COM Displacement",
        markers=False,
    )


def qc_metrics(df: pd.DataFrame) -> dict[str, float | bool]:
    if df.empty:
        raise ValueError("Cannot calculate QC metrics for an empty trajectory.")
    force = df["Steering Force (kJ/mol/nm)"].to_numpy(float)
    work = df["Cumulative Work (kJ/mol)"].to_numpy(float)
    finite = bool(np.isfinite(df.select_dtypes(include=[np.number]).to_numpy(float)).all())
    target = df["Target Coordinate (nm)"].to_numpy(float)
    monotonic_target = bool(len(target) < 2 or np.all(np.diff(target) > 0))
    return {
        "all_numeric_values_finite": finite,
        "target_coordinate_monotonic": monotonic_target,
        "peak_abs_force_kj_mol_nm": float(np.max(np.abs(force))),
        "final_work_kj_mol": float(work[-1]),
        "final_displacement_nm": float(df["Displacement (nm)"].iloc[-1]),
    }
