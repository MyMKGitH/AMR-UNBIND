from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json

# Minimum solvation buffer (nm) that must remain between the far edge of the
# ligand's intended travel and the periodic box edge. This is a conservative
# margin (smaller than a full nonbonded cutoff) meant only to catch grossly
# under-padded configurations where the ligand would approach its own
# periodic image during the pull; it is not a substitute for checking the
# solvated box size for a specific system.
_MIN_PADDING_PULL_MARGIN_NM = 0.2


@dataclass(frozen=True)
class SimulationConfig:
    """Numerical settings for a pilot SMD run.

    Values are deliberately explicit so every run can be reproduced from its
    configuration file and the generated metadata.json.
    """

    temperature_kelvin: float = 300.0
    pressure_atm: float = 1.0
    friction_per_ps: float = 1.0
    timestep_fs: float = 2.0
    ionic_strength_molar: float = 0.15
    solvation_padding_nm: float = 1.0

    nvt_equilibration_ps: float = 10.0
    npt_equilibration_ps: float = 10.0
    pull_distance_nm: float = 0.25
    pull_velocity_nm_per_ps: float = 0.005
    steering_k_kj_mol_nm2: float = 500.0

    report_interval_steps: int = 100
    checkpoint_interval_steps: int = 5000
    minimization_iterations: int = 500
    seed: int = 20260907

    protein_restraint_k_kj_mol_nm2: float = 100.0
    pH: float = 7.4

    forcefield_protein: str = "amber14-all.xml"
    forcefield_water: str = "amber14/tip3p.xml"
    small_molecule_forcefield: str = "openff-2.2.1"

    preferred_platform: str = "auto"
    cuda_device_index: str = "0"

    def validate(self) -> None:
        checks = {
            "temperature_kelvin": self.temperature_kelvin > 0,
            "pressure_atm": self.pressure_atm > 0,
            "friction_per_ps": self.friction_per_ps > 0,
            "timestep_fs": 0 < self.timestep_fs <= 4.0,
            "ionic_strength_molar": self.ionic_strength_molar >= 0,
            "solvation_padding_nm": self.solvation_padding_nm >= 0.5,
            "nvt_equilibration_ps": self.nvt_equilibration_ps >= 0,
            "npt_equilibration_ps": self.npt_equilibration_ps >= 0,
            "pull_distance_nm": self.pull_distance_nm > 0,
            "pull_velocity_nm_per_ps": self.pull_velocity_nm_per_ps > 0,
            "steering_k_kj_mol_nm2": self.steering_k_kj_mol_nm2 > 0,
            "report_interval_steps": self.report_interval_steps > 0,
            "checkpoint_interval_steps": self.checkpoint_interval_steps > 0,
            "minimization_iterations": self.minimization_iterations >= 0,
            "seed": self.seed >= 0,
            "protein_restraint_k_kj_mol_nm2": self.protein_restraint_k_kj_mol_nm2 >= 0,
            "pH": 0 <= self.pH <= 14,
            "preferred_platform": self.preferred_platform.lower() in {"auto", "cpu", "cuda", "opencl", "hip"},
            "cuda_device_index": bool(str(self.cuda_device_index).strip()) and str(self.cuda_device_index).strip().isdigit(),
            "solvation_padding_nm_vs_pull_distance_nm": (
                self.solvation_padding_nm >= self.pull_distance_nm + _MIN_PADDING_PULL_MARGIN_NM
            ),
        }
        bad = [name for name, ok in checks.items() if not ok]
        if bad:
            if "solvation_padding_nm_vs_pull_distance_nm" in bad:
                bad[bad.index("solvation_padding_nm_vs_pull_distance_nm")] = (
                    "solvation_padding_nm_vs_pull_distance_nm (solvation_padding_nm "
                    f"({self.solvation_padding_nm} nm) must be at least "
                    f"pull_distance_nm + {_MIN_PADDING_PULL_MARGIN_NM} nm "
                    f"({self.pull_distance_nm + _MIN_PADDING_PULL_MARGIN_NM} nm) so the "
                    "ligand cannot approach its own periodic image during the pull)"
                )
            raise ValueError(f"Invalid simulation configuration: {', '.join(bad)}")

    @property
    def pull_time_ps(self) -> float:
        return self.pull_distance_nm / self.pull_velocity_nm_per_ps

    @property
    def pull_steps(self) -> int:
        step_ps = self.timestep_fs / 1000.0
        return max(1, int(round(self.pull_time_ps / step_ps)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        self.validate()
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimulationConfig":
        obj = cls(**data)
        obj.validate()
        return obj

    @classmethod
    def from_json(cls, path: Path) -> "SimulationConfig":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
