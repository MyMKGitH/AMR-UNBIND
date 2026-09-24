from __future__ import annotations

from dataclasses import dataclass
import openmm as mm


@dataclass(frozen=True)
class PlatformSelection:
    name: str
    properties: dict[str, str]
    reason: str


def _available_names() -> list[str]:
    return [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]


def choose_platform(preferred: str = "auto", cuda_device_index: str = "0") -> PlatformSelection:
    preferred = preferred.lower()
    available = _available_names()

    if preferred != "auto":
        requested = {"cpu": "CPU", "cuda": "CUDA", "opencl": "OpenCL", "hip": "HIP"}[preferred]
        if requested not in available:
            raise RuntimeError(f"Requested OpenMM platform {requested} is not available. Available: {available}")
        props = {}
        if requested in {"CUDA", "OpenCL", "HIP"}:
            props["Precision"] = "mixed"
        if requested == "CUDA":
            props["DeviceIndex"] = cuda_device_index
        return PlatformSelection(requested, props, f"Requested platform: {requested}")

    for name in ("CUDA", "HIP", "OpenCL", "CPU"):
        if name in available:
            props = {}
            if name in {"CUDA", "OpenCL", "HIP"}:
                props["Precision"] = "mixed"
            if name == "CUDA":
                props["DeviceIndex"] = cuda_device_index
            return PlatformSelection(name, props, f"Auto-selected first available accelerator/platform: {name}")

    raise RuntimeError(f"No usable OpenMM platform found. Available: {available}")
