"""Environment and machine-state capture for benchmark records (D-012)."""

from __future__ import annotations

import importlib.metadata as md
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

_PACKAGES = (
    "torch",
    "torchvision",
    "tensorrt",
    "tensorrt-cu12",
    "tensorrt-cu12-libs",
    "mujoco",
    "numpy",
    "onnx",
    "nvidia-ml-py",
    "lerobot",
    "transformers",
)


def _run(cmd: list[str], cwd: Optional[Path] = None) -> Optional[str]:
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _read(path: str) -> Optional[str]:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def git_info() -> dict:
    commit = _run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
    status = _run(["git", "status", "--porcelain"], cwd=REPO_ROOT)
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def package_versions() -> dict:
    versions = {}
    for name in _PACKAGES:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            continue
    return versions


def cuda_info() -> dict:
    info: dict = {}
    nvcc = shutil.which("nvcc")
    if nvcc:
        out = _run([nvcc, "--version"])
        info["nvcc_path"] = nvcc
        release = [line for line in (out or "").splitlines() if "release" in line]
        info["nvcc_release"] = release[0].split("release", 1)[1].strip() if release else None
    torch = sys.modules.get("torch")
    if torch is not None:
        info["torch_cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(h)
        info["gpu_name"] = name.decode() if isinstance(name, bytes) else name
        drv = pynvml.nvmlSystemGetDriverVersion()
        info["driver"] = drv.decode() if isinstance(drv, bytes) else drv
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(h)
        info["compute_capability"] = f"{major}.{minor}"
        info["vram_total_mib"] = pynvml.nvmlDeviceGetMemoryInfo(h).total / 2**20
        info["driver_cuda_version"] = pynvml.nvmlSystemGetCudaDriverVersion_v2()
    except Exception as exc:
        info["nvml_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _cpu_model() -> Optional[str]:
    text = _read("/proc/cpuinfo")
    if text:
        for line in text.splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


def _on_ac_power() -> Optional[bool]:
    base = Path("/sys/class/power_supply")
    if not base.exists():
        return None
    for supply in base.iterdir():
        if _read(str(supply / "type")) == "Mains":
            online = _read(str(supply / "online"))
            if online is not None:
                return online == "1"
    return None


def machine_state() -> dict:
    """Power mode, AC state and CPU governor: the inputs that make laptop numbers noisy."""
    return {
        "hostname": platform.node(),
        "os": platform.platform(),
        "kernel": platform.release(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "on_ac_power": _on_ac_power(),
        "platform_profile": _read("/sys/firmware/acpi/platform_profile"),
        "cpu_governor": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "energy_performance_preference": _read(
            "/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference"
        ),
    }


def collect() -> dict:
    return {
        "git": git_info(),
        "python": sys.version.split()[0],
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "packages": package_versions(),
        "cuda": cuda_info(),
        "machine": machine_state(),
    }
