"""Custom CUDA kernel: build saxpy.cu with the environment's nvcc for sm_89 and time it with CUDA events."""

from __future__ import annotations

import ctypes
import shutil
import subprocess
from pathlib import Path

from ..envinfo import REPO_ROOT
from ..harness import Benchmark

KERNEL_SRC = Path(__file__).resolve().parent / "kernels" / "saxpy.cu"
BUILD_DIR = REPO_ROOT / "build" / "smoke"
ARCH = "sm_89"


def build_kernel() -> tuple[Path, list[str]]:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc not found on PATH; activate the sixgig environment (D-008)")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    lib = BUILD_DIR / "libsixgig_saxpy.so"
    cmd = [nvcc, "-O3", f"-arch={ARCH}", "-shared", "-Xcompiler", "-fPIC", "-o", str(lib), str(KERNEL_SRC)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"nvcc failed:\n{proc.stderr}")
    return lib, cmd


def run(quick: bool = False, suite: str = "phase0") -> Path:
    n = 1 << 25
    warmup, iters = (5, 20) if quick else (20, 200)
    with Benchmark("cuda_saxpy", suite=suite, params={"n": n, "dtype": "float32", "arch": ARCH}) as b:
        lib_path, cmd = build_kernel()
        b.params["nvcc_cmd"] = " ".join(Path(c).name if c.startswith("/") else c for c in cmd)
        lib = ctypes.CDLL(str(lib_path))
        fn = lib.sixgig_saxpy_bench
        fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        fn.restype = ctypes.c_int
        times = (ctypes.c_float * iters)()
        max_err = ctypes.c_float(0.0)
        rc = fn(n, warmup, iters, times, ctypes.byref(max_err))
        if rc != 0:
            raise RuntimeError(f"saxpy benchmark returned cudaError {rc}")
        if max_err.value > 1e-5:
            raise RuntimeError(f"saxpy result mismatch: max abs error {max_err.value}")
        b.add_samples(list(times), warmup=warmup, source="cuda_events")
        p50_s = sorted(times)[len(times) // 2] / 1e3
        # SAXPY reads x and y and writes y: 3 floats per element.
        b.metrics["effective_bandwidth_gbps_p50"] = 3 * 4 * n / p50_s / 1e9
        b.metrics["max_abs_error"] = max_err.value
    return b.output_path
