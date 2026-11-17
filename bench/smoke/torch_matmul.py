"""PyTorch GEMM throughput across precisions on the RTX 4050."""

from __future__ import annotations

from pathlib import Path

from ..harness import Benchmark

PRECISIONS = ("fp32", "tf32", "fp16", "bf16")


def run(quick: bool = False, suite: str = "phase0") -> list[Path]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    n = 4096
    warmup, iters = (5, 20) if quick else (20, 100)
    paths = []
    for prec in PRECISIONS:
        dtype = {"fp32": torch.float32, "tf32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[prec]
        torch.backends.cuda.matmul.allow_tf32 = prec == "tf32"
        torch.cuda.reset_peak_memory_stats()
        with Benchmark(f"torch_matmul_{prec}", suite=suite, params={"n": n, "precision": prec}) as b:
            a = torch.randn(n, n, device="cuda", dtype=dtype)
            w = torch.randn(n, n, device="cuda", dtype=dtype)
            out = torch.empty(n, n, device="cuda", dtype=dtype)
            samples = b.measure(lambda: torch.matmul(a, w, out=out), warmup=warmup, iters=iters, sync=torch.cuda.synchronize)
            p50_s = sorted(samples)[len(samples) // 2] / 1e3
            b.metrics["tflops_p50"] = 2 * n**3 / p50_s / 1e12
            b.metrics["torch_max_memory_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
            b.params["torch_cuda"] = torch.version.cuda
            b.params["device"] = torch.cuda.get_device_name(0)
            del a, w, out
        paths.append(b.output_path)
        torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = False
    return paths
