"""Benchmark runner that writes one JSON record per run (D-012).

Usage:

    with Benchmark("torch_matmul_fp16", suite="phase0", params={"n": 4096}) as b:
        b.measure(step, warmup=20, iters=200, sync=torch.cuda.synchronize)
        b.metrics["tflops"] = ...
    # JSON written on exit to results/phase0/torch_matmul_fp16-<timestamp>.json

Latencies are in milliseconds. Warm-up iterations are run but never recorded.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

from . import envinfo
from .stats import summarize
from .telemetry import GpuMonitor

RESULTS_DIR = envinfo.REPO_ROOT / "results"
SCHEMA_VERSION = 1


class Benchmark:
    def __init__(
        self,
        name: str,
        suite: str,
        params: Optional[dict] = None,
        results_dir: Optional[Path] = None,
        monitor_gpu: bool = True,
        gpu_sample_interval_s: float = 0.05,
    ):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in (".", ".."):
            raise ValueError("name may only contain letters, digits, '_', '.', '-'")
        # Suites may nest (e.g. "tmp/quick" lands in the gitignored results/tmp/), but never traverse.
        if not re.fullmatch(r"[A-Za-z0-9_-]+(/[A-Za-z0-9_-]+)*", suite):
            raise ValueError("suite must be '/'-separated segments of letters, digits, '_', '-'")
        self.name = name
        self.suite = suite
        self.params = dict(params or {})
        self.results_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR
        self.metrics: dict = {}
        self.notes: list[str] = []
        self.warmup = 0
        self.latency_ms: list[float] = []
        self.latency_source: Optional[str] = None
        self.task: Optional[dict] = None
        self.status = "ok"
        self.error: Optional[str] = None
        self.output_path: Optional[Path] = None
        self._monitor = GpuMonitor(interval_s=gpu_sample_interval_s) if monitor_gpu else None
        self._started_at: Optional[dt.datetime] = None
        self._wall_s: Optional[float] = None
        self._t0 = 0.0

    def __enter__(self) -> "Benchmark":
        self._started_at = dt.datetime.now(dt.timezone.utc).astimezone()
        self._t0 = time.perf_counter()
        if self._monitor is not None:
            self._monitor.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._wall_s = time.perf_counter() - self._t0
        if self._monitor is not None:
            self._monitor.stop()
        if exc is not None:
            self.status = "error"
            self.error = f"{exc_type.__name__}: {exc}"
        self.save()
        return False

    def measure(
        self,
        fn: Callable[[], object],
        warmup: int = 10,
        iters: int = 100,
        sync: Optional[Callable[[], object]] = None,
    ) -> list[float]:
        """Time fn() per call with a host clock. Pass sync (e.g. torch.cuda.synchronize) for GPU work."""
        if iters <= 0:
            raise ValueError("iters must be positive")
        for _ in range(warmup):
            fn()
        if sync is not None:
            sync()
        samples = []
        for _ in range(iters):
            t0 = time.perf_counter_ns()
            fn()
            if sync is not None:
                sync()
            samples.append((time.perf_counter_ns() - t0) / 1e6)
        self.warmup = warmup
        self.latency_ms = samples
        self.latency_source = "host_clock_synced" if sync is not None else "host_clock"
        return samples

    def add_samples(self, samples_ms: Sequence[float], warmup: int, source: str) -> None:
        """Record latencies measured elsewhere, e.g. with CUDA events inside a kernel benchmark."""
        if not samples_ms:
            raise ValueError("no samples")
        self.latency_ms = [float(x) for x in samples_ms]
        self.warmup = warmup
        self.latency_source = source

    def set_task_result(self, successes: int, episodes: int, seeds: Sequence[int]) -> None:
        if episodes <= 0 or not 0 <= successes <= episodes:
            raise ValueError("invalid task result")
        self.task = {
            "successes": successes,
            "episodes": episodes,
            "success_rate": successes / episodes,
            "seeds": list(seeds),
        }

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "suite": self.suite,
            "status": self.status,
            "error": self.error,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "wall_time_s": self._wall_s,
            "params": self.params,
            "warmup_iters": self.warmup,
            "latency_source": self.latency_source,
            "latency_ms": summarize(self.latency_ms) if self.latency_ms else None,
            "latency_ms_samples": self.latency_ms,
            "task": self.task,
            "metrics": self.metrics,
            "gpu_telemetry": self._monitor.summary() if self._monitor is not None else None,
            "environment": envinfo.collect(),
            "notes": self.notes,
        }

    def save(self) -> Path:
        stamp = (self._started_at or dt.datetime.now().astimezone()).strftime("%Y%m%dT%H%M%S")
        out_dir = self.results_dir / self.suite
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.name}-{stamp}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str) + "\n")
        self.output_path = path
        return path
