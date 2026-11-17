"""Latency statistics without numpy, so the harness has no hard dependencies."""

from __future__ import annotations

import math
from typing import Sequence


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (same method as numpy's default) of pre-sorted data."""
    if not sorted_values:
        raise ValueError("percentile of empty sequence")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be in [0, 100], got {q}")
    pos = (len(sorted_values) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def summarize(samples: Sequence[float]) -> dict:
    """Summary used for every latency result: count, mean, std, min, p50, p95, p99, max."""
    if not samples:
        raise ValueError("cannot summarize zero samples")
    s = sorted(float(x) for x in samples)
    n = len(s)
    mean = sum(s) / n
    var = sum((x - mean) ** 2 for x in s) / (n - 1) if n > 1 else 0.0
    return {
        "count": n,
        "mean": mean,
        "std": math.sqrt(var),
        "min": s[0],
        "p50": percentile(s, 50),
        "p95": percentile(s, 95),
        "p99": percentile(s, 99),
        "max": s[-1],
    }
