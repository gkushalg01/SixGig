"""SixGig benchmark harness (D-012).

Every run records environment versions, machine power state, GPU telemetry,
warm-up count and latency percentiles, and writes one JSON file per run.
"""

from .harness import Benchmark, RESULTS_DIR

__all__ = ["Benchmark", "RESULTS_DIR"]
