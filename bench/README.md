# Benchmark harness

Shared benchmarking code for every SixGig phase. It applies the rules in [D-012](../docs/decisions.md#d-012-benchmark-discipline) so that every number in the project can be reproduced and traced to a JSON file.

## Usage

```bash
conda activate sixgig
cd /path/to/SixGig

python -m bench.smoke            # run all Phase 0 smoke workloads
python -m bench.smoke --quick    # shorter runs
python -m bench.smoke --only cuda,trt
python -m pytest bench/tests -q  # harness unit tests
```

To benchmark your own workload:

```python
import torch
from bench import Benchmark

with Benchmark("vla_smolvla_fp16", suite="phase1", params={"denoise_steps": 10}) as b:
    b.measure(lambda: policy(batch), warmup=20, iters=200, sync=torch.cuda.synchronize)
    b.metrics["actions_per_s"] = ...
    b.set_task_result(successes=41, episodes=50, seeds=range(50))
# The JSON file is written when the block exits, including when it exits with an error.
```

## What each record contains

Records are written to `results/<suite>/<name>-<timestamp>.json`.

| Field | Contents |
|-------|----------|
| `latency_ms` | count, mean, std, min, p50, p95, p99 and max of the timed iterations. Warm-up iterations are excluded. |
| `latency_ms_samples` | every timed sample, so the statistics can be recomputed later |
| `latency_source` | how latency was timed: `host_clock_synced` (host clock with a device synchronize), `host_clock`, or `cuda_events` |
| `gpu_telemetry` | NVML readings sampled every 50 ms during the run: power, temperature, SM and memory clocks, utilization, peak memory used, performance states, and clock-limit reasons such as thermal or power cap |
| `environment` | git commit and dirty flag, package versions, nvcc, driver, compute capability, AC power, platform profile, CPU governor |
| `metrics`, `task` | workload-specific numbers, and task success over the listed seeds |

**Power readings.** This laptop sometimes reports about 590 W against a 60 W cap. Any reading above 1.5 times the enforced limit is dropped from the statistics and counted in `power_samples_rejected` ([telemetry.py](telemetry.py)).

## Phase 0 smoke workloads

| Workload | What it proves | Headline metric |
|----------|----------------|-----------------|
| `cuda` | The environment's nvcc builds an sm_89 kernel, and it runs correctly ([kernels/saxpy.cu](smoke/kernels/saxpy.cu)) | Effective memory bandwidth (GB/s) |
| `torch` | PyTorch CUDA GEMM in FP32, TF32, FP16 and BF16 | TFLOPS |
| `trt` | TensorRT builds and runs an FP16 engine whose output matches PyTorch | Latency, engine build time |
| `mujoco` | MuJoCo steps a 7-DoF arm episode stably | Steps per second, real-time factor |
