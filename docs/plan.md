# Plan

**Project:** SixGig, a dual-machine robotics AI stack built to run in real time inside the RTX 4050's 6 GB of VRAM.

Related documents: [decisions.md](./decisions.md) · [glossary.md](./glossary.md) · [progress.md](./progress.md)

## Goal

Build a simulation-only robot manipulation stack that shows HPC and GPU skill in AI inference. It should work as a portfolio piece and as a way to learn the stack in depth. Every claim is backed by a reproducible benchmark (D-012).

## Hardware

| Machine | Role | Key facts |
|---------|------|-----------|
| RTX laptop | System 1: inference and control | i7-12650H, RTX 4050 6 GB (sm_89), 23 GB RAM, Ubuntu 22.04, driver 570 (CUDA 12.8), boot fix in D-015 |
| MacBook M4 Pro | System 2 and simulator | 24 GB unified memory, MLX, no CUDA |

## Architecture

```text
 MacBook M4 Pro                                RTX 4050 laptop (6 GB VRAM)
┌──────────────────────────────┐   wired     ┌──────────────────────────────────┐
│ MuJoCo sim + camera render   │── images ──▶│ [P4] TensorRT perception         │
│                              │   state     │ [P1] optimized VLA ─▶ chunk      │
│ VLM planner on MLX (System 2)│── subtask ─▶│        │ sampling prior               │
│   language goal ─▶ subtasks  │             │        ▼                         │
│                              │◀─ actions ──│ [P2] CUDA MPPI (System 1)        │
└──────────────────────────────┘             └──────────────────────────────────┘
```

## Components

| ID | Component | Stands alone as | Phase |
|----|-----------|-----------------|-------|
| P1 | VLA inference optimization | A repo or folder with a Pareto chart on two machines | 1 |
| P2 | CUDA MPPI controller | A C++/CUDA library and ROS 2 node with scaling and profiling results | 2 |
| P5 | Dual-machine integration | The flagship demo | 3 |
| P4 | GPU perception pipeline | Stretch goal | Stretch |

P3 (GPU-parallel RL) and P6 (Gaussian splatting SLAM) are out of scope (D-002).

## Phases

### Phase 0: environment and benchmark harness

- **Driver fix (done 2026-11-18):** the NVIDIA driver loads and `nvidia-smi` shows the RTX 4050 (D-014, D-015).
- **Done 2026-11-18:** created the `sixgig` conda environment from [`env/linux-64.yml`](../env/linux-64.yml) with Python 3.12 (D-007), torch 2.14.1+cu126 (D-006), the conda-forge CUDA 12.6 toolkit (D-008) and TensorRT 11.4 (D-009).
- **Done 2026-11-18:** built the benchmark harness in [`bench/`](../bench/) (D-016). All four smoke workloads pass and their records are in [`results/phase0/`](../results/phase0/).
- **Remaining:** LIBERO (D-003). Done 2026-11-18 at the start of Phase 1 (D-019, D-021, D-024).
- **Remaining:** ROS 2 Jazzy from RoboStack (D-010), which moves to the start of Phase 2 where the MPPI node first needs it.
- **Remaining:** the Mac environment (conda, MLX, mlx-vlm, MuJoCo, RoboStack), which has to be run on the Mac.

**Exit criteria (met 2026-11-18):** a trivial CUDA kernel, a torch matmul, a TensorRT engine and a MuJoCo episode each run and produce a harness JSON file.

### Phase 1 (P1): VLA inference optimization

Model: `lerobot/smolvla_libero` through lerobot 0.7.0 (D-021), evaluated with the `lerobot-eval` wrapper in [`vla/libero_eval.py`](../vla/libero_eval.py) (D-023).

1. **Baseline (done 2026-11-18):** run the checkpoint as stored (`native`: BF16 backbone with FP32 projections) and the `fp32`, `tf32` and `fp16` modes (D-022). Record LIBERO success rate over the four suites, in-loop and microbenchmark latency percentiles, and peak VRAM. Result: `native` scores 63.5% at 142.6 ms per chunk (p50, in-loop) and 923 MiB. Other precisions do not change success but are slower. Records are in [`results/phase1/`](../results/phase1/).
2. **Optimize step by step,** with each step a separate result file:
   - quantization: INT8, then 4-bit;
   - TensorRT export of the vision encoder and action head (fallback: `torch.compile` plus quantization for the language backbone);
   - fewer denoising steps;
   - prefix caching;
   - asynchronous action chunking.
3. **Port to MLX** on the Mac and compare.

**Deliverable:** a Pareto chart of success rate against latency and VRAM, for both machines.

### Phase 2 (P2): CUDA MPPI controller

1. CUDA kernels in a structure-of-arrays layout for batched rollouts of a 7-DoF arm (Franka Panda kinematics), plus a cost kernel covering goal distance, joint limits, collisions and smoothness.
2. An MPPI update step with a C++ API, wrapped as a ROS 2 node.
3. Baselines: a multithreaded CPU MPPI and, if available, an existing GPU library.
4. Optimization with Nsight Systems and Nsight Compute: memory layout, kernel fusion, precision, CUDA graphs. Each step is documented with before-and-after numbers.

**Deliverable:** curves of rollouts per second and control rate against the number of samples, a roofline view, and a profiling write-up.

### Phase 3 (P5): dual-machine integration

1. The Mac hosts the MuJoCo sim and an MLX VLM planner that turns a language goal into JSON subtasks through constrained decoding.
2. The RTX laptop runs the optimized VLA. Its action chunk becomes the sampling prior for CUDA MPPI, which enforces constraints.
3. Transport uses ROS 2 over Zenoh on wired LAN.

**Deliverable:** a per-stage latency budget, long-horizon task success, and a demo video.

### Stretch (P4): GPU perception pipeline

- TensorRT open-vocabulary detection, depth and segmentation, with CUDA pre- and post-processing.
- Zero-copy ROS 2 transport through composable nodes.
- A scheduler that keeps perception, the VLA and MPPI within the 6 GB budget together.

**Deliverable:** a per-stage latency and VRAM table with all models loaded.

### Final: write-up

- A README with reproducible benchmark commands.
- A combined report and the demo video.
- Every chart linked to its JSON file in `results/`.

## Repository layout (planned)

```text
SixGig/
├── README.md
├── docs/            decisions.md, glossary.md, plan.md, progress.md
├── env/             conda environment files for linux-64 and osx-arm64
├── bench/           benchmark harness (Phase 0)
├── vla/             P1: baselines, export, quantization, MLX port
├── mppi/            P2: CUDA/C++ library, CMake, ROS 2 node, CPU baseline
├── perception/      P4 (stretch)
├── system/          P5: launch files, planner, bridges
├── results/         benchmark JSON and plots
└── scripts/
```

## Risks

| Risk | Mitigation |
|------|------------|
| The NVIDIA driver stops loading, or the display freezes again, after a kernel or driver update | Re-check `nvidia-smi` and which driver owns each card after every update. Keep the D-015 kernel parameters. |
| VLM backbone export to TensorRT fails | Use `torch.compile` plus quantization for the backbone and TensorRT for the rest. |
| TensorRT 11 wheel incompatible with driver 570 | Resolved 2026-11-18: TensorRT 11.4 (cu12) builds and runs on driver 570 (D-009). |
| RoboStack Jazzy is missing packages | Use Zenoh transport without ROS 2 for P5 (D-010). |
| Thermal noise in benchmarks | Follow the D-012 recording rules; run plugged in in performance mode. |
| Disk space: 56 GB free after the 15 GB `sixgig` environment; VLA weights, datasets and engines will grow | Store model weights and datasets in one cache directory and prune engines and checkpoints. |
| GPU throughput is capped by the 60 W board power limit, not by compute units | Record power and clock-limit reasons in every run (D-016), and report results as power-limited where they are. |
