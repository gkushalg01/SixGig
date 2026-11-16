# Plan

**Project:** SixGig, a dual-machine robotics AI stack built to run in real time inside the RTX 4050's 6 GB of VRAM.

Related documents: [decisions.md](./decisions.md) · [glossary.md](./glossary.md) · [progress.md](./progress.md)

## Goal

Build a simulation-only robot manipulation stack that shows HPC and GPU skill in AI inference. It should work as a portfolio piece and as a way to learn the stack in depth. Every claim is backed by a reproducible benchmark (D-012).

## Hardware

| Machine | Role | Key facts |
|---------|------|-----------|
| RTX laptop | System 1: inference and control | i7-12650H, RTX 4050 6 GB (sm_89), 23 GB RAM, Ubuntu 22.04, driver 570 |
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

- **User action first:** enroll the Secure Boot key so the NVIDIA driver loads (steps in D-005). Done when `nvidia-smi` shows the RTX 4050.
- Create the `sixgig` conda environment with Python 3.12 (D-007), torch 2.14.1+cu126 (D-006) and the conda-forge CUDA 12.6 toolkit (D-008).
- Install and smoke-test TensorRT (D-009), MuJoCo and LIBERO (D-003), and ROS 2 Jazzy from RoboStack (D-010).
- Build a benchmark harness that records the metrics in D-012 and writes JSON to `results/`.
- Set up the Mac environment: conda, MLX, mlx-vlm, MuJoCo, RoboStack.

**Exit criteria:** a trivial CUDA kernel, a torch matmul, a TensorRT engine and a MuJoCo episode each run and produce a harness JSON file.

### Phase 1 (P1): VLA inference optimization

1. **Baseline:** run a small VLA (D-004) in PyTorch at FP32 and FP16. Record LIBERO success rate, latency percentiles and peak VRAM.
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
| The NVIDIA driver does not load | Enroll the Secure Boot key (D-005). This blocks all GPU work. |
| VLM backbone export to TensorRT fails | Use `torch.compile` plus quantization for the backbone and TensorRT for the rest. |
| TensorRT 11 wheel incompatible with driver 570 | Fall back to TensorRT 10.x for CUDA 12 (D-009). |
| RoboStack Jazzy is missing packages | Use Zenoh transport without ROS 2 for P5 (D-010). |
| Thermal noise in benchmarks | Follow the D-012 recording rules; run plugged in in performance mode. |
| Root disk is 82% full (21 GB free) | Store model weights and datasets in one cache directory and prune engines and checkpoints. |
