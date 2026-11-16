# Decisions

Each decision gets a numbered entry, and entries are never deleted. When a decision changes, add a new entry and mark the old one `Superseded by D-NNN`.

Status values:

- **Accepted:** decided and in effect.
- **Proposed:** the intended choice, still to be confirmed by a check in the named phase.
- **Superseded:** replaced by a later entry.

| ID | Decision | Status |
|----|----------|--------|
| [D-001](#d-001-simulation-only) | Simulation only | Accepted |
| [D-002](#d-002-flagship-scope) | Flagship scope: dual-machine stack from P1 and P2, P4 as stretch | Accepted |
| [D-003](#d-003-mujoco-based-simulation-not-isaac) | MuJoCo-based simulation, not Isaac Sim or Isaac Lab | Accepted |
| [D-004](#d-004-small-vla-models) | Small VLA models (about 0.5B to 3B parameters) | Accepted |
| [D-005](#d-005-keep-nvidia-driver-570-and-enroll-the-secure-boot-key) | Keep NVIDIA driver 570 and enroll the Secure Boot key | Accepted |
| [D-006](#d-006-pytorch-2141-with-cuda-126) | PyTorch 2.14.1 with CUDA 12.6 | Accepted |
| [D-007](#d-007-python-312-in-a-conda-environment) | Python 3.12 in a conda environment | Accepted |
| [D-008](#d-008-cuda-126-toolkit-inside-the-conda-environment) | CUDA 12.6 toolkit inside the conda environment | Proposed |
| [D-009](#d-009-tensorrt-from-the-cuda-12-wheel) | TensorRT from the CUDA 12 wheel | Proposed |
| [D-010](#d-010-ros-2-jazzy-through-robostack-on-both-machines) | ROS 2 Jazzy through RoboStack on both machines | Proposed |
| [D-011](#d-011-split-work-across-the-two-machines-by-strength) | Split work across the two machines by strength | Accepted |
| [D-012](#d-012-benchmark-discipline) | Benchmark discipline | Accepted |
| [D-013](#d-013-living-project-documents) | Living project documents in `docs/` | Accepted |

---

## D-001: Simulation only

- **Date:** 2026-11-17
- **Status:** Accepted
- **Context:** No physical robot is available.
- **Decision:** Everything runs in simulation. Results are reported as simulated task success and measured compute performance.
- **Alternatives considered:** Buying a low-cost arm (SO-101 class). Rejected for now to keep the focus on compute.
- **Consequences:** No sim-to-real claims. Hardware-specific ROS 2 drivers are out of scope.

## D-002: Flagship scope

- **Date:** 2026-11-17
- **Status:** Accepted
- **Context:** The goals are a portfolio piece that shows HPC and GPU skill, and deeper knowledge of the stack. Six candidate projects were shortlisted (P1 to P6).
- **Decision:** Build P5, a dual-machine stack, from two components that also stand alone: P1 (VLA inference optimization) and P2 (a custom CUDA MPPI controller). P4 (GPU perception pipeline) is the only stretch goal.
- **Alternatives considered:**
  - A single project in more depth. Rejected because the flagship covers more of the stated interests while P1 and P2 still provide depth.
  - P3 (GPU-parallel RL). Dropped: many similar public repos exist.
  - P6 (Gaussian splatting SLAM). Dropped: memory-hungry on 6 GB and the least related to inference.
- **Consequences:** P1 and P2 each produce publishable results, so the project remains presentable if the integration phase slips.

## D-003: MuJoCo-based simulation, not Isaac

- **Date:** 2026-11-17
- **Status:** Accepted
- **Context:** The RTX 4050 has 6 GB of VRAM, which is below the documented minimum for Isaac Sim and Isaac Lab. The Mac has no CUDA at all.
- **Decision:** Use MuJoCo (4.0.1 on PyPI as of 2026-11-17), plus a MuJoCo-based manipulation benchmark (LIBERO or equivalent).
- **Alternatives considered:**
  - Isaac Sim or Isaac Lab. Rejected on VRAM.
  - Genesis or ManiSkill. Kept as fallbacks if LIBERO integration fails.
- **Consequences:** The simulator runs on the CPU and can be hosted on the Mac, which leaves the full 6 GB of VRAM for inference.

## D-004: Small VLA models

- **Date:** 2026-11-17
- **Status:** Accepted
- **Context:** A 7B-class VLA fits in 6 GB only with 4-bit quantization and runs too slowly for closed-loop control.
- **Decision:** Target small VLAs (SmolVLA-class, about 0.5B parameters) and quantized models of about 3B. Phase 0 picks the exact model from what lerobot 0.7.0 supports.
- **Alternatives considered:** A 4-bit 7B VLA. Rejected for latency, but kept as one data point on the Pareto chart if it fits.
- **Consequences:** The project's angle is "real-time on 6 GB", not "biggest model".

## D-005: Keep NVIDIA driver 570 and enroll the Secure Boot key

- **Date:** 2026-11-17
- **Status:** Accepted
- **Context:** Driver 570.124.04 is installed and was built by DKMS for kernel 6.8.0-57. Secure Boot is enabled. The module is signed with a local key (`/var/lib/shim-signed/mok/MOK.der`) that is not enrolled, so the kernel rejects it ("Key was rejected by service") and `nvidia-smi` fails.
- **Decision:** Keep driver 570 and enroll the existing key. The user does this because it needs sudo and a reboot:
  1. Run `sudo mokutil --import /var/lib/shim-signed/mok/MOK.der` and set a one-time password.
  2. Reboot, then in the blue MOK Manager screen choose Enroll MOK, then Continue, then Yes, and enter the password.
  3. After the reboot, check with `mokutil --test-key /var/lib/shim-signed/mok/MOK.der` and `nvidia-smi`.
- **Alternatives considered:**
  - Upgrade the driver to 580 or newer for CUDA 13. This still needs the key enrolled, and adds risk.
  - Disable Secure Boot in the BIOS. This weakens the security of the machine.
- **Consequences:** CUDA support tops out at 12.8, which constrains D-006, D-008 and D-009. Future kernel updates are re-signed with the same key automatically.

## D-006: PyTorch 2.14.1 with CUDA 12.6

- **Date:** 2026-11-17
- **Status:** Accepted
- **Context:**
  - PyTorch 2.15.0 ships only CUDA 13.2 and 13.4 builds, which need a newer driver than 570.
  - Driver 570 supports CUDA up to 12.8.
  - The newest build at or below 12.8 is torch 2.14.1+cu126; cu129 builds need driver 575 or newer.
  - lerobot 0.7.0 requires torch between 2.7 and 2.16.
- **Decision:** Install torch 2.14.1+cu126 with the matching torchvision.
- **Alternatives considered:** torch 2.15 with CUDA 13. Requires a driver upgrade; rejected in D-005.
- **Consequences:** The stack is one minor version behind the newest PyTorch, which does not matter for this project.

## D-007: Python 3.12 in a conda environment

- **Date:** 2026-11-17
- **Status:** Accepted
- **Context:**
  - System Python is 3.10.12.
  - lerobot 0.7.0 requires Python 3.12 or newer.
  - torch 2.15, MuJoCo 4.0.1 and transformers 5.x require 3.11 or newer.
  - Miniconda is already installed.
- **Decision:** Create a conda environment named `sixgig` with Python 3.12 on the RTX laptop, and an equivalent one on the Mac.
- **Alternatives considered:**
  - uv virtual environments. Not installed, and conda also supplies the non-Python dependencies D-008 and D-010 need.
  - Docker. Adds friction for GPU profiling.
- **Consequences:** Nothing is installed into system Python, and no sudo is needed for packages.

## D-008: CUDA 12.6 toolkit inside the conda environment

- **Date:** 2026-11-17
- **Status:** Proposed, confirmed in Phase 0
- **Context:** The system toolkit at `/usr/local/cuda` is 12.1, while torch uses the CUDA 12.6 runtime. Custom kernels for P2 and P4 should be built with a matching `nvcc`, and Ada GPUs (sm_89) are supported from 12.1 onward.
- **Decision:** Install `cuda-nvcc` and the CUDA 12.6 libraries from conda-forge into the `sixgig` environment. Builds target `sm_89`.
- **Alternatives considered:**
  - Use the system 12.1 toolkit. Mismatched versions risk ABI problems when building torch extensions.
  - Install 12.6 system-wide. Needs sudo.
- **Consequences:** CMake must find the environment's toolkit, not `/usr/local/cuda`.

## D-009: TensorRT from the CUDA 12 wheel

- **Date:** 2026-11-17
- **Status:** Proposed, confirmed in Phase 0
- **Context:** TensorRT 11.4.0.106 is published as both `tensorrt-cu12` and `tensorrt-cu13`. Driver 570 supports CUDA 12 only.
- **Decision:** Use `tensorrt-cu12` 11.4 from pip. Phase 0 checks that it runs on driver 570 and with the CUDA 12.6 runtime. If not, fall back to the newest TensorRT 10.x release built for CUDA 12.
- **Alternatives considered:** Installing TensorRT from the system tarball or apt. Needs sudo and is harder to reproduce.
- **Consequences:** The `trtexec` command-line tool may not come with the wheel, so engines are built from Python or C++.

## D-010: ROS 2 Jazzy through RoboStack on both machines

- **Date:** 2026-11-17
- **Status:** Proposed, confirmed in Phase 0
- **Context:**
  - ROS 2 is not installed.
  - Native ROS 2 Humble on Ubuntu 22.04 is tied to Python 3.10, which conflicts with D-007.
  - Native ROS 2 on macOS is impractical.
- **Decision:** Install ROS 2 Jazzy from RoboStack (conda-forge packages) into the conda environments on both the Linux and Mac machines.
- **Alternatives considered:**
  - Native Humble with a separate ML process bridged over Zenoh. Two Python worlds to maintain.
  - Docker with Ubuntu 24.04 and Jazzy. Awkward on the Mac.
- **Consequences:** One environment definition per machine. If RoboStack packages are missing, the fallback is a ROS-free transport (Zenoh) for P5, with ROS 2 used only on Linux.

## D-011: Split work across the two machines by strength

- **Date:** 2026-11-17
- **Status:** Accepted
- **Context:**
  - The RTX 4050 has CUDA and TensorRT but only 6 GB of VRAM.
  - The M4 Pro has 24 GB of unified memory, MLX and strong CPU cores, but no CUDA.
- **Decision:**
  - The Mac runs the simulator, camera rendering, and the VLM planner (System 2) on MLX.
  - The RTX laptop runs only inference and control (System 1): the optimized VLA, CUDA MPPI and, as a stretch, perception.
  - The machines are linked over wired LAN.
- **Alternatives considered:** Everything on one machine. That would waste the Mac and push the planner into the 6 GB VRAM.
- **Consequences:** End-to-end latency includes the network, so it must be measured as a separate stage.

## D-012: Benchmark discipline

- **Date:** 2026-11-17
- **Status:** Accepted
- **Context:** Laptop GPUs throttle with temperature and power mode, so unmanaged runs give noisy numbers.
- **Decision:** Every benchmark run records:
  - the git commit and environment versions;
  - power mode, plugged-in state, GPU clocks, temperature and power draw;
  - warm-up iterations, which are excluded from results;
  - latency p50, p95 and p99, peak VRAM and, where relevant, task success over a fixed set of seeds.

  Results are written as JSON under `results/` and plotted from there.
- **Alternatives considered:** Ad-hoc timing. Not reproducible.
- **Consequences:** Every claim in the README links to a results file.

## D-013: Living project documents

- **Date:** 2026-11-17
- **Status:** Accepted
- **Context:** The user asked for a decision log, a glossary, a plan and a progress log.
- **Decision:** Keep four documents in `docs/`: `decisions.md` (numbered records with context, decision, alternatives and consequences), `glossary.md`, `plan.md` and `progress.md`.
  - Any new choice gets a decision entry.
  - Any new component or technique gets a glossary entry.
  - Any finished step or blocker updates `progress.md`.
  - Any scope change updates `plan.md`.
- **Alternatives considered:** Documents at the repo root, or a compact decision table only. The user chose `docs/` and the numbered format.
- **Consequences:** Documentation is updated in the same commit as the change it describes.
