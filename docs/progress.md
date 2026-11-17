# Progress

Newest entries first. Each entry says what was done, what was found, and what comes next. Decision IDs refer to [decisions.md](./decisions.md).

## Status

| Todo | Phase | Status | Blocked by |
|------|-------|--------|------------|
| Fix the GPU driver (user action) | 0 | Done | — |
| Scaffold the repo and documents | 0 | Done | — |
| Set up the environment and benchmark harness (RTX laptop) | 0 | Pending | — |
| Set up the Mac environment | 0 | Pending | Repo scaffold |
| Baseline a small VLA | 1 | Pending | Environment and harness |
| Optimize VLA inference | 1 | Pending | VLA baseline |
| Port the VLA to MLX | 1 | Pending | VLA baseline, Mac environment |
| Build the CUDA MPPI controller | 2 | Pending | Environment and harness |
| Integrate the dual-machine stack | 3 | Pending | VLA optimization, CUDA MPPI, Mac environment |
| GPU perception pipeline (stretch) | Stretch | Pending | Environment and harness |
| Write-up and demo | Final | Pending | Dual-machine integration |

## Log

### 2026-11-18: GPU driver working after a display freeze

**Done**

- The user got the NVIDIA driver loading. `nvidia-smi` shows the RTX 4050 Laptop GPU with 6141 MiB, driver 570.124.04 and CUDA 12.8. The display server uses about 4 MiB of it.
- Verified the machine state:
  - The signing key is enrolled, and Secure Boot is disabled (D-014).
  - Kernel parameters `i915.force_probe=46a3 modprobe.blacklist=xe` are set permanently in GRUB (D-015).
  - `i915` drives the Intel iGPU (`card1`) and `nvidia` drives the RTX 4050 (`card2`). PRIME mode is on-demand.
- Recorded D-014 and D-015, marked D-005 as partly superseded, and added glossary entries for `i915`/`xe`, kernel parameters and PRIME on-demand.

**Found**

- **Cause of the freeze:** boot logs from 2026-11-17 to 18 show `xe` trying to claim the Alder Lake-P iGPU (46a3), which it does not officially support on kernel 6.8, followed by hung tasks and a boot that stopped logging after about 14 seconds. Blacklisting `xe` fixed it.
- `xe.blacklist=1`, tried in one boot, is not a valid kernel parameter and was ignored. `modprobe.blacklist=xe` is the correct form.
- **Power readings spike:** `nvidia-smi` sometimes reports 590 W against a 60 W cap. Of three samples one second apart, the first read 590.01 W and the next two read about 12.6 W. So the harness (D-012) must reject impossible power samples instead of averaging them. Compute capability is confirmed as 8.9 (sm_89).

**Next**

- Create the `sixgig` conda environment and build the benchmark harness.

### 2026-11-17: repository scaffold

**Done**

- Created the git repo at `/home/ats/workspace/SixGig` with the planned layout (`docs/`, `env/`, `bench/`, `vla/`, `mppi/`, `perception/`, `system/`, `results/`, `scripts/`).
- Added `README.md` and a `.gitignore` that keeps model weights, TensorRT engines, ONNX files, build outputs and profiler captures out of git.
- Added the four project documents to `docs/` (D-013).

**Next**

- User: enroll the Secure Boot key (D-005). All GPU work in Phase 0 waits on this.
- After the driver loads: create the `sixgig` conda environment and build the benchmark harness.

### 2026-11-17: planning and environment audit

**Done**

- Shortlisted six projects. Chose the flagship (P5 built from P1 and P2) with P4 as the stretch goal (D-002).
- Audited the RTX laptop:
  - Ubuntu 22.04.5, kernel 6.8.0-57, i7-12650H, 23 GB RAM.
  - 21 GB free on the root disk (82% used).
  - CUDA toolkit 12.1 at `/usr/local/cuda`; cuDNN 9.8 present.
  - No ROS 2. No TensorRT command-line tool. Only the `base` conda environment exists.
- Surveyed PyPI versions as of today:

  | Package | Version |
  |---------|---------|
  | torch | 2.15.0 (CUDA 13 builds only); newest CUDA 12 build is 2.14.1+cu126 |
  | lerobot | 0.7.0 (Python 3.12+, torch below 2.16) |
  | mujoco, mujoco-mjx, mujoco-warp | 4.0.1 |
  | tensorrt-cu12, tensorrt-cu13 | 11.4.0.106 |
  | transformers | 5.23.0 |
  | onnxruntime-gpu | 1.32.0 |
  | mlx | 0.32.4 |
  | mlx-vlm | 0.7.12 |
  | warp-lang | 1.19.0 |
  | eclipse-zenoh | 1.10.2 |

- Recorded decisions D-001 to D-013.

**Found**

- **Blocker:** the NVIDIA driver (570.124.04) does not load. Secure Boot is on, and the DKMS-built module is signed with a key that is not enrolled ("Key was rejected by service"). The fix is in D-005 and needs sudo plus a reboot.
- System Python is 3.10, below what lerobot, torch 2.15 and MuJoCo 4 need. A conda environment is required (D-007).

**Next**

- User: enroll the Secure Boot key and confirm that `nvidia-smi` works.
- Scaffold the repository with these documents (done in the entry above).
