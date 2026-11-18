# Progress

Newest entries first. Each entry says what was done, what was found, and what comes next. Decision IDs refer to [decisions.md](./decisions.md).

## Resume here

The next session starts from this list. Everything below it is history. Frequently used commands are in [commands.md](./commands.md).

1. **Activate the environment:** `conda activate sixgig` (or prefix commands with `conda run -n sixgig`), then `cd /home/ats/workspace/SixGig`. If `~/.libero/config.yaml` is missing, run `python scripts/setup_libero.py`.
2. **Phase 1 baseline is done** (see the 2026-11-18 log entries below). `native` is the reference: 63.5% success, 142.6 ms p50 per chunk in the loop, 923 MiB peak.
3. **Phase 1 optimization has started** ([plan.md](./plan.md), Phase 1 step 2). The wrapper now takes `--policy-override KEY=VALUE` (D-023). Each step gets its own result file, run with the full 400 episodes (about 35 to 40 minutes per run) and compared with `native` using `python scripts/summarize_results.py`. Order, cheapest first:
   - fewer denoising steps: `num_steps=5` (running or done, see the log), then `num_steps=3` or `2` depending on its success rate;
   - `n_action_steps` below 50 (try 25 and 10), which may also raise success (D-021);
   - `torch.compile`, then INT8 and 4-bit quantization, then TensorRT. These need new wrapper hooks, not just config overrides.
4. **Commit only from a clean tree before long runs.** Records capture the git state when they are saved, so editing tracked files during a run marks its records dirty.
5. **Mac:** set up the environment now, in parallel with the RTX runs. See **Mac plan** below.

Still open from Phase 0: ROS 2 (Phase 2).

## Mac plan

What to run on the MacBook M4 Pro, and when. The Mac and the RTX laptop work in parallel until Phase 3.

| When | Mac task | Depends on | Hands back to the RTX side |
|------|----------|------------|----------------------------|
| **Now** | Install Miniforge, clone the repo over SSH, create the environment from [`env/osx-arm64.yml`](../env/osx-arm64.yml) (draft), and run the Mac checks in [commands.md](./commands.md#mac-macbook-m4-pro): MPS, MLX, MuJoCo `cgl` rendering, harness tests | Nothing | Versions and any install errors, to finalize the env file and record the Mac environment in the docs |
| After the environment works | SmolVLA on LIBERO with PyTorch on MPS: a latency-only check, then the full 400 episodes, as the Mac's reference point | The wrapper must accept `--device mps` and `MUJOCO_GL=cgl` (small change, done on request) | Mac baseline records for the Pareto chart |
| After the Mac baseline | Port SmolVLA to MLX (Phase 1 step 3): the vision encoder and language backbone first through mlx-vlm, then the action expert; compare success and latency with MPS and with the RTX | Mac baseline | MLX records for the Pareto chart |
| Before Phase 3 (can be done any time) | Install ROS 2 Jazzy from RoboStack into the Mac environment and check `ros2 topic list` (D-010) | Mac environment | Whether D-010 holds on macOS, or Phase 3 falls back to Zenoh |
| Before Phase 3 | Connect the two machines with a wired LAN cable or adapter and measure round-trip time with `ping` | Nothing | Network latency for the Phase 3 budget (D-011) |
| Phase 3 | Run the MuJoCo sim, camera rendering and the MLX planner on the Mac, talking to the RTX laptop | Phase 1 optimization, Phase 2 CUDA MPPI | — |

## Status

| Todo | Phase | Status | Blocked by |
|------|-------|--------|------------|
| Fix the GPU driver (user action) | 0 | Done | — |
| Scaffold the repo and documents | 0 | Done | — |
| Set up the environment and benchmark harness (RTX laptop) | 0 | Done | — |
| Install LIBERO | 1 | Done | — |
| Install ROS 2 Jazzy from RoboStack (RTX laptop) | 2 | Pending | — |
| Set up the Mac environment | 0 | In progress: env file drafted, not yet created | Must be run on the Mac |
| Baseline a small VLA | 1 | Done | — |
| Optimize VLA inference | 1 | In progress | — |
| Port the VLA to MLX | 1 | Pending | VLA baseline, Mac environment |
| Build the CUDA MPPI controller | 2 | Pending | ROS 2 (for the node only; the CUDA library can start now) |
| Integrate the dual-machine stack | 3 | Pending | VLA optimization, CUDA MPPI, Mac environment |
| GPU perception pipeline (stretch) | Stretch | Pending | Environment and harness |
| Write-up and demo | Final | Pending | Dual-machine integration |

## Log

### 2026-11-18: Phase 1 optimization started: policy overrides, commands reference, Mac plan

**Done**

- Added `--policy-override KEY=VALUE` and `--tag` to [`vla/libero_eval.py`](../vla/libero_eval.py) (D-023). Overrides are forwarded to `lerobot-eval` as `--policy.KEY=VALUE`, and record names get a suffix such as `_num_steps5`.
- Checked it with a latency-only run of `num_steps=5` into `results/tmp/dev/`: the record reports `denoise_steps: 5`, and the microbenchmark p50 fell from 145.9 ms to 80.0 ms. The 3 short episodes say nothing about success yet.
- Added [`scripts/summarize_results.py`](../scripts/summarize_results.py), which prints one line per record: success, p50/p99, peak memory, denoising steps, `n_action_steps` and commit.
- Added [commands.md](./commands.md), a reference of the commands used in the project.
- Drafted [`env/osx-arm64.yml`](../env/osx-arm64.yml) for the Mac: Python 3.12, torch 2.14.1 (MPS), MuJoCo 3.8.1, lerobot 0.7.0, mlx 0.32.4 and mlx-vlm 0.7.12. All pinned versions have macOS arm64 wheels for Python 3.12 (torch needs macOS 14 or newer), and mlx-vlm's `transformers>=5.14` fits lerobot's `>=5.22,<5.23`. It has not been created on a Mac yet.
- Added the **Mac plan** above.
- Health check after resuming: `pip check` is clean, the 8 harness tests pass, and the quick smoke suite (`tmp/smoke`) passes 4/4 with numbers matching Phase 0.

**Next**

- See **Resume here** above.

### 2026-11-18: Phase 1 baseline: SmolVLA on LIBERO in four precision modes

**Done**

- Got the evaluation wrapper working (D-023). The output-directory fix and the stored-dtype precision modes (D-022) both worked on the first run.
- Fixed `--latency-only`: one episode can succeed after 2 chunk inferences, fewer than the 3 warm-up calls, so it now runs 3 short episodes.
- The microbenchmark now runs under `torch.inference_mode()`, like lerobot's rollout loop. This made no measurable difference.
- Ran latency-only checks for all four modes into the gitignored `results/tmp/dev/` and deleted the records of the earlier failed runs.
- Ran the full baseline: 4 suites × 10 tasks × 10 episodes = 400 episodes per mode, seed 1000, batch size 1. Each mode took 34 to 40 minutes, about 5 to 6 s per episode, and 2.5 hours in total. On AC power, with no other GPU load.
- Records are in [`results/phase1/`](../results/phase1/), two per mode, all with a clean tree. They were produced from commit `8c7d1ce`. After the runs, that commit was reworded to `33e4f24` to remove a co-author trailer (D-018). The tree is identical.
- D-022 is now Accepted. D-021 and D-023 have new consequences.

**Phase 1 baseline results (RTX 4050 Laptop, 60 W cap)**

| Mode | Success (400 eps) | 95% interval | Spatial / Object / Goal / Long | In-loop p50 / p99 | Microbenchmark p50 / p99 | Peak allocated |
|------|-------------------|--------------|--------------------------------|-------------------|--------------------------|----------------|
| `native` (BF16 + FP32 projections) | **63.5%** (254) | 58.7 to 68.1% | 61 / 69 / 73 / 51% | **142.6 / 158.3 ms** | 145.9 / 147.9 ms | **923 MiB** |
| `fp16` | 61.5% (246) | 56.6 to 66.1% | 59 / 64 / 75 / 48% | 148.3 / 168.1 ms | 155.7 / 159.4 ms | 975 MiB |
| `tf32` | 63.0% (252) | 58.2 to 67.6% | 59 / 70 / 80 / 43% | 180.3 / 201.6 ms | 185.9 / 190.4 ms | 1797 MiB |
| `fp32` | 63.7% (255) | 58.9 to 68.3% | 65 / 67 / 74 / 49% | 208.6 / 225.6 ms | 218.6 / 232.3 ms | 1797 MiB |

Each in-loop row has about 1,960 timed chunk inferences, after discarding 3 warm-up calls. The microbenchmark uses 100 calls after 10 warm-up calls.

**Found**

- **Precision does not affect success.** All four 95% intervals overlap, and the spread between modes (2.2 points) is well inside the interval width.
- **`native` BF16 is the fastest and smallest.** `fp32` is 1.46× slower and `tf32` 1.26× slower. They also double the memory. `fp32` hit the 60 W power cap at 90 °C in the microbenchmark.
- **FP16 brings nothing over BF16** on Ada. It is 4% slower and uses 52 MiB more, presumably from casts at the FP32 projection boundaries. There is no reason to prefer it.
- **Inference is a small part of rollout time.** One chunk inference (about 143 ms) covers 50 environment steps, which take about 1.1 s at 45 steps/s. Optimizing latency matters for real-time control rate, not for evaluation time.
- **Success is well below the SmolVLA paper.** Our 63.5% compares with about 87% average in the paper. `libero_10` (long-horizon) is weakest at 51%, and one `libero_spatial` task scored 0/10. The checkpoint revision, rename map and policy settings all match the training config (D-021), so this is recorded as the measured baseline, not a setup bug.
- **Microbenchmark vs in-loop:** the microbenchmark p50 is consistently 2 to 5% higher than in-loop p50 (D-023). The 3-sample in-loop numbers from latency-only checks vary by ±10 ms, so they only show that a mode runs.

**Next**

- See **Resume here** above.

### 2026-11-18: Phase 1 started: LIBERO, SmolVLA and the evaluation wrapper (paused)

**Done**

- Installed `lerobot[smolvla,libero]==0.7.0` into `sixgig`.
  - It brought transformers 5.22.2, robosuite 1.4.0, hf-libero 0.1.5 and accelerate 1.16.
  - numpy went from 2.5.5 to 2.2.6, and MuJoCo was pinned to 3.8.1 (D-019).
  - `pip check` is clean, and [`env/linux-64.yml`](../env/linux-64.yml) is updated.
- Moved CMake from conda to pip, 4.1.3 (D-020).
- Re-ran the Phase 0 smoke suite after the install into the scratch suite `tmp/after-lerobot`. All four pass, with numbers matching Phase 0.
- Added [`scripts/setup_libero.py`](../scripts/setup_libero.py), which writes LIBERO's config without the stdin prompt (D-024).
- Chose the policy `lerobot/smolvla_libero`, pinned to revision `a24f1f3804` (D-021).
- Checked the stock pipeline end to end with plain `lerobot-eval` on `libero_spatial` task 0, 2 episodes and seed 1000: **2/2 successes** (Wilson 95% interval 34% to 100%).
  - The evaluation took 127 s, about 63 s per episode including model load and warm-up.
  - Rollout ran at about 45 to 49 environment steps per second, and the second episode succeeded around step 79.
  - Output is in `results/tmp/lerobot-eval-check/run3/`, with 2 videos.
- Wrote the Phase 1 code (D-022, D-023). It has not yet completed a run:
  - [`vla/precision.py`](../vla/precision.py): precision modes.
  - [`vla/libero_eval.py`](../vla/libero_eval.py): wraps `lerobot-eval`, adds in-loop chunk timing and a microbenchmark, and writes harness records.
- Downloads so far: about 2.8 GB in the Hugging Face cache (checkpoint and SmolVLM2 tokenizer files) plus LIBERO assets in `~/.cache/libero/assets`. About 56 GB of disk is free.

**Found**

- **LIBERO prompts on first import:** it asks on stdin and fails under `conda run`. Fixed by D-024.
- **Camera names don't match:** `lerobot-eval` needs the camera rename map (D-021), or it stops with "Checkpoint feature missing from the simulator".
- **Checkpoint is mixed precision:** 446.8M parameters are BF16 and 3.3M are FP32 (projections). Casting the whole policy to BF16 failed with "mat1 and mat2 must have the same dtype". Led to D-022.
- **Output directory:** lerobot creates its output directory only when it saves videos. With videos disabled, writing `task_descriptions.json` failed. Fixed in the wrapper, not yet re-run.
- **Model config:** chunk size 50, `n_action_steps` 50, 10 flow-matching steps, KV cache on (`use_cache`), 512×512 padded image resize. These are the defaults Phase 1 starts optimizing from.

**Next**

- See **Resume here** above.

### 2026-11-18: Phase 0 environment and benchmark harness

**Done**

- Created the `sixgig` conda environment from [`env/linux-64.yml`](../env/linux-64.yml). It takes 15 GB.

  | Component | Version |
  |-----------|---------|
  | Python | 3.12.16 |
  | torch / torchvision | 2.14.1+cu126 / 0.29.1+cu126 (cuDNN 9.10.2) |
  | nvcc | 12.6.85, conda-forge |
  | Host compiler | GCC 13.4, conda-forge |
  | CMake | 4.5.0 |
  | TensorRT | 11.4.0.106 (`tensorrt-cu12`) |
  | MuJoCo | 4.0.1 |
  | numpy | 2.5.5 |

- Built the benchmark harness in [`bench/`](../bench/) (D-016), with 8 unit tests that pass.
- Wrote four smoke workloads. All pass from commit `7729dc9` with a clean tree. Records are in [`results/phase0/`](../results/phase0/).
- Confirmed D-008 and D-009, and recorded D-016 (harness design), D-017 (strongly typed TensorRT networks) and D-018 (no co-author trailer). Added glossary entries for the harness and environment.
- Removed the Copilot `Co-authored-by` trailer from all commits at the user's request (D-018). The file contents are unchanged.

**Phase 0 results (RTX 4050 Laptop, 60 W cap, on AC power)**

| Workload | p50 | p99 | Headline |
|----------|-----|-----|----------|
| CUDA SAXPY, 2²⁵ floats, CUDA events | 2.34 ms | 2.71 ms | 172 GB/s, 90% of the theoretical 192 GB/s; zero error |
| torch GEMM 4096³ FP32 | 25.1 ms | 25.5 ms | 5.5 TFLOPS |
| torch GEMM 4096³ TF32 | 12.9 ms | 13.3 ms | 10.7 TFLOPS |
| torch GEMM 4096³ FP16 | 6.27 ms | 6.28 ms | 21.9 TFLOPS |
| torch GEMM 4096³ BF16 | 6.15 ms | 6.20 ms | 22.4 TFLOPS |
| TensorRT FP16 MLP, 4×1024, batch 64 | 32.2 µs | 33.5 µs | 16.7 TFLOPS; build 1.3 s; max error 0.0068 vs PyTorch |
| MuJoCo 7-DoF arm, 10k steps, CPU | 9.4 µs/step | 10.7 µs/step | 106k steps/s, 212× real time |

**Found**

- **Power-limited:** every heavy GPU run hit the 60 W board cap (`sw_power_cap`). FP32 and TF32 GEMMs also showed `sw_thermal`, with SM clocks dropping to about 1700 MHz. Peak temperature was about 61 °C. GEMM throughput on this laptop is set by power, not by compute units, so later phases should report results as power-limited where they are.
- **Power glitch filter works:** an earlier full run rejected 6 readings of 590 W during SAXPY. The final run had none.
- **Dirty flag bug, fixed:** benchmark output under `results/` made the tree look dirty. The check now ignores `results/`. The first set of records was moved to the gitignored `results/tmp/superseded-dirty-flag/` and the suite was re-run.
- **No `trtexec`:** the `tensorrt-cu12` wheel does not include it (D-009).
- **No platform profile:** this laptop exposes no ACPI platform profile, so `platform_profile` is recorded as null. The CPU governor is `powersave` with energy-performance preference `performance`.

**Next**

- Phase 1: install LIBERO and baseline a small VLA (SmolVLA-class) in FP32 and FP16.
- Phase 2: start the CUDA MPPI library. ROS 2 is needed only for the node.
- Mac: set up the environment there.

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
