# Commands

Commands used regularly in this project, with what each one does. Run them from the repository root with the `sixgig` environment active unless a section says otherwise. Add new commands here when a phase introduces them (D-013).

Related documents: [plan.md](./plan.md) · [progress.md](./progress.md) · [decisions.md](./decisions.md)

## Every session (RTX laptop)

| Command | What it does |
|---------|--------------|
| `conda activate sixgig` | Activates the project environment (Python 3.12, torch 2.14.1+cu126, CUDA 12.6 toolkit, lerobot). |
| `cd ~/workspace/SixGig` | Goes to the repository root. Every command below assumes it. |
| `conda run --no-capture-output -n sixgig <cmd>` | Runs one command in the environment without activating it. `--no-capture-output` streams output live. |
| `git status --short` | Shows uncommitted changes. Commit before long runs so records are not marked dirty (D-016). |
| `nvidia-smi` | Shows GPU memory, temperature, power and running processes. Check the GPU is idle before benchmarking. |
| `cat /sys/class/power_supply/A*/online` | Prints `1` on AC power. Benchmarks for the record must run plugged in (D-012). |

## One-time setup and environment maintenance

| Command | What it does |
|---------|--------------|
| `conda env create -f env/linux-64.yml` | Creates the `sixgig` environment on the RTX laptop (about 15 GB). |
| `conda env update -f env/linux-64.yml --prune` | Brings an existing environment in line with the file after it changes. |
| `python scripts/setup_libero.py` | Writes `~/.libero/config.yaml` without LIBERO's stdin prompt. Once per machine (D-024). `--force` overwrites it. |
| `pip check` | Reports broken or conflicting package requirements. Run after any install. |
| `pip list \| grep -iE "torch\|lerobot\|mujoco\|transformers\|tensorrt"` | Shows the versions of the key packages. |

## Health checks

| Command | What it does |
|---------|--------------|
| `python -m pytest bench/tests -q` | Runs the benchmark harness unit tests. |
| `python -m bench.smoke --quick --suite tmp/smoke` | Short runs of the four Phase 0 workloads (CUDA kernel, torch GEMM, TensorRT, MuJoCo) into a scratch suite. Use after driver, kernel or environment changes. |
| `python -m bench.smoke` | Full Phase 0 smoke suite into `results/phase0/`. `--only cuda,trt` runs a subset. |
| `nvidia-smi --query-gpu=name,memory.used,temperature.gpu,power.draw,clocks.sm --format=csv -l 1` | Prints GPU state every second. Useful in a second terminal during long runs. A 590 W power reading is a known glitch (D-016). |
| `lsmod \| grep -E "nvidia\|i915\|xe"` | Shows which graphics drivers loaded. After a kernel update, `i915` and `nvidia` should be present and `xe` absent (D-015). |

## Phase 1: VLA evaluation (`vla/libero_eval.py`)

Every run writes two records: `smolvla_libero_eval_<variant>` (task success and in-loop chunk latency) and `smolvla_chunk_latency_<variant>` (microbenchmark). lerobot's own output goes to the gitignored `results/tmp/lerobot-eval/`.

| Command | What it does |
|---------|--------------|
| `python -m vla.libero_eval --precision native --latency-only --results-suite tmp/dev` | Quick check (about 1 minute): 3 short episodes on one task, then the microbenchmark. Shows the setup works and gives a rough latency. Success rate is not meaningful. |
| `python -m vla.libero_eval --precision native` | Full baseline-protocol run: 4 suites × 10 tasks × 10 episodes = 400 episodes, seed 1000, into `results/phase1/`. Takes 35 to 40 minutes. |
| `python -m vla.libero_eval --precision fp32` | Same, in another precision mode. Modes: `native`, `fp32`, `tf32`, `fp16` (D-022). |
| `python -m vla.libero_eval --precision native --policy-override num_steps=5` | Overrides a policy config field for the run. Here: 5 flow-matching (denoising) steps instead of 10. The record name gets the suffix `_num_steps5`. |
| `python -m vla.libero_eval --precision native --policy-override n_action_steps=10` | Executes 10 of the 50 actions in each chunk before predicting the next one (D-021). |
| `... --policy-override num_steps=5 --policy-override n_action_steps=25 --tag fast` | Combines overrides; `--tag` replaces the automatic record-name suffix. |
| `python -m vla.libero_eval --precision native --suites libero_spatial --task-ids 0 --episodes 2 --results-suite tmp/dev` | A small custom run: one suite, one task, two episodes. |
| `... --videos 4` | Saves the first 4 episodes as videos under `results/tmp/lerobot-eval/<variant>-<time>/videos/`. |
| `python -m vla.libero_eval --help` | Lists every option. |

## Results

| Command | What it does |
|---------|--------------|
| `python scripts/summarize_results.py` | One line per record in `results/phase1/`: success, p50/p99 latency, peak memory, denoising steps, `n_action_steps`, commit. A `*` after the commit means the tree was dirty. |
| `python scripts/summarize_results.py results/tmp/dev --name num_steps` | Same, for any suite directory, filtered by record name. |
| `rm -rf results/tmp/*` | Deletes scratch records and lerobot output. Everything under `results/tmp/` is gitignored. |
| `du -sh ~/.cache/huggingface ~/.cache/libero results/tmp` | Disk used by model downloads, LIBERO assets and scratch output. |
| `df -h /` | Free disk space (about 49 GB on 2026-11-18). |

## Git

| Command | What it does |
|---------|--------------|
| `git --no-pager log --oneline -10` | Recent commits. Records store the commit they ran from. |
| `git add -A && git commit -m "..."` | Commits code, docs and curated results together (D-013). No co-author trailer (D-018). |
| `git --no-pager diff --stat` | Summary of uncommitted changes. |

## Mac (MacBook M4 Pro)

The Mac environment file is a draft and has not been created yet. See [progress.md](./progress.md), "Mac plan".

| Command | What it does |
|---------|--------------|
| `xcode-select --install` | Installs Apple's command-line build tools (compilers, git). Once per Mac. |
| `curl -LO https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh && bash Miniforge3-MacOSX-arm64.sh` | Installs Miniforge, a conda distribution that defaults to conda-forge. |
| `git clone ats@<laptop-ip>:workspace/SixGig ~/workspace/SixGig` | Clones the repository from the RTX laptop over SSH (the laptop runs an SSH server; its LAN address was 192.168.0.103 on 2026-11-18, check with `hostname -I`). There is no GitHub remote yet. Later updates: `git pull`. |
| `conda env create -f env/osx-arm64.yml` | Creates the Mac `sixgig` environment: torch (MPS backend), MuJoCo 3.8.1, lerobot 0.7.0, MLX, mlx-vlm. |
| `python scripts/setup_libero.py` | LIBERO config, same as on Linux. |
| `python -c "import torch; print(torch.backends.mps.is_available())"` | Checks that PyTorch can use the Apple GPU. Should print `True`. |
| `python -c "import mlx.core as mx; print(mx.default_device(), mx.ones(3) * 2)"` | Checks that MLX runs on the GPU. Should print `Device(gpu, 0)`. |
| `MUJOCO_GL=cgl python -c "import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><geom size=\".1\"/></body></worldbody></mujoco>'); r = mujoco.Renderer(m); r.update_scene(mujoco.MjData(m)); print(r.render().shape)"` | Checks that MuJoCo renders offscreen on macOS (`cgl` replaces Linux's `egl`). Should print `(240, 320, 3)`. |
| `python -m pytest bench/tests -q` | Harness unit tests. They do not touch the GPU, so they are expected to pass on the Mac too. |
| `sudo powermetrics --samplers gpu_power -i 1000` | Apple GPU power and frequency every second, the Mac's counterpart to `nvidia-smi`. |
