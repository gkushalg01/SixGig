"""Evaluate a LeRobot VLA on LIBERO and record task success plus inference latency (D-012).

Runs lerobot's own `lerobot-eval` pipeline unchanged (same envs, seeds and processors),
with two hooks on the policy it builds:

* the policy is cast to the requested precision (vla/precision.py);
* every action-chunk inference (`_get_action_chunk`) is timed with CUDA syncs on both sides.

After the evaluation, a controlled microbenchmark replays the first real LIBERO
observation through `predict_action_chunk` to get clean latency statistics.

Each run writes two harness records:

    results/<suite>/smolvla_libero_eval_<precision>-<ts>.json     success + in-loop latency
    results/<suite>/smolvla_chunk_latency_<precision>-<ts>.json   controlled latency

Examples:

    python -m vla.libero_eval --precision fp32 --episodes 10
    python -m vla.libero_eval --precision native --suites libero_spatial --task-ids 0 --episodes 2 --results-suite tmp/dev
    python -m vla.libero_eval --precision fp16 --latency-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Headless rendering and quiet progress bars must be configured before mujoco/tqdm are imported.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TQDM_DISABLE", "1")

from bench import Benchmark  # noqa: E402
from bench.envinfo import REPO_ROOT  # noqa: E402

from .precision import PRECISIONS, apply_precision, describe_model  # noqa: E402

POLICY_ID = "lerobot/smolvla_libero"
# Pinned for reproducibility (D-021): the Hub commit evaluated in Phase 1.
POLICY_REVISION = "a24f1f3804"
LIBERO_SUITES = "libero_spatial,libero_object,libero_goal,libero_10"
RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}
# The first chunk calls include cuDNN autotuning and lazy CUDA initialization.
IN_LOOP_WARMUP_CALLS = 3


def _clone_batch(batch: dict) -> dict:
    import torch

    return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def _check_libero_config() -> None:
    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", Path.home() / ".libero"))
    if not (config_dir / "config.yaml").exists():
        raise SystemExit(f"{config_dir / 'config.yaml'} is missing; run: python scripts/setup_libero.py")


def build_argv(args, output_dir: Path) -> list[str]:
    precision = PRECISIONS[args.precision]
    argv = [
        "lerobot-eval",
        f"--policy.path={args.policy}",
        "--policy.device=cuda",
        "--policy.use_amp=false",
        "--env.type=libero",
        f"--env.task={args.suites}",
        f"--eval.n_episodes={args.episodes}",
        "--eval.batch_size=1",
        f"--seed={args.seed}",
        f"--tf32={str(precision.tf32).lower()}",
        f"--rename_map={json.dumps(RENAME_MAP)}",
        f"--output_dir={output_dir}",
    ]
    if args.revision:
        argv.insert(2, f"--policy.pretrained_revision={args.revision}")
    if args.task_ids:
        argv.append(f"--env.task_ids=[{args.task_ids}]")
    if args.max_episode_steps:
        argv.append(f"--eval.max_episode_steps={args.max_episode_steps}")
    return argv


def run(args) -> list[Path]:
    import torch
    import lerobot.scripts.lerobot_eval as le

    precision = PRECISIONS[args.precision]
    state: dict = {"record": True, "samples_ms": [], "captured": None, "policy": None}

    original_make_policy = le.make_policy
    original_eval_all = le.eval_policy_all

    def make_policy(*a, **k):
        policy = original_make_policy(*a, **k)
        state["loaded_model"] = describe_model(policy)
        apply_precision(policy, precision)
        state["model"] = describe_model(policy)
        state["policy"] = policy
        inner = policy._get_action_chunk

        def timed_chunk(batch, **kw):
            if state["captured"] is None:
                state["captured"] = _clone_batch(batch)
            if not state["record"]:
                return inner(batch, **kw)
            torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            out = inner(batch, **kw)
            torch.cuda.synchronize()
            state["samples_ms"].append((time.perf_counter_ns() - t0) / 1e6)
            return out

        policy._get_action_chunk = timed_chunk
        return policy

    def eval_policy_all(*a, **k):
        k["max_episodes_rendered"] = args.videos
        if args.videos == 0:
            k["videos_dir"] = None
        return original_eval_all(*a, **k)

    le.make_policy = make_policy
    le.eval_policy_all = eval_policy_all

    stamp = time.strftime("%Y%m%dT%H%M%S")
    lerobot_dir = REPO_ROOT / "results" / "tmp" / "lerobot-eval" / f"{args.precision}-{stamp}"
    # lerobot only creates output_dir as a side effect of writing videos.
    lerobot_dir.mkdir(parents=True, exist_ok=True)
    argv = build_argv(args, lerobot_dir)
    params = {
        "policy": args.policy,
        "revision": args.revision,
        "precision": precision.name,
        "precision_description": precision.description,
        "suites": args.suites.split(","),
        "task_ids": args.task_ids,
        "episodes_per_task": args.episodes,
        "start_seed": args.seed,
        "batch_size": 1,
        "max_episode_steps": args.max_episode_steps,
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "lerobot_argv": argv[1:],
    }
    paths = []

    torch.cuda.reset_peak_memory_stats()
    name = f"smolvla_libero_eval_{precision.name}"
    with Benchmark(name, suite=args.results_suite, params=params) as b:
        sys.argv = argv
        le.eval_main()
        info = json.loads((lerobot_dir / "eval_info.json").read_text())
        overall = info["overall"]
        samples = state["samples_ms"]
        if len(samples) <= IN_LOOP_WARMUP_CALLS:
            raise RuntimeError(f"only {len(samples)} chunk inferences ran; increase episodes or steps")
        b.add_samples(samples[IN_LOOP_WARMUP_CALLS:], warmup=IN_LOOP_WARMUP_CALLS, source="host_clock_synced_in_loop")
        cfg = state["policy"].config
        b.metrics.update(
            {
                "success_rate": overall["pc_success"] / 100.0,
                "success_ci95_pct": overall.get("pc_success_ci95"),
                "success_rate_by_suite": {
                    g: v["pc_success"] / 100.0 for g, v in info.get("per_group", {}).items()
                },
                "eval_wall_s": overall.get("eval_s"),
                "eval_s_per_episode": overall.get("eval_ep_s"),
                "chunk_inferences": len(samples),
                "first_chunk_ms": samples[0],
                "torch_max_memory_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "torch_max_memory_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                "chunk_size": cfg.chunk_size,
                "n_action_steps": cfg.n_action_steps,
                "denoise_steps": cfg.num_steps,
                "model_as_loaded": state["loaded_model"],
                "model": state["model"],
            }
        )
        b.metrics["per_task"] = [
            {"suite": t["task_group"], "task_id": t["task_id"], "successes": sum(map(bool, t["metrics"]["successes"])), "episodes": len(t["metrics"]["successes"])}
            for t in info.get("per_task", [])
        ]
        b.set_task_result(overall["n_success"], overall["n_episodes"], seeds=range(args.seed, args.seed + args.episodes))
        b.notes.append(f"lerobot output: {lerobot_dir.relative_to(REPO_ROOT)}")
        if args.latency_only:
            b.notes.append("latency-only run: episodes truncated, success rate is not meaningful")
    paths.append(b.output_path)

    paths.append(microbenchmark(args, state, precision))
    return paths


def microbenchmark(args, state: dict, precision) -> Path:
    import torch

    policy, batch = state["policy"], state["captured"]
    if batch is None:
        raise RuntimeError("no observation was captured during evaluation")
    state["record"] = False
    cfg = policy.config
    params = {
        "policy": args.policy,
        "revision": args.revision,
        "precision": precision.name,
        "precision_description": precision.description,
        "batch_size": 1,
        "input": "first LIBERO observation of the evaluation, after lerobot preprocessing",
        "chunk_size": cfg.chunk_size,
        "denoise_steps": cfg.num_steps,
    }
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with Benchmark(f"smolvla_chunk_latency_{precision.name}", suite=args.results_suite, params=params) as b:
        samples = b.measure(
            lambda: policy.predict_action_chunk(dict(batch)),
            warmup=args.latency_warmup,
            iters=args.latency_iters,
            sync=torch.cuda.synchronize,
        )
        p50_s = sorted(samples)[len(samples) // 2] / 1e3
        b.metrics.update(
            {
                "chunks_per_s_p50": 1.0 / p50_s,
                "actions_per_s_p50": cfg.chunk_size / p50_s,
                "torch_max_memory_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "model": state["model"],
            }
        )
    return b.output_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--precision", choices=list(PRECISIONS), required=True)
    parser.add_argument("--policy", default=POLICY_ID)
    parser.add_argument("--revision", default=POLICY_REVISION, help="Hub revision; empty string for latest")
    parser.add_argument("--suites", default=LIBERO_SUITES, help="comma-separated LIBERO suites")
    parser.add_argument("--task-ids", default="", help="comma-separated task ids within each suite (default: all)")
    parser.add_argument("--episodes", type=int, default=10, help="episodes per task")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-episode-steps", type=int, default=0)
    parser.add_argument("--videos", type=int, default=0, help="number of episodes to save as video")
    parser.add_argument("--latency-only", action="store_true", help="one short episode, then the microbenchmark")
    parser.add_argument("--latency-warmup", type=int, default=10)
    parser.add_argument("--latency-iters", type=int, default=100)
    parser.add_argument("--results-suite", default="phase1")
    args = parser.parse_args(argv)

    if args.latency_only:
        args.suites, args.task_ids, args.episodes = "libero_spatial", "0", 1
        args.max_episode_steps = args.max_episode_steps or 4 * 50
    _check_libero_config()
    for path in run(args):
        rec = json.loads(path.read_text())
        lat = rec["latency_ms"]
        task = rec.get("task") or {}
        extra = f" success={task['success_rate']:.3f} ({task['successes']}/{task['episodes']})" if task else ""
        print(f"{rec['name']}: {rec['status']} p50={lat['p50']:.2f} ms p99={lat['p99']:.2f} ms{extra} -> {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
