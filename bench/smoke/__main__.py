"""Run the Phase 0 smoke workloads.

    python -m bench.smoke                 # all four, full length
    python -m bench.smoke --quick         # short runs
    python -m bench.smoke --only cuda,trt # a subset

Exits non-zero if any workload fails. Each workload writes JSON to results/<suite>/.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback

from ..envinfo import REPO_ROOT

WORKLOADS = ("cuda", "torch", "trt", "mujoco")


def _load(name: str):
    if name == "cuda":
        from .cuda_kernel import run
    elif name == "torch":
        from .torch_matmul import run
    elif name == "trt":
        from .tensorrt_engine import run
    else:
        from .mujoco_episode import run
    return run


def _headline(path) -> str:
    rec = json.loads(path.read_text())
    lat = rec.get("latency_ms") or {}
    metrics = ", ".join(f"{k}={v:.4g}" for k, v in rec["metrics"].items() if isinstance(v, (int, float)))
    p50 = lat.get("p50")
    p99 = lat.get("p99")
    lat_txt = f"p50={p50:.4g} ms p99={p99:.4g} ms" if p50 is not None else "no latency"
    return f"{rec['name']}: {rec['status']} | {lat_txt} | {metrics} -> {path.relative_to(REPO_ROOT)}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default=",".join(WORKLOADS), help=f"comma-separated subset of {WORKLOADS}")
    parser.add_argument("--quick", action="store_true", help="fewer iterations")
    parser.add_argument("--suite", default="phase0", help="results subdirectory")
    args = parser.parse_args(argv)

    selected = [w.strip() for w in args.only.split(",") if w.strip()]
    unknown = sorted(set(selected) - set(WORKLOADS))
    if unknown:
        parser.error(f"unknown workloads: {unknown}")

    failed = []
    for name in selected:
        print(f"== {name}", flush=True)
        try:
            result = _load(name)(quick=args.quick, suite=args.suite)
            for path in result if isinstance(result, list) else [result]:
                print("  " + _headline(path), flush=True)
        except Exception:
            failed.append(name)
            traceback.print_exc()
    print(f"\n{len(selected) - len(failed)}/{len(selected)} workloads passed" + (f"; failed: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
