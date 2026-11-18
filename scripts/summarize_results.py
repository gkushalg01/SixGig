"""Print one table row per benchmark record, so runs can be compared at a glance.

    python scripts/summarize_results.py                      # results/phase1
    python scripts/summarize_results.py results/tmp/dev      # any suite directory
    python scripts/summarize_results.py results/phase1 --name libero_eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _fmt(value, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def row(path: Path) -> list[str]:
    rec = json.loads(path.read_text())
    lat = rec.get("latency_ms") or {}
    task = rec.get("task") or {}
    metrics = rec.get("metrics") or {}
    params = rec.get("params") or {}
    git = (rec.get("environment") or {}).get("git") or {}
    success = task.get("success_rate")
    return [
        rec.get("name", path.stem),
        rec.get("status", "?"),
        f"{_fmt(success and success * 100, '.1f')}% ({task.get('successes')}/{task.get('episodes')})" if task else "-",
        _fmt(lat.get("p50"), ".1f"),
        _fmt(lat.get("p99"), ".1f"),
        _fmt(metrics.get("torch_max_memory_allocated_mib"), ".0f"),
        str(metrics.get("denoise_steps", params.get("denoise_steps", "-"))),
        str(metrics.get("n_action_steps", params.get("n_action_steps", "-"))),
        (git.get("commit") or "?")[:7] + ("*" if git.get("dirty") else ""),
        path.stem.rsplit("-", 1)[-1],
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("suite_dir", nargs="?", default=str(REPO_ROOT / "results" / "phase1"))
    parser.add_argument("--name", default="", help="only records whose name contains this text")
    args = parser.parse_args(argv)

    paths = sorted(p for p in Path(args.suite_dir).glob("*.json") if args.name in p.stem)
    if not paths:
        raise SystemExit(f"no records in {args.suite_dir}")
    header = ["name", "status", "success", "p50 ms", "p99 ms", "peak MiB", "steps", "n_act", "commit", "time"]
    rows = [header] + [row(p) for p in paths]
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip())
    print("\ncommit* = dirty tree when the record was saved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
