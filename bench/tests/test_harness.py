import json

import pytest

from bench.harness import Benchmark
from bench.stats import percentile, summarize
from bench.telemetry import decode_clock_reasons, is_valid_power


def test_percentile_interpolates_like_numpy_default():
    data = [1.0, 2.0, 3.0, 4.0]
    assert percentile(data, 50) == 2.5
    assert percentile(data, 0) == 1.0
    assert percentile(data, 100) == 4.0
    assert percentile(data, 95) == pytest.approx(3.85)


def test_summarize_reports_tail_latency():
    s = summarize([1.0] * 98 + [10.0, 100.0])
    assert s["p50"] == 1.0
    assert s["max"] == 100.0
    assert s["p99"] > s["p95"]


def test_power_glitch_from_this_laptop_is_rejected():
    # nvidia-smi reported 590 W against the RTX 4050's 60 W cap (docs/progress.md, 2026-11-18).
    assert not is_valid_power(590.01, 60.0)
    assert is_valid_power(12.6, 60.0)
    assert is_valid_power(75.0, 60.0)  # brief boost above the cap is plausible
    assert not is_valid_power(-1.0, 60.0)
    assert is_valid_power(590.0, None)  # no limit known: nothing to compare against


def test_clock_reasons_decode_thermal_and_power_caps():
    assert decode_clock_reasons(0x4 | 0x40) == ["hw_thermal", "sw_power_cap"]
    assert decode_clock_reasons(0) == []


def test_measure_excludes_warmup_and_writes_record(tmp_path):
    calls = []
    with Benchmark("unit", suite="test", params={"k": 1}, results_dir=tmp_path, monitor_gpu=False) as b:
        b.measure(lambda: calls.append(1), warmup=3, iters=7)
        b.metrics["x"] = 1.5
    rec = json.loads(b.output_path.read_text())
    assert len(calls) == 10
    assert rec["warmup_iters"] == 3
    assert rec["latency_ms"]["count"] == 7
    assert len(rec["latency_ms_samples"]) == 7
    assert rec["status"] == "ok"
    assert rec["metrics"] == {"x": 1.5}
    assert rec["environment"]["git"]["commit"]
    assert b.output_path.parent == tmp_path / "test"


def test_failed_run_is_still_recorded(tmp_path):
    with pytest.raises(RuntimeError):
        with Benchmark("boom", suite="test", results_dir=tmp_path, monitor_gpu=False) as b:
            raise RuntimeError("engine build failed")
    rec = json.loads(b.output_path.read_text())
    assert rec["status"] == "error"
    assert "engine build failed" in rec["error"]


def test_suite_may_nest_but_not_traverse(tmp_path):
    b = Benchmark("x", suite="tmp/quick", results_dir=tmp_path, monitor_gpu=False)
    b.add_samples([1.0], warmup=0, source="test")
    assert b.save().parent == tmp_path / "tmp" / "quick"
    for bad in ("../escape", "a/../b", "/abs", "a//b"):
        with pytest.raises(ValueError):
            Benchmark("x", suite=bad, results_dir=tmp_path, monitor_gpu=False)


def test_task_result_validates_counts(tmp_path):
    b = Benchmark("task", suite="test", results_dir=tmp_path, monitor_gpu=False)
    b.set_task_result(successes=3, episodes=4, seeds=[0, 1, 2, 3])
    assert b.task["success_rate"] == 0.75
    with pytest.raises(ValueError):
        b.set_task_result(successes=5, episodes=4, seeds=[])
