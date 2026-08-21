"""Contract tests: infra profiling (cProfile harness + reports)."""

import json
import time

from fpl.profiling import profile_call, summarize_profile, time_phases


def _busy(seconds=0.05):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        _ = 1 + 1


def test_time_phases_reports_sorted_order():
    order = []
    phases = {
        "a": lambda: (order.append("a"), _busy(0.01)),
        "b": lambda: (order.append("b"), _busy(0.02)),
    }
    timings = time_phases(phases)
    assert list(timings) == ["a", "b"]
    assert timings["a"] > 0.0 and timings["b"] > timings["a"]


def test_profile_call_writes_prof_and_json(tmp_path):
    profile_call(_busy, name="busy", out_dir=tmp_path)
    assert (tmp_path / "busy.prof").exists()
    json_path = tmp_path / "busy.json"
    assert json_path.exists()
    loaded = json.loads(json_path.read_text())
    assert loaded["profile"] == "busy"
    assert loaded["wall_seconds"] > 0.0
    assert loaded["function_rows"], "cProfile should record at least busy itself"
    assert loaded["module_rows"], "module rollup should be non-empty"


def test_profile_rows_have_schema(tmp_path):
    report = profile_call(_busy, name="busy", out_dir=tmp_path)
    top = report["function_rows"][0]
    for key in ("function", "where", "ncalls", "tottime_s", "cumtime_s"):
        assert key in top
    module = report["module_rows"][0]
    assert {"module", "tottime_s", "cumtime_s"} <= set(module)


def test_summary_renders(tmp_path):
    report = profile_call(_busy, name="busy", out_dir=tmp_path)
    text = summarize_profile(report)
    assert "busy" in text
    assert "cumtime_s" in text and "top modules" in text