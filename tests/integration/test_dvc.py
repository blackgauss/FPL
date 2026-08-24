"""Tests for the DVC pipeline scaffold.

Covers what matters without re-running heavy stages:
- the dvc.yaml stage graph shape (names/deps/metrics),
- the metrics-writer helpers used by the stages (unit, on synthetic results),
- a fast `dvc dag` smoke test that the graph is valid DVC (skipped if the
  `dvc` dev dependency is not installed).
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest
import yaml

from fpl.experiments.artifacts import (
    flat_metric,
    write_flat_metrics,
    write_metrics_json,
)
from fpl.pipeline import ranked_candidates, validate_candidate_artifact

ROOT = Path(__file__).resolve().parents[2]
DVC = importlib.util.find_spec("dvc") is not None


def _fake_result(name="lgbm_all"):
    return {
        "name": name,
        "metrics": [
            {"cohort": "all", "mae": 1.01, "rmse": 2.0, "bias": 0.1, "n": 5802},
            {"cohort": "top10", "mae": 3.1, "rmse": 3.9, "n": 580},
        ],
        "ranking": {"spearman_rho": 0.68, "topk_hit_rate": 0.35,
                    "pairwise_concordance": 0.46, "n": 5802},
        "calibration": {"mae": 1.01, "slope": 0.87, "ece": 0.12, "n": 5802},
    }


def test_dvc_yaml_defines_expected_stages():
    spec = yaml.safe_load((ROOT / "dvc.yaml").read_text(encoding="utf-8"))
    stages = spec["stages"]
    assert set(stages) == {"ingest", "features", "train", "fit_dist",
                           "eval_experiments", "rank_report", "search",
                           "gym"}
    # the data+model spine must be present and chained by deps/outs
    assert {x["cmd"] for x in stages.values()}
    assert all(stage["deps"] for stage in stages.values())
    for name in ("train", "eval_experiments", "rank_report", "search", "gym"):
        assert stages[name]["metrics"], f"{name} must declare metrics"
    assert any("search_candidates.parquet" in path
               for path in stages["search"]["outs"])
    assert any("search_candidates.parquet" in path
               for path in stages["gym"]["deps"])
    assert (ROOT / "params.yaml").exists()
    assert any("params.yaml" in s for s in spec.get("vars", []))
    assert stages["search"]["params"] == ["pipeline.season",
                                            "pipeline.gw_start",
                                            "pipeline.gw_end", "pipeline.search"]
    assert stages["gym"]["params"] == ["pipeline.season", "pipeline.gw_start",
                                          "pipeline.gw_end", "pipeline.gym.top"]
    assert "src/fpl" in stages["search"]["deps"]
    assert "src/fpl" in stages["gym"]["deps"]


def test_metrics_writer_flat_and_roundtrip(tmp_path):
    results = [_fake_result("a"), _fake_result("b")]
    out = tmp_path / "m.json"
    write_flat_metrics(results, out)
    payload = json.loads(out.read_text())
    assert set(payload["experiments"]) == {"a", "b"}
    for flat in payload["experiments"].values():
        assert flat["mae_all"] == 1.01
        assert flat["rank_spearman_rho"] == 0.68
        assert flat["cal_slope"] == 0.87
    # single-result flattening matches the writer's per-name dict
    assert flat_metric(results[0]) == payload["experiments"]["a"]


def test_write_metrics_json_roundtrip(tmp_path):
    path = write_metrics_json({"hist_gb": {"mae": 1.0}}, tmp_path / "k.json")
    assert json.loads(path.read_text()) == {"hist_gb": {"mae": 1.0}}


def test_candidate_artifact_contract_and_order():
    class Result:
        basket = pl.DataFrame({
            "team_id": [2, 1], "player_code": [20, 10],
            "position": ["GKP", "GKP"], "price_tenths": [50, 50],
            "expected_total": [4.0, 5.0],
        })
        team_ids = (1, 2)

    artifact = ranked_candidates(Result())
    assert artifact["team_id"].to_list() == [1, 2]
    assert artifact["candidate_rank"].to_list() == [0, 1]
    validate_candidate_artifact(artifact)
    with pytest.raises(ValueError, match="expected_total"):
        validate_candidate_artifact(artifact.drop("expected_total"))


@pytest.mark.skipif(not DVC, reason="dvc dev dependency not installed")
def test_dvc_dag_renders_stage_graph():
    result = subprocess.run(
        [sys.executable, "-m", "dvc", "dag"], cwd=ROOT,
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr[-500:]
    assert "eval_experiments" in result.stdout
    assert "rank_report" in result.stdout
