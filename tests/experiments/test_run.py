"""Contract tests: fpl.experiments.run end-to-end on a synthetic store."""

from pathlib import Path

import pytest

from fpl.data.contract import load_season
from fpl.data.features import build_features
from fpl.experiments.run import run_experiment


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    from fpl.model.train import load_training
    from tests.fixtures.synthetic import build_season_tree_dense

    root = Path(tmp_path_factory.mktemp("exp"))
    build_season_tree_dense(root, n_players=40, n_gws=4)
    data = load_season(root, "2025-2026")
    feats = build_features(data.gw_stats, data.team_history, data.matches,
                           data.players)
    processed = root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    for name, frame in [("features", feats), ("gw_stats", data.gw_stats),
                        ("players", data.players)]:
        frame.write_parquet(processed / f"{name}_2025-2026.parquet")
    load_training(processed, ["2025-2026"])
    return str(processed)


def _base_spec(**overrides):
    spec = {
        "name": "probe", "model": "lgbm",
        "params": {"learning_rate": 0.05, "num_leaves": 15,
                   "min_child_samples": 5, "num_boost_round": 20},
        "seasons": ["2025-2026"],
        "split": {"fit_gw_max": 1, "cal_start": 2, "cal_end": 1,
                  "test_start": 2, "test_end": 3},
    }
    spec.update(overrides)
    return spec


class TestPointRun:
    def test_point_metrics_written(self, store):
        result = run_experiment(_base_spec(), processed=store)
        assert result["name"] == "probe"
        cohorts = {m["cohort"] for m in result["metrics"]}
        assert "all" in cohorts and "top10" in cohorts
        all_metric = next(m for m in result["metrics"] if m["cohort"] == "all")
        assert {"mae", "rmse", "bias", "n"} <= set(all_metric)

    def test_gym_actual_settlement(self, store):
        spec = _base_spec(gym={
            "season": "2025-2026", "gw_start": 2, "gw_end": 3,
            "n_teams": 2, "seed": 1, "top": 1})
        result = run_experiment(spec, processed=store)
        assert result["gym"]["settlement"] == "actual"
        assert result["gym"]["squads"] == 1
        for key in ("total_actual", "gap", "captain_weeks"):
            assert key in result["gym"]

    def test_gym_predicted_settlement_with_play_prob(self, store):
        spec = _base_spec(gym={
            "season": "2025-2026", "gw_start": 2, "gw_end": 3,
            "n_teams": 2, "seed": 1, "top": 1})
        stub = lambda squad, gw: {c: 0.9 for c in squad.codes()}  # noqa: E731
        result = run_experiment(spec, processed=store, play_prob=stub)
        assert result["gym"]["settlement"] == "predicted"

    def test_gym_with_custom_features_rejected(self, store):
        spec = _base_spec(features=["position", "now_cost"], gym={})
        with pytest.raises(ValueError, match="default feature set"):
            run_experiment(spec, processed=store)