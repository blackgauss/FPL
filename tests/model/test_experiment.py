"""Black-box tests: experiment harness and feature-subset assembly."""

import numpy as np
import polars as pl
import pytest

from fpl.model.experiment import REGISTRY, run_experiment
from fpl.model.train import assemble

FEAT = {
    "player_id": [1, 1, 2, 2],
    "player_code": [223094, 223094, 118748, 118748],
    "gw": [2, 3, 2, 3],
    "team_code": [43, 43, 3, 3],
    "opponent_team_code": [3, 3, 43, 43],
    "was_home": [True, False, False, True],
    "home_elo": [2064.0, 1991.0, 1991.0, 2064.0],
    "opponent_elo": [1991.0, 2064.0, 2064.0, 1991.0],
    "prev_points": [13, 2, 9, 3],
    "pts_avg_3": [13.0, 7.5, 9.0, 6.0],
    "pts_avg_5": [13.0, 7.5, 9.0, 6.0],
    "total_points": [2, 4, 5, 7],
    "next_points": [4, 5, 7, 6],
}
PLAYERS = {
    "player_id": [1, 2],
    "player_code": [223094, 118748],
    "team_code": [43, 3],
    "position": ["FWD", "MID"],
}
GW_STATS = {
    "player_id": [1, 1, 2, 2],
    "gw": [2, 3, 2, 3],
    "now_cost": [15.0, 14.1, 11.0, 10.5],
    "ep_next": [4.0, 4.5, 3.0, 3.5],
}


def _raw():
    return (pl.DataFrame(FEAT), pl.DataFrame(PLAYERS), pl.DataFrame(GW_STATS))


@pytest.fixture()
def pairs():
    feat, players, gw = _raw()
    td = assemble(feat, players, gw, "2025-2026")
    return td, td


class TestFeatureSubset:
    def test_subset_reduces_columns(self):
        feat, players, gw = _raw()
        sub = assemble(feat, players, gw, "2025-2026",
                       feature_columns=["position", "now_cost", "pts_avg_3"])
        assert sub.feature_names == ["position", "now_cost", "pts_avg_3"]
        assert sub.X.shape[1] == 3
        assert sub.categorical == [0]  # position remains categorical

    def test_dropping_categorical_adjusts_index(self):
        feat, players, gw = _raw()
        sub = assemble(feat, players, gw, "2025-2026",
                       feature_columns=["now_cost", "ep_next"])
        assert sub.categorical == []

    def test_default_is_full_set(self):
        feat, players, gw = _raw()
        t = assemble(feat, players, gw, "2025-2026")
        assert t.feature_names == [
            "team_code", "position", "now_cost", "ep_next", "had_match",
            "was_home", "opponent_elo", "home_elo", "pts_avg_3", "pts_avg_5",
        ]


class TestRunExperiment:
    def test_unknown_model_raises(self, pairs):
        with pytest.raises(ValueError, match="unknown model"):
            run_experiment(pairs[0], pairs[1], name="x", model="nope",
                           fit_gw_max=2, test_gw_min=3)

    def test_ridge_finite_and_scores(self, pairs):
        train, fit = pairs
        r = run_experiment(train, fit, name="r", model="ridge",
                           fit_gw_max=2, test_gw_min=3)
        assert isinstance(r.mae, float)
        assert r.n_test == 2  # gw == 3 rows

    def test_registry_has_core_models(self):
        assert {"lgbm", "hist_gb", "ridge"} <= set(REGISTRY)

    def test_test_min_equal_fit_max_is_leak_and_errors(self, pairs):
        # overlapping windows must be rejected early
        train, fit = pairs
        with pytest.raises(ValueError, match="leaky split"):
            run_experiment(train, fit, name="x", model="ridge",
                           fit_gw_max=3, test_gw_min=2)


class TestFiniteMatrix:
    def test_no_nan_in_full_assemble(self, pairs):
        assert not np.isnan(pairs[0].X).any()