"""Black-box tests: inference serving (expected points for a context)."""

import polars as pl
import pytest

from fpl.model.inference import expected_points
from fpl.model.train import assemble


@pytest.fixture(scope="module")
def model_and_td(tmp_path_factory):
    import lightgbm as lgb

    feat = pl.DataFrame({
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
    })
    players = pl.DataFrame({
        "player_id": [1, 2],
        "player_code": [223094, 118748],
        "team_code": [43, 3],
        "position": ["FWD", "MID"],
        "web_name": ["Haaland", "De Bruyne"],
    })
    gw_stats = pl.DataFrame({
        "player_id": [1, 1, 2, 2],
        "gw": [2, 3, 2, 3],
        "now_cost": [15.0, 14.1, 11.0, 10.5],
        "ep_next": [4.0, 4.5, 3.0, 3.5],
    })
    td = assemble(feat, players, gw_stats, "2025-2026")
    ds = lgb.Dataset(td.X, label=td.y, feature_name=td.feature_names,
                     categorical_feature=td.categorical)
    model = lgb.train({"objective": "regression", "metric": "mae",
                       "verbosity": -1}, ds, num_boost_round=2)
    return model, td, players


def test_predicts_target_gameweek_rows_only(model_and_td):
    # gw=3 rows have target == points of GW4 (shift semantics).
    # expected_points(gw=4) must predict on source rows where gw == 3, and
    # the report's `gw` column labels the predicted gameweek (4).
    model, td, players = model_and_td
    rep = expected_points(td, model, gw=4, players=players)
    assert set(rep.get_column("gw")) == {4}  # predicted gameweek, not source
    assert rep.height == 2  # both players have a gw=3 source row
    assert "expected_points" in rep.columns


def test_code_filter_restricts(model_and_td):
    model, td, players = model_and_td
    rep = expected_points(td, model, gw=4, players=players,
                          code_filter=[223094])
    assert rep.height == 1
    assert rep.get_column("player_code").item() == 223094


def test_horizon_across_two_gameweeks(model_and_td):
    from fpl.model.inference import expected_points_horizon

    model, td, players = model_and_td
    rep = expected_points_horizon(td, model, gw_start=3, gw_end=4,
                                  players=players)
    # gw=3 uses source rows gw-1=2; gw=4 uses source rows gw-1=3
    assert set(rep.get_column("gw")) == {3, 4}
    assert rep.height == 4
    assert sorted(rep.get_column("gw").unique()) == [3, 4]


def test_missing_gw_raises(model_and_td):
    model, td, players = model_and_td
    with pytest.raises(ValueError, match="no feature rows"):
        expected_points(td, model, gw=1, players=players)


class TestSerialization:
    @pytest.fixture()
    def ridge(self):
        import numpy as np
        from sklearn.linear_model import Ridge

        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 3))
        y = X @ np.array([1.0, 2.0, 3.0]) + 0.1 * rng.normal(size=40)
        return Ridge().fit(X, y)

    @pytest.mark.parametrize("suffix", [".pkl", ".joblib"])
    def test_sklearn_roundtrip(self, ridge, tmp_path, suffix):
        import numpy as np

        from fpl.model.inference import load_model, save_model

        p = tmp_path / f"ridge{suffix}"
        save_model(ridge, p)
        back = load_model(p)
        x = np.zeros((3, 3))
        np.testing.assert_allclose(back.predict(x), ridge.predict(x), rtol=1e-9)

    def test_unknown_suffix_raises(self, ridge, tmp_path):
        from fpl.model.inference import save_model

        with pytest.raises(ValueError, match="no serializer for '.xyz'"):
            save_model(ridge, tmp_path / "model.xyz")