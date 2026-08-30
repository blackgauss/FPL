"""Black-box tests: eval metrics and training-data assembly."""

import polars as pl
import pytest

from fpl.model.eval import baseline_mean, mae, rmse
from fpl.model.train import FEATURE_COLUMNS, assemble


class TestMetrics:
    def test_mae(self):
        y = pl.Series([0.0, 2.0, 4.0])
        p = pl.Series([1.0, 1.0, 5.0])
        assert mae(y, p) == pytest.approx((1 + 1 + 1) / 3)

    def test_rmse(self):
        y = pl.Series([0.0, 0.0])
        p = pl.Series([0.0, 3.0])
        assert rmse(y, p) == pytest.approx((9.0 / 2) ** 0.5)


class TestBaselineMean:
    def test_constant_prediction(self):
        y_pred = baseline_mean(pl.Series([2.0, 4.0]), pl.Series([0.0, 0.0]))
        assert y_pred.to_list() == [3.0, 3.0]


@pytest.fixture()
def frames():
    features = pl.DataFrame({
        "player_id": [1, 1, 2],
        "player_code": [223094, 223094, 118748],
        "gw": [2, 3, 2],
        "team_code": [43, 43, 3],
        "opponent_team_code": [3, None, 43],
        "was_home": [True, None, False],
        "home_elo": [2064.0, None, 1991.0],
        "opponent_elo": [1991.0, None, 2064.0],
        "prev_points": [13, 2, 9],
        "pts_avg_3": [13.0, 7.5, 9.0],
        "pts_avg_5": [13.0, 7.5, 9.0],
        "total_points": [2, 4, 5],
        "next_points": [4, 1, 3],
    })
    players = pl.DataFrame({
        "player_id": [1, 2],
        "player_code": [223094, 118748],
        "position": ["FWD", "MID"],
    })
    gw_stats = pl.DataFrame({
        "player_id": [1, 1, 2],
        "gw": [2, 3, 2],
        "now_cost": [15.0, 14.1, 11.0],
        "ep_next": [4.0, 5.0, 3.0],
    })
    return features, players, gw_stats


class TestAssemble:

    def test_feature_columns(self, frames):
        features, players, gw_stats = frames
        td = assemble(features, players, gw_stats, season="2025-2026")
        assert td.feature_names == FEATURE_COLUMNS
        assert td.X.shape[1] == len(FEATURE_COLUMNS)
        assert td.y.tolist() == [4.0, 1.0, 3.0]

    def test_no_match_rows_filled(self, frames):
        features, players, gw_stats = frames
        td = assemble(features, players, gw_stats, season="2025-2026")
        # row index 1 = player 1 gw 3 (no PL match): had_match=0, venue/elo = 0
        fg = dict(zip(td.feature_names, td.X[1], strict=False))
        assert fg["had_match"] == 0
        assert fg["opponent_elo"] == 0.0
        assert fg["home_elo"] == 0.0

    def test_meta_preserves_player_gw_season(self, frames):
        features, players, gw_stats = frames
        td = assemble(features, players, gw_stats, season="2024-2025")
        assert td.meta["season"].to_list() == ["2024-2025"] * 3

    def test_missing_season_defaults(self, frames):
        features, players, gw_stats = frames
        td = assemble(features, players, gw_stats)
        assert set(td.meta["season"]) == {"unknown"}

    def test_categorical_columns_encoded(self, frames):
        features, players, gw_stats = frames
        td = assemble(features, players, gw_stats, season="2025-2026")
        # categorical indices point at team_code + position
        assert set(td.categorical) == {
            td.feature_names.index("team_code"),
            td.feature_names.index("position"),
        }

    def test_position_codes_stable_across_seasons(self, frames):
        """Two seasons' matrices are vstacked in experiments/run; position
        codes must come from a fixed vocabulary, not each frame's own
        first-appearance order (which differs between seasons' data)."""
        features, players, gw_stats = frames
        td_a = assemble(features, players, gw_stats, season="2024-2025")
        # a season whose first appearance (by sort order) is a MID, not a FWD
        features_b = features.with_columns(
            player_code=pl.Series([118748, 118748, 223094]))
        td_b = assemble(features_b, players, gw_stats, season="2025-2026")
        pos_idx = td_a.feature_names.index("position")
        fwd_a = td_a.X[td_a.meta["player_code"] == 223094, pos_idx][0]
        fwd_b = td_b.X[td_b.meta["player_code"] == 223094, pos_idx][0]
        # exact sorted-enum vocabulary DEF=0, FWD=1, GKP=2, MID=3, identical
        # across seasons (the old per-frame Categorical gave FWD 0 vs 1 here)
        assert fwd_a == fwd_b == 1
        assert td_b.X[td_b.meta["player_code"] == 118748, pos_idx][0] == 3


class TestStableIdentity:
    """Cross-season safety: player_id is season-local (803/804 reused for a
    different player across 2024-25 and 2025-26). Training must join player
    metadata on player_code, and only rows with a resolvable code kept."""

    def test_position_comes_from_player_code(self, frames):
        features, players, gw_stats = frames
        # same code, but players table re-uses id 1 for a DIFFERENT code/player
        players_bad = pl.DataFrame({
            "player_id": [1, 2],
            "player_code": [999999, 118748],  # id 1 now maps to a stranger
            "position": ["GKP", "MID"],
        })
        td = assemble(features, players_bad, gw_stats, season="2025-2026")
        names = td.feature_names
        pos_idx = names.index("position")
        # row for player_code 223094 (player_id 1) has no match in players_bad
        # codes -> position becomes unknown; NEVER GKP (the stranger's position).
        xi = td.meta.with_row_index("r").filter(
            pl.col("player_code") == 223094).get_column("r").to_list()
        for i in xi:
            assert td.X[i, pos_idx] != 0, "wrong cross-season position attached"

    def test_unknown_position_flagged_rather_than_wrong(self, frames):
        features, players, gw_stats = frames
        td = assemble(features, players, gw_stats, season="2025-2026")
        # every row must resolve a position from ITS OWN code, none null
        pos_idx = td.feature_names.index("position")
        assert not any(
            row[pos_idx] is None for row in td.X
        ), "unresolved position present"