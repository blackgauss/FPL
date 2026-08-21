"""Integration: the replay gym replays a Squad through real Gameweeks.

Uses the dense synthetic season (actual gw_stats minutes/points) to verify
the gym settles real weeks, auto-substitutes on dnps, and reports
forecast-vs-actual under the same doubling/substitution rule — the
evaluation harness for the baseline and, later, captain/transfer models.
"""

from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from fpl.data.contract import load_season
from fpl.domain import Player, PlayerIdentity, PlayerState, Position, Squad
from fpl.gym import Eval, replay

_POS = ("GKP", "DEF", "MID", "FWD")


@pytest.fixture(scope="module")
def season(tmp_path_factory):
    from tests.fixtures.synthetic import build_season_tree_dense

    root = Path(tmp_path_factory.mktemp("gym"))
    build_season_tree_dense(root, n_players=40, n_gws=4)
    return load_season(root, "2025-2026")


def _build_squad(players, gw=2, captain=None):
    """A position-legal 15-player Squad from season players (11 starters:
    1 GKP / 4 DEF / 5 MID / 1 FWD, plus a 4-man bench in priority order)."""
    by_pos = {p: [] for p in _POS}
    for row in players.sort("player_code").iter_rows(named=True):
        by_pos[row["position"]].append(row)
    starters = (by_pos["GKP"][:1] + by_pos["DEF"][:4]
                + by_pos["MID"][:5] + by_pos["FWD"][:1])
    bench = [by_pos["GKP"][1], by_pos["DEF"][4], by_pos["MID"][5],
             by_pos["FWD"][1]]
    all_rows = starters + bench
    ps = [Player(PlayerIdentity(r["player_code"], r["web_name"],
                                Position(r["position"])),
                 PlayerState(r["team_code"], 50)) for r in all_rows]
    return Squad(
        players=tuple(ps), gw=gw,
        starters=tuple(r["player_code"] for r in starters),
        bench=tuple(r["player_code"] for r in bench),
        captain=captain or starters[5]["player_code"],  # a MID starter
        vice_captain=starters[1]["player_code"],        # a DEF starter
    )


class TestGymReplay:
    def test_replays_weeks_with_no_policy(self, season):
        squad = _build_squad(season.players)
        res = replay(squad, gw_stats=season.gw_stats, players=season.players,
                     weeks=2)
        assert [r.gw for r in res] == [2, 3]
        assert [r.squad.gw for r in res] == [2, 3]   # no policy -> gw advances
        for r in res:
            assert len(r.xi) == 11                   # everyone plays in fixture
            assert set(r.xi) <= set(squad.codes())
            assert r.substituted_in == ()
            assert r.actual_points == pytest.approx(
                sum(r.xi_points.values())
                + (r.xi_points[r.captain_doubled] if r.captain_doubled else 0.0))
            assert r.predicted_points is None

    def test_auto_substitution_on_dnp(self, season):
        squad = _build_squad(season.players)
        gk = squad.starters[0]
        gk_pid = season.players.filter(pl.col("player_code") == gk)[
            "player_id"].item()
        # at GW3 the starting GK sits out (minutes=0); the bench GK played
        gw_stats = season.gw_stats.with_columns(
            pl.when((pl.col("player_id") == gk_pid) & (pl.col("gw") == 3))
            .then(0).otherwise(pl.col("minutes")).alias("minutes"))
        res = replay(squad, gw_stats=gw_stats, players=season.players, weeks=2)
        r = res[1]                                   # GW3
        assert gk not in r.xi                        # dnp starter out
        assert squad.bench[0] in r.substituted_in    # bench GK in (priority)
        assert gk not in r.xi_points

    def test_predictor_and_policy(self, season):
        squad = _build_squad(season.players)
        new_cap = squad.starters[2]

        def policy(s, gw):
            return replace(s, captain=new_cap, gw=gw + 1)

        def predictor(s, gw):
            return {c: 3.0 for c in s.codes()}

        res = replay(squad, gw_stats=season.gw_stats, players=season.players,
                     weeks=2, policy=policy, predictor=predictor)
        assert res[1].squad.captain == new_cap
        for r in res:
            # forecast under the same rule as actuals: xi sum, doubled captain
            assert r.predicted_points == pytest.approx(
                len(r.xi) * 3.0 + (3.0 if r.captain_doubled else 0.0))
            assert r.actual_points >= 0


class TestEvalProtocol:
    def test_run_with_forecast_frame(self, season):
        squad = _build_squad(season.players)
        # a flat model forecast: expected = 3.0 for every member every week
        forecast = pl.DataFrame({
            "player_code": [c for c in squad.codes() for _ in (2, 3)],
            "gw": [g for _ in squad.codes() for g in (2, 3)],
            "expected_points": [3.0] * (2 * len(squad.codes())),
        })
        res = Eval(squad, gw_stats=season.gw_stats, players=season.players,
                   weeks=2, forecast=forecast,
                   name="baseline-unit").run()
        assert len(res.weeks) == 2
        assert res.total_actual == pytest.approx(
            sum(w.actual_points for w in res.weeks))
        assert res.total_predicted is not None
        assert res.gap == pytest.approx(res.total_predicted - res.total_actual)
        assert res.substitutions == 0
        assert res.captain_weeks == len(res.weeks)
        assert "baseline-unit" in res.summary()
        assert "predicted" in res.summary() and "gap" in res.summary()

    def test_predictor_and_forecast_are_exclusive(self, season):
        squad = _build_squad(season.players)
        with pytest.raises(ValueError, match="not both"):
            Eval(squad, gw_stats=season.gw_stats, players=season.players,
                 weeks=1, predictor=lambda s, g: {}, forecast=season.gw_stats)