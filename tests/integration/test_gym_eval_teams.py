"""Integration: candidate squads -> gym evals -> observability (the PoC).

Runs the real pipeline on a dense synthetic season, replays the top
candidates through Eval against actual gw_stats, and verifies the baseline's
forecast-vs-actual story is produced — including that a forced starter dnp
shows up as a real substitution under the game rules.
"""

from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from fpl.data.contract import load_season
from fpl.data.features import build_features
from fpl.gym import Eval
from fpl.pipeline import run_basket
from tests.fixtures.synthetic import build_season_tree_dense
from tests.integration.test_pipeline_runner import ToyModel


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    from fpl.model.train import load_training

    root = Path(tmp_path_factory.mktemp("gym_eval"))
    build_season_tree_dense(root, n_players=40, n_gws=4)
    data = load_season(root, "2025-2026")
    feats = build_features(data.gw_stats, data.team_history, data.matches,
                           data.players)
    out = root / "processed"
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in [("features", feats), ("gw_stats", data.gw_stats),
                        ("players", data.players)]:
        frame.write_parquet(out / f"{name}_2025-2026.parquet")
    load_training(out, ["2025-2026"])
    return out, data


def _candidates_and_forecast(store, n_teams=4):
    from fpl.model.train import load_training
    from fpl.team.scoring import score_players

    out, data = store
    model = ToyModel()
    res = run_basket(processed=str(out), season="2025-2026", gw_start=2,
                     gw_end=4, model=model, freshness=False,
                     enum_kw={"n_teams": n_teams, "seed": 1})
    td = load_training(str(out), ["2025-2026"])["2025-2026"]
    _, per_gw = score_players(td, model, gw_start=2, gw_end=4,
                              players=data.players, detail=True)
    forecast = per_gw.select("player_code", "gw", "expected_points")
    forecastable = sorted(per_gw["gw"].unique().to_list())
    return res, forecast, forecastable, data


class TestGymEvalPoC:
    def test_candidates_replay_to_evals(self, store):
        res, forecast, forecastable, data = _candidates_and_forecast(store)
        assert res.squads, "pipeline must produce candidate squads"
        assert len(forecastable) >= 1
        start, weeks = forecastable[0], len(forecastable)

        evals = []
        for squad in res.squads[:2]:
            ev = Eval(replace(squad, gw=start), gw_stats=data.gw_stats,
                      players=data.players, weeks=weeks, forecast=forecast,
                      name=f"cand-{squad.players[0].code}-{squad.gw}").run()
            evals.append(ev)
            assert len(ev.weeks) == weeks
            assert ev.weeks[0].gw == start
            assert ev.total_predicted is not None
            assert ev.gap == pytest.approx(ev.total_predicted - ev.total_actual)
            assert "predicted" in ev.summary() and "gap" in ev.summary()
        # the baseline edge is observable: candidates *differ* in actual points
        assert max(e.total_actual for e in evals) - \
            min(e.total_actual for e in evals) > 0

    def test_forced_dnp_shows_up_as_real_rule_observability(self, store):
        res, forecast, forecastable, data = _candidates_and_forecast(store)
        squad = replace(res.squads[0], gw=forecastable[0])
        gk = next(c for c in squad.starters
                  if squad.by_code()[c].position.value == "GKP")
        gk_pid = data.players.filter(pl.col("player_code") == gk)[
            "player_id"].item()
        gw = forecastable[0]
        gw_stats = data.gw_stats.with_columns(
            pl.when((pl.col("player_id") == gk_pid) & (pl.col("gw") == gw))
            .then(0).otherwise(pl.col("minutes")).alias("minutes"))

        ev = Eval(squad, gw_stats=gw_stats, players=data.players, weeks=1,
                  forecast=forecast, name="with-dnp").run()
        week = ev.weeks[0]
        assert gk not in week.xi                    # dnp starter actually out
        assert week.substituted_in                  # bench cover came on
        assert ev.substitutions >= 1
        assert gk in week.dnps