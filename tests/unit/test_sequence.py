"""Tests for one-gameweek transfer and lineup planning."""

import polars as pl
import pytest

from fpl.weekly.sequence import plan_one_week


def _inputs():
    positions = (["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3)
    rows = []
    for i, position in enumerate(positions, 1):
        rows.append({"player_id": i, "player_code": i, "web_name": f"P{i}",
                     "position": position, "team_code": 100 + i})
    rows.append({"player_id": 16, "player_code": 16, "web_name": "New MID",
                 "position": "MID", "team_code": 200})
    players = pl.DataFrame(rows)
    slot_codes = [1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 2, 7, 12, 15]
    picks = pl.DataFrame({
        "entry_id": [7] * 15, "gw": [1] * 15, "element": slot_codes,
        "position": list(range(1, 16)), "multiplier": [2] + [1] * 14,
        "is_captain": [True] + [False] * 14,
        "is_vice_captain": [False, True] + [False] * 13,
    })
    live = players.select("player_id", "player_code", "web_name", "position",
                          "team_code").with_columns(
        pl.lit(50).alias("now_cost"),
        pl.lit(100).alias("chance_of_playing_next_round"),
        pl.lit("a").alias("status"), pl.lit(True).alias("can_select"),
        pl.col("position").replace({"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4})
        .alias("element_type"),
    )
    expected = {code: 1.0 for code in range(1, 16)} | {16: 5.0}
    return picks, players, live, expected


def test_one_week_plan_returns_legal_transfer_and_lineup_options():
    picks, players, live, expected = _inputs()
    result = plan_one_week(
        picks=picks, history=pl.DataFrame(), players=players, live=live,
        expected=expected, gw=1, bank_tenths=0, top=5,
    )
    assert result["gw"] == 2
    assert len(result["options"]) == 5
    best = result["options"][0]
    assert best["transfer_in"] == "New MID"
    assert best["expected_delta"] > 0
    assert best["ownership_in"] == 0.0
    assert best["ownership_out"] == 1.0
    assert len(best["starters"]) == 11
    assert len(best["bench"]) == 4
    assert best["captain"] in best["starters"]
    assert best["vice_captain"] in best["starters"]
    assert best["captain"] != best["vice_captain"]


def test_planner_survives_midwindow_club_transfers(tmp_path, monkeypatch):
    """Live bootstrap can legally show 4 players of one club after the
    window rotates clubs; the squad is built from collected (deadline-time)
    clubs so planning must not crash."""
    picks, players, live, expected = _inputs()
    live4 = live.with_columns(
        pl.when(pl.col("player_id").is_in([3, 4, 5])).then(pl.lit(106))
        .otherwise(pl.col("team_code")).alias("team_code"))
    result = plan_one_week(
        picks=picks, history=pl.DataFrame(), players=players, live=live4,
        expected=expected, gw=1, bank_tenths=0, top=3)
    assert result["gw"] == 2  # squad validated on snapshot clubs, not live

    import fpl.weekly.sequence as seq
    picks.write_parquet(tmp_path / "picks.parquet")
    pl.DataFrame().write_parquet(tmp_path / "hist.parquet")
    players.write_parquet(tmp_path / "players_S.parquet")
    calls: dict = {}
    monkeypatch.setattr(seq, "fetch_bootstrap", lambda: {})
    monkeypatch.setattr(seq, "to_live_frame", lambda _: live4)
    monkeypatch.setattr(seq, "load_training",
                        lambda *a, **k: calls.update(tr=True) or {"S": "td"})
    monkeypatch.setattr(seq, "load_model", lambda p: "model")

    def boom(*a, **k):
        raise ValueError("no feature rows for gameweeks 2..2 (need rows at gw-1)")

    monkeypatch.setattr(seq, "score_players", boom)
    out = tmp_path / "gw2_plan.json"
    result = seq.run_from_files(
        picks_path=str(tmp_path / "picks.parquet"),
        history_path=str(tmp_path / "hist.parquet"),
        processed=str(tmp_path), season="S", model_path="m", gw=1,
        bank_tenths=0, top=3, out=str(out), entry_id=None)
    assert result["expected_source"] == "official_ep"
    assert result["gw"] == 2 and out.exists()  # scored with ep_next values


def _flat(n_value_map):
    """Constant quantile grids (degenerate distributions at a fixed value)."""
    return {code: [float(v)] * 9 for code, v in n_value_map.items()}


def test_digest_mean_and_prob_greater():
    from fpl.weekly.sequence import digest_mean, prob_greater
    uniform = [10 * q for q in (0.01, 0.05, .1, .25, .5, .75, .9, .95, .99)]
    assert digest_mean(uniform) == pytest.approx(5.0, abs=0.05)
    assert prob_greater([7.0] * 9, [3.0] * 9) == 1.0
    assert prob_greater([5.0] * 9, [5.0] * 9) == 0.5
    mixed = prob_greater([1.0] * 4 + [9.0] * 5, [5.0] * 9)
    assert 0.4 < mixed < 0.6  # 5 of 9 cells win


def test_plan_reports_xi_distribution_and_beat_probability():
    from fpl.weekly.sequence import plan_one_week
    picks, players, live, expected = _inputs()
    dist = {code: [float(v)] * 9 for code, v in expected.items()}
    result = plan_one_week(
        picks=picks, history=pl.DataFrame(), players=players, live=live,
        expected=expected, gw=1, bank_tenths=0, top=10, distributions=dist)
    hold = next(o for o in result["options"] if o["transfer_out"] is None)
    assert hold["prob_beat_hold"] == 0.5  # identical lineups: pure ties
    gainers = [o for o in result["options"] if o["expected_delta"] > 0]
    assert gainers and all(o["prob_beat_hold"] > 0.9 for o in gainers)
    assert hold["xi_q10"] == hold["xi_q90"]  # degenerate grid: no spread
    got = sum(expected[c] for c in hold["starters"]) + expected[hold["captain"]]
    assert hold["xi_q50"] == pytest.approx(got)  # XI sum + captain doubled


def test_planner_survives_observed_club_drift(tmp_path):
    """A re-collected snapshot that moved players into one club (post-window
    transfers) must not brick planning: drift is tolerated; adding a NEW
    player from the drifted club is not."""
    picks, players, live, expected = _inputs()
    drifted = players.with_columns(
        pl.when(pl.col("player_id") <= 4).then(pl.lit(101))
        .otherwise(pl.col("team_code")).alias("team_code"))
    result = plan_one_week(
        picks=picks, history=pl.DataFrame(), players=drifted, live=live,
        expected=expected, gw=1, bank_tenths=0, top=5)
    assert result["gw"] == 2  # squad assembled despite four from club 101
