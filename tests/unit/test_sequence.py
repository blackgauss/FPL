"""Tests for one-gameweek transfer and lineup planning."""

import polars as pl

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
