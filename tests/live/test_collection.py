"""Synthetic API contract tests for league and manager collection."""

import json

import polars as pl
import pytest
import requests

from fpl.live.collection import COLLECTION_SCHEMA_VERSION, collect
from fpl.live.compare import compare_team


class Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class Session:
    def __init__(self):
        self.calls = []

    def get(self, url, *, headers, timeout):
        self.calls.append(url)
        if "standings" in url:
            return Response({"standings": {"has_next": False, "results": [{
                "entry": 42, "rank": 1, "player_name": "Manager",
                "entry_name": "My Team", "total": 100, "event_total": 60,
                "last_rank": 2,
            }, {
                "entry": 77, "rank": 2, "player_name": "Rival",
                "entry_name": "Rival Team", "total": 90, "event_total": 50,
                "last_rank": 3,
            }]}})
        if url.endswith("entry/42/"):
            return Response({"id": 42, "name": "Manager", "player_first_name": "A"})
        if url.endswith("entry/42/history/"):
            return Response({"current": [{"event": 1, "points": 60},
                                         {"event": 2, "points": 70}]})
        if "/entry/42/event/" in url:
            return Response({"picks": [{
                "element": 100, "position": 1, "multiplier": 2,
                "is_captain": True, "is_vice_captain": False,
            }]})
        if "/event/" in url and "/live/" in url:
            return Response({"elements": [{"id": 10, "stats": {
                "minutes": 90, "total_points": 6,
            }}]})
        raise AssertionError(f"unexpected API request: {url}")


def test_collect_writes_manager_and_league_outputs(tmp_path):
    session = Session()
    frames = collect(league_id=9, entry_id=42, out_dir=tmp_path,
                     session=session)
    assert frames["standings"].height == 2
    assert frames["history"].height == 2
    assert frames["picks"].height == 2
    assert frames["event"].height == 2
    assert set(frames["picks"]["entry_id"].to_list()) == {42}
    metadata = json.loads((tmp_path / "collection.json").read_text())
    assert metadata["schema_version"] == COLLECTION_SCHEMA_VERSION
    assert metadata["gw_end"] == 2
    assert len(session.calls) == 7


def test_collect_league_picks_adds_other_entries(tmp_path):
    session = Session()
    base_get = session.get

    def get(url, *, headers, timeout):
        if "/entry/77/event/" in url:
            return Response({"picks": [{"element": 200, "position": 1}]})
        return base_get(url, headers=headers, timeout=timeout)

    session.get = get
    frames = collect(league_id=9, entry_id=42, out_dir=tmp_path,
                     session=session, league_picks=True)
    assert set(frames["picks"]["entry_id"].unique().to_list()) == {42, 77}


def test_collect_can_keep_manager_data_when_league_unavailable(tmp_path):
    session = Session()
    base_get = session.get

    def get(url, *, headers, timeout):
        if "standings" in url:
            response = Response({})
            response.status_code = 404
            raise requests.HTTPError(response=response)
        return base_get(url, headers=headers, timeout=timeout)

    session.get = get
    frames = collect(league_id=9, entry_id=42, out_dir=tmp_path,
                     session=session, skip_league=True)
    assert frames["standings"].height == 0
    assert frames["history"].height == 2


def test_compare_team_uses_official_history_score_and_pick_xscore():
    picks = pl.DataFrame({
        "gw": [1, 1], "element": [10, 11], "position": [1, 12],
        "multiplier": [2, 0], "is_captain": [True, False],
        "is_vice_captain": [False, True],
    })
    players = pl.DataFrame({
        "player_id": [10, 11], "player_code": [100, 101],
        "web_name": ["A", "B"], "position": ["FWD", "MID"],
    })
    stats = pl.DataFrame({
        "player_id": [10, 11], "gw": [1, 1],
        "minutes": [90, 0], "total_points": [6, 10],
    })
    forecast = pl.DataFrame({
        "player_code": [100, 101], "gw": [1, 1],
        "expected_points": [4.0, 8.0],
    })
    history = pl.DataFrame({"event": [1], "points": [12]})
    rows, summary = compare_team(
        picks=picks, history=history, players=players,
        gw_stats=stats, forecast=forecast, gw=1)
    assert summary == {"gw": 1, "xscore": 8.0, "actual_score": 12.0,
                       "history_score": 12.0, "score_source": "entry_history",
                       "error": -4.0, "player_count": 2}
    assert rows["actual_points"].to_list() == [6, 10]


def test_compare_team_requires_entry_for_multi_team_pick_artifact():
    picks = pl.DataFrame({"entry_id": [1, 2], "gw": [1, 1],
                          "element": [10, 11], "position": [1, 1],
                          "multiplier": [1, 1]})
    with pytest.raises(ValueError, match="entry_id is required"):
        compare_team(
            picks=picks, history=pl.DataFrame({"event": [1], "points": [0]}),
            players=pl.DataFrame({"player_id": [10], "player_code": [100],
                                  "web_name": ["A"], "position": ["FWD"]}),
            gw_stats=pl.DataFrame({"player_id": [10], "gw": [1],
                                   "minutes": [90], "total_points": [1]}),
            forecast=pl.DataFrame({"player_code": [100], "gw": [1],
                                    "expected_points": [1.0]}), gw=1)


def test_resolve_h2h_scores_each_gw_separately() -> None:
    """Multi-GW picks must settle PER GW: two managers tied on season
    totals but splitting the GWs 1-1 are W-L/W-L, not a draw (comparing
    totals was the historical bug once a second GW is collected)."""
    from fpl.live.collection import resolve_h2h_standings

    pos = (["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3) * 2
    ids = list(range(1, 31))
    players = pl.DataFrame({
        "player_id": ids, "player_code": [1000 + i for i in ids],
        "web_name": [f"P{i}" for i in ids], "position": pos,
        "team_code": [i % 5 + 1 for i in range(30)], "price_tenths": [50] * 30,
    })
    picks_rows = []
    for eid, base in ((1, 0), (2, 15)):
        for gw in (1, 2):
            for slot in range(15):
                picks_rows.append({
                    "entry_id": eid, "gw": gw, "element": base + slot + 1,
                    "position": slot + 1, "multiplier": 2 if slot == 0 else 1,
                    "is_captain": slot == 0, "is_vice_captain": slot == 1,
                    "element_type": 1})
    picks = pl.DataFrame(picks_rows)

    def live(pid, gw, pts, minsec=90):
        return {"player_id": pid, "gw": gw, "minutes": minsec,
                "total_points": pts}
    event_live = pl.DataFrame([
        # GW1: A = 30*2(cap) + 10 + 10 = 80 ; B = 0 (cap plays blank) + 10 = 10
        live(1, 1, 30), live(2, 1, 10), live(3, 1, 10),
        live(16, 1, 0), live(17, 1, 10),
        # GW2 mirrored: A = 10 ; B = 30*2 + 10 + 10 = 80  → tied 90 totals
        live(16, 2, 30), live(17, 2, 10), live(18, 2, 10),
        live(2, 2, 10),
    ])
    matches = pl.DataFrame({
        "entry_1_entry": [1, 2], "entry_2_entry": [2, 1],
        "entry_1_points": [0, 0], "entry_2_points": [0, 0],
        "event": [1, 2], "is_bye": [False, False],
    })
    standings = [{"entry": 1, "entry_name": "A", "player_name": "a"},
                 {"entry": 2, "entry_name": "B", "player_name": "b"}]
    out = resolve_h2h_standings(standings=standings, matches=matches,
                               picks=picks, event_live=event_live,
                               players=players).sort("entry_id")
    rows = out.rows()
    r1 = dict(zip(out.columns, rows[0], strict=True))
    r2 = dict(zip(out.columns, rows[1], strict=True))
    # GW1: A 80 (cap×2 + VC + 5pt) vs B 10 -> A; GW2 mirrored -> 1-1 each,
    # even though A's totals (100: VC inherits captaincy when the captain
    # doesn't play) exceed B's (90). Comparing totals would have made A
    # win BOTH matches — the historical bug.
    assert r1["resolved_score"] == 100.0 and r2["resolved_score"] == 90.0
    assert (r1["wins"], r1["draws"], r1["losses"], r1["league_points"]) == (1, 0, 1, 3)
    assert (r2["wins"], r2["draws"], r2["losses"], r2["league_points"]) == (1, 0, 1, 3)
