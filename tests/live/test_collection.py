"""Synthetic API contract tests for league and manager collection."""

import json

from fpl.live.collection import COLLECTION_SCHEMA_VERSION, collect


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
        raise AssertionError(f"unexpected API request: {url}")


def test_collect_writes_manager_and_league_outputs(tmp_path):
    session = Session()
    frames = collect(league_id=9, entry_id=42, out_dir=tmp_path,
                     session=session)
    assert frames["standings"].height == 2
    assert frames["history"].height == 2
    assert frames["picks"].height == 2
    assert set(frames["picks"]["entry_id"].to_list()) == {42}
    metadata = json.loads((tmp_path / "collection.json").read_text())
    assert metadata["schema_version"] == COLLECTION_SCHEMA_VERSION
    assert metadata["gw_end"] == 2
    assert len(session.calls) == 5


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
