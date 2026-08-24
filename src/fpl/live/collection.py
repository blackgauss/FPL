"""Collection of league and manager state from the FPL API.

The bootstrap snapshot is global player state. This module collects the
manager-specific state that cannot be inferred from it: classic-league
standings, entry history, and gameweek picks. Responses are normalized into
parquet outputs only after collection completes.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import requests

from fpl.live.live import HEADERS

API_ROOT = "https://fantasy.premierleague.com/api"
COLLECTION_SCHEMA_VERSION = 1


def _get_json(session: requests.Session, path: str, *, timeout: int) -> dict:
    response = session.get(f"{API_ROOT}/{path.lstrip('/')}",
                           headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def fetch_league_standings(
    league_id: int, *, session: requests.Session, timeout: int = 30,
    page_size: int = 50,
) -> list[dict]:
    """Fetch all classic-league standings pages."""
    results: list[dict] = []
    page = 1
    while True:
        payload = _get_json(
            session,
            f"leagues-classic/{league_id}/standings/?page_standings={page}"
            f"&page_new_entries=1&phase=1",
            timeout=timeout,
        )
        results.extend(payload.get("standings", {}).get("results", []))
        if not payload.get("standings", {}).get("has_next", False):
            return results
        page += 1
        if len(results) < page_size * (page - 1):
            raise ValueError("league standings pagination returned no progress")


def _history_rows(entry_id: int, payload: dict) -> list[dict]:
    return [{"entry_id": entry_id, **row} for row in payload.get("current", [])]


def _pick_rows(entry_id: int, gw: int, payload: dict) -> list[dict]:
    rows = []
    for pick in payload.get("picks", []):
        rows.append({"entry_id": entry_id, "gw": gw, **pick})
    return rows


def collect(
    *, league_id: int, entry_id: int, out_dir: str | Path,
    gw_start: int = 1, gw_end: int | None = None,
    league_picks: bool = False, skip_league: bool = False, timeout: int = 30,
    session: requests.Session | None = None,
) -> dict[str, pl.DataFrame]:
    """Collect one manager and optionally every visible league entry.

    `entry_id` is the manager/team ID shown in the FPL URL, not a league rank.
    When `gw_end` is omitted, the manager's latest completed history GW is used.
    """
    if gw_start < 1 or (gw_end is not None and gw_end < gw_start):
        raise ValueError("invalid gameweek collection range")
    session = session or requests.Session()
    output = Path(out_dir)
    standings_error = None
    try:
        standings = fetch_league_standings(league_id, session=session, timeout=timeout)
    except requests.HTTPError as exc:
        if not skip_league:
            raise
        standings = []
        standings_error = f"HTTP {exc.response.status_code if exc.response else 'error'}"
    if league_picks and not standings:
        raise ValueError("--league-picks requires accessible league standings")
    entry = _get_json(session, f"entry/{entry_id}/", timeout=timeout)
    history = _get_json(session, f"entry/{entry_id}/history/", timeout=timeout)
    latest = max((row.get("event", 0) for row in history.get("current", [])),
                 default=gw_start)
    end = gw_end or latest
    if end < gw_start:
        raise ValueError(f"no manager history in GW range {gw_start}..{end}")

    history_rows = _history_rows(entry_id, history)
    pick_rows = []
    for gw in range(gw_start, end + 1):
        pick_rows.extend(_pick_rows(
            entry_id, gw, _get_json(session, f"entry/{entry_id}/event/{gw}/picks/",
                                    timeout=timeout)))

    if league_picks:
        for standing in standings:
            other_id = int(standing["entry"])
            if other_id == entry_id:
                continue
            for gw in range(gw_start, end + 1):
                pick_rows.extend(_pick_rows(
                    other_id, gw,
                    _get_json(session, f"entry/{other_id}/event/{gw}/picks/",
                              timeout=timeout)))

    standings_rows = [{
        "league_id": league_id, "entry_id": int(row["entry"]),
        "rank": row.get("rank"), "player_name": row.get("player_name"),
        "entry_name": row.get("entry_name"), "total": row.get("total"),
        "event_total": row.get("event_total"), "last_rank": row.get("last_rank"),
    } for row in standings]
    standings_frame = pl.DataFrame(standings_rows) if standings_rows else pl.DataFrame(
        schema={"league_id": pl.Int64, "entry_id": pl.Int64, "rank": pl.Int64,
                "player_name": pl.String, "entry_name": pl.String,
                "total": pl.Int64, "event_total": pl.Int64, "last_rank": pl.Int64})
    history_frame = pl.DataFrame(history_rows)
    picks_frame = pl.DataFrame(pick_rows)
    metadata = {
        "schema_version": COLLECTION_SCHEMA_VERSION,
        "collected_at": datetime.now(UTC).isoformat(),
        "league_id": league_id, "entry_id": entry_id,
        "gw_start": gw_start, "gw_end": end,
        "league_picks": league_picks,
    }
    if standings_error:
        metadata["league_error"] = standings_error
    output.mkdir(parents=True, exist_ok=True)
    standings_frame.write_parquet(output / "league_standings.parquet")
    history_frame.write_parquet(output / "team_history.parquet")
    picks_frame.write_parquet(output / "team_picks.parquet")
    _write_json(output / "collection.json", {"entry": entry, **metadata})
    return {"standings": standings_frame, "history": history_frame,
            "picks": picks_frame}
