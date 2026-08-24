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

from fpl.domain import squad_from_frame
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


def fetch_h2h_standings(
    league_id: int, *, session: requests.Session, timeout: int = 30,
) -> list[dict]:
    """Fetch the current H2H league standings, including the AVERAGE row."""
    payload = _get_json(session, f"leagues-h2h/{league_id}/standings/",
                        timeout=timeout)
    return payload.get("standings", {}).get("results", [])


def fetch_h2h_matches(
    league_id: int, gw: int, *, session: requests.Session, timeout: int = 30,
) -> list[dict]:
    """Fetch all H2H fixtures for one gameweek."""
    results: list[dict] = []
    page = 1
    while True:
        payload = _get_json(
            session,
            f"leagues-h2h-matches/league/{league_id}/?event={gw}&page={page}",
            timeout=timeout,
        )
        results.extend(payload.get("results", []))
        if not payload.get("has_next", False):
            return results
        page += 1


def _history_rows(entry_id: int, payload: dict) -> list[dict]:
    return [{"entry_id": entry_id, **row} for row in payload.get("current", [])]


def _pick_rows(entry_id: int, gw: int, payload: dict) -> list[dict]:
    rows = []
    for pick in payload.get("picks", []):
        rows.append({"entry_id": entry_id, "gw": gw, **pick})
    return rows


def _event_rows(gw: int, payload: dict) -> list[dict]:
    return [{"gw": gw, "player_id": row["id"], **row.get("stats", {})}
            for row in payload.get("elements", [])]


def collect(
    *, league_id: int, entry_id: int, out_dir: str | Path,
    gw_start: int = 1, gw_end: int | None = None,
    league_picks: bool = False, skip_league: bool = False,
    league_type: str = "classic", resolve_h2h: bool = False,
    processed: str = "data/processed", season: str = "2026-2027",
    timeout: int = 30,
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
    if league_type not in {"classic", "h2h"}:
        raise ValueError("league_type must be 'classic' or 'h2h'")
    if resolve_h2h and league_type != "h2h":
        raise ValueError("resolve_h2h requires league_type='h2h'")
    if resolve_h2h:
        league_picks = True
    try:
        standings = (fetch_h2h_standings(league_id, session=session, timeout=timeout)
                     if league_type == "h2h" else
                     fetch_league_standings(league_id, session=session, timeout=timeout))
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
    event_rows = []
    matches = []
    if league_type == "h2h":
        matches = fetch_h2h_matches(league_id, gw_start, session=session,
                                    timeout=timeout)
    for gw in range(gw_start, end + 1):
        event_rows.extend(_event_rows(
            gw, _get_json(session, f"event/{gw}/live/", timeout=timeout)))
        pick_rows.extend(_pick_rows(
            entry_id, gw, _get_json(session, f"entry/{entry_id}/event/{gw}/picks/",
                                    timeout=timeout)))

    if league_picks:
        for standing in standings:
            if standing.get("entry") is None:
                continue
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
    } for row in standings if row.get("entry") is not None]
    standings_frame = pl.DataFrame(standings_rows) if standings_rows else pl.DataFrame(
        schema={"league_id": pl.Int64, "entry_id": pl.Int64, "rank": pl.Int64,
                "player_name": pl.String, "entry_name": pl.String,
                "total": pl.Int64, "event_total": pl.Int64, "last_rank": pl.Int64})
    history_frame = pl.DataFrame(history_rows)
    picks_frame = pl.DataFrame(pick_rows)
    event_frame = pl.DataFrame(event_rows)
    matches_frame = pl.DataFrame(matches)
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
    event_frame.write_parquet(output / "event_live.parquet")
    matches_frame.write_parquet(output / "league_matches.parquet")
    if resolve_h2h:
        resolved = resolve_h2h_standings(
            standings=standings, matches=matches_frame, picks=picks_frame,
            event_live=event_frame, players=pl.read_parquet(
                f"{processed}/players_{season}.parquet"),
        )
        resolved.write_parquet(output / "resolved_standings.parquet")
    _write_json(output / "collection.json", {"entry": entry, **metadata})
    return {"standings": standings_frame, "history": history_frame,
            "picks": picks_frame, "event": event_frame, "matches": matches_frame}


def resolve_h2h_standings(
    *, standings: list[dict], matches: pl.DataFrame, picks: pl.DataFrame,
    event_live: pl.DataFrame, players: pl.DataFrame,
) -> pl.DataFrame:
    """Resolve H2H standings from picks and event-live points.

    This intentionally uses the domain settlement rules rather than the
    partially-finalized scores returned by the entry/league endpoints.
    """
    scores: dict[int, float] = {}
    for entry_id, group in picks.group_by("entry_id", maintain_order=True):
        eid = int(entry_id[0])
        group = group.sort("position")
        selected = group.join(
            players.select("player_id", "player_code", "web_name", "position",
                           "team_code"),
            left_on="element", right_on="player_id", how="inner",
        ).with_columns(pl.lit(0).alias("price_tenths"))
        frame = selected.select("player_code", "web_name", "position_right",
                                "team_code", "price_tenths").rename(
                                    {"position_right": "position"})
        squad = squad_from_frame(frame, gw=int(group["gw"][0]))
        code_by_element = dict(zip(selected["element"], selected["player_code"],
                                   strict=False))
        squad = squad.__class__(
            players=squad.players, gw=squad.gw,
            starters=tuple(code_by_element[int(element)] for element in
                           group.filter(pl.col("position") <= 11)["element"]),
            bench=tuple(code_by_element[int(element)] for element in
                        group.filter(pl.col("position") > 11)["element"]),
            captain=int(selected.filter(pl.col("is_captain"))["player_code"].item()),
            vice_captain=int(selected.filter(pl.col("is_vice_captain"))
                             ["player_code"].item()),
        )
        actual = selected.select("element", "player_code").join(
            event_live, left_on="element", right_on="player_id", how="left",
        ).with_columns(pl.col("minutes").fill_null(0),
                       pl.col("total_points").fill_null(0))
        scores[eid] = float(squad.gw_settlement(
            dict(zip(actual["player_code"], actual["minutes"] > 0, strict=False)),
            dict(zip(actual["player_code"], actual["total_points"], strict=False)),
        ).gw_total)

    table = {int(row["entry"]): {
        "entry_id": int(row["entry"]), "entry_name": row.get("entry_name"),
        "player_name": row.get("player_name"), "resolved_score": scores[int(row["entry"])],
        "wins": 0, "draws": 0, "losses": 0, "league_points": 0,
    } for row in standings if row.get("entry") is not None and int(row["entry"]) in scores}
    for row in matches.iter_rows(named=True):
        a, b = row.get("entry_1_entry"), row.get("entry_2_entry")
        if a is None or b is None or int(a) not in table or int(b) not in table:
            continue
        a, b = int(a), int(b)
        if scores[a] == scores[b]:
            for eid in (a, b):
                table[eid]["draws"] += 1
                table[eid]["league_points"] += 1
        else:
            winner, loser = (a, b) if scores[a] > scores[b] else (b, a)
            table[winner]["wins"] += 1
            table[winner]["league_points"] += 3
            table[loser]["losses"] += 1
    return pl.DataFrame(list(table.values())).sort(
        ["league_points", "resolved_score"], descending=[True, True])
