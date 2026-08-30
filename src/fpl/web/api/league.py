"""GET /api/league/standings — collected league table + latest GW points.

H2H leagues need two kinds of truth: the official FPL table shows everyone
rank 1 / 0 pts until points are awarded, while `resolved_standings.parquet`
(from collection's resolve_h2h) carries event-settled scores and W-D-L and
derived league points. When resolved data exists it wins; classic leagues
read the plain standings.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from fpl.web.queries import Store

router = APIRouter(prefix="/league")


def get_store(request: Request) -> Store:
    return request.app.state.store


@router.get("/standings")
def get_standings(request: Request, entry_id: int | None = None) -> dict:
    store = get_store(request)
    try:
        current_gw = store.current_gw()
    except Exception:
        current_gw = 0
    if entry_id is None:
        entry_id = store.entry_id()

    resolved = store.account("resolved_standings")
    if resolved is not None and resolved.height:
        rows = resolved.sort(
            ["league_points", "resolved_score"], descending=True).to_dicts()
        for i, row in enumerate(rows, start=1):
            row["rank"] = i
            row["is_self"] = row.get("entry_id") == entry_id
        return {"available": True, "kind": "h2h_resolved",
                "current_gw": current_gw, "points_gw": current_gw,
                "entry_id": entry_id, "rows": rows}

    frame = store.account("league_standings")
    if frame is None:
        return {"available": False, "current_gw": current_gw, "rows": [],
                "reason": "no collected league standings"}

    rows = frame.to_dicts()
    gw_points: dict[int, int] = {}
    points_gw = None
    history = store.account("team_history")
    if history is not None:
        hist_rows = history.to_dicts()
        events = [h["event"] for h in hist_rows if h.get("event") is not None]
        if events:
            points_gw = max(events)
            gw_points = {h["entry_id"]: h.get("points")
                         for h in hist_rows if h.get("event") == points_gw}

    for row in rows:
        row["gw_points"] = gw_points.get(row.get("entry_id"))
        row["is_self"] = row.get("entry_id") == entry_id
    rows.sort(key=lambda r: (r.get("rank") is None, r.get("rank") or 0,
                             -(r.get("total") or 0)))

    return {"available": True, "kind": "classic", "current_gw": current_gw,
            "points_gw": points_gw, "entry_id": entry_id, "rows": rows}
