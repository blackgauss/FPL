"""GET /api/league/standings — collected league table + latest GW points."""

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

    frame = store.account("league_standings")
    if frame is None:
        return {"available": False, "current_gw": current_gw, "rows": [],
                "reason": "no collected league_standings"}

    rows = frame.to_dicts()

    # Latest collected per-entry GW score from team_history (event_live is
    # player-level; entry totals live in team_history).
    gw_points: dict[int, int] = {}
    points_gw = None
    history = store.account("team_history")
    if history is not None:
        hist_rows = history.to_dicts()
        events = [h["event"] for h in hist_rows if h.get("event") is not None]
        if events:
            points_gw = max(events)
            for h in hist_rows:
                if h.get("event") == points_gw:
                    gw_points[h["entry_id"]] = h.get("points")

    for row in rows:
        row["gw_points"] = gw_points.get(row.get("entry_id"))
        if entry_id is not None:
            row["is_self"] = row.get("entry_id") == entry_id
    rows.sort(key=lambda r: (r.get("rank") is None, r.get("rank") or 0,
                             -(r.get("total") or 0)))

    return {"available": True, "current_gw": current_gw,
            "points_gw": points_gw, "entry_id": entry_id, "rows": rows}
