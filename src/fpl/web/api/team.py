"""GET /api/team/flags — live health flags for the collected squad picks."""

from __future__ import annotations

from fastapi import APIRouter, Request

from fpl.live.filters import flag_squad_player
from fpl.web.queries import Store

router = APIRouter(prefix="/team")


def get_store(request: Request) -> Store:
    return request.app.state.store


def _default_entry(store: Store) -> int | None:
    for doc in store.account_json(r"^collection\.json$"):
        entry = doc.get("entry")
        if isinstance(entry, dict) and entry.get("id") is not None:
            return int(entry["id"])
    return None


def _comparison_summary(store: Store, gw: int) -> dict | None:
    wanted = f"gw{gw}_comparison.json"
    for doc in store.account_json(r"gw\d+_comparison"):
        if doc.get("file") == wanted:
            summary = doc.get("summary")
            return summary if isinstance(summary, dict) else None
    return None


@router.get("/flags")
def get_flags(request: Request, gw: int | None = None,
              entry_id: int | None = None) -> dict:
    store = get_store(request)
    picks = store.account("team_picks")
    if picks is None:
        return {"available": False, "reason": "no collected team_picks"}

    rows = picks.to_dicts()
    if entry_id is None:
        ids = {r["entry_id"] for r in rows if r.get("entry_id") is not None}
        entry_id = ids.pop() if len(ids) == 1 else _default_entry(store)
    if entry_id is not None:
        rows = [r for r in rows if r.get("entry_id") == entry_id]
    if gw is None:
        gws = [r["gw"] for r in rows if r.get("gw") is not None]
        gw = max(gws) if gws else 0
    rows = [r for r in rows if r.get("gw") == gw]
    if not rows:
        return {"available": False, "gw": gw, "entry_id": entry_id,
                "reason": f"no collected picks for gw {gw}"}

    # Row-wise flags via fpl.live.filters.flag_squad_player over picks x live.
    # Picks carry no price/club snapshot, so a Squad (squad_from_frame) is not
    # buildable from disk data alone -> price-move / club-transfer flags are
    # unavailable here; status flags are exact.
    live_map: dict[int, dict] = {}
    lv = store.live()
    if lv is not None:
        live_map = {r["player_id"]: r for r in lv[0].to_dicts()
                    if r.get("player_id") is not None}

    out_rows: list[dict] = []
    captain: dict | None = None
    vice_captain: dict | None = None
    for r in sorted(rows, key=lambda r: r.get("position") or 0):
        live = live_map.get(r["element"], {})
        if lv is None:
            flag = "no live snapshot"
        else:
            flag = flag_squad_player({"status": live.get("status")})
        row = {
            "player_id": r["element"],
            "web_name": live.get("web_name"),
            "player_code": live.get("player_code"),
            "slot": r.get("position"),
            "element_type": r.get("element_type"),
            "multiplier": r.get("multiplier"),
            "is_captain": bool(r.get("is_captain")),
            "is_vice_captain": bool(r.get("is_vice_captain")),
            "status": live.get("status"),
            "news": live.get("news"),
            "flag": flag,
        }
        if r.get("is_captain"):
            captain = row
        if r.get("is_vice_captain"):
            vice_captain = row
        out_rows.append(row)

    return {
        "available": True,
        "gw": gw,
        "entry_id": entry_id,
        "captain": captain,
        "vice_captain": vice_captain,
        "rows": out_rows,
        "comparison": _comparison_summary(store, gw),
    }
