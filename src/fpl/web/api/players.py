"""GET /api/players — filterable explorer table + next-GW prediction."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from fpl.web.queries import Store

router = APIRouter(prefix="/players")


def get_store(request: Request) -> Store:
    return request.app.state.store


@router.get("")
def get_players(
    request: Request,
    search: str | None = None,
    position: str | None = None,
    club: int | None = None,
    status: str | None = None,
    max_price: int | None = Query(None, description="tenths, e.g. 60 = £6.0m"),
    sort: str | None = Query(None, description=f"one of {Store.SORTABLE}"),
    dir: str = Query("asc", pattern="^(asc|desc)$"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> dict:
    store = get_store(request)
    if sort is not None and sort not in Store.SORTABLE:
        raise HTTPException(400, f"unsortable column {sort!r}")
    try:
        out = store.players(
            search=search, position=position, club=club, status=status,
            max_price=max_price, limit=limit, offset=offset,
            sort=sort, descending=dir == "desc")
    except Exception:
        return {"available": False, "season": store.season, "total": 0, "rows": []}

    out["current_gw"] = store.clock()["current"]
    return out
