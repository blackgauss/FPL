"""GET /api/players — filterable explorer table + next-GW prediction."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

from fpl.web.queries import Store

router = APIRouter(prefix="/players")


def get_store(request: Request) -> Store:
    return request.app.state.store


def current_gw(store: Store) -> int:
    try:
        return store.current_gw()
    except Exception:
        return 0


@router.get("")
def get_players(
    request: Request,
    search: str | None = None,
    position: str | None = None,
    club: int | None = None,
    status: str | None = None,
    max_price: int | None = Query(None, description="tenths, e.g. 60 = £6.0m"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> dict:
    store = get_store(request)
    try:
        out = store.players(
            search=search, position=position, club=club, status=status,
            max_price=max_price, limit=limit, offset=offset)
    except Exception:
        return {"available": False, "season": store.season, "total": 0, "rows": []}

    gw = current_gw(store)
    preds: dict[int, float] = {}
    predicted = store.predicted_next(gw + 1)
    if predicted is not None:
        preds = {r["player_code"]: r["pred"] for r in predicted.to_dicts()}
    for row in out["rows"]:
        row["pred_next"] = preds.get(row.get("player_code"))
    out["current_gw"] = gw
    return out
