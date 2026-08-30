"""GET /api/meta — season, GW clock, snapshot freshness, artifact inventory."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Request

from fpl.web.queries import Store

router = APIRouter(prefix="/meta")


def get_store(request: Request) -> Store:
    return request.app.state.store


@router.get("")
def get_meta(request: Request) -> dict:
    store = get_store(request)
    try:
        current_gw = store.current_gw()
    except Exception:
        current_gw = 0

    live: dict = {"available": False, "fetched_at": None, "age_seconds": None}
    lv = store.live()
    if lv is not None:
        live["available"] = True
        fetched_at = lv[1] or None
        live["fetched_at"] = fetched_at
        if fetched_at:
            try:
                dt = datetime.fromisoformat(fetched_at)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                live["age_seconds"] = max(
                    0.0, (datetime.now(UTC) - dt).total_seconds())
            except ValueError:
                pass

    return {
        "season": store.season,
        "current_gw": current_gw,
        "entry_id": store.entry_id(),
        "live": live,
        "artifacts": store.artifacts(),
    }
