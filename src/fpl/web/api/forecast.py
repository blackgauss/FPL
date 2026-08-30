"""GET /api/forecast — per player-GW distributional forecast rows."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from fpl.web.queries import Store

router = APIRouter(prefix="/forecast")

Q_KEYS = ["q1", "q5", "q10", "q25", "q50", "q75", "q90", "q95", "q99"]


def get_store(request: Request) -> Store:
    return request.app.state.store


class ForecastRow(BaseModel):
    player_code: int
    web_name: str | None = None
    gw: int
    pred: float | None = None
    quantiles: dict[str, Any]


class ForecastOut(BaseModel):
    season: str
    gw_start: int
    gw_end: int
    available: bool = True
    rows: list[ForecastRow]


@router.get("", response_model=ForecastOut)
def get_forecast(
    request: Request,
    player_codes: str | None = Query(
        None, description="comma-separated player codes"),
    position: str | None = None,
    gw_start: int | None = Query(
        None, ge=1, description="defaults to the next GW"),
    horizon: int = Query(5, ge=1, le=10),
) -> ForecastOut:
    store = get_store(request)
    if gw_start is None:
        try:
            gw_start = store.current_gw() + 1
        except Exception:
            gw_start = 1
    gw_end = gw_start + horizon - 1

    codes: set[int] | None = None
    if player_codes:
        try:
            codes = {int(c) for c in player_codes.split(",") if c.strip()}
        except ValueError as exc:
            raise HTTPException(400, f"bad player_codes: {exc}") from exc

    # cold build is expensive but memoized by the Store (process + webcache);
    # we simply wait for it here.
    try:
        frame = store.forecast(gw_start, gw_end)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(503, f"forecast build failed: {exc}") from exc

    rows: list[dict] = []
    for r in frame.to_dicts():
        if codes is not None and r.get("player_code") not in codes:
            continue
        if position is not None and r.get("position") != position:
            continue
        rows.append({
            "player_code": r["player_code"],
            "web_name": r.get("web_name"),
            "gw": r["gw"],
            "pred": r.get("pred"),
            "quantiles": {k: r.get(k) for k in Q_KEYS},
        })
    return ForecastOut(season=store.season, gw_start=gw_start, gw_end=gw_end,
                       rows=rows)
