"""One thin router per web resource: params -> Store -> response."""

from __future__ import annotations

from fastapi import APIRouter

from fpl.web.api import forecast, league, meta, players, research, team, transfers

API_ROUTERS: list[APIRouter] = [
    meta.router,
    players.router,
    forecast.router,
    team.router,
    transfers.router,
    league.router,
    research.router,
]
