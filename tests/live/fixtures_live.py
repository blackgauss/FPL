"""Shared fixtures for live/tests: tiny synthetic bootstrap payload + dataset.

These mimic the real FPL API `elements` structure (status enum, tenths prices)
without network. `make_payload` builds N players with controllable
status/news/price/team; `make_dataset` builds the matching local dataset the
hygiene comparison runs against.
"""

from __future__ import annotations

import polars as pl

DEFAULT_STATUSES = ["a", "a", "a", "d", "i", "s", "u", "a"]


def make_payload(
    *,
    codes: list[int] | None = None,
    prices_tenths: list[int] | None = None,
    statuses: list[str] | None = None,
    teams: list[int] | None = None,
    news: list[str] | None = None,
    n: int = 8,
) -> dict:
    """Synthetic bootstrap-static JSON: {elements: [...]} for to_live_frame."""
    codes = codes or list(range(223001, 223001 + n))
    prices = prices_tenths or [50 + (i % 8) * 10 for i in range(n)]
    statuses = statuses or (DEFAULT_STATUSES * n)[:n]
    teams = teams or [1 + (i % 5) for i in range(n)]
    news = news or ([""] * n)
    elements = []
    for i in range(n):
        elements.append({
            "id": i + 1,
            "code": codes[i],
            "web_name": f"P{i}",
            "team": teams[i],
            "team_code": teams[i] * 10,
            "element_type": 1 + (i % 4),
            "now_cost": prices[i],
            "status": statuses[i],
            "news": news[i],
            "news_added": None,
            "chance_of_playing_this_round": None,
            "chance_of_playing_next_round": None,
            "selected_by_percent": "3.4",
            "minutes": 90,
            "ep_next": "3.0",
            "ep_this": None,
            "removed": False,
            "can_select": True,
            "can_transact": True,
        })
    return {"elements": elements, "teams": [], "events": []}


def make_dataset(*, prices_tenths: list[int],
                 team_codes: list[int], n: int = 8) -> pl.DataFrame:
    """Local dataset frame aligned to make_payload player_codes 223001.."""
    return pl.DataFrame({
        "player_code": [223001 + i for i in range(n)],
        "now_cost": prices_tenths,
        "team_code": team_codes,
        "web_name": [f"P{i}" for i in range(n)],
    })