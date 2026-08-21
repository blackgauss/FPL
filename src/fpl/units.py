"""Shared units and pipeline constants (single source of truth).

The price-units trap bit several times: live FPL API reports `now_cost` in
tenths (155 = £15.5m), our dataset stores decimal millions (15.5), and the
feature store keeps decimal. All conversions funnel through this module so
nothing re-implements tenths math and drifts.
"""

from __future__ import annotations

import polars as pl

TENTHS_PER_POUND = 10          # 1.0 decimal (millions) -> 10 tenths
DEFAULT_BUDGET_TENTHS = 1000   # £100m in £0.1m units
MAX_PER_CLUB = 3

# canonical squad size requirements (FPL rules) — the single source. Both the
# search-space sizing and the enumerator must agree.
SQUAD_COUNTS = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}


def to_tenths(value: float) -> int:
    """decimal-millions price (15.5) -> tenths (155)."""
    return int(round(value * TENTHS_PER_POUND))


def to_millions(tenths: int) -> float:
    """tenths (155) -> decimal millions (15.5)."""
    return tenths / TENTHS_PER_POUND


def now_cost_to_tenths(col: pl.Expr | pl.Series) -> pl.Expr | pl.Series:
    """Normalise a decimal now_cost column/series to tenths (in place)."""
    return (col * TENTHS_PER_POUND).round().cast(pl.Int64)