"""Canonical forecast schemas + a strict normalizer.

The single source of truth for "what a forecast looks like", so scripts stop
passing ``pred`` where the gym expects ``expected_points`` and vice-versa. Any
frame missing a required key raises :class:`SchemaError` before use.
"""

from __future__ import annotations

import polars as pl

from fpl.dist import QS

PointForecast = pl.DataFrame  # player_code, gw, expected_points
DistributionForecast = pl.DataFrame  # player_code, gw, q01..q99

POINT_FORECAST_COLUMNS = ("player_code", "gw", "expected_points")
DIST_QUANTILE_COLUMNS = tuple(f"q{int(q * 100):02d}" for q in QS)
DISTRIBUTION_FORECAST_COLUMNS = ("player_code", "gw") + DIST_QUANTILE_COLUMNS


class SchemaError(ValueError):
    """A forecast frame does not satisfy the canonical schema."""


def normalize_forecast(
    frame: pl.DataFrame, *, kind: str = "point",
) -> pl.DataFrame:
    """Normalize and validate a forecast frame to the canonical schema.

    - ``kind="point"``: requires ``player_code``, ``gw``; accepts either
      ``expected_points`` or ``pred`` (aliased) as the value column.
    - ``kind="distribution"``: requires ``player_code``, ``gw`` and all
      ``q01..q99`` quantile columns.

    Raises :class:`SchemaError` on any missing/extra-required column.
    """
    if kind == "point":
        columns = set(frame.columns)
        if not {"player_code", "gw"} <= columns:
            raise SchemaError(
                f"point forecast missing player_code/gw; got {sorted(columns)}")
        if "expected_points" not in columns:
            if "pred" in columns:
                frame = frame.rename({"pred": "expected_points"})
            else:
                raise SchemaError(
                    "point forecast needs `expected_points` (or `pred`); "
                    f"got {sorted(columns)}")
        missing = [c for c in POINT_FORECAST_COLUMNS if c not in frame.columns]
        if missing:
            raise SchemaError(f"point forecast missing columns: {missing}")
        return frame.select(*POINT_FORECAST_COLUMNS)
    if kind == "distribution":
        required = list(DISTRIBUTION_FORECAST_COLUMNS)
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise SchemaError(f"distribution forecast missing columns: {missing}")
        return frame.select(*required)
    raise SchemaError(f"unknown forecast kind {kind!r}")