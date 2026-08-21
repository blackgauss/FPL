"""Contract tests: canonical forecast schema + strict normalizer."""

import polars as pl
import pytest

from fpl.experiments.forecast import (
    DISTRIBUTION_FORECAST_COLUMNS,
    POINT_FORECAST_COLUMNS,
    SchemaError,
    normalize_forecast,
)


def test_point_accepts_expected_points():
    frame = pl.DataFrame({
        "player_code": [1, 2], "gw": [31, 31], "expected_points": [4.0, 5.0],
    })
    out = normalize_forecast(frame, kind="point")
    assert out.columns == list(POINT_FORECAST_COLUMNS)
    assert out["expected_points"].to_list() == [4.0, 5.0]


def test_point_aliases_pred_columns():
    frame = pl.DataFrame({"player_code": [1], "gw": [31], "pred": [3.5]})
    out = normalize_forecast(frame, kind="point")
    assert "expected_points" in out.columns and "pred" not in out.columns
    assert out["expected_points"].to_list() == [3.5]


def test_point_missing_value_raises():
    frame = pl.DataFrame({"player_code": [1], "gw": [31]})
    with pytest.raises(SchemaError, match="expected_points"):
        normalize_forecast(frame, kind="point")


def test_missing_keys_raise():
    frame = pl.DataFrame({"gw": [31], "expected_points": [4.0]})
    with pytest.raises(SchemaError, match="player_code/gw"):
        normalize_forecast(frame, kind="point")


def test_distribution_schema_complete():
    cols = ["player_code", "gw"] + \
        [f"q{int(q * 100):02d}" for q in (0.01, 0.05, 0.10, 0.25,
                                           0.50, 0.75, 0.90, 0.95, 0.99)]
    frame = pl.DataFrame({c: [0.0] for c in cols})
    out = normalize_forecast(frame, kind="distribution")
    assert out.columns == list(DISTRIBUTION_FORECAST_COLUMNS)


def test_unknown_kind_raises():
    with pytest.raises(SchemaError, match="unknown forecast kind"):
        normalize_forecast(pl.DataFrame(), kind="nope")