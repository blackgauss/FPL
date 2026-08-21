"""Evaluation helpers for point-forecast models.

Forecasting FPL points is count-like and zero-inflated; the useful lens for
team selection is per-position calibration and whether predictions beat cheap
baselines (rolling mean, FPL's own ep_next). All metrics here are over
predicted-vs-actual arrays.
"""

from __future__ import annotations

import polars as pl


def mae(y_true: pl.Series, y_pred: pl.Series) -> float:
    return float((y_true - y_pred).abs().mean())


def rmse(y_true: pl.Series, y_pred: pl.Series) -> float:
    return float(((y_true - y_pred) ** 2).mean()) ** 0.5


def summarize(pred: pl.Series, actual: pl.Series, label: str) -> None:
    print(f"{label:<28} MAE={mae(actual, pred):.3f}  RMSE={rmse(actual, pred):.3f}")


def baseline_mean(y_train: pl.Series, y_test: pl.Series) -> pl.Series:
    """Baseline: always predict the training mean."""
    return pl.Series([float(y_train.mean())] * y_test.len())