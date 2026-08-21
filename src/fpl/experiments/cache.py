"""In-process memo caches for the experiment harness.

Building blocks are deterministic given fixed seeds, so repeated experiments
that share a config (same seasons/features/categoricals/split/model) can reuse
loaded TrainingData and fitted predictors instead of re-reading parquet and
re-fitting LightGBM. This is a per-process cache; cross-invocation caching is
deliberately left to future disk caching.

Callers must ``reset_experiment_cache()`` between logically separate runs
(tests do), because fit results are deterministic only for the same config.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

_TRAINING: dict[tuple, dict] = {}
_FIT: dict[tuple, Callable] = {}
_FORECAST: dict[tuple, object] = {}
_COUNTS = {"load_calls": 0, "load_hits": 0, "fit_calls": 0, "fit_hits": 0,
           "forecast_calls": 0, "forecast_hits": 0}


def reset_experiment_cache() -> None:
    """Clear in-process caches and counters (idempotent)."""
    _TRAINING.clear()
    _FIT.clear()
    _FORECAST.clear()
    for key in _COUNTS:
        _COUNTS[key] = 0


def cache_counts() -> dict[str, int]:
    """Call/hit counters for observability (e.g. printed by scripts)."""
    return dict(_COUNTS)


def cached_training(
    loader: Callable[[], dict],
    *,
    processed: str,
    seasons: tuple[str, ...],
    features: tuple[str, ...] | None,
    categorical: tuple[str, ...] | None,
) -> dict:
    """Memoized load_training-by-season keyed on identity of the config."""
    key = (processed, seasons, features, categorical)
    _COUNTS["load_calls"] += 1
    if key in _TRAINING:
        _COUNTS["load_hits"] += 1
        return _TRAINING[key]
    result = loader()
    _TRAINING[key] = result
    return result


def cache_id(key: tuple) -> str:
    """Stable, cross-process content id for a cache key.

    Uses a content hash (not builtin ``hash()``, which is randomized per
    process) so a config names the same disk file in every run.
    """
    import hashlib

    return hashlib.sha256(repr(key).encode()).hexdigest()[:16]


def cached_fit(
    fitter: Callable[[], Callable],
    *,
    key: tuple,
    disk_dir: str | Path | None = None,
    model_kind: str = "lgbm",
) -> Callable:
    """Memoized model fit (returns a predict callable) keyed on config.

    With `disk_dir` and `model_kind="lgbm"` this adds a cross-process cache:
    the LightGBM Booster is persisted to ``<disk_dir>/fit-<id>.txt`` and a
    fresh process loads it instead of re-fitting. Any disk failure falls back
    to fitting (never raises for cache reasons).
    """
    _COUNTS["fit_calls"] += 1
    if key in _FIT:
        _COUNTS["fit_hits"] += 1
        return _FIT[key]

    disk_path = None
    if disk_dir is not None:
        disk_path = Path(disk_dir) / f"fit-{cache_id(key)}.txt"
        if model_kind == "lgbm" and disk_path.exists():
            try:
                import lightgbm as lgb

                booster = lgb.Booster(model_file=str(disk_path))
                predict = booster.predict
                _COUNTS["fit_hits"] += 1
                _FIT[key] = predict
                return predict
            except Exception:  # noqa: BLE001 - corrupted cache -> refit
                pass

    predict = fitter()
    _FIT[key] = predict
    if disk_path is not None and model_kind == "lgbm":
        try:
            booster = getattr(predict, "__self__", None)
            if booster is not None and hasattr(booster, "save_model"):
                disk_path.parent.mkdir(parents=True, exist_ok=True)
                booster.save_model(str(disk_path))
        except Exception:  # noqa: BLE001 - persistence best-effort
            pass
    return predict


def fit_cache_key(
    *,
    processed: str,
    seasons: tuple[str, ...],
    fit_gw_max: int,
    model: str,
    params: dict | None,
    features: tuple[str, ...] | None,
    categorical: tuple[str, ...] | None,
) -> tuple:
    """Deterministic identity of a model fit.

    `processed` is deliberately part of the key: the same nominal config on a
    different data directory must not reuse a fit from another store.
    """
    return (
        processed, seasons, fit_gw_max, model,
        tuple(sorted((params or {}).items())), features, categorical,
    )


def cached_forecast(
    builder: Callable[[], object],
    *,
    key: tuple,
) -> object:
    """Memoized per-(fit, window) forecast frame (skips re-prediction)."""
    _COUNTS["forecast_calls"] += 1
    if key in _FORECAST:
        _COUNTS["forecast_hits"] += 1
        return _FORECAST[key]
    forecast = builder()
    _FORECAST[key] = forecast
    return forecast