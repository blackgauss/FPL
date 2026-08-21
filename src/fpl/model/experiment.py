"""Rapid model experimentation: a thin, config-driven harness.

The journal's "rapid experimentation loop" needs candidates compared on the
same held-out slice without code changes per experiment. An experiment is a
dict (declared in YAML):

    {model: <registry name>, params: {...}, features: [subset, ...],
     seasons: [...], fit_gw_max: int, test_gw_min: int}

A model is a function of (X_train, y_train) -> fitted model with .predict().
The registry holds the estimators we can swap; add one to try a new family.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _lgbm(params: dict):
    import lightgbm as lgb

    def make(X, y, categorical):
        ds = lgb.Dataset(X, label=y, categorical_feature=categorical)
        cfg = {"objective": "regression", "metric": "mae", "learning_rate": 0.05,
               "num_leaves": 63, "min_child_samples": 20, "verbosity": -1}
        cfg.update(params)
        fitted = lgb.train(cfg, ds, num_boost_round=200)
        return fitted.predict

    return make


def _hist_gb(params: dict):
    from sklearn.ensemble import HistGradientBoostingRegressor

    def make(X, y, categorical=None):
        cfg = {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 63,
               "min_samples_leaf": 20}
        cfg.update(params)
        return HistGradientBoostingRegressor(**cfg).fit(X, y).predict

    return make


def _ridge(params: dict):
    from sklearn.linear_model import Ridge

    def make(X, y, categorical=None):
        cfg = {"alpha": 1.0}
        cfg.update(params)
        return Ridge(**cfg).fit(X, y).predict

    return make


REGISTRY = {
    "lgbm": _lgbm,
    "hist_gb": _hist_gb,
    "ridge": _ridge,
}


@dataclass(frozen=True)
class ExperimentResult:
    name: str
    model: str
    features: list[str]
    mae: float
    rmse: float
    fit_gw_max: int
    test_gw_min: int
    n_test: int


def run_experiment(
    train_data,
    fit_data,
    *,
    name: str,
    model: str,
    params: dict | None = None,
    feature_columns: list[str] | None = None,
    fit_gw_max: int,
    test_gw_min: int,
) -> ExperimentResult:
    """Fit on (train_data + fit_data rows with gw <= fit_gw_max), score on the
    held-out fit_data slice with gw >= test_gw_min. No leakage: test strictly
    follows fit. `train_data`/`fit_data` are TrainingData objects; they must
    share the same feature schema (feature subset is applied beforehand)."""
    if model not in REGISTRY:
        raise ValueError(f"unknown model {model!r}; registry={sorted(REGISTRY)}")

    X_tr = np.vstack([train_data.X, fit_data.X[fit_data.gw <= fit_gw_max]])
    y_tr = np.concatenate([train_data.y, fit_data.y[fit_data.gw <= fit_gw_max]])

    if fit_gw_max >= test_gw_min:
        raise ValueError(
            f"leaky split: fit max GW {fit_gw_max} >= test min GW {test_gw_min}")

    mask = fit_data.gw >= test_gw_min
    X_test, y_test = fit_data.X[mask], fit_data.y[mask]
    if X_test.shape[0] == 0:
        raise ValueError(
            f"no test rows: fit max {fit_gw_max} vs test min {test_gw_min}")

    predict = REGISTRY[model](params or {})(X_tr, y_tr, train_data.categorical)
    pred = predict(X_test)

    resid = y_test - pred
    return ExperimentResult(
        name=name,
        model=model,
        features=feature_columns or [],
        mae=float(np.abs(resid).mean()),
        rmse=float(np.sqrt(np.mean(resid**2))),
        fit_gw_max=fit_gw_max,
        test_gw_min=test_gw_min,
        n_test=int(mask.sum()),
    )


def print_results(results: list[ExperimentResult]) -> None:
    print(f"{'name':<26}{'model':<10}{'MAE':>8}{'RMSE':>8}{'n':>6}  features")
    print("-" * 88)
    for r in sorted(results, key=lambda r: r.mae):
        feats = ",".join(r.features) if r.features else "ALL"
        print(f"{r.name:<26}{r.model:<10}{r.mae:>8.3f}{r.rmse:>8.3f}{r.n_test:>6}  {feats}")
    print("-" * 88)
