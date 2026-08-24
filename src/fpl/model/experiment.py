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

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl


def _lgbm(params: dict):
    import lightgbm as lgb

    def make(X, y, categorical):
        ds = lgb.Dataset(X, label=y, categorical_feature=categorical)
        cfg = {"objective": "regression", "metric": "mae", "learning_rate": 0.05,
               "num_leaves": 63, "min_child_samples": 20, "verbosity": -1,
               "seed": 0, "bagging_seed": 0, "feature_fraction_seed": 0,
               "data_random_seed": 0, "deterministic": True,
               "force_col_wise": True, "num_threads": 1}
        cfg.update(params)
        num_round = cfg.pop("num_boost_round", 200)
        fitted = lgb.train(cfg, ds, num_boost_round=num_round)
        return fitted.predict

    return make


def _hist_gb(params: dict):
    from sklearn.ensemble import HistGradientBoostingRegressor

    def make(X, y, categorical=None):
        cfg = {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 63,
               "min_samples_leaf": 20, "random_state": 0}
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
    top10_mae: float | None = None
    n_top10: int = 0


def run_experiment(
    train_data,
    fit_data,
    *,
    name: str,
    model: str,
    params: dict | None = None,
    fit_gw_max: int,
    test_gw_min: int,
    test_gw_max: int | None = None,
) -> ExperimentResult:
    """Fit on (train_data + fit_data rows with gw <= fit_gw_max), score on the
    held-out fit_data slice [test_gw_min, test_gw_max]. No leakage: test
    strictly follows fit. `fit_gw_max=0` trains on train_data only — the
    season-start (cold) setting where the test window is the NEXT season's
    first gameweeks. `train_data`/`fit_data` are TrainingData objects; the
    feature schema is baked in already and read back from
    `train_data.feature_names` for the report."""
    if model not in REGISTRY:
        raise ValueError(f"unknown model {model!r}; registry={sorted(REGISTRY)}")

    gw_max = test_gw_max if test_gw_max is not None else 10**6
    if fit_gw_max > 0:
        X_tr = np.vstack([train_data.X, fit_data.X[fit_data.gw <= fit_gw_max]])
        y_tr = np.concatenate([train_data.y, fit_data.y[fit_data.gw <= fit_gw_max]])
    else:  # season-start: train on the previous season only
        X_tr, y_tr = train_data.X, train_data.y

    if fit_gw_max >= test_gw_min:
        raise ValueError(
            f"leaky split: fit max GW {fit_gw_max} >= test min GW {test_gw_min}")

    mask = (fit_data.gw >= test_gw_min) & (fit_data.gw <= gw_max)
    X_test, y_test = fit_data.X[mask], fit_data.y[mask]
    if X_test.shape[0] == 0:
        raise ValueError(
            f"no test rows: fit max {fit_gw_max} vs test min {test_gw_min}")

    predict = REGISTRY[model](params or {})(X_tr, y_tr, train_data.categorical)
    pred = predict(X_test)

    resid = y_test - pred
    # Decision-relevant cohort: top 10% by prediction within each GW. This
    # uses prediction only, so it is available before outcomes are observed.
    test_meta = fit_data.meta.filter(pl.Series("test", mask)).select("gw")
    top10 = np.zeros(len(pred), dtype=bool)
    for gw in test_meta["gw"].unique().to_list():
        idx = np.flatnonzero(test_meta["gw"].to_numpy() == gw)
        n = max(1, int(np.ceil(len(idx) * 0.10)))
        top10[idx[np.argsort(pred[idx])[-n:]]] = True
    return ExperimentResult(
        name=name,
        model=model,
        features=list(train_data.feature_names),
        mae=float(np.abs(resid).mean()),
        rmse=float(np.sqrt(np.mean(resid**2))),
        fit_gw_max=fit_gw_max,
        test_gw_min=test_gw_min,
        n_test=int(mask.sum()),
        top10_mae=float(np.abs(resid[top10]).mean()),
        n_top10=int(top10.sum()),
    )


def print_results(results: list[ExperimentResult]) -> None:
    print(f"{'name':<26}{'model':<10}{'MAE':>8}{'RMSE':>8}{'top10':>8}{'n':>6}  features")
    print("-" * 88)
    for r in sorted(results, key=lambda r: r.mae):
        feats = ",".join(r.features) if r.features else "ALL"
        print(f"{r.name:<26}{r.model:<10}{r.mae:>8.3f}{r.rmse:>8.3f}"
              f"{r.top10_mae or 0:>8.3f}{r.n_test:>6}  {feats}")
    print("-" * 88)


def write_results(
    results: list[ExperimentResult],
    path: str | Path,
    *,
    metadata: dict | None = None,
) -> Path:
    """Write a completed, machine-readable experiment result artifact.

    The artifact is written only after every declared experiment completes;
    callers can therefore treat its presence and ``status=complete`` as proof
    that the comparison actually ran. Metadata carries the declared config,
    split, data fingerprint, git SHA, and any runner-specific context.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "complete",
        "results": [asdict(result) for result in results],
        "metadata": metadata or {},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path
