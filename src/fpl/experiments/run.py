"""The single entry point: run a declared experiment to a result dict.

This is used by the CLI, by the gym, and by tests — never bypass it. It owns
splitting, leakage, caching of assembled data, cohort metrics, the optional
gym evaluation, and the artifact contract.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import polars as pl

from fpl.experiments import cache
from fpl.experiments.candidates import candidate_squads
from fpl.experiments.cohorts import cohort_masks
from fpl.experiments.forecast import normalize_forecast
from fpl.experiments.metrics import calibration_metrics, point_metrics, ranking_metrics
from fpl.experiments.splits import (
    TemporalSplit,
    source_masks,
    validate_feature_leakage,
)
from fpl.gym import Eval
from fpl.model.experiment import REGISTRY
from fpl.model.train import load_training


class _AsModel:
    """Expose a predict callable as a minimal model object."""

    def __init__(self, predict: Callable):
        self.predict = predict


def _source_forecast(td, predict, *, gw_start: int, gw_end: int) -> pl.DataFrame:
    """Native forecast for target GWs [gw_start, gw_end] from source rows."""
    rows = []
    for gw in range(gw_start, gw_end + 1):
        mask = td.gw == (gw - 1)
        if int(mask.sum()) == 0:
            continue
        meta = td.meta.filter(mask).select("player_code")
        rows.append(meta.with_columns(
            pl.Series("expected_points", np.asarray(predict(td.X[mask])).round(3)),
            pl.lit(gw).alias("gw"),
        ))
    if not rows:
        raise ValueError(f"no source rows for target GWs {gw_start}..{gw_end}")
    return pl.concat(rows).select("player_code", "gw", "expected_points")


def run_experiment(
    spec: dict,
    *,
    processed: str = "data/processed",
    git_sha: str | None = None,
    play_prob: Callable | None = None,
) -> dict:
    """Execute one declared experiment dict and return a serializable result.

    `play_prob(squad, gw) -> dict[code, float]` enables the gym's
    predicted-settlement mode; when None the gym uses actual settlement.
    """
    seasons = spec["seasons"]
    split = TemporalSplit(**spec["split"])
    # The leakage gate (identity join, target shift, split integrity) is part
    # of the experiment contract, not just the CLI: any caller of this entry
    # point gets the same guarantee before a single model is fitted.
    latest = seasons[-1]
    validate_feature_leakage(
        pl.read_parquet(f"{processed}/features_{latest}.parquet"),
        pl.read_parquet(f"{processed}/players_{latest}.parquet"),
        split)
    name = spec["name"]
    model_name = spec["model"]
    params = spec.get("params") or {}
    feats = spec.get("features")
    cats = spec.get("categorical_columns")
    gym_cfg = spec.get("gym")

    if gym_cfg is not None and feats is not None:
        raise ValueError(
            "gym evaluation requires the default feature set; custom "
            "`features` + `gym` is not supported (candidate scoring uses the "
            "default set)")

    by_season = cache.cached_training(
        lambda: load_training(processed, seasons, feature_columns=feats,
                              categorical_columns=cats),
        processed=processed, seasons=tuple(seasons),
        features=tuple(feats) if feats else None,
        categorical=tuple(cats) if cats else None)
    train_data, fit_data = by_season[seasons[0]], by_season[seasons[-1]]
    masks = source_masks(split, fit_data.gw)

    x_train = np.vstack([train_data.X, fit_data.X[masks["train"]]])
    y_train = np.concatenate([train_data.y, fit_data.y[masks["train"]]])
    fit_key = (
        tuple(seasons), split.fit_gw_max, model_name,
        tuple(sorted((params or {}).items())),
        tuple(feats) if feats else None,
        tuple(cats) if cats else None)
    predict = cache.cached_fit(
        lambda: REGISTRY[model_name](params)(
            x_train, y_train, train_data.categorical),
        key=fit_key)
    model = _AsModel(predict)

    pred = predict(fit_data.X[masks["test"]])
    actual = fit_data.y[masks["test"]]
    source_gw = fit_data.gw[masks["test"]]
    meta_test = fit_data.meta.filter(masks["test"])
    players = pl.read_parquet(f"{processed}/players_{seasons[-1]}.parquet")
    positions = meta_test.join(
        players.select("player_code", "position"), on="player_code", how="left"
    )["position"].to_numpy()

    cohort_mask = cohort_masks(pred, source_gw, positions)
    metrics = [
        {"cohort": cohort, "n": int(mask.sum()),
         **point_metrics(actual[mask], pred[mask])}
        for cohort, mask in cohort_mask.items() if np.any(mask)
    ]
    calibration = calibration_metrics(actual, pred)
    ranking = ranking_metrics(actual, pred, source_gw)

    result: dict = {
        "name": name, "model": model_name,
        "features": list(train_data.feature_names),
        "metrics": metrics,
        "ranking": ranking,
        "calibration": calibration,
        "gym": None,
    }

    if gym_cfg is not None:
        season_gym = gym_cfg.get("season", seasons[-1])
        gw_start = gym_cfg["gw_start"]
        gw_end = gym_cfg["gw_end"]
        td_gym = by_season.get(season_gym, fit_data)
        forecast = normalize_forecast(
            _source_forecast(td_gym, predict, gw_start=gw_start, gw_end=gw_end),
            kind="point")
        pack = candidate_squads(
            processed=processed, season=season_gym, gw_start=gw_start,
            gw_end=gw_end, model=model, n_teams=gym_cfg.get("n_teams", 4),
            seed=gym_cfg.get("seed", 1))
        squads = pack.squads[: gym_cfg.get("top", 2)]
        weeks = gw_end - gw_start + 1
        gw_stats = pl.read_parquet(f"{processed}/gw_stats_{season_gym}.parquet")
        evals = [
            Eval(replace(squad, gw=gw_start), gw_stats=gw_stats,
                 players=players, weeks=weeks, forecast=forecast,
                 play_prob=play_prob, name=name).run()
            for squad in squads
        ]
        # single gym run vis-a-vis multiple candidate squads: emit the canonical
        # observability per squad, keyed by rank
        result["gym"] = {
            "settlement": evals[0].settlement,
            "squads": len(evals),
            "runs": [ev.observability() for ev in evals],
        }
    return result