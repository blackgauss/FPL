"""Attach distributional point forecasts to the team-search pipeline.

Fits residual CDFs on a held-out GW slice (no leakage), then for the
prediction horizon [gw_start, gw_end] produces a per-player-GW frame carrying
the point prediction AND the quantile structure of its distribution.

The context bin is (position, price_band): cheap players are measurably
burstier — higher coefficient-of-variation of points in every position — so
binned residuals give cheap over-hyped picks a wider/tail-heavier distribution
than premium stars at the same predicted mean. The team simulator MC-samples
squad totals from these CDFs so squads differ by variance, not just mean.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from fpl.dist import QS, fit_residual_cdfs, price_band, quantiles_of
from fpl.model.inference import load_model
from fpl.model.train import load_training


def collect_residuals(
    processed: str, seasons: list[str],
    *,
    fit_gw_max: int, test_gw_min: int,
    bins: str = "position",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Held-out (actual, predicted, context) residual material.

    Trains a fresh ege model on GW <= fit_gw_max across the two seasons, then
    predicts the held-out GW >= test_gw_min slice; returns aligned arrays so
    CDFs are fit without test rows ever entering the fit.

    `bins="position"` (default) groups residuals by position only — reliable
    at current holdout sizes. `bins="position_price"` uses (position,
    price_band), which is more discriminative but the £9+ bins are
    structurally thin (few expensive players) — revisit once more seasons
    accrue.
    """
    import lightgbm as lgb

    by_season = load_training(processed, seasons)
    first = by_season[seasons[0]]
    last = by_season[seasons[-1]]

    mask_fit = last.gw <= fit_gw_max
    X_tr = np.vstack([first.X, last.X[mask_fit]])
    y_tr = np.concatenate([first.y, last.y[mask_fit]])
    ds = lgb.Dataset(X_tr, label=y_tr,
                     feature_name=first.feature_names,
                     categorical_feature=first.categorical)
    model = lgb.train(
        {"objective": "regression", "metric": "mae", "num_leaves": 63,
         "learning_rate": 0.05, "min_child_samples": 20, "verbosity": -1},
        ds, num_boost_round=200,
    )

    mask_test = last.gw >= test_gw_min
    act = last.y[mask_test]
    pred = model.predict(last.X[mask_test])

    plr = pl.read_parquet(f"{processed}/players_{seasons[-1]}.parquet")
    gw_stats = pl.read_parquet(f"{processed}/gw_stats_{seasons[-1]}.parquet")
    meta = last.meta.filter(pl.col("gw") >= test_gw_min)
    ctx = (
        meta.join(plr.select("player_id", "position"), on="player_id", how="left")
        .join(gw_stats.select("player_id", "now_cost").group_by("player_id").agg(
            pl.col("now_cost").last()), on="player_id", how="left")
    )
    pos = ctx["position"].to_numpy().astype(object)
    if bins == "position_price":
        price = ctx["now_cost"].fill_null(0.0).to_numpy().astype(float)
        key = np.asarray(
            [(p, price_band(c)) for p, c in zip(pos, price, strict=False)],
            dtype=object)
    else:
        key = pos
    return np.asarray(act, float), np.asarray(pred, float), key


def distributional_forecast(
    processed: str,
    season: str,
    model_path: str,
    *,
    gw_start: int,
    gw_end: int,
    fit_gw_max: int = 30,
    test_gw_min: int = 31,
    bins: str = "position",
) -> pl.DataFrame:
    """Per-player-GW distributional forecast for [gw_start, gw_end].

    Returns one row per (player_code, gw): `pred` (point forecast) + a
    `quantiles` list (points at QS for the player's residual CDF, which is a
    t-digest per context bin). `bins` selects the residual binning (see
    collect_residuals).
    """
    from fpl.model.inference import expected_points_horizon

    model = load_model(model_path)
    td = load_training(processed, [season])[season]
    players = pl.read_parquet(f"{processed}/players_{season}.parquet")
    horizon = expected_points_horizon(
        td, model, gw_start=gw_start, gw_end=gw_end, players=players,
    )

    act, pred, keys = collect_residuals(
        processed, ["2024-2025", "2025-2026"],
        fit_gw_max=fit_gw_max, test_gw_min=test_gw_min, bins=bins,
    )
    cdfs = fit_residual_cdfs(act, pred, keys)
    # decompose the digest keys into join columns
    if bins == "position_price":
        cdf_df = pl.DataFrame(
            [(k[0], k[1], quantiles_of(cdfs[k]).tolist()) for k in cdfs],
            schema=["position", "price_band", "residual_qs"], orient="row",
        )
        # horizon players need their price band to pick the right CDF
        gw_price = (
            pl.read_parquet(f"{processed}/gw_stats_{season}.parquet")
            .select("player_id", "gw", "now_cost")
            .join(players.select("player_id", "player_code"),
                  on="player_id", how="left")
            .select("player_code", "gw", "now_cost")
        )
        horizon = horizon.join(gw_price, on=["player_code", "gw"], how="left")
        horizon = horizon.with_columns(
            pl.col("now_cost").fill_null(0.0).map_elements(
                price_band, return_dtype=pl.String).alias("price_band")
        )
        join_on = ["position", "price_band"]
    else:  # position only
        cdf_df = pl.DataFrame(
            [(k, quantiles_of(cdfs[k]).tolist()) for k in cdfs],
            schema=["position", "residual_qs"], orient="row",
        )
        join_on = "position"

    return (
        horizon.join(cdf_df, on=join_on, how="left")
        # missing context bin -> safe zero-noise default
        .with_columns(pl.col("residual_qs").fill_null(
            pl.lit([0.0] * len(QS), dtype=pl.List(pl.Float64))))
        .with_columns(
            (pl.col("expected_points")
             + pl.col("residual_qs").list.eval(pl.element().cast(pl.Float64)))
            .alias("quantiles_list")
        )
        .drop("residual_qs")
        .rename({"expected_points": "pred"})
        .with_columns(
            pl.col("quantiles_list").list.to_struct(
                fields=[f"q{int(q*100)}" for q in QS]).alias("quantiles_struct")
        )
        .drop(["quantiles_list", "now_cost", "price_band"] if bins == "position_price"
              else ["quantiles_list"])
        .select("player_code", "web_name", "position", "gw", "pred",
                "quantiles_struct")
    )