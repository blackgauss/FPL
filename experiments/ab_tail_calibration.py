"""A/B: global residual tails vs position/prediction-bucket tails.

Leakage-safe split:
  fit mean model: 2025-26 source GW <= 27 (+ 2024-25)
  fit residual quantiles: 2025-26 GW 28..30
  arbitrate: 2025-26 GW 31..38

The baseline adds one global residual quantile to the mean prediction. The
variant learns residual quantiles by position and predicted-points bucket.
Both arms keep the same mean model; comparison uses proper pinball loss and
coverage at q50/q90/q95. This tests tail calibration, not mean accuracy.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl

from fpl.model.leakage import validate
from fpl.model.train import CATEGORY_COLUMNS, FEATURE_COLUMNS, load_training

PROCESSED = "data/processed"
SEASONS = ["2024-2025", "2025-2026"]
TRAIN_MAX = 27
VALIDATION = (28, 30)
TEST = (31, 38)


def _bucket(pred: pl.Expr) -> pl.Expr:
    return (
        pl.when(pred < 2).then(pl.lit("low"))
        .when(pred < 4).then(pl.lit("mid"))
        .when(pred < 6).then(pl.lit("high"))
        .otherwise(pl.lit("elite"))
        .alias("bucket")
    )


def _pinball(actual: np.ndarray, pred: np.ndarray, q: float) -> float:
    error = actual - pred
    return float(np.mean(np.where(error >= 0, q * error, (q - 1) * error)))


def main() -> None:
    validate(
        pl.read_parquet(f"{PROCESSED}/features_2025-2026.parquet"),
        pl.read_parquet(f"{PROCESSED}/players_2025-2026.parquet"),
        gw_train_max=VALIDATION[1], gw_test_min=TEST[0],
    )
    print("leakage validation: PASS")

    by = load_training(
        PROCESSED, SEASONS, feature_columns=FEATURE_COLUMNS,
        categorical_columns=CATEGORY_COLUMNS,
    )
    train_data, test_data = by[SEASONS[0]], by[SEASONS[1]]
    train_mask = test_data.gw <= TRAIN_MAX
    X = np.vstack([train_data.X, test_data.X[train_mask]])
    y = np.concatenate([train_data.y, test_data.y[train_mask]])
    ds = lgb.Dataset(X, label=y, feature_name=test_data.feature_names,
                     categorical_feature=test_data.categorical)
    model = lgb.train({
        "objective": "regression", "metric": "mae", "learning_rate": 0.05,
        "num_leaves": 63, "min_child_samples": 20, "seed": 42,
        "verbosity": -1,
    }, ds, num_boost_round=200)

    rows = []
    for split, mask in [
        ("validation", (test_data.gw >= VALIDATION[0]) &
                        (test_data.gw <= VALIDATION[1])),
        ("test", (test_data.gw >= TEST[0]) & (test_data.gw <= TEST[1])),
    ]:
        pred = model.predict(test_data.X[mask])
        meta = test_data.meta.filter(mask).select("player_code", "gw")
        rows.append(meta.with_columns(
            pl.Series("pred", pred),
            pl.Series("actual", test_data.y[mask]),
            pl.lit(split).alias("split"),
        ))

    players = pl.read_parquet(f"{PROCESSED}/players_2025-2026.parquet")
    data = pl.concat(rows).join(players.select("player_code", "position"),
                                on="player_code")
    data = data.with_columns(_bucket(pl.col("pred")))
    validation = data.filter(pl.col("split") == "validation")
    test = data.filter(pl.col("split") == "test")

    print(f"tail comparison: validation={validation.height}, test={test.height}")
    print("=" * 72)
    for q in (0.50, 0.90, 0.95):
        global_q = validation.select(
            (pl.col("actual") - pl.col("pred")).quantile(q)
        ).item()
        grouped = (
            validation.with_columns(
                (pl.col("actual") - pl.col("pred")).alias("residual")
            )
            .group_by("position", "bucket")
            .agg(pl.col("residual").quantile(q).alias("residual_q"))
        )
        conditional = (
            test.join(grouped, on=["position", "bucket"], how="left")
            .with_columns(pl.col("residual_q").fill_null(global_q))
        )
        actual = test["actual"].to_numpy()
        global_pred = test["pred"].to_numpy() + global_q
        conditional_pred = (
            test["pred"].to_numpy() + conditional["residual_q"].to_numpy()
        )
        for label, pred in [("global", global_pred),
                            ("position+bucket", conditional_pred)]:
            coverage = float(np.mean(actual <= pred))
            loss = _pinball(actual, pred, q)
            print(f"q{q:.2f} {label:<17} coverage={coverage:.3f} "
                  f"pinball={loss:.3f}")


if __name__ == "__main__":
    main()
