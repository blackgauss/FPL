"""A/B: global point model vs validation-shrunk position corrections.

Four independent position models overfit slightly, while a global model can
miss position-specific behavior. This experiment blends them:

    prediction = global + alpha[position] * (position_model - global)

The alpha values are selected on source GWs 28..29 and evaluated on untouched
source GWs 30..37 (targets 31..38). All players remain in training.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl

from fpl.model.leakage import validate
from fpl.model.train import CATEGORY_COLUMNS, FEATURE_COLUMNS, load_training

P = "data/processed"
SEASONS = ["2024-2025", "2025-2026"]
POSITIONS = ["GKP", "DEF", "MID", "FWD"]


def main() -> None:
    validate(
        pl.read_parquet(f"{P}/features_2025-2026.parquet"),
        pl.read_parquet(f"{P}/players_2025-2026.parquet"),
        gw_train_max=27, gw_test_min=31,
    )
    print("leakage validation: PASS")
    by = load_training(P, SEASONS, feature_columns=FEATURE_COLUMNS,
                       categorical_columns=CATEGORY_COLUMNS)
    t24, t25 = by[SEASONS[0]], by[SEASONS[1]]
    p24 = t24.meta.join(
        pl.read_parquet(f"{P}/players_2024-2025.parquet")
        .select("player_id", "position"), on="player_id")["position"].to_numpy()
    p25 = t25.meta.join(
        pl.read_parquet(f"{P}/players_2025-2026.parquet")
        .select("player_code", "position"), on="player_code")["position"].to_numpy()
    train25 = t25.gw <= 27
    params = {"objective": "regression", "metric": "mae", "learning_rate": 0.05,
              "num_leaves": 63, "min_child_samples": 20, "seed": 42,
              "verbosity": -1}

    x = np.vstack([t24.X, t25.X[train25]])
    y = np.concatenate([t24.y, t25.y[train25]])
    global_model = lgb.train(
        params, lgb.Dataset(x, label=y, feature_name=t25.feature_names,
                            categorical_feature=t25.categorical), num_boost_round=200)
    position_models = {}
    for position in POSITIONS:
        m24, m25 = p24 == position, train25 & (p25 == position)
        xp = np.vstack([t24.X[m24], t25.X[m25]])
        yp = np.concatenate([t24.y[m24], t25.y[m25]])
        position_models[position] = lgb.train(
            params, lgb.Dataset(xp, label=yp, feature_name=t25.feature_names,
                                categorical_feature=t25.categorical),
            num_boost_round=200,
        )

    def predictions(mask):
        x_test = t25.X[mask]
        positions = p25[mask]
        global_pred = global_model.predict(x_test)
        position_pred = np.empty(len(global_pred))
        for position, model in position_models.items():
            ix = positions == position
            position_pred[ix] = model.predict(x_test[ix])
        return global_pred, position_pred, positions

    val_mask = (t25.gw >= 28) & (t25.gw <= 29)
    test_mask = (t25.gw >= 30) & (t25.gw <= 37)
    gv, pv, posv = predictions(val_mask)
    yv = t25.y[val_mask]
    alphas = {}
    for position in POSITIONS:
        ix = posv == position
        alphas[position] = min(
            (float(np.abs(yv[ix] - (gv[ix] + a * (pv[ix] - gv[ix]))).mean()), a)
            for a in np.arange(0, 1.01, 0.1)
        )[1]
    print("validation alphas:", {k: round(float(v), 1) for k, v in alphas.items()})

    gt, pt, post = predictions(test_mask)
    yt = t25.y[test_mask]
    blend = gt.copy()
    for position, alpha in alphas.items():
        ix = post == position
        blend[ix] = gt[ix] + alpha * (pt[ix] - gt[ix])
    meta = t25.meta.filter(test_mask).select("gw")
    top10 = np.zeros(len(yt), dtype=bool)
    for gw in meta["gw"].unique().to_list():
        ix = np.flatnonzero(meta["gw"].to_numpy() == gw)
        n = max(1, int(np.ceil(len(ix) * 0.10)))
        top10[ix[np.argsort(gt[ix])[-n:]]] = True
    for label, mask in [("all", np.ones(len(yt), dtype=bool)),
                        ("top10", top10)]:
        print(f"{label}: global={np.abs(yt[mask] - gt[mask]).mean():.3f} "
              f"position={np.abs(yt[mask] - pt[mask]).mean():.3f} "
              f"blend={np.abs(yt[mask] - blend[mask]).mean():.3f} "
              f"n={int(mask.sum())}")


if __name__ == "__main__":
    main()
