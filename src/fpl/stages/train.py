"""Train and evaluate a tree point-forecast model on the feature store.

Split by game week (no leakage): train on all of 2024-25 + 2025-26 GW 1..30,
held-out test on 2025-26 GW 31..38. Baselines (constant mean, persist last
GW, FPL's own ep_next) are evaluated on the same held-out rows.

Run:  python -m fpl.stages.train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from fpl.model.eval import baseline_mean, summarize
from fpl.model.inference import save_model
from fpl.model.leakage import validate
from fpl.model.train import load_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the point forecast model")
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--train-season", default="2024-2025")
    parser.add_argument("--season", default="2025-2026")
    parser.add_argument("--fit-gw-max", type=int, default=30)
    parser.add_argument("--test-gw-start", type=int, default=31)
    parser.add_argument("--test-gw-end", type=int, default=38)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    processed = args.processed
    seasons = [args.train_season, args.season]
    fit_gw_max = args.fit_gw_max
    test_gws = (args.test_gw_start, args.test_gw_end)

    by_season = load_training(processed, seasons)
    train_data, fit_data = by_season[args.train_season], by_season[args.season]

    # leakage gate (journal "Data" section): run before any fitting
    validate(
        pl.read_parquet(f"{processed}/features_{args.season}.parquet"),
        pl.read_parquet(f"{processed}/players_{args.season}.parquet"),
        gw_train_max=fit_gw_max,
        gw_test_min=test_gws[0],
    )
    print("leakage validation: PASS")

    # fit only on train (2024-25) + 2025-26 GW 1..30; never the held-out GWs
    fit_excl_test_mask = fit_data.gw <= fit_gw_max
    X_train = np.vstack([train_data.X, fit_data.X[fit_excl_test_mask]])
    y_train = np.concatenate([train_data.y, fit_data.y[fit_excl_test_mask]])

    lgb_train = lgb.Dataset(
        X_train, label=y_train,
        feature_name=train_data.feature_names,
        categorical_feature=train_data.categorical,
    )
    model = lgb.train(
        {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 20,
            "num_boost_round": 200,
            "verbosity": -1,
            "seed": args.seed,
            "bagging_seed": args.seed,
            "feature_fraction_seed": args.seed,
            "data_random_seed": args.seed,
            "deterministic": True,
            "force_col_wise": True,
            "num_threads": 1,
        },
        lgb_train,
        num_boost_round=200,
    )

    save_model(model, "data/processed/points_lgbm.txt")
    print("saved model -> data/processed/points_lgbm.txt")

    # held-out test slice
    mask = (fit_data.gw >= test_gws[0]) & (fit_data.gw <= test_gws[1])
    y_test = pl.Series("actual", fit_data.y[mask])
    pred_tree = pl.Series("tree", model.predict(fit_data.X[mask]))

    # raw rows for baselines (prev_points from feature store, ep_next from gw_stats)
    raw = pl.read_parquet(f"{processed}/features_{args.season}.parquet").filter(
        pl.col("gw").is_between(*test_gws)
    )
    gw_stats = pl.read_parquet(f"{processed}/gw_stats_{args.season}.parquet").select(
        "player_id", "gw", "ep_next"
    )
    raw = raw.join(gw_stats, on=["player_id", "gw"], how="left")
    y_raw = pl.Series("actual", raw.get_column("next_points"))
    prev_baseline = pl.Series("prev", raw.get_column("prev_points"))
    ep_baseline = pl.Series("ep", raw.get_column("ep_next").fill_null(0.0))

    print(f"\nheld-out {args.season} GW {test_gws[0]}..{test_gws[1]} "
          f"({y_test.len()} rows)")
    print("=" * 72)
    summarize(pred_tree, y_test, "LightGBM (tree)")
    summarize(prev_baseline, y_raw, "persist last GW (prev_points)")
    summarize(ep_baseline, y_raw, "FPL ep_next")
    summarize(baseline_mean(train_data.y, y_test), y_test, "constant train mean")

    # emit a flat metrics file for `dvc metrics show` (same numbers as above)
    def _err(pred: np.ndarray, act: np.ndarray) -> dict:
        return {"mae": float(np.abs(pred - act).mean()),
                "rmse": float(np.sqrt(np.mean((pred - act) ** 2)))}

    ty, ry = np.asarray(y_test), np.asarray(y_raw)
    scores = {
        "LightGBM_tree": _err(np.asarray(pred_tree), ty),
        "persist_last_gw": _err(np.asarray(prev_baseline), ry),
        "FPL_ep_next": _err(np.asarray(ep_baseline), ry),
        "constant_train_mean": _err(
            np.full(ty.shape, float(train_data.y.mean())), ty),
    }
    metrics_path = Path(processed) / "mae.json"
    metrics_path.write_text(json.dumps(scores, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    print(f"metrics -> {metrics_path}")

    print("\nfeature importances (top 8, gain):")
    for name, imp in sorted(
        zip(train_data.feature_names, model.feature_importance("gain"), strict=False),
        key=lambda t: -t[1],
    )[:8]:
        print(f"  {name:<18} {imp:9.1f}")


if __name__ == "__main__":
    main()
