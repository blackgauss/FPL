"""Train and evaluate a tree point-forecast model on the feature store.

Split by game week (no leakage): train on all of 2024-25 + 2025-26 GW 1..30,
held-out test on 2025-26 GW 31..38. Baselines (constant mean, persist last
GW, FPL's own ep_next) are evaluated on the same held-out rows.

Run:  python scripts/train_tree.py
"""

from __future__ import annotations

import lightgbm as lgb
import polars as pl

from fpl.model.eval import baseline_mean, summarize
from fpl.model.leakage import validate
from fpl.model.train import assemble

SEASONS = ["2024-2025", "2025-2026"]
FIT_GWS = (1, 30)
TEST_GWS = (31, 38)


def main() -> None:
    processed = "data/processed"

    train_parts = []
    for season in SEASONS:
        feat = pl.read_parquet(f"{processed}/features_{season}.parquet")
        players = pl.read_parquet(f"{processed}/players_{season}.parquet")
        gw_stats = pl.read_parquet(f"{processed}/gw_stats_{season}.parquet")
        train_parts.append(assemble(feat, players, gw_stats, season))

    train_data, fit_data = train_parts

    # leakage gate (journal "Data" section): run before any fitting
    validate(
        pl.read_parquet(f"{processed}/features_2025-2026.parquet"),
        pl.read_parquet(f"{processed}/players_2025-2026.parquet"),
        gw_train_max=30,
        gw_test_min=31,
    )
    print("leakage validation: PASS")

    lgb_train = lgb.Dataset(
        train_data.X, label=train_data.y,
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
        },
        lgb_train,
        num_boost_round=200,
    )

    # held-out test slice
    mask = (fit_data.gw >= TEST_GWS[0]) & (fit_data.gw <= TEST_GWS[1])
    y_test = pl.Series("actual", fit_data.y[mask])
    pred_tree = pl.Series("tree", model.predict(fit_data.X[mask]))

    # raw rows for baselines (prev_points from feature store, ep_next from gw_stats)
    raw = pl.read_parquet(f"{processed}/features_2025-2026.parquet").filter(
        pl.col("gw").is_between(*TEST_GWS)
    )
    gw_stats = pl.read_parquet(f"{processed}/gw_stats_2025-2026.parquet").select(
        "player_id", "gw", "ep_next"
    )
    raw = raw.join(gw_stats, on=["player_id", "gw"], how="left")
    y_raw = pl.Series("actual", raw.get_column("next_points"))
    prev_baseline = pl.Series("prev", raw.get_column("prev_points"))
    ep_baseline = pl.Series("ep", raw.get_column("ep_next").fill_null(0.0))

    print(f"\nheld-out 2025-26 GW {TEST_GWS[0]}..{TEST_GWS[1]} "
          f"({y_test.len()} rows)")
    print("=" * 72)
    summarize(pred_tree, y_test, "LightGBM (tree)")
    summarize(prev_baseline, y_raw, "persist last GW (prev_points)")
    summarize(ep_baseline, y_raw, "FPL ep_next")
    summarize(baseline_mean(train_data.y, y_test), y_test, "constant train mean")

    print("\nfeature importances (top 8, gain):")
    for name, imp in sorted(
        zip(train_data.feature_names, model.feature_importance("gain"), strict=False),
        key=lambda t: -t[1],
    )[:8]:
        print(f"  {name:<18} {imp:9.1f}")


if __name__ == "__main__":
    main()