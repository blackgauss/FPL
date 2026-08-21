"""A/B: baseline vs leakage-safe first-team regularity features.

Do not remove fringe players from training; give the model trailing role
context instead. For each source row, features use only the current and
previous five observed Gameweeks. The target is the following week, so target
minutes never enter the feature.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl

from fpl.model.leakage import validate
from fpl.model.train import CATEGORY_COLUMNS, FEATURE_COLUMNS, assemble, load_training

PROCESSED = "data/processed"
SEASONS = ["2024-2025", "2025-2026"]
EXTRA = ["appear_rate_5", "start_rate_5", "minutes_share_5"]


def regularity_frame(gw_stats: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for pid, group in gw_stats.group_by("player_id"):
        history = sorted(
            group.select("gw", "minutes", "starts").iter_rows(named=True),
            key=lambda row: row["gw"],
        )
        for i, row in enumerate(history):
            tail = history[max(0, i - 4): i + 1]
            rows.append({
                "player_id": int(pid[0]),
                "gw": int(row["gw"]),
                "appear_rate_5": sum((r["minutes"] or 0) > 0 for r in tail) / 5,
                "start_rate_5": sum((r["starts"] or 0) > 0 for r in tail) / 5,
                "minutes_share_5": sum(
                    min(float(r["minutes"] or 0), 90) for r in tail
                ) / 450,
            })
    return pl.DataFrame(rows)


def main() -> None:
    validate(
        pl.read_parquet(f"{PROCESSED}/features_2025-2026.parquet"),
        pl.read_parquet(f"{PROCESSED}/players_2025-2026.parquet"),
        gw_train_max=27, gw_test_min=31,
    )
    print("leakage validation: PASS")

    baseline = load_training(
        PROCESSED, SEASONS, feature_columns=FEATURE_COLUMNS,
        categorical_columns=CATEGORY_COLUMNS,
    )
    role, regularity = {}, {}
    for season in SEASONS:
        features = pl.read_parquet(f"{PROCESSED}/features_{season}.parquet")
        players = pl.read_parquet(f"{PROCESSED}/players_{season}.parquet")
        gw_stats = pl.read_parquet(f"{PROCESSED}/gw_stats_{season}.parquet")
        regularity[season] = regularity_frame(gw_stats)
        role[season] = assemble(
            features.join(regularity[season], on=["player_id", "gw"]),
            players, gw_stats, season, feature_columns=FEATURE_COLUMNS + EXTRA,
            categorical_columns=CATEGORY_COLUMNS,
        )

    params = {
        "objective": "regression", "metric": "mae", "learning_rate": 0.05,
        "num_leaves": 63, "min_child_samples": 20, "seed": 42,
        "verbosity": -1,
    }
    models = {}
    for name, data in [("baseline", baseline), ("role", role)]:
        old, current = data[SEASONS[0]], data[SEASONS[1]]
        old_train, current_train = old.gw <= 27, current.gw <= 27
        x = np.vstack([old.X[old_train], current.X[current_train]])
        y = np.concatenate([old.y[old_train], current.y[current_train]])
        ds = lgb.Dataset(x, label=y, feature_name=current.feature_names,
                         categorical_feature=current.categorical)
        models[name] = lgb.train(params, ds, num_boost_round=200)

    current = baseline[SEASONS[1]]
    test = (current.gw >= 30) & (current.gw <= 37)
    meta = (current.meta.with_row_index("_row")
            .join(regularity[SEASONS[1]], on=["player_id", "gw"], how="left")
            .sort("_row"))
    regular = ((meta["minutes_share_5"] >= 0.5)
               & (meta["start_rate_5"] >= 0.4)).to_numpy()[test]
    actual = current.y[test]

    print("holdout rows: all / first-team regular / non-regular")
    for name, data in [("baseline", baseline), ("role", role)]:
        pred = models[name].predict(data[SEASONS[1]].X[test])
        print(name)
        for label, mask in [("all", np.ones(len(actual), dtype=bool)),
                            ("regular", regular), ("nonregular", ~regular)]:
            print(f"  {label:<10} n={int(mask.sum()):4d} "
                  f"MAE={np.abs(pred[mask] - actual[mask]).mean():.3f} "
                  f"bias={(pred[mask] - actual[mask]).mean():+.3f}")


if __name__ == "__main__":
    main()
