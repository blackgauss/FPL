"""Why is model training slow? Is it expected?

Decomposes the LightGBM training stage into its components and benchmarks
them on the reference config (46k rows x 15 features, 200 rounds, 63 leaves):

  - data prep / assembly (load_training + X/y build)
  - lgb.Dataset construction, WITH categoricals and WITHOUT (isolates the
    categorical-prep cost)
  - actual lgb.train wall time + per-round cost (200 vs 50 rounds)
  - dtypes/sizes for context

Writes experiments/artifacts/profile/train_time.json (gitignored) and prints
a verdict comparing to what LightGBM should cost at this size.

Usage:
    python scripts/train_time_analysis.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from fpl.experiments.splits import TemporalSplit
from fpl.model.train import CATEGORY_COLUMNS, FEATURE_COLUMNS, load_training

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = str(ROOT / "data" / "processed")
OUT = ROOT / "experiments" / "artifacts" / "profile" / "train_time.json"
SEASONS = ("2024-2025", "2025-2026")
PARAMS = {"objective": "regression", "metric": "mae", "learning_rate": 0.05,
          "num_leaves": 63, "min_child_samples": 20, "seed": 42,
          "verbosity": -1}


def fit_model(X, y, categorical, *, rounds: int, feature_names=None):
    import lightgbm as lgb

    ds = lgb.Dataset(X, label=y, categorical_feature=categorical,
                     feature_name=feature_names or None)
    cfg = dict(PARAMS)
    cfg["num_boost_round"] = rounds
    t0 = time.perf_counter()
    lgb.train(cfg, ds, num_boost_round=rounds)
    return time.perf_counter() - t0, ds


def main() -> None:
    split = TemporalSplit(fit_gw_max=30, cal_start=31, cal_end=30,
                          test_start=31, test_end=None)
    by_season = load_training(PROCESSED, list(SEASONS))
    t24, t25 = by_season[SEASONS[0]], by_season[SEASONS[-1]]
    train_mask = t25.gw <= split.fit_gw_max

    t0 = time.perf_counter()
    x = np.vstack([t24.X, t25.X[train_mask]])
    y = np.concatenate([t24.y, t25.y[train_mask]])
    assemble_s = time.perf_counter() - t0

    cat_idx = [list(FEATURE_COLUMNS).index(c) for c in CATEGORY_COLUMNS]
    report = {
        "rows": int(x.shape[0]), "features": int(x.shape[1]),
        "dtype": str(x.dtype),
        "assemble_s": assemble_s,
    }

    # dataset construction with vs without categoricals
    import lightgbm as lgb

    t0 = time.perf_counter()
    lgb.Dataset(x, label=y, feature_name=list(FEATURE_COLUMNS),
                categorical_feature=cat_idx)
    ds_cat_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    lgb.Dataset(x, label=y, feature_name=list(FEATURE_COLUMNS))
    ds_plain_s = time.perf_counter() - t0
    report["dataset_with_cat_s"] = ds_cat_s
    report["dataset_plain_s"] = ds_plain_s

    train_200, _ = fit_model(x, y, cat_idx, rounds=200)
    train_50, _ = fit_model(x, y, cat_idx, rounds=50)
    report["train_200_s"] = train_200
    report["train_50_s"] = train_50
    report["per_round_s"] = (train_200 - train_50) / 150

    ox = x[:, ~np.isin(np.arange(x.shape[1]), cat_idx)]
    train_200_nocat, _ = fit_model(ox, y, [], rounds=200)
    report["train_200_no_categorical_s"] = train_200_nocat

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")

    print("training time decomposition")
    print("=" * 60)
    for key, value in report.items():
        print(f"  {key:<28} {value:.4f}" if isinstance(value, float)
              else f"  {key:<28} {value}")
    python_ok = train_200 < 10.0
    print("-" * 60)
    print("verdict: at ~46k rows the expected LightGBM fit for 200 rounds / "
          "63 leaves is 'seconds, not tens of seconds'.")
    print(f"  observed train_200 = {train_200:.2f}s "
          f"({'expected-ish' if python_ok else 'above expectation'})")
    print(f"  dataset(cat) = {ds_cat_s:.3f}s vs dataset(plain) = "
          f"{ds_plain_s:.3f}s "
          f"({'categoricals add real cost' if ds_cat_s > 0.5 else 'categorical prep is small'})")
    print(f"  per-round = {report['per_round_s'] * 1000:.1f}ms "
          f"({report['rows']:,} rows)")
    print(f"wrote -> {OUT}")


if __name__ == "__main__":
    main()