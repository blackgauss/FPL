"""Profile the next lever: boosting rounds (time vs quality knee).

Training is ~26ms/round at 49k rows, so fewer rounds cuts time linearly.
The question is whether quality survives. This sweeps num_boost_round and
reports fit time + holdout MAE + ranking on the SAME leakage-safe split.

Writes experiments/artifacts/profile/rounds_sensitivity.json (gitignored).

Usage:
    python scripts/rounds_sensitivity.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from fpl.experiments.cohorts import cohort_masks, group_mae
from fpl.experiments.metrics import ranking_metrics
from fpl.experiments.splits import TemporalSplit
from fpl.model.train import CATEGORY_COLUMNS, FEATURE_COLUMNS, load_training

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = str(ROOT / "data" / "processed")
OUT = ROOT / "experiments" / "artifacts" / "profile" / "rounds_sensitivity.json"
SEASONS = ("2024-2025", "2025-2026")
ROUNDS = (25, 50, 100, 200)
BASE = {"objective": "regression", "metric": "mae", "learning_rate": 0.05,
        "num_leaves": 63, "min_child_samples": 20, "seed": 42, "verbosity": -1}


def fit_predict(X_tr, y_tr, cat_idx, X_te, *, rounds):
    import lightgbm as lgb

    ds = lgb.Dataset(X_tr, label=y_tr, feature_name=list(FEATURE_COLUMNS),
                     categorical_feature=cat_idx)
    cfg = dict(BASE)
    cfg["num_boost_round"] = rounds
    t0 = time.perf_counter()
    model = lgb.train(cfg, ds, num_boost_round=rounds)
    return time.perf_counter() - t0, model.predict(X_te)


def main() -> None:
    split = TemporalSplit(fit_gw_max=30, cal_start=31, cal_end=30,
                          test_start=31, test_end=None)
    by_season = load_training(PROCESSED, list(SEASONS))
    t24, t25 = by_season[SEASONS[0]], by_season[SEASONS[-1]]
    train_mask = t25.gw <= split.fit_gw_max
    test_mask = t25.gw >= split.test_start

    X_tr = np.vstack([t24.X, t25.X[train_mask]])
    y_tr = np.concatenate([t24.y, t25.y[train_mask]])
    X_te, y_te = t25.X[test_mask], t25.y[test_mask]
    src_te = t25.gw[test_mask]
    cat_idx = [list(FEATURE_COLUMNS).index(c) for c in CATEGORY_COLUMNS]

    players = __import__("polars").read_parquet(
        f"{PROCESSED}/players_2025-2026.parquet")
    positions = t25.meta.filter(test_mask).join(
        players.select("player_code", "position"), on="player_code", how="left"
    )["position"].to_numpy()

    rows = []
    for rounds in ROUNDS:
        fit_s, pred = fit_predict(X_tr, y_tr, cat_idx, X_te, rounds=rounds)
        mae = float(np.abs(pred - y_te).mean())
        ranking = ranking_metrics(y_te, pred, src_te)
        top10 = group_mae(y_te, pred,
                          cohort_masks(pred, src_te, positions))["top10"]
        rows.append({
            "rounds": rounds, "fit_s": round(fit_s, 3),
            "mae": round(mae, 4), "top10_mae": round(top10, 4),
            "spearman": round(ranking["spearman_rho"], 4),
            "rank_topk": round(ranking["topk_hit_rate"], 4),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    print("boosting rounds sensitivity")
    print("=" * 72)
    print(f"{'rounds':>6} {'fit_s':>7} {'mae':>7} {'top10_mae':>10} "
          f"{'spearman':>9} {'rank_topk':>10}")
    for r in rows:
        print(f"{r['rounds']:>6} {r['fit_s']:>7.2f} {r['mae']:>7.3f} "
              f"{r['top10_mae']:>10.3f} {r['spearman']:>9.3f} "
              f"{r['rank_topk']:>10.3f}")
    best = min(rows, key=lambda r: (r["mae"] - min(x["mae"] for x in rows))**2
               + (r["fit_s"] / 100))
    print("-" * 72)
    print(f"knee candidate: {best['rounds']} rounds, fit {best['fit_s']:.2f}s, "
          f"MAE {best['mae']:.3f}")
    print(f"wrote -> {OUT}")


if __name__ == "__main__":
    main()