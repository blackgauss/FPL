"""A/B: baseline vs + matchup featurization, graded by the gym.

The simplest covariance/matchup indication (per analysis/Investigations.md):
tell the model which team it faces. Everything else is held fixed — same
data split, same LightGBM params/seed, same candidate squads, same actuals.
Only the feature set differs:

    A (baseline): FEATURE_COLUMNS
    B (matchup) : FEATURE_COLUMNS + opponent_team_code (categorical)

Verdict is two-sided: holdout MAE (out-of-sample) AND the gym's
forecast-vs-actual gap on real gameweeks — a paired toggle, not one number.

Run:  python experiments/ab_matchup_model.py
"""

from __future__ import annotations

from dataclasses import replace

import lightgbm as lgb
import numpy as np
import polars as pl

from fpl.gym import Eval
from fpl.model.eval import summarize
from fpl.model.train import CATEGORY_COLUMNS, FEATURE_COLUMNS, load_training
from fpl.pipeline import run_basket
from fpl.team.scoring import score_players

PROCESSED = "data/processed"
SEASONS = ["2024-2025", "2025-2026"]
TRAIN_SRC_MAX = 29          # source rows <= 29 (next target <= 30)
EVAL_SOURCES = (30, 37)     # source rows -> targets 31..38 (the holdout)
SEED = 42

PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "num_boost_round": 200,
    "seed": SEED,
    "verbosity": -1,
}


def fit(feature_columns: list[str], categorical_columns: list[str]):
    by_season = load_training(PROCESSED, SEASONS, feature_columns=feature_columns,
                              categorical_columns=categorical_columns)
    t24, t25 = by_season[SEASONS[0]], by_season[SEASONS[1]]
    keep = t25.gw <= TRAIN_SRC_MAX
    X = np.vstack([t24.X, t25.X[keep]])
    y = np.concatenate([t24.y, t25.y[keep]])
    ds = lgb.Dataset(X, label=y, feature_name=t25.feature_names,
                     categorical_feature=t25.categorical)
    model = lgb.train(PARAMS, ds, num_boost_round=200)
    return model, t25


def main() -> None:
    players = pl.read_parquet(f"{PROCESSED}/players_2025-2026.parquet")
    gw_stats = pl.read_parquet(f"{PROCESSED}/gw_stats_2025-2026.parquet")

    cfgs = {
        "A_baseline": (list(FEATURE_COLUMNS), list(CATEGORY_COLUMNS)),
        "B_matchup": (list(FEATURE_COLUMNS) + ["opponent_team_code"],
                      list(CATEGORY_COLUMNS) + ["opponent_team_code"]),
    }

    models, tds = {}, {}
    for name, (feats, cats) in cfgs.items():
        models[name], tds[name] = fit(feats, cats)

    # --- holdout MAE (out-of-sample, same rows for both) ---
    mask = (tds["A_baseline"].gw >= EVAL_SOURCES[0]) & \
           (tds["A_baseline"].gw <= EVAL_SOURCES[1])
    y_test = tds["A_baseline"].y[mask]
    print(f"\nholdout (targets GW {EVAL_SOURCES[0]+1}..{EVAL_SOURCES[1]+1}, "
          f"{y_test.size} rows)")
    print("=" * 72)
    for name, td in tds.items():
        summarize(pl.Series("pred", models[name].predict(td.X[mask])),
                  pl.Series("actual", y_test), name)
        _ = td  # keep ref

    # --- gym A/B: same squads, same actuals, only the forecast changes ---
    base_model = models["A_baseline"]
    res = run_basket(
        processed=PROCESSED, season=SEASONS[1], gw_start=31, gw_end=38,
        model=base_model, freshness=False,
        enum_kw={"n_teams": 2, "seed": 1})
    forecasts = {}
    for name, model in models.items():
        _, per_gw = score_players(tds[name], model, gw_start=31, gw_end=38,
                                  players=players, detail=True)
        forecasts[name] = per_gw.select("player_code", "gw", "expected_points")

    forecastable = sorted(forecasts["A_baseline"]["gw"].unique().to_list())
    start, weeks = forecastable[0], len(forecastable)
    print(f"\ngym: candidate squads {len(res.squads)} | forecastable GWs "
          f"{start}..{start + weeks - 1} (fixed squad, fixed actuals)")
    print("=" * 72)

    for i, squad in enumerate(res.squads[:2]):
        s = replace(squad, gw=start)
        evals = {name: Eval(s, gw_stats=gw_stats, players=players, weeks=weeks,
                            forecast=forecasts[name], name=f"{name}-{squad.players[0].code}").run()
                 for name in cfgs}
        for name in cfgs:
            e = evals[name]
            print(f"  squad{i} {name:<14} actual {e.total_actual:6.1f}  "
                  f"predicted {e.total_predicted:6.1f}  gap {e.gap:+.1f}")
        gap_a = evals["A_baseline"].gap
        gap_b = evals["B_matchup"].gap
        if gap_a is not None and gap_b is not None:
            winner = "B_matchup" if abs(gap_b) < abs(gap_a) else "A_baseline"
            print(f"      -> |gap| {abs(gap_a):.1f} vs {abs(gap_b):.1f} "
                  f"(winner: {winner})")

    # which matchup feature actually mattered?
    print("\nfeature importances (gain, B only):")
    for fname, imp in sorted(
        zip(tds["B_matchup"].feature_names,
            models["B_matchup"].feature_importance("gain"), strict=False),
        key=lambda t: -t[1],
    )[:10]:
        print(f"  {fname:<18} {imp:9.1f}")


if __name__ == "__main__":
    main()