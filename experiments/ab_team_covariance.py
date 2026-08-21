"""A/B: is a team-flag covariance model useful for squad evaluation?

The mean of a squad's points depends only on per-player expectations, but its
*variance* depends on how teammates / same-fixture opponents co-move. Baseline
squad evaluation treats players as I.I.D. (var = sum of per-player variances);
the covariance-aware version adds the learned structure:

    var(squad) = sum_i sigma_i^2
               + 2*alpha * (# same-team pairs in the scoring XI)
               - 2*beta  * (# shared-fixture opponent pairs in the XI)

alpha (teammates co-move, +) and beta (opponents in the same match co-move, -)
are estimated from TRAINING residuals only (gw <= 29). The gym arbitrates on
holdout gameweeks (31..38): for each candidate squad-week, actual settled
points vs the predicted distribution; a well-calibrated variance gives
z = (actual - mean)/std with mean|z| and std(z) close to what the model 
claims. IID ignores within-squad correlation -> overconfident -> |z| too large.

Leakage: the gate runs first (train gw_max 29 vs test gw_min 31); nothing
from the holdout touches any fit.

Run:  python experiments/ab_team_covariance.py
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from fpl.model.leakage import validate
from fpl.model.train import CATEGORY_COLUMNS, FEATURE_COLUMNS, load_training
from fpl.pipeline import run_basket
from fpl.team.scoring import score_players

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = str(ROOT / "data" / "processed")
SEASONS = ["2024-2025", "2025-2026"]
SEASON = SEASONS[1]
TRAIN_SRC_MAX = 29          # rows gw <= 29 (targets <= 30)
EVAL_GWS = (31, 38)         # holdout targets
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


def main() -> None:
    validate(
        pl.read_parquet(f"{PROCESSED}/features_{SEASON}.parquet"),
        pl.read_parquet(f"{PROCESSED}/players_{SEASON}.parquet"),
        gw_train_max=TRAIN_SRC_MAX, gw_test_min=EVAL_GWS[0])
    print("leakage validation: PASS")

    players = pl.read_parquet(f"{PROCESSED}/players_{SEASON}.parquet")
    gw_stats = pl.read_parquet(f"{PROCESSED}/gw_stats_{SEASON}.parquet")
    matches = pl.read_parquet(f"{PROCESSED}/matches_{SEASON}.parquet")

    # --- train baseline model on train rows only (both seasons; 25-26 gw<=29) ---
    by_season = load_training(PROCESSED, SEASONS, feature_columns=FEATURE_COLUMNS,
                              categorical_columns=CATEGORY_COLUMNS)
    t24, td = by_season[SEASONS[0]], by_season[SEASON]
    keep = td.gw <= TRAIN_SRC_MAX
    X = np.vstack([t24.X, td.X[keep]])
    y = np.concatenate([t24.y, td.y[keep]])
    ds = lgb.Dataset(X, label=y,
                     feature_name=td.feature_names, categorical_feature=td.categorical)
    model = lgb.train(PARAMS, ds, num_boost_round=200)

    # --- covariance from TRAIN residuals (actual - predict) on 25-26 train rows ---
    resid = td.y[keep] - model.predict(td.X[keep])
    feat_train = pl.read_parquet(f"{PROCESSED}/features_{SEASON}.parquet").select(
        "player_code", "gw", "team_code", "opponent_team_code")
    rf = (feat_train.filter(pl.col("gw") <= TRAIN_SRC_MAX)
          .with_columns(pl.Series("resid", resid))
          .drop_nulls(subset=["opponent_team_code"]))

    # per-player residual variance (fallback = pooled median)
    sig2_by = rf.group_by("player_code").agg(pl.col("resid").var().alias("sig2"))
    pooled = float(rf["resid"].var())
    sd2i: dict[int, float] = {}
    for r in sig2_by.iter_rows(named=True):
        v = r["sig2"]
        sd2i[int(r["player_code"])] = float(v) if v is not None else pooled

    def sig2(code):
        return sd2i.get(int(code), pooled)

    # alpha: pooled same-team pairwise residual product (players i,j, same gw, same team)
    pair_products = rf.join(rf, on=["gw", "team_code"]).filter(
        pl.col("player_code") < pl.col("player_code_right")) \
        .with_columns((pl.col("resid") * pl.col("resid_right")).alias("prod"))
    alpha = float(pair_products["prod"].mean())

    # beta: shared-fixture opponent pairs (players in teams that faced each other that gw)
    opp_pairs = rf.join(rf, on=["gw"]).filter(
        (pl.col("opponent_team_code") == pl.col("team_code_right")) &
        (pl.col("team_code") == pl.col("opponent_team_code_right")) &
        (pl.col("player_code") < pl.col("player_code_right"))).with_columns(
        (pl.col("resid") * pl.col("resid_right")).alias("prod"))
    beta = float(opp_pairs["prod"].mean())

    print(f"learned residual structure (train, {rf.height} rows):")
    print(f"  pooled player variance sigma^2 = {pooled:.4f}")
    print(f"  alpha (same-team residual cov) = {alpha:.4f}")
    print(f"  beta  (shared-fixture opp cov) = {beta:.4f}")

    # --- candidate squads (baseline basket, fixed across both arms) ---
    res = run_basket(processed=PROCESSED, season=SEASON, gw_start=EVAL_GWS[0],
                     gw_end=EVAL_GWS[1], model=model, freshness=False,
                     enum_kw={"n_teams": 6, "seed": 1})
    _, per_gw = score_players(td, model, gw_start=EVAL_GWS[0], gw_end=EVAL_GWS[1],
                              players=players, detail=True)
    expected_by = {(int(r["player_code"]), int(r["gw"])): float(r["expected_points"])
                   for r in (per_gw.select("player_code", "gw", "expected_points")
                             .iter_rows(named=True))}

    # helper: same-team pairs and shared-fixture opponent pairs within a player set at a gw
    def team_of():
        return {int(r["player_code"]): int(r["team_code"])
                for r in players.select("player_code", "team_code").iter_rows(named=True)}

    team_code = team_of()

    def counts_in(set_codes, gw):
        codes = list(set_codes)
        same = 0
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                if team_code.get(codes[i]) == team_code.get(codes[j]):
                    same += 1
        opp = 0
        fam = set(matches.filter(pl.col("gw") == gw).select("home_team", "away_team")
                  .iter_rows())
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                ti, tj = team_code.get(codes[i]), team_code.get(codes[j])
                if (ti, tj) in fam or (tj, ti) in fam:
                    opp += 1
        return same, opp

    z_iid, z_cov = [], []
    for squad in res.squads[:6]:
        s = replace(squad, gw=EVAL_GWS[0])
        for gw in range(EVAL_GWS[0], EVAL_GWS[1] + 1):
            rows = (gw_stats.filter(pl.col("gw") == gw)
                    .join(players.select("player_id", "player_code"), on="player_id")
                    .filter(pl.col("player_code").is_in(s.codes())))
            played = {int(r["player_code"]): (r["minutes"] or 0) > 0
                      for r in rows.select("player_code", "minutes").iter_rows(named=True)}
            points = {int(r["player_code"]): float(r["total_points"] or 0.0)
                      for r in rows.select("player_code", "total_points").iter_rows(named=True)}
            settle = s.gw_settlement(played, points)
            S = settle.playing
            if not S:
                continue
            mean = sum(expected_by.get((c, gw), 0.0) * (2 if c == settle.captain_doubled else 1)
                       for c in S)
            actual = settle.gw_total
            var1 = sum(sig2(c) for c in S)
            same, opp = counts_in(S, gw)
            var2 = var1 + 2 * alpha * same - 2 * beta * opp
            if var1 <= 1e-9 or var2 <= 1e-9:
                continue
            z_iid.append((actual - mean) / np.sqrt(var1))
            z_cov.append((actual - mean) / np.sqrt(var2))

    z_iid, z_cov = np.array(z_iid), np.array(z_cov)
    print(f"\ngym arbitration: {len(z_iid)} squad-week actuals, "
          f"holdout GWs {EVAL_GWS[0]}..{EVAL_GWS[1]}")
    print("=" * 72)
    print(f"  IID (baseline)        : mean|z| = {np.abs(z_iid).mean():.3f}  "
          f"std(z) = {z_iid.std():.3f}")
    print(f"  covariance-aware       : mean|z| = {np.abs(z_cov).mean():.3f}  "
          f"std(z) = {z_cov.std():.3f}")
    print("  (well-calibrated variance => std(z) near 1; IID overestimates the "
          "misses if it is > 1)")


if __name__ == "__main__":
    main()