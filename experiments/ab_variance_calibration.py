"""A/B: variance calibration vs raw IID squad sampling, gym-arbitrated.

Experiments 2-3 showed squad-week uncertainty is dominated by VARIANCE SCALE
(hold-out squad-total errors far exceed in-sample / empirical variance). This
tests the simplest fix: a single scalar calibration factor k fitted on a
VALIDATION window (gw 28..30), never the arbiter window (gw 31..38):

    calibrated std = k * sampled std

Arms (same marginals, same mean — only the variance handling differs):
  raw        : z = (actual - mean) / std_sampled
  calibrated : z = (actual - mean) / (k * std_sampled)

Calibrated variance => std(z) and mean|z| move toward 1.

Leakage: marginal distributions from gw <= 27; k from gw 28..30; artificial
arbitration (the gym window) is gw 31..38. Gate + seed first.

Run:  python experiments/ab_variance_calibration.py
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl

from fpl.model.inference import load_model
from fpl.model.leakage import validate
from fpl.pipeline import run_basket

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = str(ROOT / "data" / "processed")
SEASON = "2025-2026"
MARG_SRC_MAX = 27           # marginals from gw <= 27 (train)
VAL_GWS = (28, 30)          # calibration window (k)
TEST_GWS = (31, 38)         # the arbiter (gym window)
N_DRAWS = 200
SEED = 42


def main() -> None:
    validate(
        pl.read_parquet(f"{PROCESSED}/features_{SEASON}.parquet"),
        pl.read_parquet(f"{PROCESSED}/players_{SEASON}.parquet"),
        gw_train_max=VAL_GWS[1], gw_test_min=TEST_GWS[0])
    print("leakage validation: PASS")

    players = pl.read_parquet(f"{PROCESSED}/players_{SEASON}.parquet")
    gw_stats = pl.read_parquet(f"{PROCESSED}/gw_stats_{SEASON}.parquet")

    # per-player marginal point distributions from gw <= MARG_SRC_MAX
    train = (gw_stats.filter(pl.col("gw") <= MARG_SRC_MAX)
             .join(players.select("player_id", "player_code"), on="player_id"))
    marg: dict[int, np.ndarray] = {}
    for (code,), g in train.group_by("player_code"):
        marg[int(code)] = g["total_points"].to_numpy().astype(float)
    rng = np.random.default_rng(SEED)

    model = load_model(f"{PROCESSED}/points_lgbm.txt")
    res = run_basket(processed=PROCESSED, season=SEASON, gw_start=TEST_GWS[0],
                     gw_end=TEST_GWS[1], model=model, freshness=False,
                     enum_kw={"n_teams": 4, "seed": 1})
    squads = res.squads[:4]

    def eval_week(squad, gw):
        rows = (gw_stats.filter(pl.col("gw") == gw)
                .join(players.select("player_id", "player_code"), on="player_id")
                .filter(pl.col("player_code").is_in(squad.codes())))
        played = {int(r["player_code"]): (r["minutes"] or 0) > 0
                  for r in rows.select("player_code", "minutes").iter_rows(named=True)}
        points = {int(r["player_code"]): float(r["total_points"] or 0.0)
                  for r in rows.select("player_code", "total_points").iter_rows(named=True)}
        settle = squad.gw_settlement(played, points)
        S = list(settle.playing)
        if len(S) < 4:
            return None
        tot = np.zeros(N_DRAWS)
        for d in range(N_DRAWS):
            tot[d] = sum(
                marg[c][rng.integers(0, len(marg[c]))] if c in marg else 0.0
                for c in S)
        return (settle.gw_total, float(tot.mean()), float(tot.std()))

    def collect(gw_range):
        out = []
        for gw in range(*gw_range):
            for squad in squads:
                r = eval_week(replace(squad, gw=gw), gw)
                if r is not None:
                    out.append(r)
        return out  # list of (actual, mean, std)

    val = collect(VAL_GWS)
    # calibration factor: sqrt( mean[(actual-mean)^2] / mean[var_sampled] )
    k = float(np.sqrt(
        np.mean([(a - m) ** 2 for a, m, s in val])
        / (0.0 if not val else np.mean([s ** 2 for a, m, s in val]))))

    test = collect(TEST_GWS)
    z_raw = np.array([(a - m) / s for a, m, s in test])
    z_cal = np.array([(a - m) / (k * s) for a, m, s in test])

    print(f"\nvalidation {VAL_GWS[0]}..{VAL_GWS[1]}: {len(val)} squad-weeks -> "
          f"calibration factor k = {k:.3f}")
    print("gym arbitration (test gw 31..38):")
    print("=" * 72)
    print(f"  raw        : mean|z| = {np.abs(z_raw).mean():.3f}  std(z) = {z_raw.std():.3f}")
    print(f"  calibrated : mean|z| = {np.abs(z_cal).mean():.3f}  std(z) = {z_cal.std():.3f}")


if __name__ == "__main__":
    main()