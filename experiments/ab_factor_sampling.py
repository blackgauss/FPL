"""A/B: IID marginal sampling vs learned team/fixture factor sampling.

Uses the existing dist_2025-2026.parquet marginal quantiles. The factor arm
preserves those marginals, but adds shared Gaussian shocks:

  z_i = sqrt(rho_team) * team_shock
      + sqrt(rho_fixture) * side_i * fixture_shock
      + sqrt(1-rho_team-rho_fixture) * idio_i

The rhos are learned from TRAIN residual covariance only. IID is the same
sampler with both rhos set to zero. The gym settles actual points using real
minutes, bench priorities, and captain rules; the comparison measures squad
distribution calibration.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.special import ndtr

from fpl.dist import QS
from fpl.gym import Eval
from fpl.model.inference import load_model
from fpl.model.leakage import validate
from fpl.model.train import CATEGORY_COLUMNS, FEATURE_COLUMNS, load_training
from fpl.pipeline import run_basket

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = str(ROOT / "data" / "processed")
SEASON = "2025-2026"
TRAIN_MAX = 29
TEST = (31, 33)
N_DRAWS = 300
SEED = 42


def learn_rhos() -> tuple[float, float]:
    """Estimate shared residual covariance from pre-holdout rows only."""
    by = load_training(PROCESSED, ["2024-2025", SEASON],
                       feature_columns=FEATURE_COLUMNS,
                       categorical_columns=CATEGORY_COLUMNS)
    t24, t25 = by["2024-2025"], by[SEASON]
    keep = t25.gw <= TRAIN_MAX
    x = np.vstack([t24.X, t25.X[keep]])
    y = np.concatenate([t24.y, t25.y[keep]])
    ds = lgb.Dataset(x, label=y, feature_name=t25.feature_names,
                     categorical_feature=t25.categorical)
    model = lgb.train({"objective": "regression", "learning_rate": .05,
                       "num_leaves": 63, "min_child_samples": 20,
                       "seed": SEED, "verbosity": -1}, ds, num_boost_round=200)
    residual = t25.y[keep] - model.predict(t25.X[keep])
    features = pl.read_parquet(f"{PROCESSED}/features_{SEASON}.parquet")
    rf = (features.filter(pl.col("gw") <= TRAIN_MAX)
          .select("player_code", "gw", "team_code", "opponent_team_code")
          .with_columns(pl.Series("resid", residual)).drop_nulls())
    sigma2 = float(rf["resid"].var())
    same = (rf.join(rf, on=["gw", "team_code"])
            .filter(pl.col("player_code") < pl.col("player_code_right"))
            .select((pl.col("resid") * pl.col("resid_right")).alias("x")))
    opp = (rf.join(rf, on=["gw"])
           .filter((pl.col("opponent_team_code") == pl.col("team_code_right")) &
                   (pl.col("team_code") == pl.col("opponent_team_code_right")) &
                   (pl.col("player_code") < pl.col("player_code_right")))
           .select((pl.col("resid") * pl.col("resid_right")).alias("x")))
    alpha = float(same["x"].mean())
    beta = float(opp["x"].mean())
    return float(np.clip(alpha / sigma2, 0, .25)), float(np.clip(beta / sigma2, 0, .25))


def sample_totals(q, teams, fixtures, sides, *, rho_team, rho_fixture,
                  rng):
    """Sample squad totals; q is (players, quantiles), preserving marginals."""
    n_players = q.shape[0]
    team_ids, team_ix = np.unique(teams, return_inverse=True)
    fix_ids, fix_ix = np.unique(fixtures, return_inverse=True)
    team_z = rng.standard_normal((N_DRAWS, len(team_ids)))
    fix_z = rng.standard_normal((N_DRAWS, len(fix_ids)))
    idio = rng.standard_normal((N_DRAWS, n_players))
    residual = max(1.0 - rho_team - rho_fixture, 0.0) ** .5
    z = (rho_team ** .5) * team_z[:, team_ix]
    z += (rho_fixture ** .5) * fix_z[:, fix_ix] * sides
    z += residual * idio
    u = ndtr(z)
    draws = np.empty_like(u)
    qs = np.asarray(QS)
    for i in range(n_players):
        draws[:, i] = np.interp(u[:, i], qs, q[i])
    return draws.sum(axis=1), draws


def main() -> None:
    validate(pl.read_parquet(f"{PROCESSED}/features_{SEASON}.parquet"),
             pl.read_parquet(f"{PROCESSED}/players_{SEASON}.parquet"),
             gw_train_max=TRAIN_MAX, gw_test_min=TEST[0])
    print("leakage validation: PASS")
    rho_team, rho_fixture = learn_rhos()
    print(f"learned rhos: team={rho_team:.4f}, fixture={rho_fixture:.4f}")

    players = pl.read_parquet(f"{PROCESSED}/players_{SEASON}.parquet")
    gw_stats = pl.read_parquet(f"{PROCESSED}/gw_stats_{SEASON}.parquet")
    dist = pl.read_parquet(f"{PROCESSED}/dist_{SEASON}.parquet")
    features = pl.read_parquet(f"{PROCESSED}/features_{SEASON}.parquet")
    model = load_model(f"{PROCESSED}/points_lgbm.txt")
    res = run_basket(processed=PROCESSED, season=SEASON, gw_start=TEST[0],
                     gw_end=TEST[1], model=model, freshness=False,
                     enum_kw={"n_teams": 4, "seed": 1})
    context = (features.with_columns((pl.col("gw") + 1).alias("target_gw"))
               .select("player_code", "target_gw", "team_code",
                       "opponent_team_code", "was_home"))
    d = dist.join(context, left_on=["player_code", "gw"],
                  right_on=["player_code", "target_gw"], how="left")
    qcols = [f"q{int(q * 100):02d}" for q in QS]
    d = d.with_columns(pl.col("quantiles_struct").struct.rename_fields(qcols)
                       .alias("q"))

    rng = np.random.default_rng(SEED)
    z = {"iid": [], "factor": []}
    for squad in res.squads[:4]:
        s = replace(squad, gw=TEST[0])
        for gw in range(TEST[0], TEST[1] + 1):
            # Use the actual settled XI for apples-to-apples distribution grading.
            ev = Eval(s, gw_stats=gw_stats, players=players, weeks=1,
                      forecast=d.select(
                          "player_code", "gw",
                          pl.col("pred").alias("expected_points")),
                      name="factor-ab").run().weeks[0]
            codes = list(ev.xi)
            rows = d.filter((pl.col("gw") == gw) &
                            pl.col("player_code").is_in(codes))
            if rows.height != len(codes):
                continue
            rows = rows.sort("player_code")
            q = np.asarray([[r["q"][name] for name in qcols]
                            for r in rows.iter_rows(named=True)], dtype=float)
            teams = np.asarray(rows["team_code"].to_list())
            opp = rows["opponent_team_code"].to_list()
            fixtures = np.asarray([tuple(sorted((int(t), int(o))))
                                   if o is not None else (int(t), -1)
                                   for t, o in zip(teams, opp, strict=False)])
            _, fix_ix = np.unique(fixtures, axis=0, return_inverse=True)
            sides = np.asarray([1.0 if bool(x) else -1.0
                                for x in rows["was_home"].fill_null(False)])
            iid, _ = sample_totals(q, teams, fix_ix, sides,
                                   rho_team=0, rho_fixture=0, rng=rng)
            factor, _ = sample_totals(q, teams, fix_ix, sides,
                                      rho_team=rho_team,
                                      rho_fixture=rho_fixture, rng=rng)
            z["iid"].append((ev.actual_points - iid.mean()) / iid.std())
            z["factor"].append((ev.actual_points - factor.mean()) / factor.std())

    print(f"gym A/B: {len(z['iid'])} squad-weeks, {N_DRAWS} draws")
    for name in ("iid", "factor"):
        values = np.asarray(z[name])
        print(f"  {name:<8} mean|z|={np.abs(values).mean():.3f} "
              f"std(z)={values.std():.3f}")


if __name__ == "__main__":
    main()
