"""A/B: match-state factor model vs IID, gym-arbitrated.

Theoretical ideal is P(X_1..X_22) per match, but that is 22-dim with <2000
matches -> singular/noise (see analysis/Investigations.md). Factor cheat:
conditional independence given the MATCH STATE

    P(state) x prod_i P(X_i | state)

where state = (own goals, conceded) per team in the match. Teammates share
their team's goals/conceded; opponents share the joint outcome; the covariance
is induced structurally, no correlation matrix. Sampler is model-free: state
distribution and P(X | state) come from TRAIN gameweeks only (gw <= 29); the
gym arbitrates on holdout squads-week actuals (gw 31..38).

Two arms, same marginals, differ only in coupling:
  IID      : sample each player's marginal points independently.
  factor   : sample a match state, then each player conditional on it.

Metrics on 200 draws per squad-week: z = (actual - mean)/std (calibrated => 1,
IID is overconfident), plus: does the factor sampler REPRODUCE the measured
same-team correlation (~0.05) while IID gives ~0?

Velocity: model-free (no lgb fit), and the conditional tables are cached to
experiments/artifacts/ so re-runs skip recomputation.

Run:  python experiments/ab_match_state_factor.py
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
ARTIFACTS = ROOT / "experiments" / "artifacts"
SEASON = "2025-2026"
TRAIN_SRC_MAX = 29
EVAL_GWS = (31, 38)
N_DRAWS = 200
SEED = 42


def _load_train_tables() -> tuple[np.ndarray, dict, dict]:
    """state samples + conditional-per-position and marginal-per-player tables.

    Cached to artifacts (velocity): built from gw<=TRAIN_SRC_MAX matches +
    gw_stats, so nothing from the holdout is used.
    """
    cache = ARTIFACTS / f"factor_tables_{SEASON}_gw{EVAL_GWS[0]}to{EVAL_GWS[1]}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return (np.asarray(z["states"]), z["cond"].item(), z["marg"].item())

    players = pl.read_parquet(f"{PROCESSED}/players_{SEASON}.parquet")
    gw = pl.read_parquet(f"{PROCESSED}/gw_stats_{SEASON}.parquet")
    matches = pl.read_parquet(f"{PROCESSED}/matches_{SEASON}.parquet")

    # match states: (team goals, conceded) pairs from every finished train match
    states = np.asarray(
        matches.filter((pl.col("gw") <= TRAIN_SRC_MAX) & (pl.col("finished")))
        .select("home_score", "away_score").drop_nulls().to_numpy(), dtype=int)

    # per (player, gw, team) goals/conceded via the match the team played
    long = []
    for role, goal_col, opp_col in [("home", "home_score", "away_score"),
                                    ("away", "away_score", "home_score")]:
        m = matches.filter(pl.col("gw") <= TRAIN_SRC_MAX) \
            .select("gw", pl.col(role + "_team").alias("team_code"),
                    pl.col(goal_col).alias("goals"),
                    pl.col(opp_col).alias("conceded"))
        long.append(m)
    state_by_team_gw = pl.concat(long).drop_nulls()

    rows = (gw.filter(pl.col("gw") <= TRAIN_SRC_MAX)
            .join(players.select("player_id", "player_code", "position",
                                 "team_code"), on="player_id")
            .join(state_by_team_gw, on=["team_code", "gw"], how="left"))

    def own_bin(g): return min(int(g), 4)
    def conc_bin(c): return min(int(c), 3)

    rows = rows.with_columns(
        pl.col("goals").map_elements(own_bin, return_dtype=pl.Int8).alias("ob"),
        pl.col("conceded").map_elements(conc_bin, return_dtype=pl.Int8).alias("cb"),
    ).drop_nulls(subset=["ob", "cb"])

    cond: dict = {}
    for (pos, ob, cb), g in rows.group_by(["position", "ob", "cb"]):
        cond[(pos, ob, cb)] = g["total_points"].to_numpy().astype(float)
    marg: dict = {}
    for (code,), g in rows.group_by("player_code"):
        marg[int(code)] = g["total_points"].to_numpy().astype(float)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    np.savez(cache, states=states, cond=cond, marg=marg)
    print(f"cached factor tables -> {cache}")
    return states, cond, marg


def main() -> None:
    validate(
        pl.read_parquet(f"{PROCESSED}/features_{SEASON}.parquet"),
        pl.read_parquet(f"{PROCESSED}/players_{SEASON}.parquet"),
        gw_train_max=TRAIN_SRC_MAX, gw_test_min=EVAL_GWS[0])
    print("leakage validation: PASS")

    states, cond, marg = _load_train_tables()
    players = pl.read_parquet(f"{PROCESSED}/players_{SEASON}.parquet")
    gw_stats = pl.read_parquet(f"{PROCESSED}/gw_stats_{SEASON}.parquet")
    matches = pl.read_parquet(f"{PROCESSED}/matches_{SEASON}.parquet")
    team_of = {int(r["player_code"]): int(r["team_code"])
               for r in players.select("player_code", "team_code").iter_rows(named=True)}
    pos_of = {int(r["player_code"]): r["position"]
              for r in players.select("player_code", "position").iter_rows(named=True)}

    # candidate squads via the stored model (no training this run)
    model = load_model(f"{PROCESSED}/points_lgbm.txt")
    res = run_basket(processed=PROCESSED, season=SEASON, gw_start=EVAL_GWS[0],
                     gw_end=EVAL_GWS[1], model=model, freshness=False,
                     enum_kw={"n_teams": 4, "seed": 1})

    rng = np.random.default_rng(SEED)

    def sample_state():
        s = states[rng.integers(0, len(states))]
        return (int(s[0]), int(s[1]))

    def paired(state):
        # unordered (goals, conceded) for the two teams: both assignments equal p
        g, c = state
        return (g, c) if rng.random() < 0.5 else (c, g)

    z_iid, z_fac = [], []
    corr_same = {"iid": [], "fac": []}

    for squad in res.squads[:4]:
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
            S = list(settle.playing)
            if not S or len(S) < 4:
                continue
            actual = settle.gw_total

            # each distinct (home_team, away_team) fixture gets its own state
            fam = {tuple(sorted(r)) for r in matches.filter(pl.col("gw") == gw)
                   .select("home_team", "away_team").drop_nulls().iter_rows()}
            fix_state = {}
            for pair in fam:
                g1, g2 = paired(sample_state())
                fix_state[pair] = (g1, g2)

            def goals_for(code, fam=fam, fix_state=fix_state):
                t = team_of[code]
                opp = next((pair for pair in fam if t in pair), None)
                g, c = fix_state[opp] if opp else (0, 0)
                return (g, c) if opp[0] == t else (c, g)

            tot_iid, tot_fac = np.zeros(N_DRAWS), np.zeros(N_DRAWS)
            seller_one = seller_two = None
            # find a same-team pair among S for the correlation check
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    if team_of.get(S[i]) == team_of.get(S[j]) and \
                            team_of.get(S[i]) is not None:
                        seller_one, seller_two = S[i], S[j]
                        break
                if seller_one is not None:
                    break

            sj_pair = {"iid": ([], []), "fac": ([], [])}
            for d in range(N_DRAWS):
                # a FRESH match state per draw: the shared factor varies across
                # draws, which is what induces teammate/opponent covariance
                fix_state = {}
                for pair in fam:
                    g1, g2 = paired(sample_state())
                    fix_state[pair] = (g1, g2)
                t_i = np.zeros(len(S))
                t_f = np.zeros(len(S))
                for k, code in enumerate(S):
                    m = marg.get(code)
                    mpts = m[rng.integers(0, len(m))] if (m is not None and len(m)) else 0.0
                    t_i[k] = mpts
                    g, c = goals_for(code)
                    cell = cond.get((pos_of[code], g, c))
                    fpts = cell[rng.integers(0, len(cell))] \
                        if (cell is not None and len(cell)) else mpts
                    t_f[k] = fpts
                tot_iid[d] = t_i.sum()
                tot_fac[d] = t_f.sum()
                if seller_one is not None:
                    i1, i2 = S.index(seller_one), S.index(seller_two)
                    sj_pair["iid"][0].append(t_i[i1])
                    sj_pair["fac"][0].append(t_f[i1])
                    sj_pair["iid"][1].append(t_i[i2])
                    sj_pair["fac"][1].append(t_f[i2])

            if tot_iid.std() > 0 and tot_fac.std() > 0:
                z_iid.append((actual - tot_iid.mean()) / tot_iid.std())
                z_fac.append((actual - tot_fac.mean()) / tot_fac.std())
            if seller_one is not None and len(sj_pair["fac"][0]) > 40:
                for arm, (a, b) in sj_pair.items():
                    c = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else 0.0
                    corr_same[arm].append(c)

    z_iid, z_fac = np.array(z_iid), np.array(z_fac)
    print(f"\ngym arbitration: {len(z_iid)} squad-week actuals, "
          f"{N_DRAWS} draws each")
    print("=" * 72)
    for arm, z in [("IID", z_iid), ("factor", z_fac)]:
        print(f"  {arm:<9}  mean|z| = {np.abs(z).mean():.3f}  std(z) = {z.std():.3f}")
    print("\nreproduced same-team correlation (simulated teammate totals):")
    for arm, cs in corr_same.items():
        print(f"  {arm:<9} mean r = {np.mean(cs):.4f} over {len(cs)} pairs "
              f"(data says ~0.054)")
    print("  (IID ~ 0; factor should approach the real same-team co-movement)")


if __name__ == "__main__":
    main()