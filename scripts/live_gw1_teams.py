"""GW1 candidate teams: 2026-27 pre-season selection with live reconcile.

The 2026-27 feature store's GW1 rows describe the current (pre-season) world:
carryover form, live prices, ep_next, status. Scoring those rows directly
gives each player's expected points for the upcoming gameweek (GW1). Then the
live snapshot reconciles clubs/injuries into the input BEFORE construction
(fpl.live.current), so `max_per_club`, the pool, and enumeration all see the
current world — and a per-squad live-flag column shows any player who is
injured / missing from the live roster / transferred / price-moved.

Usage:
    python scripts/live_gw1_teams.py [--cache data/raw/fpl_api/live.json]
"""

from __future__ import annotations

import argparse

import polars as pl

from fpl.live.current import reconcile_player_clubs
from fpl.live.filters import flag_squad_player, suggest
from fpl.live.live import load_live_state
from fpl.model.inference import load_model
from fpl.model.train import load_training
from fpl.team.enumerate import greedy_teams
from fpl.team.filtering import filter_pool


def main() -> None:
    ap = argparse.ArgumentParser(description="2026-27 GW1 candidate teams, live-reconciled")
    ap.add_argument("--season", default="2026-2027")
    ap.add_argument("--gw", type=int, default=1)
    ap.add_argument("--cache", default="data/raw/fpl_api/live.json")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--n-teams", type=int, default=20)
    args = ap.parse_args()

    live, fetched = load_live_state(args.cache, max_age_seconds=3600)
    players = pl.read_parquet(f"{args.processed}/players_{args.season}.parquet")
    gw_stats = pl.read_parquet(f"{args.processed}/gw_stats_{args.season}.parquet")
    model = load_model(f"{args.processed}/points_lgbm.txt")

    # GW1 rows: features are the pre-season/current world -> expected GW1 points
    td = load_training(args.processed, [args.season],
                       require_target=False)[args.season]
    mask = td.gw == args.gw
    scored = td.meta.filter(pl.col("gw") == args.gw).with_columns(
        pl.Series("expected_total", model.predict(td.X[mask]))
    )
    scored = scored.join(players.select("player_id", "player_code", "web_name",
                                        "position"),
                         on="player_id", how="left")
    # availability frame for filter_pool (price from gw_stats)
    avail = gw_stats.join(players.select("player_id", "player_code"),
                          on="player_id", how="left").select(
        "player_code", "now_cost").with_columns(
        pl.col("now_cost").cast(pl.Float64),
        pl.lit(1).alias("minutes_in_window"))
    scored = scored.with_columns(pl.lit(1).alias("minutes_in_window"))

    # reconcile the current world BEFORE construction (clubs/injuries/missing)
    scored_live = reconcile_player_clubs(scored, live)
    pool = filter_pool(scored_live, avail, top_k_per_position=25,
                       max_per_team=4, reserve_top=20)
    basket = greedy_teams(pool, n_teams=args.n_teams, seed=1)

    # per-squad live flags
    live_cols = live.select("player_code", "status", "now_cost", "team_code")
    teams = pl.read_parquet(f"{args.processed}/teams_{args.season}.parquet").select(
        pl.col("code").alias("team_code"), pl.col("name").alias("club"))
    bm = (
        basket
        .join(players.select("player_code", "web_name", "team_code"),
              on="player_code", how="left")
        .join(teams, on="team_code", how="left")
        .join(live_cols, on="player_code", how="left", suffix="_live")
        .with_columns(
            (pl.col("now_cost_live") - (pl.col("now_cost") * 10).round())
            .alias("price_diff_tenths"),
        )
        .with_columns(
            pl.struct(["status", "team_code", "team_code_live", "price_diff_tenths"])
            .map_elements(lambda r: flag_squad_player(dict(r)),
                          return_dtype=pl.String).alias("problems")
        )
    )

    print(f"\nlive {fetched} | 2026-27 GW{args.gw} candidates "
          f"(scored {scored_live.height} players, pool {pool.height}, "
          f"basket {basket['team_id'].n_unique()} squads)\n"
          f"{'='*104}")

    # pick the first few DISTINCT squads for display (team_ids skip empties)
    distinct_ids = (
        bm.with_columns(pl.col("player_code").sort().over("team_id").alias("sig"))
        .select("team_id", "sig").unique(subset=["team_id", "sig"])
        .unique(subset="sig", keep="first")["team_id"].to_list()
    )
    seen = 0
    for tid in distinct_ids:
        if seen >= 3:
            break
        sq = bm.filter(pl.col("team_id") == tid).sort(
            "position", "expected_total", descending=[False, True])
        if sq.height == 0:
            continue
        seen += 1
        cost = sq["price_tenths"].sum() / 10
        exp = sq["expected_total"].sum()
        print(f"\n### squad {tid} | £{cost:.1f}m | expected GW{args.gw} {exp:.1f}")
        for row in sq.iter_rows(named=True):
            print(f"  {row['web_name']:<26} {row['club'] or '?':<14} "
                  f"status={row['status'] or '-':>1}  {row['problems'][:56]}")

    keep = suggest(live).sum()
    print(f"\n  {int(keep)}/{live.height} live players are selectable "
          f"(suggest mask)")
    nd = bm.group_by("team_id").agg(pl.col("player_code").sort().alias("sig"))
    print(f"  basket distinctness: {nd.unique(subset='sig').height} of "
          f"{nd.height} squads")


if __name__ == "__main__":
    main()