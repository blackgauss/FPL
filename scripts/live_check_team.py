"""Check a candidate team basket against live FPL API state.

Reproduces the team-search basket the way `inspect_teams.py` does, then joins
the LIVE snapshot and flags for every squad: players who are injured /
suspended / unavailable, players who transferred clubs (team mismatch), and
price moves — i.e. the "my candidate team had missing/injured players" problem,
made detectable.

Usage:
    python scripts/live_check_team.py \
        --season 2025-2026 --gw 2 --gw-end 4 \
        --cache data/raw/fpl_api/live.json
"""

from __future__ import annotations

import argparse

import polars as pl

from fpl.live.agreement import to_tenths
from fpl.live.filters import flag_squad_player
from fpl.live.live import load_live_state
from fpl.model.inference import load_model
from fpl.model.train import load_training
from fpl.team.enumerate import greedy_teams
from fpl.team.filtering import availability_from_gw_stats, filter_pool
from fpl.team.scoring import score_players


def main() -> None:
    ap = argparse.ArgumentParser(description="Detect missing/injured players in a team basket")
    ap.add_argument("--season", default="2025-2026")
    ap.add_argument("--gw", type=int, default=2)
    ap.add_argument("--gw-end", type=int, default=None)
    ap.add_argument("--cache", default="data/raw/fpl_api/live.json")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--n-teams", type=int, default=20, help="how many squads to build")
    args = ap.parse_args()
    gw_end = args.gw_end or args.gw

    # 1) reproducibly build the candidate basket
    players = pl.read_parquet(f"{args.processed}/players_{args.season}.parquet")
    gw_stats = pl.read_parquet(f"{args.processed}/gw_stats_{args.season}.parquet")
    teams = pl.read_parquet(f"{args.processed}/teams_{args.season}.parquet")
    model = load_model(f"{args.processed}/points_lgbm.txt")
    td = load_training(args.processed, [args.season])[args.season]
    scored, _ = score_players(td, model, gw_start=args.gw, gw_end=gw_end,
                              players=players, detail=True)
    avail = availability_from_gw_stats(gw_stats, players,
                                       gw_start=args.gw, gw_end=gw_end)
    pool = filter_pool(scored, avail, top_k_per_position=25, max_per_team=4,
                       reserve_top=20)
    basket = greedy_teams(pool, n_teams=args.n_teams, seed=1)

    # 2) live state (rate-limit-safe; cached)
    live, fetched = load_live_state(args.cache, max_age_seconds=3600)
    print(f"\nlive snapshot {fetched} | {live.height} players | "
          f"basket {basket['team_id'].n_unique()} squads\n")
    print("=" * 100)

    # 3) per-squad: join live + flag problems
    live_cols = live.select(
        "player_code", "web_name", "status", "news", "team_code", "now_cost")
    basket_meta = (
        basket
        .join(players.select("player_code", "web_name", "team_code"),
              on="player_code", how="left")
        .join(teams.select(pl.col("code").alias("team_code"),
                           pl.col("name").alias("club")), on="team_code", how="left")
        .join(live_cols, on="player_code", how="left", suffix="_live")
        .with_columns(
            to_tenths(pl.col("now_cost"), 10).alias("ds_price_tenths"),
        )
        .with_columns(
            (pl.col("now_cost_live") - pl.col("ds_price_tenths"))
            .alias("price_diff_tenths"),
        )
    )

    basket_meta = basket_meta.with_columns(
        # per-row flag from the shared, tested helper
        pl.struct(["status", "team_code", "team_code_live", "price_diff_tenths"])
        .map_elements(lambda r: flag_squad_player(dict(r)), return_dtype=pl.String)
        .alias("problems")
    )

    # rank squads by expected total and print the best few, then a problem rollup
    best = basket_meta.group_by("team_id").agg(
        pl.col("expected_total").sum().alias("exp"),
        pl.col("problems").alias("problems"),
    ).with_columns(
        pl.col("problems").list.len().alias("problem_count"),
        pl.col("problems").list.filter(
            pl.element().ne("ok")).alias("real_problems"),
    ).sort("exp", descending=True)

    print("=== best squad + live-detected problems ===")
    best_team = best.head(1)
    tid = best_team.get_column("team_id").item()
    sq = basket_meta.filter(pl.col("team_id") == tid).sort(
        "position", "expected_total", descending=[False, True])
    for row in sq.iter_rows(named=True):
        print(f"  {row['web_name']:<28} {row['club'] or '?':<14} "
              f"status={row['status'] or '-':>1}  {row['problems'][:52]}")

    print("\n=== problem rollup across all squads ===")
    flat = basket_meta.filter(pl.col("problems") != "ok")
    print(flat.group_by(pl.col("web_name"), pl.col("problems")).len().sort(
        "len", descending=True).head(12))
    print(f"\n{flat.height} of {basket.height} team-starts have a live problem "
          f"(injured/suspended/transferred/price-moved)")


if __name__ == "__main__":
    main()