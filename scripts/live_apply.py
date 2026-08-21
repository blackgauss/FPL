"""Apply live FPL API state as on-the-fly filters to the team-search pool.

Fetches the bootstrap-static snapshot (rate-limit-safe, cached), builds the
live filter masks, drops injured/suspended/unavailable players from the search
pool, and reports data hygiene (price/team drift) between live and the dataset.

Transfers/price are reported, not filtered by default: a transferred player may
still be selectable in FPL, only at a different club — flag and review rather
than silently exclude.

Usage:
    python scripts/live_apply.py [--season 2026-2027] [--cache ...]
"""

from __future__ import annotations

import argparse

import polars as pl

from fpl.live.agreement import hygiene_summary
from fpl.live.filters import filter_frame_by_code, suggest
from fpl.live.live import load_live_state
from fpl.model.inference import load_model
from fpl.model.train import load_training
from fpl.team.filtering import availability_from_gw_stats, filter_pool
from fpl.team.scoring import score_players


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply live FPL filters to team search")
    parser.add_argument("--season", default="2026-2027")
    parser.add_argument("--gw", type=int, default=2)
    parser.add_argument("--gw-end", type=int, default=None)
    parser.add_argument("--cache", default="data/raw/fpl_api/live.json")
    parser.add_argument("--processed", default="data/processed")
    args = parser.parse_args()
    gw_end = args.gw_end or args.gw

    live, fetched_at = load_live_state(args.cache, max_age_seconds=3600)
    print(f"live snapshot (cached/fetched) {fetched_at} | players={live.height}")

    # existing pipeline is untouched; live filters are applied on the result
    model = load_model(f"{args.processed}/points_lgbm.txt")
    players = pl.read_parquet(f"{args.processed}/players_{args.season}.parquet")
    gw_stats = pl.read_parquet(f"{args.processed}/gw_stats_{args.season}.parquet")
    td = load_training(args.processed, [args.season])[args.season]
    scored, _ = score_players(td, model, gw_start=args.gw, gw_end=gw_end,
                              players=players, detail=True)
    avail = availability_from_gw_stats(gw_stats, players,
                                       gw_start=args.gw, gw_end=gw_end)
    pool = filter_pool(scored, avail, top_k_per_position=25, max_per_team=4,
                       reserve_top=20)

    # on-the-fly: drop players suggest() says not to play
    filtered = filter_frame_by_code(pool, live, suggest(live))
    dropped = pool.height - filtered.height
    print(f"pool: {pool.height} -> {filtered.height} after live filters "
          f"({dropped} excluded: injured/suspended/unavailable)")

    # hygiene: how much does live differ from the dataset these predictions came from?
    summ = hygiene_summary(
        live,
        scored.join(avail.select("player_code", "now_cost"), on="player_code",
                    how="left"),
        dataset_price_col="now_cost",
        dataset_team_col="team_code",
        price_scale=10.0,  # dataset now_cost is decimal millions -> tenths
    )
    print("hygiene (live vs dataset):", summ.to_dict(as_series=False))


if __name__ == "__main__":
    main()