"""Apply live FPL API state to the team-search construction input.

Fetches the bootstrap-static snapshot (rate-limit-safe, cached), reconciles the
scored pool with the CURRENT world BEFORE filtering/enumeration runs:
- transferred players get their live club (so max_per_club uses current clubs)
- players missing from the live roster (transferred out of FPL) are dropped
- injured/suspended/unavailable players are excluded via suggest()

Reports data hygiene (price/team drift) between live and the dataset.

Usage:
    python scripts/live_apply.py [--season 2026-2027] [--cache ...]
"""

from __future__ import annotations

import argparse

import polars as pl

from fpl.live.agreement import hygiene_summary
from fpl.live.current import construction_input
from fpl.live.filters import suggest
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

    # 1) score the pool exactly as before (prediction is snapshot-agnostic)
    model = load_model(f"{args.processed}/points_lgbm.txt")
    players = pl.read_parquet(f"{args.processed}/players_{args.season}.parquet")
    gw_stats = pl.read_parquet(f"{args.processed}/gw_stats_{args.season}.parquet")
    td = load_training(args.processed, [args.season])[args.season]
    scored, _ = score_players(td, model, gw_start=args.gw, gw_end=gw_end,
                              players=players, detail=True)

    # 2) reconcile with the current world BEFORE construction: transfers,
    #    missing players, and availability all feed the pool/team-cap
    scored_live = construction_input(scored, live, suggest(live))
    dropped = scored.height - scored_live.height
    print(f"scored {scored.height} -> {scored_live.height} after current-world "
          f"reconcile ({dropped} removed: missing from live / unavailable)")

    # 3) build the pool from the reconciled input (current clubs used)
    avail = availability_from_gw_stats(gw_stats, players,
                                       gw_start=args.gw, gw_end=gw_end)
    pool = filter_pool(scored_live, avail, top_k_per_position=25, max_per_team=4,
                       reserve_top=20)
    print(f"pool after construction on current clubs/injuries: {pool.height} players")

    # 4) hygiene: how much does live differ from the dataset predictions used?
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


if __name__ == "__main__":
    main()