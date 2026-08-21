"""Run the team-search harness and print value + weaknesses.

Usage:
    python scripts/train_tree.py                        # once: write the model
    python scripts/search_teams.py --season 2025-2026 --gw 31 --gw-end 33
    python scripts/search_teams.py --enum greedy --n-teams 20
"""

from __future__ import annotations

import argparse

from fpl.model.inference import load_model
from fpl.team.harness import REGISTRY, run


def main() -> None:
    parser = argparse.ArgumentParser(description="Search over teams")
    parser.add_argument("--season", default="2025-2026")
    parser.add_argument("--gw", type=int, default=31)
    parser.add_argument("--gw-end", type=int, default=None)
    parser.add_argument("--enum", default="greedy", choices=sorted(REGISTRY["enumerate"]))
    parser.add_argument("--value", default="h2h", choices=sorted(REGISTRY["value"]))
    parser.add_argument("--n-teams", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=150)
    parser.add_argument("--dist", default=None,
                        help="path to dist_{season}.parquet (required for h2h_dist)")
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--model", default="data/processed/points_lgbm.txt")
    args = parser.parse_args()
    gw_end = args.gw_end or args.gw

    model = load_model(args.model)
    dist = None
    if args.value == "h2h_dist":
        import polars as pl

        dist = pl.read_parquet(args.dist or
                               f"{args.processed}/dist_{args.season}.parquet")

    res = run(
        processed=args.processed, season=args.season,
        gw_start=args.gw, gw_end=gw_end, model=model,
        enum=args.enum, value_fn=args.value,
        enum_kw={"n_teams": args.n_teams, "seed": args.seed},
        value_kw={"n_samples": args.n_samples, "seed": args.seed},
        dist_forecast=dist,
    )

    n, lg = res.search_size
    print(f"\nseason={res.season} GW {res.gw_start}..{res.gw_end} | "
          f"pool={res.pool.height} players | "
          f"search space (#teams) ~= 10^{lg:.2f}")
    print(f"basket: {res.basket['team_id'].n_unique()} squads\n")

    print("=== value ===")
    cols = ["team_id", "played", "wins", "losses", "draws", "exp_wins",
            "win_ratio", "avg_edge"]
    show = [c for c in cols if c in res.value.columns]
    print(res.value.select(*show).head(10))
    print("\n=== weaknesses ===")
    print(res.weakness.head(10))


if __name__ == "__main__":
    main()