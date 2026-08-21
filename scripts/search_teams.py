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
    parser.add_argument("--n-teams", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--model", default="data/processed/points_lgbm.txt")
    args = parser.parse_args()
    gw_end = args.gw_end or args.gw

    model = load_model(args.model)
    res = run(
        processed=args.processed, season=args.season,
        gw_start=args.gw, gw_end=gw_end, model=model,
        enum=args.enum, enum_kw={"n_teams": args.n_teams, "seed": args.seed},
    )

    n, lg = res.search_size
    print(f"\nseason={res.season} GW {res.gw_start}..{res.gw_end} | "
          f"pool={res.pool.height} players | "
          f"search space (#teams) ~= 10^{lg:.2f}")
    print(f"basket: {res.basket['team_id'].n_unique()} squads\n")

    print("=== H2H value ===")
    print(res.value.select("team_id", "played", "wins", "losses", "draws",
                           "win_ratio", "avg_edge").head(10))
    print("\n=== weaknesses ===")
    print(res.weakness.head(10))


if __name__ == "__main__":
    main()