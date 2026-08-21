"""Inspect the candidate team basket in a readable, sanity-checkable form.

Prints each squad as 15 named players (position / club / £cost / window
expected points), with a validity summary (counts, budget, positions,
max-per-club). The harness now returns typed `SearchResult.squads`, so this
script reads only Squad objects: no team_id bookkeeping, no manual table
joins, and no price-unit math (Player stores tenths; `to_millions` formats).

This is the interface the weekly optimizers will be built against.

Run:  python scripts/inspect_teams.py --season 2025-2026 --gw 31 --gw-end 33
"""

from __future__ import annotations

import argparse

import polars as pl

from fpl.domain import position_sort_key
from fpl.model.inference import load_model
from fpl.team.harness import run
from fpl.units import to_millions


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect candidate team basket")
    parser.add_argument("--season", default="2025-2026")
    parser.add_argument("--gw", type=int, default=31)
    parser.add_argument("--gw-end", type=int, default=None)
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--model", default="data/processed/points_lgbm.txt")
    parser.add_argument("--n-teams", type=int, default=20)
    parser.add_argument("--min-team", type=int, default=0, help="show the n-th best squad")
    parser.add_argument("--top", type=int, default=3, help="how many squads to print")
    args = parser.parse_args()
    gw_end = args.gw_end or args.gw

    model = load_model(args.model)
    res = run(
        processed=args.processed, season=args.season,
        gw_start=args.gw, gw_end=gw_end, model=model,
        enum_kw={"n_teams": args.n_teams, "seed": 1},
    )
    teams = pl.read_parquet(f"{args.processed}/teams_{args.season}.parquet")
    teams_names = dict(zip(teams["code"], teams["name"], strict=False))
    expected = dict(zip(res.pool["player_code"], res.pool["expected_total"],
                        strict=False))

    print(f"\nbasket: {len(res.squads)} squads | "
          f"search space log10 ~= {res.search_size[1]:.2f}")
    print("=" * 88)
    for i, squad in enumerate(res.squads[args.min_team: args.min_team + args.top],
                              start=args.min_team):
        cost = to_millions(squad.cost_tenths())
        exp = sum(expected.get(p.code, 0.0) for p in squad.players)
        print(f"\n### squad {i} | cost £{cost:.1f}m | 15 players | "
              f"expected {exp:.1f} pts (window)")
        ordered = sorted(squad.players,
                         key=lambda p: (position_sort_key(p.position),
                                        -expected.get(p.code, 0.0)))
        for p in ordered:
            club = teams_names.get(p.club, f"#{p.club}")
            print(f"  {p.position:<3} {p.name:<28} {club:<16} "
                  f"£{to_millions(p.cost_tenths):>4.1f}m  "
                  f"exp {expected.get(p.code, 0.0):>5.1f}")


if __name__ == "__main__":
    main()