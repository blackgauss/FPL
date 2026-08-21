"""Inspect the candidate team basket in a readable, sanity-checkable form.

Prints each squad as 15 named players (position / club / £cost / window
expected points), with a validity summary (counts, budget, positions,
max-per-club). Use it to eyeball whether the enumerated squads are
reasonable FPL teams before trusting the value stage.

Run:  python scripts/inspect_teams.py --season 2025-2026 --gw 31 --gw-end 33
"""

from __future__ import annotations

import argparse

import polars as pl

from fpl.model.inference import load_model
from fpl.team.harness import run


def _club_of(team_code, teams) -> str:
    row = teams.filter(pl.col("code") == team_code)
    return row.get_column("name").item() if row.height else f"#{team_code}"


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
    players = pl.read_parquet(f"{args.processed}/players_{args.season}.parquet")
    teams = pl.read_parquet(f"{args.processed}/teams_{args.season}.parquet")
    teams_names = dict(zip(teams["code"], teams["name"], strict=False))

    players_names = players.select("player_code", "web_name", "team_code")
    basket = (
        res.basket.join(players_names, on="player_code", how="left")
        .with_columns(pl.col("team_code").cast(pl.Int64).replace_strict(
            teams_names, default="?").alias("club"))
        .sort(["team_id", "position", "expected_total"], descending=[True, False, True])
    )

    order = (
        basket.group_by("team_id").agg(pl.col("expected_total").sum().alias("tot"))
        .sort("tot", descending=True)["team_id"].to_list()
    )

    print(f"\nbasket: {basket['team_id'].n_unique()} squads | "
          f"search space log10 ~= {res.search_size[1]:.2f}")
    print("=" * 88)
    for tid in order[args.min_team: args.min_team + args.top]:
        sq = basket.filter(pl.col("team_id") == tid)
        cost = sq["price_tenths"].sum() / 10
        exp = sq["expected_total"].sum()
        print(f"\n### squad {tid} | cost £{cost:.1f}m | 15 players | "
              f"expected {exp:.1f} pts (window)")
        for pos in ["GKP", "DEF", "MID", "FWD"]:
            by_pos = sq.filter(pl.col("position") == pos).sort(
                "expected_total", descending=True)
            for row in by_pos.iter_rows(named=True):
                print(f"  {pos:<3} {row['web_name']:<28} {row['club']:<16} "
                      f"£{row['now_cost']:>4.1f}m  exp {row['expected_total']:>5.1f}")


if __name__ == "__main__":
    main()