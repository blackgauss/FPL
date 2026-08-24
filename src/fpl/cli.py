"""Command-line dispatch for the package pipeline."""

from __future__ import annotations

import argparse

from fpl.live import collection
from fpl.stages import gym, search


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FPL pipeline stages")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    search_parser = subparsers.add_parser("search")
    _common_args(search_parser)
    search_parser.add_argument("--enum", default="greedy")
    search_parser.add_argument("--value", default="h2h")
    search_parser.add_argument("--n-teams", type=int, default=20)
    search_parser.add_argument("--seed", type=int, default=0)
    search_parser.add_argument("--n-samples", type=int, default=150)
    search_parser.add_argument("--out", required=True)
    search_parser.add_argument("--metrics-out", required=True)

    gym_parser = subparsers.add_parser("gym")
    _common_args(gym_parser)
    gym_parser.add_argument("--candidates", required=True)
    gym_parser.add_argument("--top", type=int, default=3)
    gym_parser.add_argument("--out", required=True)
    gym_parser.add_argument("--metrics-out", required=True)

    collect_parser = subparsers.add_parser("collect", help="collect league/team state")
    collect_parser.add_argument("--league-id", type=int, required=True)
    collect_parser.add_argument("--entry-id", type=int, required=True)
    collect_parser.add_argument("--out", required=True)
    collect_parser.add_argument("--gw-start", type=int, default=1)
    collect_parser.add_argument("--gw-end", type=int)
    collect_parser.add_argument("--league-picks", action="store_true")

    args = parser.parse_args()
    if args.stage == "collect":
        collection.collect(
            league_id=args.league_id, entry_id=args.entry_id, out_dir=args.out,
            gw_start=args.gw_start, gw_end=args.gw_end,
            league_picks=args.league_picks,
        )
    elif args.stage == "search":
        search.run(
            processed=args.processed, season=args.season, gw_start=args.gw,
            gw_end=args.gw_end or args.gw, model_path=args.model, enum=args.enum,
            value_fn=args.value, n_teams=args.n_teams, seed=args.seed,
            n_samples=args.n_samples, out=args.out, metrics_out=args.metrics_out,
        )
    else:
        gym.run(
            processed=args.processed, season=args.season, gw_start=args.gw,
            gw_end=args.gw_end or args.gw, model_path=args.model,
            candidates_path=args.candidates, top=args.top, out=args.out,
            metrics_out=args.metrics_out,
        )


def _common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--season", default="2025-2026")
    parser.add_argument("--gw", type=int, default=31)
    parser.add_argument("--gw-end", type=int)
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--model", default="data/processed/points_lgbm.txt")
