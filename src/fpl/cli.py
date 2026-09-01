"""Command-line dispatch for the package pipeline."""

from __future__ import annotations

import argparse
import json

from fpl.live import collection, compare
from fpl.stages import gym, search
from fpl.weekly import sequence


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
    collect_parser.add_argument("--season", default="2026-2027")
    collect_parser.add_argument("--processed", default="data/processed")
    collect_parser.add_argument("--gw-start", type=int, default=1)
    collect_parser.add_argument("--gw-end", type=int)
    collect_parser.add_argument("--league-picks", action="store_true")
    collect_parser.add_argument("--league-type", choices=("classic", "h2h"),
                                default="classic")
    collect_parser.add_argument("--resolve-h2h", action="store_true",
                                help="write standings resolved from event-live points")
    collect_parser.add_argument("--skip-league", action="store_true",
                                help="collect manager data if standings are unavailable")

    compare_parser = subparsers.add_parser("compare", help="compare team to forecast")
    compare_parser.add_argument("--picks", required=True)
    compare_parser.add_argument("--history", required=True)
    compare_parser.add_argument("--processed", default="data/processed")
    compare_parser.add_argument("--season", default="2026-2027")
    compare_parser.add_argument("--model", default="data/processed/points_lgbm.txt")
    compare_parser.add_argument("--gw", type=int, required=True)
    compare_parser.add_argument("--out")
    compare_parser.add_argument("--event-live",
                                help="collected event_live.parquet for authoritative points")
    compare_parser.add_argument("--entry-id", type=int,
                                help="manager ID when picks contain multiple teams")
    compare_parser.add_argument("--official-forecast", action="store_true",
                                help="use FPL ep_this for current GW when model GW0 is unavailable")

    plan_parser = subparsers.add_parser("plan", help="plan the next gameweek")
    plan_parser.add_argument("--picks", required=True)
    plan_parser.add_argument("--history", required=True)
    plan_parser.add_argument("--entry-id", type=int, required=True)
    plan_parser.add_argument("--processed", default="data/processed")
    plan_parser.add_argument("--season", default="2026-2027")
    plan_parser.add_argument("--model", default="data/processed/points_lgbm.txt")
    plan_parser.add_argument("--gw", type=int, required=True,
                             help="current gameweek; plans gw+1")
    plan_parser.add_argument("--bank-tenths", type=int, default=0)
    plan_parser.add_argument("--sell-bottom-pct", type=int, default=40,
                             help="only sell players in this bottom fraction"
                                  " of the squad by expected points (100 = off)")
    plan_parser.add_argument("--top", type=int, default=10)
    plan_parser.add_argument("--out")

    args = parser.parse_args()
    if args.stage == "collect":
        collection.collect(
            league_id=args.league_id, entry_id=args.entry_id, out_dir=args.out,
            gw_start=args.gw_start, gw_end=args.gw_end,
            league_picks=args.league_picks, skip_league=args.skip_league,
            league_type=args.league_type, resolve_h2h=args.resolve_h2h,
            processed=args.processed, season=args.season,
        )
    elif args.stage == "compare":
        payload = compare.write_comparison(
            picks_path=args.picks, history_path=args.history,
            processed=args.processed, season=args.season, model_path=args.model,
            gw=args.gw, out=args.out, event_live_path=args.event_live,
            official_forecast=args.official_forecast, entry_id=args.entry_id,
        )
        print(json.dumps(payload["summary"], indent=2))
    elif args.stage == "plan":
        payload = sequence.run_from_files(
            picks_path=args.picks, history_path=args.history,
            processed=args.processed, season=args.season, model_path=args.model,
            gw=args.gw, bank_tenths=args.bank_tenths, top=args.top,
            sell_bottom_q=args.sell_bottom_pct / 100.0, out=args.out,
            entry_id=args.entry_id,
        )
        print(json.dumps(payload, indent=2))
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
