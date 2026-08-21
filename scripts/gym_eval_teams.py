"""Proof of concept: candidate squads -> gym evals -> observability.

Runs the real selection pipeline (score -> filter -> enumerate -> hydrate),
then replays the top candidate Squads through the gym against ACTUAL
gameweek outcomes (who played, what they scored), settling each week with the
real auto-sub / captain rules. Every squad gets an EvalResult whose summary
shows forecast-vs-actual and the unlearned features (captains, subs, dnps)
so the baseline's edge (or lack of it) is visible directly.

Usage:
    python scripts/train_tree.py              # once: write the model
    python scripts/gym_eval_teams.py --season 2025-2026 --gw 31 --gw-end 33
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import polars as pl

from fpl.gym import Eval
from fpl.model.inference import load_model
from fpl.model.train import load_training
from fpl.pipeline import run_basket
from fpl.team.scoring import score_players


def main() -> None:
    ap = argparse.ArgumentParser(description="Candidate squads -> gym evals")
    ap.add_argument("--season", default="2025-2026")
    ap.add_argument("--gw", type=int, default=31)
    ap.add_argument("--gw-end", type=int, default=None)
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--model", default="data/processed/points_lgbm.txt")
    ap.add_argument("--n-teams", type=int, default=20)
    ap.add_argument("--top", type=int, default=3, help="how many squads to eval")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    gw_end = args.gw_end or args.gw

    model = load_model(args.model)

    res = run_basket(
        processed=args.processed, season=args.season,
        gw_start=args.gw, gw_end=gw_end, model=model,
        enum_kw={"n_teams": args.n_teams, "seed": args.seed},
    )
    players = pl.read_parquet(f"{args.processed}/players_{args.season}.parquet")
    gw_stats = pl.read_parquet(f"{args.processed}/gw_stats_{args.season}.parquet")
    td = load_training(args.processed, [args.season])[args.season]
    _, per_gw = score_players(td, model, gw_start=args.gw, gw_end=gw_end,
                              players=players, detail=True)

    forecastable = sorted(per_gw["gw"].unique().to_list())
    start, end = forecastable[0], forecastable[-1]
    weeks = end - start + 1
    forecast = per_gw.select("player_code", "gw", "expected_points")

    print(f"\nseason {args.season}: {len(forecastable)} forecastable GWs "
          f"{start}..{end}; evaluating top {args.top} of {len(res.squads)} "
          f"candidates\n" + "=" * 96)

    evals = []
    for i, squad in enumerate(res.squads[:args.top]):
        s = replace(squad, gw=start)
        ev = Eval(s, gw_stats=gw_stats, players=players, weeks=weeks,
                  forecast=forecast, name=f"cand-{i}").run()
        evals.append(ev)
        print(ev.summary())
        print()

    print("=" * 96)
    best = max(evals, key=lambda e: e.total_actual)
    exact = max(evals, key=lambda e: -(e.gap or 0.0))
    print(f"most actual points : {best.spec.name} "
          f"({best.total_actual:.1f} vs predicted {best.total_predicted:.1f})")
    print(f"least under-predict: {exact.spec.name} "
          f"(gap {(exact.gap or 0.0):+.1f})")
    print(f"baseline edge: candidate evals differ by up to "
          f"{max(e.total_actual for e in evals) - min(e.total_actual for e in evals):.1f} "
          f"actual points — the unlearned features (captains/subs/who-plays) "
          f"are where improvement lives.")


if __name__ == "__main__":
    main()