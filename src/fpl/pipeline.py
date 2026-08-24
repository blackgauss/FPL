"""Composable pipeline runner: one path from raw data to a candidate basket.

Every later stage (captain, transfers, horizon sim) should build on this single
runner rather than re-implementing the chain. It is pure orchestration: it
delegates to leaf modules and inserts two **gates** that must hold for any team
selection:

    1. freshness — the feature store is non-empty and live hasn't drifted too
                    far from it (fpl.live.freshness).
    2. leakage   — model training must run validate() before fitting
                    (fpl.model.leakage); callers choosing a split must provide
                    its window (enforced by run_experiment, not duplicated).

Separation of concerns: leaf modules (score/filter/enumerate/harness) stay pure
over their inputs; this module owns the wiring and the gates. The harness
accepts an injected `scored` (the seam) so live reconcile can slot in upstream.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import polars as pl

from fpl.live.current import construction_input
from fpl.live.filters import suggest
from fpl.team.enumerate import greedy_teams  # noqa: F401  (registry default)
from fpl.team.harness import basket_squads
from fpl.team.harness import run as harness_run
from fpl.team.scoring import score_players

CANDIDATE_COLUMNS = frozenset({
    "team_id", "player_code", "position", "price_tenths", "expected_total",
})


def run_basket(
    *,
    processed: str,
    season: str,
    gw_start: int,
    gw_end: int,
    model,
    live: pl.DataFrame | None = None,
    freshness: bool = True,
    **harness_kw,
):
    """score -> reconcile(live) -> filter -> enumerate -> value, with gates.

    When `live` is given it is reconciled into the scored input first (club
    transfers, injured/removed) so construction never sees a stale team — the
    harness is handed a pre-scored frame via its `scored` seam.

    Returns a fpl.team.harness.SearchResult. `harness_kw` (enum/value/...)
    forwards to the harness; `pool_kw`, `enum_kw`, `value_fn` etc. are legal.
    """
    from fpl.live.freshness import check_season_has_rows
    from fpl.model.train import load_training

    if freshness:
        check_season_has_rows(processed, season)

    players = pl.read_parquet(f"{processed}/players_{season}.parquet")
    td = load_training(processed, [season])[season]
    scored, per_gw = score_players(
        td, model, gw_start=gw_start, gw_end=gw_end, players=players,
        detail=True)

    if live is not None:
        scored = construction_input(scored, live, suggest(live))

    return harness_run(
        processed=processed, season=season, gw_start=gw_start, gw_end=gw_end,
        model=model, scored=scored, per_gw=per_gw, players=players,
        **harness_kw,
    )


def write_search_stage(
    *, processed: str, season: str, gw_start: int, gw_end: int, model_path: str,
    enum: str, value_fn: str, n_teams: int, seed: int, n_samples: int,
    out: str, metrics_out: str,
) -> None:
    """Run search and persist the candidate artifact consumed by gym."""
    from fpl.model.inference import load_model

    model = load_model(model_path)
    dist = None
    if value_fn == "h2h_dist":
        dist = pl.read_parquet(f"{processed}/dist_{season}.parquet")
    result = harness_run(
        processed=processed, season=season, gw_start=gw_start, gw_end=gw_end,
        model=model, enum=enum, value_fn=value_fn,
        enum_kw={"n_teams": n_teams, "seed": seed},
        value_kw={"n_samples": n_samples, "seed": seed}
        if value_fn == "h2h_dist" else {},
        dist_forecast=dist,
    )
    candidates = ranked_candidates(result)
    output = Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    candidates.write_parquet(output)
    metrics_path = Path(metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({
        "candidate_count": len(result.squads),
        "pool_players": result.pool.height,
        "search_space": result.search_size[0],
        "search_space_log10": result.search_size[1],
    }, indent=2) + "\n", encoding="utf-8")


def write_gym_stage(
    *, processed: str, season: str, gw_start: int, gw_end: int,
    model_path: str, candidates_path: str, top: int, out: str,
    metrics_out: str,
) -> None:
    """Replay the search artifact and persist gym observability."""
    from fpl.gym import Eval
    from fpl.model.inference import load_model
    from fpl.model.train import load_training

    model = load_model(model_path)
    players = pl.read_parquet(f"{processed}/players_{season}.parquet")
    gw_stats = pl.read_parquet(f"{processed}/gw_stats_{season}.parquet")
    td = load_training(processed, [season])[season]
    _, per_gw = score_players(
        td, model, gw_start=gw_start, gw_end=gw_end, players=players, detail=True,
    )
    forecastable = sorted(per_gw["gw"].unique().to_list())
    start, end = forecastable[0], forecastable[-1]
    candidates = pl.read_parquet(candidates_path)
    validate_candidate_artifact(candidates)
    candidates = candidates.sort("candidate_rank") if "candidate_rank" in candidates.columns \
        else candidates
    squads = basket_squads(candidates, players, gw=gw_start)
    forecast = per_gw.select("player_code", "gw", "expected_points")
    evals = [
        Eval(replace(squad, gw=start), gw_stats=gw_stats, players=players,
             weeks=end - start + 1, forecast=forecast, name=f"cand-{i}").run()
        for i, (_, squad) in enumerate(squads[:top])
    ]
    payload = {"season": season, "gw_start": start, "gw_end": end,
               "runs": [evaluation.observability() for evaluation in evals]}
    output = Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    metrics_path = Path(metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    actual = [evaluation.total_actual for evaluation in evals]
    gaps = [evaluation.gap or 0.0 for evaluation in evals]
    metrics_path.write_text(json.dumps({
        "evaluated": len(evals),
        "best_actual": max(actual),
        "mean_actual": sum(actual) / len(actual),
        "mean_gap": sum(gaps) / len(gaps),
    }, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Package CLI used by DVC; stage composition stays in this module."""
    parser = argparse.ArgumentParser(description="Run FPL pipeline stages")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    search = subparsers.add_parser("search")
    _common_args(search)
    search.add_argument("--enum", default="greedy")
    search.add_argument("--value", default="h2h")
    search.add_argument("--n-teams", type=int, default=20)
    search.add_argument("--seed", type=int, default=0)
    search.add_argument("--n-samples", type=int, default=150)
    search.add_argument("--out", required=True)
    search.add_argument("--metrics-out", required=True)

    gym = subparsers.add_parser("gym")
    _common_args(gym)
    gym.add_argument("--candidates", required=True)
    gym.add_argument("--top", type=int, default=3)
    gym.add_argument("--out", required=True)
    gym.add_argument("--metrics-out", required=True)

    args = parser.parse_args()
    if args.stage == "search":
        write_search_stage(
            processed=args.processed, season=args.season, gw_start=args.gw,
            gw_end=args.gw_end or args.gw, model_path=args.model, enum=args.enum,
            value_fn=args.value, n_teams=args.n_teams, seed=args.seed,
            n_samples=args.n_samples, out=args.out, metrics_out=args.metrics_out,
        )
    else:
        write_gym_stage(
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


def ranked_candidates(result) -> pl.DataFrame:
    """Lower a SearchResult to the stable parquet contract for downstream stages."""
    frames = [
        result.basket.filter(pl.col("team_id") == team_id)
        .with_columns(pl.lit(rank).alias("candidate_rank"))
        for rank, team_id in enumerate(result.team_ids)
    ]
    candidates = pl.concat(frames, how="vertical") if frames else result.basket
    validate_candidate_artifact(candidates)
    return candidates


def validate_candidate_artifact(candidates: pl.DataFrame) -> None:
    """Check the minimum schema required to hydrate candidates into Squad values."""
    missing = CANDIDATE_COLUMNS - set(candidates.columns)
    if missing:
        raise ValueError(
            "candidate artifact missing required columns: "
            + ", ".join(sorted(missing)))
