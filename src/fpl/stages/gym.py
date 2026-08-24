"""DVC gym replay stage."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import polars as pl

from fpl.gym import Eval
from fpl.model.inference import load_model
from fpl.model.train import load_training
from fpl.stages.search import validate_candidate_artifact
from fpl.team.harness import basket_squads
from fpl.team.scoring import score_players


def run(
    *, processed: str, season: str, gw_start: int, gw_end: int,
    model_path: str, candidates_path: str, top: int, out: str,
    metrics_out: str,
) -> None:
    """Replay the search artifact and persist gym observability."""
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
    if "candidate_rank" in candidates.columns:
        candidates = candidates.sort("candidate_rank")
    squads = basket_squads(candidates, players, gw=gw_start)
    forecast = per_gw.select("player_code", "gw", "expected_points")
    evals = [
        Eval(replace(squad, gw=start), gw_stats=gw_stats, players=players,
             weeks=end - start + 1, forecast=forecast, name=f"cand-{i}").run()
        for i, (_, squad) in enumerate(squads[:top])
    ]
    payload = {"schema_version": 1, "season": season,
               "gw_start": start, "gw_end": end,
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
