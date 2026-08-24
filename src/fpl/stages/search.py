"""DVC search stage and candidate-artifact contract."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from fpl.model.inference import load_model
from fpl.team.harness import run as harness_run

CANDIDATE_COLUMNS = frozenset({
    "team_id", "player_code", "position", "price_tenths", "expected_total",
    "artifact_schema_version",
})


def validate_candidate_artifact(candidates: pl.DataFrame) -> None:
    """Check the minimum schema required to hydrate candidates into squads."""
    missing = CANDIDATE_COLUMNS - set(candidates.columns)
    if missing:
        raise ValueError(
            "candidate artifact missing required columns: "
            + ", ".join(sorted(missing)))


def ranked_candidates(result) -> pl.DataFrame:
    """Lower a SearchResult to the stable parquet contract for downstream stages."""
    frames = [
        result.basket.filter(pl.col("team_id") == team_id)
        .with_columns(
            pl.lit(rank).alias("candidate_rank"),
            pl.lit(1).alias("artifact_schema_version"),
        )
        for rank, team_id in enumerate(result.team_ids)
    ]
    candidates = pl.concat(frames, how="vertical") if frames else result.basket
    validate_candidate_artifact(candidates)
    return candidates


def run(
    *, processed: str, season: str, gw_start: int, gw_end: int,
    model_path: str, enum: str, value_fn: str, n_teams: int, seed: int,
    n_samples: int, out: str, metrics_out: str,
) -> None:
    """Run search and persist the candidate artifact consumed by gym."""
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
