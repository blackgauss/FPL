"""Stage 1: score every player over a gameweek horizon.

Given the assembled TrainingData and a trained model, produce one predicted
expected-points value per player for a planning window [gw_start, gw_end] —
the "initial player scoring" the team search filters over.

Reuses expected_points_horizon (row gw=k predicts points in gw=k+1 == GW k+1).
Aggregation over the window is a simple sum; keeping per-GW detail lets later
stages (H2H by week, weaknesses) drill down without re-predicting.
"""

from __future__ import annotations

import polars as pl

from fpl.model.inference import expected_points_horizon


def score_players(
    td,
    model,
    *,
    gw_start: int,
    gw_end: int,
    players: pl.DataFrame,
    detail: bool = False,
) -> pl.DataFrame:
    """Expected points per player over [gw_start, gw_end].

    Returns one row per player with `expected_total` (sum over the window) and
    `expected_mean_per_gw`, joined to player identity. With detail=True also
    returns the per-GW frame (one row per player-GW).
    """
    horizon = expected_points_horizon(
        td, model, gw_start=gw_start, gw_end=gw_end, players=players,
    )

    agg = (
        horizon.group_by("player_code", "web_name", "position", "team_code")
        .agg(
            pl.col("expected_points").sum().alias("expected_total"),
            pl.col("expected_points").mean().alias("expected_mean_per_gw"),
        )
        .sort("expected_total", descending=True)
    )

    if detail:
        return agg, horizon
    return agg