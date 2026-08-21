"""Leakage validation for the modeling pipeline.

The journal (docs/journal/Week1.md, "Data") requires a validation step — with
both data-level and statistical checks — run BEFORE any training. This module
encodes the guarantees the pipeline relies on:

1. Identity: player_id is season-local and gets reused for different players
   across seasons (803/804 shared IDs collide in 2024-25 vs 2025-26). Any join
   of player metadata must use the stable `player_code`.
2. Causality: a feature row for GW(k) must never contain outcome information
   from GW(k) or later. Concretely, the feature-store target comes from the
   NEXT GW via a per-player backward shift.
3. Split integrity: training/test splits are by game week, so no row in train
   has a gw >= min(test gw).

Each check returns a list of violations (empty == pass). Pure and loosely
coupled — call them in the training script or CI before fitting a model.
"""

from __future__ import annotations

import polars as pl


def check_identity_joins_on_stable_code(
    features: pl.DataFrame, players: pl.DataFrame
) -> list[str]:
    """Player metadata used in training must key on `player_code`.

    Checks that features carry both IDs and that no feature row is keyed by a
    player_id whose code (in the supplied players frame) is a different player
    — the signature of a cross-season ID collision.
    """
    problems: list[str] = []
    if "player_code" not in features.columns:
        problems.append("features missing stable player_code column (required)")
    if "player_id" not in players.columns:
        problems.append("players missing player_id (reference frame)")
    if problems:
        return problems

    merged = features.select("player_id", "player_code", "gw").join(
        players.select("player_id", pl.col("player_code").alias("players_code")),
        on="player_id",
        how="left",
    )
    mismatch = merged.filter(
        pl.col("players_code").is_not_null()
        & (pl.col("player_code") != pl.col("players_code"))
    )
    if mismatch.height:
        problems.append(
            f"{mismatch.height} feature rows keyed by a player_id whose code in "
            "the players frame is a different player (cross-season collision)"
        )
    return problems


def check_target_is_next_gw(features: pl.DataFrame) -> list[str]:
    """Statistical causality check: next_points == the following GW's
    total_points for the same player (a clean per-player forward shift)."""
    required = {"player_id", "gw", "total_points", "next_points"}
    if required - set(features.columns):
        return ["missing core columns for target-shift check"]

    df = features.sort("player_id", "gw").with_columns(
        pl.col("total_points").shift(-1).over("player_id").alias("expected_next")
    )
    bad = df.filter(
        pl.col("expected_next").is_not_null()
        & (pl.col("next_points") != pl.col("expected_next"))
    )
    if bad.height:
        return [
            f"{bad.height} rows where next_points != next-GW total_points "
            "(broken target shift — leakage or baggage)"
        ]
    return []


def check_split_no_future_in_train(gw_train_max: int, gw_test_min: int) -> list[str]:
    """Split integrity: training GWs must strictly precede test GWs."""
    if gw_train_max >= gw_test_min:
        return [
            f"train max GW {gw_train_max} >= test min GW {gw_test_min} "
            "(leakage across the split)"
        ]
    return []


def validate(
    features: pl.DataFrame,
    players: pl.DataFrame,
    gw_train_max: int,
    gw_test_min: int,
) -> None:
    """Run all checks; raise ValueError listing any violations."""
    problems = (
        check_identity_joins_on_stable_code(features, players)
        + check_target_is_next_gw(features)
        + check_split_no_future_in_train(gw_train_max, gw_test_min)
    )
    if problems:
        raise ValueError("\n".join(problems))
