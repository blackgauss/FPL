"""Inference: serve a trained model as "expected points of these players".

Row semantics (feature store): the row at gw=k holds features observed before
GW k's deadline; its target `next_points` is that player's points in GW k+1.
So expected points for gameweek G come from predictions on rows where gw == G-1.

The happy path: we already have an assembled `TrainingData` (features + meta)
for the season of interest and a trained model. `expected_points` masks to the
rows for the target gameweek and returns a name-filled report.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


def expected_points(
    td,
    model,
    *,
    gw: int,
    players: pl.DataFrame,
    code_filter: list[int] | None = None,
) -> pl.DataFrame:
    """Predict next-GW points for the gameweek `gw`.

    `td` is a TrainingData whose rows have `gw` values; we predict on the rows
    with gw == gw-1 (their target IS gw's points). `players` carries names for
    the report; `code_filter` optionally restricts to a list of player codes.

    Returns: (player_code, web_name, team, position, gw, expected_points).
    """
    mask = td.gw == (gw - 1)
    if mask.sum() == 0:
        raise ValueError(
            f"no feature rows for gw={gw} (need training rows at gw={gw - 1})")

    pred = np.asarray(model.predict(td.X[mask])).round(3)
    meta = td.meta.filter(pl.col("gw") == (gw - 1),)

    report = (
        meta.with_columns(pl.Series("expected_points", pred))
        .join(
            players.select("player_id", "player_code", "web_name", "position",
                           "team_code"),
            on=["player_id", "player_code"],
            how="left",
        )
        .select("player_code", "web_name", "position", "team_code",
                "gw", "expected_points")
    )
    if code_filter is not None:
        report = report.filter(pl.col("player_code").is_in(code_filter))
    return report.sort("expected_points", descending=True)


def load_model(path: str | Path):
    """Load a lightgbm Booster saved via Booster.save_model."""
    import lightgbm as lgb

    return lgb.Booster(model_file=str(path))


def save_model(model, path: str | Path) -> None:
    """Persist a lightgbm Booster to disk."""
    model.save_model(str(path))