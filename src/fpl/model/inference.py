"""Inference: serve a trained model as "expected points of these players".

Row semantics (feature store): the row at gw=k holds features observed before
GW k's deadline; its target `next_points` is that player's points in GW k+1.
So expected points for gameweek G come from predictions on rows where gw == G-1.

Model-family independence: `save_model`/`load_model` dispatch on a registry so
any estimator with `.predict(X)` can be served — adding a family is one dict
entry in SERIALIZERS, not an edit to serving. The model object itself is opaque
to `expected_points`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


def _save_txt(model, path: Path) -> None:
    """LightGBM Booster persistence (Booster.save_model)."""
    model.save_model(str(path))


def _load_txt(path: Path):
    import lightgbm as lgb

    return lgb.Booster(model_file=str(path))


def _save_pickle(model, path: Path) -> None:
    import pickle

    with open(path, "wb") as fh:
        pickle.dump(model, fh)


def _load_pickle(path: Path):
    import pickle

    with open(path, "rb") as fh:
        return pickle.load(fh)


def _save_joblib(model, path: Path) -> None:
    import joblib

    joblib.dump(model, path)


def _load_joblib(path: Path):
    import joblib

    return joblib.load(path)


SERIALIZERS: dict[str, tuple | None] = {
    ".txt": (_save_txt, _load_txt),        # lightgbm Booster
    ".pkl": (_save_pickle, _load_pickle),  # any pickleable estimator
    ".joblib": (_save_joblib, _load_joblib),
}


def save_model(model, path: str | Path) -> Path:
    """Persist a model; format chosen by filename suffix via SERIALIZERS."""
    path = Path(path)
    key = path.suffix
    if key not in SERIALIZERS:
        raise ValueError(
            f"no serializer for {key!r}; pick one of {sorted(SERIALIZERS)}")
    SERIALIZERS[key][0](model, path)
    return path


def load_model(path: str | Path):
    """Load a model persisted via save_model (format from filename suffix)."""
    path = Path(path)
    key = path.suffix
    if key not in SERIALIZERS:
        raise ValueError(
            f"no deserializer for {key!r}; pick one of {sorted(SERIALIZERS)}")
    return SERIALIZERS[key][1](path)


def expected_points_horizon(
    td,
    model,
    *,
    gw_start: int,
    gw_end: int,
    players: pl.DataFrame,
    code_filter: list[int] | None = None,
) -> pl.DataFrame:
    """Expected points for gameweeks [gw_start, gw_end], one row per player-GW.

    Row semantics: the feature-store row at gw=k predicts points in gw=k+1, so
    gameweek G uses rows where gw == G-1. `code_filter` restricts player_codes.
    Returns (player_code, web_name, position, team_code, gw, expected_points).
    """
    rows = []
    for gw in range(gw_start, gw_end + 1):
        source_gw = gw - 1
        mask = td.gw == source_gw
        if mask.sum() == 0:
            continue  # no data for this GW (e.g. beyond a season's end)
        pred = np.asarray(model.predict(td.X[mask])).round(3)
        meta = td.meta.filter(pl.col("gw") == source_gw)
        frame = (
            meta.with_columns(pl.Series("expected_points", pred))
            .with_columns(pl.lit(gw).alias("gw"))
        )
        rows.append(frame)

    if not rows:
        raise ValueError(
            f"no feature rows for gameweeks {gw_start}..{gw_end} "
            f"(need rows at gw-1 for each)")

    report = (
        pl.concat(rows)
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
    return report.sort(["gw", "expected_points"], descending=[False, True])


def expected_points(
    td,
    model,
    *,
    gw: int,
    players: pl.DataFrame,
    code_filter: list[int] | None = None,
) -> pl.DataFrame:
    """Predict next-GW points for the single gameweek `gw`.

    `td` is a TrainingData whose rows have `gw` values; we predict on the rows
    with gw == gw-1 (their target IS gw's points). `players` carries names for
    the report; `code_filter` optionally restricts to a list of player codes.

    Returns: (player_code, web_name, team, position, gw, expected_points).
    """
    return expected_points_horizon(
        td, model, gw_start=gw, gw_end=gw,
        players=players, code_filter=code_filter,
    )


# (serializers defined above; module ends cleanly after load_model)