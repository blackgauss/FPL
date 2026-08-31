"""Compare a collected manager team with model expectations."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import polars as pl

from fpl.domain import squad_from_frame


def compare_team(
    *, picks: pl.DataFrame, history: pl.DataFrame, players: pl.DataFrame,
    gw_stats: pl.DataFrame, forecast: pl.DataFrame, gw: int,
    event_live: pl.DataFrame | None = None, entry_id: int | None = None,
) -> tuple[pl.DataFrame, dict]:
    """Return player-level actual/predicted points and team-level summary.

    The official actual score comes from the FPL entry history, which already
    includes captain fallback and automatic substitutions. Player actuals are
    joined from local GW stats for diagnosis; xScore uses the submitted pick
    multipliers and therefore represents the pre-match forecasted team score.
    """
    selected = picks.filter(pl.col("gw") == gw)
    if "entry_id" in selected.columns:
        ids = selected["entry_id"].unique().to_list()
        if entry_id is None:
            if len(ids) != 1:
                raise ValueError("entry_id is required when picks contain multiple teams")
            entry_id = int(ids[0])
        selected = selected.filter(pl.col("entry_id") == entry_id)
    if selected.height == 0:
        raise ValueError(f"no collected picks for GW {gw}")
    actual_source = event_live if event_live is not None else gw_stats
    actual = actual_source.filter(pl.col("gw") == gw).select(
        "player_id", "minutes", "total_points")
    rows = (
        selected.with_columns(pl.col("element").alias("player_id"))
        .join(
            players.select("player_id", "player_code", "web_name", "position"),
            on="player_id", how="left",
        )
        .join(actual, on="player_id", how="left")
        .join(forecast.filter(pl.col("gw") == gw).select(
            "player_code", "expected_points"), on="player_code", how="left")
        .with_columns(
            pl.col("total_points").fill_null(0).alias("actual_points"),
            pl.col("expected_points").fill_null(0.0),
            (pl.col("expected_points") * pl.col("multiplier"))
            .alias("weighted_expected"),
        )
        .select(
            "gw", "position", "player_id", "player_code", "web_name",
            "multiplier", "is_captain", "is_vice_captain", "minutes",
            "actual_points", "expected_points", "weighted_expected",
        )
        .sort("position")
    )
    history_row = history.filter(pl.col("event") == gw)
    if history_row.height == 0:
        raise ValueError(f"no collected history for GW {gw}")
    history_score = float(history_row["points"].item())
    actual_score = history_score
    score_source = "entry_history"
    if event_live is not None:
        squad_frame = (
            selected.join(
                players.select("player_id", "player_code", "web_name", "position",
                               "team_code"),
                left_on="element", right_on="player_id", how="inner",
            )
            .with_columns(
                pl.col("position_right").cast(pl.String).replace({
                    "1": "GKP", "2": "DEF", "3": "MID", "4": "FWD",
                }).alias("position"),
                pl.lit(0).alias("price_tenths"),
            )
            .select("player_code", "web_name", "position", "team_code",
                    "price_tenths")
        )
        base = squad_from_frame(squad_frame, gw=gw)
        by_slot = rows.sort("position")
        squad = replace(
            base,
            starters=tuple(by_slot.filter(pl.col("position") <= 11)
                            ["player_code"].to_list()),
            bench=tuple(by_slot.filter(pl.col("position") > 11)
                        ["player_code"].to_list()),
            captain=int(by_slot.filter(pl.col("is_captain"))["player_code"].item()),
            vice_captain=int(by_slot.filter(pl.col("is_vice_captain"))
                             ["player_code"].item()),
        )
        points = dict(zip(rows["player_code"], rows["actual_points"], strict=False))
        played = dict(zip(rows["player_code"], rows["minutes"] > 0, strict=False))
        actual_score = float(squad.gw_settlement(played, points).gw_total)
        score_source = "event_live_settlement"
    xscore = float(rows["weighted_expected"].sum())
    summary = {
        "gw": gw,
        "xscore": xscore,
        "actual_score": actual_score,
        "history_score": history_score,
        "score_source": score_source,
        "error": xscore - actual_score,
        "player_count": rows.height,
    }
    return rows, summary


def write_comparison(
    *, picks_path: str, history_path: str, processed: str, season: str,
    model_path: str, gw: int, out: str | None = None,
    event_live_path: str | None = None,
    official_forecast: bool = False, entry_id: int | None = None,
) -> dict:
    """Load collected state and write/return a team comparison."""
    from fpl.model.inference import load_model
    from fpl.model.train import load_training
    from fpl.team.scoring import score_players

    players = pl.read_parquet(f"{processed}/players_{season}.parquet")
    gw_stats = pl.read_parquet(f"{processed}/gw_stats_{season}.parquet")
    # target next_points only exists for COMPLETED GWs; scoring an in-progress
    # or upcoming GW must still work on the feature rows (see team.distribution)
    td = load_training(processed, [season], require_target=False)[season]
    if official_forecast:
        from fpl.live.live import fetch_bootstrap, to_live_frame

        forecast = (
            to_live_frame(fetch_bootstrap())
            .select("player_id", "ep_this")
            .join(players.select("player_id", "player_code"), on="player_id")
            .select("player_code", pl.lit(gw).alias("gw"),
                    pl.col("ep_this").cast(pl.Float64).alias("expected_points"))
        )
    else:
        model = load_model(model_path)
        _, forecast = score_players(
            td, model, gw_start=gw, gw_end=gw, players=players, detail=True)
    rows, summary = compare_team(
        picks=pl.read_parquet(picks_path),
        history=pl.read_parquet(history_path), players=players,
        gw_stats=gw_stats, forecast=forecast, gw=gw,
        event_live=pl.read_parquet(event_live_path) if event_live_path else None,
        entry_id=entry_id,
    )
    payload = {"summary": summary, "players": rows.to_dicts()}
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
