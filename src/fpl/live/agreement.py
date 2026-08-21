"""Data-hygiene checks: does live FPL API state agree with our local dataset?

Team selection must reflect the real, current world: prices move, players get
transferred/injured. This module compares the live snapshot (fpl.live.live)
against the local feature/dataset frames and reports disagreements so they can
be surfaced (or filters applied) before any selection is trusted.

Comparisons are keyed on the stable `player_code`.

PRICE UNITS (the classic trap): live `now_cost` is in FPL tenths (155 = £15.5m).
Our local dataset stores decimal millions (15.5). Comparisons must normalize
to the same units first — see `to_tenths`/`price_scale` — otherwise every
player looks like a price move. `price_scale` says "dataset now_cost × scale =
tenths": 10 for decimal-millions datasets, 1 for already-tenths.
"""

from __future__ import annotations

import polars as pl


def to_tenths(price: pl.Series, scale: float = 10.0) -> pl.Series:
    """Convert a dataset price column to FPL tenths units.

    scale = 10 when the dataset stores decimal millions (e.g. 15.5 -> 155);
    scale = 1 when it already stores tenths. Floating, then rounded, so the
    comparison is not derailed by representation noise.
    """
    return (price.cast(pl.Float64) * scale).round().cast(pl.Int64)


def price_diff_tenths(
    live: pl.DataFrame, dataset_now_cost: pl.Series, *, scale: float = 10.0
) -> pl.Series:
    """live.now_cost (tenths) minus the dataset price normalized to tenths."""
    return live.get_column("now_cost") - to_tenths(dataset_now_cost, scale)


def report_agreement(
    live: pl.DataFrame,
    dataset: pl.DataFrame,
    *,
    dataset_price_col: str,
    dataset_team_col: str,
    price_scale: float = 10.0,
) -> pl.DataFrame:
    """Compare live state to a dataset frame (aligned by player_code order).

    `dataset` must carry player_code + price (in dataset units; `price_scale`
    normalizes to tenths) + team_code. Returns per-player frame:
    player_code, web_name, live vs normalized dataset price + diff (tenths),
    live vs dataset team_code (transfers), live status, and `matched_to_dataset`
    (whether the player existed in the dataset).
    """
    d = dataset.sort("player_code").select(
        "player_code", pl.col(dataset_price_col).alias("ds_price_raw"),
        pl.col(dataset_team_col).alias("ds_team"),
    )
    out = (
        live.sort("player_code")
        .join(d, on="player_code", how="left")
        .with_columns(
            to_tenths(pl.col("ds_price_raw"), price_scale).alias("ds_price"),
        )
        .with_columns(
            (pl.col("now_cost") - pl.col("ds_price")).alias("price_diff_tenths"),
            (pl.col("team_code").eq(pl.col("ds_team"))).alias("team_ok"),
            (pl.col("ds_price").is_not_null()).alias("matched_to_dataset"),
        )
        .select(
            "player_code", "web_name", "now_cost", "ds_price", "price_diff_tenths",
            "team_code", "ds_team", "team_ok", "status", "news",
            "matched_to_dataset",
        )
    )
    return out


def hygiene_summary(
    live: pl.DataFrame,
    dataset: pl.DataFrame,
    *,
    dataset_price_col: str = "now_cost",
    dataset_team_col: str = "team_code",
    price_scale: float = 10.0,
) -> pl.DataFrame:
    """Aggregate hygiene: counts of mismatches by kind. Always a small frame."""
    rep = report_agreement(live, dataset,
                           dataset_price_col=dataset_price_col,
                           dataset_team_col=dataset_team_col,
                           price_scale=price_scale)
    return pl.DataFrame({
        "in_live": [rep.height],
        "matched_to_dataset": [int(rep["matched_to_dataset"].sum())],
        "price_moved": [int((rep["price_diff_tenths"].abs() > 0).sum())],
        "team_transferred": [int((~rep["team_ok"]).sum())],
        "not_available": [int(rep["status"].is_in(["i", "s", "u"]).sum())],
    })


def filterable_flag(live: pl.DataFrame, dataset: pl.DataFrame,
                    *, dataset_team_col: str = "team_code") -> pl.Series:
    """Boolean per live-ordered player: whether this player should be excluded
    because live disagrees with the dataset (transferred, unavailable)."""
    rep = report_agreement(live, dataset, dataset_price_col="now_cost",
                           dataset_team_col=dataset_team_col)
    return (~rep["team_ok"] | rep["status"].is_in(["i", "s", "u"])).alias("exclude")