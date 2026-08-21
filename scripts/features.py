"""Build the player-GW feature store from the parquet dataset.

Reads data/processed (produced by scripts/ingest.py), writes one feature
parquet per season:
    data/processed/features_{season}.parquet
The feature schema is defined in src/fpl/data/features.py.

Usage:
    python scripts/features.py --config config/data.yaml [--season <s>]
"""

from __future__ import annotations

import argparse

import polars as pl

from fpl.data.config import load_config
from fpl.data.features import build_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Build player-GW feature store")
    parser.add_argument("--config", required=True, help="path to data.yaml")
    parser.add_argument("--season", default=None, help="season label (default: all available)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seasons = [args.season] if args.season else cfg.seasons.available
    root = cfg.processed_dir

    # carryover: each player's trailing rolling stats from the PREVIOUS
    # season's feature store, used to seed the next season's GW1 (season start
    # has no current-season history yet). Built from the already-written store,
    # so iteration order matters: process seasons oldest->newest.
    carryover: pl.DataFrame | None = None
    for season in seasons:
        gw_stats = pl.read_parquet(root / f"gw_stats_{season}.parquet")
        team_history = pl.read_parquet(root / f"team_history_{season}.parquet")
        matches = pl.read_parquet(root / f"matches_{season}.parquet")
        players = pl.read_parquet(root / f"players_{season}.parquet")
        features = build_features(gw_stats, team_history, matches, players,
                                  carryover=carryover)
        out = root / f"features_{season}.parquet"
        features.write_parquet(out)
        print(f"{out}: {features.height} rows")
        carryover = features.group_by("player_code").agg(
            pl.col("prev_points").last().alias("prev_points"),
            pl.col("pts_avg_3").last().alias("pts_avg_3"),
            pl.col("pts_avg_5").last().alias("pts_avg_5"),
        )


if __name__ == "__main__":
    main()