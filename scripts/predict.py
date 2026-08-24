"""Serve expected points for a player collection across a GW window.

Loads the persisted tree model, assembles the current season's features, and
reports expected next-GW points for each gameweek in [--gw, --gw-end].

Run:  python -m fpl.stages.train   # first, to write points_lgbm.txt
      python scripts/predict.py --season 2025-2026 --gw 31
      python scripts/predict.py --season 2025-2026 --gw 31 --gw-end 35
"""

from __future__ import annotations

import argparse

import polars as pl

from fpl.model.inference import expected_points_horizon, load_model
from fpl.model.train import load_training

# Example squad: stable player codes (Haaland, Foden, Bowen, Gabriel, Palmer)
CODES = [223094, 209244, 239719, 117242, 112520]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", default="2025-2026")
    parser.add_argument("--gw", type=int, default=31)
    parser.add_argument("--gw-end", type=int, default=None)
    args = parser.parse_args()
    gw_end = args.gw_end or args.gw

    model = load_model("data/processed/points_lgbm.txt")
    processed = "data/processed"
    players = pl.read_parquet(f"{processed}/players_{args.season}.parquet")
    td = load_training(processed, [args.season])[args.season]

    report = expected_points_horizon(
        td, model, gw_start=args.gw, gw_end=gw_end,
        players=players, code_filter=CODES,
    )
    print(f"\nExpected next-GW points for GW {args.gw}..{gw_end}:")
    print(report)


if __name__ == "__main__":
    main()
