"""Demo: serve expected points for a player collection + gameweek context.

Loads the persisted tree model, assembles the current season's features, and
reports expected next-GW points for the given GW. Collect a team by editing
CODES below (stable player_code ids).

Run:  python scripts/train_tree.py   # first, to write points_lgbm.txt
      python scripts/predict.py --season 2025-2026 --gw 31
"""

from __future__ import annotations

import argparse

import polars as pl

from fpl.model.inference import expected_points, load_model
from fpl.model.train import assemble

# Example squad: stable player codes (Haaland, Foden, Bowen, Gabriel, Palmer)
CODES = [223094, 209244, 239719, 117242, 112520]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", default="2025-2026")
    parser.add_argument("--gw", type=int, default=31)
    args = parser.parse_args()

    model = load_model("data/processed/points_lgbm.txt")
    processed = "data/processed"
    feat = pl.read_parquet(f"{processed}/features_{args.season}.parquet")
    players = pl.read_parquet(f"{processed}/players_{args.season}.parquet")
    gw_stats = pl.read_parquet(f"{processed}/gw_stats_{args.season}.parquet")

    td = assemble(feat, players, gw_stats, args.season)
    report = expected_points(td, model, gw=args.gw,
                             players=players, code_filter=CODES)
    print(f"\nExpected next-GW points at gameweek {args.gw} "
          f"(model trained on earlier GWs):")
    print(report)


if __name__ == "__main__":
    main()