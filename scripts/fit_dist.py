"""Fit residual-CDF distributional forecasts and write them to parquet.

Computes, once, the per-player-GW distributional forecast the team-search
`h2h_dist` value stage consumes: for each player-GW in the horizon, the point
prediction plus the quantile structure of its points distribution (context
bin = position). Writes data/processed/dist_{season}.parquet.

Usage:
    python scripts/fit_dist.py --season 2025-2026 --gw 31 --gw-end 33
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fpl.team.distribution import distributional_forecast


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit + cache distributional forecast")
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--season", default="2025-2026")
    parser.add_argument("--gw", type=int, default=31)
    parser.add_argument("--gw-end", type=int, default=33)
    parser.add_argument("--model", default="data/processed/points_lgbm.txt")
    args = parser.parse_args()
    gw_end = args.gw_end or args.gw

    df = distributional_forecast(
        args.processed, args.season, args.model,
        gw_start=args.gw, gw_end=gw_end,
    )
    out = Path(args.processed) / f"dist_{args.season}.parquet"
    df.write_parquet(out)
    print(f"wrote {out} ({df.height} rows)")


if __name__ == "__main__":
    main()