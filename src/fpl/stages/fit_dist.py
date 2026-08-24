"""Fit heteroskedastic distributional forecasts and write them to parquet.

Trains a point model on fit GWs, fits sigma(X) + a global standardized-
residual t-digest on the held-out slice, and produces per-player-GW CDFs
(pred + sigma*z_q) for the horizon. Writes data/processed/dist_{season}.parquet.

Usage:
    python -m fpl.stages.fit_dist --season 2025-2026 --gw 31 --gw-end 33
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
    parser.add_argument("--fit-gw-max", type=int, default=30)
    parser.add_argument("--test-gw-min", type=int, default=31)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    gw_end = args.gw_end or args.gw

    df = distributional_forecast(
        args.processed, args.season, args.gw, gw_end,
        fit_gw_max=args.fit_gw_max, test_gw_min=args.test_gw_min, seed=args.seed,
    )
    out = Path(args.processed) / f"dist_{args.season}.parquet"
    df.write_parquet(out)
    print(f"wrote {out} ({df.height} rows)")


if __name__ == "__main__":
    main()
