"""Thin entrypoint for the ingest job, driven by YAML config.

Usage:
    python scripts/ingest.py --config config/data.yaml --season 2025-2026
"""

from __future__ import annotations

import argparse

from fpl.data.config import load_config
from fpl.data.ingest import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Build parquet dataset from FPL-Core CSVs")
    parser.add_argument("--config", required=True, help="path to data.yaml")
    parser.add_argument(
        "--season",
        default=None,
        help="season label, e.g. 2025-2026 (default: config seasons.default)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    season = args.season or cfg.seasons.default
    for path in run(cfg.fpl_core_root, season, cfg.processed_dir):
        print(path)


if __name__ == "__main__":
    main()