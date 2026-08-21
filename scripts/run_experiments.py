"""Run the experiment registry and print a comparison.

Assembles shared training pairs per season; a per-experiment feature subset is
applied by calling `assemble` with that subset. All scores are on the same
held-out GW window, so rows are comparable.

Usage:  python scripts/run_experiments.py [--config config/experiments.yaml]
"""

from __future__ import annotations

import argparse

import polars as pl
import yaml

from fpl.model.experiment import print_results, run_experiment
from fpl.model.train import assemble


def main() -> None:
    parser = argparse.ArgumentParser(description="Run candidate model experiments")
    parser.add_argument("--config", default="config/experiments.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)

    seasons = spec["seasons"]
    fit_gw_max = spec.get("fit_gw_max", 30)
    test_gw_min = spec.get("test_gw_min", 31)
    root = "data/processed"

    seasons_data = {}
    for season in seasons:
        feat = pl.read_parquet(f"{root}/features_{season}.parquet")
        players = pl.read_parquet(f"{root}/players_{season}.parquet")
        gw = pl.read_parquet(f"{root}/gw_stats_{season}.parquet")
        seasons_data[season] = (feat, players, gw)

    results = []
    for name, exp in spec["experiments"].items():
        feats = exp.get("features")  # None => full set
        train, fit = (
            assemble(*seasons_data[s], s, feature_columns=feats)
            for s in seasons
        )
        results.append(run_experiment(
            train, fit, name=name, model=exp["model"], params=exp.get("params"),
            fit_gw_max=fit_gw_max, test_gw_min=test_gw_min,
        ))

    print_results(results)


if __name__ == "__main__":
    main()
