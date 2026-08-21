"""Run the experiment registry and print a comparison.

Assembles shared training pairs per season; a per-experiment feature subset is
applied by calling `assemble` with that subset. All scores are on the same
held-out GW window, so rows are comparable.

Usage:  python scripts/run_experiments.py [--config config/experiments.yaml]
"""

from __future__ import annotations

import argparse

import yaml

from fpl.model.experiment import print_results, run_experiment
from fpl.model.train import load_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Run candidate model experiments")
    parser.add_argument("--config", default="config/experiments.yaml")
    parser.add_argument("--season-start", action="store_true",
                        help="use the season_start block: train on prior season, test early GWs")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)

    seasons = spec["seasons"]
    root = "data/processed"
    if args.season_start and "season_start" in spec:
        ss = spec["season_start"]
        fit_gw_max = ss.get("fit_gw_max", 0)
        test_gw_min = 1
        test_gw_max = ss.get("test_gw_max", 5)
    else:
        fit_gw_max = spec.get("fit_gw_max", 30)
        test_gw_min = spec.get("test_gw_min", 31)
        test_gw_max = None

    results = []
    for name, exp in spec["experiments"].items():
        feats = exp.get("features")  # None => full set
        by_season = load_training(root, seasons, feature_columns=feats)
        train, fit = by_season[seasons[0]], by_season[seasons[-1]]
        results.append(run_experiment(
            train, fit, name=name, model=exp["model"], params=exp.get("params"),
            fit_gw_max=fit_gw_max, test_gw_min=test_gw_min, test_gw_max=test_gw_max,
        ))

    print_results(results)


if __name__ == "__main__":
    main()
