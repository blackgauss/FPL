"""Run the experiment registry and print a comparison.

Assembles shared training pairs per season; a per-experiment feature subset is
applied by calling `assemble` with that subset. All scores are on the same
held-out GW window, so rows are comparable.

Usage:  python scripts/run_experiments.py [--config config/experiments.yaml]
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import polars as pl
import yaml

from fpl.model.experiment import print_results, run_experiment, write_results
from fpl.model.leakage import validate
from fpl.model.train import load_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Run candidate model experiments")
    parser.add_argument("--config", default="config/experiments.yaml")
    parser.add_argument("--season-start", action="store_true",
                        help="use the season_start block: train on prior season, test early GWs")
    parser.add_argument("--output", default="experiments/artifacts/model_experiments.json",
                        help="completed JSON result artifact")
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

    latest = seasons[-1]
    validate(
        pl.read_parquet(f"{root}/features_{latest}.parquet"),
        pl.read_parquet(f"{root}/players_{latest}.parquet"),
        gw_train_max=fit_gw_max, gw_test_min=test_gw_min,
    )
    print("leakage validation: PASS")

    results = []
    loaded = {}
    for name, exp in spec["experiments"].items():
        feats = exp.get("features")  # None => full set
        cats = exp.get("categorical_columns")
        key = (
            tuple(feats) if feats is not None else None,
            tuple(cats) if cats is not None else None,
        )
        if key not in loaded:
            loaded[key] = load_training(root, seasons, feature_columns=feats,
                                        categorical_columns=cats)
        by_season = loaded[key]
        train, fit = by_season[seasons[0]], by_season[seasons[-1]]
        results.append(run_experiment(
            train, fit, name=name, model=exp["model"], params=exp.get("params"),
            fit_gw_max=fit_gw_max, test_gw_min=test_gw_min, test_gw_max=test_gw_max,
        ))

    print_results(results)
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        sha = None
    artifact = write_results(
        results, args.output,
        metadata={
            "config": str(Path(args.config)),
            "seasons": seasons,
            "fit_gw_max": fit_gw_max,
            "test_gw_min": test_gw_min,
            "test_gw_max": test_gw_max,
            "git_sha": sha,
        },
    )
    print(f"completed result artifact -> {artifact}")


if __name__ == "__main__":
    main()
