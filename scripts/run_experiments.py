"""Run declared experiment(s) through fpl.experiments and write an artifact.

Usage:
    python scripts/run_experiments.py \
        --config config/experiments_matchup.yaml \
        --output experiments/artifacts/matchup.json
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import polars as pl
import yaml

from fpl.experiments.artifacts import write_artifact, write_failed_artifact
from fpl.experiments.run import run_experiment
from fpl.experiments.splits import TemporalSplit, validate_feature_leakage


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run declared experiments")
    parser.add_argument("--config", default="config/experiments_matchup.yaml")
    parser.add_argument("--output",
                        default="experiments/artifacts/gym_experiments.json")
    parser.add_argument("--season-start", action="store_true",
                        help="train on prior season, test early GWs")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)
    seasons = spec["seasons"]
    processed = "data/processed"
    git_sha = _git_sha()

    if args.season_start:
        split = TemporalSplit(
            fit_gw_max=0, cal_start=1, cal_end=0, test_start=1,
            test_end=spec["season_start"].get("test_gw_max", 5))
    else:
        split = TemporalSplit(**spec["split"])

    latest = seasons[-1]
    validate_feature_leakage(
        pl.read_parquet(f"{processed}/features_{latest}.parquet"),
        pl.read_parquet(f"{processed}/players_{latest}.parquet"),
        split)
    print("leakage validation: PASS")

    metadata = {
        "config": str(Path(args.config)),
        "seasons": seasons,
        "split": split.__dict__,
        "git_sha": git_sha,
    }

    outcome_path = Path(args.output)
    results = []
    try:
        for name, exp in spec["experiments"].items():
            print(f"running {name!r} ...")
            results.append(run_experiment(
                {"name": name, "seasons": seasons, "split": split.__dict__,
                 **exp},
                processed=processed, git_sha=git_sha,
            ))
        write_artifact(outcome_path, results, metadata=metadata)
    except Exception as exc:  # noqa: BLE001 - write a failed artifact
        write_failed_artifact(outcome_path, exc, metadata=metadata)
        print(f"FAILED experiment -> wrote {outcome_path}")
        raise
    print(f"completed result artifact -> {outcome_path}")


if __name__ == "__main__":
    main()