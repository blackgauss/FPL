"""Run declared experiment(s) through fpl.experiments and write an artifact.

Usage:
    python scripts/run_experiments.py \
        --config config/experiments_matchup.yaml \
        --output experiments/artifacts/matchup.json \
        --parallel 2 \
        --cache-dir experiments/artifacts/.cache
"""

from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
import yaml

from fpl.experiments import cache
from fpl.experiments.artifacts import write_artifact, write_failed_artifact
from fpl.experiments.run import run_experiment
from fpl.experiments.splits import TemporalSplit, validate_feature_leakage

DEFAULT_CACHE = Path(__file__).resolve().parents[1] / "experiments" / "artifacts" / ".cache"


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
    parser.add_argument("--parallel", type=int, default=1,
                        help="max worker ARMS (LightGBM releases the GIL while "
                             "training; disks/time shared)")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE),
                        help="cross-process fit cache directory; empty string "
                             "disables")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)
    seasons = spec["seasons"]
    processed = "data/processed"
    git_sha = _git_sha()

    # warm the lazy heavy imports BEFORE the worker pool so parallel arms do
    # not race a circular-importing first `import lightgbm` / scipy
    import lightgbm  # noqa: F401
    from scipy import stats  # noqa: F401  (used by metrics.spearman)

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
        "parallel": args.parallel,
    }

    outcome_path = Path(args.output)
    results = []
    cache_dir = args.cache_dir.strip() if args.cache_dir.strip() else None
    try:
        specs = [
            {"name": name, "seasons": seasons, "split": split.__dict__, **exp}
            for name, exp in spec["experiments"].items()
        ]
        if args.parallel > 1:
            with ThreadPoolExecutor(max_workers=args.parallel) as pool:
                results = list(pool.map(
                    lambda s: run_experiment(s, processed=processed,
                                             git_sha=git_sha,
                                             cache_dir=cache_dir),
                    specs))
        else:
            for s in specs:
                print(f"running {s['name']!r} ...")
                results.append(run_experiment(s, processed=processed,
                                              git_sha=git_sha,
                                              cache_dir=cache_dir))
        write_artifact(outcome_path, results, metadata=metadata)
    except Exception as exc:  # noqa: BLE001 - write a failed artifact
        write_failed_artifact(outcome_path, exc, metadata=metadata)
        print(f"FAILED experiment -> wrote {outcome_path}")
        raise
    print(f"completed result artifact -> {outcome_path}")
    print(f"experiment cache: {cache.cache_counts()}")


if __name__ == "__main__":
    main()