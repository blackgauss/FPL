"""Profile the reference pipeline: where does the time actually go.

Stages profiled (each into experiments/artifacts/profile/):

  - data_prep/load_training        - reading the feature store + assembly.
  - training/fit_registry          - LightGBM fit over the train split.
  - inference/fit_and_predict      - fit + predict on the holdout slice.
  - comparison/run_experiment      - full declared-run harness (fit + cohorts
                                     + ranking/calibration metrics).
  - candidate/run_basket           - candidate-squad generation + value.

Each stage is cProfiled (`*.prof` + structured `*.json`) and the wall-clock
per scope is printed at the end.

Usage:
    python scripts/profile_pipeline.py
    python scripts/profile_pipeline.py --config config/experiments_ranking.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from fpl.experiments.run import run_experiment
from fpl.experiments.splits import TemporalSplit
from fpl.model.experiment import REGISTRY
from fpl.model.train import load_training
from fpl.profiling import profile_call, summarize_profile

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = str(ROOT / "data" / "processed")
PROFILE_DIR = ROOT / "experiments" / "artifacts" / "profile"


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the reference pipeline")
    parser.add_argument("--config", default="config/experiments_ranking.yaml")
    parser.add_argument("--experiment", default="lgbm_all")
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seasons = raw["seasons"]
    split = TemporalSplit(**raw["split"])

    by_season = load_training(PROCESSED, seasons)
    t24, t25 = by_season[seasons[0]], by_season[seasons[-1]]
    train_mask = t25.gw <= split.fit_gw_max
    test_mask = (t25.gw >= split.test_start) & (
        t25.gw <= split.test_end if split.test_end is not None else True)

    def fit() :
        x = np.vstack([t24.X, t25.X[train_mask]])
        y = np.concatenate([t24.y, t25.y[train_mask]])
        return REGISTRY["lgbm"]({})(x, y, t24.categorical)

    def fit_and_predict():
        fn = fit()
        return fn(t25.X[test_mask])

    def declared_run():
        return run_experiment({
            "name": args.experiment, "seasons": seasons, "split": raw["split"],
            "model": "lgbm"}, processed=PROCESSED)

    def candidates():
        from fpl.experiments.candidates import candidate_squads
        from fpl.model.inference import load_model

        return candidate_squads(
            processed=PROCESSED, season=seasons[-1], gw_start=31, gw_end=33,
            model=load_model(f"{PROCESSED}/points_lgbm.txt"),
            n_teams=2, seed=1)

    reports = {
        "data_prep/load_training": (
            lambda: load_training(PROCESSED, seasons), "data_prep_load"),
        "training/fit_registry": (fit, "training_fit"),
        "inference/fit_and_predict_holdout": (fit_and_predict, "inference_holdout"),
        "comparison/run_experiment": (declared_run, "run_experiment"),
        "candidate/run_basket": (candidates, "candidate_basket"),
    }

    profiled = {}
    for label, (fn, name) in reports.items():
        profiled[label] = profile_call(fn, name=name, out_dir=PROFILE_DIR)

    print("\nwall-clock per scope (cProfile wall_seconds):")
    print("=" * 64)
    for label, report in profiled.items():
        print(f"  {label:<38} {report['wall_seconds']:7.2f}s")
    print("\n--- data_prep/load_training ---")
    print(summarize_profile(profiled["data_prep/load_training"]))
    print("\n--- training/fit_registry ---")
    print(summarize_profile(profiled["training/fit_registry"]))
    print("\n--- comparison/run_experiment ---")
    print(summarize_profile(profiled["comparison/run_experiment"]))
    print("\n--- candidate/run_basket ---")
    print(summarize_profile(profiled["candidate/run_basket"]))
    print(f"\nprofiles written to {PROFILE_DIR}")


if __name__ == "__main__":
    main()