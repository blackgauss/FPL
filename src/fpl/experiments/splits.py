"""Explicit temporal splits for model experiments.

Semantics (feature store): a row at source GW k observes pre-deadline info
for target GW k+1 (its ``next_points``). All windows below are SOURCE rows;
the model predicts plus-one targets.

The runner enforces ``fit < cal < test`` so every metric is out-of-window and
no calibration touches the arbitration slice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fpl.model.leakage import check_split_no_future_in_train, validate


@dataclass(frozen=True)
class TemporalSplit:
    """Source-GW windows for fit / calibration / test."""

    fit_gw_max: int                 # last train source row
    cal_start: int                  # first calibration source row
    cal_end: int                    # last calibration source row (inclusive)
    test_start: int                 # first test source row
    test_end: int | None = None     # last test source row (None = season end)

    def problems(self) -> list[str]:
        problems: list[str] = []
        problems += check_split_no_future_in_train(self.fit_gw_max, self.test_start)
        empty_cal = self.cal_end < self.cal_start
        if not empty_cal and not (self.cal_start <= self.cal_end):
            problems.append("calibration window is empty or reversed")
        if not empty_cal and not (self.fit_gw_max < self.cal_start):
            problems.append(
                f"fit max GW {self.fit_gw_max} >= cal start {self.cal_start}")
        if not (self.cal_end < self.test_start):
            problems.append(
                f"cal end {self.cal_end} >= test start {self.test_start}")
        return problems

    def validate(self) -> None:
        problems = self.problems()
        if problems:
            raise ValueError("invalid temporal split: " + "; ".join(problems))


def source_masks(split: TemporalSplit, source_gw: np.ndarray) -> dict[str, np.ndarray]:
    """Return boolean masks over rows for train/calibration/test.

    `source_gw` is the per-row source GW array (e.g. TrainingData.gw).
    """
    test_mask = (source_gw >= split.test_start) & (
        source_gw <= split.test_end if split.test_end is not None else True)
    return {
        "train": source_gw <= split.fit_gw_max,
        "calibration": (source_gw >= split.cal_start) & (source_gw <= split.cal_end),
        "test": test_mask,
    }


def validate_feature_leakage(
    features, players, split: TemporalSplit,
) -> None:
    """Call the leakage gate for the declared split before any fitting."""
    split.validate()
    validate(features, players,
             gw_train_max=split.fit_gw_max, gw_test_min=split.test_start)