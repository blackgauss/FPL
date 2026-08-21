"""Contract tests: cohorts + metrics."""

import numpy as np

from fpl.experiments.cohorts import cohort_masks, group_mae
from fpl.experiments.metrics import pinball, point_metrics, rmse


def test_top10_mask_per_gw():
    pred = np.asarray([1.0, 2.0, 3.0, 0.1, 9.0, 8.0, 7.0, 6.0])
    gw = np.asarray([1, 1, 1, 1, 2, 2, 2, 2])
    masks = cohort_masks(pred, gw, top=0.3)
    assert masks["all"].all()
    # top two of each gw
    assert masks["top10"][[1, 2, 4, 5]].all()
    assert not masks["top10"][[0, 3, 6, 7]].any()


def test_top10_by_position():
    pred = np.asarray([5.0, 4.0, 3.0, 2.0])
    gw = np.asarray([1, 1, 1, 1])
    pos = np.asarray(["MID", "DEF", "MID", "DEF"])
    masks = cohort_masks(pred, gw, positions=pos)
    assert masks["top10_by_position"][0] and not masks["top10_by_position"][2]
    assert masks["top10_by_position"][1] and not masks["top10_by_position"][3]


def test_group_mae():
    actual = np.asarray([1.0, 3.0, 2.0])
    pred = np.asarray([2.0, 2.0, 2.0])
    masks = {"all": np.ones(3, dtype=bool), "none": np.zeros(3, dtype=bool)}
    grouped = group_mae(actual, pred, masks)
    assert "none" not in grouped
    assert grouped["all"] == (1.0 + 1.0 + 0.0) / 3.0


def test_point_metrics():
    actual = np.asarray([2.0, 4.0])
    pred = np.asarray([3.0, 3.0])
    metrics = point_metrics(actual, pred)
    assert metrics["mae"] == 1.0
    assert metrics["rmse"] == pytest.approx(1.0)
    assert metrics["bias"] == 0.0
    assert metrics["n"] == 2


def test_rmse_and_pinball():
    actual = np.asarray([0.0, 10.0])
    pred = np.asarray([5.0, 5.0])
    assert rmse(actual, pred) == pytest.approx(5.0)
    p = pinball(actual, pred, q=0.9)
    assert p == pytest.approx(2.5)  # 0.5 + 4.5, averaged


import pytest  # noqa: E402