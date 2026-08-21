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


class TestRankingMetrics:
    def test_spearman_perfect_and_antitone(self):
        from fpl.experiments.metrics import spearman

        actual = np.asarray([1.0, 2.0, 3.0, 4.0])
        assert spearman(actual, actual) == 1.0
        assert spearman(actual, actual[::-1]) == -1.0

    def test_topk_hit_rate(self):
        from fpl.experiments.metrics import topk_hit_rate

        actual = np.asarray([1.0, 3.0, 2.0])
        pred = np.asarray([1.0, 3.0, 2.0])
        gw = np.asarray([1, 1, 1])
        assert topk_hit_rate(actual, pred, gw, top=0.3) == 1.0
        # model's top player (3.0 pred) is NOT the actual top (5.0) -> miss
        assert topk_hit_rate(np.asarray([1.0, 5.0, 2.0]),
                             np.asarray([3.0, 1.0, 2.0]), gw, top=0.3) == 0.0

    def test_pairwise_concordance(self):
        from fpl.experiments.metrics import pairwise_concordance

        actual = np.asarray([1.0, 2.0, 3.0])
        assert pairwise_concordance(actual, actual) == 1.0
        assert pairwise_concordance(actual, actual[::-1]) == 0.0


class TestCalibrationMetrics:
    def test_slope_and_ece_perfect(self):
        from fpl.experiments.metrics import cal_line, calibration_metrics, ece

        pred = np.linspace(1, 10, 100)
        actual = pred.copy()  # perfectly calibrated mean line
        assert cal_line(actual, pred)["slope"] == pytest.approx(1.0, abs=1e-9)
        assert cal_line(actual, pred)["intercept"] == pytest.approx(0.0, abs=1e-9)
        assert ece(actual, pred, bins=10) == 0.0
        cal = calibration_metrics(actual, pred)
        assert cal["mae"] == 0.0 and cal["variance_ratio"] == pytest.approx(1.0)

    def test_bad_intercept_detected(self):
        from fpl.experiments.metrics import cal_line

        pred = np.linspace(1, 10, 100)
        actual = pred + 3.0  # systematic over/under states
        line = cal_line(actual, pred)
        assert line["slope"] == pytest.approx(1.0, abs=1e-9)
        assert line["intercept"] == pytest.approx(3.0, abs=1e-6)

    def test_ece_detects_binnable_miscalibration(self):
        from fpl.experiments.metrics import ece

        pred = np.array([0.0, 0.1, 0.9, 1.0])
        actual = np.array([0.5, 0.5, 0.5, 0.5])
        assert ece(actual, pred, bins=4) > 0.0


import pytest  # noqa: E402