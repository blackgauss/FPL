"""Contract tests: temporal splits + leakage integration."""

import numpy as np
import pytest

from fpl.experiments.splits import TemporalSplit, source_masks


def test_valid_split_masks():
    split = TemporalSplit(fit_gw_max=27, cal_start=28, cal_end=29, test_start=30)
    assert split.problems() == []
    gw = np.arange(1, 35)
    masks = source_masks(split, gw)
    assert list(masks["train"]) == [g <= 27 for g in gw]
    assert list(masks["calibration"]) == [(28 <= g <= 29) for g in gw]
    assert list(masks["test"]) == [g >= 30 for g in gw]


def test_fit_not_before_cal_rejected():
    split = TemporalSplit(fit_gw_max=29, cal_start=28, cal_end=30, test_start=31)
    with pytest.raises(ValueError, match="invalid temporal split"):
        split.validate()


def test_cal_not_before_test_rejected():
    split = TemporalSplit(fit_gw_max=27, cal_start=28, cal_end=31, test_start=30)
    with pytest.raises(ValueError, match="invalid temporal split"):
        split.validate()


def test_empty_cal_allowed():
    # e.g. season-start / no-curation slices
    split = TemporalSplit(fit_gw_max=0, cal_start=1, cal_end=0, test_start=1)
    assert split.problems() == []


def test_leakage_gate_runs_for_declared_split(tmp_path):
    from fpl.data.contract import load_season
    from fpl.data.features import build_features
    from fpl.experiments.splits import validate_feature_leakage
    from tests.fixtures.synthetic import build_season_tree_dense

    root = tmp_path / "s"
    build_season_tree_dense(root, n_players=40, n_gws=4)
    data = load_season(root, "2025-2026")
    feats = build_features(data.gw_stats, data.team_history, data.matches,
                           data.players)
    validate_feature_leakage(
        feats, data.players, TemporalSplit(
            fit_gw_max=1, cal_start=2, cal_end=2, test_start=3))