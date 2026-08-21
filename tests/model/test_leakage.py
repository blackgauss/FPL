"""Black-box tests: leakage validation checks."""

import polars as pl
import pytest

from fpl.model.leakage import (
    check_identity_joins_on_stable_code,
    check_split_no_future_in_train,
    check_target_is_next_gw,
    validate,
)


def _features(**overrides):
    data = {
        "player_id": [1, 1, 2],
        "player_code": [223094, 223094, 118748],
        "gw": [2, 3, 2],
        "total_points": [2, 4, 5],
        "next_points": [4, 1, 3],
    }
    data.update(overrides)
    return pl.DataFrame(data)


class TestIdentity:
    def test_missing_stable_code_reported(self):
        f = _features()
        f = f.drop("player_code")
        problems = check_identity_joins_on_stable_code(f, pl.DataFrame(
            {"player_id": [1], "player_code": [223094]}))
        assert problems and "player_code" in problems[0]

    def test_cross_season_collision_detected(self):
        # player_id 1 maps to a different code in the players frame
        f = _features()
        players = pl.DataFrame({
            "player_id": [1, 2],
            "player_code": [999999, 118748],  # id 1 now belongs to a stranger
        })
        problems = check_identity_joins_on_stable_code(f, players)
        assert problems and "different player" in problems[0]

    def test_clean_features_pass(self):
        players = pl.DataFrame({
            "player_id": [1, 2],
            "player_code": [223094, 118748],
        })
        assert check_identity_joins_on_stable_code(_features(), players) == []


class TestTargetShift:
    def test_broken_shift_detected(self):
        # next_points for player 1 gw 2 should equal total_points of gw 3 (4)
        f = _features()
        f = f.with_columns(pl.when(pl.col("gw").is_in([2, 3]) & (pl.col("player_id") == 1))
                           .then(pl.lit(-9)).otherwise(pl.col("next_points"))
                           .alias("next_points"))
        problems = check_target_is_next_gw(f)
        assert problems and "broken target" in problems[0]

    def test_clean_shift_passes(self):
        assert check_target_is_next_gw(_features()) == []


class TestSplit:
    def test_overlapping_split_detected(self):
        problems = check_split_no_future_in_train(31, 30)
        assert problems and "leakage across the split" in problems[0]

    def test_clean_split_passes(self):
        assert check_split_no_future_in_train(30, 31) == []


class TestValidation:
    def test_raises_on_any_violation(self):
        bad = _features().drop("player_code")
        with pytest.raises(ValueError, match="player_code"):
            validate(bad, pl.DataFrame({"player_id": [1]}), 30, 31)

    def test_passes_when_clean(self):
        players = pl.DataFrame({
            "player_id": [1, 2],
            "player_code": [223094, 118748],
        })
        validate(_features(), players, gw_train_max=30, gw_test_min=31)