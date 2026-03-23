"""Amount-deviation features are prior-only; these tests pin the exact values."""

import numpy as np
import pytest

from src.fraud.features_behaviour import (
    card_history_statistics,
    compute_behaviour_features,
    is_round_amount,
    percentile_rank,
    round_increment,
    zscore,
)


def test_zscore_floor_prevents_explosion():
    assert zscore(300.0, 100.0, 0.0, 5.0) == 40.0
    assert zscore(110.0, 100.0, 10.0, 1.0) == 1.0
    with pytest.raises(ValueError):
        zscore(1.0, 0.0, 0.0, 0.0)


def test_percentile_rank_is_strict_less_than():
    assert percentile_rank([100.0, 100.0], 300.0) == 1.0
    assert percentile_rank([100.0, 300.0], 200.0) == 0.5
    assert np.isnan(percentile_rank([], 1.0))


def test_round_amount_detection():
    assert round_increment(100.0) == 100.0
    assert round_increment(250.0) == 50.0
    assert round_increment(37.42) == 0.0
    assert is_round_amount(500.0)
    assert not is_round_amount(37.42)
