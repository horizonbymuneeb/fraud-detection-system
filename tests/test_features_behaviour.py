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


def test_history_statistics_use_only_prior_rows():
    stats = card_history_statistics(np.array([100.0, 100.0, 300.0]))
    assert stats["history_count"].tolist() == [0, 1, 2]
    assert np.isnan(stats["history_mean"][0])
    assert stats["history_mean"][1] == 100.0
    assert stats["history_mean"][2] == 100.0
    assert stats["history_std"][2] == 0.0
    assert stats["history_median"][2] == 100.0
    assert stats["history_max"][2] == 100.0
    assert stats["history_percentile"][2] == 1.0


def test_behaviour_features_exact_values():
    frame = pd_frame([100.0, 100.0, 300.0])
    features = compute_behaviour_features(frame, min_history=3, zscore_floor=5.0)
    assert np.isnan(features["dev_zscore"].iloc[0])
    assert features["dev_zscore"].iloc[1] == 0.0
    assert features["dev_zscore"].iloc[2] == 40.0
    assert features["dev_median_ratio"].iloc[2] == 3.0
    assert features["dev_percentile"].iloc[2] == 1.0
    assert features["dev_cold_start"].tolist() == [1, 1, 1]
    assert features["dev_log_amount"].iloc[0] == pytest.approx(np.log1p(100.0))


def test_cold_start_clears_after_min_history():
    frame = pd_frame([100.0, 120.0, 90.0, 400.0])
    features = compute_behaviour_features(frame, min_history=3, zscore_floor=5.0)
    assert features["dev_cold_start"].tolist() == [1, 1, 1, 0]


def test_behaviour_features_require_columns():
    frame = pd_frame([10.0])
    with pytest.raises(ValueError, match="amount"):
        compute_behaviour_features(frame.drop(columns=["amount"]), 1, 5.0)
    with pytest.raises(ValueError, match="non-negative"):
        compute_behaviour_features(frame, -1, 5.0)


def pd_frame(amounts):
    import pandas as pd

    return pd.DataFrame(
        {
            "card_id": ["c1"] * len(amounts),
            "timestamp": pd.date_range("2024-01-01", periods=len(amounts), freq="h"),
            "amount": amounts,
        }
    )
