"""Velocity features are checked against hand-counted windows."""

import numpy as np
import pandas as pd
import pytest

from src.fraud.features_velocity import (
    burst_flag,
    card_groups,
    compute_velocity_features,
    distinct_window_counts,
    hours_since_previous,
    sum_ratio,
    timestamps_to_ns,
    velocity_columns,
    velocity_summary,
    window_counts_sums,
    window_to_ns,
)


@pytest.fixture()
def hand_frame():
    return pd.DataFrame(
        {
            "card_id": ["c1", "c1", "c1", "c1", "c2"],
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 00:00",
                    "2024-01-01 00:30",
                    "2024-01-01 02:00",
                    "2024-01-01 02:00",
                    "2024-01-01 00:10",
                ]
            ),
            "amount": [10.0, 20.0, 30.0, 40.0, 50.0],
            "merchant_id": ["m1", "m2", "m3", "m3", "m9"],
        }
    )


def test_window_ns_and_negative_window_rejected():
    assert window_to_ns(1) == 3_600_000_000_000
    with pytest.raises(ValueError):
        window_to_ns(0)


def test_timestamps_to_ns_rejects_nat():
    with pytest.raises(ValueError, match="NaT"):
        timestamps_to_ns(pd.Series([pd.Timestamp("2024-01-01"), pd.NaT]))


def test_window_counts_exclude_current_and_older_than_window():
    times = timestamps_to_ns(pd.Series(pd.to_datetime(
        ["2024-01-01 00:00", "2024-01-01 00:30", "2024-01-01 02:00"]
    )))
    counts, sums = window_counts_sums(times, np.array([10.0, 20.0, 30.0]), window_to_ns(1))
    assert counts.tolist() == [0, 1, 0]
    assert sums.tolist() == [0.0, 10.0, 0.0]


def test_duplicate_timestamps_resolve_to_earlier_row_only():
    times = timestamps_to_ns(pd.Series(pd.to_datetime(["2024-01-01 02:00", "2024-01-01 02:00"])))
    counts, sums = window_counts_sums(times, np.array([5.0, 7.0]), window_to_ns(1))
    assert counts.tolist() == [0, 1]
    assert sums.tolist() == [0.0, 5.0]


def test_distinct_window_counts():
    times = timestamps_to_ns(pd.Series(pd.to_datetime(
        ["2024-01-01 00:00", "2024-01-01 00:10", "2024-01-01 00:20", "2024-01-01 03:00"]
    )))
    codes = np.array([1, 1, 2, 1])
    assert distinct_window_counts(times, codes, window_to_ns(1)).tolist() == [0, 1, 1, 0]


def test_hours_since_previous_and_burst_flag():
    times = timestamps_to_ns(pd.Series(pd.to_datetime(
        ["2024-01-01 00:00", "2024-01-01 00:30", "2024-01-01 01:00"]
    )))
    gaps = hours_since_previous(times)
    assert gaps[0] == -1.0
    assert np.allclose(gaps[1:], [0.5, 0.5])
    assert burst_flag(np.array([3, 4, 5])).tolist() == [0, 1, 1]
    with pytest.raises(ValueError):
        burst_flag(np.array([1]), threshold=0)


def test_sum_ratio_is_bounded_and_guards_zero():
    ratio = sum_ratio(np.array([0.0, 5.0, 10.0]), np.array([0.0, 10.0, 10.0]))
    assert ratio.tolist() == [0.0, 0.5, 1.0]


def test_hand_computed_window_features(hand_frame):
    features = compute_velocity_features(hand_frame, windows=(1, 24))
    card1 = features.iloc[:4]
    assert card1["vel_count_1h"].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert card1["vel_sum_1h"].tolist() == [0.0, 10.0, 0.0, 30.0]
    assert card1["vel_count_24h"].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert card1["vel_sum_24h"].tolist() == [0.0, 10.0, 30.0, 60.0]
    assert card1["vel_prior_count"].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert card1["vel_hours_since_prev"].tolist()[0] == -1.0
    assert card1["vel_distinct_merchants_24h"].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_new_card_flag_and_short_window_share(hand_frame):
    features = compute_velocity_features(hand_frame, windows=(1, 24))
    assert features["vel_is_new_card"].iloc[0] == 1
    assert features["vel_is_new_card"].iloc[1] == 0
    assert 0.0 <= features["vel_sum_share_short"].iloc[3] <= 1.0
    assert np.isnan(features["vel_count_1h"]).sum() == 0


def test_feature_alignment_uses_position_even_with_shuffled_index(hand_frame):
    shuffled = hand_frame.sort_values("timestamp", ascending=False)
    features = compute_velocity_features(shuffled, windows=(1,))
    assert (features.index == shuffled.index).all()
    first_row = features.loc[hand_frame.index[0]]
    assert first_row["vel_prior_count"] == 0


def test_missing_columns_raise(hand_frame):
    with pytest.raises(ValueError, match="merchant_id"):
        compute_velocity_features(hand_frame.drop(columns=["merchant_id"]))
