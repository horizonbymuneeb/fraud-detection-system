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
