"""Cyclic time encoding, risk tables and first-sighting flags."""

import numpy as np
import pandas as pd
import pytest

from src.fraud.features_context import (
    apply_category_frequency,
    category_risk_table,
    context_features,
    context_columns,
    country_risk_scores,
    cyclic_encoding,
    day_of_week,
    default_category_risk,
    first_seen_flags,
    fit_category_frequency,
    hour_of_day,
    is_night,
    is_weekend_down,
    time_of_day_bucket,
)


@pytest.fixture()
def context_frame():
    return pd.DataFrame(
        {
            "card_id": ["c1", "c1", "c2"],
            "timestamp": pd.to_datetime(
                ["2024-01-06 00:00", "2024-01-06 13:00", "2024-01-08 23:30"]
            ),
            "merchant_id": ["m1", "m1", "m7"],
            "merchant_category": ["groceries", "groceries", "crypto"],
            "country": ["PK", "PK", "RU"],
        }
    )


def test_cyclic_encoding_matches_trigonometry():
    sin, cos = cyclic_encoding(np.array([0, 6, 12, 18]), 24)
    assert sin[0] == pytest.approx(0.0, abs=1e-12)
    assert cos[0] == pytest.approx(1.0, abs=1e-12)
    assert sin[1] == pytest.approx(1.0, abs=1e-12)
    assert sin[2] == pytest.approx(0.0, abs=1e-12)
    assert cos[2] == pytest.approx(-1.0, abs=1e-12)
    with pytest.raises(ValueError):
        cyclic_encoding(np.array([1]), 1)


def test_time_buckets_and_flags(context_frame):
    hours = hour_of_day(context_frame["timestamp"])
    assert hours.tolist() == [0, 13, 23]
    assert time_of_day_bucket(hours).tolist() == [0, 2, 3]
    assert is_night(hours).tolist() == [1, 0, 0]
    with pytest.raises(ValueError):
        time_of_day_bucket(np.array([24]))
