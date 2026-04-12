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


def test_day_of_week_and_weekend(context_frame):
    days = day_of_week(context_frame["timestamp"])
    assert days.tolist() == [5, 5, 0]
    assert is_weekend_down(days).tolist() == [1, 1, 0]


def test_risk_tables_and_overrides():
    defaults = default_category_risk()
    assert defaults["crypto"] > defaults["groceries"]
    overridden = category_risk_table({"crypto": 0.1})
    assert overridden["crypto"] == 0.1
    scores = country_risk_scores(pd.Series(["PK", "RU", "ZZ"]))
    assert scores[1] > scores[0]
    assert scores[2] == 0.5


def test_first_seen_flags_are_causal(context_frame):
    flags = first_seen_flags(context_frame, "merchant_id")
    assert flags.tolist() == [1, 0, 1]
    countries = first_seen_flags(context_frame, "country")
    assert countries.tolist() == [1, 0, 1]
    with pytest.raises(ValueError):
        first_seen_flags(context_frame, "missing")


def test_category_frequency_fit_and_apply():
    freq = fit_category_frequency(pd.DataFrame({"merchant_category": ["a", "a", "b"]}))
    assert freq["a"] == pytest.approx(2 / 3)
    applied = apply_category_frequency(pd.Series(["a", "b", "unseen"]), freq)
    assert applied[0] == pytest.approx(2 / 3)
    assert applied[2] == pytest.approx(freq.min())
    with pytest.raises(ValueError):
        fit_category_frequency(pd.DataFrame({"merchant_category": []}))


def test_context_features_shape_and_columns(context_frame):
    features = context_features(context_frame)
    assert list(features.columns) == context_columns()
    assert len(features) == len(context_frame)
    assert features["ctx_is_night"].tolist() == [1.0, 0.0, 0.0]
    assert features["ctx_new_merchant"].tolist() == [1.0, 0.0, 1.0]
    assert features["ctx_country_risk"].iloc[2] > features["ctx_country_risk"].iloc[0]


def test_context_features_require_columns(context_frame):
    with pytest.raises(ValueError, match="country"):
        context_features(context_frame.drop(columns=["country"]))
