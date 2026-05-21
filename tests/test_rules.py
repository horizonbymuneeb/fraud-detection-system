"""Rule baseline behaviour on hand-built feature rows."""

import numpy as np
import pandas as pd
import pytest

from src.fraud.rules import (
    RULE_WEIGHTS,
    evaluate_rules,
    rule_flags,
    rule_predict,
    rule_score,
    triggered_rules,
    tune_rule_threshold,
)


@pytest.fixture()
def feature_rows():
    return pd.DataFrame(
        {
            "vel_count_1h": [0.0, 6.0, 1.0],
            "dev_zscore": [0.0, 7.0, 4.0],
            "dev_median_ratio": [1.0, 2.0, 9.0],
            "ctx_country_risk": [0.1, 0.9, 0.1],
            "ctx_new_country": [0.0, 1.0, 0.0],
            "ctx_is_night": [0.0, 0.0, 0.0],
            "ctx_new_merchant": [0.0, 0.0, 1.0],
        }
    )


def test_flags_fire_on_expected_rows(feature_rows):
    flags = rule_flags(feature_rows)
    assert flags.loc[0].sum() == 0
    assert flags.loc[1, "burst_1h"] and flags.loc[1, "extreme_amount"]
    assert flags.loc[1, "foreign_first"]
    assert flags.loc[2, "new_merchant_spike"]


def test_rule_score_is_weighted_sum(feature_rows):
    scores = rule_score(feature_rows)
    assert scores[0] == 0.0
    assert scores[1] == pytest.approx(
        RULE_WEIGHTS["burst_1h"] + RULE_WEIGHTS["extreme_amount"] + RULE_WEIGHTS["foreign_first"]
    )
    assert scores[2] == pytest.approx(RULE_WEIGHTS["new_merchant_spike"] + RULE_WEIGHTS["amount_ratio"])
    assert rule_predict(scores).tolist() == [False, True, True]
    with pytest.raises(ValueError):
        rule_predict(scores, threshold=0)
