"""The rule baseline every model has to beat.

Rules are what a fraud team already has on day one: a burst of card-present
transactions, an amount far above the card's own median, a first-time foreign
merchant, a large late-night purchase. The baseline matters because it shares
the same features as the model, so the comparison isolates the model's benefit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RULE_WEIGHTS: dict[str, float] = {
    "burst_1h": 2.0,
    "extreme_amount": 2.0,
    "amount_ratio": 1.5,
    "foreign_first": 1.5,
    "night_large": 1.0,
    "new_merchant_spike": 1.0,
}
RULE_THRESHOLD = 2.0


def rule_flags(features: pd.DataFrame) -> pd.DataFrame:
    """Boolean trigger columns; every rule is a single, explainable condition."""
    required = {"vel_count_1h", "dev_zscore", "dev_median_ratio", "ctx_country_risk", "ctx_new_country"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"rule baseline needs features: {sorted(missing)}")
    night = features.get("ctx_is_night", pd.Series(0, index=features.index))
    new_merchant = features.get("ctx_new_merchant", pd.Series(0, index=features.index))
    return pd.DataFrame(
        {
            "burst_1h": features["vel_count_1h"] >= 5,
            "extreme_amount": features["dev_zscore"] >= 6.0,
            "amount_ratio": features["dev_median_ratio"] >= 8.0,
            "foreign_first": (features["ctx_country_risk"] >= 0.6) & (features["ctx_new_country"] > 0),
            "night_large": (night > 0) & (features["dev_median_ratio"] >= 4.0),
            "new_merchant_spike": (new_merchant > 0) & (features["dev_zscore"] >= 3.0),
        },
        index=features.index,
    )


def rule_score(features: pd.DataFrame) -> np.ndarray:
    """Weighted sum of triggered rules (0 = nothing fired)."""
    flags = rule_flags(features)
    weights = np.array([RULE_WEIGHTS[column] for column in flags.columns], dtype=np.float64)
    return flags.to_numpy(dtype=np.float64) @ weights


def rule_predict(scores: np.ndarray, threshold: float = RULE_THRESHOLD) -> np.ndarray:
    """Flag a transaction when the weighted rule score reaches ``threshold``."""
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    return np.asarray(scores, dtype=np.float64) >= threshold


def triggered_rules(features: pd.DataFrame, position: int) -> list[str]:
    """Names of the rules that fired for one positional row."""
    flags = rule_flags(features.iloc[[position]])
    return [column for column in flags.columns if bool(flags.iloc[0][column])]
