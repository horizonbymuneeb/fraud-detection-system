"""Per-card spend behaviour: is this amount unusual *for this card*?

The comparison baseline is the card's own past, never the future: for row ``i``
the statistics use rows ``0..i-1`` only. A cold-start card therefore reports
``dev_history_count < min_history`` and the pipeline imputes population-level
defaults instead of pretending a single prior purchase defines a pattern.
"""
from __future__ import annotations

from bisect import bisect_left, insort

import numpy as np
import pandas as pd

from .config import Config

REQUIRED_COLUMNS = ("card_id", "timestamp", "amount")
ROUND_INCREMENTS = (1.0, 5.0, 10.0, 50.0, 100.0, 500.0)


def zscore(value: float, mean: float, std: float, floor: float) -> float:
    """Standardised deviation with a floor on the denominator."""
    if floor <= 0:
        raise ValueError("floor must be positive")
    return float((value - mean) / max(std, floor))


def percentile_rank(sorted_values: list[float], value: float) -> float:
    """Fraction of prior values strictly below ``value`` (0..1)."""
    if not sorted_values:
        return float("nan")
    return bisect_left(sorted_values, value) / len(sorted_values)


def round_increment(amount: float) -> float:
    """Largest round increment (1/5/10/50/100/500) that divides ``amount``."""
    for step in reversed(ROUND_INCREMENTS):
        if amount >= step and abs(amount / step - round(amount / step)) < 1e-9:
            return step
    return 0.0


def is_round_amount(amount: float) -> bool:
    """True when the amount is a whole multiple of 50 or 100."""
    return round_increment(amount) >= 50.0
