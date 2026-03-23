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
