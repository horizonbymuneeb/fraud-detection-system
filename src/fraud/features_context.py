"""Context features: time of day, merchant category risk, and first sightings.

Time is encoded cyclically (sin/cos) rather than as a raw integer so 23:00 and
00:00 sit next to each other, and a raw day-of-week column so weekly patterns
stay interpretable. Merchant-category risk is a fixed domain table, overridable
from ``config.yaml``; category frequency is fitted on the training window only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .generator import CATEGORY_PROFILES, COUNTRY_RISK

TIME_BUCKETS: tuple[str, ...] = ("night", "morning", "afternoon", "evening")
NIGHT_HOURS: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
WEEKEND_DAYS: tuple[int, ...] = (5, 6)


def default_category_risk() -> dict[str, float]:
    """Risk table derived from the generator's category profiles."""
    return {category: profile[2] for category, profile in CATEGORY_PROFILES.items()}


def category_risk_table(overrides: dict[str, float] | None = None) -> dict[str, float]:
    """Default risk table with config overrides applied on top."""
    table = default_category_risk()
    if overrides:
        for category, value in overrides.items():
            table[str(category)] = float(value)
    return table


def cyclic_encoding(values: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray]:
    """Map integer-valued cyclical features onto the unit circle."""
    if period <= 1:
        raise ValueError("period must be > 1")
    angle = 2.0 * np.pi * np.asarray(values, dtype=np.float64) / period
    return np.sin(angle), np.cos(angle)
