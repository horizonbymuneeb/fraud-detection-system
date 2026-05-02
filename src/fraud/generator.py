"""Synthetic card-transaction generator with labelled fraud patterns.

The generator exists so the pipeline can be exercised end to end without a
data licence. It is deliberately not a toy: cards have a home country, a
preferred category mix, a spend scale and a personal activity rate, fraud is
injected at a configurable base rate, and each injected ring belongs to a named
pattern so recall can be reported per pattern instead of as one number.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import FRAUD_TYPES, Config

# category: (median amount, log-sigma, merchant risk, online share)
CATEGORY_PROFILES: dict[str, tuple[float, float, float, float]] = {
    "groceries": (48.0, 0.55, 0.05, 0.10),
    "restaurants": (32.0, 0.60, 0.10, 0.15),
    "fuel": (60.0, 0.45, 0.20, 0.05),
    "electronics": (420.0, 0.90, 0.55, 0.55),
    "fashion": (110.0, 0.70, 0.20, 0.45),
    "travel": (650.0, 0.80, 0.45, 0.70),
    "gift_cards": (200.0, 0.50, 0.85, 0.60),
    "jewellery": (900.0, 0.70, 0.60, 0.30),
    "crypto": (500.0, 1.00, 0.95, 0.95),
    "utilities": (85.0, 0.30, 0.03, 0.65),
    "pharmacy": (25.0, 0.50, 0.05, 0.10),
    "streaming": (15.0, 0.25, 0.02, 1.00),
}
