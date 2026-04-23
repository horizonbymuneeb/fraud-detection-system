"""Transaction-velocity features: how busy is this card in the last 1h/24h/7d?

A transaction is only ever scored against transactions that happened *before*
it. The counters here are therefore computed over the half-open interval
``(t - window, t)`` and never include the row being scored, which is what makes
the features usable in a streaming scorer and leak-free across a time split.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import Config

NS_PER_HOUR = 3_600_000_000_000
REQUIRED_COLUMNS = ("card_id", "timestamp", "amount", "merchant_id")
DEFAULT_WINDOWS_HOURS: tuple[int, ...] = (1, 24, 168)
MERCHANT_WINDOW_HOURS = 24


def window_to_ns(hours: int) -> int:
    """Convert a window expressed in hours to integer nanoseconds."""
    if hours <= 0:
        raise ValueError("window must be a positive number of hours")
    return int(hours) * NS_PER_HOUR


def timestamps_to_ns(timestamps: pd.Series) -> np.ndarray:
    """Convert a datetime series to int64 UTC nanoseconds, rejecting NaT."""
    values = pd.to_datetime(timestamps)
    if values.isna().any():
        raise ValueError("timestamps contain NaT")
    return values.to_numpy(dtype="datetime64[ns]").astype("int64")
