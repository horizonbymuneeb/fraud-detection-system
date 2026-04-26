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


def card_groups(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return a mapping card_id -> positional indices sorted by timestamp.

    The frame must have a unique RangeIndex so positions can be used as a key.
    """
    if not frame.index.is_unique:
        raise ValueError("frame index must be unique")
    working = frame.copy()
    working["_pos"] = np.arange(len(working))
    ordered = working.sort_values(["card_id", "timestamp"], kind="mergesort")
    groups: dict[str, np.ndarray] = {}
    for card, block in ordered.groupby("card_id", sort=False):
        groups[str(card)] = block["_pos"].to_numpy(dtype=np.int64)
    return groups


def window_counts_sums(
    times_ns: np.ndarray, amounts: np.ndarray, window_ns: int
) -> tuple[np.ndarray, np.ndarray]:
    """Count and summed amount of prior rows inside ``(t - window, t)``.

    ``times_ns`` must be increasing. Row ``i`` only looks at rows ``< i``, so a
    row can never contribute to its own features (duplicate timestamps resolve
    in favour of the earlier row, matching an online scorer).
    """
    n = len(times_ns)
    if n != len(amounts):
        raise ValueError("times and amounts must have equal length")
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(amounts, dtype=np.float64)))
    positions = np.arange(n)
    left = np.searchsorted(times_ns, times_ns - window_ns, side="right")
    counts = positions - left
    sums = cumulative[positions] - cumulative[left]
    return counts.astype(np.int64), sums
