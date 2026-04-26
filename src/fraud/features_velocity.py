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


def distinct_window_counts(
    times_ns: np.ndarray, codes: np.ndarray, window_ns: int
) -> np.ndarray:
    """Number of distinct codes in ``(t - window, t)`` using a two-pointer count.

    ``codes`` must be small non-negative integers (pandas ``factorize`` output).
    """
    n = len(times_ns)
    if n != len(codes):
        raise ValueError("times and codes must have equal length")
    out = np.zeros(n, dtype=np.int64)
    counts: dict[int, int] = {}
    left = 0
    distinct = 0
    for i in range(n):
        limit = times_ns[i] - window_ns
        while left < i and times_ns[left] <= limit:
            code = int(codes[left])
            counts[code] -= 1
            if counts[code] == 0:
                distinct -= 1
            left += 1
        out[i] = distinct
        code = int(codes[i])
        counts[code] = counts.get(code, 0) + 1
        if counts[code] == 1:
            distinct += 1
    return out


def hours_since_previous(times_ns: np.ndarray) -> np.ndarray:
    """Hours since the card's previous transaction; -1 for the first one."""
    gaps = np.full(len(times_ns), -1.0, dtype=np.float64)
    if len(times_ns) > 1:
        gaps[1:] = np.diff(times_ns) / NS_PER_HOUR
    return gaps


def burst_flag(count_1h: np.ndarray, threshold: int = 4) -> np.ndarray:
    """1 when a card produced ``threshold`` or more transactions in one hour."""
    if threshold < 1:
        raise ValueError("threshold must be >= 1")
    return (count_1h >= threshold).astype(np.int8)


def sum_ratio(short_sum: np.ndarray, long_sum: np.ndarray) -> np.ndarray:
    """Share of the long-window volume that landed inside the short window."""
    ratio = np.zeros(len(short_sum), dtype=np.float64)
    usable = long_sum > 0
    ratio[usable] = short_sum[usable] / long_sum[usable]
    return np.clip(ratio, 0.0, 1.0)


def _card_velocity(
    times_ns: np.ndarray,
    amounts: np.ndarray,
    merchant_codes: np.ndarray,
    windows: Sequence[int],
) -> dict[str, np.ndarray]:
    """Compute every velocity column for a single card's ordered history."""
    out: dict[str, np.ndarray] = {}
    for hours in windows:
        counts, sums = window_counts_sums(times_ns, amounts, window_to_ns(hours))
        out[f"vel_count_{hours}h"] = counts
        out[f"vel_sum_{hours}h"] = sums
    out["vel_hours_since_prev"] = hours_since_previous(times_ns)
    out["vel_prior_count"] = np.arange(len(times_ns), dtype=np.int64)
    out["vel_distinct_merchants_24h"] = distinct_window_counts(
        times_ns, merchant_codes, window_to_ns(MERCHANT_WINDOW_HOURS)
    )
    out["vel_is_new_card"] = (out["vel_prior_count"] == 0).astype(np.int8)
    smallest = min(windows)
    largest = max(windows)
    if smallest != largest:
        out["vel_sum_share_short"] = sum_ratio(
            out[f"vel_sum_{smallest}h"], out[f"vel_sum_{largest}h"]
        )
        out["vel_count_share_short"] = sum_ratio(
            out[f"vel_count_{smallest}h"], out[f"vel_count_{largest}h"]
        )
    out["vel_burst_1h"] = burst_flag(out[f"vel_count_{smallest}h"])
    return out
