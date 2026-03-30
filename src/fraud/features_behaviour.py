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


def card_history_statistics(amounts: np.ndarray) -> dict[str, np.ndarray]:
    """Prior-only mean/std/median/percentile/maximum for one card's amounts.

    Uses an incremental sorted list so the median and percentile are exact and
    the whole pass stays O(n log n) rather than O(n^2).
    """
    n = len(amounts)
    out = {
        "history_count": np.zeros(n, dtype=np.int64),
        "history_mean": np.full(n, np.nan),
        "history_std": np.full(n, np.nan),
        "history_median": np.full(n, np.nan),
        "history_percentile": np.full(n, np.nan),
        "history_max": np.full(n, np.nan),
    }
    ordered: list[float] = []
    total = 0.0
    total_sq = 0.0
    running_max = -np.inf
    for i, amount in enumerate(amounts):
        if i > 0:
            mean = total / i
            variance = max(0.0, total_sq / i - mean * mean)
            out["history_count"][i] = i
            out["history_mean"][i] = mean
            out["history_std"][i] = float(np.sqrt(variance))
            out["history_median"][i] = float(ordered[i // 2]) if i % 2 else float(
                0.5 * (ordered[i // 2 - 1] + ordered[i // 2])
            )
            out["history_percentile"][i] = percentile_rank(ordered, float(amount))
            out["history_max"][i] = running_max
        insort(ordered, float(amount))
        total += float(amount)
        total_sq += float(amount) ** 2
        running_max = max(running_max, float(amount))
    return out


def deviation_features(
    amounts: np.ndarray,
    stats: dict[str, np.ndarray],
    min_history: int,
    zscore_floor: float,
) -> dict[str, np.ndarray]:
    """Turn prior-history statistics into model features for the current row."""
    mean = stats["history_mean"]
    std = stats["history_std"]
    median = stats["history_median"]
    count = stats["history_count"]
    has_history = count > 0

    z = np.full(len(amounts), np.nan)
    z[has_history] = (amounts[has_history] - mean[has_history]) / np.maximum(
        std[has_history], zscore_floor
    )
    ratio = np.full(len(amounts), np.nan)
    usable = has_history & (median > 0)
    ratio[usable] = amounts[usable] / median[usable]
    vs_max = np.full(len(amounts), np.nan)
    vs_max[has_history] = amounts[has_history] / np.maximum(
        stats["history_max"][has_history], 1.0
    )
    return {
        "dev_zscore": z,
        "dev_median_ratio": ratio,
        "dev_vs_max": vs_max,
        "dev_percentile": stats["history_percentile"],
        "dev_history_count": count.astype(np.float64),
        "dev_cold_start": (count < min_history).astype(np.int8),
        "dev_log_amount": np.log1p(amounts),
        "dev_is_round": np.array([is_round_amount(float(a)) for a in amounts], dtype=np.int8),
        "dev_round_increment": np.array([round_increment(float(a)) for a in amounts], dtype=np.float64),
    }


def compute_behaviour_features(
    frame: pd.DataFrame, min_history: int, zscore_floor: float
) -> pd.DataFrame:
    """Return the amount-deviation block aligned with ``frame``'s index."""
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"behaviour features need columns: {sorted(missing)}")
    if min_history < 0:
        raise ValueError("min_history must be non-negative")
    working = frame.copy()
    working["_pos"] = np.arange(len(working))
    ordered = working.sort_values(["card_id", "timestamp"], kind="mergesort")
    amounts = frame["amount"].to_numpy(dtype=np.float64)
    block: dict[str, np.ndarray] = {}
    for _card, group in ordered.groupby("card_id", sort=False):
        positions = group["_pos"].to_numpy(dtype=np.int64)
        stats = card_history_statistics(amounts[positions])
        feats = deviation_features(amounts[positions], stats, min_history, zscore_floor)
        for name, values in feats.items():
            target = block.get(name)
            if target is None:
                target = np.zeros(len(frame), dtype=np.float64)
                block[name] = target
            target[positions] = values
    return pd.DataFrame(block, index=frame.index).astype(np.float64)


def behaviour_columns() -> list[str]:
    """Names of the columns :func:`compute_behaviour_features` produces."""
    return [
        "dev_zscore",
        "dev_median_ratio",
        "dev_vs_max",
        "dev_percentile",
        "dev_history_count",
        "dev_cold_start",
        "dev_log_amount",
        "dev_is_round",
        "dev_round_increment",
    ]


def behaviour_features_for_config(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Compute behaviour features using the tunables from ``config.yaml``."""
    return compute_behaviour_features(
        frame, cfg.features.min_history_txns, cfg.features.amount_zscore_floor
    )
