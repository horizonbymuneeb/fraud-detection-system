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


def hour_of_day(timestamps: pd.Series) -> np.ndarray:
    """Integer hour in ``[0, 23]`` for each timestamp."""
    return pd.to_datetime(timestamps).dt.hour.to_numpy(dtype=np.int64)


def day_of_week(timestamps: pd.Series) -> np.ndarray:
    """Integer day of week, Monday = 0."""
    return pd.to_datetime(timestamps).dt.day_of_week.to_numpy(dtype=np.int64)


def time_of_day_bucket(hours: np.ndarray) -> np.ndarray:
    """Bucket an hour into night / morning / afternoon / evening."""
    hours = np.asarray(hours, dtype=np.int64)
    if np.any((hours < 0) | (hours > 23)):
        raise ValueError("hours must be in [0, 23]")
    return np.where(hours < 6, 0, np.where(hours < 12, 1, np.where(hours < 18, 2, 3)))


def is_night(hours: np.ndarray) -> np.ndarray:
    """1 for the 00:00-05:59 window."""
    return np.isin(np.asarray(hours), NIGHT_HOURS).astype(np.int8)


def is_weekend_down(days: np.ndarray) -> np.ndarray:
    """1 for Saturday and Sunday."""
    return np.isin(np.asarray(days), WEEKEND_DAYS).astype(np.int8)


def risk_lookup(values: pd.Series | np.ndarray, table: dict[str, float], default: float) -> np.ndarray:
    """Map categorical values to risks, defaulting for unseen categories."""
    if not table:
        raise ValueError("risk table must not be empty")
    return np.array(
        [float(table.get(str(value), default)) for value in np.asarray(values)], dtype=np.float64
    )


def category_risk_scores(categories: pd.Series, overrides: dict[str, float] | None = None) -> np.ndarray:
    """Risk score per merchant category."""
    return risk_lookup(categories, category_risk_table(overrides), default=0.35)


def country_risk_scores(countries: pd.Series) -> np.ndarray:
    """Risk score per transaction country."""
    return risk_lookup(countries, COUNTRY_RISK, default=0.5)


def fit_category_frequency(train_frame: pd.DataFrame) -> pd.Series:
    """Share of training transactions per merchant category."""
    if train_frame.empty:
        raise ValueError("cannot fit category frequency on an empty frame")
    if "merchant_category" not in train_frame:
        raise ValueError("frame needs a merchant_category column")
    return train_frame["merchant_category"].value_counts(normalize=True).sort_index()


def apply_category_frequency(categories: pd.Series, frequency: pd.Series) -> np.ndarray:
    """Frequency lookups; unseen categories get the table's minimum share."""
    if frequency.empty:
        return np.full(len(categories), np.nan)
    fallback = float(frequency.min())
    return np.array(
        [float(frequency.get(str(value), fallback)) for value in categories.to_numpy()],
        dtype=np.float64,
    )


def first_seen_flags(frame: pd.DataFrame, column: str) -> np.ndarray:
    """1 where ``column`` appears for the first time for that card.

    Rows are evaluated in timestamp order, so the flag is causal: the first
    sighting of a merchant or country is exactly what a scorer would see.
    """
    if column not in frame:
        raise ValueError(f"frame is missing column: {column}")
    working = frame.copy()
    working["_pos"] = np.arange(len(working))
    ordered = working.sort_values(["card_id", "timestamp"], kind="mergesort")
    flags = np.zeros(len(frame), dtype=np.int8)
    for _card, group in ordered.groupby("card_id", sort=False):
        seen: set[str] = set()
        for pos, value in zip(group["_pos"].to_numpy(), group[column].astype(str).to_numpy()):
            if value not in seen:
                seen.add(value)
                flags[int(pos)] = 1
    return flags


def context_features(
    frame: pd.DataFrame,
    category_risk_overrides: dict[str, float] | None = None,
    category_frequency: pd.Series | None = None,
) -> pd.DataFrame:
    """Return the calendar / merchant-risk / novelty feature block."""
    required = {"card_id", "timestamp", "merchant_id", "merchant_category", "country"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"context features need columns: {sorted(missing)}")
    hours = hour_of_day(frame["timestamp"])
    days = day_of_week(frame["timestamp"])
    hour_sin, hour_cos = cyclic_encoding(hours, 24)
    day_sin, day_cos = cyclic_encoding(days, 7)
    frequency = (
        fit_category_frequency(frame) if category_frequency is None else category_frequency
    )
    block = {
        "ctx_hour_sin": hour_sin,
        "ctx_hour_cos": hour_cos,
        "ctx_dow_sin": day_sin,
        "ctx_dow_cos": day_cos,
        "ctx_hour": hours.astype(np.float64),
        "ctx_dow": days.astype(np.float64),
        "ctx_time_of_day": time_of_day_bucket(hours).astype(np.float64),
        "ctx_is_night": is_night(hours).astype(np.float64),
        "ctx_is_weekend": is_weekend_down(days).astype(np.float64),
        "ctx_category_risk": category_risk_scores(frame["merchant_category"], category_risk_overrides),
        "ctx_country_risk": country_risk_scores(frame["country"]),
        "ctx_category_share": apply_category_frequency(frame["merchant_category"], frequency),
        "ctx_new_merchant": first_seen_flags(frame, "merchant_id").astype(np.float64),
        "ctx_new_country": first_seen_flags(frame, "country").astype(np.float64),
    }
    return pd.DataFrame(block, index=frame.index).astype(np.float64)


def context_columns() -> list[str]:
    """Names of the columns :func:`context_features` produces."""
    return [
        "ctx_hour_sin", "ctx_hour_cos", "ctx_dow_sin", "ctx_dow_cos",
        "ctx_hour", "ctx_dow", "ctx_time_of_day", "ctx_is_night",
        "ctx_is_weekend", "ctx_category_risk", "ctx_country_risk",
        "ctx_category_share", "ctx_new_merchant", "ctx_new_country",
    ]


def context_features_for_config(
    frame: pd.DataFrame, cfg: Config, category_frequency: pd.Series | None = None
) -> pd.DataFrame:
    """Context block using the category-risk overrides from ``config.yaml``."""
    return context_features(frame, dict(cfg.features.category_risk), category_frequency)
