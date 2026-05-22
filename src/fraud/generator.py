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
HOME_COUNTRIES: tuple[str, ...] = ("PK", "AE", "US", "GB")
FOREIGN_COUNTRIES: tuple[str, ...] = ("RU", "NG", "CN", "BR", "TR", "VN")
COUNTRY_RISK: dict[str, float] = {
    "PK": 0.05, "AE": 0.05, "US": 0.05, "GB": 0.05,
    "RU": 0.90, "NG": 0.75, "CN": 0.45, "BR": 0.55, "TR": 0.50, "VN": 0.60,
}
CHANNELS: tuple[str, ...] = ("chip", "contactless", "online")
NIGHT_HOURS: tuple[int, ...] = (1, 2, 3, 4, 5)
RISK_TIERS: tuple[str, ...] = ("low", "medium", "high")
TRANSACTION_COLUMNS: tuple[str, ...] = (
    "txn_id", "card_id", "timestamp", "amount", "merchant_id",
    "merchant_category", "country", "channel", "is_fraud", "fraud_type",
)


@dataclass(frozen=True)
class CardProfile:
    """Behavioural profile of one cardholder."""

    card_id: str
    home_country: str
    preferred: tuple[str, ...]
    spend_scale: float
    credit_limit: float
    activity: float
    risk_tier: str


def generate_card_population(n_cards: int, rng: np.random.Generator) -> list[CardProfile]:
    """Build ``n_cards`` cards with drawn geography, habits and limits."""
    if n_cards <= 0:
        raise ValueError("n_cards must be positive")
    categories = list(CATEGORY_PROFILES)
    cards: list[CardProfile] = []
    for i in range(n_cards):
        n_pref = int(rng.integers(2, 5))
        preferred = tuple(rng.choice(categories, size=n_pref, replace=False).tolist())
        cards.append(
            CardProfile(
                card_id=f"C{i:05d}",
                home_country=str(rng.choice(HOME_COUNTRIES, p=[0.55, 0.25, 0.12, 0.08])),
                preferred=preferred,
                spend_scale=float(np.clip(rng.lognormal(0.0, 0.35), 0.4, 3.5)),
                credit_limit=float(rng.choice([500, 1000, 2500, 5000, 10000])),
                activity=float(np.clip(rng.gamma(2.0, 0.45), 0.05, 3.0)),
                risk_tier=str(rng.choice(RISK_TIERS, p=[0.7, 0.25, 0.05])),
            )
        )
    return cards


def draw_category(card: CardProfile, rng: np.random.Generator) -> str:
    """Pick the merchant category for a legitimate purchase on ``card``."""
    if rng.random() < 0.72:
        return str(rng.choice(card.preferred))
    return str(rng.choice(list(CATEGORY_PROFILES)))


def draw_amount(category: str, scale: float, rng: np.random.Generator) -> float:
    """Lognormal amount around the category median, rounded to cents."""
    median, sigma, _risk, _online = CATEGORY_PROFILES[category]
    return round(float(rng.lognormal(np.log(median * scale), sigma)), 2)


def draw_country(card: CardProfile, category: str, rng: np.random.Generator) -> str:
    """Most purchases happen at home; online travel is the usual exception."""
    online_share = CATEGORY_PROFILES[category][3]
    foreign_share = 0.02 + 0.06 * online_share
    if rng.random() < foreign_share:
        return str(rng.choice(FOREIGN_COUNTRIES))
    return card.home_country


def draw_channel(category: str, rng: np.random.Generator) -> str:
    """Channel mix: online categories are mostly card-not-present."""
    online_share = CATEGORY_PROFILES[category][3]
    if rng.random() < online_share:
        return "online"
    return str(rng.choice(["chip", "contactless"]))


def draw_merchant(category: str, rng: np.random.Generator) -> str:
    """Stable-looking merchant identifier scoped by category."""
    return f"{category[:4].upper()}-{int(rng.integers(1, 400)):03d}"


def draw_timestamps(day: pd.Timestamp, activity: float, rng: np.random.Generator) -> list[pd.Timestamp]:
    """Poisson transaction count for a card-day, with a human hour-of-day mix."""
    count = int(rng.poisson(activity))
    stamps: list[pd.Timestamp] = []
    for _ in range(count):
        if rng.random() < 0.78:
            hour = int(rng.integers(8, 23))
        else:
            hour = int(rng.integers(0, 24))
        minute = int(rng.integers(0, 60))
        second = int(rng.integers(0, 60))
        stamps.append(day + pd.Timedelta(hours=hour, minutes=minute, seconds=second))
    return stamps


def make_legit_transaction(
    card: CardProfile, timestamp: pd.Timestamp, rng: np.random.Generator
) -> dict[str, object]:
    """Assemble one legitimate transaction row."""
    category = draw_category(card, rng)
    return {
        "card_id": card.card_id,
        "timestamp": timestamp,
        "amount": draw_amount(category, card.spend_scale, rng),
        "merchant_id": draw_merchant(category, rng),
        "merchant_category": category,
        "country": draw_country(card, category, rng),
        "channel": draw_channel(category, rng),
        "is_fraud": 0,
        "fraud_type": None,
    }


def generate_legit_transactions(
    cards: Sequence[CardProfile], n_days: int, start_date: str, rng: np.random.Generator
) -> list[dict[str, object]]:
    """Walk every card over every day and collect legitimate transactions."""
    rows: list[dict[str, object]] = []
    start = pd.Timestamp(start_date).normalize()
    # Weekend shopping bump keeps the day-of-week feature informative.
    for offset in range(n_days):
        day = start + pd.Timedelta(days=offset)
        day_factor = 1.18 if day.day_of_week >= 5 else 1.0
        for card in cards:
            for ts in draw_timestamps(day, card.activity * day_factor, rng):
                rows.append(make_legit_transaction(card, ts, rng))
    return rows


def make_amount_fraud(card: CardProfile, ts: pd.Timestamp, rng: np.random.Generator) -> list[dict[str, object]]:
    """Suspiciously large purchase, typically electronics or gift cards."""
    category = str(rng.choice(["electronics", "gift_cards", "jewellery", "crypto"]))
    amount = round(float(card.credit_limit * rng.uniform(0.55, 0.98)), 2)
    return [{
        "card_id": card.card_id, "timestamp": ts, "amount": amount,
        "merchant_id": draw_merchant(category, rng), "merchant_category": category,
        "country": card.home_country, "channel": "online", "is_fraud": 1,
        "fraud_type": "amount",
    }]


def make_velocity_fraud(card: CardProfile, ts: pd.Timestamp, rng: np.random.Generator) -> list[dict[str, object]]:
    """Burst of 5-8 card-present attempts within about an hour."""
    n = int(rng.integers(5, 9))
    rows: list[dict[str, object]] = []
    for k in range(n):
        category = str(rng.choice(["fuel", "groceries", "gift_cards", "pharmacy"]))
        rows.append({
            "card_id": card.card_id,
            "timestamp": ts + pd.Timedelta(minutes=int(k * rng.integers(3, 9))),
            "amount": round(float(rng.uniform(15.0, 120.0) * card.spend_scale), 2),
            "merchant_id": draw_merchant(category, rng),
            "merchant_category": category,
            "country": card.home_country,
            "channel": "chip",
            "is_fraud": 1,
            "fraud_type": "velocity",
        })
    return rows


def make_geo_fraud(card: CardProfile, ts: pd.Timestamp, rng: np.random.Generator) -> list[dict[str, object]]:
    """Card suddenly used abroad, in a country it has never appeared in."""
    category = str(rng.choice(["electronics", "travel", "crypto"]))
    country = str(rng.choice(FOREIGN_COUNTRIES))
    return [{
        "card_id": card.card_id, "timestamp": ts,
        "amount": round(float(rng.uniform(200.0, 1800.0) * card.spend_scale), 2),
        "merchant_id": draw_merchant(category, rng), "merchant_category": category,
        "country": country, "channel": "online", "is_fraud": 1, "fraud_type": "geo",
    }]


def make_night_fraud(card: CardProfile, ts: pd.Timestamp, rng: np.random.Generator) -> list[dict[str, object]]:
    """Late-night purchase forced into the 01:00-05:00 window."""
    category = str(rng.choice(["gift_cards", "electronics", "streaming"]))
    night_ts = ts.normalize() + pd.Timedelta(hours=int(rng.choice(NIGHT_HOURS)), minutes=int(rng.integers(0, 60)))
    return [{
        "card_id": card.card_id, "timestamp": night_ts,
        "amount": round(float(rng.uniform(80.0, 900.0) * card.spend_scale), 2),
        "merchant_id": draw_merchant(category, rng), "merchant_category": category,
        "country": card.home_country, "channel": "online", "is_fraud": 1,
        "fraud_type": "night",
    }]


FRAUD_BUILDERS = {
    "velocity": make_velocity_fraud,
    "amount": make_amount_fraud,
    "geo": make_geo_fraud,
    "night": make_night_fraud,
}


def allocate_fraud_counts(n_fraud: int, mix: dict[str, float]) -> dict[str, int]:
    """Split a fraud budget across patterns, giving leftovers to the largest share."""
    known = {k: max(0.0, float(mix.get(k, 0.0))) for k in FRAUD_TYPES}
    total = sum(known.values())
    if total <= 0:
        raise ValueError("fraud_mix must contain at least one positive share")
    shares = {k: v / total for k, v in known.items()}
    counts = {k: int(np.floor(n_fraud * s)) for k, s in shares.items()}
    remainder = n_fraud - sum(counts.values())
    winners = sorted(shares, key=lambda k: (-shares[k], k))
    for i in range(remainder):
        counts[winners[i % len(winners)]] += 1
    return counts


def fraud_budget(n_legit: int, base_fraud_rate: float) -> int:
    """Number of fraud *rows* that yields ``base_fraud_rate`` over the stream."""
    if not 0.0 <= base_fraud_rate < 1.0:
        raise ValueError("base_fraud_rate must be in [0, 1)")
    if base_fraud_rate == 0.0:
        return 0
    return max(1, int(round(n_legit * base_fraud_rate / (1.0 - base_fraud_rate))))


def expected_rows_per_event() -> dict[str, float]:
    """Average rows one injected event produces per pattern.

    A velocity ring is a burst, so one event is several rows; the other three
    patterns are single rows. The event budget is divided by these weights so
    the stream lands on the configured row-level fraud rate.
    """
    return {"velocity": 6.5, "amount": 1.0, "geo": 1.0, "night": 1.0}


def allocate_fraud_events(
    n_fraud_rows: int, mix: dict[str, float], rows_per_event: dict[str, float] | None = None
) -> dict[str, int]:
    """Split a row-level fraud budget into event counts per pattern."""
    if n_fraud_rows <= 0:
        return {fraud_type: 0 for fraud_type in FRAUD_TYPES}
    weights = expected_rows_per_event() if rows_per_event is None else rows_per_event
    shares = {k: max(0.0, float(mix.get(k, 0.0))) for k in FRAUD_TYPES}
    total = sum(shares.values())
    if total <= 0:
        raise ValueError("fraud_mix must contain at least one positive share")
    row_budget = {k: n_fraud_rows * shares[k] / total for k in FRAUD_TYPES}
    events = {
        k: int(round(row_budget[k] / max(weights.get(k, 1.0), 1e-9)))
        for k in FRAUD_TYPES
    }
    for k in FRAUD_TYPES:
        if shares[k] > 0 and events[k] == 0:
            events[k] = 1
    return events


def inject_fraud(
    legit_rows: Sequence[dict[str, object]],
    cards: Sequence[CardProfile],
    mix: dict[str, float],
    base_fraud_rate: float,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    """Draw fraud events over the same cards and time span as the legit stream."""
    if not legit_rows:
        return []
    n_fraud_rows = fraud_budget(len(legit_rows), base_fraud_rate)
    events = allocate_fraud_events(n_fraud_rows, mix)
    start = min(r["timestamp"] for r in legit_rows)  # type: ignore[type-var]
    end = max(r["timestamp"] for r in legit_rows)  # type: ignore[type-var]
    span_seconds = max(1, int((end - start).total_seconds()))
    injected: list[dict[str, object]] = []
    for fraud_type in FRAUD_TYPES:
        builder = FRAUD_BUILDERS[fraud_type]
        for _ in range(events[fraud_type]):
            card = cards[int(rng.integers(0, len(cards)))]
            ts = start + pd.Timedelta(seconds=int(rng.integers(0, span_seconds)))
            injected.extend(builder(card, ts, rng))
    return injected


def generate_transactions(cfg: Config) -> pd.DataFrame:
    """Generate the full labelled stream and return it sorted by time."""
    rng = np.random.default_rng(cfg.seed)
    cards = generate_card_population(cfg.data.n_cards, rng)
    legit = generate_legit_transactions(
        cards, cfg.data.n_days, cfg.data.start_date, rng
    )
    fraud = inject_fraud(
        legit, cards, dict(cfg.data.fraud_mix), cfg.data.base_fraud_rate, rng
    )
    frame = pd.DataFrame(legit + fraud, columns=list(TRANSACTION_COLUMNS))
    frame = frame.sort_values(["timestamp", "card_id"], kind="mergesort").reset_index(drop=True)
    frame["txn_id"] = [f"T{i:07d}" for i in range(len(frame))]
    return frame


def validate_transactions(frame: pd.DataFrame) -> dict[str, object]:
    """Cheap integrity checks on a transaction frame; returns a summary dict."""
    missing = set(TRANSACTION_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"transaction frame is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("transaction frame is empty")
    if (frame["amount"] <= 0).any():
        raise ValueError("transaction amounts must be positive")
    if not frame["timestamp"].is_monotonic_increasing:
        raise ValueError("transactions must be sorted by timestamp")
    unknown = set(frame["fraud_type"].dropna().unique()) - set(FRAUD_TYPES)
    if unknown:
        raise ValueError(f"unknown fraud_type values: {sorted(unknown)}")
    if (frame.loc[frame["is_fraud"] == 0, "fraud_type"].notna()).any():
        raise ValueError("legitimate transactions must not carry a fraud_type")
    return summarise_transactions(frame)


def summarise_transactions(frame: pd.DataFrame) -> dict[str, object]:
    """Counts used by the CLI and the README's data section."""
    n = int(len(frame))
    fraud = frame["is_fraud"].astype(int)
    by_type = frame.loc[frame["is_fraud"] == 1, "fraud_type"].value_counts().to_dict()
    return {
        "transactions": n,
        "cards": int(frame["card_id"].nunique()),
        "fraud": int(fraud.sum()),
        "fraud_rate": float(fraud.mean()) if n else 0.0,
        "fraud_by_type": {str(k): int(v) for k, v in by_type.items()},
        "start": str(frame["timestamp"].min()),
        "end": str(frame["timestamp"].max()),
        "mean_amount": float(frame["amount"].mean()),
    }


def chronological_split(
    frame: pd.DataFrame, train_fraction: float, gap_hours: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time, dropping a gap so test rows have full velocity history."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if gap_hours < 0:
        raise ValueError("gap_hours must be non-negative")
    ordered = frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    boundary = int(len(ordered) * train_fraction)
    boundary_ts = ordered.loc[boundary, "timestamp"]
    train = ordered.iloc[:boundary].copy()
    test_start = boundary_ts + pd.Timedelta(hours=gap_hours)
    test = ordered.loc[ordered["timestamp"] >= test_start].copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def split_metadata(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, object]:
    """Describe a finished split, including any rows lost to the gap."""
    return {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_fraud": int(train["is_fraud"].sum()),
        "test_fraud": int(test["is_fraud"].sum()),
        "train_end": str(train["timestamp"].max()),
        "test_start": str(test["timestamp"].min()),
        "boundary_hours": float(
            (test["timestamp"].min() - train["timestamp"].max()).total_seconds() / 3600.0
        ),
    }


def write_transactions(frame: pd.DataFrame, path: str | Path) -> Path:
    """Persist a transaction frame as CSV with ISO timestamps."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    out.to_csv(path, index=False)
    return path


def read_transactions(path: str | Path) -> pd.DataFrame:
    """Read a CSV written by :func:`write_transactions`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"transaction file not found: {path}")
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=False)
    frame = frame.sort_values(["timestamp", "card_id"], kind="mergesort").reset_index(drop=True)
    return frame


def generate_and_save(cfg: Config) -> tuple[pd.DataFrame, Path]:
    """Generate, validate, write to ``cfg.data.output_dir`` and return both."""
    frame = generate_transactions(cfg)
    validate_transactions(frame)
    path = write_transactions(frame, Path(cfg.data.output_dir) / "transactions.csv")
    return frame, path
