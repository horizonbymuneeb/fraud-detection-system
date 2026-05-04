"""The generator is test infrastructure for everything else, so it must be deterministic."""

import numpy as np
import pytest

from src.fraud.config import load_config
from src.fraud.generator import (
    allocate_fraud_counts,
    allocate_fraud_events,
    chronological_split,
    expected_rows_per_event,
    fraud_budget,
    generate_transactions,
    read_transactions,
    summarise_transactions,
    validate_transactions,
    write_transactions,
)


def test_same_seed_gives_identical_frames(tiny_config):
    a = generate_transactions(tiny_config)
    b = generate_transactions(tiny_config)
    assert a.shape == b.shape
    assert (a["amount"].to_numpy() == b["amount"].to_numpy()).all()
    assert (a["timestamp"].to_numpy() == b["timestamp"].to_numpy()).all()


def test_rows_are_time_sorted_and_ids_unique(tiny_frame):
    assert tiny_frame["timestamp"].is_monotonic_increasing
    assert tiny_frame["txn_id"].is_unique
    assert (tiny_frame["amount"] > 0).all()


def test_fraud_rate_is_close_to_the_configured_base_rate(tiny_config, tiny_frame):
    summary = summarise_transactions(tiny_frame)
    assert 0.03 < summary["fraud_rate"] < 0.07
    assert summary["transactions"] == len(tiny_frame)


def test_every_injected_pattern_appears(tiny_frame):
    fraud = tiny_frame[tiny_frame["is_fraud"] == 1]
    assert set(fraud["fraud_type"].unique()) == {"velocity", "amount", "geo", "night"}
    assert fraud["timestamp"].notna().all()


def test_legitimate_rows_have_no_fraud_type(tiny_frame):
    legit = tiny_frame[tiny_frame["is_fraud"] == 0]
    assert legit["fraud_type"].isna().all()


def test_fraud_budget_matches_requested_rate():
    assert fraud_budget(1000, 0.005) == 5
    assert fraud_budget(1000, 0.0) == 0
    with pytest.raises(ValueError):
        fraud_budget(10, 1.5)


def test_event_allocation_accounts_for_burst_rows():
    events = allocate_fraud_events(110, {"velocity": 0.5, "amount": 0.5})
    rows = expected_rows_per_event()
    produced = events["velocity"] * rows["velocity"] + events["amount"] * rows["amount"]
    assert abs(produced - 110) < 10
    assert events["velocity"] < events["amount"]


def test_allocate_counts_sums_to_budget():
    counts = allocate_fraud_counts(10, {"velocity": 0.4, "amount": 0.3, "geo": 0.2, "night": 0.1})
    assert sum(counts.values()) == 10


def test_validation_rejects_unsorted_and_negative(tmp_path):
    frame = generate_transactions(load_config("config.yaml", {"data": {"n_cards": 20, "n_days": 5}}))
    bad = frame.copy()
    bad.loc[bad.index[-1], "amount"] = -5
    with pytest.raises(ValueError, match="positive"):
        validate_transactions(bad)
    shuffled = frame.sort_values("timestamp", ascending=False).reset_index(drop=True)
    with pytest.raises(ValueError, match="sorted"):
        validate_transactions(shuffled)
