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
