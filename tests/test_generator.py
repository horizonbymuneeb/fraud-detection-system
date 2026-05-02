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
