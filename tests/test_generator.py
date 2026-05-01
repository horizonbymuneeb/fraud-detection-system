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
