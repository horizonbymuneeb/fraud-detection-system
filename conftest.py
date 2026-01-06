"""Make ``src`` importable from any working directory and share fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fraud.config import load_config  # noqa: E402
from src.fraud.feature_pipeline import prepare_datasets  # noqa: E402
from src.fraud.generator import generate_transactions  # noqa: E402
from src.fraud.trainer import fit_model  # noqa: E402

REAL_CONFIG = ROOT / "config.yaml"
TINY_OVERRIDES = {
    "seed": 7,
    "data": {"n_cards": 120, "n_days": 20, "base_fraud_rate": 0.05},
    "split": {"train_fraction": 0.7, "boundary_gap_hours": 24},
    "model": {
        "n_estimators": 25,
        "max_depth": 3,
        "min_samples_leaf": 5,
        "n_iter_no_change": 10,
        "validation_fraction": 0.2,
    },
}


@pytest.fixture(scope="session")
def tiny_config():
    """Small but realistic config: same code path, tiny data volume."""
    return load_config(REAL_CONFIG, TINY_OVERRIDES)


@pytest.fixture(scope="session")
def tiny_frame(tiny_config):
    """Generated transaction stream used by most tests."""
    return generate_transactions(tiny_config)


@pytest.fixture(scope="session")
def tiny_datasets(tiny_config, tiny_frame):
    """Prepared feature matrices for the tiny stream."""
    return prepare_datasets(tiny_frame, tiny_config)


@pytest.fixture(scope="session")
def fitted(tiny_config, tiny_datasets):
    """A fitted model result for attribution/trainer tests."""
    return fit_model(tiny_datasets["X_train"], tiny_datasets["y_train"], tiny_config)
