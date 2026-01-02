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
