"""Training entry point: generate or load data, fit, tune the threshold, save.

    python -m src.fraud.train --config config.yaml

The threshold is tuned on the tail of the training window (never on the test
window), then the model is refitted on the full training window and evaluated
once on the held-out future.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .attribution import PathAttributor, save_explanations
from .config import Config, describe_config, load_config
from .cost_threshold import CostMatrix, cost_matrix_from_config, tune_threshold
from .evaluation import per_fraud_type_recall, summary_report
from .feature_pipeline import (
    assert_no_future_leakage,
    prepare_datasets,
    save_manifest,
)
from .generator import generate_and_save, read_transactions, validate_transactions
from .rules import rule_score, tune_rule_threshold, evaluate_rules
from .trainer import (
    dump_json,
    feature_importances,
    fit_model,
    model_summary,
    predict_scores,
    save_model,
    temporal_holdout,
    threshold_metrics,
)

DEFAULT_ARTIFACT_DIR = "artifacts"


def build_argument_parser() -> argparse.ArgumentParser:
    """CLI surface: config path, optional data path, artifact directory."""
    parser = argparse.ArgumentParser(description="Train the card-fraud detector")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--data", default=None, help="existing transactions.csv (skips generation)")
    parser.add_argument("--artifacts", default=DEFAULT_ARTIFACT_DIR, help="output directory")
    parser.add_argument("--skip-leak-check", action="store_true", help="skip the leakage assertion")
    return parser
