"""Evaluate a saved model bundle on a transaction stream.

    python -m src.fraud.evaluate --config config.yaml --model artifacts/model.pkl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, load_config
from .cost_threshold import compare_thresholds, cost_matrix_from_config
from .evaluation import calibration_table, summary_report
from .feature_pipeline import load_manifest, prepare_datasets
from .generator import read_transactions, validate_transactions
from .trainer import dump_json, load_model, predict_scores


def build_argument_parser() -> argparse.ArgumentParser:
    """CLI surface for offline evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a saved fraud model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default="artifacts/model.pkl")
    parser.add_argument("--data", default="data/transactions.csv")
    parser.add_argument("--output", default="artifacts/evaluation.json")
    return parser


def load_bundle(model_path: str) -> dict[str, object]:
    """Load the model bundle and insist on the feature names being present."""
    bundle = load_model(model_path)
    if not bundle.get("feature_names"):
        raise ValueError("model bundle has no feature_names")
    return bundle
