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


def load_or_generate(cfg: Config, data_path: str | None) -> pd.DataFrame:
    """Read an existing stream or generate one from the config."""
    if data_path:
        frame = read_transactions(data_path)
        validate_transactions(frame)
        return frame
    csv_path = Path(cfg.data.output_dir) / "transactions.csv"
    if csv_path.exists():
        frame = read_transactions(csv_path)
        validate_transactions(frame)
        return frame
    frame, _path = generate_and_save(cfg)
    return frame


def tune_on_validation(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    amount_train: np.ndarray,
    cfg: Config,
    matrix: CostMatrix,
) -> dict[str, object]:
    """Fit on the first part of train, tune the cost threshold on the last part."""
    fit_idx, tune_idx = temporal_holdout(len(X_train), cfg.model.validation_fraction)
    if len(np.unique(y_train[fit_idx])) < 2:
        return {"threshold": 0.5, "note": "training window too small for tuning; fell back to 0.5"}
    holdout_model = fit_model(X_train.iloc[fit_idx], y_train[fit_idx], cfg)
    holdout_scores = predict_scores(holdout_model.model, X_train.iloc[tune_idx])
    report = tune_threshold(
        amount_train[tune_idx], y_train[tune_idx], holdout_scores, matrix
    )
    report["tuned_on"] = int(len(tune_idx))
    return report
