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


def run_evaluation(
    bundle: dict[str, object], cfg: Config, data_path: str, manifest_path: str | Path
) -> dict[str, object]:
    """Recompute the test-split metrics for a saved model."""
    frame = read_transactions(data_path)
    validate_transactions(frame)
    datasets = prepare_datasets(frame, cfg)
    model = bundle["model"]
    scores = predict_scores(model, datasets["X_test"])
    matrix = cost_matrix_from_config(cfg)
    threshold = float(bundle.get("metadata", {}).get("threshold", 0.5))
    report = summary_report(datasets["y_test"], scores, threshold, fp_budget=int(0.001 * len(scores)))
    report["threshold_curve"] = compare_thresholds(
        datasets["amount_test"], datasets["y_test"], scores, matrix
    ).to_dict(orient="records")
    report["calibration"] = calibration_table(datasets["y_test"], scores).to_dict(orient="records")
    report["manifest"] = load_manifest(manifest_path).get("n_features")
    return report


def main(argv: list[str] | None = None) -> int:
    """Load, evaluate, write the JSON report and print the headline numbers."""
    args = build_argument_parser().parse_args(argv)
    cfg = load_config(args.config)
    bundle = load_bundle(args.model)
    report = run_evaluation(bundle, cfg, args.data, Path(args.model).parent / "feature_manifest.json")
    dump_json(args.output, report)
    print(json.dumps({k: report[k] for k in ("n", "positives", "precision", "recall", "threshold") if k in report}, indent=2))
    print(f"report: {args.output}")
    return 0
