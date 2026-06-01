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


def collect_metrics(
    test_report: dict[str, object],
    threshold: float,
    fraud_types: pd.Series,
    y_test: np.ndarray,
    scores: np.ndarray,
) -> dict[str, object]:
    """Attach per-pattern recall to the summary report."""
    report = dict(test_report)
    report["per_fraud_type_recall"] = per_fraud_type_recall(fraud_types, y_test, scores, threshold)
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the full training path and print a compact report."""
    args = build_argument_parser().parse_args(argv)
    cfg = load_config(args.config)
    print(f"config: {describe_config(cfg)}")

    frame = load_or_generate(cfg, args.data)
    summary = validate_transactions(frame)
    print(f"data: {summary['transactions']} rows, {summary['fraud']} fraud "
          f"({summary['fraud_rate']:.3%}), types={summary['fraud_by_type']}")

    if not args.skip_leak_check:
        # Full-frame feature rebuild vs truncated prefixes: fails if any feature
        # of a row depends on a transaction that happened after it.
        assert_no_future_leakage(frame, cfg, sample_rows=3)
        print("leak check: passed")

    datasets = prepare_datasets(frame, cfg)
    matrix = cost_matrix_from_config(cfg)
    tuning = tune_on_validation(
        datasets["X_train"], datasets["y_train"], datasets["amount_train"], cfg, matrix
    )
    threshold = float(tuning["threshold"])
    print(f"threshold tuning: {json.dumps(tuning, default=float)[:200]}")

    result = fit_model(datasets["X_train"], datasets["y_train"], cfg)
    scores = predict_scores(result.model, datasets["X_test"])
    report = summary_report(
        datasets["y_test"], scores, threshold, fp_budget=int(0.001 * len(datasets["y_test"]))
    )
    report = collect_metrics(
        report, threshold, datasets["fraud_types_test"], datasets["y_test"], scores
    )
    report["model"] = model_summary(result)
    report["tuning"] = tuning
    report["threshold_metrics_at_half"] = threshold_metrics(
        datasets["y_test"], scores, 0.5
    )

    rules_raw = rule_score(datasets["X_test"])
    rule_threshold = tune_rule_threshold(datasets["y_train"], rule_score(datasets["X_train"])) \
        if datasets["y_train"].sum() else 2.0
    report["rule_baseline"] = evaluate_rules(datasets["y_test"], rules_raw, float(rule_threshold))

    artifact_dir = Path(args.artifacts)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    save_model(
        result,
        artifact_dir / "model.pkl",
        metadata={"threshold": threshold, "config": args.config, "leak_check": not args.skip_leak_check},
    )
    save_manifest(artifact_dir / "feature_manifest.json", cfg)
    importances = feature_importances(result.model, result.feature_names)
    report["top_features"] = importances.head(10).to_dict()
    dump_json(artifact_dir / "metrics.json", report)
    attributor = PathAttributor(result.model, result.feature_names)
    explanations = attributor.explain(datasets["X_test"])
    save_explanations(artifact_dir / "explanations.json", explanations, sample=5)
    print(
        f"test: n={report['n']} positives={report['positives']} "
        f"precision={report['precision']:.3f} recall={report['recall']:.3f} "
        f"threshold={threshold:.4f}"
    )
    print(f"artifacts: {artifact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
