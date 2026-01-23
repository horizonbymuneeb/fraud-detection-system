"""End-to-end: generate data, train, persist artifacts, evaluate the bundle."""

import json

import yaml

from src.fraud.config import load_config, read_yaml
from src.fraud.evaluate import main as evaluate_main
from src.fraud.generator import generate_transactions, write_transactions
from src.fraud.train import main as train_main


def write_small_config(path):
    raw = read_yaml("config.yaml")
    raw["seed"] = 11
    raw["data"].update({"n_cards": 80, "n_days": 14, "base_fraud_rate": 0.04})
    raw["split"].update({"train_fraction": 0.7, "boundary_gap_hours": 12})
    raw["model"].update(
        {"n_estimators": 15, "max_depth": 2, "min_samples_leaf": 3, "n_iter_no_change": 5}
    )
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_train_then_evaluate(tmp_path):
    config_path = write_small_config(tmp_path / "config.yaml")
    cfg = load_config(config_path)
    frame = generate_transactions(cfg)
    data_path = write_transactions(frame, tmp_path / "transactions.csv")
    artifacts = tmp_path / "artifacts"

    assert train_main([
        "--config", str(config_path),
        "--data", str(data_path),
        "--artifacts", str(artifacts),
    ]) == 0
    for name in ("model.pkl", "feature_manifest.json", "metrics.json", "explanations.json"):
        assert (artifacts / name).exists(), name
    metrics = json.loads((artifacts / "metrics.json").read_text())
    assert metrics["n"] > 0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert "rule_baseline" in metrics
    assert "per_fraud_type_recall" in metrics
    assert len(metrics["top_features"]) == 10

    report_path = tmp_path / "evaluation.json"
    assert evaluate_main([
        "--config", str(config_path),
        "--model", str(artifacts / "model.pkl"),
        "--data", str(data_path),
        "--output", str(report_path),
    ]) == 0
    report = json.loads(report_path.read_text())
    assert report["n"] > 0
    assert report["threshold_curve"]
    assert report["calibration"]
