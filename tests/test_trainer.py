"""Imbalance handling, temporal holdout and persistence."""

import numpy as np
import pandas as pd
import pytest

from src.fraud.trainer import (
    build_estimator,
    class_balance,
    feature_importances,
    fit_model,
    grouped_importances,
    load_json,
    load_model,
    model_summary,
    predict_labels,
    predict_scores,
    save_model,
    scale_pos_weight,
    temporal_holdout,
    threshold_metrics,
)


def test_scale_pos_weight_is_negatives_over_positives():
    labels = np.array([1] + [0] * 9)
    assert scale_pos_weight(labels) == 9.0
    with pytest.raises(ValueError, match="no positive"):
        scale_pos_weight(np.zeros(5, dtype=int))


def test_class_balance_counts():
    balance = class_balance(np.array([0, 0, 0, 1]))
    assert balance == {"n": 4, "positives": 1, "negatives": 3, "prior": 0.25}


def test_build_estimator_disables_early_stopping_for_small_data(tiny_config):
    small = build_estimator(tiny_config, 100)
    large = build_estimator(tiny_config, 5000)
    assert small.get_params()["n_estimators"] == tiny_config.model.n_estimators
    assert large.get_params()["n_iter_no_change"] == tiny_config.model.n_iter_no_change


def test_fit_predict_and_determinism(tiny_config, tiny_datasets):
    result = fit_model(tiny_datasets["X_train"], tiny_datasets["y_train"], tiny_config)
    scores = predict_scores(result.model, tiny_datasets["X_test"])
    again = fit_model(tiny_datasets["X_train"], tiny_datasets["y_train"], tiny_config)
    assert np.allclose(scores, predict_scores(again.model, tiny_datasets["X_test"]))
    assert ((scores >= 0) & (scores <= 1)).all()
    assert result.n_positive == int(tiny_datasets["y_train"].sum())


def test_fit_rejects_single_class_and_short_holdout(tiny_config, tiny_datasets):
    with pytest.raises(ValueError, match="both classes"):
        fit_model(tiny_datasets["X_train"], np.zeros(len(tiny_datasets["X_train"]), dtype=int), tiny_config)
    with pytest.raises(ValueError):
        temporal_holdout(3, 0.2)


def test_temporal_holdout_splits_at_the_end():
    fit_idx, tune_idx = temporal_holdout(100, 0.2)
    assert fit_idx.max() < tune_idx.min()
    assert len(fit_idx) + len(tune_idx) == 100
    assert len(tune_idx) == 20


def test_threshold_metrics_are_exact():
    y = np.array([1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.4, 0.8, 0.2, 0.1])
    metrics = threshold_metrics(y, scores, 0.5)
    assert metrics == {
        "threshold": 0.5,
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "tn": 2,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }
    with pytest.raises(ValueError):
        predict_labels(scores, 0.0)
    with pytest.raises(ValueError):
        threshold_metrics(y, scores, 1.5)


def test_importances_align_with_columns(tiny_config, tiny_datasets, fitted):
    importances = feature_importances(fitted.model, fitted.feature_names)
    assert set(importances.index) == set(fitted.feature_names)
    assert importances.is_monotonic_decreasing
    assert importances.sum() == pytest.approx(1.0, abs=1e-9)
    grouped = grouped_importances(importances, lambda name: name.split("_")[0])
    assert grouped.sum() == pytest.approx(1.0, abs=1e-9)


def test_save_and_load_round_trip(tmp_path, fitted):
    path = save_model(fitted, tmp_path / "model.pkl", metadata={"threshold": 0.3})
    bundle = load_model(path)
    assert bundle["metadata"]["threshold"] == 0.3
    assert bundle["feature_names"] == fitted.feature_names
    assert model_summary(fitted)["n_features"] == len(fitted.feature_names)
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "absent.pkl")
