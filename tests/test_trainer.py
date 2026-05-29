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
