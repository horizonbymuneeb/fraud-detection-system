"""Ranking metrics are validated against hand-built score lists."""

import numpy as np
import pandas as pd
import pytest

from src.fraud.evaluation import (
    average_precision,
    calibration_table,
    confusion_counts,
    per_fraud_type_recall,
    precision_at_k,
    recall_at_fp_budget,
    recall_at_k,
    roc_auc,
    summary_report,
    threshold_at_fp_budget,
    top_fraction_lift,
)

LABELS = np.array([0, 1, 0, 1])
SCORES = np.array([0.9, 0.8, 0.7, 0.6])


def test_precision_and_recall_at_k():
    assert precision_at_k(LABELS, SCORES, 2) == 0.5
    assert precision_at_k(LABELS, SCORES, 4) == 0.5
    assert recall_at_k(LABELS, SCORES, 2) == 0.5
    assert recall_at_k(LABELS, SCORES, 4) == 1.0
    with pytest.raises(ValueError):
        precision_at_k(LABELS, SCORES, 0)
    with pytest.raises(ValueError):
        recall_at_k(np.zeros(4, dtype=int), SCORES, 2)


def test_threshold_at_fixed_fp_budget():
    labels = np.array([1, 0, 0])
    scores = np.array([0.9, 0.8, 0.7])
    assert threshold_at_fp_budget(labels, scores, 1) == 0.8
    assert threshold_at_fp_budget(labels, scores, 0) == 0.9
    report = recall_at_fp_budget(labels, scores, 1)
    assert report["recall"] == 1.0
    assert report["precision"] == 0.5
    assert report["false_positives"] == 1
    with pytest.raises(ValueError):
        threshold_at_fp_budget(labels, scores, -1)


def test_roc_auc_and_average_precision():
    assert roc_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc_auc(LABELS, SCORES) == pytest.approx(0.25)
    assert roc_auc(LABELS, 1.0 - SCORES) == pytest.approx(0.75)
    assert average_precision(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    with pytest.raises(ValueError):
        roc_auc(np.zeros(4, dtype=int), SCORES)
    with pytest.raises(ValueError):
        average_precision(np.zeros(4, dtype=int), SCORES)
