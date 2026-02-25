"""Cost-matrix selection is checked against hand-computed losses."""

import numpy as np
import pandas as pd
import pytest

from src.fraud.cost_threshold import (
    CostMatrix,
    break_even_probability,
    compare_thresholds,
    cost_matrix_from_config,
    expected_loss,
    loss_curve,
    optimal_threshold,
    tune_threshold,
)

AMOUNTS = np.array([100.0, 200.0])
LABELS = np.array([1, 0])
SCORES = np.array([0.9, 0.2])
MATRIX = CostMatrix(false_positive_cost=10.0, fraud_loss_cost=50.0, amount_loss_multiplier=1.0)


def test_cost_matrix_validation_and_missed_cost():
    with pytest.raises(ValueError):
        CostMatrix(false_positive_cost=0)
    with pytest.raises(ValueError):
        CostMatrix(fraud_loss_cost=-1)
    with pytest.raises(ValueError):
        CostMatrix(amount_loss_multiplier=-0.5)
    assert MATRIX.missed_fraud_cost(100.0) == 150.0


def test_expected_loss_matches_hand_calculation():
    assert expected_loss(AMOUNTS, LABELS, SCORES, 0.5, MATRIX) == 0.0
    assert expected_loss(AMOUNTS, LABELS, SCORES, 0.1, MATRIX) == 10.0
    assert expected_loss(AMOUNTS, LABELS, SCORES, 0.95, MATRIX) == 150.0
    with pytest.raises(ValueError):
        expected_loss(AMOUNTS, LABELS, SCORES, 1.5, MATRIX)


def test_loss_curve_thresholds_and_optimum():
    curve = loss_curve(AMOUNTS, LABELS, SCORES, MATRIX)
    assert curve["threshold"].tolist() == [0.2, 0.9]
    assert curve["loss"].tolist() == [10.0, 0.0]
    threshold, loss = optimal_threshold(curve)
    assert threshold == 0.9 and loss == 0.0
    with pytest.raises(ValueError):
        optimal_threshold(pd.DataFrame({"threshold": [], "loss": []}))


def test_loss_curve_rejects_empty_input():
    with pytest.raises(ValueError):
        loss_curve(np.array([]), np.array([]), np.array([]), MATRIX)


def test_break_even_probability():
    assert break_even_probability(MATRIX) == pytest.approx(10.0 / 60.0)
    assert break_even_probability(CostMatrix(1.0, 1.0, 0.0)) == 0.5


def test_tune_threshold_reports_saving_against_half():
    report = tune_threshold(AMOUNTS, LABELS, SCORES, MATRIX)
    assert report["threshold"] == 0.9
    assert report["expected_loss"] == 0.0
    assert report["loss_at_half"] == 0.0
    assert report["saving_vs_half"] == 0.0
    assert 0.0 < report["break_even"] < 1.0


def test_compare_thresholds_frame():
    table = compare_thresholds(AMOUNTS, LABELS, SCORES, MATRIX, thresholds=(0.1, 0.5))
    assert table["threshold"].tolist() == [0.1, 0.5]
    assert table["loss"].tolist() == [10.0, 0.0]
    assert table["flagged"].tolist() == [2, 1]


def test_cost_matrix_from_config():
    from src.fraud.config import load_config

    cfg = load_config("config.yaml")
    matrix = cost_matrix_from_config(cfg)
    assert matrix.false_positive_cost == cfg.cost.false_positive_cost
    assert matrix.fraud_loss_cost == cfg.cost.fraud_loss_cost
