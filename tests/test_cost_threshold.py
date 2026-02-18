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
