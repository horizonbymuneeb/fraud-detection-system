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
