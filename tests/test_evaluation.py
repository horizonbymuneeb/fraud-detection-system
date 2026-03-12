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
