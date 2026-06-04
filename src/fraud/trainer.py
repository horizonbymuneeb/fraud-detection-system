"""Model training: scale_pos_weight, temporal validation and persistence.

The positive class is ~0.5% of the stream, so every weight decision matters.
Rather than oversampling (which inflates training time and makes predicted
probabilities lie), the model gets ``scale_pos_weight = negatives / positives``
and the decision threshold is chosen later from the cost matrix.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from .config import Config


@dataclass(frozen=True)
class TrainResult:
    """A fitted model plus the facts needed to reproduce its predictions."""

    model: GradientBoostingClassifier
    feature_names: list[str]
    scale_pos_weight: float
    n_train: int
    n_positive: int
    train_prior: float


def scale_pos_weight(labels: np.ndarray) -> float:
    """Weight applied to the minority class: negatives / positives."""
    labels = np.asarray(labels).astype(int)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0:
        raise ValueError("training labels contain no positive examples")
    return float(negatives / positives)
