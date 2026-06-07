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


def class_balance(labels: np.ndarray) -> dict[str, float]:
    """Counts and prior for the training labels."""
    labels = np.asarray(labels).astype(int)
    positives = int(labels.sum())
    return {
        "n": int(len(labels)),
        "positives": positives,
        "negatives": int(len(labels) - positives),
        "prior": float(positives / len(labels)) if len(labels) else 0.0,
    }


def build_estimator(cfg: Config, n_rows: int) -> GradientBoostingClassifier:
    """Create the GBDT, disabling early stopping when data is too small for it."""
    params: dict[str, Any] = dict(
        n_estimators=cfg.model.n_estimators,
        learning_rate=cfg.model.learning_rate,
        max_depth=cfg.model.max_depth,
        min_samples_leaf=cfg.model.min_samples_leaf,
        subsample=cfg.model.subsample,
        max_features=cfg.model.max_features,
        random_state=cfg.model.random_state,
    )
    # Early stopping needs two populated classes on both sides of its internal
    # split; with a small training window that is not guaranteed.
    if n_rows >= 400 and cfg.model.n_iter_no_change > 0:
        params["n_iter_no_change"] = cfg.model.n_iter_no_change
        params["validation_fraction"] = cfg.model.validation_fraction
    return GradientBoostingClassifier(**params)


def fit_model(X: pd.DataFrame, y: np.ndarray, cfg: Config) -> TrainResult:
    """Fit the classifier on already-imputed, already-clipped features."""
    if len(X) != len(y):
        raise ValueError("X and y disagree on row count")
    labels = np.asarray(y).astype(int)
    if labels.min() == labels.max():
        raise ValueError("training labels must contain both classes")
    weight = scale_pos_weight(labels)
    model = build_estimator(cfg, len(X))
    model.fit(X, labels, sample_weight=np.where(labels == 1, weight, 1.0))
    return TrainResult(
        model=model,
        feature_names=list(X.columns),
        scale_pos_weight=weight,
        n_train=int(len(X)),
        n_positive=int(labels.sum()),
        train_prior=float(labels.mean()),
    )
