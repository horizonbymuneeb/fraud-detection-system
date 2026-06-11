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


def predict_scores(model: GradientBoostingClassifier, X: pd.DataFrame) -> np.ndarray:
    """Fraud probability for every row."""
    if len(X) == 0:
        return np.zeros(0, dtype=np.float64)
    return model.predict_proba(X)[:, 1]


def predict_labels(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean decisions at ``threshold`` (inclusive)."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be in (0, 1)")
    return np.asarray(scores) >= threshold


def threshold_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Confusion-derived metrics at one threshold."""
    labels = np.asarray(y).astype(int)
    preds = predict_labels(scores, threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def temporal_holdout(n_rows: int, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Positional split of the training window: first part fit, last part tune."""
    if not 0.01 <= fraction < 0.9:
        raise ValueError("fraction must be in [0.01, 0.9)")
    if n_rows < 4:
        raise ValueError("need at least four rows for a temporal holdout")
    cut = int(n_rows * (1.0 - fraction))
    cut = min(max(cut, 1), n_rows - 1)
    fit_idx = np.arange(0, cut)
    tune_idx = np.arange(cut, n_rows)
    return fit_idx, tune_idx


def feature_importances(model: GradientBoostingClassifier, columns: list[str]) -> pd.Series:
    """Mean decrease in impurity, sorted descending."""
    if len(columns) != len(model.feature_importances_):
        raise ValueError("column count does not match the fitted model")
    series = pd.Series(model.feature_importances_, index=columns, dtype=np.float64)
    return series.sort_values(ascending=False)


def grouped_importances(importances: pd.Series, group_of: Any) -> pd.Series:
    """Sum importances by feature group (velocity / behaviour / context)."""
    grouped: dict[str, float] = {}
    for name, value in importances.items():
        grouped[group_of(name)] = grouped.get(group_of(name), 0.0) + float(value)
    return pd.Series(grouped, dtype=np.float64).sort_values(ascending=False)


def model_summary(result: TrainResult) -> dict[str, object]:
    """Facts about a fitted model, logged by the training CLI."""
    model = result.model
    return {
        "n_train": result.n_train,
        "n_positive": result.n_positive,
        "scale_pos_weight": result.scale_pos_weight,
        "train_prior": result.train_prior,
        "n_estimators": int(model.n_estimators_),
        "n_features": len(result.feature_names),
    }


def save_model(result: TrainResult, path: str | Path, metadata: dict[str, object] | None = None) -> Path:
    """Pickle the model and write the side-car metadata JSON next to it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(
            {
                "model": result.model,
                "feature_names": result.feature_names,
                "metadata": metadata or {},
                "train": model_summary(result),
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return path


def load_model(path: str | Path) -> dict[str, object]:
    """Load a bundle written by :func:`save_model`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"model not found: {path}")
    with path.open("rb") as handle:
        bundle = pickle.load(handle)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"{path} does not contain a model bundle")
    return bundle


def dump_json(path: str | Path, payload: dict[str, object]) -> Path:
    """Write a metrics/report mapping as JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return path


def load_json(path: str | Path) -> dict[str, object]:
    """Read a JSON mapping, failing loudly when the file is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"json file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
