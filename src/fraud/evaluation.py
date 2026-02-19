"""Ranking metrics that match how a fraud team runs the queue.

A fraud model is judged by what an analyst can act on: how many true frauds sit
in the top N alerts (precision@k), how much recall is available at a fixed
false-positive budget, and whether the score is monotone enough for a threshold
to be meaningful (ROC AUC, average precision).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _validate(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) != len(scores):
        raise ValueError("labels and scores must be the same length")
    if len(labels) == 0:
        raise ValueError("empty input")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be binary")
    if np.isnan(scores).any():
        raise ValueError("scores contain NaN")
    return labels, scores


def precision_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Fraction of the top ``k`` scores that are fraud."""
    labels, scores = _validate(labels, scores)
    if k < 1:
        raise ValueError("k must be >= 1")
    k = min(k, len(labels))
    top = np.argsort(-scores, kind="mergesort")[:k]
    return float(labels[top].mean())
