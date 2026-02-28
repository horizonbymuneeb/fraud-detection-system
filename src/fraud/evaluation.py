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


def recall_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Share of all frauds captured by the top ``k`` scores."""
    labels, scores = _validate(labels, scores)
    if k < 1:
        raise ValueError("k must be >= 1")
    positives = int(labels.sum())
    if positives == 0:
        raise ValueError("no positives to recall")
    k = min(k, len(labels))
    top = np.argsort(-scores, kind="mergesort")[:k]
    return float(labels[top].sum() / positives)


def threshold_at_fp_budget(labels: np.ndarray, scores: np.ndarray, fp_budget: int) -> float:
    """Highest threshold whose flagged count (all of it false positives) fits."""
    labels, scores = _validate(labels, scores)
    if fp_budget < 0:
        raise ValueError("fp_budget must be non-negative")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    flagged = 0
    false_positives = 0
    threshold = float(np.nextafter(sorted_scores.max(), np.inf))
    for score, label in zip(sorted_scores, sorted_labels):
        next_fp = false_positives + (1 - int(label))
        if next_fp > fp_budget:
            break
        flagged += 1
        false_positives = next_fp
        threshold = float(score)
    if flagged == 0:
        return float(np.nextafter(sorted_scores.max(), np.inf))
    return threshold


def recall_at_fp_budget(labels: np.ndarray, scores: np.ndarray, fp_budget: int) -> dict[str, float]:
    """Recall and precision achievable without exceeding ``fp_budget`` alerts."""
    labels, scores = _validate(labels, scores)
    threshold = threshold_at_fp_budget(labels, scores, fp_budget)
    flagged = scores >= threshold
    tp = int((flagged & (labels == 1)).sum())
    fp = int((flagged & (labels == 0)).sum())
    positives = int(labels.sum())
    return {
        "threshold": threshold,
        "recall": float(tp / positives) if positives else 0.0,
        "precision": float(tp / max(tp + fp, 1)),
        "alerts": int(flagged.sum()),
        "false_positives": fp,
    }


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve; raises when only one class is present."""
    labels, scores = _validate(labels, scores)
    if labels.min() == labels.max():
        raise ValueError("ROC AUC needs both classes")
    return float(roc_auc_score(labels, scores))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve (sklearn's average precision)."""
    labels, scores = _validate(labels, scores)
    if labels.sum() == 0:
        raise ValueError("average precision needs at least one positive")
    return float(average_precision_score(labels, scores))


def confusion_counts(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, int]:
    """TP/FP/TN/FN at one probability threshold."""
    labels, scores = _validate(labels, scores)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    flagged = scores >= threshold
    return {
        "tp": int((flagged & (labels == 1)).sum()),
        "fp": int((flagged & (labels == 0)).sum()),
        "tn": int((~flagged & (labels == 0)).sum()),
        "fn": int((~flagged & (labels == 1)).sum()),
    }


def per_fraud_type_recall(
    fraud_types: pd.Series, labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float]:
    """Recall per injected pattern; legitimate rows keep ``None``."""
    labels, scores = _validate(labels, scores)
    types = fraud_types.reset_index(drop=True)
    if len(types) != len(labels):
        raise ValueError("fraud_types must align with labels")
    flagged = scores >= threshold
    report: dict[str, float] = {}
    for fraud_type in sorted(types.dropna().unique()):
        mask = (types == fraud_type).to_numpy()
        total = int(mask.sum())
        if total:
            report[str(fraud_type)] = float(flagged[mask].sum() / total)
    return report
