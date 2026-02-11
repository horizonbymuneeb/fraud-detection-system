"""Cost-matrix threshold selection.

A 0.5 probability threshold silently assumes a false positive and a missed
fraud cost the same. They do not: reviewing a flagged transaction costs minutes
of an analyst's time, while a missed fraud costs the transaction amount plus
chargeback handling. This module turns that trade-off into one number, the
expected loss per transaction, and picks the threshold that minimises it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostMatrix:
    """Per-event costs, in the same currency as the transaction amounts."""

    false_positive_cost: float = 3.0
    fraud_loss_cost: float = 250.0
    amount_loss_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.false_positive_cost <= 0:
            raise ValueError("false_positive_cost must be positive")
        if self.fraud_loss_cost <= 0:
            raise ValueError("fraud_loss_cost must be positive")
        if self.amount_loss_multiplier < 0:
            raise ValueError("amount_loss_multiplier must be non-negative")

    def missed_fraud_cost(self, amount: float | np.ndarray) -> np.ndarray:
        """Cost of letting a fraud of this amount through (fee + loss)."""
        return self.fraud_loss_cost + self.amount_loss_multiplier * np.asarray(amount, dtype=np.float64)


def cost_matrix_from_config(cfg: object) -> CostMatrix:
    """Build a :class:`CostMatrix` from a loaded ``Config``."""
    cost = getattr(cfg, "cost")
    return CostMatrix(
        false_positive_cost=float(cost.false_positive_cost),
        fraud_loss_cost=float(cost.fraud_loss_cost),
        amount_loss_multiplier=float(cost.amount_loss_multiplier),
    )


def expected_loss(
    amounts: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    matrix: CostMatrix,
) -> float:
    """Total expected loss at one threshold.

    ``sum over false positives of fp_cost + sum over missed frauds of
    (fraud_loss_cost + amount * multiplier)``.
    """
    amounts = np.asarray(amounts, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    if not (len(amounts) == len(labels) == len(scores)):
        raise ValueError("amounts, labels and scores must be the same length")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    flagged = scores >= threshold
    false_positives = int((flagged & (labels == 0)).sum())
    missed = (~flagged) & (labels == 1)
    return float(
        false_positives * matrix.false_positive_cost
        + matrix.missed_fraud_cost(amounts[missed]).sum()
    )


def loss_curve(
    amounts: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    matrix: CostMatrix,
) -> pd.DataFrame:
    """Expected loss for every distinct threshold in the score distribution."""
    amounts = np.asarray(amounts, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    if len(amounts) == 0:
        raise ValueError("cannot build a loss curve from zero rows")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    sorted_amounts = amounts[order]
    positive = sorted_labels == 1
    cumulative_pos = np.cumsum(positive)
    cumulative_pos_amount = np.cumsum(sorted_amounts * positive)
    total_positive = int(positive.sum())
    n = len(sorted_scores)
    thresholds = np.unique(sorted_scores)
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        idx = int(np.searchsorted(sorted_scores, threshold, side="left"))
        positives_below = int(cumulative_pos[idx - 1]) if idx > 0 else 0
        amount_below = float(cumulative_pos_amount[idx - 1]) if idx > 0 else 0.0
        tp = total_positive - positives_below
        flagged = n - idx
        fp = flagged - tp
        fn = positives_below
        loss = (
            fp * matrix.false_positive_cost
            + fn * matrix.fraud_loss_cost
            + amount_below * matrix.amount_loss_multiplier
        )
        rows.append(
            {
                "threshold": float(threshold),
                "fp": float(fp),
                "fn": float(fn),
                "loss": float(loss),
                "loss_per_txn": float(loss / n),
            }
        )
    return pd.DataFrame(rows)


def optimal_threshold(curve: pd.DataFrame) -> tuple[float, float]:
    """Threshold with the lowest expected loss; ties break to the higher one."""
    if curve.empty:
        raise ValueError("loss curve is empty")
    best = curve.loc[curve["loss"].idxmin()]
    return float(best["threshold"]), float(best["loss"])
