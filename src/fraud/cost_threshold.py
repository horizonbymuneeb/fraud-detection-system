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
