"""Per-prediction additive attribution computed from the booster itself.

For a gradient boosted tree the raw margin is ``base_margin + lr * sum(trees)``.
Inside one tree, walking from the root to the leaf visits a chain of splits; the
change in the leaf value at each split belongs to the feature that split on.
Summing those deltas over the visited path and then over all trees gives an
exact additive decomposition of the model output:

    raw_margin(x) = base + sum_f phi_f(x)      (no approximation, no sampling)

This is the path-dependent variant of tree SHAP without interaction terms. It
answers "how much did this feature move the log-odds for *this* transaction",
which is the question a fraud analyst actually asks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier


def initial_estimate(model: GradientBoostingClassifier) -> float:
    """Raw prediction of the booster's init estimator (log-odds of the prior)."""
    if not hasattr(model, "estimators_"):
        raise ValueError("model is not fitted")
    init = getattr(model, "init_", None)
    prior = getattr(init, "class_prior_", None)
    if prior is not None and len(prior) == 2:
        p1 = float(prior[1])
        p0 = float(prior[0])
        if p1 <= 0.0 or p0 <= 0.0:
            raise ValueError("degenerate class prior; cannot compute base margin")
        return float(np.log(p1 / p0))
    return 0.0
