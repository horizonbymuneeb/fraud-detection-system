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


def base_margin(model: GradientBoostingClassifier) -> float:
    """Constant offset shared by every prediction.

    That is the init raw prediction plus the (learning-rate-scaled) value stored
    at the root of every tree: each row passes through every root, so the root
    values are a constant that belongs with the base, not with any feature.
    """
    if not hasattr(model, "estimators_"):
        raise ValueError("model is not fitted")
    root_sum = 0.0
    for i in range(model.estimators_.shape[0]):
        tree = model.estimators_[i, 0]
        root_sum += float(np.ravel(tree.tree_.value)[0])
    return float(initial_estimate(model) + model.learning_rate * root_sum)


def extract_stages(model: GradientBoostingClassifier) -> tuple[float, list[object], float]:
    """Return ``(learning_rate, trees, base_margin)`` for a fitted model."""
    if not hasattr(model, "estimators_"):
        raise ValueError("model is not fitted")
    trees = [model.estimators_[i, 0] for i in range(model.estimators_.shape[0])]
    return float(model.learning_rate), trees, base_margin(model)


def tree_path_contribution(tree: object, row: np.ndarray, n_features: int, out: np.ndarray) -> float:
    """Add this tree's path deltas to ``out`` and return the leaf value."""
    node = 0
    values = np.ravel(tree.tree_.value)
    while tree.tree_.children_left[node] != -1:
        feature = int(tree.tree_.feature[node])
        threshold = float(tree.tree_.threshold[node])
        child = (
            tree.tree_.children_left[node]
            if row[feature] <= threshold
            else tree.tree_.children_right[node]
        )
        out[feature] += float(values[child] - values[node])
        node = int(child)
    return float(values[node])


def contributions_for_row(
    model: GradientBoostingClassifier, row: np.ndarray, n_features: int | None = None
) -> np.ndarray:
    """Exact per-feature log-odds contributions for one row."""
    row = np.asarray(row, dtype=np.float64).ravel()
    n_features = n_features or len(row)
    learning_rate, trees, _base = extract_stages(model)
    contributions = np.zeros(n_features, dtype=np.float64)
    total = 0.0
    for tree in trees:
        total += tree_path_contribution(tree, row, n_features, contributions)
    return contributions * learning_rate


def contribution_matrix(
    model: GradientBoostingClassifier, X: pd.DataFrame | np.ndarray
) -> tuple[float, np.ndarray]:
    """Contributions for every row; returns ``(base_margin, matrix)``."""
    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("X must be two-dimensional")
    learning_rate, trees, base = extract_stages(model)
    matrix = np.zeros_like(values, dtype=np.float64)
    for tree in trees:
        for i in range(values.shape[0]):
            tree_path_contribution(tree, values[i], values.shape[1], matrix[i])
    return base, matrix * learning_rate


def raw_margin_from_contributions(base: float, contributions: np.ndarray) -> np.ndarray:
    """Reconstruct the raw margin from the additive decomposition."""
    return base + np.sum(contributions, axis=1)


def probability_from_margin(margin: float | np.ndarray) -> np.ndarray:
    """Sigmoid that maps a raw log-odds margin to a probability."""
    m = np.clip(np.asarray(margin, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-m))


def additivity_error(model: GradientBoostingClassifier, X: pd.DataFrame) -> float:
    """Largest difference between reconstructed and model raw margins."""
    base, matrix = contribution_matrix(model, X)
    reconstructed = raw_margin_from_contributions(base, matrix)
    reference = model.decision_function(X)
    return float(np.max(np.abs(reconstructed - reference)))


def top_contributors(contributions: np.ndarray, columns: list[str], k: int = 5) -> list[dict[str, object]]:
    """The ``k`` features with the largest absolute contribution, signed."""
    if k < 1:
        raise ValueError("k must be >= 1")
    order = np.argsort(-np.abs(contributions))[:k]
    return [
        {"feature": columns[i], "contribution": float(contributions[i])}
        for i in order
    ]


def grouped_contributions(contributions: np.ndarray, columns: list[str]) -> pd.DataFrame:
    """Sum contributions by feature group (velocity / behaviour / context)."""
    from .feature_pipeline import feature_group

    frame = pd.DataFrame({"feature": columns, "contribution": contributions})
    frame["group"] = [feature_group(name) for name in frame["feature"]]
    grouped = frame.groupby("group", sort=False)["contribution"].sum()
    return pd.DataFrame({"group": grouped.index, "contribution": grouped.to_numpy()})


def global_attribution(model: GradientBoostingClassifier, X: pd.DataFrame) -> pd.Series:
    """Mean absolute contribution per feature across rows, sorted."""
    _base, matrix = contribution_matrix(model, X)
    mean_abs = np.mean(np.abs(matrix), axis=0)
    return pd.Series(mean_abs, index=list(X.columns), dtype=np.float64).sort_values(ascending=False)


def explain_row(
    model: GradientBoostingClassifier,
    row: pd.Series,
    columns: list[str] | None = None,
    threshold: float = 0.5,
    top_k: int = 5,
) -> dict[str, object]:
    """Analyst-facing explanation for one transaction."""
    columns = columns or list(row.index)
    values = np.asarray(row[columns], dtype=np.float64)
    base, matrix = contribution_matrix(model, pd.DataFrame([values], columns=columns))
    contributions = matrix[0]
    margin = float(base + contributions.sum())
    probability = float(probability_from_margin(margin))
    return {
        "base_margin": float(base),
        "raw_margin": margin,
        "probability": probability,
        "decision": "block" if probability >= threshold else "allow",
        "top_features": top_contributors(contributions, columns, top_k),
        "additivity_ok": bool(abs(margin - float(model.decision_function(pd.DataFrame([values], columns=columns))[0])) < 1e-9),
    }


@dataclass(frozen=True)
class PathAttributor:
    """Reusable wrapper so a scorer does not re-walk the model per call site."""

    model: GradientBoostingClassifier
    feature_names: list[str]

    def explain(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Attribution matrix indexed like ``X`` with feature columns."""
        base, matrix = contribution_matrix(self.model, X)
        frame = pd.DataFrame(matrix, columns=self.feature_names)
        frame.insert(0, "base_margin", base)
        frame["raw_margin"] = frame["base_margin"] + frame[self.feature_names].sum(axis=1)
        frame["probability"] = probability_from_margin(frame["raw_margin"].to_numpy())
        return frame

    def additivity_error(self, X: pd.DataFrame) -> float:
        """Forward to the module-level check for this model."""
        return additivity_error(self.model, X)

    def top_for(self, X: pd.DataFrame, row_position: int = 0, k: int = 5) -> list[dict[str, object]]:
        """Top contributors for one positional row of ``X``."""
        _base, matrix = contribution_matrix(self.model, X.iloc[[row_position]])
        return top_contributors(matrix[0], self.feature_names, k)
