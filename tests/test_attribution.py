"""The additive decomposition must reconstruct the model margin exactly."""

import json

import numpy as np
import pandas as pd
import pytest

from src.fraud.attribution import (
    PathAttributor,
    additivity_error,
    base_margin,
    contribution_matrix,
    contributions_for_row,
    explain_row,
    grouped_contributions,
    probability_from_margin,
    raw_margin_from_contributions,
    save_explanations,
    top_contributors,
)


def test_additivity_is_exact_on_the_test_split(fitted, tiny_datasets):
    error = additivity_error(fitted.model, tiny_datasets["X_test"])
    assert error < 1e-9


def test_contributions_match_decision_function_row_by_row(fitted, tiny_datasets):
    base, matrix = contribution_matrix(fitted.model, tiny_datasets["X_test"].head(20))
    reconstructed = raw_margin_from_contributions(base, matrix)
    reference = fitted.model.decision_function(tiny_datasets["X_test"].head(20))
    assert np.allclose(reconstructed, reference, atol=1e-9)
    single = contributions_for_row(fitted.model, tiny_datasets["X_test"].iloc[0].to_numpy())
    assert np.allclose(single, matrix[0], atol=1e-9)


def test_probability_matches_predict_proba(fitted, tiny_datasets):
    base, matrix = contribution_matrix(fitted.model, tiny_datasets["X_test"].head(15))
    probabilities = probability_from_margin(raw_margin_from_contributions(base, matrix))
    expected = predict_proba_column(fitted, tiny_datasets["X_test"].head(15))
    assert np.allclose(probabilities, expected, atol=1e-9)


def predict_proba_column(result, X):
    return result.model.predict_proba(X)[:, 1]


def test_base_margin_is_constant(fitted, tiny_datasets):
    value = base_margin(fitted.model)
    assert isinstance(value, float)
    X = tiny_datasets["X_test"].head(5)
    recon = np.array(
        [
            raw_margin_from_contributions(value, contribution_matrix(fitted.model, X.iloc[[i]])[1])[0]
            for i in range(len(X))
        ]
    )
    assert np.allclose(recon, fitted.model.decision_function(X), atol=1e-9)


def test_top_contributors_ordering_and_grouped_sums(fitted, tiny_datasets):
    base, matrix = contribution_matrix(fitted.model, tiny_datasets["X_test"].head(10))
    top = top_contributors(matrix[0], fitted.feature_names, k=5)
    assert len(top) == 5
    magnitudes = [abs(item["contribution"]) for item in top]
    assert magnitudes == sorted(magnitudes, reverse=True)
    with pytest.raises(ValueError):
        top_contributors(matrix[0], fitted.feature_names, k=0)

    grouped = grouped_contributions(matrix[0], fitted.feature_names)
    assert abs(grouped["contribution"].sum() - matrix[0].sum()) < 1e-9
    assert set(grouped["group"]) <= {"velocity", "behaviour", "context"}
