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
