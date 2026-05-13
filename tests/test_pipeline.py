"""Leakage, split masks and train-only transforms are the pipeline's contract."""

import numpy as np
import pandas as pd
import pytest

from src.fraud.feature_pipeline import (
    apply_clipper,
    apply_imputer,
    assemble_feature_frame,
    assert_no_future_leakage,
    feature_columns,
    feature_group,
    fit_category_frequency,
    fit_clipper,
    fit_imputer,
    make_xy,
    prepare_datasets,
    save_manifest,
    load_manifest,
    split_masks,
)


def test_training_rows_do_not_change_when_test_rows_appear(tiny_frame, tiny_config):
    train_mask, _test_mask = split_masks(tiny_frame, tiny_config)
    train_only = tiny_frame.loc[train_mask]
    frequency = fit_category_frequency(train_only)
    full_features = assemble_feature_frame(tiny_frame, tiny_config, frequency)
    train_features = assemble_feature_frame(train_only, tiny_config, frequency)
    assert np.allclose(
        full_features.loc[train_mask].to_numpy(),
        train_features.to_numpy(),
        equal_nan=True,
    )


def test_leakage_assertion_passes_for_causal_features(tiny_frame, tiny_config):
    assert assert_no_future_leakage(tiny_frame, tiny_config, sample_rows=3) is True


def test_split_masks_are_disjoint_and_respect_gap(tiny_frame, tiny_config):
    train_mask, test_mask = split_masks(tiny_frame, tiny_config)
    assert not (train_mask & test_mask).any()
    assert train_mask.sum() + test_mask.sum() < len(tiny_frame)
    train_end = tiny_frame.loc[train_mask, "timestamp"].max()
    test_start = tiny_frame.loc[test_mask, "timestamp"].min()
    assert test_start >= train_end + pd.Timedelta(hours=tiny_config.split.boundary_gap_hours)


def test_imputer_fills_nan_with_train_medians():
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, np.nan, np.nan]})
    medians = fit_imputer(X)
    assert medians["a"] == pytest.approx(2.0)
    assert medians["b"] == 0.0
    filled = apply_imputer(X, medians)
    assert filled.isna().sum().sum() == 0
    assert filled["a"].tolist() == [1.0, 2.0, 3.0]


def test_clipper_caps_values_at_fitted_quantile():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 1000.0]})
    bounds = fit_clipper(X, 0.75)
    clipped = apply_clipper(X, bounds)
    assert clipped["a"].max() == bounds["a"] < 1000.0
    with pytest.raises(ValueError):
        fit_clipper(X, 0.2)


def test_make_xy_aligns_labels_and_validates_mask():
    frame = pd.DataFrame({"is_fraud": [0, 1, 0]})
    features = pd.DataFrame({"f": [0.1, 0.2, 0.3]})
    mask = np.array([True, True, False])
    X, y = make_xy(features, frame, mask)
    assert len(X) == 2 and y.tolist() == [0, 1]
    with pytest.raises(ValueError):
        make_xy(features, frame, np.array([False, False, False]))
    with pytest.raises(ValueError):
        make_xy(features, frame, np.array([1, 0, 0]))
