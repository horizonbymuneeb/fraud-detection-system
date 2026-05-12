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
