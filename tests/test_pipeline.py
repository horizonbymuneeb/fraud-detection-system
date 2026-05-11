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
