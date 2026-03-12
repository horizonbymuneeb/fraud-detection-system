"""Assemble the feature matrix and split it without leaking across time.

Leakage policy, stated once and enforced by :func:`assert_no_future_leakage`:

* velocity and behaviour features are functions of strictly earlier rows, so
  computing them on the full timeline cannot leak future information;
* category frequency and imputation/clipping parameters are fitted on the
  training window only and then applied to the test window;
* a row shifted later never changes an earlier row's features.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .features_behaviour import behaviour_features_for_config
from .features_context import context_features_for_config, fit_category_frequency
from .features_velocity import velocity_features_for_config

GROUP_PREFIX = {"vel_": "velocity", "dev_": "behaviour", "ctx_": "context"}
MANIFEST_FILENAME = "feature_manifest.json"


def assemble_feature_frame(
    frame: pd.DataFrame, cfg: Config, category_frequency: pd.Series | None = None
) -> pd.DataFrame:
    """Concatenate the three feature blocks, in a fixed column order."""
    blocks = [
        velocity_features_for_config(frame, cfg),
        behaviour_features_for_config(frame, cfg),
        context_features_for_config(frame, cfg, category_frequency),
    ]
    features = pd.concat(blocks, axis=1)
    return features[feature_columns(cfg)]


def feature_columns(cfg: Config | None = None) -> list[str]:
    """Every model feature, in the order the model will see it."""
    from .features_behaviour import behaviour_columns
    from .features_context import context_columns
    from .features_velocity import velocity_columns

    windows = cfg.features.velocity_windows_hours if cfg else (1, 24, 168)
    return velocity_columns(windows) + behaviour_columns() + context_columns()


def feature_group(name: str) -> str:
    """Group a feature name by its prefix, for per-group importance rollups."""
    for prefix, group in GROUP_PREFIX.items():
        if name.startswith(prefix):
            return group
    raise ValueError(f"feature name has no known prefix: {name}")
