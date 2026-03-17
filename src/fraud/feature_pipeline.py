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


def split_masks(frame: pd.DataFrame, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Boolean train/test masks matching :func:`generator.chronological_split`.

    The gap is expressed in hours: test rows inside the gap after the boundary
    are excluded so each test row has at least a 7-day velocity history.
    """
    ordered = frame.sort_values("timestamp", kind="mergesort")
    n = len(ordered)
    if n < 2:
        raise ValueError("need at least two rows to split")
    boundary = int(n * cfg.split.train_fraction)
    boundary = min(max(boundary, 1), n - 1)
    boundary_ts = ordered["timestamp"].iloc[boundary]
    test_start = boundary_ts + pd.Timedelta(hours=cfg.split.boundary_gap_hours)
    train_mask = np.zeros(len(frame), dtype=bool)
    test_mask = np.zeros(len(frame), dtype=bool)
    train_positions = ordered.index[:boundary].to_numpy() if not isinstance(
        ordered.index, pd.RangeIndex
    ) else ordered.index[:boundary].to_numpy()
    train_mask[train_positions] = True
    test_positions = ordered.index[ordered["timestamp"] >= test_start].to_numpy()
    test_mask[test_positions] = True
    test_mask &= ~train_mask
    return train_mask, test_mask


def fit_imputer(features: pd.DataFrame) -> dict[str, float]:
    """Column medians from the training window; all-NaN columns impute to 0."""
    medians: dict[str, float] = {}
    for column in features.columns:
        values = features[column].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        medians[column] = float(np.median(finite)) if finite.size else 0.0
    return medians


def apply_imputer(features: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    """Fill non-finite values with the fitted per-column medians."""
    out = features.copy()
    for column, value in medians.items():
        if column in out:
            out[column] = out[column].fillna(value).replace([np.inf, -np.inf], value)
    return out


def fit_clipper(features: pd.DataFrame, quantile: float) -> dict[str, float]:
    """Upper winsorising bound per column, fitted on the training window."""
    if not 0.5 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0.5, 1]")
    bounds: dict[str, float] = {}
    for column in features.columns:
        values = features[column].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        bounds[column] = float(np.quantile(finite, quantile)) if finite.size else 0.0
    return bounds


def apply_clipper(features: pd.DataFrame, bounds: dict[str, float]) -> pd.DataFrame:
    """Clip each column at its fitted upper bound."""
    out = features.copy()
    for column, bound in bounds.items():
        if column in out:
            out[column] = out[column].clip(upper=bound)
    return out


def make_xy(features: pd.DataFrame, frame: pd.DataFrame, mask: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Select rows by ``mask`` and return ``(X, y)`` with the label as int8."""
    if len(features) != len(frame):
        raise ValueError("features and frame must have the same length")
    if mask.dtype != bool or mask.shape != (len(frame),):
        raise ValueError("mask must be a boolean array matching the frame")
    if mask.sum() == 0:
        raise ValueError("mask selects no rows")
    X = features.loc[mask].reset_index(drop=True)
    y = frame.loc[mask, "is_fraud"].to_numpy(dtype=np.int8)
    return X, y


def prepare_datasets(frame: pd.DataFrame, cfg: Config) -> dict[str, object]:
    """Build features, fit train-only transforms and produce both splits.

    Returns a dict with ``X_train``, ``y_train``, ``X_test``, ``y_test`` plus
    the fitted ``medians``/``bounds`` and ``category_frequency`` so a scorer can
    reproduce the exact transform later.
    """
    train_mask, test_mask = split_masks(frame, cfg)
    category_frequency = fit_category_frequency(frame.loc[train_mask])
    features = assemble_feature_frame(frame, cfg, category_frequency)
    X_train_raw, y_train = make_xy(features, frame, train_mask)
    X_test_raw, y_test = make_xy(features, frame, test_mask)
    medians = fit_imputer(X_train_raw)
    bounds = fit_clipper(
        apply_imputer(X_train_raw, medians), cfg.features.clip_quantile_high
    )
    X_train = apply_clipper(apply_imputer(X_train_raw, medians), bounds)
    X_test = apply_clipper(apply_imputer(X_test_raw, medians), bounds)
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "amount_train": frame.loc[train_mask, "amount"].to_numpy(dtype=np.float64),
        "amount_test": frame.loc[test_mask, "amount"].to_numpy(dtype=np.float64),
        "fraud_types_test": frame.loc[test_mask, "fraud_type"],
        "train_mask": train_mask,
        "test_mask": test_mask,
        "medians": medians,
        "bounds": bounds,
        "category_frequency": category_frequency,
        "features": features,
    }


def assert_no_future_leakage(
    frame: pd.DataFrame, cfg: Config, sample_rows: int = 3, seed: int = 0
) -> bool:
    """Recompute a sampled row's features from its card's past and compare.

    Velocity, behaviour, novelty and risk features of a transaction depend only
    on earlier rows *of the same card* (category frequency is passed in fixed),
    so truncating the rest of the frame must not change the sampled row. If it
    does, some feature is looking at the future and the assertion names it.
    """
    if sample_rows < 1:
        raise ValueError("sample_rows must be >= 1")
    ordered = frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    if len(ordered) < 3:
        raise ValueError("need at least three rows to check for leakage")
    frequency = fit_category_frequency(ordered)
    full = assemble_feature_frame(ordered, cfg, frequency)
    rng = np.random.default_rng(seed)
    candidates = np.arange(1, len(ordered))
    chosen = np.sort(
        rng.choice(candidates, size=min(sample_rows, len(candidates)), replace=False)
    )
    cards = ordered["card_id"].to_numpy()
    for row in chosen:
        row = int(row)
        card_positions = np.flatnonzero(cards == cards[row])
        prefix = ordered.iloc[card_positions[card_positions <= row]]
        prefix_features = assemble_feature_frame(prefix, cfg, frequency)
        expected = full.iloc[row].to_numpy(dtype=np.float64)
        actual = prefix_features.iloc[-1].to_numpy(dtype=np.float64)
        if not np.allclose(expected, actual, equal_nan=True, rtol=1e-12, atol=1e-12):
            bad = [
                column
                for column, left, right in zip(full.columns, expected, actual)
                if not np.isclose(left, right, equal_nan=True)
            ]
            raise AssertionError(f"future leakage at row {row} (card {cards[row]}): {bad}")
    return True


def feature_manifest(cfg: Config) -> list[dict[str, object]]:
    """Structured description of every feature, written next to the model."""
    return [
        {"name": name, "group": feature_group(name), "source": "config.yaml"}
        for name in feature_columns(cfg)
    ]


def save_manifest(path: str | Path, cfg: Config) -> Path:
    """Write the feature manifest as JSON (used by the scorer to detect drift)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_features": len(feature_columns(cfg)),
        "features": feature_manifest(cfg),
        "velocity_windows_hours": list(cfg.features.velocity_windows_hours),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_manifest(path: str | Path) -> dict[str, object]:
    """Read a manifest written by :func:`save_manifest`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
