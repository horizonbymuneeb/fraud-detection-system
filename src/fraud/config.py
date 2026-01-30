"""Typed, validated access to ``config.yaml``.

The rest of the package never reads YAML directly: it receives a :class:`Config`
instance. Keeping parsing and validation in one place means a typo in the config
fails at load time with a message naming the offending key, not deep inside a
feature function.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
REQUIRED_SECTIONS = ("data", "features", "split", "model", "cost")
FRAUD_TYPES = ("velocity", "amount", "geo", "night")


@dataclass(frozen=True)
class DataConfig:
    """How the synthetic transaction stream is generated."""

    n_cards: int = 2500
    n_days: int = 90
    start_date: str = "2024-01-01"
    activity_rate: float = 0.9
    base_fraud_rate: float = 0.005
    fraud_mix: Mapping[str, float] = field(
        default_factory=lambda: {"velocity": 0.35, "amount": 0.25, "geo": 0.2, "night": 0.2}
    )
    output_dir: str = "data"


@dataclass(frozen=True)
class FeatureConfig:
    """Feature-engineering knobs shared by every feature module."""

    velocity_windows_hours: tuple[int, ...] = (1, 24, 168)
    min_history_txns: int = 3
    amount_zscore_floor: float = 5.0
    clip_quantile_high: float = 0.999
    category_risk: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SplitConfig:
    """Chronological train/test split."""

    train_fraction: float = 0.70
    boundary_gap_hours: int = 168


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for the scikit-learn gradient boosting classifier."""

    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 3
    min_samples_leaf: int = 25
    subsample: float = 0.85
    max_features: str = "sqrt"
    l2_regularization: float = 0.1
    validation_fraction: float = 0.1
    n_iter_no_change: int = 20
    random_state: int = 20240117


@dataclass(frozen=True)
class CostConfig:
    """The business cost matrix used to choose the decision threshold."""

    false_positive_cost: float = 3.0
    fraud_loss_cost: float = 250.0
    amount_loss_multiplier: float = 1.0
    threshold_grid_points: int = 401
