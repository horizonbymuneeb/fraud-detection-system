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


@dataclass(frozen=True)
class Config:
    """Full experiment configuration."""

    seed: int
    data: DataConfig
    features: FeatureConfig
    split: SplitConfig
    model: ModelConfig
    cost: CostConfig


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from ``path``, raising ``FileNotFoundError`` if absent."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping, got {type(raw).__name__}")
    return raw


def require_section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Return ``raw[name]`` as a mapping, or raise ``ValueError`` naming it."""
    if name not in raw:
        raise ValueError(f"config is missing required section: {name!r}")
    section = raw[name]
    if not isinstance(section, Mapping):
        raise ValueError(f"config section {name!r} must be a mapping")
    return section


def _as_int(value: Any, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be an integer, got {value!r}")
    if float(value) != int(value):
        raise ValueError(f"{key} must be an integer, got {value!r}")
    return int(value)


def _as_float(value: Any, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number, got {value!r}")
    return float(value)


def _as_windows(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("features.velocity_windows_hours must be a non-empty list")
    windows = tuple(_as_int(v, "features.velocity_windows_hours") for v in value)
    if any(w <= 0 for w in windows):
        raise ValueError("velocity windows must be positive")
    return tuple(sorted(set(windows)))


def build_config(raw: Mapping[str, Any]) -> Config:
    """Parse a raw YAML mapping into a validated :class:`Config`."""
    data_raw = require_section(raw, "data")
    features_raw = require_section(raw, "features")
    split_raw = require_section(raw, "split")
    model_raw = require_section(raw, "model")
    cost_raw = require_section(raw, "cost")

    mix = data_raw.get("fraud_mix", {})
    if not isinstance(mix, Mapping) or set(mix) - set(FRAUD_TYPES):
        raise ValueError(f"data.fraud_mix keys must be a subset of {FRAUD_TYPES}")
    mix_values = {str(k): _as_float(v, f"data.fraud_mix.{k}") for k, v in mix.items()}
    if mix_values and abs(sum(mix_values.values()) - 1.0) > 1e-6:
        raise ValueError("data.fraud_mix must sum to 1.0")

    cfg = Config(
        seed=_as_int(raw.get("seed", 0), "seed"),
        data=DataConfig(
            n_cards=_as_int(data_raw.get("n_cards", 2500), "data.n_cards"),
            n_days=_as_int(data_raw.get("n_days", 90), "data.n_days"),
            start_date=str(data_raw.get("start_date", "2024-01-01")),
            activity_rate=_as_float(data_raw.get("activity_rate", 0.9), "data.activity_rate"),
            base_fraud_rate=_as_float(
                data_raw.get("base_fraud_rate", 0.005), "data.base_fraud_rate"
            ),
            fraud_mix=mix_values,
            output_dir=str(data_raw.get("output_dir", "data")),
        ),
        features=FeatureConfig(
            velocity_windows_hours=_as_windows(
                features_raw.get("velocity_windows_hours", [1, 24, 168])
            ),
            min_history_txns=_as_int(
                features_raw.get("min_history_txns", 3), "features.min_history_txns"
            ),
            amount_zscore_floor=_as_float(
                features_raw.get("amount_zscore_floor", 5.0), "features.amount_zscore_floor"
            ),
            clip_quantile_high=_as_float(
                features_raw.get("clip_quantile_high", 0.999), "features.clip_quantile_high"
            ),
            category_risk={
                str(k): _as_float(v, f"features.category_risk.{k}")
                for k, v in (features_raw.get("category_risk") or {}).items()
            },
        ),
        split=SplitConfig(
            train_fraction=_as_float(split_raw.get("train_fraction", 0.7), "split.train_fraction"),
            boundary_gap_hours=_as_int(
                split_raw.get("boundary_gap_hours", 168), "split.boundary_gap_hours"
            ),
        ),
        model=ModelConfig(
            n_estimators=_as_int(model_raw.get("n_estimators", 300), "model.n_estimators"),
            learning_rate=_as_float(model_raw.get("learning_rate", 0.05), "model.learning_rate"),
            max_depth=_as_int(model_raw.get("max_depth", 3), "model.max_depth"),
            min_samples_leaf=_as_int(
                model_raw.get("min_samples_leaf", 25), "model.min_samples_leaf"
            ),
            subsample=_as_float(model_raw.get("subsample", 0.85), "model.subsample"),
            max_features=str(model_raw.get("max_features", "sqrt")),
            l2_regularization=_as_float(
                model_raw.get("l2_regularization", 0.1), "model.l2_regularization"
            ),
            validation_fraction=_as_float(
                model_raw.get("validation_fraction", 0.1), "model.validation_fraction"
            ),
            n_iter_no_change=_as_int(
                model_raw.get("n_iter_no_change", 20), "model.n_iter_no_change"
            ),
            random_state=_as_int(
                model_raw.get("random_state", raw.get("seed", 0)), "model.random_state"
            ),
        ),
        cost=CostConfig(
            false_positive_cost=_as_float(
                cost_raw.get("false_positive_cost", 3.0), "cost.false_positive_cost"
            ),
            fraud_loss_cost=_as_float(
                cost_raw.get("fraud_loss_cost", 250.0), "cost.fraud_loss_cost"
            ),
            amount_loss_multiplier=_as_float(
                cost_raw.get("amount_loss_multiplier", 1.0), "cost.amount_loss_multiplier"
            ),
            threshold_grid_points=_as_int(
                cost_raw.get("threshold_grid_points", 401), "cost.threshold_grid_points"
            ),
        ),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    """Raise ``ValueError`` when a value is outside its meaningful range."""
    problems: list[str] = []
    if cfg.data.n_cards <= 0:
        problems.append("data.n_cards must be positive")
    if cfg.data.n_days < 2:
        problems.append("data.n_days must be at least 2")
    if not 0.0 < cfg.data.base_fraud_rate < 0.2:
        problems.append("data.base_fraud_rate must be in (0, 0.2)")
    if not 0.0 < cfg.data.activity_rate <= 10.0:
        problems.append("data.activity_rate must be in (0, 10]")
    if cfg.features.min_history_txns < 0:
        problems.append("features.min_history_txns must be >= 0")
    if cfg.features.amount_zscore_floor <= 0:
        problems.append("features.amount_zscore_floor must be positive")
    if not 0.5 <= cfg.features.clip_quantile_high <= 1.0:
        problems.append("features.clip_quantile_high must be in [0.5, 1]")
    if not 0.3 <= cfg.split.train_fraction <= 0.95:
        problems.append("split.train_fraction must be in [0.3, 0.95]")
    if cfg.split.boundary_gap_hours < 0:
        problems.append("split.boundary_gap_hours must be >= 0")
    if cfg.model.n_estimators <= 0:
        problems.append("model.n_estimators must be positive")
    if not 0.0 < cfg.model.learning_rate <= 1.0:
        problems.append("model.learning_rate must be in (0, 1]")
    if not 0.0 < cfg.model.subsample <= 1.0:
        problems.append("model.subsample must be in (0, 1]")
    if not 0.0 < cfg.model.validation_fraction < 1.0:
        problems.append("model.validation_fraction must be in (0, 1)")
    if not set(cfg.data.fraud_mix) <= set(FRAUD_TYPES):
        problems.append(f"data.fraud_mix may only contain {FRAUD_TYPES}")
    if cfg.cost.false_positive_cost <= 0:
        problems.append("cost.false_positive_cost must be positive")
    if cfg.cost.fraud_loss_cost <= 0:
        problems.append("cost.fraud_loss_cost must be positive")
    if cfg.cost.threshold_grid_points < 3:
        problems.append("cost.threshold_grid_points must be >= 3")
    if problems:
        raise ValueError("invalid config: " + "; ".join(problems))


def load_config(
    path: str | Path = DEFAULT_CONFIG_PATH, overrides: Mapping[str, Any] | None = None
) -> Config:
    """Load ``path`` and apply shallow per-section ``overrides``.

    ``overrides`` is meant for tests and small experiments: ``{"data": {"n_cards": 40}}``.
    """
    raw = read_yaml(path)
    if overrides:
        for section, values in overrides.items():
            if section in raw and isinstance(raw[section], Mapping) and isinstance(values, Mapping):
                raw[section] = {**raw[section], **values}
            else:
                raw[section] = values
    return build_config(raw)


def config_to_dict(cfg: Config) -> dict[str, Any]:
    """Round-trip a config back to plain YAML-ready structure."""
    data = asdict(cfg)
    data["features"]["velocity_windows_hours"] = list(cfg.features.velocity_windows_hours)
    return data


def describe_config(cfg: Config) -> str:
    """One-line human summary used by the training CLI."""
    return (
        f"cards={cfg.data.n_cards} days={cfg.data.n_days} "
        f"fraud_rate={cfg.data.base_fraud_rate:.3%} "
        f"windows={cfg.features.velocity_windows_hours} "
        f"trees={cfg.model.n_estimators} "
        f"fp_cost={cfg.cost.false_positive_cost:g} fn_cost={cfg.cost.fraud_loss_cost:g}"
    )
