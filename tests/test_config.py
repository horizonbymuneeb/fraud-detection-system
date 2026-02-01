"""Config parsing and validation are the first line of defence against typos."""

import pytest

from src.fraud.config import (
    DEFAULT_CONFIG_PATH,
    build_config,
    config_to_dict,
    describe_config,
    load_config,
    read_yaml,
    require_section,
    validate_config,
)


def test_repository_config_loads_and_is_internally_consistent():
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert cfg.data.n_cards > 0
    assert cfg.features.velocity_windows_hours == tuple(
        sorted(cfg.features.velocity_windows_hours)
    )
    assert abs(sum(cfg.data.fraud_mix.values()) - 1.0) < 1e-6
    assert cfg.model.n_estimators >= 50


def test_overrides_are_applied_per_section():
    cfg = load_config(DEFAULT_CONFIG_PATH, {"data": {"n_cards": 17}, "model": {"n_estimators": 3}})
    assert cfg.data.n_cards == 17
    assert cfg.model.n_estimators == 3
    assert cfg.data.n_days > 0


def test_missing_section_is_named_in_the_error():
    with pytest.raises(ValueError, match="cost"):
        build_config({"seed": 1, "data": {}, "features": {}, "split": {}, "model": {}})
