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


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_yaml(tmp_path / "nope.yaml")


def test_invalid_fraud_rate_and_costs_are_rejected():
    raw = read_yaml(DEFAULT_CONFIG_PATH)
    raw["data"]["base_fraud_rate"] = 0.5
    with pytest.raises(ValueError, match="base_fraud_rate"):
        build_config(raw)

    raw = read_yaml(DEFAULT_CONFIG_PATH)
    raw["cost"]["fraud_loss_cost"] = -1.0
    with pytest.raises(ValueError, match="fraud_loss_cost"):
        build_config(raw)


def test_fraud_mix_must_sum_to_one():
    raw = read_yaml(DEFAULT_CONFIG_PATH)
    raw["data"]["fraud_mix"] = {"velocity": 0.5, "amount": 0.2}
    with pytest.raises(ValueError, match="sum to 1.0"):
        build_config(raw)


def test_unknown_fraud_mix_key_is_rejected():
    raw = read_yaml(DEFAULT_CONFIG_PATH)
    raw["data"]["fraud_mix"] = {"velocity": 1.0, "chargeback": 0.0}
    with pytest.raises(ValueError, match="fraud_mix"):
        build_config(raw)


def test_require_section_rejects_scalars():
    with pytest.raises(ValueError, match="mapping"):
        require_section({"data": 3}, "data")


def test_round_trip_to_dict_and_description():
    cfg = load_config(DEFAULT_CONFIG_PATH)
    as_dict = config_to_dict(cfg)
    assert as_dict["data"]["n_cards"] == cfg.data.n_cards
    assert as_dict["features"]["velocity_windows_hours"] == list(
        cfg.features.velocity_windows_hours
    )
    text = describe_config(cfg)
    assert "cards=" in text and "trees=" in text
