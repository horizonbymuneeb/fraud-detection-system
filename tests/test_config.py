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
