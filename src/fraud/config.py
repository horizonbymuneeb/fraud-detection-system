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
