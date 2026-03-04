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
