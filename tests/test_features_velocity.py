"""Velocity features are checked against hand-counted windows."""

import numpy as np
import pandas as pd
import pytest

from src.fraud.features_velocity import (
    burst_flag,
    card_groups,
    compute_velocity_features,
    distinct_window_counts,
    hours_since_previous,
    sum_ratio,
    timestamps_to_ns,
    velocity_columns,
    velocity_summary,
    window_counts_sums,
    window_to_ns,
)
