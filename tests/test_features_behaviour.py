"""Amount-deviation features are prior-only; these tests pin the exact values."""

import numpy as np
import pytest

from src.fraud.features_behaviour import (
    card_history_statistics,
    compute_behaviour_features,
    is_round_amount,
    percentile_rank,
    round_increment,
    zscore,
)
