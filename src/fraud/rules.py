"""The rule baseline every model has to beat.

Rules are what a fraud team already has on day one: a burst of card-present
transactions, an amount far above the card's own median, a first-time foreign
merchant, a large late-night purchase. The baseline matters because it shares
the same features as the model, so the comparison isolates the model's benefit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RULE_WEIGHTS: dict[str, float] = {
    "burst_1h": 2.0,
    "extreme_amount": 2.0,
    "amount_ratio": 1.5,
    "foreign_first": 1.5,
    "night_large": 1.0,
    "new_merchant_spike": 1.0,
}
