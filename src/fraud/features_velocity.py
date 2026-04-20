"""Transaction-velocity features: how busy is this card in the last 1h/24h/7d?

A transaction is only ever scored against transactions that happened *before*
it. The counters here are therefore computed over the half-open interval
``(t - window, t)`` and never include the row being scored, which is what makes
the features usable in a streaming scorer and leak-free across a time split.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import Config

NS_PER_HOUR = 3_600_000_000_000
