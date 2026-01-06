"""The additive decomposition must reconstruct the model margin exactly."""

import json

import numpy as np
import pandas as pd
import pytest

from src.fraud.attribution import (
    PathAttributor,
    additivity_error,
    base_margin,
    contribution_matrix,
    contributions_for_row,
    explain_row,
    grouped_contributions,
    probability_from_margin,
    raw_margin_from_contributions,
    save_explanations,
    top_contributors,
)
