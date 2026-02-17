"""Evaluate a saved model bundle on a transaction stream.

    python -m src.fraud.evaluate --config config.yaml --model artifacts/model.pkl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, load_config
from .cost_threshold import compare_thresholds, cost_matrix_from_config
from .evaluation import calibration_table, summary_report
from .feature_pipeline import load_manifest, prepare_datasets
from .generator import read_transactions, validate_transactions
from .trainer import dump_json, load_model, predict_scores
