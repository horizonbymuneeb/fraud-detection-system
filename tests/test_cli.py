"""End-to-end: generate data, train, persist artifacts, evaluate the bundle."""

import json

import yaml

from src.fraud.config import load_config, read_yaml
from src.fraud.evaluate import main as evaluate_main
from src.fraud.generator import generate_transactions, write_transactions
from src.fraud.train import main as train_main
