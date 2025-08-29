"""Utility functions for production ML."""
import numpy as np
import torch
import random
import json
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

def save_metrics(metrics: Dict[str, float], path: str) -> None:
    """Save evaluation metrics to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {path}")

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_device() -> torch.device:
    """Get the best available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def format_number(n: int) -> str:
    """Format large numbers with K/M/B suffixes."""
    for unit in ['', 'K', 'M', 'B']:
        if abs(n) < 1000:
            return f"{n:.1f}{unit}"
        n /= 1000
    return f"{n:.1f}T"

# Fix timestamp feature extraction timezone bug [2025-06-11T16:59:01]

# Add model monitoring dashboard for drift [2025-06-12T15:48:29]

# Add transaction velocity feature engineering [2025-06-17T11:26:40]

# Add model monitoring dashboard for drift [2025-06-24T18:12:21]

# Fix memory issue in batch inference worker [2025-06-25T12:56:37]

# Add anomaly detection as backup classifier [2025-06-30T14:16:18]

# Fix memory issue in batch inference worker [2025-07-01T09:05:38]

# WIP: tuning neural networks for fraud patterns [2025-07-01T20:27:44]

# Update dashboard for fraud metrics visualization [2025-07-06T14:54:30]

# Implement ensemble of XGBoost and RandomForest [2025-07-07T19:24:10]

# Add model explainability with SHAP values [2025-07-09T17:04:59]

# Update dashboard for fraud metrics visualization [2025-07-14T10:37:19]

# Update deployment scripts for Docker container [2025-07-16T16:57:24]

# Implement cross-validation for time series [2025-07-21T11:17:13]

# Update feature store integration for batch [2025-07-21T16:54:44]

# WIP: tuning neural networks for fraud patterns [2025-07-24T11:01:52]

# WIP: tuning neural networks for fraud patterns [2025-08-05T12:14:14]

# Implement ensemble of XGBoost and RandomForest [2025-08-08T10:22:17]

# Implement cross-validation for time series [2025-08-10T10:48:28]

# Add model monitoring dashboard for drift [2025-08-20T20:51:21]

# Add logging for prediction audit trail [2025-08-20T13:13:21]

# Add logging for prediction audit trail [2025-08-25T15:58:26]

# Implement XGBoost fraud detection classifier [2025-08-28T09:55:08]

# Implement XGBoost fraud detection classifier [2025-08-29T12:18:01]
