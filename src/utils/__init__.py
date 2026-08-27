"""Shared utilities: new config/logging/metrics + notebook-compat legacy APIs.

``compute_metrics`` keeps the legacy (torch-tensor delta) signature that
notebooks use; the new numpy-flavoured implementation remains reachable via
``src.utils.metrics`` for the training pipeline.
"""

from src.utils.config import load_config

# Backward compatibility: notebooks use `from src.utils import FocalLoss,
# compute_metrics, save_checkpoint, load_checkpoint` with torch tensors.
from src.utils.legacy import FocalLoss, compute_metrics, load_checkpoint, save_checkpoint
from src.utils.logger import get_logger
from src.utils.metrics import format_metrics

__all__ = [
    "FocalLoss",
    "compute_metrics",
    "format_metrics",
    "get_logger",
    "load_checkpoint",
    "load_config",
    "save_checkpoint",
]
