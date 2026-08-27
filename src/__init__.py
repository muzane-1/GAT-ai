"""Production-grade AML detection via PyTorch Geometric GATv2.

The legacy convenience imports remain available so notebooks written against
the previous layout keep working.
"""

from src.dataset import load_aml_dataset
from src.model import GATv2, GATv2AMLModel
from src.utils.legacy import FocalLoss, compute_metrics, load_checkpoint, save_checkpoint

__version__ = "2.0.0"
__all__ = [
    "FocalLoss",
    "GATv2",
    "GATv2AMLModel",
    "compute_metrics",
    "load_aml_dataset",
    "load_checkpoint",
    "save_checkpoint",
]
