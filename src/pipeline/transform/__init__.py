"""Graph transformation kernel: canonical transaction tables → PyG objects.

This is the shared transformation core consumed by the orchestration layer's
``transform`` step (features → graph construction → positional encodings →
neighbor sampling).
"""

from src.pipeline.transform.features import FEATURE_COLUMNS, compute_node_features
from src.pipeline.transform.graph_builder import build_pyg_data, compute_node_labels
from src.pipeline.transform.positional_encoding import (
    laplacian_positional_encoding,
    random_walk_structural_encoding,
)
from src.pipeline.transform.sampling import make_neighbor_loader

__all__ = [
    "FEATURE_COLUMNS",
    "build_pyg_data",
    "compute_node_features",
    "compute_node_labels",
    "laplacian_positional_encoding",
    "make_neighbor_loader",
    "random_walk_structural_encoding",
]
