"""Data ingestion, feature engineering, and PyG graph construction."""

from src.data_pipeline.auto_fetch import (
    auto_fetch,
    fetch_to_pyg,
    list_candidate_datasets,
    sanitize_transactions,
    validate_transactions,
)
from src.data_pipeline.features import FEATURE_COLUMNS, compute_node_features
from src.data_pipeline.graph_builder import build_pyg_data, compute_node_labels
from src.data_pipeline.ingestion import fetch_transactions, generate_synthetic_transactions

__all__ = [
    "FEATURE_COLUMNS",
    "auto_fetch",
    "build_pyg_data",
    "compute_node_features",
    "compute_node_labels",
    "fetch_to_pyg",
    "fetch_transactions",
    "generate_synthetic_transactions",
    "list_candidate_datasets",
    "sanitize_transactions",
    "validate_transactions",
]
