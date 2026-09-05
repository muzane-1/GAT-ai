"""Data ingestion, feature engineering, and PyG graph construction."""

from src.data_pipeline.auto_fetch import (
    DatasetCandidate,
    assess_reliability,
    auto_fetch,
    discover_and_verify,
    discover_candidates,
    fetch_to_pyg,
    generate_search_queries,
    handoff,
    list_candidate_datasets,
    sanitize_transactions,
    validate_transactions,
    verified_summary,
    verify_candidates,
)
from src.data_pipeline.features import FEATURE_COLUMNS, compute_node_features
from src.data_pipeline.graph_builder import build_pyg_data, compute_node_labels
from src.data_pipeline.ingestion import fetch_transactions, generate_synthetic_transactions
from src.data_pipeline.positional_encoding import (
    laplacian_positional_encoding,
    random_walk_structural_encoding,
)
from src.data_pipeline.sampling import make_neighbor_loader

__all__ = [
    "DatasetCandidate",
    "FEATURE_COLUMNS",
    "assess_reliability",
    "auto_fetch",
    "build_pyg_data",
    "laplacian_positional_encoding",
    "make_neighbor_loader",
    "compute_node_features",
    "compute_node_labels",
    "discover_and_verify",
    "discover_candidates",
    "fetch_to_pyg",
    "fetch_transactions",
    "generate_search_queries",
    "generate_synthetic_transactions",
    "random_walk_structural_encoding",
    "handoff",
    "list_candidate_datasets",
    "sanitize_transactions",
    "validate_transactions",
    "verified_summary",
    "verify_candidates",
]
