"""Unified AML data pipeline — three MLOps operational layers.

- **Discovery & Ingestion** (:mod:`src.pipeline.discovery`) — agentic dataset
  discovery, Playwright scraping with CAPTCHA surfacing, CSV/HF fetching with
  a deterministic synthetic fallback, and LangGraph/CrewAI-ready agent nodes.
- **Validation & Quality** (:mod:`src.pipeline.validation`) — Pandera schema
  contract, dataset-quality scoring, and the PyG substructure verifier.
- **Orchestration & Pipeline** (:mod:`src.pipeline.orchestration`) — step-based
  ``fetch → validate → transform`` execution (ZenML / Kedro compatible).

The shared graph transformation kernel lives in
:mod:`src.pipeline.transform` and is consumed by the orchestration layer.

This package facade re-exports the full public API so callers can simply
``from src.pipeline import auto_fetch, fetch_to_pyg, build_pyg_data, ...``.
"""

from src.pipeline.discovery.agents import (
    DISCOVERY_NODES,
    DiscoveryState,
    SequentialDiscoveryGraph,
    build_discovery_graph,
    run_discovery_agents,
)
from src.pipeline.discovery.auto_fetch import (
    DatasetCandidate,
    assess_reliability,
    auto_fetch,
    candidates_from_json_records,
    discover_and_verify,
    discover_candidates,
    fetch_to_pyg,
    generate_search_queries,
    handoff,
    handoff_to_storage,
    list_candidate_datasets,
    sanitize_transactions,
    search_web_browser,
    search_web_resilient,
    validate_transactions,
    verified_summary,
    verify_candidates,
)
from src.pipeline.discovery.ingestion import (
    CANONICAL_COLUMNS,
    fetch_transactions,
    generate_synthetic_transactions,
    normalize_columns,
)
from src.pipeline.discovery.scraper import (
    CaptchaChallenge,
    PlaywrightScraper,
    ScraperConfig,
)
from src.pipeline.orchestration.pipeline import Pipeline, build_default_pipeline, run_pipeline
from src.pipeline.orchestration.steps import Step, fetch_step, step, transform_step, validate_step
from src.pipeline.transform.features import FEATURE_COLUMNS, compute_node_features
from src.pipeline.transform.graph_builder import build_pyg_data, compute_node_labels
from src.pipeline.transform.positional_encoding import (
    laplacian_positional_encoding,
    random_walk_structural_encoding,
)
from src.pipeline.transform.sampling import make_neighbor_loader
from src.pipeline.validation.pandera_schema import validate_transaction_schema
from src.pipeline.validation.scoring import evaluate_candidate_dataset

__all__ = [
    "CANONICAL_COLUMNS",
    "DISCOVERY_NODES",
    "DatasetCandidate",
    "DiscoveryState",
    "FEATURE_COLUMNS",
    "Pipeline",
    "PlaywrightScraper",
    "ScraperConfig",
    "SequentialDiscoveryGraph",
    "Step",
    "assess_reliability",
    "auto_fetch",
    "build_default_pipeline",
    "build_discovery_graph",
    "build_pyg_data",
    "candidates_from_json_records",
    "CaptchaChallenge",
    "compute_node_features",
    "compute_node_labels",
    "discover_and_verify",
    "discover_candidates",
    "evaluate_candidate_dataset",
    "fetch_step",
    "fetch_to_pyg",
    "fetch_transactions",
    "generate_search_queries",
    "generate_synthetic_transactions",
    "handoff",
    "handoff_to_storage",
    "laplacian_positional_encoding",
    "list_candidate_datasets",
    "make_neighbor_loader",
    "normalize_columns",
    "random_walk_structural_encoding",
    "run_discovery_agents",
    "run_pipeline",
    "sanitize_transactions",
    "search_web_browser",
    "search_web_resilient",
    "step",
    "transform_step",
    "validate_step",
    "validate_transaction_schema",
    "validate_transactions",
    "verified_summary",
    "verify_candidates",
]
