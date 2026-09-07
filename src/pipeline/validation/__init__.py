"""Layer 2 — Validation & Quality.

Modular validation engine for AML transaction tables and graphs:

- :mod:`src.pipeline.validation.schema` — how much canonical schema is present.
- :mod:`src.pipeline.validation.health` — non-null ratio, positive amounts,
  timestamp validity.
- :mod:`src.pipeline.validation.topology` — node/edge counts, connectivity,
  AML class ratio.
- :mod:`src.pipeline.validation.scoring` — aggregates the above into a
  weighted quality score.
- :mod:`src.pipeline.validation.pandera_schema` — Pandera contract checked
  *before* graph construction.
- :mod:`src.pipeline.validation.substructure_verifier` — self-verification of
  GNN predictions against local AML motifs (directed cycles, smurfing fans,
  dense/k-core subgraphs); produces verification scores, false positive/negative
  suspects and retrain triggers for the update pipeline.
"""

from src.pipeline.validation.health import evaluate_data_health
from src.pipeline.validation.pandera_schema import (
    HAS_PANDERA,
    TRANSACTION_SCHEMA,
    SchemaValidationError,
    validate_transaction_schema,
)
from src.pipeline.validation.schema import (
    CANONICAL_SCHEMA,
    SCHEMA_ROLES,
    evaluate_schema_fit,
    resolve_column,
)
from src.pipeline.validation.scoring import WEIGHTS, evaluate_candidate_dataset
from src.pipeline.validation.substructure_verifier import (
    DEFAULT_VERIFIER_CONFIG,
    SubstructureVerifierConfig,
    detect_structural_trigger,
    summarise_verification,
    verify_predictions,
)
from src.pipeline.validation.topology import evaluate_graph_topology

__all__ = [
    "CANONICAL_SCHEMA",
    "DEFAULT_VERIFIER_CONFIG",
    "HAS_PANDERA",
    "SCHEMA_ROLES",
    "TRANSACTION_SCHEMA",
    "SchemaValidationError",
    "SubstructureVerifierConfig",
    "WEIGHTS",
    "detect_structural_trigger",
    "evaluate_candidate_dataset",
    "evaluate_data_health",
    "evaluate_graph_topology",
    "evaluate_schema_fit",
    "resolve_column",
    "summarise_verification",
    "validate_transaction_schema",
    "verify_predictions",
]
