"""Modular evaluation engine for AML graph datasets.

Each module is a single, testable concern:

- :mod:`src.eval.schema` — how much canonical schema is present.
- :mod:`src.eval.health` — non-null ratio, positive amounts, timestamp validity.
- :mod:`src.eval.topology` — node/edge counts, connectivity, AML class ratio.
- :mod:`src.eval.scoring` — aggregates the above into a weighted quality score.
"""

from src.eval.health import evaluate_data_health
from src.eval.schema import (
    CANONICAL_SCHEMA,
    SCHEMA_ROLES,
    evaluate_schema_fit,
    resolve_column,
)
from src.eval.scoring import WEIGHTS, evaluate_candidate_dataset
from src.eval.topology import evaluate_graph_topology

__all__ = [
    "CANONICAL_SCHEMA",
    "SCHEMA_ROLES",
    "WEIGHTS",
    "evaluate_schema_fit",
    "evaluate_data_health",
    "evaluate_graph_topology",
    "evaluate_candidate_dataset",
    "resolve_column",
]
