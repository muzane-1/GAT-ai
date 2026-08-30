"""Weighted dataset scoring — the aggregation layer of the eval engine.

Combines the schema, health and topology sub-evaluators into a single
``weighted_score`` that ranks candidate AML graph datasets before the ingestion
pipeline makes its hard validation checks.
"""

from typing import Any

import pandas as pd

from src.eval.health import evaluate_data_health
from src.eval.schema import evaluate_schema_fit
from src.eval.topology import evaluate_graph_topology

#: Relative importance of each sub-evaluator when ranking candidates.
WEIGHTS: dict[str, float] = {
    "schema_fit": 0.3,
    "data_health": 0.3,
    "graph_topology": 0.2,
    "aml_balance": 0.2,
}


def evaluate_candidate_dataset(df_raw: pd.DataFrame) -> dict[str, Any]:
    """Aggregate sub-evaluators into a weighted quality score.

    Args:
        df_raw: Raw (unsanitised) candidate transaction table.

    Returns:
        Dict of scores — ``schema_fit``, ``data_health``, ``graph_topology``,
        ``aml_balance``, ``weighted_score`` — plus the raw ``nodes``,
        ``edges`` and ``aml_ratio`` metrics. The return shape is compatible
        with the original ``src.data_pipeline.auto_fetch`` implementation so
        existing callers keep working.
    """
    schema = evaluate_schema_fit(df_raw)
    health = evaluate_data_health(df_raw)
    topology = evaluate_graph_topology(df_raw)

    weighted_score = (
        schema["score"] * WEIGHTS["schema_fit"]
        + health["score"] * WEIGHTS["data_health"]
        + topology["score"] * WEIGHTS["graph_topology"]
        + topology["aml_balance"] * WEIGHTS["aml_balance"]
    )

    return {
        "schema_fit": schema["score"],
        "data_health": health["score"],
        "graph_topology": topology["score"],
        "aml_balance": topology["aml_balance"],
        "weighted_score": float(weighted_score),
        "nodes": topology["nodes"],
        "edges": topology["edges"],
        "aml_ratio": topology["aml_ratio"],
    }
