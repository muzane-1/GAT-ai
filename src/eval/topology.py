"""Graph-topology evaluation for raw AML transaction tables.

Assesses the graph structure implied by the transaction edges: node and edge
counts, a connectivity ratio (edges per node, capped at 1), and the AML class
ratio. A useful class ratio is strictly between 0 and 0.5 — if every row (or no
row) is flagged, the table carries little discriminative signal.
"""

from typing import Any

import pandas as pd

from src.eval.schema import resolve_column


def evaluate_graph_topology(df: pd.DataFrame) -> dict[str, Any]:
    """Score node/edge structure, connectivity ratio and AML class balance.

    Args:
        df: Raw transaction table.

    Returns:
        Dict with ``score`` (0..1-ish graph-size proxy, kept compatible with
        the earlier pipeline), ``nodes``, ``edges``, ``connectivity_ratio``,
        ``aml_ratio``, ``aml_balance`` and ``non_zero_aml`` boolean.
    """
    src_col = resolve_column(df, "src")
    dst_col = resolve_column(df, "dst")
    nodes = 0
    edges = 0
    connectivity_ratio = 0.0
    if src_col is not None and dst_col is not None:
        src_vals = df[src_col].astype(str)
        dst_vals = df[dst_col].astype(str)
        nodes = int(pd.concat([src_vals, dst_vals]).nunique())
        edges = int(len(df))
        if nodes > 0 and edges > 0:
            connectivity_ratio = round(min(1.0, edges / nodes), 6)

    graph_score = (nodes + edges + connectivity_ratio) / 3.0 if (nodes > 0 and edges > 0) else 0.0

    aml_col = resolve_column(df, "is_laundering")
    aml_ratio = 0.0
    if aml_col is not None:
        aml_vals = pd.to_numeric(df[aml_col], errors="coerce").fillna(0)
        aml_ratio = float(aml_vals.mean()) if len(aml_vals) else 0.0

    aml_balance = 1.0 if 0 < aml_ratio < 0.5 else max(0.0, 1.0 - abs(aml_ratio - 0.25) / 0.25)

    return {
        "score": float(graph_score),
        "nodes": nodes,
        "edges": edges,
        "connectivity_ratio": connectivity_ratio,
        "aml_ratio": aml_ratio,
        "aml_balance": aml_balance,
        "non_zero_aml": aml_ratio > 0,
    }
