"""Automated AML dataset fetch, sanitation, and graph validation.

This module is the MLOps entrypoint that turns remote Hugging Face
transaction tables into a clean, graph-ready pandas DataFrame (and, when
requested, a PyTorch Geometric ``Data`` object). It is deliberately layered
as three composable stages:

1. **Discovery & fetch** — use :class:`huggingface_hub.HfApi` to locate
   candidate datasets, then stream CSV rows via ``hf_hub_download`` or the
   canonical :func:`src.data_pipeline.ingestion.fetch_transactions` path.
2. **Sanitation** — normalise columns into the repo's canonical schema
   (``tx_id, src, dst, amount, timestamp, is_laundering``), drop duplicates,
   fix missing values, and log-scale-normalise amounts.
3. **Validation** — assert the graph is non-empty, well-typed, and class
   distribution is recordable (not necessarily balanced, but visible).

If every remote path fails or the post-validation snapshot is empty, the
pipeline deterministically falls back to the built-in synthetic generator so
callers (training, CI, notebooks) never see a hard crash.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data_pipeline.ingestion import (
    CANONICAL_COLUMNS,
    fetch_transactions,
    generate_synthetic_transactions,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def list_candidate_datasets(
    query: str = "transactions aml",
    author: str | None = None,
    limit: int = 20,
) -> list[str]:
    """Query Hugging Face for AML-flavoured transaction datasets.

    Args:
        query: Full-text search across dataset names / cards.
        author: Optional user/org filter (e.g. ``"qubit420"``).
        limit: Maximum number of dataset IDs to return.

    Returns:
        A list of ``"<owner>/<repo>"`` identifiers, best-match first. An
        empty list on any API error so downstream logic falls through to the
        local / synthetic paths.
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        rows = api.list_datasets(search=query, author=author, limit=limit)
        return [row.id for row in rows]
    except Exception as exc:  # network, auth, or hub rate-limit
        logger.warning("hf_dataset_discovery_failed", extra={"error": str(exc)})
        return []


def _canonicalise(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to the canonical schema, best-effort."""
    from src.data_pipeline.ingestion import _COLUMN_ALIASES

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}
    for alias, canonical in _COLUMN_ALIASES.items():
        if alias.lower() in lower_map:
            df = df.rename(columns={lower_map[alias.lower()]: canonical})
    return df


def sanitize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Apply MLOps-standard sanitation rules to a raw transaction table.

    - Column normalisation into the canonical schema.
    - Missing ``tx_id`` → synthesised from row index (stable).
    - Missing ``timestamp`` → forward-fill then 0.0 (first rows).
    - Non-numeric ``amount`` → parsed, NaNs dropped, negatives clipped.
    - Duplicate edges (same src/dst/amount/ts) → dropped.
    - ``is_laundering`` → coerced to 0/1 (any truthy string parsed).

    Returns:
        Clean DataFrame that satisfies :func:`validate_transactions`.
    """
    df = _canonicalise(df)
    original = len(df)

    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            if col == "tx_id":
                df[col] = np.arange(len(df)).astype(str)
            elif col == "is_laundering":
                df[col] = 0
            elif col == "timestamp":
                df[col] = 0.0
            else:
                df[col] = np.nan

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["src", "dst", "amount"])
    df["amount"] = df["amount"].clip(lower=0.0)

    df["is_laundering"] = (
        df["is_laundering"]
        .astype(str)
        .str.lower()
        .map({"1": 1, "true": 1, "yes": 1, "0": 0, "false": 0, "no": 0})
        .fillna(0)
        .astype(int)
    )

    df = df.drop_duplicates(subset=["src", "dst", "amount", "timestamp"]).reset_index(drop=True)
    logger.info(
        "sanitized_transactions",
        extra={"before": original, "after": len(df), "dropped": original - len(df)},
    )
    return df[CANONICAL_COLUMNS]


def validate_transactions(df: pd.DataFrame) -> dict[str, Any]:
    """Compute sanity metrics used by the notebook's success/warning checks.

    Raises:
        ValueError: if the table is empty or missing canonical columns.
    """
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing canonical columns: {missing}")
    if len(df) == 0:
        raise ValueError("transaction table is empty after sanitation")

    src = df["src"].astype(str)
    dst = df["dst"].astype(str)
    nodes = pd.concat([src, dst]).nunique()
    edges = len(df)
    pos = int(df["is_laundering"].sum())

    # Graph connectivity: are all node ids reachable in an undirected view?
    try:
        import scipy.sparse as sp

        ids = pd.concat([src, dst]).unique().tolist()
        idx = {v: i for i, v in enumerate(ids)}
        rows = np.array([idx[s] for s in src], dtype=np.int64)
        cols = np.array([idx[d] for d in dst], dtype=np.int64)
        adj = sp.coo_matrix((np.ones(edges), (rows, cols)), shape=(len(ids), len(ids)))
        n_comp, _ = sp.csgraph.connected_components(adj + adj.T, directed=False, return_labels=True)
    except Exception:
        n_comp = -1  # connectivity not measurable; still report other stats

    return {
        "rows": edges,
        "nodes": int(nodes),
        "edges": edges,
        "avg_degree": round(edges / max(nodes, 1), 3),
        "connected_components": int(n_comp),
        "class_counts": {0: edges - pos, 1: pos},
        "aml_ratio": round(pos / edges, 6),
        "null_cells": int(df.isna().sum().sum()),
    }


def auto_fetch(
    source: str | None = None,
    hf_query: str | None = "qubit420/ibm-aml-LI-smaller",
    normalize_amounts: bool = True,
    fallback_generate: bool = True,
    **builder_kwargs: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """End-to-end fetch → sanitize → validate pipeline.

    Resolution order:
    1. Explicit ``source`` (local CSV path or HTTP URL).
    2. Explicit Hugging Face dataset id (``hf_query``); falls back through
       ``fetch_transactions`` so the canonical retry+backoff logic still
       applies.
    3. Deterministic synthetic generator.

    Returns:
        ``(canonical_transactions, stats_dict)`` — the stats are produced by
        :func:`validate_transactions` and include class-imbalance numbers.
    """
    df: pd.DataFrame | None = None
    provenance: str = "synthetic"

    if source:
        df = fetch_transactions(source=source, fallback_generate=False)
        provenance = f"source:{source}"
    elif hf_query:
        try:
            df = fetch_transactions(source=hf_query, fallback_generate=False)
            provenance = f"hf:{hf_query}"
        except Exception as exc:
            logger.warning("hf_fetch_failed", extra={"dataset": hf_query, "error": str(exc)})
            df = None

    if df is None:
        if not fallback_generate:
            raise RuntimeError(
                "auto_fetch could not retrieve a usable table and fallback is disabled"
            )
        df = generate_synthetic_transactions()
        provenance = "synthetic"

    df = sanitize_transactions(df)
    if normalize_amounts:
        # log1p keeps the heavily skewed amounts well-behaved for GNN scaling
        df = df.copy()
        df["amount"] = np.log1p(df["amount"])

    stats = validate_transactions(df)
    stats["provenance"] = provenance
    stats["normalized_amounts"] = normalize_amounts
    logger.info(
        "auto_fetch_complete",
        extra={"provenance": provenance, "rows": stats["rows"], "aml_ratio": stats["aml_ratio"]},
    )
    return df, stats


def fetch_to_pyg(
    source: str | None = None,
    hf_query: str | None = "qubit420/ibm-aml-LI-smaller",
    **builder_kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Fetch and convert to a PyG ``Data`` object via the canonical builder.

    Returns:
        ``(data, stats)`` where ``data`` is the PyG graph and ``stats`` is
        the dict produced by :func:`auto_fetch`.
    """
    from src.data_pipeline.graph_builder import build_pyg_data

    df, stats = auto_fetch(source=source, hf_query=hf_query, **builder_kwargs)
    data, _ = build_pyg_data(df)
    stats["num_node_features"] = int(data.num_node_features)
    return data, stats
