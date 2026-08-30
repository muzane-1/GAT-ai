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

# Default keywords and tags for dynamic dataset discovery
DEFAULT_KEYWORDS = ["aml", "anti-money-laundering", "financial-graph", "transaction-fraud"]
DEFAULT_TAGS = ["finance", "graph", "fraud", "aml"]


def list_candidate_datasets(
    keywords: list[str] | None = None,
    tags: list[str] | None = None,
    author: str | None = None,
    limit: int = 20,
) -> list[str]:
    """Query Hugging Face for AML-flavoured transaction datasets using dynamic keywords and tags.

    Args:
        keywords: List of keywords to search for in dataset names/cards.
        tags: List of tags to filter datasets.
        author: Optional user/org filter (e.g. ``"qubit420"``).
        limit: Maximum number of dataset IDs to return.

    Returns:
        A list of ``"<owner>/<repo>"`` identifiers, best-match first. An
        empty list on any API error so downstream logic falls through to the
        local / synthetic paths.
    """
    keywords = keywords or DEFAULT_KEYWORDS
    tags = tags or DEFAULT_TAGS

    try:
        from huggingface_hub import HfApi

        api = HfApi()
        query = " ".join(keywords + tags)
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


def evaluate_candidate_dataset(df_raw: pd.DataFrame) -> dict[str, float]:
    """Evaluate a candidate dataset using a weighted quality score.

    Args:
        df_raw: Raw transaction DataFrame to evaluate.

        Returns:
        A dictionary containing the evaluation scores and metrics.
    """
    from src.data_pipeline.ingestion import _COLUMN_ALIASES

    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Schema Fit: Check for canonical columns or aliases
    schema_fit_score = 0.0
    canonical_columns_present = [col for col in CANONICAL_COLUMNS if col in df.columns]
    alias_columns_present = [alias for alias in _COLUMN_ALIASES if alias in df.columns]
    schema_fit_score = (len(canonical_columns_present) + len(alias_columns_present)) / len(
        CANONICAL_COLUMNS
    )

    # Data Health: Non-null ratio, positive amounts, valid timestamps
    non_null_ratio = df.notna().mean().mean()
    positive_amount_ratio = 0.0
    valid_timestamp_ratio = 0.0

    if "amount" in df.columns or any(alias in df.columns for alias in ["value"]):
        amount_col = (
            "amount"
            if "amount" in df.columns
            else [alias for alias in ["value"] if alias in df.columns][0]
        )
        df["amount"] = pd.to_numeric(df[amount_col], errors="coerce")
        positive_amount_ratio = (df["amount"] > 0).mean()

    if "timestamp" in df.columns or any(alias in df.columns for alias in ["timestamp_seconds"]):
        timestamp_col = (
            "timestamp"
            if "timestamp" in df.columns
            else [alias for alias in ["timestamp_seconds"] if alias in df.columns][0]
        )
        try:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
            valid_timestamp_ratio = df[timestamp_col].notna().mean()
        except Exception:
            valid_timestamp_ratio = 0.0

    data_health_score = (non_null_ratio + positive_amount_ratio + valid_timestamp_ratio) / 3

    # Graph Topology & Balance: Node/edge counts, connectivity, AML class ratio
    graph_topology_score = 0.0
    aml_balance_score = 0.0

    src_col = "src" if "src" in df.columns else ("source" if "source" in df.columns else None)
    dst_col = "dst" if "dst" in df.columns else ("target" if "target" in df.columns else None)

    if src_col and dst_col:
        nodes = pd.concat([df[src_col], df[dst_col]]).nunique()
        edges = len(df)

        if nodes > 0 and edges > 0:
            connectivity_ratio = min(1.0, edges / nodes)
            graph_topology_score = (nodes + edges + connectivity_ratio) / 3

    if "is_laundering" in df.columns or any(
        alias in df.columns for alias in ["label", "laundering"]
    ):
        aml_col = (
            "is_laundering"
            if "is_laundering" in df.columns
            else [alias for alias in ["label", "laundering"] if alias in df.columns][0]
        )
        df[aml_col] = pd.to_numeric(df[aml_col], errors="coerce").fillna(0)
        aml_ratio = df[aml_col].mean()

        if 0 < aml_ratio < 0.5:
            aml_balance_score = 1.0
        else:
            aml_balance_score = max(0.0, 1.0 - abs(aml_ratio - 0.25) / 0.25)

    # Weighted quality score
    weights = {
        "schema_fit": 0.3,
        "data_health": 0.3,
        "graph_topology": 0.2,
        "aml_balance": 0.2,
    }

    weighted_score = (
        schema_fit_score * weights["schema_fit"]
        + data_health_score * weights["data_health"]
        + graph_topology_score * weights["graph_topology"]
        + aml_balance_score * weights["aml_balance"]
    )

    return {
        "schema_fit": schema_fit_score,
        "data_health": data_health_score,
        "graph_topology": graph_topology_score,
        "aml_balance": aml_balance_score,
        "weighted_score": weighted_score,
        "nodes": int(nodes) if "nodes" in locals() else 0,
        "edges": edges if "edges" in locals() else 0,
        "aml_ratio": aml_ratio if "aml_ratio" in locals() else 0.0,
    }


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
    hf_query: str | None = None,
    hf_keywords: list[str] | None = None,
    hf_tags: list[str] | None = None,
    normalize_amounts: bool = True,
    fallback_generate: bool = True,
    **builder_kwargs: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """End-to-end fetch → sanitize → validate pipeline with dynamic multi-source discovery.

    Resolution order:
    1. Explicit ``source`` (local CSV path or HTTP URL).
    2. Explicit Hugging Face dataset id (``hf_query``).
    3. Dynamic discovery of Hugging Face datasets using keywords and tags.
    4. Deterministic synthetic generator.

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
        # Dynamic discovery of Hugging Face datasets
        candidate_datasets = list_candidate_datasets(
            keywords=hf_keywords or DEFAULT_KEYWORDS,
            tags=hf_tags or DEFAULT_TAGS,
            limit=20,
        )

        if candidate_datasets:
            logger.info(f"Discovered {len(candidate_datasets)} candidate datasets")

            # Evaluate and rank candidate datasets
            evaluated_datasets = []
            for dataset_id in candidate_datasets:
                try:
                    df_raw = fetch_transactions(source=dataset_id, fallback_generate=False)
                    evaluation_scores = evaluate_candidate_dataset(df_raw)
                    evaluated_datasets.append((dataset_id, evaluation_scores, df_raw))
                    logger.info(
                        f"Evaluated dataset {dataset_id}: "
                        f"score={evaluation_scores['weighted_score']:.3f}"
                    )
                except Exception as exc:
                    logger.warning(
                        f"Failed to fetch or evaluate dataset {dataset_id}",
                        extra={"error": str(exc)},
                    )
                    continue

            if evaluated_datasets:
                # Sort by weighted score in descending order
                evaluated_datasets.sort(key=lambda x: x[1]["weighted_score"], reverse=True)

                # Select the highest-scoring dataset that passes hard validation checks
                for dataset_id, evaluation_scores, df_raw in evaluated_datasets:
                    try:
                        df = sanitize_transactions(df_raw)
                        stats = validate_transactions(df)

                        # Hard validation checks
                        if (
                            stats["rows"] > 0
                            and stats["nodes"] > 0
                            and 0 < stats["aml_ratio"] < 0.5
                        ):
                            provenance = f"hf:{dataset_id}"
                            logger.info(
                                f"Selected highest-scoring dataset {dataset_id} "
                                f"with score {evaluation_scores['weighted_score']:.3f}"
                            )
                            break
                    except Exception as exc:
                        logger.warning(
                            f"Dataset {dataset_id} failed validation",
                            extra={"error": str(exc)},
                        )
                        df = None
                        continue

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
    hf_query: str | None = None,
    hf_keywords: list[str] | None = None,
    hf_tags: list[str] | None = None,
    **builder_kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Fetch and convert to a PyG ``Data`` object via the canonical builder.

    Returns:
        ``(data, stats)`` where ``data`` is the PyG graph and ``stats`` is
        the dict produced by :func:`auto_fetch`.
    """
    from src.data_pipeline.graph_builder import build_pyg_data

    df, stats = auto_fetch(
        source=source,
        hf_query=hf_query,
        hf_keywords=hf_keywords,
        hf_tags=hf_tags,
        **builder_kwargs,
    )
    data, _ = build_pyg_data(df)
    stats["num_node_features"] = int(data.num_node_features)
    return data, stats
