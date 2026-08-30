"""Unit tests for the automated dataset fetch / validate pipeline."""

import pandas as pd
import pytest

from src.data_pipeline.auto_fetch import (
    auto_fetch,
    fetch_to_pyg,
    sanitize_transactions,
    validate_transactions,
)
from src.data_pipeline.ingestion import CANONICAL_COLUMNS


def _raw_messy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": ["A", "B", "B", "A", "C"],
            "target": ["B", "A", "A", "B", "C"],
            "value": ["10.5", "22", "22", "not_a_number", "700"],
            "timestamp_seconds": [1000.0, 1001.0, 1001.0, 1002.0, 1003.0],
            "label": ["0", "1", "1", "0", "true"],
        }
    )


def test_sanitize_normalizes_schema_and_types() -> None:
    df = sanitize_transactions(_raw_messy_frame())
    assert list(df.columns) == CANONICAL_COLUMNS
    assert df["amount"].ge(0).all()
    assert df["is_laundering"].isin([0, 1]).all()
    # "not_a_number" row dropped + 1 duplicate edge dropped
    assert len(df) == 3


def test_validate_reports_topology_and_imbalance() -> None:
    df = sanitize_transactions(_raw_messy_frame())
    stats = validate_transactions(df)
    assert stats["rows"] == 3 and stats["nodes"] == 3
    assert stats["aml_ratio"] == pytest.approx(2 / 3)
    assert stats["null_cells"] == 0


def test_validate_rejects_empty_frame() -> None:
    with pytest.raises(ValueError, match="missing canonical columns"):
        validate_transactions(pd.DataFrame())


def test_auto_fetch_synthetic_fallback_and_pyg() -> None:
    df, stats = auto_fetch(hf_query=None)
    assert stats["provenance"] == "synthetic"
    assert stats["normalized_amounts"] is True

    data, _ = fetch_to_pyg(hf_query=None)
    assert data.num_nodes > 0 and data.num_edges > 0
    assert data.x is not None and data.edge_index.shape[0] == 2


def test_evaluate_candidate_dataset() -> None:
    from src.data_pipeline.auto_fetch import evaluate_candidate_dataset

    df = pd.DataFrame(
        {
            "src": ["A", "B", "C"],
            "dst": ["B", "C", "A"],
            "amount": [10.5, 22.0, 700.0],
            "timestamp": [1000.0, 1001.0, 1002.0],
            "is_laundering": [0, 1, 0],
        }
    )

    scores = evaluate_candidate_dataset(df)
    assert "weighted_score" in scores
    assert "schema_fit" in scores
    assert "data_health" in scores
    assert "graph_topology" in scores
    assert "aml_balance" in scores


def test_list_candidate_datasets() -> None:
    from src.data_pipeline.auto_fetch import list_candidate_datasets

    # Test with default keywords and tags
    datasets = list_candidate_datasets()
    assert isinstance(datasets, list)

    # Test with custom keywords and tags
    datasets = list_candidate_datasets(keywords=["aml"], tags=["finance"])
    assert isinstance(datasets, list)
