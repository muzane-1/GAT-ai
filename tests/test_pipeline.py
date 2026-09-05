"""Unit tests for ingestion retry/fallback logic and PyG graph building."""

from unittest import mock

import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from src.data_pipeline.graph_builder import build_pyg_data
from src.data_pipeline.ingestion import (
    CANONICAL_COLUMNS,
    fetch_transactions,
    generate_synthetic_transactions,
    normalize_columns,
)
from src.data_pipeline.positional_encoding import (
    laplacian_positional_encoding,
    random_walk_structural_encoding,
)
from src.data_pipeline.sampling import make_neighbor_loader


def test_synthetic_generation_schema() -> None:
    """Synthetic fallback produces the canonical schema and both classes."""
    df = generate_synthetic_transactions(
        n_accounts=100, n_transactions=500, fraud_ratio=0.05, seed=7
    )
    assert list(df.columns) == CANONICAL_COLUMNS
    assert 0 < df["is_laundering"].sum() < len(df)
    assert (df["amount"] > 0).all()


def test_rows_and_aliasing_normalization() -> None:
    """Legacy column aliases are mapped to canonical names."""
    raw = pd.DataFrame(
        {
            "transaction_id": [0],
            "source": ["A"],
            "target": ["B"],
            "value": [10.0],
            "timestamp": [1000],
            "label": [0],
        }
    )
    normalized = normalize_columns(raw)
    assert list(normalized.columns) == CANONICAL_COLUMNS
    assert normalized.iloc[0]["src"] == "A"
    assert normalized.iloc[0]["dst"] == "B"


def test_missing_columns_raise() -> None:
    """Incomplete tables fail loudly instead of silently corrupting."""
    with pytest.raises(ValueError):
        normalize_columns(pd.DataFrame({"tx_id": [0]}))


def test_fetch_retries_then_fallback() -> None:
    """A permanently broken source falls back to synthetic data."""
    with mock.patch("src.data_pipeline.ingestion.requests.get") as mocked_get:
        mocked_get.side_effect = RuntimeError("network down")
        df = fetch_transactions(
            "https://example.com/tx.csv",
            n_retry=2,
            backoff_seconds=0.001,
            fallback_generate=True,
            fallback_kwargs={"n_accounts": 20, "n_transactions": 50},
        )
    assert mocked_get.call_count == 2
    assert list(df.columns) == CANONICAL_COLUMNS


def test_fetch_raises_when_fallback_disabled() -> None:
    """Fallback-to-synthetic is disabled → RuntimeError surfaces."""
    with mock.patch("src.data_pipeline.ingestion.requests.get") as mocked_get:
        mocked_get.side_effect = RuntimeError("network down")
        with pytest.raises(RuntimeError):
            fetch_transactions(
                "https://example.com/tx.csv",
                n_retry=1,
                backoff_seconds=0.001,
                fallback_generate=False,
            )


def test_graph_builder_shapes_and_scaling() -> None:
    """Graph building yields expected shapes, scaler reuse, and labels."""
    df = generate_synthetic_transactions(n_accounts=40, n_transactions=200, seed=1)
    data, scaler = build_pyg_data(df)
    assert data.num_nodes == len(set(df["src"]).union(set(df["dst"])))
    assert data.edge_index.shape == (2, len(df))
    assert data.x.shape[1] == 9  # 9 engineered node features
    assert data.edge_attr.shape[1] == 2  # scaled amount + timestamp delta
    assert data.y.dtype == torch.long

    # Scaler reuse (e.g. validation graph) must not refit.
    data2, scaler2 = build_pyg_data(df, scaler=scaler)
    assert scaler2 is scaler
    assert data2.num_edges == len(df)

    # Feature columns are standardised to zero mean.
    assert abs(data.x.numpy().mean()) < 1e-4
    assert data.lap_pe.shape == (data.num_nodes, 8)
    assert data.rw_pe.shape == (data.num_nodes, 8)


def test_laplacian_and_random_walk_encodings_are_structural() -> None:
    """Encodings have stable shapes and RW self-return probabilities."""
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    lap_pe = laplacian_positional_encoding(edge_index, num_nodes=3, num_embeddings=2)
    rw_pe = random_walk_structural_encoding(edge_index, num_nodes=3, walk_length=2)
    assert lap_pe.shape == (3, 2)
    assert rw_pe.shape == (3, 2)
    assert torch.allclose(rw_pe[:, 0], torch.tensor([0.0, 0.0, 0.0]))


def test_neighbor_loader_batch_contract() -> None:
    """Sampling exposes PyG's node/edge batch contract at the configured size."""
    data = Data(
        x=torch.randn(1000, 3),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        y=torch.zeros(1000, dtype=torch.long),
    )
    loader = make_neighbor_loader(data, batch_size=1000)
    assert loader.batch_size == 1000
    assert loader.data.num_nodes == 1000
