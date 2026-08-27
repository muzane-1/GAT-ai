"""Unit tests for graph topology feature generation."""

import pandas as pd
import pytest

from src.data_pipeline.features import FEATURE_COLUMNS, compute_node_features


@pytest.fixture()
def toy_transactions() -> pd.DataFrame:
    """Two laundering txs (label → 1) and a handful of benign transfers."""
    rows = [
        {
            "tx_id": 0,
            "src": "A",
            "dst": "B",
            "amount": 100.0,
            "timestamp": 1000,
            "is_laundering": 1,
        },
        {
            "tx_id": 1,
            "src": "B",
            "dst": "C",
            "amount": 100.0,
            "timestamp": 1500,
            "is_laundering": 1,
        },
        {"tx_id": 2, "src": "C", "dst": "D", "amount": 50.0, "timestamp": 1200, "is_laundering": 0},
        {"tx_id": 3, "src": "D", "dst": "A", "amount": 50.0, "timestamp": 1700, "is_laundering": 0},
        {"tx_id": 4, "src": "A", "dst": "D", "amount": 25.0, "timestamp": 2000, "is_laundering": 0},
    ]
    return pd.DataFrame(rows)


def test_feature_columns_present(toy_transactions: pd.DataFrame) -> None:
    """All topology features are produced for every account."""
    features = compute_node_features(toy_transactions)
    for column in FEATURE_COLUMNS:
        assert column in features.columns
    assert set(features.index) == {"A", "B", "C", "D"}


def test_degree_and_amounts(toy_transactions: pd.DataFrame) -> None:
    """Degree and amount statistics match the toy edge list."""
    features = compute_node_features(toy_transactions)
    assert features.loc["A", "out_degree"] == 2
    assert features.loc["A", "in_degree"] == 1
    assert features.loc["D", "in_degree"] == 2
    assert pytest.approx(features.loc["A", "mean_amount_sent"]) == (100.0 + 25.0) / 2


def test_label_propagation(toy_transactions: pd.DataFrame) -> None:
    """Any account touching a laundering edge is flagged positive."""
    features = compute_node_features(toy_transactions)
    assert features.loc["A", "label"] == 1
    assert features.loc["B", "label"] == 1
    assert features.loc["C", "label"] == 1
    assert features.loc["D", "label"] == 0


def test_velocity_window(toy_transactions: pd.DataFrame) -> None:
    """Velocity counts only transactions inside the recent window."""
    features = compute_node_features(toy_transactions, velocity_window_seconds=600)  # max_ts=2000
    assert features.loc["A", "sent_velocity"] == 1  # only tx 4 inside window
    assert features.loc["B", "received_velocity"] == 0
    assert features.loc["D", "received_velocity"] == 1
    assert features.loc["B", "sent_velocity"] == 1  # tx id 1 at t=1500 inside window


def test_empty_input(toy_transactions: pd.DataFrame) -> None:
    """Empty input returns an empty table, not an exception."""
    empty = compute_node_features(
        pd.DataFrame(columns=["tx_id", "src", "dst", "amount", "timestamp", "is_laundering"])
    )
    assert empty.empty
