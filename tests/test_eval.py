"""Unit tests for the modular eval package (schema, health, topology, scoring)."""

import pandas as pd
import pytest

from src.eval import (
    CANONICAL_SCHEMA,
    SCHEMA_ROLES,
    WEIGHTS,
    evaluate_candidate_dataset,
    evaluate_data_health,
    evaluate_graph_topology,
    evaluate_schema_fit,
    resolve_column,
)


def _canonical_frame(rows: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tx_id": list(range(rows)),
            "src": [f"A{i}" for i in range(rows)],
            "dst": [f"B{i}" for i in range(rows)],
            "amount": [10.0 + i for i in range(rows)],
            "timestamp": [1_700_000_000 + i for i in range(rows)],
            "is_laundering": ([0, 1, 0, 0, 1] * rows)[:rows],
        }
    )


# --- schema -------------------------------------------------------------


def test_schema_fit_roles_and_constants() -> None:
    assert set(SCHEMA_ROLES) == set(CANONICAL_SCHEMA)
    assert CANONICAL_SCHEMA == ["tx_id", "src", "dst", "amount", "timestamp", "is_laundering"]


def test_schema_fit_full_canonical() -> None:
    result = evaluate_schema_fit(_canonical_frame())
    assert result["score"] == 1.0
    assert result["missing"] == []
    assert result["mapped"] == 6


def test_schema_fit_aliases_mapped() -> None:
    df = pd.DataFrame(
        {
            "transaction_id": [0],
            "source": ["A"],
            "target": ["B"],
            "value": [10.0],
            "timestamp_seconds": [1000.0],
            "label": [0],
        }
    )
    result = evaluate_schema_fit(df)
    assert result["score"] == 1.0
    assert result["found"]["tx_id"] == "transaction_id"
    assert result["found"]["src"] == "source"
    assert result["found"]["dst"] == "target"
    assert result["found"]["amount"] == "value"
    assert result["found"]["timestamp"] == "timestamp_seconds"
    assert result["found"]["is_laundering"] == "label"


def test_schema_fit_mixed_case_and_whitespace() -> None:
    df = pd.DataFrame({" Tx_ID ": [0], "Source": ["A"], "TARGET": ["B"]})
    result = evaluate_schema_fit(df)
    assert result["found"]["tx_id"] == " Tx_ID "
    assert result["found"]["src"] == "Source"
    assert result["found"]["dst"] == "TARGET"
    assert result["score"] == pytest.approx(3 / 6)


def test_schema_fit_partial_and_empty() -> None:
    partial = evaluate_schema_fit(pd.DataFrame({"tx_id": [1], "amount": [3.0]}))
    assert partial["score"] == pytest.approx(2 / 6)
    assert "src" in partial["missing"] and "is_laundering" in partial["missing"]
    empty = evaluate_schema_fit(pd.DataFrame())
    assert empty["score"] == 0.0
    assert empty["mapped"] == 0


def test_resolve_column_unknown_role() -> None:
    assert resolve_column(_canonical_frame(), "not_a_role") is None
    assert resolve_column(pd.DataFrame({"x": [1]}), "src") is None


# --- health -------------------------------------------------------------


def test_health_perfect_frame() -> None:
    result = evaluate_data_health(_canonical_frame())
    assert result["non_null_ratio"] == 1.0
    assert result["positive_amount_ratio"] == 1.0
    assert result["valid_timestamp_ratio"] == 1.0
    assert result["score"] == pytest.approx(1.0)


def test_health_null_negative_and_bad_timestamps() -> None:
    df = pd.DataFrame(
        {
            "amount": [10.0, -5.0, None],
            "timestamp": ["2020-01-01", "not-a-date", "2020-01-03"],
        }
    )
    result = evaluate_data_health(df)
    assert result["positive_amount_ratio"] == pytest.approx(1 / 3)
    assert result["valid_timestamp_ratio"] == pytest.approx(2 / 3)
    assert result["non_null_ratio"] < 1.0
    assert result["score"] == pytest.approx((result["non_null_ratio"] + 1 / 3 + 2 / 3) / 3)


def test_health_aliased_columns() -> None:
    df = pd.DataFrame({"value": ["10.5", "22"], "timestamp_seconds": [1000, 1001]})
    result = evaluate_data_health(df)
    assert result["positive_amount_ratio"] == 1.0
    assert result["valid_timestamp_ratio"] == 1.0


def test_health_empty_and_missing_columns() -> None:
    empty = evaluate_data_health(pd.DataFrame())
    assert empty["score"] == 0.0
    bare = evaluate_data_health(pd.DataFrame({"tx_id": [1, 2]}))
    assert bare["positive_amount_ratio"] == 0.0
    assert bare["valid_timestamp_ratio"] == 0.0


def test_health_unparseable_timestamps_guard() -> None:
    df = pd.DataFrame({"timestamp": [object()]})
    result = evaluate_data_health(df)
    assert result["valid_timestamp_ratio"] == 0.0


# --- topology -----------------------------------------------------------


def test_topology_metrics() -> None:
    df = pd.DataFrame(
        {
            "src": ["A", "B", "C"],
            "dst": ["B", "C", "A"],
            "amount": [1, 2, 3],
            "is_laundering": [0, 1, 0],
        }
    )
    result = evaluate_graph_topology(df)
    assert result["nodes"] == 3
    assert result["edges"] == 3
    assert result["connectivity_ratio"] == 1.0
    assert result["aml_ratio"] == pytest.approx(1 / 3)
    assert result["aml_balance"] == 1.0
    assert result["non_zero_aml"] is True


def test_topology_aml_edge_cases() -> None:
    base = {"src": ["A", "B"], "dst": ["B", "C"], "amount": [1, 2]}
    zero = evaluate_graph_topology(pd.DataFrame({**base, "is_laundering": [0, 0]}))
    assert zero["aml_ratio"] == 0.0
    assert zero["aml_balance"] == 0.0
    assert zero["non_zero_aml"] is False
    all_pos = evaluate_graph_topology(pd.DataFrame({**base, "is_laundering": [1, 1]}))
    assert all_pos["aml_ratio"] == 1.0
    assert all_pos["aml_balance"] == 0.0
    half = evaluate_graph_topology(pd.DataFrame({**base, "is_laundering": [1, 0]}))
    assert half["aml_ratio"] == pytest.approx(0.5)


def test_topology_connectivity_clamps_at_one() -> None:
    # More edges than nodes still caps the ratio at 1.0.
    df = pd.DataFrame({"src": ["A", "A", "B"], "dst": ["B", "C", "C"], "is_laundering": [0, 0, 0]})
    result = evaluate_graph_topology(df)
    assert result["nodes"] == 3
    assert result["edges"] == 3
    assert result["connectivity_ratio"] == 1.0


def test_topology_no_src_or_dst() -> None:
    result = evaluate_graph_topology(pd.DataFrame({"amount": [1, 2, 3]}))
    assert result["nodes"] == 0
    assert result["edges"] == 0
    assert result["score"] == 0.0
    assert result["connectivity_ratio"] == 0.0


# --- scoring ------------------------------------------------------------


def test_scoring_weights_sum_to_one() -> None:
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)
    assert set(WEIGHTS) == {"schema_fit", "data_health", "graph_topology", "aml_balance"}


def test_candidate_scoring_aggregation() -> None:
    result = evaluate_candidate_dataset(_canonical_frame())
    assert set(result) == {
        "schema_fit",
        "data_health",
        "graph_topology",
        "aml_balance",
        "weighted_score",
        "nodes",
        "edges",
        "aml_ratio",
    }
    expected = (
        result["schema_fit"] * WEIGHTS["schema_fit"]
        + result["data_health"] * WEIGHTS["data_health"]
        + result["graph_topology"] * WEIGHTS["graph_topology"]
        + result["aml_balance"] * WEIGHTS["aml_balance"]
    )
    assert result["weighted_score"] == pytest.approx(expected)
    assert result["nodes"] == 10  # src A0..A4 + dst B0..B4
    assert result["edges"] == 5


def test_candidate_empty_frame_scores_zero() -> None:
    result = evaluate_candidate_dataset(pd.DataFrame())
    assert result["weighted_score"] == 0.0
    assert result["schema_fit"] == 0.0
    assert result["data_health"] == 0.0
    assert result["graph_topology"] == 0.0
    assert result["aml_balance"] == 0.0


def test_candidate_alias_mapping_aggregation() -> None:
    df = pd.DataFrame(
        {
            "transaction_id": [0, 1],
            "source": ["A", "B"],
            "target": ["B", "A"],
            "value": ["10.5", "22"],
            "timestamp_seconds": [1000.0, 1001.0],
            "label": ["0", "1"],
        }
    )
    result = evaluate_candidate_dataset(df)
    assert result["schema_fit"] == 1.0
    assert result["nodes"] == 2
    assert result["edges"] == 2
    assert result["aml_ratio"] == pytest.approx(0.5)


def test_candidate_backward_compat_shapes() -> None:
    """The old auto_fetch callers expected exactly these keys; keep them."""
    df = pd.DataFrame(
        {
            "src": ["A", "B", "C"],
            "dst": ["B", "C", "A"],
            "amount": [10.5, 22.0, 700.0],
            "timestamp": [1000.0, 1001.0, 1002.0],
            "is_laundering": [0, 1, 0],
        }
    )
    result = evaluate_candidate_dataset(df)
    assert "weighted_score" in result
    assert "schema_fit" in result
    assert "data_health" in result
    assert "graph_topology" in result
    assert "aml_balance" in result
    assert result["aml_ratio"] == pytest.approx(1 / 3)
