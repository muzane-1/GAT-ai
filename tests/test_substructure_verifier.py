"""Unit tests for the Substructure-Aware Verification self-evaluation engine.

The fixtures are hand-built motif graphs (directed laundering cycles, smurfing
fans, isolated pairs) so every structural expectation is exact and the suite
stays fast, deterministic and network-free.
"""

import importlib
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from src.eval import (
    DEFAULT_VERIFIER_CONFIG,
    detect_structural_trigger,
    summarise_verification,
    verify_predictions,
)
from src.eval.substructure_verifier import (
    _cycle_length_through,
    _density,
    _directed_adjacency,
    _max_core_number,
    _undirected_edges,
)


def _motif_graph() -> Data:
    """15-node graph embedding every canonical AML motif plus quiet noise.

    Layout:
      * nodes 0-2  — directed 3-cycle (layered laundering loop)
      * node 3     — fan-out hub (4 receivers, smurfing)
      * node 12    — fan-in sink (4 senders, smurfing)
      * nodes 13-14— isolated pair (no AML structure at all)
    """
    edges = [
        (0, 1),
        (1, 2),
        (2, 0),  # directed cycle
        (3, 4),
        (3, 5),
        (3, 6),
        (3, 7),  # fan-out hub
        (8, 12),
        (9, 12),
        (10, 12),
        (11, 12),  # fan-in sink
        (13, 14),  # benign isolated pair
    ]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(num_nodes=15, edge_index=edge_index)


def _probs(alert_nodes: set[int], num_nodes: int = 15) -> torch.Tensor:
    probs = torch.full((num_nodes,), 0.05)
    for node in alert_nodes:
        probs[node] = 0.9
    return probs


def _labels() -> torch.Tensor:
    y = torch.zeros(15, dtype=torch.long)
    for node in (0, 1, 2, 3, 12):  # true AML accounts
        y[node] = 1
    return y


# ---------------------------------------------------------------------------
# Motif primitives
# ---------------------------------------------------------------------------


def test_directed_adjacency_is_sorted_and_loop_free() -> None:
    edge_index = torch.tensor([[2, 1, 1, 0], [1, 2, 2, 0]]).t().contiguous()
    adj = _directed_adjacency(edge_index, num_nodes=3)
    assert adj[0] == []  # self-loop dropped
    assert adj[1] == [2]  # duplicate removed
    assert adj[2] == [1]


def test_cycle_length_found_and_absent() -> None:
    cycle_adj = _directed_adjacency(torch.tensor([[0, 1, 2], [1, 2, 0]]), num_nodes=3)
    assert _cycle_length_through(cycle_adj, 0, max_len=3, max_paths=1000) == 3

    open_adj = _directed_adjacency(torch.tensor([[0, 1], [1, 2]]), num_nodes=3)
    assert _cycle_length_through(open_adj, 0, max_len=3, max_paths=1000) == 0

    # Budget exhaustion degrades to "no cycle" instead of hanging.
    assert _cycle_length_through(cycle_adj, 0, max_len=3, max_paths=0) == 0


def test_max_core_and_density() -> None:
    # Triangle (0,1,2) + pendant edge (2,3): max core = 2, density = 4/6.
    edges = [(0, 1), (1, 2), (0, 2), (2, 3)]
    assert _max_core_number(4, edges) == 2
    assert _density(4, edges) == pytest.approx(4 / 6)
    assert _max_core_number(0, []) == 0
    assert _density(1, []) == 0.0


def test_undirected_edges_dedupes_and_orders() -> None:
    edge_index = torch.tensor([[1, 0, 2], [0, 1, 2]])
    assert _undirected_edges(edge_index, num_nodes=3) == [(0, 1)]  # self-loop dropped


# ---------------------------------------------------------------------------
# verify_predictions end-to-end on the motif fixture
# ---------------------------------------------------------------------------


def test_cycle_and_smurfing_alerts_are_verified() -> None:
    data = _motif_graph()
    report = verify_predictions(
        data, _probs({0, 1, 2, 3, 12, 13}), y=_labels(), config=DEFAULT_VERIFIER_CONFIG
    )

    assert report["inspected_nodes"] > 0
    assert report["verification_score"] > 0.0

    by_node = {f["node"]: f for f in report["findings"]}
    assert by_node[0]["cycle_length"] == 3
    assert "directed_cycle_len_3" in by_node[0]["motifs"]
    assert by_node[0]["score"] >= DEFAULT_VERIFIER_CONFIG.confirm_threshold
    assert by_node[3]["fan_out"] == 4
    assert "fan_out_smurfing" in by_node[3]["motifs"]
    assert by_node[12]["fan_in"] == 4
    assert "fan_in_smurfing" in by_node[12]["motifs"]


def test_unsupported_alert_is_flagged_false_positive() -> None:
    data = _motif_graph()
    report = verify_predictions(data, _probs({13}), y=_labels())
    assert 13 in report["suspected_false_positives"]
    assert report["structural_misclassification_rate"] > 0.0
    assert any("node 13" in line for line in report["structural_feedback"])


def test_structurally_loud_quiet_node_is_flagged_false_negative() -> None:
    data = _motif_graph()
    # Node 0 sits on a laundering cycle but the model scores it quietly.
    report = verify_predictions(data, _probs({3, 12}), y=_labels())

    assert 0 in report["suspected_false_negatives"]
    finding = {f["node"]: f for f in report["findings"]}[0]
    assert finding["score"] >= DEFAULT_VERIFIER_CONFIG.rescue_threshold


def test_false_positive_rate_fires_retrain_signal() -> None:
    data = _motif_graph()
    # Alerts on the cycle (supported) AND on both benign isolated nodes (not).
    report = verify_predictions(
        data, _probs({0, 1, 2, 13, 14}), y=_labels(), config=DEFAULT_VERIFIER_CONFIG
    )
    assert set(report["suspected_false_positives"]) == {13, 14}
    assert report["retrain_signal"] is True


def test_clean_predictions_produce_no_signal() -> None:
    data = _motif_graph()
    report = verify_predictions(
        data, _probs({0, 1, 2, 3, 12}), y=_labels(), config=DEFAULT_VERIFIER_CONFIG
    )
    assert report["suspected_false_positives"] == []
    assert report["suspected_false_negatives"] == []
    assert report["retrain_signal"] is False
    assert report["verified"] is True


def test_verification_is_deterministic() -> None:
    data = _motif_graph()
    probs = _probs({0, 1, 2, 3, 12, 13})
    first = verify_predictions(data, probs, y=_labels())
    second = verify_predictions(data, probs, y=_labels())
    assert first["findings"] == second["findings"]
    assert first["suspected_false_positives"] == second["suspected_false_positives"]


# ---------------------------------------------------------------------------
# Degenerate inputs (crash-resilience contract)
# ---------------------------------------------------------------------------


def test_empty_graph_yields_shape_stable_empty_report() -> None:
    empty_graph = Data(num_nodes=0, edge_index=torch.zeros(2, 0, dtype=torch.long))
    report = verify_predictions(empty_graph, torch.zeros(0))
    assert report["verified"] is False
    assert report["inspected_nodes"] == 0
    assert report["retrain_signal"] is False
    assert set(report) == {
        "verified",
        "reason",
        "verification_score",
        "verified_rate",
        "structural_misclassification_rate",
        "inspected_nodes",
        "alert_nodes",
        "suspected_false_positives",
        "suspected_false_negatives",
        "findings",
        "structural_feedback",
        "retrain_signal",
    }


def test_probability_count_mismatch_is_rejected_gracefully() -> None:
    report = verify_predictions(_motif_graph(), torch.ones(7))
    assert report["verified"] is False
    assert "node count" in report["reason"]


# ---------------------------------------------------------------------------
# Trigger decision + summary shape
# ---------------------------------------------------------------------------


def test_structural_trigger_thresholds() -> None:
    empty = detect_structural_trigger({"inspected_nodes": 0, "alert_nodes": []})
    assert empty["triggered"] is False

    hot = detect_structural_trigger(
        {
            "inspected_nodes": 10,
            "alert_nodes": [1, 2],
            "structural_misclassification_rate": 0.40,
            "verified_rate": 0.60,
        }
    )
    assert hot["triggered"] is True
    assert "misclassification rate" in hot["reason"]

    unsupported = detect_structural_trigger(
        {
            "inspected_nodes": 10,
            "alert_nodes": [1, 2],
            "structural_misclassification_rate": 0.0,
            "verified_rate": 0.20,
        }
    )
    assert unsupported["triggered"] is True
    assert "supported" in unsupported["reason"]

    calm = detect_structural_trigger(
        {
            "inspected_nodes": 10,
            "alert_nodes": [1, 2],
            "structural_misclassification_rate": 0.05,
            "verified_rate": 0.90,
        }
    )
    assert calm["triggered"] is False


def test_summarise_verification_shape() -> None:
    report = verify_predictions(_motif_graph(), _probs({0, 1, 2, 13, 14}), y=_labels())
    summary = summarise_verification(report)
    assert set(summary) == {
        "verification_score",
        "verified_rate",
        "structural_misclassification_rate",
        "inspected_nodes",
        "suspected_false_positives",
        "suspected_false_negatives",
        "retrain_signal",
    }
    assert summary["suspected_false_positives"] == 2
    assert summary["retrain_signal"] is True


# ---------------------------------------------------------------------------
# update_pipeline integration (structural retrain trigger)
# ---------------------------------------------------------------------------


def _import_update_pipeline() -> object:
    return importlib.import_module("scripts.update_pipeline")


def test_combine_triggers_structural_fires_retrain() -> None:
    update = _import_update_pipeline()
    drift = {"drifted": False, "reason": "metric stable"}
    structural = {"decision": {"triggered": True, "reason": "structural misclassification"}}

    combined = update.combine_triggers(drift, structural)
    assert combined["drifted"] is True
    assert combined["structural_trigger"] is True
    assert "structural:" in combined["reason"]


def test_combine_triggers_without_structural_decision() -> None:
    update = _import_update_pipeline()
    drift = {"drifted": True, "reason": "pr_auc dropped 10%"}

    combined = update.combine_triggers(drift, None)
    assert combined["drifted"] is True
    assert combined["structural_trigger"] is False
    assert combined["reason"] == "pr_auc dropped 10%"


def test_structural_verification_disabled_returns_none(tmp_path: Path) -> None:
    update = _import_update_pipeline()
    config = {
        "monitoring": {"substructure_verification": False},
        "paths": {"best_checkpoint": str(tmp_path / "best.pt")},
    }
    assert update.run_structural_verification(config) is None


def test_structural_verification_skips_without_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch
    from torch_geometric.data import Data

    update = _import_update_pipeline()
    tiny = Data(
        num_nodes=3,
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        y=torch.zeros(3, dtype=torch.long),
    )
    monkeypatch.setattr(update, "_load_verification_graph", lambda config: tiny)
    config = {
        "monitoring": {"substructure_verification": True},
        "paths": {"best_checkpoint": str(tmp_path / "missing.pt")},
    }
    assert update.run_structural_verification(config) is None


def test_structural_verification_never_crashes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    update = _import_update_pipeline()

    def _explode(config: dict) -> None:
        raise RuntimeError("graph source exploded")

    monkeypatch.setattr(update, "_load_verification_graph", _explode)
    config = {
        "monitoring": {"substructure_verification": True},
        "paths": {"best_checkpoint": str(tmp_path / "best.pt")},
    }
    assert update.run_structural_verification(config) is None
