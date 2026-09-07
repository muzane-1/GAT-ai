"""Substructure-aware self-verification of GNN predictions.

This module closes the self-evaluation loop of the AML pipeline: after the
GATv2 model assigns suspicion probabilities to accounts, the verifier inspects
the *local topology* around the highest-impact predictions and asks a simple
question — *does the surrounding graph actually look like money laundering?*

For every inspected node it extracts a bounded k-hop ego subgraph
(:func:`torch_geometric.utils.k_hop_subgraph`) and scores three canonical AML
motifs with pure-torch/Python primitives (no extra dependencies):

* **Directed cycle** — layered laundering loops (A→B→C→A). Detected with a
  budget-capped BFS for the shortest directed cycle through the node.
* **Smurfing fan** — high fan-in (many senders) or fan-out (many receivers)
  degree, the hallmark of structuring/smurfing.
* **Dense subgraph / k-core** — tightly connected ego neighbourhoods, found by
  iterative k-core peeling and edge-density on the (undirected) ego graph.

The per-node evidence is combined into a single ``verification_score`` in
``[0, 1]``. Predictions whose structure contradicts the model are surfaced as
``suspected_false_positives`` (alert without structural support) or
``suspected_false_negatives`` (quiet prediction on a structurally loud node),
together with human-readable ``structural_feedback``.

The module is fully decoupled from training: it consumes any PyG ``Data``
object plus a probability tensor, never imports the model, and — per the
repository's crash-resilience contract — degrades to a well-formed empty
report instead of raising on degenerate input. :func:`detect_structural_trigger`
converts a report into a retrain decision compatible with
``scripts/update_pipeline.py``'s drift engine.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

from src.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    "DEFAULT_VERIFIER_CONFIG",
    "SubstructureVerifierConfig",
    "detect_structural_trigger",
    "summarise_verification",
    "verify_predictions",
]


@dataclass(frozen=True, slots=True)
class SubstructureVerifierConfig:
    """Thresholds and work budgets for structural self-verification.

    All attributes are deterministic knobs; nothing here is learned, so the
    verifier is reproducible and auditable in production.
    """

    #: Radius of the ego subgraph extracted around each inspected node.
    k_hop: int = 2
    #: Hard cap on inspected nodes per call (predicted positives first, then
    #: borderline negatives for false-negative rescue).
    top_k: int = 128
    #: Probability at/above which a node is treated as an alert.
    positive_threshold: float = 0.5
    #: Verification score below which an alert lacks structural support (FP).
    confirm_threshold: float = 0.45
    #: Verification score above which a non-alert is structurally loud (FN).
    rescue_threshold: float = 0.75
    #: Longest directed cycle searched through a node (BFS depth bound).
    cycle_max_len: int = 3
    #: Total BFS path expansions allowed per node's cycle search (guard).
    cycle_max_paths: int = 20_000
    #: In-degree at which fan-in smurfing evidence saturates.
    fan_in_threshold: int = 4
    #: Out-degree at which fan-out smurfing evidence saturates.
    fan_out_threshold: int = 4
    #: Minimum ego-subgraph size before density evidence is meaningful.
    min_core_size: int = 3
    #: Undirected ego density that counts as a "dense" subgraph (score 1.0).
    density_target: float = 0.5
    #: Relative contribution of each motif to the verification score. Calibrated
    #: so one saturated motif (0.50 / 0.25+dense) confirms an alert and two
    #: saturated motifs (>=0.75) rescue a false negative.
    motif_weights: dict[str, float] = field(
        default_factory=lambda: {"cycle": 0.50, "fan": 0.25, "dense": 0.25}
    )
    #: Share of structurally contradicted predictions that fires a retrain.
    misclassification_rate_threshold: float = 0.15
    #: Minimum share of alerts that must be structurally supported.
    verified_rate_floor: float = 0.50


DEFAULT_VERIFIER_CONFIG = SubstructureVerifierConfig()


# ---------------------------------------------------------------------------
# Empty-report template (shape-stable for downstream consumers)
# ---------------------------------------------------------------------------


def _empty_report(reason: str) -> dict[str, Any]:
    """Return a well-formed, zeroed verification report."""
    return {
        "verified": False,
        "reason": reason,
        "verification_score": 0.0,
        "verified_rate": 0.0,
        "structural_misclassification_rate": 0.0,
        "inspected_nodes": 0,
        "alert_nodes": [],
        "suspected_false_positives": [],
        "suspected_false_negatives": [],
        "findings": [],
        "structural_feedback": [],
        "retrain_signal": False,
    }


# ---------------------------------------------------------------------------
# Motif primitives (deterministic, budget-capped, dependency-free)
# ---------------------------------------------------------------------------


def _directed_adjacency(edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
    """Deterministic successor lists from an edge index (self-loops dropped)."""
    adj: list[list[int]] = [[] for _ in range(num_nodes)]
    for u, v in zip(edge_index[0].tolist(), edge_index[1].tolist(), strict=False):
        if u != v:
            adj[u].append(v)
    return [sorted(set(nbrs)) for nbrs in adj]


def _cycle_length_through(adj: list[list[int]], start: int, max_len: int, max_paths: int) -> int:
    """Shortest directed cycle through ``start`` (0 when none or budget spent).

    Breadth-first search over simple paths leaving ``start``; the first time
    ``start`` is re-reached yields the minimal cycle length. ``max_paths``
    bounds total edge expansions so adversarial hub nodes cannot stall the
    verifier — an exhausted budget simply reports "no cycle found".
    """
    if not adj[start]:
        return 0
    queue: deque[tuple[int, int, tuple[int, ...]]] = deque()
    for nbr in adj[start]:
        if nbr == start:
            return 1
        queue.append((nbr, 1, (nbr,)))
    expansions = 0
    while queue:
        node, depth, path = queue.popleft()
        for nbr in adj[node]:
            expansions += 1
            if expansions > max_paths:
                return 0
            if nbr == start:
                return depth + 1
            if depth + 1 >= max_len or nbr in path:
                continue
            queue.append((nbr, depth + 1, (*path, nbr)))
    return 0


def _undirected_edges(edge_index: torch.Tensor, num_nodes: int) -> list[tuple[int, int]]:
    """Deduplicated undirected edge list of a (relabeled) subgraph."""
    seen: set[tuple[int, int]] = set()
    for u, v in zip(edge_index[0].tolist(), edge_index[1].tolist(), strict=False):
        if u == v or u >= num_nodes or v >= num_nodes:
            continue
        seen.add((u, v) if u < v else (v, u))
    return sorted(seen)


def _max_core_number(num_nodes: int, edges: list[tuple[int, int]]) -> int:
    """Max k-core of an undirected graph via iterative min-degree peeling."""
    if num_nodes <= 0:
        return 0
    adj: list[set[int]] = [set() for _ in range(num_nodes)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    degrees = {node: len(adj[node]) for node in range(num_nodes)}
    core = 0
    while degrees:
        node = min(degrees, key=lambda n: (degrees[n], n))
        deg = degrees.pop(node)
        core = max(core, deg)
        for nbr in adj[node]:
            if nbr in degrees:
                degrees[nbr] -= 1
    return core


def _density(num_nodes: int, edges: list[tuple[int, int]]) -> float:
    """Edge density of a simple undirected graph in ``[0, 1]``."""
    if num_nodes < 2:
        return 0.0
    return len(edges) / (num_nodes * (num_nodes - 1) / 2.0)


# ---------------------------------------------------------------------------
# Per-node structural evidence
# ---------------------------------------------------------------------------


def _node_evidence(
    data: Data,
    node: int,
    full_in_degree: int,
    full_out_degree: int,
    config: SubstructureVerifierConfig,
) -> dict[str, Any]:
    """Extract and score AML motifs in the k-hop ego subgraph of ``node``.

    Args:
        data: PyG graph (CPU tensors expected; moved defensively).
        node: Global node index under inspection.
        full_in_degree: In-degree of ``node`` in the *whole* graph.
        full_out_degree: Out-degree of ``node`` in the *whole* graph.
        config: Verifier thresholds.

    Returns:
        Evidence mapping with ``node``, ``score`` (0..1), boolean/numeric motif
        fields (``cycle_length``, ``fan_in``, ``fan_out``, ``core_number``,
        ``ego_density``, ``ego_nodes``) and the matched ``motifs`` name list.
    """
    edge_index = data.edge_index.detach().cpu()
    num_nodes = int(data.num_nodes)
    seed = torch.tensor([node], dtype=torch.long)

    # ``k_hop_subgraph`` walks a single direction per call; the AML ego
    # neighbourhood needs both senders and receivers, so extract with each
    # flow and union the node sets before rebuilding the directed sub-edges.
    downstream, _, _, _ = k_hop_subgraph(
        seed,
        num_hops=config.k_hop,
        edge_index=edge_index,
        relabel_nodes=False,
        num_nodes=num_nodes,
        flow="target_to_source",
    )
    upstream, _, _, _ = k_hop_subgraph(
        seed,
        num_hops=config.k_hop,
        edge_index=edge_index,
        relabel_nodes=False,
        num_nodes=num_nodes,
        flow="source_to_target",
    )
    subset = torch.cat([downstream, upstream]).unique()
    local = {global_id: idx for idx, global_id in enumerate(subset.tolist())}
    sub_src: list[int] = []
    sub_dst: list[int] = []
    for u, v in zip(edge_index[0].tolist(), edge_index[1].tolist(), strict=False):
        if u in local and v in local:
            sub_src.append(local[u])
            sub_dst.append(local[v])
    sub_edge_index = torch.tensor([sub_src, sub_dst], dtype=torch.long)
    sub_nodes = len(local)
    local_node = local[node]

    dir_adj = _directed_adjacency(sub_edge_index, sub_nodes)
    und_edges = _undirected_edges(sub_edge_index, sub_nodes)

    cycle_length = _cycle_length_through(
        dir_adj, local_node, config.cycle_max_len, config.cycle_max_paths
    )
    fan_in_score = min(1.0, full_in_degree / max(1, config.fan_in_threshold))
    fan_out_score = min(1.0, full_out_degree / max(1, config.fan_out_threshold))
    fan_score = max(fan_in_score, fan_out_score)
    core_number = _max_core_number(sub_nodes, und_edges)
    ego_density = _density(sub_nodes, und_edges)
    dense_score = (
        min(1.0, ego_density / max(1e-9, config.density_target))
        if sub_nodes >= config.min_core_size
        else 0.0
    )

    weights = config.motif_weights
    score = min(
        1.0,
        weights.get("cycle", 0.0) * (1.0 if cycle_length else 0.0)
        + weights.get("fan", 0.0) * fan_score
        + weights.get("dense", 0.0) * dense_score,
    )

    motifs: list[str] = []
    if cycle_length:
        motifs.append(f"directed_cycle_len_{cycle_length}")
    if fan_out_score >= 1.0:
        motifs.append("fan_out_smurfing")
    if fan_in_score >= 1.0:
        motifs.append("fan_in_smurfing")
    if dense_score >= 1.0:
        motifs.append("dense_subgraph")

    return {
        "node": node,
        "score": float(score),
        "cycle_length": cycle_length,
        "fan_in": full_in_degree,
        "fan_out": full_out_degree,
        "core_number": core_number,
        "ego_density": round(float(ego_density), 6),
        "ego_nodes": sub_nodes,
        "motifs": motifs,
    }


def verify_predictions(
    data: Data,
    probs: torch.Tensor,
    y: torch.Tensor | None = None,
    *,
    config: SubstructureVerifierConfig | None = None,
    candidate_nodes: list[int] | None = None,
) -> dict[str, Any]:
    """Verify model predictions against local AML graph substructures.

    The top-``top_k`` alerts (highest predicted probability) are inspected
    first; the most borderline non-alerts are inspected afterwards so that
    structurally loud but quietly scored accounts can be flagged as suspected
    false negatives.

    Args:
        data: PyG graph (``edge_index`` required, shape ``(2, E)``).
        probs: Positive-class probability per node (shape ``(N,)``).
        y: Optional ground-truth node labels (``0/1``). When given, suspected
            contradictions are cross-checked against the labels and only
            label-confirming cases count toward the misclassification rate.
        config: Optional verifier configuration override.
        candidate_nodes: Optional explicit inspection list (bypasses top-k
            selection); useful for targeted audits.

    Returns:
        Report mapping (always shape-stable, even for empty/degenerate input):
        ``verification_score``, ``verified_rate``,
        ``structural_misclassification_rate``, ``inspected_nodes``,
        ``alert_nodes``, ``suspected_false_positives``,
        ``suspected_false_negatives``, ``findings``, ``structural_feedback``
        and ``retrain_signal``.
    """
    cfg = config or DEFAULT_VERIFIER_CONFIG

    try:
        probs_t = torch.as_tensor(probs, dtype=torch.float32).detach().cpu().flatten()
        edge_index = data.edge_index.detach().cpu()
        num_nodes = int(data.num_nodes)
    except Exception as exc:  # noqa: BLE001 - crash-resilience contract
        logger.warning("verifier_input_error", extra={"error": str(exc)})
        return _empty_report(f"malformed verifier input: {exc}")

    if edge_index.dim() != 2 or edge_index.size(0) != 2 or num_nodes == 0:
        return _empty_report("graph has no nodes or edge_index is malformed")
    if probs_t.numel() != num_nodes:
        return _empty_report(f"probability count ({probs_t.numel()}) != node count ({num_nodes})")

    # Deterministic ordering (stable sort on descending probability).
    order = torch.argsort(probs_t, descending=True, stable=True).tolist()
    probs_list = probs_t.tolist()
    is_alert = [p >= cfg.positive_threshold for p in probs_list]
    alert_nodes = [n for n in order if is_alert[n]][: cfg.top_k]
    # Borderline negatives: highest-probability nodes still below the threshold.
    borderline_negatives = [n for n in order if not is_alert[n]][: cfg.top_k]
    inspected = (
        candidate_nodes if candidate_nodes is not None else alert_nodes + borderline_negatives
    )

    in_degree = torch.bincount(edge_index[1], minlength=num_nodes).tolist()
    out_degree = torch.bincount(edge_index[0], minlength=num_nodes).tolist()
    findings = [
        _node_evidence(data, node, int(in_degree[node]), int(out_degree[node]), cfg)
        for node in inspected
    ]

    label_values: list[int] = []
    if y is not None:
        try:
            label_values = torch.as_tensor(y).detach().cpu().flatten().tolist()
        except Exception:  # noqa: BLE001 - labels are optional, never fatal
            label_values = []

    suspected_fp: list[int] = []
    suspected_fn: list[int] = []
    confirmed_fp = confirmed_fn = 0
    feedback: list[str] = []
    for evidence in findings:
        node = evidence["node"]
        predicted_alert = is_alert[node]
        score = evidence["score"]
        label = int(label_values[node]) if node < len(label_values) else None

        if predicted_alert and score < cfg.confirm_threshold:
            suspected_fp.append(node)
            if label == 1:
                note = "label agrees with model but structure is quiet"
            else:
                confirmed_fp += 1
                note = "confirmed false positive" if label == 0 else "unsupported alert"
            feedback.append(
                f"node {node}: alert p={probs_list[node]:.3f} lacks structural support "
                f"(score={score:.2f}) - {note}"
            )
        elif not predicted_alert and score >= cfg.rescue_threshold:
            suspected_fn.append(node)
            if label == 0:
                note = "label agrees with model but structure is loud"
            else:
                confirmed_fn += 1
                note = "confirmed false negative" if label == 1 else "missed structural alert"
            feedback.append(
                f"node {node}: quiet p={probs_list[node]:.3f} but strong structural evidence "
                f"(score={score:.2f}) - {note}"
            )

    inspected_alerts = [e for e in findings if is_alert[e["node"]]]
    supported = [e for e in inspected_alerts if e["score"] >= cfg.confirm_threshold]
    verification_score = (
        sum(e["score"] for e in inspected_alerts) / len(inspected_alerts)
        if inspected_alerts
        else 0.0
    )
    verified_rate = len(supported) / len(inspected_alerts) if inspected_alerts else 0.0

    contradicted = (
        confirmed_fp + confirmed_fn if label_values else len(suspected_fp) + len(suspected_fn)
    )
    # Denominator = prediction-relevant nodes: alerts plus structural rescues.
    relevant = len(alert_nodes) + len(suspected_fn)
    misclassification_rate = contradicted / relevant if relevant else 0.0

    report: dict[str, Any] = {
        "verified": bool(findings)
        and misclassification_rate <= cfg.misclassification_rate_threshold,
        "reason": "verified" if findings else "no candidates inspected",
        "verification_score": round(float(verification_score), 6),
        "verified_rate": round(float(verified_rate), 6),
        "structural_misclassification_rate": round(float(misclassification_rate), 6),
        "inspected_nodes": len(findings),
        "alert_nodes": alert_nodes,
        "suspected_false_positives": suspected_fp,
        "suspected_false_negatives": suspected_fn,
        "findings": findings,
        "structural_feedback": feedback,
        "retrain_signal": False,
    }
    report["retrain_signal"] = detect_structural_trigger(report, config=cfg)["triggered"]
    if feedback:
        logger.info("substructure_verification_feedback", extra={"findings": len(feedback)})
    return report


def summarise_verification(report: dict[str, Any]) -> dict[str, Any]:
    """Reduce a full verification report to registry/metrics-history shape."""
    return {
        "verification_score": float(report.get("verification_score", 0.0)),
        "verified_rate": float(report.get("verified_rate", 0.0)),
        "structural_misclassification_rate": float(
            report.get("structural_misclassification_rate", 0.0)
        ),
        "inspected_nodes": int(report.get("inspected_nodes", 0)),
        "suspected_false_positives": len(report.get("suspected_false_positives", [])),
        "suspected_false_negatives": len(report.get("suspected_false_negatives", [])),
        "retrain_signal": bool(report.get("retrain_signal", False)),
    }


def detect_structural_trigger(
    report: dict[str, Any],
    *,
    config: SubstructureVerifierConfig | None = None,
    rate_threshold: float | None = None,
    verified_floor: float | None = None,
) -> dict[str, Any]:
    """Convert a verification report into a drift-style retrain decision.

    Mirrors the decision shape of ``scripts.update_pipeline.detect_drift`` so
    the two engines compose cleanly inside the update pipeline.

    Args:
        report: Output of :func:`verify_predictions`.
        config: Verifier config supplying default thresholds.
        rate_threshold: Overrides the structural misclassification threshold.
        verified_floor: Overrides the verified-rate floor.

    Returns:
        ``{"triggered": bool, "reason": str, "misclassification_rate": float,
        "verified_rate": float}``.
    """
    cfg = config or DEFAULT_VERIFIER_CONFIG
    rate_threshold = (
        cfg.misclassification_rate_threshold if rate_threshold is None else rate_threshold
    )
    verified_floor = cfg.verified_rate_floor if verified_floor is None else verified_floor
    rate = float(report.get("structural_misclassification_rate", 0.0))
    support = float(report.get("verified_rate", 0.0))
    inspected = int(report.get("inspected_nodes", 0))

    if inspected == 0:
        return {
            "triggered": False,
            "reason": "no nodes inspected - nothing to verify",
            "misclassification_rate": 0.0,
            "verified_rate": 0.0,
        }
    if rate > rate_threshold:
        return {
            "triggered": True,
            "reason": f"structural misclassification rate {rate:.2%} exceeds "
            f"threshold {rate_threshold:.2%}",
            "misclassification_rate": rate,
            "verified_rate": support,
        }
    if report.get("alert_nodes") and support < verified_floor:
        return {
            "triggered": True,
            "reason": f"only {support:.2%} of alerts are structurally supported "
            f"(floor {verified_floor:.2%})",
            "misclassification_rate": rate,
            "verified_rate": support,
        }
    return {
        "triggered": False,
        "reason": "predictions structurally consistent",
        "misclassification_rate": rate,
        "verified_rate": support,
    }
