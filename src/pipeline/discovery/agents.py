"""Agentic discovery scaffolding — LangGraph / CrewAI ready.

The deterministic discovery core in
:mod:`src.pipeline.discovery.auto_fetch` is wrapped as *graph nodes*: pure
functions over a shared :class:`DiscoveryState` blackboard that return partial
state updates. That is exactly the node contract of LangGraph's ``StateGraph``
and trivially adaptable to CrewAI tasks, while keeping the deterministic core
dependency-free.

Two execution backends are supported:

* **LangGraph** — when ``langgraph`` is installed, :func:`build_discovery_graph`
  returns a compiled ``StateGraph`` running ``plan_queries → discover → verify``.
* **Built-in fallback** — :class:`SequentialDiscoveryGraph` executes the same
  nodes in order with zero additional dependencies (default in CI).

Plugging an LLM planner (LangGraph node, CrewAI agent, ...) means replacing
:func:`plan_queries_node` with a callable sharing its signature.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict, cast

from src.pipeline.discovery.auto_fetch import (
    DatasetCandidate,
    discover_candidates,
    generate_search_queries,
    verified_summary,
    verify_candidates,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DiscoveryState(TypedDict, total=False):
    """Shared blackboard state flowing through the discovery agent nodes."""

    # --- planner inputs -----------------------------------------------------
    assets: list[str]
    aml_terms: list[str]
    formats: list[str]
    providers: str
    max_queries: int
    per_provider_limit: int
    offline: bool
    download_dir: str
    top_k: int
    # --- node outputs -------------------------------------------------------
    queries: list[str]
    candidates: list[DatasetCandidate]
    verified: list[Any]
    summary: list[dict[str, Any]]
    errors: list[str]


def plan_queries_node(state: DiscoveryState) -> dict[str, Any]:
    """Heuristic (LLM-swappable) query planner node."""
    queries = generate_search_queries(
        assets=state.get("assets") or (),
        aml_terms=state.get("aml_terms") or (),
        formats=state.get("formats") or (),
        max_queries=state.get("max_queries", 24),
    )
    return {"queries": queries}


def discover_node(state: DiscoveryState) -> dict[str, Any]:
    """Fan the planned queries out to every enabled discovery provider."""
    try:
        candidates = discover_candidates(
            assets=state.get("assets") or (),
            aml_terms=state.get("aml_terms") or (),
            formats=state.get("formats") or (),
            providers=state.get("providers", "all"),
            max_queries=state.get("max_queries", 24),
            per_provider_limit=state.get("per_provider_limit", 8),
            offline=state.get("offline", False),
        )
    except Exception as exc:  # noqa: BLE001 - discovery must degrade gracefully
        logger.warning("discover_node_failed", extra={"error": str(exc)})
        return {"candidates": [], "errors": [f"discover_failed: {exc}"]}
    return {"candidates": candidates}


def verify_node(state: DiscoveryState) -> dict[str, Any]:
    """Download and re-score the top-K candidates on their real raw bytes."""
    candidates = state.get("candidates") or []
    if not candidates:
        return {"verified": [], "summary": []}
    try:
        verified = verify_candidates(
            candidates,
            state.get("download_dir", "data/discovery"),
            top_k=state.get("top_k", 3),
            strict=False,  # keep weak candidates; summary exposes quality gates
        )
    except Exception as exc:  # noqa: BLE001 - verification must degrade gracefully
        logger.warning("verify_node_failed", extra={"error": str(exc)})
        return {"verified": [], "summary": [], "errors": [f"verify_failed: {exc}"]}
    return {"verified": verified, "summary": verified_summary(verified)}


#: Default node wiring: plan → discover → verify.
DISCOVERY_NODES: tuple[tuple[str, Callable[[DiscoveryState], dict[str, Any]]], ...] = (
    ("plan_queries", plan_queries_node),
    ("discover", discover_node),
    ("verify", verify_node),
)


class SequentialDiscoveryGraph:
    """Dependency-free executor with a LangGraph-compatible interface."""

    def __init__(self) -> None:
        self._nodes: list[tuple[str, Callable[[DiscoveryState], dict[str, Any]]]] = list(
            DISCOVERY_NODES
        )

    def add_node(
        self, name: str, fn: Callable[[DiscoveryState], dict[str, Any]]
    ) -> SequentialDiscoveryGraph:
        """Append a custom node (agent) to the end of the sequence."""
        self._nodes.append((name, fn))
        return self

    def invoke(self, state: DiscoveryState | None = None) -> DiscoveryState:
        """Run every node in order, merging partial state updates."""
        merged: DiscoveryState = dict(state or {})  # type: ignore[assignment]
        for name, fn in self._nodes:
            update = fn(merged)
            merged.update(cast(DiscoveryState, update))
            logger.info("agent_node_completed", extra={"node": name})
        return merged


def build_discovery_graph() -> Any:
    """Compile the plan → discover → verify graph.

    Returns a compiled LangGraph ``StateGraph`` when ``langgraph`` is
    installed, otherwise a :class:`SequentialDiscoveryGraph`.
    """
    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        return SequentialDiscoveryGraph()

    graph: Any = StateGraph(DiscoveryState)
    graph.add_node("plan_queries", plan_queries_node)
    graph.add_node("discover", discover_node)
    graph.add_node("verify", verify_node)
    graph.set_entry_point("plan_queries")
    graph.add_edge("plan_queries", "discover")
    graph.add_edge("discover", "verify")
    graph.add_edge("verify", END)
    return graph.compile()


def run_discovery_agents(state: DiscoveryState | None = None) -> DiscoveryState:
    """Execute the agentic discovery graph end to end."""
    graph = build_discovery_graph()
    result = graph.invoke(dict(state or {}))
    return result  # type: ignore[return-value]
