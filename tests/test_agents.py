"""Unit tests for the LangGraph/CrewAI-ready discovery agent scaffolding."""

from typing import Any

from src.pipeline.discovery.agents import (
    DiscoveryState,
    SequentialDiscoveryGraph,
    build_discovery_graph,
    discover_node,
    plan_queries_node,
    run_discovery_agents,
    verify_node,
)


def test_plan_queries_node_returns_queries() -> None:
    """The planner node produces a deterministic, non-empty query plan."""
    state: DiscoveryState = {"assets": ["btc"], "aml_terms": ["aml"], "formats": ["csv"]}
    update = plan_queries_node(state)
    assert update["queries"]
    assert all("btc" in query for query in update["queries"])


def test_discover_node_offline_is_deterministic() -> None:
    """Offline discovery short-circuits to an empty candidate list."""
    state: DiscoveryState = {"offline": True}
    update = discover_node(state)
    assert update["candidates"] == []
    assert update.get("errors", []) == []


def test_verify_node_without_candidates_is_noop() -> None:
    """No candidates means no verification work and no errors."""
    update = verify_node({"candidates": []})
    assert update["verified"] == []
    assert update["summary"] == []


def test_sequential_graph_merges_node_updates() -> None:
    """The dependency-free graph runs plan → discover → verify in order."""
    graph = SequentialDiscoveryGraph()
    executed: list[str] = []

    def tracking_node(state: DiscoveryState) -> dict[str, Any]:
        executed.append("tracking")
        return {}

    graph.add_node("tracking", tracking_node)
    result = graph.invoke({"offline": True, "assets": ["btc"], "aml_terms": ["aml"]})
    assert executed == ["tracking"]
    assert result["queries"]
    assert result["candidates"] == []
    assert result["verified"] == []


def test_build_discovery_graph_has_invoke() -> None:
    """Both backends expose the same LangGraph-style invoke interface."""
    graph = build_discovery_graph()
    assert hasattr(graph, "invoke")


def test_run_discovery_agents_end_to_end_offline() -> None:
    """Full agent run stays deterministic and error-free in offline mode."""
    result = run_discovery_agents(
        {
            "offline": True,
            "assets": ["btc", "eth"],
            "aml_terms": ["aml", "fraud"],
            "formats": ["csv"],
            "max_queries": 4,
        }
    )
    assert result["queries"]
    assert result["candidates"] == []
    assert result["verified"] == []
    assert result["summary"] == []
    assert result.get("errors", []) == []
