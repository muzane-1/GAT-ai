"""Unit tests for the step-based orchestration layer (ZenML/Kedro style)."""

from typing import Any

import pandas as pd
import pytest
from torch_geometric.data import Data

from src.pipeline.orchestration.pipeline import Pipeline, build_default_pipeline, run_pipeline
from src.pipeline.orchestration.steps import Step, fetch_step, step, transform_step, validate_step

SMALL_CONFIG: dict[str, Any] = {
    "data": {
        "fallback_n_accounts": 40,
        "fallback_n_transactions": 120,
        "fallback_fraud_ratio": 0.1,
        "fallback_seed": 42,
    }
}


def test_step_decorator_metadata() -> None:
    """The ``step`` decorator captures name, inputs, and outputs."""

    @step("double", inputs=("x",), outputs=("y",), description="doubles a value")
    def double(x: int) -> int:
        return x * 2

    assert isinstance(double, Step)
    assert double.name == "double"
    assert double.inputs == ("x",)
    assert double.outputs == ("y",)
    assert double.description == "doubles a value"
    assert double.function(2) == 4


def test_default_pipeline_has_three_layers() -> None:
    """The canonical pipeline is fetch → validate → transform."""
    pipeline = build_default_pipeline()
    assert [s.name for s in pipeline.steps] == ["fetch", "validate", "transform"]
    assert pipeline.steps[0].inputs == ()
    assert pipeline.steps[1].inputs == ("raw_df",)
    assert pipeline.steps[2].inputs == ("validated_df",)


def test_pipeline_threads_artifacts_through_catalog() -> None:
    """Outputs of one step are bound to the declared inputs of the next."""

    @step("produce", outputs=("value",))
    def produce() -> int:
        return 21

    @step("consume", inputs=("value",), outputs=("result",))
    def consume(value: int) -> int:
        return value * 2

    pipeline = Pipeline([produce, consume])
    catalog = pipeline.run()
    assert catalog == {"value": 21, "result": 42}


def test_run_pipeline_end_to_end() -> None:
    """fetch → validate → transform yields a graph plus quality metrics."""
    catalog = run_pipeline(config=SMALL_CONFIG)
    assert catalog["fetch_stats"]["provenance"] == "synthetic"
    assert isinstance(catalog["raw_df"], pd.DataFrame)
    assert catalog["validation_report"]["pandera_validated"] is True
    assert catalog["validation_report"]["rows"] > 0
    assert isinstance(catalog["graph"], Data)
    assert catalog["graph"].num_nodes > 0
    assert catalog["scaler"] is not None


def test_fetch_step_falls_back_to_synthetic() -> None:
    """A missing configured source degrades to the synthetic generator."""
    config = {"data": {"raw_source": "definitely/not/a/file.csv", "fallback_generate": True}}
    df, stats = fetch_step(config=config)
    assert stats["provenance"] == "synthetic"
    assert stats["rows"] == len(df)


def test_transform_step_builds_graph() -> None:
    """The transform step converts a validated frame into a PyG graph."""
    raw_df, _ = fetch_step(config=SMALL_CONFIG)
    validated, report = validate_step(raw_df)
    graph, scaler = transform_step(validated)
    assert isinstance(graph, Data)
    assert scaler is not None
    assert report["aml_ratio"] > 0


def test_pipeline_preserves_prepopulated_catalog() -> None:
    """A pre-populated catalog entry satisfies a step's declared input."""

    @step("echo", inputs=("seed",), outputs=("echo",))
    def echo(seed: int) -> int:
        return seed

    pipeline = Pipeline([echo])
    catalog = pipeline.run({"seed": 7})
    assert catalog["echo"] == 7


def test_run_pipeline_with_explicit_missing_source_falls_back() -> None:
    """An unreachable explicit source still produces a usable graph."""
    catalog = run_pipeline(source="definitely/not/a/file.csv", config=SMALL_CONFIG)
    assert isinstance(catalog["graph"], Data)
    assert catalog["validation_report"]["rows"] > 0


@pytest.mark.parametrize("missing", ["raw_df", "validated_df", "graph"])
def test_catalog_contains_all_core_artifacts(missing: str) -> None:
    """Every core artifact is present in the final catalog."""
    catalog = run_pipeline(config=SMALL_CONFIG)
    assert missing in catalog
