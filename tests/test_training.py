"""End-to-end pipeline integrity: short training run + drift detection."""

import json
from pathlib import Path
from typing import Any

import pytest

from src.training.train import train_model
from src.utils import load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


@pytest.fixture(scope="session")
def short_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Execute a real (short) training run once for the module."""
    config = load_config(CONFIG_PATH)
    run_dir = tmp_path_factory.mktemp("checkpoints")
    summary = train_model(config, epochs=2, run_dir=run_dir, quiet=True)
    summary["_run_dir"] = str(run_dir)
    return summary


def test_short_training_produces_checkpoint(short_run: dict[str, Any]) -> None:
    """A short training run writes a usable checkpoint and metrics."""
    checkpoint = Path(short_run["checkpoint"])
    assert checkpoint.exists()
    assert "test_metrics" in short_run
    assert "best_val_metrics" in short_run


def test_drift_detector_flags_missing_history() -> None:
    """Drift decision fires when metrics history has never been written."""
    import scripts.update_pipeline as update_pipeline

    decision = update_pipeline.detect_drift([], metric="pr_auc")
    assert decision["drifted"] is True


def test_drift_detector_relative_drop() -> None:
    """A relative drop in the tracked metric triggers retraining."""
    import scripts.update_pipeline as update_pipeline

    history = [
        {
            "timestamp": 1,
            "summary": {"test_metrics": {"pr_auc": 0.9}},
        },
        {
            "timestamp": 2,
            "summary": {"test_metrics": {"pr_auc": 0.80}},
        },
    ]
    decision = update_pipeline.detect_drift(
        history, metric="pr_auc", relative_threshold=0.05, absolute_floor=0.0
    )
    assert decision["drifted"] is True
    assert "dropped" in decision["reason"]


def test_metrics_history_schema(short_run: dict[str, Any]) -> None:
    """The appended metrics history follows the documented schema."""
    config = load_config(CONFIG_PATH)
    metrics_file = Path(config["paths"]["metrics_history"])
    history = json.loads(metrics_file.read_text())
    latest = history[-1]
    assert "timestamp" in latest and "summary" in latest
    assert "test_metrics" in latest["summary"]
