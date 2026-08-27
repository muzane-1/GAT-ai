"""Automated retraining engine — checkpoint tracking and drift detection.

This script is the production hook: it is invoked on a schedule (CI cron,
Airflow DAG, or cron) and decides whether the deployed GNN model has drifted
enough to trigger a full retraining cycle.

Logic:

1. Load the central config and the metrics history written by every
   training run.
2. Compute the most recent primary metric (default: PR-AUC) and compare it
   against the historical best relative drop threshold and the absolute
   floor from ``config.monitoring``.
3. If drift is detected, trigger :func:`src.training.train.train_model`
   and register the newly produced checkpoint in ``registry.json``.

Exit code ``0`` either way; message lines detail the decision.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train import train_model  # noqa: E402
from src.utils import get_logger, load_config  # noqa: E402

logger = get_logger("update_pipeline")

REGISTRY_FILE = "registry.json"


def load_metrics(path: Path) -> list[dict[str, Any]]:
    """Read the metrics history array; empty list when absent."""
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("Unreadable metrics file: %s", path)
        return []


def primary_metric(entry: dict[str, Any], metric: str) -> float | None:
    """Extract a primary metric value from a history summary entry."""
    metrics = entry.get("summary", {}).get("test_metrics", {})
    value = metrics.get(metric)
    return float(value) if value is not None else None


def detect_drift(
    history: list[dict[str, Any]],
    metric: str = "pr_auc",
    relative_threshold: float = 0.05,
    absolute_floor: float = 0.60,
) -> dict[str, Any]:
    """Decide whether the tracked metric has drifted below acceptable bounds.

    Args:
        history: Metrics history entries.
        metric: Metric key to track (e.g. ``pr_auc``).
        relative_threshold: Maximum tolerated drop vs. the historical best.
        absolute_floor: Hard lower bound under which retraining always fires.

    Returns:
        Decision mapping with keys ``drifted``, ``reason``, ``latest``,
        and ``best``.
    """
    values = [(entry["timestamp"], primary_metric(entry, metric)) for entry in history]
    values = [(ts, value) for ts, value in values if value is not None]
    if not values:
        return {"drifted": True, "reason": "no metrics available", "latest": None, "best": None}

    best = max(v for _, v in values)
    latest = values[-1][1]
    if latest < absolute_floor:
        return {
            "drifted": True,
            "reason": f"{metric}={latest:.4f} below absolute floor {absolute_floor}",
            "latest": latest,
            "best": best,
        }
    if best > 0 and (best - latest) / best > relative_threshold:
        return {
            "drifted": True,
            "reason": f"{metric} dropped {(best - latest) / best:.1%} vs historical best",
            "latest": latest,
            "best": best,
        }
    return {"drifted": False, "reason": "metric stable", "latest": latest, "best": best}


def register_checkpoint(registry_path: Path, checkpoint_path: str, metrics: dict[str, Any]) -> None:
    """Append a checkpoint entry to the registry file."""
    registry: list[dict[str, Any]] = []
    if registry_path.exists():
        registry = json.loads(registry_path.read_text())
    registry.append(
        {
            "timestamp": time.time(),
            "checkpoint": checkpoint_path,
            "test_metrics": metrics.get("test_metrics", {}),
            "val_metrics": metrics.get("best_val_metrics", {}),
        }
    )
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2))
    logger.info("Registered checkpoint %s", checkpoint_path)


def run(config_path: str = "config/config.yaml") -> dict[str, Any]:
    """Execute monitoring; retrain and register when drift is detected.

    Args:
        config_path: YAML config path.

    Returns:
        Decision result mapping (same as :func:`detect_drift` plus an
        optional ``retrained`` flag and checkpoint path).
    """
    config = load_config(config_path)
    monitoring = config["monitoring"]
    paths = config["paths"]

    history = load_metrics(Path(paths["metrics_history"]))
    decision = detect_drift(
        history,
        metric=monitoring.get("drift_metric", "pr_auc"),
        relative_threshold=monitoring.get("drift_threshold", 0.05),
        absolute_floor=monitoring.get("absolute_floor", 0.0),
    )
    logger.info("drift decision: %s", decision["reason"])

    if decision["drifted"]:
        logger.info("Triggering retraining")
        summary = train_model(config)
        registry_path = Path(paths["checkpoints_dir"]) / REGISTRY_FILE
        register_checkpoint(registry_path, summary["checkpoint"], summary)
        decision["retrained"] = True
        decision["checkpoint"] = summary["checkpoint"]
    else:
        decision["retrained"] = False

    return decision


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Monitor model drift and trigger retraining")
    parser.add_argument("--config", default="config/config.yaml", help="YAML config path")
    args = parser.parse_args()
    decision = run(args.config)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
