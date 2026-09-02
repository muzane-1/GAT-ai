"""Readiness dry-run: 1-epoch mini-batch smoke test + checkpoint round-trip.

Mirrors the pre-training sanity script (``scripts/verify_readiness.py``) so the
full ``fetch_to_pyg`` → ``NeighborLoader`` → forward → loss → backward →
optimiser-step → checkpoint-integrity chain is verified automatically in CI.
"""

from importlib import import_module
from math import isfinite
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@pytest.fixture()
def readiness_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Any]:
    """Run the 1-epoch dry-run once into an isolated temp checkpoints dir."""
    verify = import_module("scripts.verify_readiness")
    ckpt_path = tmp_path_factory.mktemp("checkpoints") / "test_model.pt"
    return verify.dry_run(
        config_path=str(CONFIG_PATH),
        epochs=1,
        checkpoint_path=str(ckpt_path),
        keep_checkpoint=False,
    )


def test_dry_run_pipeline_ready(readiness_report: dict[str, Any]) -> None:
    """The full fetch→sample→forward→loss→backward→step loop succeeds."""
    assert readiness_report["pipeline_ready"] is True
    assert readiness_report["graph"]["nodes"] > 0
    assert readiness_report["graph"]["edges"] > 0
    assert readiness_report["neighbor_loader"]["mini_batches"] >= 1
    assert readiness_report["optimizer"]["step"] is True


def test_dry_run_shapes_and_losses_finite(readiness_report: dict[str, Any]) -> None:
    """x/edge_index/edge_attr/y shapes align and both losses are finite."""
    checks = readiness_report["checks"]
    assert all(checks.values()), checks
    loss = readiness_report["loss"]
    assert isfinite(loss["focal"])
    assert isfinite(loss["weighted_bce"])
    assert isfinite(readiness_report["optimizer"]["grad_norm"])


def test_dry_run_checkpoint_roundtrip(readiness_report: dict[str, Any]) -> None:
    """Dummy checkpoint saves, reloads with matching state, and cleans up."""
    ckpt = readiness_report["checkpoint"]
    assert ckpt["state_match"] is True
    assert ckpt["cleaned"] is True
    assert not Path(ckpt["path"]).exists()
