"""End-to-end dry-run: Agentic Discovery → Playwright fetch → Sanitation →
PyG artifact creation → Modal Volume mount.

Network- and browser-free: every remote boundary (HTTP search, Playwright
scraping) is mocked, so the suite verifies the crash-resilient contract —
any failing stage deterministically degrades to the synthetic graph generator —
plus the artifact round-trip into the (simulated) ``project-data-vol`` mount.
"""

import asyncio
from importlib import import_module
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from src.pipeline.discovery.ingestion import CANONICAL_COLUMNS, generate_synthetic_transactions
from src.storage import pipeline as storage_pipeline
from src.training import modal_train
from src.training.train import train_model
from src.utils import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]

AUTO_FETCH = import_module("src.pipeline.discovery.auto_fetch")
SCRAPER = import_module("src.pipeline.discovery.scraper")


def _valid_frame() -> pd.DataFrame:
    """Deterministic synthetic transaction table (canonical schema)."""
    return generate_synthetic_transactions(n_accounts=30, n_transactions=120, seed=11)


def _training_frame() -> pd.DataFrame:
    """Larger synthetic table with enough positives for stratified splits."""
    return generate_synthetic_transactions(
        n_accounts=200, n_transactions=800, fraud_ratio=0.10, seed=11
    )


@pytest.fixture(autouse=True)
def _clear_discovery_cache() -> None:
    """Provider results are cached for a few minutes; keep tests isolated."""
    AUTO_FETCH._discovery_cache.clear()
    yield
    AUTO_FETCH._discovery_cache.clear()


# ---------------------------------------------------------------------------
# 1. Simulated total network failure → deterministic synthetic fallback
# ---------------------------------------------------------------------------


def test_network_failure_falls_back_to_synthetic_graph() -> None:
    """A dead network at every remote path still yields a valid PyG graph."""
    with mock.patch(
        "src.pipeline.discovery.ingestion.requests.get",
        side_effect=RuntimeError("network down"),
    ):
        data, stats = AUTO_FETCH.fetch_to_pyg(sources=["https://unreachable.invalid/tx.csv"])

    assert stats["provenance"] == "synthetic"
    assert stats["normalized_amounts"] is True
    assert data.num_nodes > 0 and data.num_edges > 0
    assert data.x is not None and data.x.shape[1] == 9
    assert data.edge_index.shape[0] == 2


def test_scraper_failure_degrades_search_to_empty() -> None:
    """A crashing browser batch returns no candidates instead of raising."""

    class ExplodingScraper:
        def __init__(self, config: object, audio_solver: object = None) -> None:
            pass

        async def __aenter__(self) -> "ExplodingScraper":
            return self

        async def __aexit__(self, *exc: object) -> bool:
            return False

        async def scrape_json(self, url: str) -> list[dict[str, object]]:
            raise RuntimeError("browser crashed")

    with mock.patch.object(SCRAPER, "PlaywrightScraper", ExplodingScraper):
        records = asyncio.run(SCRAPER.scrape_urls(["https://a.example/x.json"]))

    assert records == []


# ---------------------------------------------------------------------------
# 2. Discovery → Playwright delegation (src.pipeline.discovery.scraper)
# ---------------------------------------------------------------------------


def test_search_web_browser_delegates_to_scraper(monkeypatch: pytest.MonkeyPatch) -> None:
    """The browser fallback routes keyless search through the Playwright scraper."""
    captured: dict[str, object] = {}

    async def fake_scrape_urls(urls: list[str], *, config: object = None) -> list[object]:
        captured["urls"] = urls
        return [
            {
                "items": [
                    {
                        "html_url": "https://github.com/acme/aml-graph",
                        "name": "aml-graph",
                        "description": "bitcoin laundering edges",
                    }
                ]
            },
            {"unrelated": True},
            "garbage-not-a-dict",
        ]

    monkeypatch.setattr(SCRAPER, "scrape_urls", fake_scrape_urls)
    candidates = AUTO_FETCH.search_web_browser("btc money laundering")

    assert captured["urls"], "keyless JSON endpoints must be queried"
    assert len(candidates) == 1
    assert candidates[0].provider == "web-browser"
    assert candidates[0].id.startswith("web-browser:")
    assert candidates[0].title == "aml-graph"


def test_search_web_resilient_falls_back_to_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the requests-based search fails, the Playwright path is engaged."""
    monkeypatch.setattr(AUTO_FETCH, "search_web", lambda *args, **kwargs: [])
    called: dict[str, list[str]] = {}

    async def fake_scrape_urls(urls: list[str], **kwargs: object) -> list[object]:
        called["urls"] = urls
        return []

    monkeypatch.setattr(SCRAPER, "scrape_urls", fake_scrape_urls)

    assert AUTO_FETCH.search_web_resilient("ethereum fraud") == []
    assert called["urls"]


def test_candidates_from_json_records_is_malformed_tolerant() -> None:
    records: list[object] = [None, 42, {"items": ["bad", {"name": "no-url"}]}, "x"]
    assert AUTO_FETCH.candidates_from_json_records(records) == []


# ---------------------------------------------------------------------------
# 3. Seamless handoff: discovery → strict storage sanitation
# ---------------------------------------------------------------------------


def test_handoff_to_storage_persists_parquet_and_pyg(tmp_path: Path) -> None:
    """A valid dataset is sanitized, log-normalized, and persisted."""
    frame = _valid_frame()
    results = AUTO_FETCH.handoff_to_storage([frame], artifact_root=tmp_path / "vol")

    assert len(results) == 1
    entry = results[0]
    assert entry["status"] == "persisted"
    parquet_path = Path(entry["parquet"])
    pyg_path = Path(entry["pyg"])
    assert parquet_path.exists() and pyg_path.exists()

    stored = pd.read_parquet(parquet_path)
    assert list(stored.columns) == CANONICAL_COLUMNS

    merged = stored.merge(frame, on="tx_id", suffixes=("", "_raw"))
    assert len(merged) == len(frame)  # no rows lost (unique tx_id, no dupes)
    assert np.allclose(merged["amount"], np.log1p(merged["amount_raw"]))

    data, scaler = storage_pipeline.load_pyg_dataset(pyg_path)
    assert data.num_nodes > 0 and data.num_edges == len(frame)
    assert scaler is not None


def test_handoff_to_storage_rejects_missing_target_label(tmp_path: Path) -> None:
    """Datasets without the supervisory target are strictly rejected."""
    frame = _valid_frame().drop(columns=["is_laundering"])
    results = AUTO_FETCH.handoff_to_storage([frame], artifact_root=tmp_path / "vol")

    assert results[0]["status"] == "rejected"
    assert "is_laundering" in results[0]["reason"]
    if (tmp_path / "vol").exists():
        assert not any((tmp_path / "vol").rglob("*.parquet"))


def test_handoff_to_storage_rejects_unverified_candidate(tmp_path: Path) -> None:
    """A candidate that failed the reliability gate never reaches storage."""
    candidate = AUTO_FETCH.DatasetCandidate(id="test:bad", provider="test", title="bad", url="u")
    assessment = AUTO_FETCH.DatasetAssessment(
        candidate=candidate,
        quality_score=10.0,
        schema_fit=0.0,
        data_health=0.0,
        graph_topology=0.0,
        has_explicit_label=False,
        has_edge_connections=False,
        verified=False,
        reasons=("no explicit label", "no edge endpoints"),
    )
    unverified = AUTO_FETCH.VerifiedDataset(candidate=candidate, assessment=assessment)

    results = AUTO_FETCH.handoff_to_storage([unverified], artifact_root=tmp_path / "vol")
    assert results[0]["status"] == "rejected"


def test_handoff_to_storage_never_crashes_on_garbage(tmp_path: Path) -> None:
    """Malformed input degrades to a rejected entry; the batch continues."""
    results = AUTO_FETCH.handoff_to_storage(
        ["not-a-dataset", _valid_frame()], artifact_root=tmp_path / "vol"
    )
    assert [entry["status"] for entry in results] == ["rejected", "persisted"]


# ---------------------------------------------------------------------------
# 4. Modal shim alignment: mount-aware artifact loading
# ---------------------------------------------------------------------------


def test_modal_volume_path_honours_mount_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(storage_pipeline.MODAL_MOUNT_ENV, str(tmp_path / "data"))
    resolved = storage_pipeline.modal_volume_path("transactions.parquet")
    assert resolved == tmp_path / "data" / "transactions.parquet"


def test_load_training_inputs_prefers_pyg_artifact(tmp_path: Path) -> None:
    frame = _valid_frame()
    storage_pipeline.persist_transactions(
        frame, "transactions.parquet", root=tmp_path, log_amount=False
    )
    kind, payload = modal_train.load_training_inputs(
        pyg_path=str(tmp_path / "transactions.pt"),
        parquet_path=str(tmp_path / "transactions.parquet"),
    )
    assert kind == "pyg"
    data, scaler = payload
    assert data.num_nodes > 0
    assert data.edge_index.shape[0] == 2
    assert scaler is not None


def test_load_training_inputs_parquet_fallback(tmp_path: Path) -> None:
    frame = _valid_frame()
    storage_pipeline.persist_transactions(
        frame, "transactions.parquet", root=tmp_path, log_amount=False, write_pyg=False
    )
    kind, payload = modal_train.load_training_inputs(
        pyg_path=str(tmp_path / "missing.pt"),
        parquet_path=str(tmp_path / "transactions.parquet"),
    )
    assert kind == "parquet"
    assert list(payload.columns) == CANONICAL_COLUMNS


def test_load_training_inputs_missing_artifacts_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        modal_train.load_training_inputs(
            pyg_path=str(tmp_path / "missing.pt"),
            parquet_path=str(tmp_path / "missing.parquet"),
        )


# ---------------------------------------------------------------------------
# 5. Full-loop 1-epoch dry run: artifact → GAT training
# ---------------------------------------------------------------------------


def test_one_epoch_training_from_persisted_artifact(tmp_path: Path) -> None:
    """Discovery → storage → Volume mount → GAT training, end to end."""
    frame = _training_frame()
    results = AUTO_FETCH.handoff_to_storage([frame], artifact_root=tmp_path / "vol")
    assert results[0]["status"] == "persisted"

    data, scaler = storage_pipeline.load_pyg_dataset(results[0]["pyg"])

    config = load_config(PROJECT_ROOT / "config" / "config.yaml")
    checkpoints_dir = tmp_path / "checkpoints"
    config["paths"].update(
        {
            "checkpoints_dir": str(checkpoints_dir),
            "metrics_history": str(checkpoints_dir / "metrics_history.json"),
            "best_checkpoint": str(checkpoints_dir / "best.pt"),
        }
    )
    config["training"]["epochs"] = 1

    summary = train_model(
        config,
        epochs=1,
        run_dir=checkpoints_dir,
        quiet=True,
        data=data,
        scaler=scaler,
    )

    assert summary["best_epoch"] >= 1
    assert "test_metrics" in summary
    assert Path(summary["checkpoint"]).exists()


# ---------------------------------------------------------------------------
# 6. Modal delegation guard (train.py --modal → modal_train.run_remote)
# ---------------------------------------------------------------------------


def test_run_remote_requires_modal() -> None:
    """The CLI delegation reports a useful error when Modal is absent."""
    if modal_train.modal is not None:
        pytest.skip("modal is installed; remote delegation runs on Modal infra")
    with pytest.raises(RuntimeError, match="Modal is required"):
        modal_train.run_remote()
