"""Fully automated real end-to-end discovery test (zero human intervention)."""

from importlib import import_module
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch_geometric.loader import NeighborLoader

AUTO_FETCH = import_module("src.pipeline.discovery.auto_fetch")


def _realistic_frame(seed_nodes: int = 30) -> pd.DataFrame:
    from src.pipeline.discovery.ingestion import generate_synthetic_transactions

    return generate_synthetic_transactions(
        n_accounts=seed_nodes, n_transactions=seed_nodes * 6, fraud_ratio=0.1, seed=7
    )


def _candidate(cid: str, provider: str) -> AUTO_FETCH.DatasetCandidate:
    return AUTO_FETCH.DatasetCandidate(
        id=cid, provider=provider, title=cid, url=f"https://example.test/{cid}"
    )

def test_fully_automated_real_loop_no_human_intervention(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Search → score → auto-select → PyG → GNN train step, no prompts."""
    AUTO_FETCH._discovery_cache.clear()
    good = _candidate("huggingface:acme/aml-good", "huggingface")
    mediocre = _candidate("github:acme/aml-ok", "github")
    bad = _candidate("web:acme/aml-bad", "web")
    monkeypatch.setattr(
        AUTO_FETCH, "discover_candidates", lambda **kwargs: [bad, mediocre, good]
    )
    frames = {
        good.id: _realistic_frame(40),
        mediocre.id: _realistic_frame(40).assign(is_laundering=0),
        bad.id: pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
    }

    def _fake_verify(candidates, download_dir, **kwargs):
        out = []
        for cand in candidates:
            raw = frames[cand.id]
            try:
                clean = AUTO_FETCH.sanitize_transactions(raw)
                assessment = AUTO_FETCH.assess_reliability(cand, df=clean)
            except Exception:
                assessment = AUTO_FETCH.assess_reliability(cand, df=None)
                clean = raw
            out.append(AUTO_FETCH.VerifiedDataset(cand, assessment, tmp_path, clean))
        return out

    monkeypatch.setattr(AUTO_FETCH, "verify_candidates", _fake_verify)
    result = AUTO_FETCH.auto_discover_source(top_k=3, download_dir=tmp_path)
    assert result["status"] == "selected"
    assert result["candidate"].id == good.id
    assert len(result["scored"]) == 3
    events = [e["event"] for e in result["decision_log"]]
    assert "search_complete" in events and "source_selected" in events
    assert AUTO_FETCH.select_source([(bad, 5.0), (mediocre, 40.0), (good, 95.0)]) == good
    from src.models import AdaptiveFocalLoss
    from src.pipeline.validation.pandera_schema import validate_transaction_schema
    from src.training.train import build_gnn_model, gnn_train_step

    validated = validate_transaction_schema(result["raw_df"])
    data, _scaler, info = AUTO_FETCH.transform_to_PyG(validated)
    assert info["num_nodes"] > 0 and data.x.shape[1] == 9
    loader = NeighborLoader(
        data,
        num_neighbors=[5, 5],
        batch_size=16,
        input_nodes=torch.arange(min(16, data.num_nodes)),
    )
    batch = next(iter(loader))
    model = build_gnn_model("graphsage", in_channels=data.num_node_features)
    criterion = AdaptiveFocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in model.parameters()]
    step = gnn_train_step(model, batch, criterion, optimizer)
    assert step["loss"] == step["loss"]
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    AUTO_FETCH._discovery_cache.clear()


def test_auto_loop_falls_back_without_humans(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Below-threshold → next-best-real, then synthetic last resort."""
    AUTO_FETCH._discovery_cache.clear()
    only = _candidate("github:acme/weak", "github")
    monkeypatch.setattr(AUTO_FETCH, "discover_candidates", lambda **kwargs: [only])

    def _fake_verify(candidates, download_dir, **kwargs):
        raw = pd.DataFrame({"a": [1], "b": [2]})
        item = AUTO_FETCH.VerifiedDataset(
            only, AUTO_FETCH.assess_reliability(only, df=None), tmp_path, raw
        )
        return [item]

    monkeypatch.setattr(AUTO_FETCH, "verify_candidates", _fake_verify)
    result = AUTO_FETCH.auto_discover_source(
        top_k=1, download_dir=tmp_path, min_quality=100.0
    )
    assert result["status"] == "fallback_real"
    assert result["candidate"].id == only.id
    monkeypatch.setattr(AUTO_FETCH, "verify_candidates", lambda *a, **k: [])
    result2 = AUTO_FETCH.auto_discover_source(top_k=1, download_dir=tmp_path)
    assert result2["status"] == "synthetic"
    assert len(result2["df"]) > 0
    AUTO_FETCH._discovery_cache.clear()
