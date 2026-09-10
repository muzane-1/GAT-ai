"""Unit tests for the agentic, multi-provider dataset discovery pipeline.

These tests are network-free: provider adapters are mocked/stubbed and the
strict reliability gates are asserted on in-memory frames. They cover the
query generator, provider parsing, reliability scoring (including the hard
rejection of un-labelled / edge-less datasets), download/verification and the
handoff helpers.
"""

import sys
import types
from importlib import import_module
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

AUTO_FETCH = import_module("src.pipeline.discovery.auto_fetch")


@pytest.fixture(autouse=True)
def _clear_discovery_cache() -> None:
    """Providers cache their results for a few minutes; keep tests isolated."""
    AUTO_FETCH._discovery_cache.clear()


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tx_id": [0, 1, 2],
            "src": ["A", "B", "C"],
            "dst": ["B", "C", "A"],
            "amount": [10.5, 22.0, 700.0],
            "timestamp": [1000.0, 1001.0, 1002.0],
            "is_laundering": [0, 1, 0],
        }
    )


def _candidate(*, label_ok: bool = True, edge_ok: bool = True) -> AUTO_FETCH.DatasetCandidate:
    columns = ["tx_id", "amount", "timestamp"]
    if label_ok:
        columns.append("label")
    if edge_ok:
        columns.extend(["source", "target"])
    return AUTO_FETCH.DatasetCandidate(
        id="test:1",
        provider="test",
        title="test dataset",
        url="https://example.test/ds",
        metadata={"columns": columns},
    )


# ---------------------------------------------------------------------------
# Query generator
# ---------------------------------------------------------------------------


def test_generate_search_queries_cross_product_and_deterministic() -> None:
    first = AUTO_FETCH.generate_search_queries(
        assets=["btc"], aml_terms=["money laundering"], formats=["csv"]
    )
    second = AUTO_FETCH.generate_search_queries(
        assets=["btc"], aml_terms=["money laundering"], formats=["csv"]
    )
    assert first == second
    assert "btc money laundering" in first
    assert "btc money laundering csv" in first


def test_generate_search_queries_respects_cap_and_empties() -> None:
    queries = AUTO_FETCH.generate_search_queries(max_queries=3)
    assert len(queries) == 3
    assert AUTO_FETCH.generate_search_queries(assets=[], aml_terms=["fraud"]) == []
    assert AUTO_FETCH.generate_search_queries(assets=["eth"], aml_terms=[]) == []


def test_generate_search_queries_defaults_cover_crypto_aml_formats() -> None:
    queries = AUTO_FETCH.generate_search_queries()
    blob = "\n".join(queries).lower()
    for asset in ("bitcoin", "ethereum", "solana"):
        assert asset in blob
    assert "transaction graph" in blob
    assert "csv" in blob or "parquet" in blob


# ---------------------------------------------------------------------------
# Reliability scoring (strict label + edge gates, 0-100 quality score)
# ---------------------------------------------------------------------------


def test_assess_reliability_accepts_label_and_edges() -> None:
    assessment = AUTO_FETCH.assess_reliability(_candidate(), df=_valid_frame())
    assert assessment.verified is True
    assert 0.0 <= assessment.quality_score <= 100.0
    assert assessment.quality_score >= AUTO_FETCH.MIN_QUALITY_SCORE
    assert assessment.has_explicit_label is True
    assert assessment.has_edge_connections is True


def test_assess_reliability_rejects_missing_label() -> None:
    frame = _valid_frame().drop(columns=["is_laundering"])
    assessment = AUTO_FETCH.assess_reliability(_candidate(label_ok=False), df=frame)
    assert assessment.verified is False
    assert assessment.quality_score < AUTO_FETCH.MIN_QUALITY_SCORE
    assert assessment.has_explicit_label is False
    assert any("label" in reason for reason in assessment.reasons)


def test_assess_reliability_rejects_missing_edges() -> None:
    frame = _valid_frame().drop(columns=["src", "dst"])
    assessment = AUTO_FETCH.assess_reliability(_candidate(edge_ok=False), df=frame)
    assert assessment.verified is False
    assert assessment.quality_score < AUTO_FETCH.MIN_QUALITY_SCORE
    assert assessment.has_edge_connections is False


def test_assess_reliability_scores_on_metadata_when_no_df() -> None:
    assessment = AUTO_FETCH.assess_reliability(_candidate())
    assert assessment.metadata["assessed_from"] == "metadata"
    assert 0.0 <= assessment.quality_score <= 100.0


def test_rank_candidates_prefers_verified() -> None:
    good = AUTO_FETCH.DatasetCandidate(
        id="t:good",
        provider="t",
        title="good",
        url="https://a",
        metadata={"columns": ["tx_id", "source", "target", "amount", "timestamp", "label"]},
    )
    poor = AUTO_FETCH.DatasetCandidate(id="t:poor", provider="t", title="poor", url="https://b")
    ranked = AUTO_FETCH.rank_candidates([poor, good])
    assert ranked == [(good, AUTO_FETCH.assess_reliability(good))]


def test_filter_candidates_drops_unverified() -> None:
    candidates = [_candidate(label_ok=False), _candidate()]
    filtered = AUTO_FETCH.filter_candidates(candidates, require_verified=False)
    assert {c.id for c in filtered} == {candidate.id for candidate in candidates}
    verified_only = AUTO_FETCH.filter_candidates(candidates)
    assert verified_only == [_candidate()]


# ---------------------------------------------------------------------------
# Provider adapters
# ---------------------------------------------------------------------------


def test_search_huggingface_parses_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = types.ModuleType("huggingface_hub")
    api = mock.Mock()
    api.list_datasets.return_value = [mock.Mock(id="acme/aml-graph")]
    fake_module.HfApi = mock.Mock(return_value=api)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    candidates = AUTO_FETCH.search_huggingface("bitcoin aml")
    assert len(candidates) == 1
    assert candidates[0].id == "huggingface:acme/aml-graph"
    assert candidates[0].download_url is not None


def test_search_github_parses_items(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "items": [
            {
                "full_name": "acme/aml-graphs",
                "name": "aml-graphs",
                "html_url": "https://github.com/acme/aml-graphs",
                "description": "bitcoin laundering transactions",
                "license": {"spdx_id": "MIT"},
                "topics": ["aml", "graph"],
            }
        ]
    }
    monkeypatch.setattr(AUTO_FETCH, "_request_json", mock.Mock(return_value=payload))
    candidates = AUTO_FETCH.search_github("bitcoin transaction graph csv")
    assert len(candidates) == 1
    assert candidates[0].id == "github:acme/aml-graphs"
    assert candidates[0].license_info == "MIT"


def test_search_github_failure_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        AUTO_FETCH, "_request_json", mock.Mock(side_effect=RuntimeError("rate limit"))
    )
    assert AUTO_FETCH.search_github("bitcoin aml") == []


def test_search_kaggle_requires_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    assert AUTO_FETCH.search_kaggle("bitcoin laundering") == []
    # Automated fallback: missing creds -> public Hugging Face AML datasets.
    sentinel = [
        AUTO_FETCH.DatasetCandidate(
            id="huggingface:a/b", provider="huggingface", title="t", url="u"
        )
    ]
    monkeypatch.setattr(AUTO_FETCH, "search_huggingface_split", lambda query, **kwargs: sentinel)
    candidates, provider = AUTO_FETCH.search_kaggle_with_fallback("bitcoin laundering")
    assert provider == "huggingface"
    assert candidates == sentinel
    assert AUTO_FETCH.has_kaggle_credentials() is False


def test_split_compound_query_single_keywords() -> None:
    """Compound HF queries split into short single-keyword queries."""
    assert AUTO_FETCH.split_compound_query("bitcoin money laundering csv") == [
        "bitcoin",
        "money",
        "laundering",
        "csv",
    ]
    assert AUTO_FETCH.split_compound_query("AML, transactions; elliptic") == [
        "aml",
        "transactions",
        "elliptic",
    ]
    assert AUTO_FETCH.split_compound_query("  ") == []


def test_search_huggingface_split_fans_out_keywords(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Split search fans each keyword out and de-duplicates merged results."""
    calls: list[str] = []

    def _fake_search(query: str, *, limit: int = 8, timeout: float = 15.0) -> list:
        calls.append(query)
        return [
            AUTO_FETCH.DatasetCandidate(
                id=f"huggingface:{query}/ds", provider="huggingface", title=query, url="x"
            )
        ]

    monkeypatch.setattr(AUTO_FETCH, "search_huggingface", _fake_search)
    results = AUTO_FETCH.search_huggingface_split("AML transactions", limit=8)
    assert calls == ["aml", "transactions"]
    assert [c.id for c in results] == ["huggingface:aml/ds", "huggingface:transactions/ds"]


def test_score_data_source_quality_and_select_source() -> None:
    """Automated gate ranks/filters/selects the best real source (no HITL)."""
    import pandas as pd

    good = pd.DataFrame(
        {
            "tx_id": [0, 1, 2],
            "src": ["A", "B", "C"],
            "dst": ["B", "C", "A"],
            "amount": [10.5, 22.0, 700.0],
            "timestamp": [1000.0, 1001.0, 1002.0],
            "is_laundering": [0, 1, 0],
        }
    )
    bad = good.copy()
    bad["is_laundering"] = 0
    good_gate = AUTO_FETCH.score_data_source_quality(good)
    bad_gate = AUTO_FETCH.score_data_source_quality(bad)
    assert good_gate["verified"] is True
    selected = AUTO_FETCH.select_source(
        [("bad", bad_gate["quality_score"]), ("good", good_gate["quality_score"])]
    )
    assert selected == "good"
    assert AUTO_FETCH.select_source([("bad", 1.0)]) is None
    cand = AUTO_FETCH.DatasetCandidate(
        id="test:1",
        provider="test",
        title="t",
        url="u",
        metadata={"columns": ["tx_id", "label", "source", "target"]},
    )
    cand_gate = AUTO_FETCH.score_data_source_quality(cand)
    assert "quality_score" in cand_gate and "verified" in cand_gate


def test_transform_to_PyG_validates_and_builds_graph() -> None:
    """Real tables validate (Pandera) and become PyG graphs with PE + degrees."""
    import pandas as pd

    frame = pd.DataFrame(
        {
            "tx_id": [0, 1, 2],
            "src": ["A", "B", "C"],
            "dst": ["B", "C", "A"],
            "amount": [10.5, 22.0, 700.0],
            "timestamp": [1000.0, 1001.0, 1002.0],
            "is_laundering": [0, 1, 0],
        }
    )
    data, _scaler, info = AUTO_FETCH.transform_to_PyG(frame)
    assert data.num_nodes == 3 and data.num_edges == 3
    assert data.x.shape[1] == 9  # FEATURE_COLUMNS incl. degree features
    assert data.lap_pe is not None and data.rw_pe is not None
    assert info["num_nodes"] == 3


def test_gnn_train_step_and_model_factory() -> None:
    """GNN factory builds GATv2/GraphSAGE; train step runs fwd/loss/bwd/opt."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from torch_geometric.data import Data

    from src.models import AdaptiveFocalLoss
    from src.training.train import build_gnn_model, gnn_train_step

    data = Data(
        x=torch.randn(8, 9),
        edge_index=torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]]),
        edge_attr=torch.randn(6, 2),
        y=torch.tensor([0, 1, 0, 0, 1, 0, 0, 1]),
    )
    data.batch_size = 4
    for arch in ("gatv2", "graphsage"):
        model = build_gnn_model(
            arch,
            in_channels=9,
            hidden_channels=16,
            num_layers=1,
            edge_dim=int(data.edge_attr.shape[1]) if data.edge_attr is not None else None,
        )
        before = [p.detach().clone() for p in model.parameters()]
        criterion = AdaptiveFocalLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        info = gnn_train_step(model, data, criterion, optimizer)
        assert info["loss"] == info["loss"]  # finite
        assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=False))


def test_kaggle_plugin_falls_back_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discovery 'kaggle' plugin auto-falls-back to public HF (no manual creds)."""
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setattr(AUTO_FETCH, "search_huggingface", lambda query, **kw: [])
    assert AUTO_FETCH._PROVIDERS["kaggle"]("bitcoin laundering") == []


def test_search_kaggle_import_failure_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAGGLE_USERNAME", "u")
    monkeypatch.setenv("KAGGLE_KEY", "k")
    monkeypatch.setitem(sys.modules, "kaggle.api.kaggle_api_extended", None)
    assert AUTO_FETCH.search_kaggle("bitcoin laundering") == []


def test_search_web_parses_result_anchors(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_response = mock.Mock()
    fake_response.text = (
        '<a class="result__a" href="//duckduckgo.com/l/?uddg='
        'https%3A%2F%2Fexample.com%2Faml.csv&amp;rut=a">AML Bitcoin transactions</a>'
    )
    fake_response.raise_for_status = mock.Mock()
    monkeypatch.setattr(AUTO_FETCH.requests, "get", mock.Mock(return_value=fake_response))
    candidates = AUTO_FETCH.search_web("bitcoin transaction graph")
    assert len(candidates) == 1
    assert candidates[0].url == "https://example.com/aml.csv"


def test_search_web_failure_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(AUTO_FETCH.requests, "get", mock.Mock(side_effect=RuntimeError("blocked")))
    assert AUTO_FETCH.search_web("bitcoin aml") == []


# ---------------------------------------------------------------------------
# Discovery orchestration
# ---------------------------------------------------------------------------


def test_discover_candidates_offline_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    def _explode(_query: str) -> list[AUTO_FETCH.DatasetCandidate]:
        raise AssertionError("network must not be touched in offline mode")

    monkeypatch.setitem(AUTO_FETCH._PROVIDERS, "github", _explode)
    assert AUTO_FETCH.discover_candidates(offline=True, providers="all") == []


def test_discover_candidates_merges_and_dedupes(monkeypatch: pytest.MonkeyPatch) -> None:
    one = AUTO_FETCH.DatasetCandidate(
        id="github:a/b", provider="github", title="b", url="https://1"
    )
    two = AUTO_FETCH.DatasetCandidate(
        id="kaggle:acme/ds", provider="kaggle", title="a", url="https://2"
    )
    dup = AUTO_FETCH.DatasetCandidate(
        id="kaggle:acme/ds", provider="kaggle", title="a2", url="https://2"
    )

    monkeypatch.setitem(AUTO_FETCH._PROVIDERS, "github", mock.Mock(return_value=[one]))
    monkeypatch.setitem(AUTO_FETCH._PROVIDERS, "kaggle", mock.Mock(return_value=[two, dup]))
    monkeypatch.setattr(AUTO_FETCH, "generate_search_queries", mock.Mock(return_value=["q1"]))

    found = AUTO_FETCH.discover_candidates(
        providers="github,kaggle", max_queries=1, per_provider_limit=4
    )
    assert {c.id for c in found} == {"github:a/b", "kaggle:acme/ds"}


def test_discover_and_verify_no_candidates_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(AUTO_FETCH, "discover_candidates", mock.Mock(return_value=[]))
    assert AUTO_FETCH.discover_and_verify(offline=True) == []


def test_list_candidate_datasets_uses_generated_keywords(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = types.ModuleType("huggingface_hub")
    api = mock.Mock()
    api.list_datasets.return_value = [mock.Mock(id="acme/always-pass")]
    fake_module.HfApi = mock.Mock(return_value=api)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)
    monkeypatch.setattr(AUTO_FETCH, "_env_providers", mock.Mock(return_value=[]))

    datasets = AUTO_FETCH.list_candidate_datasets()
    assert datasets == ["acme/always-pass"]
    # First query should be a generated keyword (e.g. "bitcoin").
    first_search = api.list_datasets.call_args_list[0].kwargs["search"]
    assert "bitcoin" in first_search


def test_list_candidate_datasets_merges_extra_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    api = mock.Mock()
    api.list_datasets.return_value = [mock.Mock(id="acme/hf")]
    fake_module = types.ModuleType("huggingface_hub")
    fake_module.HfApi = mock.Mock(return_value=api)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)
    monkeypatch.setattr(
        AUTO_FETCH, "_extra_provider_ids", mock.Mock(return_value=["github:acme/aml"])
    )

    datasets = AUTO_FETCH.list_candidate_datasets(discovery_providers="github")
    assert "github:acme/aml" in datasets


# ---------------------------------------------------------------------------
# Download / verify
# ---------------------------------------------------------------------------


def test_download_dataset_local_path(tmp_path: Path) -> None:
    source = tmp_path / "raw.csv"
    source.write_text("a,b\n1,2\n")
    candidate = AUTO_FETCH.DatasetCandidate(
        id="test:local", provider="test", title="local", url="x", download_url=str(source)
    )
    destination = AUTO_FETCH.download_dataset(candidate, tmp_path / "out")
    assert destination is not None
    assert destination.read_text() == "a,b\n1,2\n"


def test_download_dataset_file_scheme(tmp_path: Path) -> None:
    source = tmp_path / "raw.csv"
    source.write_text("1,2\n")
    candidate = AUTO_FETCH.DatasetCandidate(
        id="test:file",
        provider="test",
        title="file",
        url="x",
        download_url=f"file://{source}",
    )
    destination = AUTO_FETCH.download_dataset(candidate, tmp_path / "out")
    assert destination is not None
    assert destination.name == "raw.csv"


def test_download_dataset_unsupported_scheme_raises(tmp_path: Path) -> None:
    candidate = AUTO_FETCH.DatasetCandidate(
        id="test:ftp", provider="test", title="ftp", url="x", download_url="ftp://host/x.csv"
    )
    with pytest.raises(RuntimeError, match="Unsupported download scheme"):
        AUTO_FETCH.download_dataset(candidate, tmp_path)


def test_download_dataset_missing_url_returns_none(tmp_path: Path) -> None:
    candidate = AUTO_FETCH.DatasetCandidate(id="test:nouri", provider="test", title="nu", url="x")
    assert AUTO_FETCH.download_dataset(candidate, tmp_path) is None


def test_verify_candidates_selects_verified(tmp_path: Path) -> None:
    raw_path = tmp_path / "good.csv"
    _valid_frame().to_csv(raw_path, index=False)
    good = AUTO_FETCH.DatasetCandidate(
        id="test:good",
        provider="test",
        title="good",
        url="x",
        download_url=str(raw_path),
        metadata={"columns": ["tx_id", "src", "dst", "amount", "timestamp", "is_laundering"]},
    )
    bad = AUTO_FETCH.DatasetCandidate(
        id="test:bad",
        provider="test",
        title="bad",
        url="y",
        metadata={"columns": ["tx_id", "src", "dst"]},
    )
    verified = AUTO_FETCH.verify_candidates([bad, good], tmp_path, top_k=4, timeout=2.0)
    assert [v.candidate.id for v in verified] == ["test:good"]
    assert verified[0].local_path is not None
    assert len(AUTO_FETCH.verified_summary(verified)) == 1


# ---------------------------------------------------------------------------
# Pipeline handoffs
# ---------------------------------------------------------------------------


def test_handoff_to_ingestion_normalises_schema(tmp_path: Path) -> None:
    frame, stats = AUTO_FETCH.handoff_to_ingestion(
        _valid_frame().rename(columns={"src": "source", "dst": "target"}),
        output=tmp_path / "canonical.csv",
    )
    assert stats["rows"] == 3
    assert (tmp_path / "canonical.csv").exists()
    assert frame.columns.tolist() == AUTO_FETCH.CANONICAL_COLUMNS


def test_handoff_to_features_reports_imbalance() -> None:
    features, info = AUTO_FETCH.handoff_to_features(_valid_frame())
    assert len(features) == 3
    assert info["n_accounts"] == 3
    assert info["class_counts"] == {0: 1, 1: 2}
    assert info["positive_ratio"] > 0.0


def test_handoff_to_graph_builder_returns_pyg_data() -> None:
    data, info = AUTO_FETCH.handoff_to_graph_builder(_valid_frame())
    assert data.num_nodes == 3
    assert data.num_edges == 3
    assert data.y is not None and data.y.shape[0] == 3
    assert info["num_node_features"] == 9


def test_handoff_runs_full_chain() -> None:
    result = AUTO_FETCH.handoff(_valid_frame())
    expected = {
        "canonical_transactions",
        "features",
        "graph",
        "ingestion",
        "feature_stats",
        "graph_info",
    }
    assert set(result) == expected
    assert result["graph"].num_nodes == result["feature_stats"]["n_accounts"]
