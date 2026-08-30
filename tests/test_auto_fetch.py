"""Unit tests for the automated dataset fetch / validate pipeline."""

from importlib import import_module
from unittest import mock

import pandas as pd
import pytest

from src.data_pipeline.auto_fetch import (
    auto_fetch,
    fetch_to_pyg,
    sanitize_transactions,
    validate_transactions,
)
from src.data_pipeline.ingestion import CANONICAL_COLUMNS

# NB: the package re-exports the `auto_fetch` function under the same name,
# shadowing the submodule for attribute access, so import the module explicitly.
AUTO_FETCH_MODULE = import_module("src.data_pipeline.auto_fetch")


def _install_fake_hf(monkeypatch: pytest.MonkeyPatch, api: mock.Mock) -> None:
    """Inject a fake ``huggingface_hub`` module so discovery works offline."""
    import sys
    import types

    fake_module = types.ModuleType("huggingface_hub")
    fake_module.HfApi = mock.Mock(return_value=api)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)


def _raw_messy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source": ["A", "B", "B", "A", "C"],
            "target": ["B", "A", "A", "B", "C"],
            "value": ["10.5", "22", "22", "not_a_number", "700"],
            "timestamp_seconds": [1000.0, 1001.0, 1001.0, 1002.0, 1003.0],
            "label": ["0", "1", "1", "0", "true"],
        }
    )


def test_sanitize_normalizes_schema_and_types() -> None:
    df = sanitize_transactions(_raw_messy_frame())
    assert list(df.columns) == CANONICAL_COLUMNS
    assert df["amount"].ge(0).all()
    assert df["is_laundering"].isin([0, 1]).all()
    # "not_a_number" row dropped + 1 duplicate edge dropped
    assert len(df) == 3


def test_validate_reports_topology_and_imbalance() -> None:
    df = sanitize_transactions(_raw_messy_frame())
    stats = validate_transactions(df)
    assert stats["rows"] == 3 and stats["nodes"] == 3
    assert stats["aml_ratio"] == pytest.approx(2 / 3)
    assert stats["null_cells"] == 0


def test_validate_rejects_empty_frame() -> None:
    with pytest.raises(ValueError, match="missing canonical columns"):
        validate_transactions(pd.DataFrame())


def test_auto_fetch_synthetic_fallback_and_pyg() -> None:
    df, stats = auto_fetch(hf_query=None)
    assert stats["provenance"] == "synthetic"
    assert stats["normalized_amounts"] is True

    data, _ = fetch_to_pyg(hf_query=None)
    assert data.num_nodes > 0 and data.num_edges > 0
    assert data.x is not None and data.edge_index.shape[0] == 2


def test_evaluate_candidate_dataset() -> None:
    from src.data_pipeline.auto_fetch import evaluate_candidate_dataset

    df = pd.DataFrame(
        {
            "src": ["A", "B", "C"],
            "dst": ["B", "C", "A"],
            "amount": [10.5, 22.0, 700.0],
            "timestamp": [1000.0, 1001.0, 1002.0],
            "is_laundering": [0, 1, 0],
        }
    )

    scores = evaluate_candidate_dataset(df)
    assert "weighted_score" in scores
    assert "schema_fit" in scores
    assert "data_health" in scores
    assert "graph_topology" in scores
    assert "aml_balance" in scores


def test_list_candidate_datasets() -> None:
    from src.data_pipeline.auto_fetch import list_candidate_datasets

    # Test with default keywords and tags
    datasets = list_candidate_datasets()
    assert isinstance(datasets, list)

    # Test with custom keywords and tags
    datasets = list_candidate_datasets(keywords=["aml"], tags=["finance"])
    assert isinstance(datasets, list)


def test_list_candidate_datasets_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path: the HF API returns rows that are mapped to dataset ids."""
    from src.data_pipeline.auto_fetch import list_candidate_datasets

    api = mock.Mock()
    api.list_datasets.return_value = [mock.Mock(id="owner/ds-a"), mock.Mock(id="owner/ds-b")]
    _install_fake_hf(monkeypatch, api)

    datasets = list_candidate_datasets(keywords=["aml"], tags=["finance"], author="acme", limit=5)
    assert datasets == ["owner/ds-a", "owner/ds-b"]
    api.list_datasets.assert_called_once_with(search=mock.ANY, author="acme", limit=5)


def test_list_candidate_datasets_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default keywords and tags are passed through to the HF API."""
    from src.data_pipeline.auto_fetch import list_candidate_datasets

    api = mock.Mock()
    api.list_datasets.return_value = []
    _install_fake_hf(monkeypatch, api)

    list_candidate_datasets()
    search = api.list_datasets.call_args.kwargs["search"]
    assert "aml" in search
    assert "finance" in search


def test_evaluate_candidate_dataset_aliases() -> None:
    """Aliased columns are resolved for scoring (value / timestamp_seconds / label)."""
    from src.data_pipeline.auto_fetch import evaluate_candidate_dataset

    df = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "A"],
            "value": ["10.5", "22"],
            "timestamp_seconds": [1000.0, 1001.0],
            "label": ["0", "1"],
        }
    )
    scores = evaluate_candidate_dataset(df)
    assert scores["nodes"] == 2
    assert scores["edges"] == 2
    assert scores["aml_ratio"] == 0.5


def test_evaluate_timestamp_parse_failure() -> None:
    """Unparseable timestamps land in the scoring except-branch."""
    from src.data_pipeline.auto_fetch import evaluate_candidate_dataset

    df = pd.DataFrame(
        {
            "src": ["A", "B"],
            "dst": ["B", "A"],
            "amount": [10.0, 20.0],
            "timestamp": [1000.0, 1001.0],
            "is_laundering": [0, 1],
        }
    )
    with mock.patch("pandas.to_datetime", side_effect=ValueError("boom")):
        scores = evaluate_candidate_dataset(df)
    assert scores["data_health"] >= 0.0


def test_evaluate_aml_balance_saturated() -> None:
    """An all-positive class ratio saturates the balance penalty branch."""
    from src.data_pipeline.auto_fetch import evaluate_candidate_dataset

    df = pd.DataFrame(
        {
            "src": ["A", "B"],
            "dst": ["B", "C"],
            "amount": [10.0, 20.0],
            "timestamp": [1000.0, 1001.0],
            "is_laundering": [1, 1],
        }
    )
    scores = evaluate_candidate_dataset(df)
    assert scores["aml_ratio"] == 1.0
    assert scores["aml_balance"] == 0.0


def test_sanitize_backfills_missing_defaults() -> None:
    """Missing tx_id / timestamp / is_laundering default to synthetic-safe values."""
    df = pd.DataFrame({"source": ["A", "B"], "target": ["B", "C"], "value": ["10.5", "22"]})
    clean = sanitize_transactions(df)
    assert list(clean.columns) == CANONICAL_COLUMNS
    assert (clean["is_laundering"] == 0).all()
    assert (clean["timestamp"] == 0.0).all()
    assert clean["tx_id"].astype(int).tolist() == [0, 1]


def test_sanitize_drops_rows_with_missing_amount() -> None:
    """Rows whose amounts cannot be parsed are dropped, not corrupted."""
    df = pd.DataFrame({"source": ["A"], "target": ["B"]})
    clean = sanitize_transactions(df)
    assert len(clean) == 0


def test_validate_rejects_empty_after_sanitation() -> None:
    empty = pd.DataFrame(columns=CANONICAL_COLUMNS)
    with pytest.raises(ValueError, match="empty after sanitation"):
        validate_transactions(empty)


def test_validate_connectivity_failure_returns_minus_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If scipy connectivity measurement fails, stats degrade gracefully."""
    df = sanitize_transactions(_raw_messy_frame())
    monkeypatch.setattr(
        "scipy.sparse.csgraph.connected_components",
        mock.Mock(side_effect=ValueError("boom")),
    )
    stats = validate_transactions(df)
    assert stats["connected_components"] == -1


def test_auto_fetch_explicit_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit local-file source short-circuits the discovery pipeline."""
    raw = pd.DataFrame(
        {
            "tx_id": [0, 1],
            "src": ["A", "B"],
            "dst": ["B", "C"],
            "amount": [10.0, 20.0],
            "timestamp": [1000.0, 1001.0],
            "is_laundering": [0, 1],
        }
    )

    def _fake_fetch(source: str, fallback_generate: bool = True) -> pd.DataFrame:
        assert fallback_generate is False
        return raw.copy()

    monkeypatch.setattr(AUTO_FETCH_MODULE, "fetch_transactions", _fake_fetch)
    monkeypatch.setattr(AUTO_FETCH_MODULE, "list_candidate_datasets", lambda **kwargs: [])

    df, stats = auto_fetch(source="data/raw/transactions.csv")
    assert stats["provenance"] == "source:data/raw/transactions.csv"
    assert stats["rows"] == 2
    assert list(df.columns) == CANONICAL_COLUMNS


def test_auto_fetch_hf_query_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fetchable explicit HF dataset id is ingested directly."""
    raw = pd.DataFrame(
        {
            "tx_id": [0, 1, 2],
            "src": ["A", "B", "C"],
            "dst": ["B", "C", "A"],
            "amount": [10.0, 20.0, 30.0],
            "timestamp": [1000.0, 1001.0, 1002.0],
            "is_laundering": [0, 0, 1],
        }
    )

    def _fake_fetch(source: str, fallback_generate: bool = True) -> pd.DataFrame:
        assert source == "acme/canonical"
        return raw.copy()

    monkeypatch.setattr(AUTO_FETCH_MODULE, "fetch_transactions", _fake_fetch)

    df, stats = auto_fetch(hf_query="acme/canonical")
    assert stats["provenance"] == "hf:acme/canonical"
    assert stats["rows"] == 3


def test_auto_fetch_hf_failure_falls_back_to_synthetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreachable HF dataset falls through to the synthetic generator."""

    def _fake_fetch(source: str, fallback_generate: bool = True) -> pd.DataFrame:
        raise RuntimeError(f"unreachable: {source}")

    monkeypatch.setattr(AUTO_FETCH_MODULE, "fetch_transactions", _fake_fetch)
    monkeypatch.setattr(AUTO_FETCH_MODULE, "list_candidate_datasets", lambda **kwargs: [])

    df, stats = auto_fetch(hf_query="acme/unreachable")
    assert stats["provenance"] == "synthetic"
    assert stats["rows"] > 0


def test_auto_fetch_discovery_ranks_and_selects(monkeypatch: pytest.MonkeyPatch) -> None:
    """Discovery loop skips failures and picks the highest-scoring valid dataset."""
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
    poor = good.copy()
    poor["is_laundering"] = 1  # aml_ratio == 1.0 -> fails the 0 < ratio < 0.5 check
    unreadable = pd.DataFrame(
        {
            "tx_id": ["0"],
            "src": [None],
            "dst": [None],
            "amount": [float("nan")],
            "timestamp": [0.0],
            "is_laundering": [0],
        }
    )
    frames = {
        "acme/network-fail": None,
        "acme/empty": unreadable,
        "acme/poor": poor,
        "acme/good": good,
    }

    def _fake_fetch(source: str, fallback_generate: bool = True) -> pd.DataFrame:
        if source in frames:
            if source == "acme/network-fail":
                raise RuntimeError("network down")
            return frames[source].copy()  # type: ignore[union-attr]
        raise RuntimeError(f"unknown source {source}")

    monkeypatch.setattr(AUTO_FETCH_MODULE, "fetch_transactions", _fake_fetch)
    monkeypatch.setattr(
        AUTO_FETCH_MODULE,
        "list_candidate_datasets",
        lambda **kwargs: ["acme/network-fail", "acme/empty", "acme/poor", "acme/good"],
    )

    df, stats = auto_fetch(hf_query="acme/nonexistent")
    assert stats["provenance"] == "hf:acme/good"
    assert stats["rows"] == 3
    assert stats["aml_ratio"] == pytest.approx(1 / 3)


def test_auto_fetch_fallback_disabled_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """With fallback disabled and no reachable source, auto_fetch raises."""
    monkeypatch.setattr(AUTO_FETCH_MODULE, "list_candidate_datasets", lambda **kwargs: [])

    with pytest.raises(RuntimeError, match="fallback is disabled"):
        auto_fetch(hf_query=None, fallback_generate=False)


def test_list_candidate_datasets_merges_extra_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """repo_ids / local_paths / synthetic_ids are merged onto HF results."""
    from src.data_pipeline.auto_fetch import list_candidate_datasets

    api = mock.Mock()
    api.list_datasets.return_value = [mock.Mock(id="owner/ds-a")]
    _install_fake_hf(monkeypatch, api)

    datasets = list_candidate_datasets(
        repo_ids=["owner/ds-b"],
        local_paths=["data/raw/transactions.csv"],
        synthetic_ids=["tiny"],
    )
    assert datasets == [
        "owner/ds-a",
        "owner/ds-b",
        "data/raw/transactions.csv",
        "synthetic:tiny",
    ]


def test_list_candidate_datasets_api_failure_still_returns_extras(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An HF API failure degrades to the configured explicit sources."""
    from src.data_pipeline.auto_fetch import list_candidate_datasets

    api = mock.Mock()
    api.list_datasets.side_effect = RuntimeError("no network")
    _install_fake_hf(monkeypatch, api)

    datasets = list_candidate_datasets(
        repo_ids=["owner/z"], local_paths=["x.csv"], synthetic_ids=["s"]
    )
    assert datasets == ["owner/z", "x.csv", "synthetic:s"]


def test_auto_fetch_sources_synthetic_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured synthetic source is evaluated and selected via discovery."""
    monkeypatch.setattr(
        AUTO_FETCH_MODULE,
        "list_candidate_datasets",
        lambda **kwargs: ["synthetic:tiny"],
    )
    df, stats = auto_fetch(
        sources=["synthetic:tiny"],
        synthetic_sources={"tiny": {"n_accounts": 20, "n_transactions": 80}},
    )
    assert stats["provenance"] == "synthetic:tiny"
    assert stats["rows"] > 0
    assert set(df["is_laundering"].unique()).issubset({0, 1})


def test_auto_fetch_sources_local_path_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A local CSV path source is routed through fetch_transactions."""
    raw = pd.DataFrame(
        {
            "tx_id": [0, 1, 2],
            "src": ["A", "B", "C"],
            "dst": ["B", "C", "A"],
            "amount": [10.0, 20.0, 30.0],
            "timestamp": [1000.0, 1001.0, 1002.0],
            "is_laundering": [0, 0, 1],  # aml_ratio 1/3 passes the hard check
        }
    )

    def _fake_fetch(source: str, fallback_generate: bool = True) -> pd.DataFrame:
        assert source == "data/raw/transactions.csv"
        return raw.copy()

    monkeypatch.setattr(AUTO_FETCH_MODULE, "fetch_transactions", _fake_fetch)
    monkeypatch.setattr(AUTO_FETCH_MODULE, "list_candidate_datasets", lambda **kwargs: [])

    df, stats = auto_fetch(sources=["data/raw/transactions.csv"])
    assert stats["provenance"] == "source:data/raw/transactions.csv"
    assert stats["rows"] == 3


def test_auto_fetch_sources_ranked_by_score(monkeypatch: pytest.MonkeyPatch) -> None:
    """Candidates are sorted by weighted score; best valid one is selected."""
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
    poor = good.copy()
    poor["is_laundering"] = 1  # aml_ratio == 1.0 fails the hard check

    frames = {"acme/good": good, "acme/poor": poor}

    def _fake_fetch(source: str, fallback_generate: bool = True) -> pd.DataFrame:
        return frames[source].copy()

    monkeypatch.setattr(AUTO_FETCH_MODULE, "fetch_transactions", _fake_fetch)
    monkeypatch.setattr(
        AUTO_FETCH_MODULE,
        "list_candidate_datasets",
        lambda **kwargs: ["acme/good", "acme/poor"],
    )

    df, stats = auto_fetch(sources=["acme/poor", "acme/good"])
    assert stats["provenance"] == "hf:acme/good"
    assert stats["rows"] == 3


def test_auto_fetch_sources_all_fail_validation_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidates that fail sanitation fall through to the synthetic generator."""
    monkeypatch.setattr(AUTO_FETCH_MODULE, "list_candidate_datasets", lambda **kwargs: [])

    def _fake_fetch(source: str, fallback_generate: bool = True) -> pd.DataFrame:
        # No amount column -> every row is dropped by sanitize -> validate raises.
        return pd.DataFrame({"source": ["A"], "target": ["B"]})

    monkeypatch.setattr(AUTO_FETCH_MODULE, "fetch_transactions", _fake_fetch)

    df, stats = auto_fetch(sources=["data/broken.csv"])
    assert stats["provenance"] == "synthetic"
    assert stats["rows"] > 0


def test_fetch_to_pyg_threads_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """fetch_to_pyg stays compatible while forwarding the new source params."""
    monkeypatch.setattr(
        AUTO_FETCH_MODULE,
        "list_candidate_datasets",
        lambda **kwargs: ["synthetic:tiny"],
    )

    data, stats = fetch_to_pyg(
        sources=["synthetic:tiny"],
        synthetic_sources={"tiny": {"n_accounts": 20, "n_transactions": 80}},
    )
    assert data.num_nodes > 0
    assert data.num_edges > 0
    assert data.x is not None and data.edge_index.shape[0] == 2
    assert stats["provenance"] == "synthetic:tiny"
    assert stats["num_node_features"] == 9
