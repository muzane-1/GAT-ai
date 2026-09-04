"""Automated AML dataset fetch, agentic discovery, and graph validation.

This module is the MLOps entrypoint that turns remote transaction tables into a
clean, graph-ready pandas DataFrame (and, when requested, a PyTorch Geometric
``Data`` object). It is layered as composable stages:

1. **Agentic discovery** — a heuristic (LLM-swappable) query planner builds
   crypto-asset x AML/graph-term x format queries and fans them out to the
   Kaggle API, GitHub Search API, Hugging Face Hub and a keyless web search
   (see :func:`generate_search_queries`, :func:`discover_candidates`). Each
   candidate is scored 0-100 (:func:`assess_reliability`); datasets lacking an
   explicit target label (``label`` / ``is_illicit`` / ``is_laundering`` ...)
   or edge connections (``source``/``target`` nodes) are **strictly rejected**.
2. **Fetch & sanitation** — normalise columns into the canonical schema
   (``tx_id, src, dst, amount, timestamp, is_laundering``), drop duplicates,
   fix missing values, and log-scale-normalise amounts.
3. **Validation & handoff** — assert the graph is non-empty and well-typed,
   then hand the raw matrices to ``ingestion``, the PyG graph to
   ``graph_builder`` and the derived features to ``features`` via the
   ``handoff*`` helpers.

If every remote path fails or the post-validation snapshot is empty, the
pipeline deterministically falls back to the built-in synthetic generator so
callers (training, CI, notebooks) never see a hard crash.
"""

from __future__ import annotations

import html
import os
import re
import shutil
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import numpy as np
import pandas as pd
import requests

from src.data_pipeline.ingestion import (
    CANONICAL_COLUMNS,
    fetch_transactions,
    generate_synthetic_transactions,
)
from src.eval.scoring import evaluate_candidate_dataset
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default keywords and tags for dynamic dataset discovery
DEFAULT_KEYWORDS = ["aml", "anti-money-laundering", "financial-graph", "transaction-fraud"]
DEFAULT_TAGS = ["finance", "graph", "fraud", "aml"]


# ---------------------------------------------------------------------------
# Agentic dataset discovery — typed result records & scoring plan
# ---------------------------------------------------------------------------

#: Quality Score (0-100) at/above which a dataset is considered task-useful.
MIN_QUALITY_SCORE: float = 60.0

#: Toggle for which providers auto_fetch queries (comma-separated subset of
#: {"kaggle", "github", "huggingface", "web"}).
DISCOVERY_PROVIDERS_ENV: str = "AML_DISCOVERY_PROVIDERS"

#: Crisp labels treated as an explicit supervisory target. A dataset must
#: expose at least one of these to pass the strict quality gate.
EXPLICIT_LABEL_ALIASES: tuple[str, ...] = (
    "is_illicit",
    "label",
    "is_laundering",
    "is_launder",
    "laundering",
    "is_fraud",
    "fraud",
    "flag",
)

#: Column names that give us the edge endpoints (``source``/``target`` nodes).
EDGE_ENDPOINT_ALIASES: dict[str, tuple[str, ...]] = {
    "src": ("src", "source", "sender", "from_account", "from", "node_from"),
    "dst": ("dst", "target", "receiver", "to_account", "to", "node_to"),
}

#: 0-100 score plan: the repo eval engine feeds schema/health/topology/balance
#: (scaled to 50/20/10/10) and the two explicit gates add 10 points each.
_SCHEMA_WT: float = 50.0
_HEALTH_WT: float = 20.0
_TOPOLOGY_WT: float = 10.0
_CLASS_BALANCE_WT: float = 10.0
_LABEL_WT: float = 10.0
_EDGE_WT: float = 10.0


@dataclass(frozen=True, slots=True)
class DatasetCandidate:
    """A discovered dataset reference, before any bytes are downloaded."""

    id: str
    provider: str
    title: str
    url: str
    download_url: str | None = None
    source_path: str | None = None
    description: str = ""
    license_info: str | None = None
    size_bytes: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DatasetAssessment:
    """Reliability verdict for a candidate (schema/label/graph/download)."""

    candidate: DatasetCandidate
    quality_score: float
    schema_fit: float
    data_health: float
    graph_topology: float
    has_explicit_label: bool
    has_edge_connections: bool
    verified: bool
    reasons: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VerifiedDataset:
    """A candidate that cleared the strict quality gates, with its raw bytes."""

    candidate: DatasetCandidate
    assessment: DatasetAssessment
    local_path: Path | None = None
    raw_df: pd.DataFrame | None = None


# ---------------------------------------------------------------------------
# LLM/heuristic search-query generator
# ---------------------------------------------------------------------------

DEFAULT_CRYPTO_ASSETS: tuple[str, ...] = ("bitcoin", "ethereum", "solana")
DEFAULT_AML_TERMS: tuple[str, ...] = (
    "transaction graph",
    "illicit",
    "money laundering",
    "fraud",
    "anti-money-laundering",
    "financial graph",
    "suspicious",
    "aml",
)
DEFAULT_FORMATS: tuple[str, ...] = ("csv", "parquet", "pyg", "networkx", "graph")


def generate_search_queries(
    assets: Iterable[str] = DEFAULT_CRYPTO_ASSETS,
    aml_terms: Iterable[str] = DEFAULT_AML_TERMS,
    formats: Iterable[str] = DEFAULT_FORMATS,
    max_queries: int = 24,
) -> list[str]:
    """Deterministically generate a breadth-first set of search queries.

    Iterates terms x format-variants x assets so the earliest queries already
    cover every crypto asset and the CSV/Parquet/PyG/NetworkX vocabularies
    (useful under a small ``max_queries`` cap). Iteration order is fixed (no
    RNG) so repeated runs in CI are stable. The generator is a cheap heuristic
    planner; an LLM-backed planner can replace it behind this exact signature.
    """
    assets_l = [str(a).strip().lower() for a in assets if str(a).strip()]
    terms_l = [str(t).strip().lower() for t in aml_terms if str(t).strip()]
    formats_l = [str(f).strip().lower() for f in formats if str(f).strip()]
    if not assets_l or not terms_l:
        return []

    queries: list[str] = []
    seen: set[str] = set()
    format_cycle: list[str | None] = [None, *formats_l]
    for term in terms_l:
        for fmt in format_cycle:
            for asset in assets_l:
                if len(queries) >= max_queries:
                    break
                query = f"{asset} {term}" if fmt is None else f"{asset} {term} {fmt}"
                if query not in seen:
                    seen.add(query)
                    queries.append(query)
            if len(queries) >= max_queries:
                break
        if len(queries) >= max_queries:
            break
    return queries[:max_queries]


def _generated_keywords(max_queries: int = 8) -> list[str]:
    """Flatten generated queries into a compact, de-duplicated keyword list."""
    words: list[str] = []
    for query in generate_search_queries(max_queries=min(max_queries, 6)):
        for word in query.split():
            if word not in words:
                words.append(word)
    return words[:max_queries]


# ---------------------------------------------------------------------------
# Provider adapters: Kaggle / GitHub / Hugging Face / Web search
# ---------------------------------------------------------------------------

_GH_SEARCH_URL: str = "https://api.github.com/search/repositories"
_DDG_SEARCH_URL: str = "https://html.duckduckgo.com/html/"
_USER_AGENT: str = "GAT-ai-AML-discovery/2.0"

_PROVIDER_ORDER: tuple[str, ...] = ("github", "kaggle", "huggingface", "web")
_PROVIDERS: dict[str, Callable[..., list[DatasetCandidate]]] = {}


def _request_json(url: str, *, params: dict[str, Any] | None = None, timeout: float = 12.0) -> Any:
    """GET a JSON payload, attaching the optional GitHub token."""
    headers = {"User-Agent": _USER_AGENT}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    response = requests.get(url, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def search_huggingface(
    query: str, *, limit: int = 8, timeout: float = 15.0
) -> list[DatasetCandidate]:
    """Query the Hugging Face Hub dataset registry (best-effort)."""
    try:
        from huggingface_hub import HfApi

        rows = HfApi().list_datasets(search=query, limit=limit)
    except Exception as exc:  # noqa: BLE001 - discovery is best-effort
        logger.warning("huggingface_search_failed", extra={"query": query, "error": str(exc)})
        return []

    out: list[DatasetCandidate] = []
    for row in rows:
        rid = getattr(row, "id", None) or str(row)
        out.append(
            DatasetCandidate(
                id=f"huggingface:{rid}",
                provider="huggingface",
                title=rid,
                url=f"https://huggingface.co/datasets/{rid}",
                download_url=f"https://huggingface.co/datasets/{rid}/resolve/main/",
                metadata={"needs_resolution": True},
            )
        )
    return out


def search_github(query: str, *, limit: int = 8, timeout: float = 15.0) -> list[DatasetCandidate]:
    """Search GitHub repositories for AML graph datasets."""
    try:
        payload = _request_json(
            _GH_SEARCH_URL,
            params={"q": query, "per_page": min(limit, 100)},
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("github_search_failed", extra={"query": query, "error": str(exc)})
        return []

    out: list[DatasetCandidate] = []
    for item in payload.get("items", []):
        full_name = item.get("full_name") or item.get("name") or ""
        if not full_name:
            continue
        license_meta = item.get("license")
        spdx = license_meta.get("spdx_id") if isinstance(license_meta, dict) else None
        out.append(
            DatasetCandidate(
                id=f"github:{full_name}",
                provider="github",
                title=item.get("name") or full_name,
                url=item.get("html_url") or full_name,
                description=item.get("description") or "",
                license_info=spdx,
                metadata={
                    "stars": item.get("stargazers_count"),
                    "forks": item.get("forks_count"),
                    "updated_at": item.get("updated_at"),
                    "topics": list(item.get("topics") or []),
                },
            )
        )
    return out


def _as_int(value: Any) -> int | None:
    """Best-effort int coercion for provider size fields."""
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def search_kaggle(query: str, *, limit: int = 8) -> list[DatasetCandidate]:
    """Search Kaggle datasets via the official client (requires credentials)."""
    if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        logger.warning("kaggle_search_skipped", extra={"reason": "credentials missing"})
        return []
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore[import-untyped]

        client = KaggleApi()
        client.authenticate()
        results = list(client.dataset_list(search=query))[:limit]
    except Exception as exc:  # noqa: BLE001
        logger.warning("kaggle_search_failed", extra={"query": query, "error": str(exc)})
        return []

    out: list[DatasetCandidate] = []
    for ds in results:
        ref = str(getattr(ds, "ref", "") or "").strip() or str(getattr(ds, "title", "") or "")
        if not ref:
            continue
        size = _as_int(getattr(ds, "total_bytes", None)) or _as_int(getattr(ds, "totalBytes", None))
        out.append(
            DatasetCandidate(
                id=f"kaggle:{ref}",
                provider="kaggle",
                title=str(getattr(ds, "title", "") or ref),
                url=getattr(ds, "url", "") or f"https://www.kaggle.com/datasets/{ref}",
                download_url=f"https://www.kaggle.com/api/v1/datasets/download/{ref}",
                description=str(getattr(ds, "description", "") or ""),
                size_bytes=size,
                metadata={"usability_rating": getattr(ds, "usability_rating", None)},
            )
        )
    return out


def _resolve_ddg_link(href: str) -> str:
    """Decode a DuckDuckGo redirect link to the real target URL."""
    if "uddg=" in href:
        return unquote(href.split("uddg=", 1)[1].split("&", 1)[0])
    return html.unescape(href)


def search_web(query: str, *, limit: int = 8, timeout: float = 15.0) -> list[DatasetCandidate]:
    """Keyless web search via the DuckDuckGo HTML endpoint (best-effort)."""
    try:
        response = requests.get(
            _DDG_SEARCH_URL,
            params={"q": query, "kl": "us-en"},
            headers={"User-Agent": _USER_AGENT},
            timeout=timeout,
        )
        response.raise_for_status()
        anchors = re.findall(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            response.text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("web_search_failed", extra={"query": query, "error": str(exc)})
        return []

    out: list[DatasetCandidate] = []
    seen_urls: set[str] = set()
    for href, anchor in anchors[:limit]:
        url = _resolve_ddg_link(href)
        title = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", anchor)).split())
        if not title or url in seen_urls:
            continue
        seen_urls.add(url)
        out.append(DatasetCandidate(id=f"web:{url}", provider="web", title=title, url=url))
    return out


_PLUGINS: tuple[tuple[str, Callable[..., list[DatasetCandidate]]], ...] = (
    ("huggingface", search_huggingface),
    ("github", search_github),
    ("kaggle", search_kaggle),
    ("web", search_web),
)
_PROVIDERS.update(_PLUGINS)
# ---------------------------------------------------------------------------
# Discovery orchestration & caching
# ---------------------------------------------------------------------------

_CACHE_TTL_SECONDS: float = 300.0
_discovery_cache: dict[tuple[str, str], tuple[float, list[DatasetCandidate]]] = {}


def _normalise_providers(providers: str | Iterable[str]) -> list[str]:
    """Normalise a user provider selection to an ordered, valid list."""
    if isinstance(providers, str):
        raw = [p.strip().lower() for p in providers.split(",") if p.strip()]
    else:
        raw = [str(p).strip().lower() for p in providers if str(p).strip()]
    if "all" in raw:
        return list(_PROVIDER_ORDER)
    return [p for p in raw if p in _PROVIDERS]


def _env_providers() -> list[str]:
    """Extra providers enabled via the ``${AML_DISCOVERY_PROVIDERS}`` env var."""
    value = os.environ.get(DISCOVERY_PROVIDERS_ENV, "")
    return [p for p in _normalise_providers(value) if p != "huggingface"]


def _extra_provider_ids(providers: Sequence[str], limit: int) -> list[str]:
    """Materialise candidate ids from extra network providers (HF already done)."""
    ids: list[str] = []
    for provider in providers:
        fn = _PROVIDERS[provider]
        for query in generate_search_queries(max_queries=2):
            for candidate in _cached_search(provider, query, fn, use_cache=True)[:limit]:
                if candidate.id not in ids:
                    ids.append(candidate.id)
            if ids:
                break
    return ids


def _cached_search(
    provider: str,
    query: str,
    fn: Callable[..., list[DatasetCandidate]],
    *,
    use_cache: bool,
) -> list[DatasetCandidate]:
    key = (provider, query)
    if use_cache and key in _discovery_cache:
        cached_at, cached = _discovery_cache[key]
        if time.time() - cached_at < _CACHE_TTL_SECONDS:
            return list(cached)
    results = _safe_search(fn, query)
    if use_cache:
        _discovery_cache[key] = (time.time(), list(results))
    return results


def _safe_search(fn: Callable[..., list[DatasetCandidate]], query: str) -> list[DatasetCandidate]:
    try:
        return list(fn(query))
    except Exception as exc:  # noqa: BLE001 - a failed provider must not abort discovery
        logger.warning("provider_search_failed", extra={"query": query, "error": str(exc)})
        return []


def _dedupe_candidates(candidates: Iterable[DatasetCandidate]) -> list[DatasetCandidate]:
    by_id: dict[str, DatasetCandidate] = {}
    for candidate in candidates:
        if candidate.id not in by_id:
            by_id[candidate.id] = candidate
    return list(by_id.values())


def _preliminary_score(candidate: DatasetCandidate) -> float:
    """Cheap metadata heuristic used to order discovery results."""
    text = " ".join(
        (candidate.title, candidate.description, " ".join(map(str, candidate.metadata.values())))
    ).lower()
    score = 10.0
    for keyword in ("bitcoin", "btc", "ethereum", "eth", "solana"):
        if keyword in text:
            score += 8
    for keyword in ("aml", "money launder", "illicit", "fraud", "suspicious"):
        if keyword in text:
            score += 8
    for keyword in ("transaction", "graph", "network"):
        if keyword in text:
            score += 5
    for keyword in ("csv", "parquet", "networkx", "pyg"):
        if keyword in text:
            score += 3
    if candidate.size_bytes:
        score += 2
    return score


def discover_candidates(
    *,
    assets: Iterable[str] = DEFAULT_CRYPTO_ASSETS,
    aml_terms: Iterable[str] = DEFAULT_AML_TERMS,
    formats: Iterable[str] = DEFAULT_FORMATS,
    providers: str | Iterable[str] = "all",
    max_queries: int = 24,
    per_provider_limit: int = 8,
    offline: bool = False,
    use_cache: bool = True,
) -> list[DatasetCandidate]:
    """Run the agentic discovery sweep across every enabled provider.

    Generates queries with :func:`generate_search_queries`, fans them out to the
    configured providers (guarded + cached), merges the results, de-duplicates
    and returns them ordered by the metadata heuristic. ``offline=True``
    short-circuits to ``[]`` so CI and key-less environments stay deterministic.
    """
    if offline:
        logger.info("discovery_offline", extra={"reason": "offline flag set"})
        return []

    providers_l = _normalise_providers(providers)
    if not providers_l:
        return []
    queries = generate_search_queries(assets, aml_terms, formats, max_queries=max_queries)
    if not queries:
        return []

    collected: dict[str, list[DatasetCandidate]] = {p: [] for p in providers_l}
    for provider in providers_l:
        fn = _PROVIDERS[provider]
        for query in queries:
            if len(collected[provider]) >= per_provider_limit:
                break
            seen_ids = {c.id for c in collected[provider]}
            for candidate in _cached_search(provider, query, fn, use_cache=use_cache):
                if candidate.id not in seen_ids:
                    collected[provider].append(candidate)

    merged = _dedupe_candidates(c for group in collected.values() for c in group)
    merged.sort(key=_preliminary_score, reverse=True)
    logger.info(
        "discovery_complete",
        extra={"providers": providers_l, "candidates": len(merged)},
    )
    return merged


def _df_has_explicit_label(df: pd.DataFrame) -> bool:
    """True when any canonical label alias exists among the DataFrame columns."""
    lower = {str(c).strip().lower(): c for c in df.columns}
    return any(alias in lower for alias in EXPLICIT_LABEL_ALIASES)


def _df_has_edge_connections(df: pd.DataFrame) -> bool:
    """True when both source-node and target-node columns are present."""
    lower = {str(c).strip().lower(): c for c in df.columns}
    src_ok = any(alias in lower for alias in EDGE_ENDPOINT_ALIASES["src"])
    dst_ok = any(alias in lower for alias in EDGE_ENDPOINT_ALIASES["dst"])
    return src_ok and dst_ok


def _metadata_columns(candidate: DatasetCandidate) -> list[str]:
    """Extract a ``columns``-like hint from a candidate's metadata."""
    for key in ("columns", "schema_columns", "fields"):
        value = candidate.metadata.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value]
        if isinstance(value, str):
            return [word for word in re.split(r"[\s,;]+", value) if word]
    return []


def _metadata_has_label(candidate: DatasetCandidate) -> bool:
    columns = [c.lower() for c in _metadata_columns(candidate)]
    for column in columns:
        if any(alias in column for alias in EXPLICIT_LABEL_ALIASES):
            return True
    text = f"{candidate.title} {candidate.description}".lower()
    return any(kw in text for kw in ("is_illicit", "is_laundering", "is_fraud"))


def _metadata_has_edges(candidate: DatasetCandidate) -> bool:
    columns = [c.lower() for c in _metadata_columns(candidate)]
    src_ok = any(
        any(alias in column for alias in EDGE_ENDPOINT_ALIASES["src"]) for column in columns
    )
    dst_ok = any(
        any(alias in column for alias in EDGE_ENDPOINT_ALIASES["dst"]) for column in columns
    )
    return src_ok and dst_ok


def _metadata_schema_fit(candidate: DatasetCandidate) -> float:
    """Estimate schema fit (0-1) from ``columns``-style metadata."""
    columns = [c.lower() for c in _metadata_columns(candidate)]
    if not columns:
        return 0.0
    roles: dict[str, tuple[str, ...]] = {
        "tx_id": ("tx_id", "transaction_id", "txid"),
        "src": EDGE_ENDPOINT_ALIASES["src"],
        "dst": EDGE_ENDPOINT_ALIASES["dst"],
        "amount": ("amount", "value", "amt"),
        "timestamp": ("timestamp", "ts", "datetime", "date"),
        "is_laundering": EXPLICIT_LABEL_ALIASES,
    }
    found = 0
    for aliases in roles.values():
        if any(any(alias in column for alias in aliases) for column in columns):
            found += 1
    return round(found / len(roles), 4)


def _quality_score(
    schema_fit: float,
    data_health: float,
    graph_topology: float,
    aml_balance: float,
    has_label: bool,
    has_edges: bool,
) -> float:
    """Fold sub-metrics and the two hard gates into a 0-100 Quality Score.

    The repo eval engine covers schema (50), health (20), topology (10) and
    class balance (10); the explicit-label and edge-connection gates add 10
    points each. A candidate missing a full gate is hard-capped below the
    :data:`MIN_QUALITY_SCORE` floor so it can never be verified.
    """
    base = (
        schema_fit * _SCHEMA_WT
        + data_health * _HEALTH_WT
        + graph_topology * _TOPOLOGY_WT
        + aml_balance * _CLASS_BALANCE_WT
    )
    score = base
    if has_label:
        score += _LABEL_WT
    if has_edges:
        score += _EDGE_WT
    if not (has_label and has_edges):
        score = min(score, MIN_QUALITY_SCORE - 1.0)
    return round(min(max(score, 0.0), 100.0), 2)


def assess_reliability(
    candidate: DatasetCandidate,
    df: pd.DataFrame | None = None,
) -> DatasetAssessment:
    """Score a candidate 0-100 and apply the strict label + edge gates.

    When a real table is available (post-download) the repo's ``src.eval``
    engine measures schema fit / data health / topology and the gates are
    checked on actual columns. Without a table, the same contract is evaluated
    against the candidate's metadata heuristics (``columns``/``fields``).

    Returns:
        :class:`DatasetAssessment`: ``verified=True`` only when both hard gates
        pass and the Quality Score reaches :data:`MIN_QUALITY_SCORE`.
    """
    if df is not None and not df.empty:
        scores = evaluate_candidate_dataset(df)
        schema_fit = float(scores["schema_fit"])
        data_health = float(scores["data_health"])
        graph_topology = float(scores["graph_topology"])
        aml_balance = float(scores["aml_balance"])
        has_label = _df_has_explicit_label(df)
        has_edges = _df_has_edge_connections(df)
        assessed_from = "dataframe"
    else:
        schema_fit = _metadata_schema_fit(candidate)
        data_health = 0.5
        graph_topology = 0.5
        aml_balance = 0.5
        has_label = _metadata_has_label(candidate)
        has_edges = _metadata_has_edges(candidate)
        assessed_from = "metadata"

    quality = _quality_score(
        schema_fit, data_health, graph_topology, aml_balance, has_label, has_edges
    )
    reasons: list[str] = []
    if not has_label:
        reasons.append("missing explicit target label (label/is_illicit)")
    if not has_edges:
        reasons.append("missing source/target edge columns")
    if has_label and has_edges and quality >= MIN_QUALITY_SCORE:
        reasons.append(f"quality {quality:.1f}/100 meets the {MIN_QUALITY_SCORE:.0f} floor")
        verified = True
    else:
        verified = False
        if not reasons:
            reasons.append(f"quality {quality:.1f}/100 below the {MIN_QUALITY_SCORE:.0f} floor")

    metadata = {
        **candidate.metadata,
        "assessed_from": assessed_from,
        "download_url": candidate.download_url,
    }
    return DatasetAssessment(
        candidate=candidate,
        quality_score=quality,
        schema_fit=schema_fit,
        data_health=data_health,
        graph_topology=graph_topology,
        has_explicit_label=has_label,
        has_edge_connections=has_edges,
        verified=verified,
        reasons=tuple(reasons),
        metadata=metadata,
    )


def rank_candidates(
    candidates: Iterable[DatasetCandidate],
    *,
    min_quality: float = MIN_QUALITY_SCORE,
    require_verified: bool = True,
) -> list[tuple[DatasetCandidate, DatasetAssessment]]:
    """Rank candidates by reliability; optionally drop un-verified ones.

    Returns ``(candidate, assessment)`` pairs, best Quality Score first with a
    stable ``id`` tie-break so rankings are reproducible.
    """
    pairs = [(c, assess_reliability(c)) for c in candidates]
    if require_verified:
        pairs = [(c, a) for c, a in pairs if a.verified and a.quality_score >= min_quality]
    return sorted(pairs, key=lambda pair: (-pair[1].quality_score, pair[0].id))


def filter_candidates(
    candidates: Iterable[DatasetCandidate],
    *,
    min_quality: float = MIN_QUALITY_SCORE,
    require_verified: bool = True,
) -> list[DatasetCandidate]:
    """Return only candidates that pass the reliability gates (sorted)."""
    return [
        candidate
        for candidate, _ in rank_candidates(
            candidates, min_quality=min_quality, require_verified=require_verified
        )
    ]


# ---------------------------------------------------------------------------
# Download / verification of discovered candidates
# ---------------------------------------------------------------------------


def _safe_filename(name: str) -> str:
    """Turn a URL fragment / repo ref into a safe local file name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return cleaned or "dataset.csv"


def _is_local_path(url: str) -> bool:
    """True for plain filesystem paths incl. Windows drive letters."""
    if url.startswith(("http://", "https://", "file://")):
        return False
    return os.path.isabs(url) or Path(url).exists()


def download_dataset(
    candidate: DatasetCandidate,
    download_dir: Path | str,
    *,
    timeout: float = 30.0,
) -> Path | None:
    """Download a candidate's raw file into ``download_dir`` and return its path.

    Supports ``http(s)://``, ``file://`` and plain local paths. Returns ``None``
    when the candidate has no ``download_url``.
    """
    url = candidate.download_url
    if not url:
        logger.warning("no_download_url", extra={"id": candidate.id})
        return None

    dest_dir = Path(download_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if url.startswith("file://"):
        source = Path(url[len("file://") :])
        if not source.exists():
            return None
        destination = dest_dir / _safe_filename(source.name)
        shutil.copy(source, destination)
        return destination

    if url.startswith(("http://", "https://")):
        name = _safe_filename(Path(url).name) or f"{_safe_filename(candidate.id)}.csv"
        destination = dest_dir / name
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        with open(destination, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 16):
                handle.write(chunk)
        return destination

    if _is_local_path(url):
        path = Path(url)
        return path if path.exists() else None

    raise RuntimeError(f"Unsupported download scheme for {candidate.id}: {url!r}")


def _read_table(path: Path) -> pd.DataFrame:
    """Read a downloaded file into a DataFrame (CSV/JSON/Parquet)."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path)
    return pd.read_csv(path)


def verify_candidates(
    candidates: Iterable[DatasetCandidate],
    download_dir: Path | str,
    *,
    top_k: int = 3,
    timeout: float = 30.0,
    strict: bool = True,
) -> list[VerifiedDataset]:
    """Download top-K candidates and re-score them on the real raw table.

    Args:
        candidates: Discovery results (already metadata-sorted).
        download_dir: Directory for the raw downloads.
        top_k: How many top-ranked candidates to materialise.
        timeout: HTTP download timeout in seconds.
        strict: When ``True``, only candidates that pass the label+edge gates on
            the downloaded data are returned.

    Returns:
        Verified datasets (candidate + assessment + local raw path + raw table).
    """
    ranked = rank_candidates(candidates, min_quality=0.0, require_verified=False)
    verified: list[VerifiedDataset] = []
    for candidate, _meta in ranked[:top_k]:
        try:
            local = download_dataset(candidate, download_dir, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 - a failed download is not fatal
            logger.warning("download_failed", extra={"id": candidate.id, "error": str(exc)})
            continue
        if local is None or not local.exists():
            continue
        try:
            raw = _read_table(local)
        except Exception as exc:  # noqa: BLE001 - unreadable bytes are skipped
            logger.warning("read_failed", extra={"path": str(local), "error": str(exc)})
            continue
        assessment = assess_reliability(candidate, df=raw)
        if strict and not assessment.verified:
            continue
        verified.append(VerifiedDataset(candidate, assessment, local, raw))

    verified.sort(key=lambda v: v.assessment.quality_score, reverse=True)
    logger.info(
        "verification_complete",
        extra={"attempted": min(len(ranked), top_k), "verified": len(verified)},
    )
    return verified


# ---------------------------------------------------------------------------
# One-shot discovery + verification summary
# ---------------------------------------------------------------------------


def discover_and_verify(
    *,
    assets: Iterable[str] = DEFAULT_CRYPTO_ASSETS,
    aml_terms: Iterable[str] = DEFAULT_AML_TERMS,
    formats: Iterable[str] = DEFAULT_FORMATS,
    providers: str | Iterable[str] = "all",
    max_queries: int = 24,
    per_provider_limit: int = 8,
    offline: bool = False,
    download_dir: Path | str = "data/discovery",
    top_k: int = 3,
    timeout: float = 30.0,
    strict: bool = True,
) -> list[VerifiedDataset]:
    """One-shot agentic pipeline: generate queries -> discover -> verify.

    Downloads the top-K candidates, re-scores them on the real raw table and
    applies the strict label/edge gates. Returns verified datasets holding the
    metadata, Quality Score and the raw download path.
    """
    candidates = discover_candidates(
        assets=assets,
        aml_terms=aml_terms,
        formats=formats,
        providers=providers,
        max_queries=max_queries,
        per_provider_limit=per_provider_limit,
        offline=offline,
    )
    if not candidates:
        logger.info("no_candidates_discovered")
        return []
    return verify_candidates(candidates, download_dir, top_k=top_k, timeout=timeout, strict=strict)


def verified_summary(verified: Iterable[VerifiedDataset]) -> list[dict[str, Any]]:
    """Flatten verified datasets into JSON-serialisable metadata records."""
    summary: list[dict[str, Any]] = []
    for item in verified:
        summary.append(
            {
                "id": item.candidate.id,
                "provider": item.candidate.provider,
                "title": item.candidate.title,
                "url": item.candidate.url,
                "download_url": item.candidate.download_url,
                "local_path": str(item.local_path) if item.local_path else None,
                "quality_score": item.assessment.quality_score,
                "has_explicit_label": item.assessment.has_explicit_label,
                "has_edge_connections": item.assessment.has_edge_connections,
                "schema_fit": item.assessment.schema_fit,
                "data_health": item.assessment.data_health,
                "graph_topology": item.assessment.graph_topology,
                "reasons": list(item.assessment.reasons),
            }
        )
    return summary


# ---------------------------------------------------------------------------
# Integration handoffs -> ingestion / features / graph_builder
# ---------------------------------------------------------------------------


def handoff_to_ingestion(
    raw: pd.DataFrame | Path | str,
    *,
    output: Path | str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalise raw matrices via :mod:`src.data_pipeline.ingestion`.

    Paths/URLs are read and normalised with ``fetch_transactions``; in-memory
    frames go through ``sanitize_transactions`` (which reuses ingestion's alias
    map). Returns the canonical table plus validation stats.
    """
    if isinstance(raw, pd.DataFrame):
        frame = sanitize_transactions(raw)
    else:
        frame = fetch_transactions(source=str(raw), fallback_generate=False)
    stats = validate_transactions(frame)
    if output is not None:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_path, index=False)
        stats["output_path"] = str(out_path)
    return frame, stats


def handoff_to_features(
    df: pd.DataFrame,
    *,
    velocity_window_seconds: float = 86_400.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Forward a canonical table to :mod:`src.data_pipeline.features`.

    Returns the per-account feature table (feature mapping + scaling inputs)
    together with class-imbalance metadata used downstream for re-weighting.
    """
    from src.data_pipeline.features import FEATURE_COLUMNS, compute_node_features

    features = compute_node_features(df, velocity_window_seconds=velocity_window_seconds)
    counts = {int(k): int(v) for k, v in features["label"].value_counts().sort_index().items()}
    total = max(int(len(features)), 1)
    positive = int(counts.get(1, 0))
    positive_ratio = positive / total
    info: dict[str, Any] = {
        "feature_columns": FEATURE_COLUMNS,
        "n_accounts": total,
        "class_counts": counts,
        "positive_ratio": round(positive_ratio, 6),
        "imbalanced": positive_ratio < 0.2,
    }
    return features, info


def handoff_to_graph_builder(
    df: pd.DataFrame,
    **builder_kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Hand raw edges/nodes to :mod:`src.data_pipeline.graph_builder` (PyG)."""
    from src.data_pipeline.graph_builder import build_pyg_data

    data, scaler = build_pyg_data(df, **builder_kwargs)
    info: dict[str, Any] = {
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.num_edges),
        "num_node_features": int(data.num_node_features),
        "edge_features": int(data.edge_attr.shape[1]) if data.edge_attr is not None else 0,
        "scaler": scaler,
    }
    return data, info


def handoff(
    df: pd.DataFrame,
    *,
    velocity_window_seconds: float = 86_400.0,
    ingestion_output: Path | str | None = None,
    **graph_kwargs: Any,
) -> dict[str, Any]:
    """Run the full ingestion -> features -> graph handoff chain.

    Returns a dict with ``canonical_transactions``, ``features``, ``graph`` and
    per-stage stats so callers can inspect every handoff point.
    """
    canonical, ingest_stats = handoff_to_ingestion(df, output=ingestion_output)
    feature_table, feature_stats = handoff_to_features(
        canonical, velocity_window_seconds=velocity_window_seconds
    )
    graph, graph_info = handoff_to_graph_builder(canonical, **graph_kwargs)
    return {
        "canonical_transactions": canonical,
        "features": feature_table,
        "graph": graph,
        "ingestion": ingest_stats,
        "feature_stats": feature_stats,
        "graph_info": graph_info,
    }


def list_candidate_datasets(
    keywords: list[str] | None = None,
    tags: list[str] | None = None,
    author: str | None = None,
    limit: int = 20,
    repo_ids: list[str] | None = None,
    local_paths: list[str] | None = None,
    synthetic_ids: list[str] | None = None,
    discovery_providers: str | Iterable[str] | None = None,
) -> list[str]:
    """Discover candidate AML-flavoured transaction datasets.

    Queries the Hugging Face Hub across the given keywords/tags and, on top of
    that, merges an explicit array of extra sources onto the discovery results:
    HF repository ids (``"owner/repo"``), local CSV paths
    (``"data/raw/transactions.csv"``), synthetic generator specs
    (``"synthetic:boosted"``) and, when enabled, other network providers
    (Kaggle / GitHub / web) are all treated as first-class candidates.

    Args:
        keywords: Search keywords for dataset names/cards. When ``None`` a
            deterministic keyword set is generated by
            :func:`generate_search_queries`.
        tags: Search tags to filter datasets.
        author: Optional user/org filter (e.g. ``"qubit420"``).
        limit: Maximum number of HF datasets to return.
        repo_ids: Extra Hugging Face repository identifiers to include.
        local_paths: Extra local CSV paths to include as candidates.
        synthetic_ids: Synthetic generator spec names (e.g. ``"default"``).
        discovery_providers: Optional extra network providers to query
            (``"github,kaggle,web"`` or ``"all"``). Defaults to the
            ``${AML_DISCOVERY_PROVIDERS}`` env var; the Hugging Face Hub is
            always queried first.

    Returns:
        A list of ``"<owner>/<repo>"`` identifiers and configured extra
        sources, best-match first. Empty on any API error so downstream
        logic falls through to the local / synthetic paths.
    """
    keywords = keywords or _generated_keywords()
    tags = tags or DEFAULT_TAGS
    providers = (
        _normalise_providers(discovery_providers) if discovery_providers else _env_providers()
    )

    try:
        from huggingface_hub import HfApi

        api = HfApi()
        query = " ".join(keywords + tags)
        rows = api.list_datasets(search=query, author=author, limit=limit)
        datasets = [row.id for row in rows]
    except Exception as exc:  # network, auth, or hub rate-limit
        logger.warning("hf_dataset_discovery_failed", extra={"error": str(exc)})
        datasets = []

    extra_ids = _extra_provider_ids(providers, max(limit, 1))
    extras = [
        *(repo_ids or []),
        *(local_paths or []),
        *(f"synthetic:{spec}" for spec in (synthetic_ids or [])),
        *extra_ids,
    ]
    for extra in extras:
        if extra not in datasets:
            datasets.append(extra)
    return datasets


def _canonicalise(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to the canonical schema, best-effort."""
    from src.data_pipeline.ingestion import _COLUMN_ALIASES

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}
    for alias, canonical in _COLUMN_ALIASES.items():
        if alias.lower() in lower_map:
            df = df.rename(columns={lower_map[alias.lower()]: canonical})
    return df


def sanitize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Apply MLOps-standard sanitation rules to a raw transaction table.

    - Column normalisation into the canonical schema.
    - Missing ``tx_id`` → synthesised from row index (stable).
    - Missing ``timestamp`` → forward-fill then 0.0 (first rows).
    - Non-numeric ``amount`` → parsed, NaNs dropped, negatives clipped.
    - Duplicate edges (same src/dst/amount/ts) → dropped.
    - ``is_laundering`` → coerced to 0/1 (any truthy string parsed).

    Returns:
        Clean DataFrame that satisfies :func:`validate_transactions`.
    """
    df = _canonicalise(df)
    original = len(df)

    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            if col == "tx_id":
                df[col] = np.arange(len(df)).astype(str)
            elif col == "is_laundering":
                df[col] = 0
            elif col == "timestamp":
                df[col] = 0.0
            else:
                df[col] = np.nan

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["src", "dst", "amount"])
    df["amount"] = df["amount"].clip(lower=0.0)

    df["is_laundering"] = (
        df["is_laundering"]
        .astype(str)
        .str.lower()
        .map({"1": 1, "true": 1, "yes": 1, "0": 0, "false": 0, "no": 0})
        .fillna(0)
        .astype(int)
    )

    df = df.drop_duplicates(subset=["src", "dst", "amount", "timestamp"]).reset_index(drop=True)
    logger.info(
        "sanitized_transactions",
        extra={"before": original, "after": len(df), "dropped": original - len(df)},
    )
    return df[CANONICAL_COLUMNS]


def validate_transactions(df: pd.DataFrame) -> dict[str, Any]:
    """Compute sanity metrics used by the notebook's success/warning checks.

    Raises:
        ValueError: if the table is empty or missing canonical columns.
    """
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing canonical columns: {missing}")
    if len(df) == 0:
        raise ValueError("transaction table is empty after sanitation")

    src = df["src"].astype(str)
    dst = df["dst"].astype(str)
    nodes = pd.concat([src, dst]).nunique()
    edges = len(df)
    pos = int(df["is_laundering"].sum())

    # Graph connectivity: are all node ids reachable in an undirected view?
    try:
        import scipy.sparse as sp

        ids = pd.concat([src, dst]).unique().tolist()
        idx = {v: i for i, v in enumerate(ids)}
        rows = np.array([idx[s] for s in src], dtype=np.int64)
        cols = np.array([idx[d] for d in dst], dtype=np.int64)
        adj = sp.coo_matrix((np.ones(edges), (rows, cols)), shape=(len(ids), len(ids)))
        n_comp, _ = sp.csgraph.connected_components(adj + adj.T, directed=False, return_labels=True)
    except Exception:
        n_comp = -1  # connectivity not measurable; still report other stats

    return {
        "rows": edges,
        "nodes": int(nodes),
        "edges": edges,
        "avg_degree": round(edges / max(nodes, 1), 3),
        "connected_components": int(n_comp),
        "class_counts": {0: edges - pos, 1: pos},
        "aml_ratio": round(pos / edges, 6),
        "null_cells": int(df.isna().sum().sum()),
    }


def _spec_kind(spec: str) -> str:
    """Classify a candidate source spec: synthetic / path / hf."""
    lower = spec.lower()
    if lower == "synthetic" or lower.startswith("synthetic:"):
        return "synthetic"
    if (
        lower.startswith(("http://", "https://"))
        or Path(spec).suffix.lower() == ".csv"
        or Path(spec).exists()
    ):
        return "path"
    return "hf"


def _load_candidate(spec: str, synthetic_sources: dict[str, dict[str, Any]] | None) -> pd.DataFrame:
    """Load a single candidate source: HF repo, local path, or synthetic."""
    if _spec_kind(spec) == "synthetic":
        name = spec.split(":", 1)[1] if ":" in spec else "default"
        kwargs = (synthetic_sources or {}).get(name, {})
        return generate_synthetic_transactions(**kwargs)
    return fetch_transactions(source=spec, fallback_generate=False)


def auto_fetch(
    source: str | None = None,
    hf_query: str | None = None,
    hf_keywords: list[str] | None = None,
    hf_tags: list[str] | None = None,
    sources: list[str] | None = None,
    synthetic_sources: dict[str, dict[str, Any]] | None = None,
    discovery_providers: str | Iterable[str] | None = None,
    normalize_amounts: bool = True,
    fallback_generate: bool = True,
    **builder_kwargs: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """End-to-end fetch → sanitize → validate pipeline with dynamic multi-source discovery.

    Resolution order:
    1. Explicit ``source`` (local CSV path or HTTP URL).
    2. Explicit Hugging Face dataset id (``hf_query``).
    3. Dynamic discovery: heuristic queries (see :func:`generate_search_queries`)
       fan out over Hugging Face — plus Kaggle/GitHub/web when enabled through
       ``discovery_providers`` or ``${AML_DISCOVERY_PROVIDERS}`` — merged with
       any explicit ``sources`` array (HF repo ids, local CSV paths,
       ``synthetic:<name>`` specs). Candidates are ranked by weighted score and
       the highest-scoring one that passes hard validation checks is selected.
    4. Deterministic synthetic generator (if quality gates all fail).

    For strict, agentic multi-provider search use :func:`discover_candidates`
    and :func:`discover_and_verify` directly — they return verified metadata and
    raw download paths without going through this legacy-compatible wrapper.

    Returns:
        ``(canonical_transactions, stats_dict)`` — the stats are produced by
        :func:`validate_transactions` and include class-imbalance numbers.
    """
    df: pd.DataFrame | None = None
    provenance: str = "synthetic"

    if source:
        df = fetch_transactions(source=source, fallback_generate=False)
        provenance = f"source:{source}"
    elif hf_query:
        try:
            df = fetch_transactions(source=hf_query, fallback_generate=False)
            provenance = f"hf:{hf_query}"
        except Exception as exc:
            logger.warning("hf_fetch_failed", extra={"dataset": hf_query, "error": str(exc)})
            df = None

    if df is None:
        # Dynamic discovery of Hugging Face datasets
        candidate_datasets = list_candidate_datasets(
            keywords=hf_keywords or DEFAULT_KEYWORDS,
            tags=hf_tags or DEFAULT_TAGS,
            limit=20,
            discovery_providers=discovery_providers,
        )
        for spec in sources or []:
            if spec not in candidate_datasets:
                candidate_datasets.append(spec)

        if candidate_datasets:
            logger.info(f"Discovered {len(candidate_datasets)} candidate datasets")

            # Evaluate and rank candidate datasets
            evaluated_datasets = []
            for dataset_id in candidate_datasets:
                try:
                    df_raw = _load_candidate(dataset_id, synthetic_sources)
                    evaluation_scores = evaluate_candidate_dataset(df_raw)
                    evaluated_datasets.append((dataset_id, evaluation_scores, df_raw))
                    logger.info(
                        f"Evaluated dataset {dataset_id}: "
                        f"score={evaluation_scores['weighted_score']:.3f}"
                    )
                except Exception as exc:
                    logger.warning(
                        f"Failed to fetch or evaluate dataset {dataset_id}",
                        extra={"error": str(exc)},
                    )
                    continue

            if evaluated_datasets:
                # Sort by weighted score in descending order
                evaluated_datasets.sort(key=lambda x: x[1]["weighted_score"], reverse=True)

                # Select the highest-scoring dataset that passes hard validation checks
                for dataset_id, evaluation_scores, df_raw in evaluated_datasets:
                    try:
                        df = sanitize_transactions(df_raw)
                        stats = validate_transactions(df)

                        # Hard validation checks
                        if (
                            stats["rows"] > 0
                            and stats["nodes"] > 0
                            and 0 < stats["aml_ratio"] < 0.5
                        ):
                            kind = _spec_kind(dataset_id)
                            if kind == "synthetic":
                                # Spec already carries the synthetic: prefix.
                                provenance = dataset_id
                            else:
                                label = {"path": "source", "hf": "hf"}[kind]
                                provenance = f"{label}:{dataset_id}"
                            logger.info(
                                f"Selected highest-scoring dataset {dataset_id} "
                                f"with score {evaluation_scores['weighted_score']:.3f}"
                            )
                            break
                    except Exception as exc:
                        logger.warning(
                            f"Dataset {dataset_id} failed validation",
                            extra={"error": str(exc)},
                        )
                        df = None
                        continue

    if df is None:
        if not fallback_generate:
            raise RuntimeError(
                "auto_fetch could not retrieve a usable table and fallback is disabled"
            )
        df = generate_synthetic_transactions()
        provenance = "synthetic"

    df = sanitize_transactions(df)
    if normalize_amounts:
        # log1p keeps the heavily skewed amounts well-behaved for GNN scaling
        df = df.copy()
        df["amount"] = np.log1p(df["amount"])

    stats = validate_transactions(df)
    stats["provenance"] = provenance
    stats["normalized_amounts"] = normalize_amounts
    logger.info(
        "auto_fetch_complete",
        extra={"provenance": provenance, "rows": stats["rows"], "aml_ratio": stats["aml_ratio"]},
    )
    return df, stats


def fetch_to_pyg(
    source: str | None = None,
    hf_query: str | None = None,
    hf_keywords: list[str] | None = None,
    hf_tags: list[str] | None = None,
    sources: list[str] | None = None,
    synthetic_sources: dict[str, dict[str, Any]] | None = None,
    discovery_providers: str | Iterable[str] | None = None,
    **builder_kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Fetch and convert to a PyG ``Data`` object via the canonical builder.

    Returns:
        ``(data, stats)`` where ``data`` is the PyG graph and ``stats`` is
        the dict produced by :func:`auto_fetch`.
    """
    from src.data_pipeline.graph_builder import build_pyg_data

    df, stats = auto_fetch(
        source=source,
        hf_query=hf_query,
        hf_keywords=hf_keywords,
        hf_tags=hf_tags,
        sources=sources,
        synthetic_sources=synthetic_sources,
        discovery_providers=discovery_providers,
        **builder_kwargs,
    )
    data, _ = build_pyg_data(df)
    stats["num_node_features"] = int(data.num_node_features)
    return data, stats
