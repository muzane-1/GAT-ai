"""Transaction data ingestion with retry logic and a synthetic fallback.

The canonical transaction schema used across the repository is::

    tx_id, src, dst, amount, timestamp, is_laundering

where ``src``/``dst`` are account identifiers (any hashable), ``amount`` is
positive, ``timestamp`` is a UNIX epoch (seconds) or an ISO-8601 string, and
``is_laundering`` flags suspicious transactions. Column aliases are
normalised on load so notebooks built on older naming keep working.
"""

import difflib
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)

CANONICAL_COLUMNS = ["tx_id", "src", "dst", "amount", "timestamp", "is_laundering"]

_COLUMN_ALIASES: dict[str, str] = {
    "transaction_id": "tx_id",
    "txId": "tx_id",
    "source": "src",
    "sender": "src",
    "from_account": "src",
    "from_address": "src",
    "target": "dst",
    "receiver": "dst",
    "to_account": "dst",
    "to_address": "dst",
    "value": "amount",
    "timestamp_seconds": "timestamp",
    "label": "is_laundering",
    "is_launder": "is_laundering",
    "laundering": "is_laundering",
}

#: Critical columns — the pipeline cannot proceed if these are entirely absent.
CRITICAL_COLUMNS: tuple[str, ...] = ("src", "dst", "amount", "is_laundering")

#: Minimum fuzzy-match confidence to auto-accept a column mapping.
FUZZY_MATCH_THRESHOLD: float = 0.6

#: Warn (but continue) when an auto-mapping falls below this confidence.
FUZZY_WARN_THRESHOLD: float = 0.8


def _normalise_header(name: str) -> str:
    """Lowercase + strip separators so 'Amount Paid' ~ 'amount_paid' ~ 'amount'."""
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def fuzzy_map_columns(
    columns: list[str],
    *,
    threshold: float = FUZZY_MATCH_THRESHOLD,
) -> dict[str, tuple[str, float]]:
    """Map arbitrary dataset headers onto canonical columns via fuzzy matching.

    Exact alias hits (``_COLUMN_ALIASES`` + canonical names) win first at
    confidence 1.0.  Remaining headers are compared with
    :func:`difflib.SequenceMatcher` against canonical names *and* known
    aliases (normalised: case/separator-insensitive).  Returns
    ``{raw_column: (canonical, confidence)}`` for matches at/above
    ``threshold``; low-confidence columns are skipped with a warning log
    (never a hard stop here — the caller decides on critical columns).
    """
    mapping: dict[str, tuple[str, float]] = {}
    used_canonical: set[str] = set()
    # Candidate pool: canonical names plus every known alias spelling.
    pool: dict[str, str] = {c: c for c in CANONICAL_COLUMNS}
    for alias, canonical in _COLUMN_ALIASES.items():
        pool.setdefault(alias, canonical)
    pool_norm = {_normalise_header(k): v for k, v in pool.items()}

    for raw in columns:
        key = str(raw)
        norm = _normalise_header(key)
        if norm in pool_norm and pool_norm[norm] not in used_canonical:
            mapping[key] = (pool_norm[norm], 1.0)
            used_canonical.add(pool_norm[norm])
            continue
        best: tuple[str, float] | None = None
        for probe, canonical in pool_norm.items():
            if canonical in used_canonical:
                continue
            ratio = difflib.SequenceMatcher(None, norm, probe).ratio()
            if best is None or ratio > best[1]:
                best = (canonical, ratio)
        if best is not None and best[1] >= threshold:
            mapping[key] = best
            used_canonical.add(best[0])
            if best[1] < FUZZY_WARN_THRESHOLD:
                logger.warning(
                    "fuzzy_column_mapping_low_confidence",
                    extra={"column": key, "mapped_to": best[0], "confidence": round(best[1], 3)},
                )
        else:
            logger.warning(
                "fuzzy_column_mapping_no_match",
                extra={"column": key, "best_guess": best[0] if best else None},
            )
    return mapping


def generate_synthetic_transactions(
    n_accounts: int = 400,
    n_transactions: int = 6000,
    fraud_ratio: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a deterministic synthetic AML transaction graph.

    The generator mixes a random baseline flow with injected laundering
    motifs (fan-in/fan-out structuring circles) so that GNN topology
    features carry a genuine signal.

    Args:
        n_accounts: Number of distinct accounts to simulate.
        n_transactions: Number of transactions to emit.
        fraud_ratio: Fraction of accounts flagged as laundering amplifiers.
        seed: RNG seed for full determinism.

    Returns:
        DataFrame following the canonical transaction schema.
    """
    rng = np.random.default_rng(seed)
    base_time = 1_700_000_000

    n_bad = max(2, int(n_accounts * fraud_ratio))
    bad_accounts = rng.choice(n_accounts, size=n_bad, replace=False)
    innocent_accounts = np.setdiff1d(np.arange(n_accounts), bad_accounts)

    rows: list[dict[str, Any]] = []
    tx_id = 0

    # Structuring rings among flagged accounts only. Restricting laundering
    # edges to flagged accounts keeps both labels and topology features
    # coherent: high fan-out/fan-in, just-under-threshold amounts, and a
    # short time span give the model a discriminative motif (smurfing).
    n_laundering = max(1, int(n_transactions * 0.08))
    for _ in range(n_laundering):
        src, dst = rng.choice(bad_accounts, size=2, replace=False)
        rows.append(
            {
                "tx_id": tx_id,
                "src": int(src),
                "dst": int(dst),
                "amount": round(float(rng.uniform(850.0, 990.0)), 2),
                "timestamp": int(base_time + rng.integers(0, 43_200)),  # <12h burst velocity
                "is_laundering": 1,
            }
        )
        tx_id += 1

    # Legitimate background flow among innocent accounts only.
    while len(rows) < n_transactions:
        src, dst = rng.choice(innocent_accounts, size=2, replace=False)
        rows.append(
            {
                "tx_id": tx_id,
                "src": int(src),
                "dst": int(dst),
                "amount": round(float(rng.lognormal(mean=5.0, sigma=1.2)), 2),
                "timestamp": int(base_time + rng.integers(0, 300_000)),
                "is_laundering": 0,
            }
        )
        tx_id += 1

    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    logger.info(
        "Generated %d synthetic transactions over %d accounts (%.2f%% flagged)",
        len(df),
        n_accounts,
        fraud_ratio * 100,
    )
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename aliased columns to the canonical schema and validate it.

    Static ``_COLUMN_ALIASES`` hits apply first; any still-missing canonical
    column is resolved with fuzzy header matching
    (:func:`fuzzy_map_columns`, e.g. ``"Amount Paid"`` → ``amount``,
    ``"From Bank"`` → ``src``).  Low-confidence fuzzy hits log a warning but
    never stop the pipeline; only a completely missing *critical* column
    (``amount``/``src``/``dst``/``is_laundering``) raises.

    Args:
        df: Raw transaction table.

    Returns:
        Table with canonical column names, sorted appropriately.

    Raises:
        ValueError: If canonical columns are missing after normalisation.
    """
    df = df.rename(columns=_COLUMN_ALIASES)
    missing = [col for col in CANONICAL_COLUMNS if col not in df.columns]
    if missing:
        fuzzy = fuzzy_map_columns([str(c) for c in df.columns])
        rename: dict[str, str] = {}
        for raw, (canonical, _conf) in fuzzy.items():
            if canonical in missing and raw in df.columns:
                rename[raw] = canonical
        if rename:
            logger.info("fuzzy_column_mapping_applied", extra={"mapping": rename})
            df = df.rename(columns=rename)
        missing = [col for col in CANONICAL_COLUMNS if col not in df.columns]
    if missing:
        critical = [c for c in missing if c in CRITICAL_COLUMNS]
        if critical:
            raise ValueError(f"Missing required columns after normalisation: {missing}")
        logger.warning("noncritical_columns_missing", extra={"missing": missing})
        for col in missing:
            df[col] = 0 if col != "tx_id" else range(len(df))

    if pd.api.types.is_string_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("int64") // 10**9

    df = df[CANONICAL_COLUMNS]
    return df


def _fetch_remote_csv(url: str, timeout: float) -> pd.DataFrame:
    """Download a CSV over HTTP(S)."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    from io import StringIO

    return pd.read_csv(StringIO(response.text))


def fetch_transactions(
    source: str,
    n_retry: int = 3,
    backoff_seconds: float = 1.0,
    timeout_seconds: float = 20.0,
    fallback_generate: bool = True,
    fallback_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Fetch the transaction table with retries and a synthetic fallback.

    Args:
        source: Local path or HTTP(S) URL pointing to a CSV.
        n_retry: Maximum attempt count for both remote and local reads.
        backoff_seconds: Initial backoff; doubled after each failure.
        timeout_seconds: HTTP timeout for remote sources.
        fallback_generate: When ``True``, unreachable sources fall back to
            :func:`generate_synthetic_transactions` instead of raising.
        fallback_kwargs: Extra keyword arguments forwarded to the fallback
            generator (e.g. ``n_accounts``).

    Returns:
        Canonical transaction table.

    Raises:
        RuntimeError: If fetching fails and ``fallback_generate`` is disabled.
    """
    is_remote = source.startswith(("http://", "https://"))
    last_error: Exception | None = None
    delay = backoff_seconds

    for attempt in range(1, n_retry + 1):
        try:
            if is_remote:
                df = _fetch_remote_csv(source, timeout=timeout_seconds)
            else:
                df = pd.read_csv(Path(source))
            logger.info("Loaded %d transactions from %s", len(df), source)
            return normalize_columns(df)
        except Exception as exc:  # noqa: BLE001 - retry must survive any transient failure
            last_error = exc
            logger.warning("Fetch attempt %d/%d failed: %s", attempt, n_retry, exc)
            if attempt < n_retry:
                time.sleep(delay)
                delay *= 2

    if not fallback_generate:
        raise RuntimeError(
            f"Failed to fetch transactions from {source!r} after {n_retry} attempts"
        ) from last_error

    logger.warning("Falling back to synthetic transaction generation")
    return generate_synthetic_transactions(**(fallback_kwargs or {}))
