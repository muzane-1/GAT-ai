"""Transaction data ingestion with retry logic and a synthetic fallback.

The canonical transaction schema used across the repository is::

    tx_id, src, dst, amount, timestamp, is_laundering

where ``src``/``dst`` are account identifiers (any hashable), ``amount`` is
positive, ``timestamp`` is a UNIX epoch (seconds) or an ISO-8601 string, and
``is_laundering`` flags suspicious transactions. Column aliases are
normalised on load so notebooks built on older naming keep working.
"""

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
    "target": "dst",
    "receiver": "dst",
    "to_account": "dst",
    "value": "amount",
    "timestamp_seconds": "timestamp",
    "label": "is_laundering",
    "is_launder": "is_laundering",
    "laundering": "is_laundering",
}


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
        raise ValueError(f"Missing required columns after normalisation: {missing}")

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
