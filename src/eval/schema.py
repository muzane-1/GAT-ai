"""Schema-fit evaluation for raw AML transaction tables.

The canonical transaction schema — :data:`CANONICAL_SCHEMA` — is the layout the
rest of the pipeline expects (``tx_id, src, dst, amount, timestamp,
is_laundering``). Real-world datasets use a zoo of aliases (``source``,
``target``, ``value``, ``label``, ...), so the evaluator scores how much of the
canonical schema is present, mapping columns case- and whitespace-insensitively
via :data:`SCHEMA_ROLES`.
"""

from typing import Any

import pandas as pd

#: Canonical transaction schema used throughout the repository.
CANONICAL_SCHEMA = ["tx_id", "src", "dst", "amount", "timestamp", "is_laundering"]

#: Accepted column spellings per canonical role (canonical + known aliases).
SCHEMA_ROLES: dict[str, tuple[str, ...]] = {
    "tx_id": ("tx_id", "transaction_id", "txid"),
    "src": ("src", "source", "sender", "from_account", "from"),
    "dst": ("dst", "target", "receiver", "to_account", "to"),
    "amount": ("amount", "value", "transaction_amount", "amt"),
    "timestamp": ("timestamp", "timestamp_seconds", "ts", "datetime", "date", "time"),
    "is_laundering": (
        "is_laundering",
        "is_launder",
        "laundering",
        "label",
        "is_fraud",
        "fraud",
        "flag",
    ),
}


def resolve_column(df: pd.DataFrame, role: str) -> str | None:
    """Map a canonical role to an actual column, tolerant of case/whitespace."""
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for alias in SCHEMA_ROLES.get(role, ()):
        key = str(alias).strip().lower()
        if key in lower_map:
            return lower_map[key]
    return None


def evaluate_schema_fit(df: pd.DataFrame) -> dict[str, Any]:
    """Measure how much of the canonical schema is present in ``df``.

    Args:
        df: Raw transaction table.

    Returns:
        Dict with ``score`` (0..1), ``found`` (role -> actual column name),
        ``missing`` (roles without a mapped column) and ``mapped`` (count).
    """
    found: dict[str, str] = {}
    for role in CANONICAL_SCHEMA:
        col = resolve_column(df, role)
        if col is not None:
            found[role] = col
    missing = [role for role in CANONICAL_SCHEMA if role not in found]
    return {
        "score": len(found) / len(CANONICAL_SCHEMA),
        "found": found,
        "missing": missing,
        "mapped": len(found),
    }
