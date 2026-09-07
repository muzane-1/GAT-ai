"""Pandera contract for the canonical AML transaction table.

The validation layer runs *before* graph construction: any frame entering
``build_pyg_data`` must carry the canonical transaction columns (``src`` /
``dst`` — also accepted as ``from_address`` / ``to_address`` — plus ``amount``,
``timestamp`` and ``is_laundering``) with well-typed, non-null values.

Pandera is an optional dependency: importing this module never fails, but
:func:`validate_transaction_schema` raises a descriptive ``RuntimeError`` when
the library is missing so operators get an actionable message.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - exercised only when pandera is absent
    import pandera.pandas as pa
    from pandera.errors import SchemaErrors

    HAS_PANDERA = True
except ImportError:  # pragma: no cover
    pa = None  # type: ignore[assignment]
    SchemaErrors = Exception  # type: ignore[assignment,misc]
    HAS_PANDERA = False


class SchemaValidationError(ValueError):
    """Raised when a transaction table fails the Pandera schema contract."""


#: Columns that must exist (after alias normalisation) before graph building.
REQUIRED_TRANSACTION_COLUMNS: tuple[str, ...] = (
    "src",
    "dst",
    "amount",
    "timestamp",
    "is_laundering",
)

if HAS_PANDERA:  # pragma: no branch - depends on optional dependency
    TRANSACTION_SCHEMA: Any = pa.DataFrameSchema(
        {
            # tx_id may be filled by the sanitizer, so it stays optional.
            "tx_id": pa.Column(dtype=None, required=False, nullable=True),
            "src": pa.Column(dtype=None, nullable=False),
            "dst": pa.Column(dtype=None, nullable=False),
            "amount": pa.Column("float64", checks=pa.Check.ge(0), nullable=False, coerce=True),
            "timestamp": pa.Column("float64", nullable=False, coerce=True),
            "is_laundering": pa.Column(
                "int64", checks=pa.Check.isin([0, 1]), nullable=False, coerce=True
            ),
        },
        strict=False,  # extra feature columns are allowed through
        coerce=True,
    )
else:  # pragma: no cover
    TRANSACTION_SCHEMA = None


def _alias_map() -> dict[str, str]:
    """Column alias map (lazy import keeps the layer dependency-cycle free)."""
    from src.pipeline.discovery.ingestion import _COLUMN_ALIASES

    return dict(_COLUMN_ALIASES)


def validate_transaction_schema(
    df: pd.DataFrame,
    *,
    normalize: bool = True,
    lazy: bool = True,
) -> pd.DataFrame:
    """Validate a transaction frame against the Pandera canonical schema.

    Args:
        df: Raw (possibly aliased) transaction table.
        normalize: Rename aliased columns (``from_address`` → ``src``, ...)
            before validating.
        lazy: Collect *all* failures into one report instead of raising on
            the first broken cell.

    Returns:
        The validated (and, when ``normalize`` is set, canonicalised) frame.

    Raises:
        RuntimeError: When ``pandera`` is not installed.
        SchemaValidationError: When the frame violates the contract.
    """
    if not HAS_PANDERA or TRANSACTION_SCHEMA is None:
        raise RuntimeError(
            "pandera is required for transaction schema validation; "
            "install it with `pip install pandera`"
        )

    frame = df.rename(columns=_alias_map()).copy() if normalize else df.copy()
    missing = [c for c in REQUIRED_TRANSACTION_COLUMNS if c not in frame.columns]
    if missing:
        raise SchemaValidationError(f"missing required transaction columns: {missing}")

    try:
        validated = TRANSACTION_SCHEMA.validate(frame, lazy=lazy)
    except SchemaErrors as exc:
        raise SchemaValidationError(str(exc.failure_cases)) from exc

    logger.info("pandera_validation_passed", extra={"rows": int(len(validated))})
    return validated
