"""Unit tests for the Pandera transaction schema contract."""

import numpy as np
import pandas as pd
import pytest

from src.pipeline.discovery.ingestion import generate_synthetic_transactions
from src.pipeline.validation.pandera_schema import (
    HAS_PANDERA,
    REQUIRED_TRANSACTION_COLUMNS,
    SchemaValidationError,
    validate_transaction_schema,
)

pytestmark = pytest.mark.skipif(not HAS_PANDERA, reason="pandera is not installed")


def test_canonical_frame_passes() -> None:
    """A clean canonical table validates unchanged (modulo coercion)."""
    df = generate_synthetic_transactions(n_accounts=20, n_transactions=50, seed=1)
    validated = validate_transaction_schema(df)
    for column in REQUIRED_TRANSACTION_COLUMNS:
        assert column in validated.columns
    assert (validated["amount"] >= 0).all()
    assert set(validated["is_laundering"].unique()).issubset({0, 1})


def test_aliased_columns_are_normalised() -> None:
    """``from_address`` / ``to_address`` aliases map to ``src`` / ``dst``."""
    raw = pd.DataFrame(
        {
            "tx_id": [0, 1],
            "from_address": ["A", "B"],
            "to_address": ["B", "C"],
            "amount": [10.0, 20.0],
            "timestamp": [1000, 2000],
            "is_laundering": [0, 1],
        }
    )
    validated = validate_transaction_schema(raw)
    assert "src" in validated.columns
    assert "dst" in validated.columns
    assert validated["src"].tolist() == ["A", "B"]
    assert validated["dst"].tolist() == ["B", "C"]


def test_missing_required_column_raises() -> None:
    """A missing canonical column fails with a descriptive error."""
    raw = pd.DataFrame(
        {
            "src": ["A"],
            "dst": ["B"],
            "timestamp": [1000],
            "is_laundering": [0],
        }
    )
    with pytest.raises(SchemaValidationError, match="amount"):
        validate_transaction_schema(raw, normalize=False)


def test_negative_amount_raises() -> None:
    """Negative amounts violate the schema contract."""
    raw = pd.DataFrame(
        {
            "src": ["A"],
            "dst": ["B"],
            "amount": [-5.0],
            "timestamp": [1000],
            "is_laundering": [0],
        }
    )
    with pytest.raises(SchemaValidationError):
        validate_transaction_schema(raw, normalize=False)


def test_invalid_label_raises() -> None:
    """Labels outside {0, 1} violate the schema contract."""
    raw = pd.DataFrame(
        {
            "src": ["A"],
            "dst": ["B"],
            "amount": [5.0],
            "timestamp": [1000],
            "is_laundering": [2],
        }
    )
    with pytest.raises(SchemaValidationError):
        validate_transaction_schema(raw, normalize=False)


def test_lazy_validation_collects_all_failures() -> None:
    """Lazy mode reports every failing cell instead of the first one."""
    raw = pd.DataFrame(
        {
            "src": ["A", "B"],
            "dst": ["B", "C"],
            "amount": [-1.0, -2.0],
            "timestamp": [1000, 2000],
            "is_laundering": [0, 0],
        }
    )
    with pytest.raises(SchemaValidationError) as excinfo:
        validate_transaction_schema(raw, normalize=False)
    # Both broken amounts are reported in the collected failure cases.
    assert str(excinfo.value).count("-1") >= 1 and str(excinfo.value).count("-2") >= 1


def test_extra_feature_columns_are_allowed() -> None:
    """Strict mode is off: additional feature columns pass through."""
    raw = pd.DataFrame(
        {
            "src": ["A"],
            "dst": ["B"],
            "amount": [5.0],
            "timestamp": [1000],
            "is_laundering": [1],
            "gas_fee": [np.float64(0.001)],
        }
    )
    validated = validate_transaction_schema(raw, normalize=False)
    assert "gas_fee" in validated.columns
