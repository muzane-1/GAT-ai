"""Data-health evaluation for raw AML transaction tables.

Health is a data-quality signal independent of schema/topology: how complete
are the cells (non-null ratio), how well behaved are the amounts (share of
strictly positive values), and how parseable are the timestamps.
"""

from typing import Any

import pandas as pd

from src.eval.schema import resolve_column


def evaluate_data_health(df: pd.DataFrame) -> dict[str, Any]:
    """Score data quality: non-null ratio, positive amounts, valid timestamps.

    Args:
        df: Raw transaction table.

    Returns:
        Dict with ``score`` (0..1, the mean of the three sub-ratios) plus the
        individual ``non_null_ratio``, ``positive_amount_ratio`` and
        ``valid_timestamp_ratio`` metrics.
    """
    non_null_ratio = float(df.notna().mean().mean()) if len(df) else 0.0

    amount_col = resolve_column(df, "amount")
    positive_amount_ratio = 0.0
    if amount_col is not None:
        amounts = pd.to_numeric(df[amount_col], errors="coerce")
        positive_amount_ratio = float((amounts > 0).mean()) if len(amounts) else 0.0

    timestamp_col = resolve_column(df, "timestamp")
    valid_timestamp_ratio = 0.0
    if timestamp_col is not None:
        try:
            timestamps = pd.to_datetime(df[timestamp_col], errors="coerce")
            valid_timestamp_ratio = float(timestamps.notna().mean())
        except Exception:  # noqa: BLE001 - unparseable timestamps degrade gracefully
            valid_timestamp_ratio = 0.0

    score = (non_null_ratio + positive_amount_ratio + valid_timestamp_ratio) / 3.0
    return {
        "score": score,
        "non_null_ratio": non_null_ratio,
        "positive_amount_ratio": positive_amount_ratio,
        "valid_timestamp_ratio": valid_timestamp_ratio,
    }
