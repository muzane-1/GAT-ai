"""ZenML / Kedro compatible step definitions for the AML pipeline.

A :class:`Step` is a named callable with *declared artifact inputs/outputs*
(``inputs`` / ``outputs`` tuples) — the same metadata ZenML's ``@step`` and
Kedro's node abstraction expect — so each function below can be lifted into
either framework without changing its body.

The three canonical steps mirror the MLOps layers:

1. :func:`fetch_step`      — discovery & ingestion layer (fetch/discover).
2. :func:`validate_step`   — validation & quality layer (sanitize + Pandera).
3. :func:`transform_step`  — graph transformation kernel (PyG ``Data``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.pipeline.discovery.auto_fetch import (
    sanitize_transactions,
    validate_transactions,
)
from src.pipeline.discovery.ingestion import (
    fetch_transactions,
    generate_synthetic_transactions,
)
from src.pipeline.transform.graph_builder import build_pyg_data
from src.pipeline.validation.pandera_schema import validate_transaction_schema
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Step:
    """A named pipeline step with declared artifact inputs and outputs."""

    name: str
    function: Callable[..., Any]
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    description: str = ""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the wrapped function directly (ZenML-style step call)."""
        return self.function(*args, **kwargs)


def step(
    name: str,
    *,
    inputs: tuple[str, ...] = (),
    outputs: tuple[str, ...] = (),
    description: str = "",
) -> Callable[[Callable[..., Any]], Step]:
    """Wrap a plain function into a :class:`Step` (ZenML/Kedro style)."""

    def decorator(func: Callable[..., Any]) -> Step:
        return Step(
            name=name,
            function=func,
            inputs=tuple(inputs),
            outputs=tuple(outputs),
            description=description,
        )

    return decorator


@step(
    "fetch",
    outputs=("raw_df", "fetch_stats"),
    description="Discovery & ingestion: fetch a local/remote CSV or fall back to synthetic data.",
)
def fetch_step(
    source: str | None = None, config: dict[str, Any] | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch the raw transaction table (no network by default)."""
    data_cfg = (config or {}).get("data", {})
    if source is None:
        configured = data_cfg.get("raw_source")
        # Only honour the configured source when it actually exists; otherwise
        # go straight to the deterministic synthetic generator (no retry loop).
        if configured and Path(str(configured)).exists():
            source = str(configured)

    if source:
        df = fetch_transactions(
            source=source,
            n_retry=int(data_cfg.get("fetch_retry_attempts", 3)),
            backoff_seconds=float(data_cfg.get("fetch_retry_backoff_seconds", 1.0)),
            timeout_seconds=float(data_cfg.get("fetch_timeout_seconds", 20.0)),
            fallback_generate=bool(data_cfg.get("fallback_generate", True)),
            fallback_kwargs={
                "n_accounts": int(data_cfg.get("fallback_n_accounts", 400)),
                "n_transactions": int(data_cfg.get("fallback_n_transactions", 6000)),
                "fraud_ratio": float(data_cfg.get("fallback_fraud_ratio", 0.02)),
                "seed": int(data_cfg.get("fallback_seed", 42)),
            },
        )
        stats: dict[str, Any] = {"provenance": f"source:{source}"}
    else:
        df = generate_synthetic_transactions(
            n_accounts=int(data_cfg.get("fallback_n_accounts", 400)),
            n_transactions=int(data_cfg.get("fallback_n_transactions", 6000)),
            fraud_ratio=float(data_cfg.get("fallback_fraud_ratio", 0.02)),
            seed=int(data_cfg.get("fallback_seed", 42)),
        )
        stats = {"provenance": "synthetic"}
    stats["rows"] = int(len(df))
    return df, stats


@step(
    "validate",
    inputs=("raw_df",),
    outputs=("validated_df", "validation_report"),
    description="Validation & quality: sanitize, enforce the Pandera contract, compute metrics.",
)
def validate_step(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Sanitize and validate the transaction table before graph construction."""
    cleaned = sanitize_transactions(raw_df)
    validated = validate_transaction_schema(cleaned)
    report = validate_transactions(validated)
    report["pandera_validated"] = True
    return validated, report


@step(
    "transform",
    inputs=("validated_df",),
    outputs=("graph", "scaler"),
    description="Graph kernel: build the PyG Data object with features and encodings.",
)
def transform_step(validated_df: pd.DataFrame, **builder_kwargs: Any) -> tuple[Any, Any]:
    """Convert the validated table into a PyG ``Data`` object."""
    return build_pyg_data(validated_df, **builder_kwargs)
