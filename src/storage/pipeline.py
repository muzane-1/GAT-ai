"""JSONL cleaning and durable Parquet/PyG dataset storage.

Every artifact written by this module lands under the Modal ``project-data-vol``
Volume mount (``/data`` by default, overridable through the
``AML_MODAL_DATA_MOUNT`` environment variable for local dry-runs) so that
``src.training.modal_train`` reads exactly the paths the pipeline produces:

* ``<mount>/<relative>.parquet`` — sanitized canonical transaction table.
* ``<mount>/<relative>.pt``      — ``{"data": PyG Data, "scaler": scaler}``.

The canonical schema is strictly enforced:
``tx_id, src, dst, amount, timestamp, is_laundering``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.pipeline.discovery.ingestion import CANONICAL_COLUMNS, normalize_columns
from src.pipeline.transform.graph_builder import build_pyg_data
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODAL_VOLUME_NAME = "project-data-vol"

#: Environment variable overriding the local mount point of the data Volume.
MODAL_MOUNT_ENV = "AML_MODAL_DATA_MOUNT"

#: Default mount point of ``project-data-vol`` inside Modal functions.
DEFAULT_MODAL_MOUNT = "/data"

#: Default relative artifact paths, shared with ``modal_train``.
DEFAULT_PARQUET_RELPATH = "transactions.parquet"
DEFAULT_PYG_RELPATH = "transactions.pt"


def get_modal_volume(volume_name: str = MODAL_VOLUME_NAME) -> Any:
    """Resolve a Modal Volume lazily, keeping local tooling dependency-free."""
    try:
        import modal
    except ImportError as exc:
        raise RuntimeError("Modal is required to resolve a remote data volume") from exc
    return modal.Volume.from_name(volume_name, create_if_missing=True)


def get_data_mount() -> Path:
    """Return the effective root path for Volume-backed artifacts.

    Prefers ``${AML_MODAL_DATA_MOUNT}`` so tests and local dry-runs can point
    the "volume" at a temporary directory; defaults to ``/data``.
    """
    return Path(os.environ.get(MODAL_MOUNT_ENV, DEFAULT_MODAL_MOUNT))


def modal_volume_path(
    relative_path: str | Path,
    volume_name: str = MODAL_VOLUME_NAME,  # noqa: ARG001 - call-site compatibility
) -> Path:
    """Return the conventional mounted path for a Modal Volume artifact.

    ``project-data-vol`` is mounted at ``/data`` (or ``${AML_MODAL_DATA_MOUNT}``),
    so the volume name is resolved by the mount itself rather than as a nested
    directory — this keeps artifact paths identical locally and inside Modal.
    """
    if not str(relative_path) or Path(relative_path).is_absolute():
        raise ValueError("relative_path must be a non-empty relative path")
    return get_data_mount() / Path(relative_path)


def load_jsonl(path: str | Path) -> pd.DataFrame:
    """Read JSON Lines and normalize it to the canonical transaction schema."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} must contain an object")
            records.append(value)
    return normalize_columns(pd.DataFrame(records))


def validate_canonical(frame: pd.DataFrame) -> pd.DataFrame:
    """Strictly validate the canonical transaction schema (hard quality gate).

    Args:
        frame: Candidate table (column aliases normalized on the fly).

    Returns:
        The frame reduced to ``CANONICAL_COLUMNS``.

    Raises:
        ValueError: If edge endpoints (``src``/``dst``), the supervisory target
            (``is_laundering``) or any other canonical column is missing, or
            the table is empty.
    """
    if frame is None or frame.empty:
        raise ValueError("transaction table is empty")
    normalized = normalize_columns(frame.copy())
    missing = [col for col in CANONICAL_COLUMNS if col not in normalized.columns]
    if missing:
        raise ValueError(f"missing canonical columns: {missing}")
    for column in ("src", "dst"):
        if normalized[column].isna().any():
            raise ValueError(f"column '{column}' contains missing edge endpoints")
    if normalized["is_laundering"].isna().any():
        raise ValueError("column 'is_laundering' (supervisory target) is required")
    return normalized[CANONICAL_COLUMNS]


def log_normalize_amount(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply log-scale (``log1p``) normalization to non-negative amounts."""
    normalized = frame.copy()
    amounts = pd.to_numeric(normalized["amount"], errors="coerce").fillna(0.0)
    normalized["amount"] = np.log1p(amounts.clip(lower=0.0))
    return normalized


def transform_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize, coerce, and remove duplicate transaction records."""
    cleaned = validate_canonical(frame)
    cleaned["amount"] = pd.to_numeric(cleaned["amount"], errors="raise")
    cleaned["timestamp"] = pd.to_numeric(cleaned["timestamp"], errors="raise").astype("int64")
    cleaned["is_laundering"] = cleaned["is_laundering"].astype("int64").clip(0, 1)
    return cleaned.drop_duplicates(subset=["tx_id"]).reset_index(drop=True)


def write_parquet(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a cleaned table to Parquet, using DuckDB when available."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cleaned = transform_transactions(frame)
    try:
        import duckdb
    except ImportError:
        cleaned.to_parquet(output, index=False)
    else:
        duckdb.connect().execute(
            "COPY (SELECT * FROM cleaned) TO ? (FORMAT PARQUET)", [str(output)]
        ).close()
    return output


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Load and validate a Parquet transaction table."""
    return transform_transactions(pd.read_parquet(path))


def write_pyg_dataset(frame: pd.DataFrame, path: str | Path) -> Path:
    """Build and serialize the repository's canonical PyG graph."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data, scaler = build_pyg_data(transform_transactions(frame))
    torch.save({"data": data, "scaler": scaler}, output)
    return output


def load_pyg_dataset(path: str | Path) -> tuple[Any, Any]:
    """Load a serialized ``{"data", "scaler"}`` PyG artifact."""
    payload = torch.load(  # nosec B614 - trusted, locally produced artifact
        Path(path), weights_only=False
    )
    if not isinstance(payload, dict) or "data" not in payload:
        raise ValueError(f"PyG artifact at {path} does not contain a 'data' entry")
    return payload["data"], payload.get("scaler")


def persist_transactions(
    frame: pd.DataFrame,
    relative_path: str | Path = DEFAULT_PARQUET_RELPATH,
    *,
    root: str | Path | None = None,
    log_amount: bool = True,
    write_pyg: bool = True,
) -> dict[str, Any]:
    """Sanitize, (optionally) log-normalize, and persist a transaction table.

    This is the single handoff point between dataset discovery
    (``src.pipeline.discovery.auto_fetch``) and durable storage: it validates the
    canonical schema strictly, removes duplicates, applies log-scale
    normalization to ``amount`` and writes both Parquet and PyG ``.pt``
    artifacts under the Modal Volume mount.

    Args:
        frame: Raw candidate transaction table.
        relative_path: Relative artifact path (``.parquet`` suffix enforced).
        root: Explicit artifact root (e.g. a test tmp dir). Defaults to the
            Modal mount (``/data`` / ``${AML_MODAL_DATA_MOUNT}``).
        log_amount: Apply ``log1p`` normalization to ``amount`` (default True).
        write_pyg: Also write the PyG ``.pt`` artifact (default True).

    Returns:
        Mapping with ``parquet``, ``pyg`` paths and ``rows``/``columns`` stats.

    Raises:
        ValueError: When the strict canonical schema gate fails.
    """
    base = Path(root) if root is not None else get_data_mount()
    base.mkdir(parents=True, exist_ok=True)

    cleaned = transform_transactions(frame)
    if log_amount:
        cleaned = log_normalize_amount(cleaned)

    parquet_path = base / str(relative_path)
    if parquet_path.suffix == "":
        parquet_path = parquet_path.with_suffix(".parquet")
    write_parquet(cleaned, parquet_path)

    result: dict[str, Any] = {
        "parquet": str(parquet_path),
        "rows": int(len(cleaned)),
        "columns": list(cleaned.columns),
        "amount_log_normalized": bool(log_amount),
    }

    if write_pyg:
        pyg_path = parquet_path.with_suffix(".pt")
        write_pyg_dataset(cleaned, pyg_path)
        result["pyg"] = str(pyg_path)

    logger.info(
        "persisted_transactions",
        extra={"parquet": result["parquet"], "pyg": result.get("pyg"), "rows": result["rows"]},
    )
    return result
