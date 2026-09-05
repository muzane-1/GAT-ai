"""JSONL cleaning and durable Parquet/PyG dataset storage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.data_pipeline.graph_builder import build_pyg_data
from src.data_pipeline.ingestion import normalize_columns

MODAL_VOLUME_NAME = "project-data-vol"


def get_modal_volume(volume_name: str = MODAL_VOLUME_NAME) -> Any:
    """Resolve a Modal Volume lazily, keeping local tooling dependency-free."""
    try:
        import modal
    except ImportError as exc:
        raise RuntimeError("Modal is required to resolve a remote data volume") from exc
    return modal.Volume.from_name(volume_name, create_if_missing=True)


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


def transform_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize, coerce, and remove duplicate transaction records."""
    cleaned = normalize_columns(frame.copy())
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


def modal_volume_path(relative_path: str | Path, volume_name: str = MODAL_VOLUME_NAME) -> Path:
    """Return the conventional mounted path for a Modal Volume artifact."""
    if not str(relative_path) or Path(relative_path).is_absolute():
        raise ValueError("relative_path must be a non-empty relative path")
    return Path("/data") / volume_name / Path(relative_path)
