"""Conversion of raw transactions into PyG ``Data`` objects.

Node features are standard-scaled (amount-like columns passed through
``log1p`` first to damp heavy skew), edge features carry the scaled amount
and a normalised timestamp delta, and node labels are attached for
supervised training.
"""

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

from src.data_pipeline import features as feature_module
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Amount-like columns whose zero-inflated log transform stabilises scale.
_LOG1P_COLUMNS = [
    "mean_amount_sent",
    "mean_amount_received",
    "total_amount_sent",
    "total_amount_received",
]


def build_pyg_data(
    df: pd.DataFrame,
    velocity_window_seconds: float = 86400.0,
    scaler: StandardScaler | None = None,
) -> tuple[Data, StandardScaler]:
    """Build a PyG ``Data`` object from a canonical transaction table.

    Args:
        df: Canonical transaction table.
        velocity_window_seconds: Window forwarded to the feature generator.
        scaler: Optional pre-fitted scaler (e.g. reusing the training split
            statistics for a validation graph).

    Returns:
        The PyG ``Data`` object and the scaler fitted on node features.
    """
    node_features = feature_module.compute_node_features(
        df, velocity_window_seconds=velocity_window_seconds
    )
    account_to_index = {account: idx for idx, account in enumerate(node_features.index)}

    x_raw = node_features[feature_module.FEATURE_COLUMNS].copy()
    for column in _LOG1P_COLUMNS:
        x_raw[column] = np.log1p(x_raw[column])

    if scaler is None:
        scaler = StandardScaler().fit(x_raw.values.astype(np.float64))
    x = torch.tensor(scaler.transform(x_raw.values.astype(np.float64)), dtype=torch.float32)

    src_idx = df["src"].map(account_to_index).astype("int64").to_numpy()
    dst_idx = df["dst"].map(account_to_index).astype("int64").to_numpy()
    edge_index = torch.tensor(np.stack([src_idx, dst_idx]), dtype=torch.long)  # shape (2, E)

    y = torch.tensor(node_features["label"].to_numpy(), dtype=torch.long)

    # Edge features: log-amount + normalised timestamp span.
    amounts = np.log1p(df["amount"].to_numpy(dtype=np.float64))
    ts_min, ts_max = df["timestamp"].min(), df["timestamp"].max()
    norm_ts = (df["timestamp"].to_numpy(dtype=np.float64) - ts_min) / max(ts_max - ts_min, 1)
    edge_attr = torch.tensor(
        np.stack([amounts, norm_ts], axis=1), dtype=torch.float32
    )  # shape (E, 2)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    logger.info(
        "Built graph: %d nodes, %d edges, %d node features, %d edge features",
        data.num_nodes,
        data.num_edges,
        x.shape[1],
        edge_attr.shape[1],
    )
    return data, scaler


def compute_node_labels(df: pd.DataFrame) -> dict[Any, int]:
    """Return a raw account-id → label mapping (utility for notebooks)."""
    dirty = df[df["is_laundering"] == 1]
    dirty_accounts = set(dirty["src"]).union(set(dirty["dst"]))
    accounts = set(df["src"]).union(set(df["dst"]))
    return {account: int(account in dirty_accounts) for account in accounts}
