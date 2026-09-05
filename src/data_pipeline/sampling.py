"""Memory-bounded graph sampling helpers."""

from __future__ import annotations

from typing import Any

from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader


def make_neighbor_loader(
    data: Data,
    batch_size: int = 2000,
    num_neighbors: list[int] | tuple[int, ...] = (-1, -1),
    shuffle: bool = True,
    **kwargs: Any,
) -> NeighborLoader:
    """Create a NeighborLoader capped to the recommended 1k–4k node range."""
    if not 1000 <= batch_size <= 4000:
        raise ValueError("batch_size must be between 1000 and 4000 nodes")
    return NeighborLoader(
        data,
        input_nodes=None,
        num_neighbors=list(num_neighbors),
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs,
    )
