"""Precomputed structural positional encodings for PyG graphs."""

from __future__ import annotations

import torch
from torch_geometric.utils import to_undirected


def laplacian_positional_encoding(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_embeddings: int = 8,
) -> torch.Tensor:
    """Return the first non-trivial eigenvectors of the normalized Laplacian.

    The computation is intentionally performed during graph construction. The
    resulting dense tensor is small (``num_nodes x num_embeddings``) and can
    then be moved to the training device with the rest of the graph.
    """
    if num_embeddings < 0:
        raise ValueError("num_embeddings must be non-negative")
    if num_embeddings == 0:
        return torch.empty((num_nodes, 0), dtype=torch.float32)
    if num_nodes < 1:
        return torch.empty((0, num_embeddings), dtype=torch.float32)

    undirected = to_undirected(edge_index, num_nodes=num_nodes)
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float64)
    adjacency[undirected[0], undirected[1]] = 1.0
    degree = adjacency.sum(dim=1)
    inv_sqrt_degree = degree.clamp_min(1.0).pow(-0.5)
    normalized_adjacency = inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]
    laplacian = torch.eye(num_nodes, dtype=torch.float64) - normalized_adjacency
    eigenvectors = torch.linalg.eigh(laplacian).eigenvectors

    # The constant eigenvector is the trivial first component.
    available = max(0, num_nodes - 1)
    result = torch.zeros((num_nodes, num_embeddings), dtype=torch.float32)
    if available:
        width = min(num_embeddings, available)
        result[:, :width] = eigenvectors[:, 1 : width + 1].to(torch.float32)
    return result


def random_walk_structural_encoding(
    edge_index: torch.Tensor,
    num_nodes: int,
    walk_length: int = 8,
) -> torch.Tensor:
    """Return diagonal probabilities of successive random-walk transition powers."""
    if walk_length < 0:
        raise ValueError("walk_length must be non-negative")
    if walk_length == 0:
        return torch.empty((num_nodes, 0), dtype=torch.float32)

    undirected = to_undirected(edge_index, num_nodes=num_nodes)
    transition = torch.zeros((num_nodes, num_nodes), dtype=torch.float64)
    transition[undirected[0], undirected[1]] = 1.0
    degree = transition.sum(dim=1)
    transition = transition / degree.clamp_min(1.0)[:, None]

    power = torch.eye(num_nodes, dtype=torch.float64)
    encodings: list[torch.Tensor] = []
    for _ in range(walk_length):
        power = power @ transition
        encodings.append(torch.diagonal(power))
    return torch.stack(encodings, dim=1).to(torch.float32)
