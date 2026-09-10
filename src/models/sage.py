"""GraphSAGE encoder for AML node classification (neighborhood aggregation)."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    """Multi-layer GraphSAGE with LayerNorm, residuals and dropout."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        aggr: str = "mean",
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.dropout = nn.Dropout(dropout)
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList(
            [SAGEConv(hidden_channels, hidden_channels, aggr=aggr) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers)])
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del edge_attr  # SAGE aggregation is topology-only; kept for API parity.
        x = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms, strict=True):
            residual = x
            x = conv(x, edge_index)
            x = norm(x)
            x = nn.functional.elu(x)
            x = self.dropout(x)
            x = residual + x
        return self.classifier(x)

    def predict_prob(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the positive-class probability per node, shape ``(N,)``."""
        logits = self.forward(x, edge_index, edge_attr=edge_attr)
        return torch.softmax(logits, dim=-1)[:, 1]
