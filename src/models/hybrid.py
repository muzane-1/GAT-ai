"""Local-global GATv2 and graph-transformer node classifier."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv


class GATv2GraphTransformer(nn.Module):
    """GraphGPS-style block combining local GATv2 and global self-attention."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.3,
        edge_dim: int | None = None,
        lap_pe_dim: int = 0,
        rw_pe_dim: int = 0,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if hidden_channels % heads:
            raise ValueError("hidden_channels must be divisible by heads")

        self.input_proj = nn.Linear(in_channels + lap_pe_dim + rw_pe_dim, hidden_channels)
        self.local_layers = nn.ModuleList()
        self.global_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        for _ in range(num_layers):
            self.local_layers.append(
                GATv2Conv(
                    hidden_channels,
                    hidden_channels // heads,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    add_self_loops=False,
                    share_weights=True,
                )
            )
            self.global_layers.append(
                nn.MultiheadAttention(
                    hidden_channels,
                    heads,
                    dropout=dropout,
                    batch_first=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_channels))
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        lap_pe: torch.Tensor | None = None,
        rw_pe: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return node logits while keeping PE separate from base features."""
        pe_parts = [part for part in (lap_pe, rw_pe) if part is not None]
        if pe_parts:
            x = torch.cat([x, *pe_parts], dim=-1)
        x = self.input_proj(x)
        for local, global_attention, norm in zip(
            self.local_layers, self.global_layers, self.norms, strict=True
        ):
            residual = x
            local_output = local(x, edge_index, edge_attr=edge_attr)
            global_output, _ = global_attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
            x = norm(residual + self.dropout(local_output + global_output.squeeze(0)))
            x = torch.nn.functional.gelu(x)
        return self.classifier(x)

    def predict_prob(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        lap_pe: torch.Tensor | None = None,
        rw_pe: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return positive-class probabilities for each node."""
        logits = self.forward(x, edge_index, edge_attr, lap_pe, rw_pe)
        return torch.softmax(logits, dim=-1)[:, 1]
