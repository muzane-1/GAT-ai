"""GATv2 encoder for AML node classification.

The network stacks PyG ``GATv2Conv`` layers with:

* **Multi-head attention** — heads are concatenated by default so channel
  width is preserved between layers (grid-friendly for tuning).
* **LayerNorm + residual (skip) connections** — stabilise deeper stacks and
  prevent over-smoothing.
* **Dropout** on both activations and attention weights — regularisation
  tuned for heavily imbalanced AML graphs.

Output is two logits (benign/laundering) per node; a softmax + picking the
positive class probability happens in training/evaluation.
"""

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv


class GATv2Net(nn.Module):
    """Multi-layer GATv2 with LayerNorm, residuals and dropout.

    Args:
        in_channels: Node feature count.
        hidden_channels: Channel width shared by every layer.
        num_layers: Number of graph convolutional blocks (>= 1).
        heads: Attention heads per layer (>= 1).
        dropout: Dropout probability applied to activations and attention.
        concat_heads: Concatenate (``True``) or average (``False``) heads.
        edge_dim: Optional edge-feature dimension forwarded to every layer.
        num_classes: Output logits width; default binary.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.3,
        concat_heads: bool = True,
        edge_dim: int | None = None,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if concat_heads and hidden_channels % heads != 0:
            raise ValueError("hidden_channels must be divisible by heads when concatenating")

        self.dropout = nn.Dropout(dropout)
        self.heads = heads
        self.concat_heads = concat_heads

        head_out = hidden_channels // heads if concat_heads else hidden_channels

        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GATv2Conv(
                    hidden_channels,
                    head_out,
                    heads=heads,
                    concat=concat_heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    add_self_loops=False,
                    share_weights=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Node features, shape ``(N, in_channels)``.
            edge_index: COO edge indices, shape ``(2, E)``.
            edge_attr: Optional edge features, shape ``(E, edge_dim)``.

        Returns:
            Logits of shape ``(N, num_classes)``.
        """
        x = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms, strict=True):
            residual = x
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = nn.functional.elu(x)
            x = self.dropout(x)
            x = residual + x  # skip connection guards against over-smoothing
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
