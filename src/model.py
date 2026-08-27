"""Backward-compatible shim: re-export the refactored GATv2 architecture.

Notebooks continue to do ``from src.model import GATv2`` (or the legacy
``GATv2AMLModel``) with the original constructor signature; the underlying
implementation lives in :class:`src.models.gatv2.GATv2Net`.
"""

from src.models.gatv2 import GATv2Net


class GATv2(GATv2Net):
    """Legacy-named wrapper matching the original notebook constructor.

    Args:
        in_channels: Node feature count.
        hidden_channels: Channel width per layer.
        out_channels: Output logits width (legacy name; default binary).
        edge_dim: Edge feature dimension.
        heads: Attention heads per layer.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int,
        heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        # Pick the largest head count (<= heads) that divides hidden_channels
        # so concatenated heads preserve channel parity for skip connections.
        resolved_heads = next((h for h in range(int(heads), 0, -1) if hidden_channels % h == 0), 1)
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=3,
            heads=resolved_heads,
            dropout=dropout,
            concat_heads=True,
            edge_dim=edge_dim,
            num_classes=out_channels,
        )


# Legacy notebooks additionally import this historical alias.
GATv2AMLModel = GATv2

__all__ = ["GATv2", "GATv2AMLModel", "GATv2Net"]
