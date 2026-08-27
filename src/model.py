import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm


class GATv2(nn.Module):
    """
    Professional Deep 4-Layer GATv2 Architecture for AML Fraud Detection.
    Includes Residual Connections, Layer Normalization, and Multi-hop Attention.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, heads=4, dropout=0.2):
        super(GATv2, self).__init__()
        self.dropout = dropout

        # Layer 1: Input Graph Feature Encoder
        self.conv1 = GATv2Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            edge_dim=edge_dim
        )
        self.norm1 = LayerNorm(hidden_channels * heads)

        # Layer 2: Deep Feature Representation (Intermediate)
        self.conv2 = GATv2Conv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            edge_dim=edge_dim
        )
        self.norm2 = LayerNorm(hidden_channels * heads)

        # Layer 3: High-Level Multi-Hop Graph Analysis
        self.conv3 = GATv2Conv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            edge_dim=edge_dim
        )
        self.norm3 = LayerNorm(hidden_channels * heads)

        # Layer 4: Output Classification Layer
        self.conv4 = GATv2Conv(
            in_channels=hidden_channels * heads,
            out_channels=out_channels,
            heads=1,
            concat=False,
            edge_dim=edge_dim
        )

    def forward(self, x, edge_index, edge_attr):
        """
        Forward propagation with Skip Connections and Layer Normalization.
        """
        # --- Layer 1 ---
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.norm1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # --- Layer 2 (With Residual Skip Connection) ---
        residual1 = x
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.norm2(x + residual1)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # --- Layer 3 (With Residual Skip Connection) ---
        residual2 = x
        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        x = self.norm3(x + residual2)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # --- Layer 4 (Final Classification Readout) ---
        x = self.conv4(x, edge_index, edge_attr=edge_attr)

        return x