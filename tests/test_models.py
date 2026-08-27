"""Unit tests for the GATv2 network and adaptive focal loss."""

import pytest
import torch
from torch_geometric.data import Data

from src.models import AdaptiveFocalLoss, GATv2Net


def _toy_graph(
    n_nodes: int = 20, n_edges: int = 60, in_channels: int = 9, edge_dim: int = 2
) -> Data:
    torch.manual_seed(0)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    return Data(
        x=torch.randn(n_nodes, in_channels),
        edge_index=edge_index,
        edge_attr=torch.randn(n_edges, edge_dim),
        y=torch.randint(0, 2, (n_nodes,)),
    )


def test_forward_pass_shapes() -> None:
    """Forward pass returns (N, 2) logits and (N,) probabilities."""
    data = _toy_graph()
    model = GATv2Net(in_channels=9, hidden_channels=32, heads=4, edge_dim=2)
    logits = model(data.x, data.edge_index, data.edge_attr)
    assert logits.shape == (data.num_nodes, 2)
    assert torch.isfinite(logits).all()

    probs = model.predict_prob(data.x, data.edge_index, data.edge_attr)
    assert probs.shape == (data.num_nodes,)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_backward_gradients_flow() -> None:
    """Gradients propagate through the encoder during training."""
    data = _toy_graph()
    model = GATv2Net(in_channels=9, hidden_channels=32, heads=4, edge_dim=2)
    model.train()
    logits = model(data.x, data.edge_index, data.edge_attr)
    loss = logits.sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).any() for g in grads)


def test_without_edge_features() -> None:
    """Model works when edge features are disabled (edge_dim=None)."""
    data = _toy_graph()
    model = GATv2Net(in_channels=9, hidden_channels=32, edge_dim=None)
    logits = model(data.x, data.edge_index)
    assert logits.shape == (data.num_nodes, 2)


def test_invalid_head_scales_raise() -> None:
    """Non-divisible heads vs hidden_channels are rejected at init."""
    with pytest.raises(ValueError):
        GATv2Net(in_channels=9, hidden_channels=10, heads=3)


def test_focal_loss_penalises_focal_regions() -> None:
    """Focal loss (gamma > 0) down-weights easy, confident examples."""
    # Confident but not saturated logits so (1-pt)^gamma is meaningful.
    logits = torch.tensor([[2.0, -2.0], [-2.0, 2.0]], dtype=torch.float32)
    targets = torch.tensor([0, 1])
    ce = AdaptiveFocalLoss(init_alpha=0.5, init_gamma=0.0)
    focal = AdaptiveFocalLoss(init_alpha=0.5, init_gamma=2.0)
    ce_value = ce(logits, targets).item()
    focal_value = focal(logits, targets).item()
    assert focal_value < ce_value


def test_adaptive_alpha_gamma_update() -> None:
    """update_history shifts alpha/gamma based on FN vs FP rates."""
    criterion = AdaptiveFocalLoss(init_alpha=0.25, init_gamma=2.0)
    initial = criterion.current_params()
    criterion.update_history(false_negative_rate=0.8, false_positive_rate=0.1)
    updated = criterion.current_params()
    assert updated["alpha"] > initial["alpha"]
    assert updated["gamma"] > initial["gamma"]

    # Clamping: repeatedly drive imbalance and ensure bounds hold.
    for _ in range(50):
        criterion.update_history(false_negative_rate=1.0, false_positive_rate=0.0)
    assert criterion.min_alpha <= criterion.current_params()["alpha"] <= criterion.max_alpha
    assert criterion.min_gamma <= criterion.current_params()["gamma"] <= criterion.max_gamma


def test_adaptive_loss_params_persist_in_state_dict() -> None:
    """Alpha/gamma buffers travel with checkpoints."""
    criterion = AdaptiveFocalLoss(init_alpha=0.30, init_gamma=1.5)
    state = criterion.state_dict()
    assert "alpha" in state and "gamma" in state
    assert float(state["alpha"]) == pytest.approx(0.30)
    assert float(state["gamma"]) == pytest.approx(1.5)
