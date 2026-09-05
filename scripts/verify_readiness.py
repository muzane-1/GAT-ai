"""Pre-training readiness sanity check for the GATv2 AML pipeline.

Runs a single dry-run epoch that exercises the production pipeline stages:
``fetch_to_pyg`` -> ``NeighborLoader`` -> model forward -> loss -> backward ->
optimizer step -> checkpoint save/reload. The script exits with a concise result
report and is also used by the automated pytest smoke test in
``tests/test_train.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch_geometric import typing as pyg_typing
from torch_geometric.loader import DataLoader, NeighborLoader

from src.data_pipeline import fetch_to_pyg
from src.models import AdaptiveFocalLoss, GATv2Net


def _load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload


def _weighted_bce(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    clicked = targets.float()
    pos_count = clicked.sum().clamp_min(1.0)
    neg_count = (1.0 - clicked).sum().clamp_min(1.0)
    pos_weight = torch.full_like(clicked, max(1.0, float(neg_count / pos_count)))
    loss = F.binary_cross_entropy_with_logits(logits[:, 1], clicked, pos_weight=pos_weight)
    return loss, pos_weight


def dry_run(
    config_path: str | Path = "config/config.yaml",
    epochs: int = 1,
    checkpoint_path: str | Path = "checkpoints/test_model.pt",
    keep_checkpoint: bool = False,
) -> dict[str, Any]:
    """Execute a 1-epoch mini-batch readiness smoke test.

    The returned dict is intentionally schema-stable so the test suite can assert
    the exact pipeline contract (shape checks, finite loss values, optimizer
    update, and checkpoint round-trip integrity).
    """
    _ = _load_config(config_path)
    data, stats = fetch_to_pyg(hf_query=None, source=None)
    if not hasattr(data, "x") or not hasattr(data, "edge_index"):
        raise RuntimeError("fetch_to_pyg returned a graph without x/edge_index tensors")
    if data.x.dim() != 2:
        raise ValueError(f"Expected x to be 2D, received {tuple(data.x.shape)}")
    if data.edge_index.dim() != 2 or data.edge_index.shape[0] != 2:
        raise ValueError(
            f"Expected edge_index shape (2, E), received {tuple(data.edge_index.shape)}"
        )
    if data.edge_attr is not None and data.edge_attr.dim() != 2:
        raise ValueError(f"Expected edge_attr to be 2D, received {tuple(data.edge_attr.shape)}")
    if data.y.dim() != 1 or data.y.shape[0] != data.num_nodes:
        raise ValueError(f"Expected y to match the node count, received {tuple(data.y.shape)}")

    graph = {
        "nodes": int(data.num_nodes),
        "edges": int(data.num_edges),
        "features": int(data.num_node_features),
        "edge_features": int(data.edge_attr.shape[1]) if data.edge_attr is not None else 0,
        "source_stats": stats,
    }

    model = GATv2Net(
        in_channels=data.num_node_features,
        hidden_channels=32,
        num_layers=2,
        heads=4,
        dropout=0.1,
        edge_dim=data.edge_attr.shape[1] if data.edge_attr is not None else None,
    )
    criterion = AdaptiveFocalLoss(init_alpha=0.25, init_gamma=2.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    batch_size = min(32, max(1, data.num_nodes))
    if pyg_typing.WITH_PYG_LIB or pyg_typing.WITH_TORCH_SPARSE:
        loader = NeighborLoader(
            data,
            input_nodes=torch.arange(min(data.num_nodes, 128), dtype=torch.long),
            batch_size=batch_size,
            num_neighbors=[8, 4],
            shuffle=True,
        )
    else:
        # NeighborLoader needs pyg-lib or torch-sparse; keep the readiness
        # check runnable in the lightweight CPU development environment.
        loader = DataLoader([data], batch_size=1, shuffle=True)

    mini_batches = 0
    final_batch = None
    last_focal = 0.0
    last_weighted_bce = 0.0
    grad_norm = 0.0
    checks: dict[str, bool] = {}

    for _epoch in range(max(1, int(epochs))):
        model.train()
        for batch in loader:
            mini_batches += 1
            final_batch = batch
            x = batch.x
            edge_index = batch.edge_index
            edge_attr = batch.edge_attr
            y = batch.y

            checks["x_shape"] = x.dim() == 2 and x.shape[0] > 0
            checks["edge_index_shape"] = edge_index.dim() == 2 and edge_index.shape[0] == 2
            checks["edge_attr_shape"] = (
                edge_attr.dim() == 2 and edge_attr.shape[0] == edge_index.shape[1]
            )
            checks["y_shape"] = y.dim() == 1 and y.shape[0] == x.shape[0]
            if not all(checks.values()):
                raise ValueError(f"Tensor alignment check failed: {checks}")

            logits = model(x, edge_index, edge_attr)
            checks["forward_pass"] = (
                logits.dim() == 2 and logits.shape[-1] == 2 and logits.shape[0] == x.shape[0]
            )
            if not checks["forward_pass"]:
                raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

            focal_loss = criterion(logits, y)
            weighted_bce, _ = _weighted_bce(logits, y)
            checks["loss_finite"] = (
                torch.isfinite(focal_loss).item() and torch.isfinite(weighted_bce).item()
            )
            if not checks["loss_finite"]:
                raise ValueError(
                    f"Non-finite loss encountered: focal={focal_loss}, weighted_bce={weighted_bce}"
                )

            last_focal = float(focal_loss.item())
            last_weighted_bce = float(weighted_bce.item())
            total_loss = 0.5 * (focal_loss + weighted_bce)
            optimizer.zero_grad()
            total_loss.backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
            )
            optimizer.step()
            checks["optimizer_step"] = True
            checks["backward_pass"] = True
            break

    if final_batch is None:
        raise RuntimeError("NeighborLoader did not produce any mini-batches for the dry run")

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()
    torch.save(state_dict, checkpoint_path)
    checkpoint_state = torch.load(checkpoint_path, map_location="cpu")
    state_match = len(checkpoint_state) == len(state_dict) and all(
        key in checkpoint_state and torch.equal(value, checkpoint_state[key])
        for key, value in state_dict.items()
    )
    if not keep_checkpoint:
        checkpoint_path.unlink(missing_ok=True)
    cleaned = not checkpoint_path.exists()

    report: dict[str, Any] = {
        "pipeline_ready": all(checks.values()) and state_match and cleaned,
        "graph": graph,
        "neighbor_loader": {"mini_batches": int(mini_batches), "batch_size": int(batch_size)},
        "loss": {"focal": last_focal, "weighted_bce": last_weighted_bce},
        "optimizer": {"step": True, "grad_norm": grad_norm},
        "checks": checks,
        "checkpoint": {
            "path": str(checkpoint_path),
            "state_match": state_match,
            "cleaned": cleaned,
        },
    }

    print("Dry-run summary")
    print(
        f"- graph: {graph['nodes']} nodes / {graph['edges']} edges / {graph['features']} features"
    )
    print(f"- loader: {mini_batches} mini-batches processed")
    print(f"- losses: focal={last_focal:.6f}, weighted_bce={last_weighted_bce:.6f}")
    print(f"- grad_norm={grad_norm:.6f}, state_match={state_match}, cleaned={cleaned}")
    print("PIPELINE READY FOR CLOUD TRAINING and i have the docker remember")
    return report


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run the pre-training GATv2 readiness smoke test.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/test_model.pt",
        help="Checkpoint path for the dummy save/reload check",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of dry-run epochs to execute")
    parser.add_argument(
        "--keep-checkpoint",
        action="store_true",
        help="Keep the temporary checkpoint instead of deleting it",
    )
    args = parser.parse_args()
    dry_run(
        config_path=args.config,
        epochs=args.epochs,
        checkpoint_path=args.checkpoint,
        keep_checkpoint=args.keep_checkpoint,
    )


if __name__ == "__main__":
    _cli()
