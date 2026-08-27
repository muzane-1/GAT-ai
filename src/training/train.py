"""Training pipeline — early stopping, gradient clipping and metrics logging.

Runs full-graph (transductive) node classification. Splits nodes into
stratified train/val/test masks, optimises the adaptive focal loss, tracks
F1 / ROC-AUC / PR-AUC per epoch and checkpoints the best validation
checkpoint. All figures are appended to ``checkpoints/metrics_history.json``
for the update pipeline's drift monitoring.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.data_pipeline import build_pyg_data, fetch_transactions
from src.models import AdaptiveFocalLoss, GATv2Net
from src.utils import format_metrics, get_logger, load_config
from src.utils.metrics import compute_metrics

logger = get_logger(__name__)


def make_masks(y: torch.Tensor, val_ratio: float, test_ratio: float, seed: int) -> tuple:
    """Build stratified train/val/test boolean masks.

    Args:
        y: Node labels, shape ``(N,)``.
        val_ratio: Fraction reserved for validation.
        test_ratio: Fraction reserved for the final test split.
        seed: Random state for the split.

    Returns:
        ``(train_mask, val_mask, test_mask)`` as boolean tensors.
    """
    indices = np.arange(len(y))
    labels = y.numpy()

    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    stratify = labels if (len(np.unique(labels)) == 2 and n_pos >= 2 and n_neg >= 2) else None
    train_val_ratio = val_ratio + test_ratio
    train_idx, tmp_idx, _, tmp_labels = train_test_split(
        indices, labels, test_size=train_val_ratio, random_state=seed, stratify=stratify
    )
    relative_test = test_ratio / train_val_ratio
    second_stratify = tmp_labels if len(np.unique(tmp_labels)) == 2 else None
    val_idx, test_idx = train_test_split(
        tmp_idx, test_size=relative_test, random_state=seed, stratify=second_stratify
    )

    train_mask = torch.zeros(len(y), dtype=torch.bool)
    val_mask = torch.zeros(len(y), dtype=torch.bool)
    test_mask = torch.zeros(len(y), dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def evaluate(
    model: GATv2Net,
    data: Any,
    mask: torch.Tensor,
) -> dict[str, float]:
    """Compute classification metrics over a boolean node mask.

    Args:
        model: Trained/eval model.
        data: PyG ``Data`` object.
        mask: Boolean node subset mask.

    Returns:
        Metric mapping from :func:`src.utils.metrics.compute_metrics` plus
        ``expected_fnr`` and ``expected_fpr`` fields used by the adaptive
        loss update rule.
    """
    model.eval()
    with torch.no_grad():
        probs = model.predict_prob(data.x, data.edge_index, data.edge_attr)
    y_true = data.y[mask].numpy()
    y_prob = probs[mask].numpy()

    metrics = compute_metrics(y_true, y_prob)
    positives = y_true == 1
    negatives = y_true == 0
    fn_rate = float(np.mean(y_prob[positives] < 0.5)) if positives.any() else 0.0
    fp_rate = float(np.mean(y_prob[negatives] >= 0.5)) if negatives.any() else 0.0
    metrics["fn_rate"] = fn_rate
    metrics["fp_rate"] = fp_rate
    return metrics


def train_model(
    config: dict[str, Any],
    epochs: int | None = None,
    run_dir: str | Path | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    """Full training run used by the CLI, tuner and update pipeline alike.

    Args:
        config: Parsed YAML configuration.
        epochs: Optional epoch override (tuner passes a smaller value).
        run_dir: Optional output-directory override.
        quiet: When ``True``, epoch-by-epoch logging is suppressed.

    Returns:
        Summary mapping with ``best_val_metrics``, ``test_metrics``,
        ``best_epoch`` and ``checkpoint`` keys.
    """
    data_cfg = config["data"]
    model_cfg = config["model"]
    loss_cfg = config["loss"]
    train_cfg = config["training"]
    paths_cfg = config["paths"]

    out_dir = Path(run_dir or paths_cfg["checkpoints_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = train_cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    fallback_kwargs = {
        "n_accounts": data_cfg.get("fallback_n_accounts", 400),
        "n_transactions": data_cfg.get("fallback_n_transactions", 6000),
        "fraud_ratio": data_cfg.get("fallback_fraud_ratio", 0.02),
        "seed": data_cfg.get("fallback_seed", 42),
    }
    df = fetch_transactions(
        data_cfg["raw_source"],
        n_retry=data_cfg.get("fetch_retry_attempts", 3),
        backoff_seconds=data_cfg.get("fetch_retry_backoff_seconds", 1.0),
        timeout_seconds=data_cfg.get("fetch_timeout_seconds", 20.0),
        fallback_generate=data_cfg.get("fallback_generate", True),
        fallback_kwargs=fallback_kwargs,
    )

    data, scaler = build_pyg_data(
        df, velocity_window_seconds=float(data_cfg.get("velocity_window_seconds", 86400))
    )
    train_mask, val_mask, test_mask = make_masks(
        data.y,
        val_ratio=train_cfg["val_ratio"],
        test_ratio=train_cfg["test_ratio"],
        seed=seed,
    )

    edge_dim = int(data.edge_attr.shape[1]) if model_cfg.get("use_edge_features", True) else None
    model = GATv2Net(
        in_channels=data.num_features,
        hidden_channels=model_cfg["hidden_channels"],
        num_layers=model_cfg["num_layers"],
        heads=model_cfg["heads"],
        dropout=model_cfg["dropout"],
        concat_heads=model_cfg.get("concat_heads", True),
        edge_dim=edge_dim,
    )
    criterion = AdaptiveFocalLoss(
        init_alpha=loss_cfg["init_alpha"],
        init_gamma=loss_cfg["init_gamma"],
        min_alpha=loss_cfg.get("min_alpha", 0.05),
        max_alpha=loss_cfg.get("max_alpha", 0.95),
        min_gamma=loss_cfg.get("min_gamma", 1.0),
        max_gamma=loss_cfg.get("max_gamma", 5.0),
        alpha_update_rate=loss_cfg.get("alpha_update_rate", 0.05),
        gamma_update_rate=loss_cfg.get("gamma_update_rate", 0.1),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 5e-4),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=train_cfg.get("scheduler_factor", 0.5),
        patience=train_cfg.get("scheduler_patience", 8),
    )

    max_epochs = int(epochs or train_cfg["epochs"])
    patience = train_cfg["early_stopping_patience"]
    clip_norm = train_cfg.get("grad_clip_norm", 1.0)

    history: list[dict[str, Any]] = []
    best_val_pr_auc = -1.0
    best_epoch = -1
    best_val_metrics: dict[str, float] = {}
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index, data.edge_attr if edge_dim else None)
        loss = criterion(logits[train_mask], data.y[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        val_metrics = evaluate(model, data, val_mask)
        scheduler.step(val_metrics["pr_auc"] if not np.isnan(val_metrics["pr_auc"]) else 0.0)
        criterion.update_history(val_metrics["fn_rate"], val_metrics["fp_rate"])

        improved = val_metrics["pr_auc"] > best_val_pr_auc
        if np.isnan(val_metrics["pr_auc"]):
            improved = best_val_pr_auc < 0.0
        if improved:
            best_val_pr_auc = val_metrics["pr_auc"]
            best_epoch = epoch
            best_val_metrics = val_metrics
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "criterion_state_dict": criterion.state_dict(),
                    "scaler": scaler,
                    "config": config,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                out_dir / "best.pt",
            )
            patience_counter = 0
        else:
            patience_counter += 1

        history.append(
            {
                "epoch": epoch,
                "loss": float(loss.item()),
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "alpha": criterion.current_params()["alpha"],
                "gamma": criterion.current_params()["gamma"],
            }
        )
        if not quiet:
            logger.info(
                "epoch %03d | loss=%.4f | %s | alpha=%.3f gamma=%.2f",
                epoch,
                loss.item(),
                format_metrics(val_metrics),
                criterion.current_params()["alpha"],
                criterion.current_params()["gamma"],
            )

        if patience_counter >= patience:
            logger.info("Early stopping at epoch %d (best: %d)", epoch, best_epoch)
            break

    # Final test evaluation with the best checkpoint.
    checkpoint = torch.load(out_dir / "best.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, data, test_mask)
    logger.info("test %s", format_metrics(test_metrics))

    summary = {
        "best_epoch": best_epoch,
        "best_val_metrics": {
            k: v for k, v in best_val_metrics.items() if k != "fn_rate" and k != "fp_rate"
        },
        "test_metrics": {
            k: v for k, v in test_metrics.items() if k != "fn_rate" and k != "fp_rate"
        },
        "checkpoint": str(out_dir / "best.pt"),
    }

    metrics_entry = {
        "timestamp": time.time(),
        "history": history,
        "summary": summary,
    }
    metrics_file = Path(paths_cfg["metrics_history"])
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(metrics_file.read_text()) if metrics_file.exists() else []
    existing.append(metrics_entry)
    metrics_file.write_text(json.dumps(existing, indent=2))

    return summary


def main() -> None:
    """CLI entrypoint for the full training pipeline."""
    parser = argparse.ArgumentParser(description="Train the AML GATv2 model")
    parser.add_argument("--config", default="config/config.yaml", help="YAML config path")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch override")
    parser.add_argument("--output", default=None, help="Checkpoint directory override")
    args = parser.parse_args()

    config = load_config(args.config)
    summary = train_model(config, epochs=args.epochs, run_dir=args.output)
    logger.info("summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
