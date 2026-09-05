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
from torch import nn

from src.data_pipeline import build_pyg_data, fetch_transactions
from src.models import AdaptiveFocalLoss, GATv2GraphTransformer, GATv2Net
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
    model: nn.Module,
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
        if isinstance(model, GATv2GraphTransformer):
            probs = model.predict_prob(
                data.x,
                data.edge_index,
                data.edge_attr,
                getattr(data, "lap_pe", None),
                getattr(data, "rw_pe", None),
            )
        else:
            probs = model.predict_prob(data.x, data.edge_index, data.edge_attr)
    y_true = data.y[mask].detach().cpu().numpy()
    y_prob = probs[mask].detach().cpu().numpy()

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
    if model_cfg.get("architecture", "gatv2") == "hybrid":
        model: nn.Module = GATv2GraphTransformer(
            in_channels=data.num_features,
            hidden_channels=model_cfg["hidden_channels"],
            num_layers=model_cfg["num_layers"],
            heads=model_cfg["heads"],
            dropout=model_cfg["dropout"],
            edge_dim=edge_dim,
            lap_pe_dim=data.lap_pe.size(1),
            rw_pe_dim=data.rw_pe.size(1),
        )
    else:
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
    max_epochs = int(epochs or train_cfg["epochs"])
    patience = train_cfg["early_stopping_patience"]
    clip_norm = train_cfg.get("grad_clip_norm", 1.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data = data.to(device)
    train_mask, val_mask, test_mask = (
        train_mask.to(device),
        val_mask.to(device),
        test_mask.to(device),
    )
    amp_enabled = bool(train_cfg.get("amp", True) and device.type == "cuda")
    scaler_amp = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    warmup_epochs = min(int(train_cfg.get("warmup_epochs", 5)), max_epochs)
    min_lr = float(train_cfg.get("scheduler_min_lr", 1e-5))

    def schedule(epoch: int) -> float:
        if epoch <= warmup_epochs:
            return epoch / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(max_epochs - warmup_epochs, 1)
        return (min_lr / train_cfg["lr"]) + (1 - min_lr / train_cfg["lr"]) * (
            1 + np.cos(np.pi * progress)
        ) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    mlflow_run = None
    if config.get("tracking", {}).get("enabled", False):
        import mlflow

        mlflow.set_experiment(config["tracking"].get("experiment_name", "aml-gnn"))
        mlflow_run = mlflow.start_run()
        mlflow.log_params({"architecture": model_cfg.get("architecture", "gatv2"), **train_cfg})

    history: list[dict[str, Any]] = []
    best_val_pr_auc = -1.0
    best_epoch = -1
    best_val_metrics: dict[str, float] = {}
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            if isinstance(model, GATv2GraphTransformer):
                logits = model(
                    data.x,
                    data.edge_index,
                    data.edge_attr if edge_dim else None,
                    data.lap_pe,
                    data.rw_pe,
                )
            else:
                logits = model(data.x, data.edge_index, data.edge_attr if edge_dim else None)
            loss = criterion(logits[train_mask], data.y[train_mask])
        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        scaler_amp.step(optimizer)
        scaler_amp.update()

        val_metrics = evaluate(model, data, val_mask)
        scheduler.step()
        criterion.update_history(val_metrics["fn_rate"], val_metrics["fp_rate"])
        if mlflow_run is not None:
            import mlflow

            mlflow.log_metrics(
                {
                    "loss": float(loss.item()),
                    "pr_auc": val_metrics["pr_auc"],
                    "f1": val_metrics["f1"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )

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
    checkpoint = torch.load(  # nosec B614
        out_dir / "best.pt", weights_only=False
    )
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
    if mlflow_run is not None:
        import mlflow

        mlflow.log_artifact(str(out_dir / "best.pt"))
        mlflow.end_run()

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
