import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing extreme class imbalance in graph fraud detection tasks.
    Down-weights easy positive/negative examples and focuses on hard fraudulent nodes.
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def compute_metrics(
    y_true: torch.Tensor, y_pred_prob: torch.Tensor, threshold: float = 0.5
) -> dict:
    """
    Computes standard classification evaluation metrics for imbalanced datasets.
    """
    y_true_np = y_true.cpu().numpy()
    y_prob_np = torch.softmax(y_pred_prob, dim=-1)[:, 1].detach().cpu().numpy()
    y_pred_np = (y_prob_np >= threshold).astype(int)

    metrics = {
        "f1_score": f1_score(y_true_np, y_pred_np, zero_division=0),
        "precision": precision_score(y_true_np, y_pred_np, zero_division=0),
        "recall": recall_score(y_true_np, y_pred_np, zero_division=0),
        "roc_auc": roc_auc_score(y_true_np, y_prob_np) if len(np.unique(y_true_np)) > 1 else 0.0,
    }
    return metrics


def save_checkpoint(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, filepath: str
):
    """
    Saves PyTorch Geometric model weights and optimizer state to disk.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None):
    """
    Loads saved model weights and state dict for evaluation or prediction.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No checkpoint found at '{filepath}'")

    checkpoint = torch.load(filepath, weights_only=True)  # nosec B614
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("epoch", 0)
