"""Adaptive Focal Loss with dynamic alpha/gamma weighting.

Plain cross-entropy is dominated by benign accounts in AML graphs with
≈ 1–5 % positives. Focal loss re-weights the per-class ``alpha`` and the
focusing exponent ``gamma``; on top of that this module *adaptively*
re-tunes both quantities using the observed false-negative (FN) and
false-positive (FP) rates, so the optimizer is pushed to penalise missed
laundering detections when the FN/FN+FP balance deteriorates.

Update rule (see :meth:`AdaptiveFocalLoss.update_history`):

* ``alpha`` shifts toward the positive class when the FN rate exceeds the
  FP rate (missed detections) and vice versa.
* ``gamma`` increases while FNs dominate (harder focus on difficult
  positives) and decreases when FPs dominate.
"""

import torch
from torch import nn


class AdaptiveFocalLoss(nn.Module):
    """Binary/single-logit focal loss with adaptive alpha/gamma.

    Args:
        init_alpha: Weight for the positive (laundering) class.
        init_gamma: Initial focusing exponent.
        min_alpha / max_alpha: Clamping bounds for dynamic alpha updates.
        min_gamma / max_gamma: Clamping bounds for dynamic gamma updates.
        alpha_update_rate: Step size of the alpha update.
        gamma_update_rate: Step size of the gamma update.
    """

    def __init__(
        self,
        init_alpha: float = 0.25,
        init_gamma: float = 2.0,
        min_alpha: float = 0.05,
        max_alpha: float = 0.95,
        min_gamma: float = 1.0,
        max_gamma: float = 5.0,
        alpha_update_rate: float = 0.05,
        gamma_update_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.register_buffer("alpha", torch.tensor(float(init_alpha)))
        self.register_buffer("gamma", torch.tensor(float(init_gamma)))
        self.min_alpha = float(min_alpha)
        self.max_alpha = float(max_alpha)
        self.min_gamma = float(min_gamma)
        self.max_gamma = float(max_gamma)
        self.alpha_update_rate = float(alpha_update_rate)
        self.gamma_update_rate = float(gamma_update_rate)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Raw logits of shape ``(N, 2)``.
            targets: Integer labels of shape ``(N,)``.

        Returns:
            Scalar loss.
        """
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        targets = targets.long()

        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)

        loss = -alpha_t * torch.pow(1.0 - pt, self.gamma) * log_pt
        return loss.mean()

    @torch.no_grad()
    def update_history(self, false_negative_rate: float, false_positive_rate: float) -> None:
        """Adapt alpha/gamma from the observed FN and FP rates.

        Args:
            false_negative_rate: Share of positives missed by the model.
            false_positive_rate: Share of negatives wrongly flagged.
        """
        fnr = float(false_negative_rate)
        fpr = float(false_positive_rate)
        imbalance = fnr - fpr  # > 0 → missed positives dominate, boost focus

        new_alpha = self.alpha + self.alpha_update_rate * imbalance
        self.alpha.fill_(float(new_alpha.clamp(self.min_alpha, self.max_alpha)))

        new_gamma = self.gamma + self.gamma_update_rate * imbalance
        self.gamma.fill_(float(new_gamma.clamp(self.min_gamma, self.max_gamma)))

    def current_params(self) -> dict[str, float]:
        """Return the live alpha/gamma values (nice for logging)."""
        return {"alpha": float(self.alpha), "gamma": float(self.gamma)}
