"""
model/losses.py
═══════════════
Multi-task loss function للـ FitnessModel.

Total Loss = α·L_exercise + β·L_state + γ·L_form

حيث:
  L_exercise : Focal Loss (يركز على الأمثلة الصعبة — يعالج class imbalance)
  L_state    : Cross-Entropy with label smoothing
  L_form     : MSE + Huber Loss hybrid (robust to outliers)

الـ Focal Loss مهم هنا لأن:
  - بعض التمارين (زي Burpee) أصعب من غيرها
  - الـ class distribution مش balanced
  - الـ model لازم يركز على الـ hard examples
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
#  Focal Loss
# ═══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    Parameters
    ----------
    gamma   : focusing parameter (0 = standard CE, 2 = standard focal)
    alpha   : class weight tensor or None
    reduction: "mean" | "sum" | "none"
    """

    def __init__(
        self,
        gamma:     float = 2.0,
        alpha:     float = None,
        reduction: str   = "mean",
    ) -> None:
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,   # (B, C)
        targets: torch.Tensor,  # (B,) long
    ) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")   # (B,)
        p_t     = torch.exp(-ce_loss)
        focal   = (1 - p_t) ** self.gamma * ce_loss

        if self.alpha is not None:
            focal = self.alpha * focal

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


# ═══════════════════════════════════════════════════════════════════════════════
#  Form Score Loss
# ═══════════════════════════════════════════════════════════════════════════════

class FormScoreLoss(nn.Module):
    """
    Hybrid loss for form score regression:
      L = MSE + Huber

    - MSE: penalizes large errors strongly
    - Huber: robust to label noise (form annotations can be noisy)

    weight_mse + weight_huber should sum to 1.0
    """

    def __init__(self, weight_mse: float = 0.5, weight_huber: float = 0.5, delta: float = 0.2) -> None:
        super().__init__()
        self.w_mse   = weight_mse
        self.w_huber = weight_huber
        self.huber   = nn.HuberLoss(delta=delta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse   = F.mse_loss(pred, target)
        huber = self.huber(pred, target)
        return self.w_mse * mse + self.w_huber * huber


# ═══════════════════════════════════════════════════════════════════════════════
#  Multi-task Loss
# ═══════════════════════════════════════════════════════════════════════════════

class MultiTaskLoss(nn.Module):
    """
    Combined loss for the 3-head FitnessModel.

    L_total = weight_ex · L_exercise
            + weight_state · L_state
            + weight_form · L_form

    Defaults:
      weight_ex    = 1.0  (exercise classification — primary task)
      weight_state = 1.0  (rep state — equally important)
      weight_form  = 0.5  (form score — slightly less weight, noisier labels)

    Parameters
    ----------
    weight_ex    : weight for exercise classification loss
    weight_state : weight for rep state loss
    weight_form  : weight for form score loss
    focal_gamma  : gamma for Focal Loss (0 disables focal effect)
    label_smooth : label smoothing for rep state CE
    """

    def __init__(
        self,
        weight_ex:    float = 1.0,
        weight_state: float = 1.0,
        weight_form:  float = 0.5,
        focal_gamma:  float = 2.0,
        label_smooth: float = 0.1,
    ) -> None:
        super().__init__()
        self.w_ex    = weight_ex
        self.w_state = weight_state
        self.w_form  = weight_form

        self.loss_ex    = FocalLoss(gamma=focal_gamma)
        self.loss_state = nn.CrossEntropyLoss(label_smoothing=label_smooth)
        self.loss_form  = FormScoreLoss()

    def forward(
        self,
        ex_logits:    torch.Tensor,   # (B, 10)
        state_logits: torch.Tensor,   # (B, 3)
        form_score:   torch.Tensor,   # (B,)
        y_ex:         torch.Tensor,   # (B,) long
        y_state:      torch.Tensor,   # (B,) long
        y_form:       torch.Tensor,   # (B,) float
    ) -> Tuple[torch.Tensor, dict]:
        """
        Returns:
          total_loss : scalar tensor
          breakdown  : dict with individual losses for logging
        """
        l_ex    = self.loss_ex(ex_logits, y_ex)
        l_state = self.loss_state(state_logits, y_state)
        l_form  = self.loss_form(form_score, y_form)

        total = self.w_ex * l_ex + self.w_state * l_state + self.w_form * l_form

        breakdown = {
            "loss_total":    float(total),
            "loss_exercise": float(l_ex),
            "loss_state":    float(l_state),
            "loss_form":     float(l_form),
        }

        return total, breakdown
