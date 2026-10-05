"""
model/architecture.py
═════════════════════
Multi-Task LSTM Fitness Model

Architecture:
                 Input: (Batch, Seq=32, Features=34)
                        ↓
                 Input Projection: Linear(34 → 128) + LayerNorm
                        ↓
                 Temporal Encoder: BiLSTM (128 → 256, 2 layers, dropout=0.3)
                        ↓
              ┌──────── Context Vector (mean + last hidden) ────────┐
              │                                                      │
              ▼                        ▼                            ▼
     Exercise Head            Rep State Head                Form Score Head
     FC(512→128→10)           FC(512→128→3)                FC(512→64→1)
     Softmax                  Softmax                       Sigmoid
              │                        │                            │
              ▼                        ▼                            ▼
     Exercise class (0-9)     State (0=ready,1=down,2=up)   Score (0.0-1.0)

لماذا BiLSTM:
  - يشوف الـ sequence من الاتجاهين → يفهم context أحسن
  - أخف من Transformer في inference
  - مناسب لـ sequences زي 32 frame

الحجم التقريبي للموديل: ~2.5M parameter (~10MB)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Constants ─────────────────────────────────────────────────────────────────

N_FEATURES  = 34   # 17 keypoints × 2 coords (x,y)
N_EXERCISES = 15   # عدد التمارين (updated from 10 → 15 in v4)
N_STATES    = 3    # ready / down / up


# ═══════════════════════════════════════════════════════════════════════════════
#  Building blocks
# ═══════════════════════════════════════════════════════════════════════════════

class InputProjection(nn.Module):
    """
    Projects raw keypoint features to a richer embedding space.
    Also handles the (0,0) missing keypoint sentinel gracefully:
    missing joints get projected to a learned 'missing' embedding.
    """

    def __init__(self, in_features: int = N_FEATURES, out_features: int = 128) -> None:
        super().__init__()
        self.proj  = nn.Linear(in_features, out_features)
        self.norm  = nn.LayerNorm(out_features)
        self.act   = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 34) → (B, T, 128)"""
        return self.act(self.norm(self.proj(x)))


class TemporalEncoder(nn.Module):
    """
    Bi-directional LSTM encoder.
    Captures temporal motion patterns across the 32-frame window.
    """

    def __init__(
        self,
        input_size:  int = 128,
        hidden_size: int = 128,   # each direction — total output = 256
        num_layers:  int = 2,
        dropout:     float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size   = input_size,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            bidirectional= True,
            dropout      = dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.out_size = hidden_size * 2   # bidirectional → ×2

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, 128)
        Returns:
          seq_out  : (B, T, 256)  — all timestep outputs
          context  : (B, 256)     — global context (mean pooling)
        """
        seq_out, _ = self.lstm(x)
        seq_out    = self.dropout(seq_out)
        context    = seq_out.mean(dim=1)   # mean pooling over time
        return seq_out, context


class AttentionPool(nn.Module):
    """
    Soft attention pooling over the LSTM sequence.
    Learns WHICH frames are most informative for each task.
    """

    def __init__(self, hidden_size: int = 256) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, seq_out: torch.Tensor) -> torch.Tensor:
        """seq_out: (B, T, H) → (B, H) weighted sum"""
        scores  = self.attn(seq_out).squeeze(-1)    # (B, T)
        weights = F.softmax(scores, dim=-1)          # (B, T)
        pooled  = (weights.unsqueeze(-1) * seq_out).sum(dim=1)  # (B, H)
        return pooled


class ClassificationHead(nn.Module):
    """Generic FC head for classification tasks."""

    def __init__(self, in_features: int, hidden: int, n_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RegressionHead(nn.Module):
    """Generic FC head for regression tasks (outputs 0-1 via sigmoid)."""

    def __init__(self, in_features: int, hidden: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main model
# ═══════════════════════════════════════════════════════════════════════════════

class FitnessModel(nn.Module):
    """
    Multi-task fitness model.

    Inputs:
        x : (Batch, Seq=32, 34)  — normalized keypoint sequence

    Outputs:
        exercise_logits : (B, 10) — raw logits for exercise classification
        state_logits    : (B, 3)  — raw logits for rep state (ready/down/up)
        form_score      : (B,)    — form quality score in [0, 1]

    During inference: apply softmax to logits, use form_score directly.
    During training : use CrossEntropyLoss on logits (handles softmax internally).
    """

    def __init__(
        self,
        n_features:  int   = N_FEATURES,
        n_exercises: int   = N_EXERCISES,
        n_states:    int   = N_STATES,
        hidden_size: int   = 128,
        n_layers:    int   = 2,
        dropout:     float = 0.3,
    ) -> None:
        super().__init__()

        self.n_features  = n_features
        self.n_exercises = n_exercises
        self.n_states    = n_states

        # Encoder components
        self.input_proj = InputProjection(n_features, hidden_size)
        self.encoder    = TemporalEncoder(hidden_size, hidden_size, n_layers, dropout)
        self.attn_pool  = AttentionPool(hidden_size * 2)   # bidirectional

        enc_out_size = hidden_size * 2   # 256 after BiLSTM

        # Three task-specific heads
        self.exercise_head = ClassificationHead(enc_out_size, 128, n_exercises, dropout)
        self.state_head    = ClassificationHead(enc_out_size, 128, n_states,    dropout)
        self.form_head     = RegressionHead(enc_out_size, 64, dropout)

        self._init_weights()

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, T, 34)

        Returns:
          exercise_logits : (B, 10)
          state_logits    : (B, 3)
          form_score      : (B,)
        """
        # ── 1. Project input features
        proj = self.input_proj(x)          # (B, T, 128)

        # ── 2. Temporal encoding
        seq_out, context = self.encoder(proj)   # (B, T, 256), (B, 256)

        # ── 3. Attention pooling (task-aware context)
        attn_context = self.attn_pool(seq_out)   # (B, 256)

        # Combine mean pooling + attention pooling
        combined = context + attn_context         # (B, 256)

        # ── 4. Task heads
        exercise_logits = self.exercise_head(combined)   # (B, 10)
        state_logits    = self.state_head(combined)      # (B, 3)
        form_score      = self.form_head(combined)       # (B,)

        return exercise_logits, state_logits, form_score

    # ── Inference convenience ─────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self, x: torch.Tensor
    ) -> Tuple[int, str, float, float]:
        """
        Single-sample inference. x: (T, 34) or (1, T, 34).

        Returns:
          exercise_idx  : int
          state_name    : str  ("ready" | "down" | "up")
          form_score    : float  [0, 1]
          exercise_conf : float  confidence of exercise prediction
        """
        self.eval()
        if x.dim() == 2:
            x = x.unsqueeze(0)   # add batch dim

        ex_logits, st_logits, form_sc = self(x)

        ex_probs  = F.softmax(ex_logits, dim=-1)[0]
        st_probs  = F.softmax(st_logits, dim=-1)[0]

        ex_idx    = int(ex_probs.argmax())
        st_idx    = int(st_probs.argmax())
        form_val  = float(form_sc[0])
        ex_conf   = float(ex_probs[ex_idx])

        state_names = {0: "ready", 1: "down", 2: "up"}
        return ex_idx, state_names[st_idx], form_val, ex_conf

    # ── Weight init ───────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    # ── Model info ────────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        params = self.count_parameters()
        return (
            f"FitnessModel(\n"
            f"  n_features={self.n_features}, n_exercises={self.n_exercises}, "
            f"n_states={self.n_states}\n"
            f"  Parameters: {params:,} (~{params*4/1e6:.1f} MB)\n"
            f")"
        )


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model(
    hidden_size: int   = 128,
    n_layers:    int   = 2,
    dropout:     float = 0.3,
    device:      str   = "cpu",
) -> FitnessModel:
    """Create and return a FitnessModel on the given device."""
    model = FitnessModel(
        hidden_size = hidden_size,
        n_layers    = n_layers,
        dropout     = dropout,
    ).to(device)
    print(f"[Model] {model}")
    return model


if __name__ == "__main__":
    # Quick sanity check
    m = build_model(device="cpu")
    x = torch.randn(4, 32, 34)   # batch=4, seq=32, features=34
    ex_l, st_l, form = m(x)
    print(f"exercise_logits : {ex_l.shape}")   # (4, 10)
    print(f"state_logits    : {st_l.shape}")   # (4, 3)
    print(f"form_score      : {form.shape}")   # (4,)
    print("Architecture OK ✓")
