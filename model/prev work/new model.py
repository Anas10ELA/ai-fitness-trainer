"""
scripts/train.py  ←  model definition section
══════════════════════════════════════════════
ST-GCN Lite — drop-in replacement for the baseline ST-GCN.

Why "Lite"?
-----------
The baseline model (64→128→256, 9 blocks, ~2.59M params) was overfitting
severely on a small dataset (~100 videos × 5× augmentation ≈ 500 sequences):
    Train acc: 84.5%    Val acc: 39.2%

STGCNLite reduces the channel width from 64/128/256 → 24/48/96 while keeping
the same 7-block depth and the same 2× temporal downsampling schedule.
This gives ~339k parameters — 13% of the baseline — which is calibrated to be
large enough for 7-state × N-exercise classification but small enough to
generalize from a few hundred training sequences.

Parameter comparison
--------------------
  Baseline STGCNFull  :  2,590,196 params  (64→128→256, 9 blocks)
  STGCNLite           :    339,346 params  (24→48→96,   7 blocks)  ← this file
  Reduction           :  ×7.6 fewer parameters

Architecture
------------
  data_bn          : BatchNorm1d(C × V = 2 × 17 = 34)
  Block 1 (in=2,  out=24, stride=1)  ← no residual on first block
  Block 2 (in=24, out=24, stride=1)
  Block 3 (in=24, out=48, stride=2)  ← T: 60 → 30
  Block 4 (in=48, out=48, stride=1)
  Block 5 (in=48, out=96, stride=2)  ← T: 30 → 15
  Block 6 (in=96, out=96, stride=1)
  Block 7 (in=96, out=96, stride=1)
  Global average pool over (T=15, V=17) → (N, 96)
  fc               : Linear(96, num_classes=7)

Each STGCNBlock:
  SpatialGraphConv → BN → ReLU → TemporalConv(kt=9) → BN → Dropout → + residual → ReLU

The adjacency matrix A (17×17) is loaded from graph_A.npy and registered as a
non-trainable buffer.  A learnable edge_importance mask (same shape as A) is
multiplied element-wise so the model can re-weight but not invent connections.

Interface (identical to baseline STGCN)
-----------------------------------------
  Input  : (N, C=2, T=60, V=17, M=1)   float32
  Output : (N, num_classes)             float32   logits (no softmax)

  model = STGCNLite(A, in_channels=2, num_classes=7)
  logits = model(x)    # x shape (N, 2, 60, 17, 1)

Checkpoint format (saved by scripts/train.py)
----------------------------------------------
  {
      "epoch":       int,
      "model_state": OrderedDict,   ← from STGCNLite.state_dict()
      "val_loss":    float,
      "val_acc":     float,
      "args":        dict,          ← train CLI args (includes kt, dropout)
      "state_names": list[str],
  }
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Constants (must match build_dataset.py) ───────────────────────────────────

N_KEYPOINTS: int = 17       # V — COCO keypoints
N_COORDS:    int = 2        # C — x, y
N_STATES:    int = 7        # output classes


# ═══════════════════════════════════════════════════════════════════════════════
#  Spatial Graph Convolution
# ═══════════════════════════════════════════════════════════════════════════════

class SpatialGraphConv(nn.Module):
    """
    One spatial graph convolution step.

    Forward
    -------
    x : (N, C_in, T, V)
    1. 1×1 Conv2d lifts C_in → C_out                   → (N, C_out, T, V)
    2. einsum with A (V,V): x'_{nctw} = Σ_v x_{nctv}·A_{vw}
       → neighbour aggregation over the skeleton graph
    3. A is a frozen buffer; edge_importance re-weights connections without
       inventing new ones (learnable V×V mask).
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        A:            torch.Tensor,   # (V, V)
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.register_buffer("A", A)
        self.edge_importance = nn.Parameter(torch.ones_like(A))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)                                  # (N, C_out, T, V)
        A = self.A * self.edge_importance
        return torch.einsum("nctv,vw->nctw", x, A).contiguous()


# ═══════════════════════════════════════════════════════════════════════════════
#  ST-GCN Block (spatial + temporal + residual)
# ═══════════════════════════════════════════════════════════════════════════════

class STGCNBlock(nn.Module):
    """
    One ST-GCN block:
        SpatialGraphConv → BN → ReLU
        → TemporalConv(kt×1) → BN → Dropout
        → + residual → ReLU

    The temporal conv kernel (kt, 1) slides only along the time axis T;
    symmetric padding ((kt-1)//2, 0) preserves T at stride 1.
    stride > 1 halves T for temporal downsampling.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        A:            torch.Tensor,
        kt:           int   = 9,
        stride:       int   = 1,
        dropout:      float = 0.3,
        residual:     bool  = True,
    ) -> None:
        super().__init__()
        pad = ((kt - 1) // 2, 0)   # pad time only

        # Spatial
        self.sgc   = SpatialGraphConv(in_channels, out_channels, A)
        self.bn_s  = nn.BatchNorm2d(out_channels)

        # Temporal
        self.tcn   = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=(kt, 1),
            stride=(stride, 1),
            padding=pad,
        )
        self.bn_t  = nn.BatchNorm2d(out_channels)
        self.drop  = nn.Dropout(dropout, inplace=True)
        self.relu  = nn.ReLU(inplace=True)

        # Residual branch
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x   = self.relu(self.bn_s(self.sgc(x)))
        x   = self.bn_t(self.drop(self.tcn(x)))
        return self.relu(x + res)


# ═══════════════════════════════════════════════════════════════════════════════
#  STGCNLite — the full model
# ═══════════════════════════════════════════════════════════════════════════════

class STGCNLite(nn.Module):
    """
    Lite ST-GCN for small-dataset fitness action recognition.

    Channel plan  :  2 → 24 → 24 → 48 → 48 → 96 → 96 → 96
    Blocks        :  7  (2 temporal stride-2 downs, same schedule as full model)
    Parameters    :  ~339,346  (vs ~2,590,196 in the baseline)
    Dropout       :  0.3  (increased from baseline 0.5 for smaller model)
    Input shape   :  (N, C=2, T=60, V=17, M=1)
    Output shape  :  (N, num_classes=7)            logits
    """

    # Channel plan: (in_ch, out_ch, temporal_stride)
    # Stride-2 blocks are at positions 3 and 5 → T: 60 → 30 → 15
    _CHANNEL_PLAN = [
        (N_COORDS, 24, 1),   # Block 1 — no residual (in_ch ≠ out_ch, handled)
        (24,       24, 1),   # Block 2
        (24,       48, 2),   # Block 3 — T: 60 → 30
        (48,       48, 1),   # Block 4
        (48,       96, 2),   # Block 5 — T: 30 → 15
        (96,       96, 1),   # Block 6
        (96,       96, 1),   # Block 7
    ]

    def __init__(
        self,
        A:           torch.Tensor,
        in_channels: int   = N_COORDS,
        num_classes: int   = N_STATES,
        kt:          int   = 9,
        dropout:     float = 0.3,
    ) -> None:
        super().__init__()

        V = A.shape[0]   # 17

        # Input batch-normalisation: normalises over (C × V) channels per frame
        self.data_bn = nn.BatchNorm1d(in_channels * V)

        # Override first block's in_channels to match the constructor argument
        # so the model works even if in_channels != N_COORDS (e.g., 3-channel input)
        plan = list(self._CHANNEL_PLAN)
        plan[0] = (in_channels, plan[0][1], plan[0][2])

        self.blocks = nn.ModuleList([
            STGCNBlock(
                in_channels  = ci,
                out_channels = co,
                A            = A,
                kt           = kt,
                stride       = stride,
                dropout      = dropout,
                residual     = (i > 0),   # no residual on first block
            )
            for i, (ci, co, stride) in enumerate(plan)
        ])

        final_channels = plan[-1][1]   # 96
        self.fc = nn.Linear(final_channels, num_classes)

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, C=2, T=60, V=17, M=1)

        Returns
        -------
        logits : (N, num_classes)
        """
        N, C, T, V, M = x.size()

        # Fold M into batch: (N, M, C, T, V) → (N·M, C, T, V)
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)

        # data_bn expects (N·M, C·V, T)
        x = x.permute(0, 1, 3, 2).contiguous().view(N * M, C * V, T)
        x = self.data_bn(x)
        x = x.view(N * M, C, V, T).permute(0, 1, 3, 2).contiguous()  # (N·M, C, T, V)

        # ST-GCN backbone
        for block in self.blocks:
            x = block(x)                                 # (N·M, 96, T', V)

        # Global average pool over T and V → (N·M, 96, 1, 1) → (N·M, 96)
        x = F.adaptive_avg_pool2d(x, 1).view(N, M, -1)

        # Average over persons (M=1 here) → (N, 96)
        x = x.mean(dim=1)

        return self.fc(x)                                # (N, num_classes)

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def num_parameters(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
