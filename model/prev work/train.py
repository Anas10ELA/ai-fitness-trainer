"""
scripts/train.py
════════════════
ST-GCN Lite Model Architecture + Training Loop (single-file, COCO-17 skeleton)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ── Constants (must match build_dataset.py) ──────────────────────────────────

N_KEYPOINTS = 17        # V
N_COORDS    = 2         # C
N_STATES    = 7         # output classes
STATE_NAMES = ["neutral", "down", "up", "plank", "jump", "extended", "flexed"]


# ═══════════════════════════════════════════════════════════════════════════════
#  Model — Spatial Graph Convolution
# ═══════════════════════════════════════════════════════════════════════════════

class SpatialGraphConv(nn.Module):
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
#  Model — ST-GCN block (spatial + temporal + residual)
# ═══════════════════════════════════════════════════════════════════════════════

class STGCNBlock(nn.Module):
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
#  Model — STGCNLite (The Full Model)
# ═══════════════════════════════════════════════════════════════════════════════

class STGCNLite(nn.Module):
    _CHANNEL_PLAN = [
        (N_COORDS, 24, 1),   # Block 1
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
        self.data_bn = nn.BatchNorm1d(in_channels * V)

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
                residual     = (i > 0),   
            )
            for i, (ci, co, stride) in enumerate(plan)
        ])

        final_channels = plan[-1][1]   # 96
        self.fc = nn.Linear(final_channels, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        x = x.permute(0, 1, 3, 2).contiguous().view(N * M, C * V, T)
        x = self.data_bn(x)
        x = x.view(N * M, C, V, T).permute(0, 1, 3, 2).contiguous() 

        for block in self.blocks:
            x = block(x)                                 

        x = F.adaptive_avg_pool2d(x, 1).view(N, M, -1)
        x = x.mean(dim=1)

        return self.fc(x)                                


# ═══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_split(npz_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    data = np.load(npz_path, allow_pickle=True)
    X = torch.from_numpy(data["X"].astype(np.float32))      
    y = torch.from_numpy(data["y_state"].astype(np.int64))  
    return X, y


def make_loader(X: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=torch.cuda.is_available())


# ═══════════════════════════════════════════════════════════════════════════════
#  Train / evaluate one epoch
# ═══════════════════════════════════════════════════════════════════════════════

def run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    optimizer=None,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_correct, total_n = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for X, y in loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(X)
            loss   = criterion(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss    += loss.item() * X.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total_n       += X.size(0)

    avg_loss = total_loss / max(total_n, 1)
    accuracy = total_correct / max(total_n, 1)
    return avg_loss, accuracy


# ═══════════════════════════════════════════════════════════════════════════════
#  Main training routine
# ═══════════════════════════════════════════════════════════════════════════════

def train(args) -> None:
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}"
          f"{'  (' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else ''}")

    graph_path = data_dir / "graph_A.npy"
    if not graph_path.exists():
        raise FileNotFoundError(f"Adjacency matrix not found at {graph_path}. Run build_dataset.py first.")
    A = torch.from_numpy(np.load(graph_path).astype(np.float32))   
    print(f"[Graph]  Adjacency matrix loaded  shape={tuple(A.shape)}")

    X_train, y_train = load_split(data_dir / "train.npz")
    X_val,   y_val   = load_split(data_dir / "val.npz")
    print(f"[Data]   train X={tuple(X_train.shape)}  y={tuple(y_train.shape)}")
    print(f"[Data]   val   X={tuple(X_val.shape)}  y={tuple(y_val.shape)}")

    counts = torch.bincount(y_train, minlength=N_STATES)
    print("[Data]   train state counts: " +
          "  ".join(f"{STATE_NAMES[i]}={counts[i].item()}" for i in range(N_STATES)))

    train_loader = make_loader(X_train, y_train, args.batch_size, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   args.batch_size, shuffle=False)

    # ── استخدام الموديل الجديد هنا ──────────────────────────────────────
    model = STGCNLite(A, in_channels=N_COORDS, num_classes=N_STATES,
                  kt=args.kt, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model]  ST-GCN Lite trainable params={n_params:,}")

    class_weights = (counts.sum() / (counts.float() + 1e-6)).clamp(max=10.0).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience, min_lr=1e-6
    )

    best_val_loss = float("inf")
    best_epoch    = 0
    epochs_no_imp = 0
    ckpt_path     = out_dir / "stgcn_best.pt"

    print(f"\n{'═' * 64}")
    print(f"  {'Epoch':>5} {'TrLoss':>8} {'TrAcc':>7} {'VaLoss':>8} {'VaAcc':>7} {'LR':>9}  ")
    print(f"{'─' * 64}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, device)

        scheduler.step(va_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        improved = va_loss < best_val_loss - args.min_delta
        flag = ""
        if improved:
            best_val_loss = va_loss
            best_epoch    = epoch
            epochs_no_imp = 0
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "val_loss":     va_loss,
                "val_acc":      va_acc,
                "args":         vars(args),
                "state_names":  STATE_NAMES,
            }, ckpt_path)
            flag = " ★"
        else:
            epochs_no_imp += 1

        print(f"  {epoch:>5d} {tr_loss:>8.4f} {tr_acc:>7.3f} "
              f"{va_loss:>8.4f} {va_acc:>7.3f} {lr_now:>9.2e}"
              f"  ({time.time()-t0:4.1f}s){flag}")

        if epochs_no_imp >= args.patience:
            print(f"{'─' * 64}")
            print(f"  Early stopping at epoch {epoch} "
                  f"(no val improvement for {args.patience} epochs)")
            break

    print(f"{'═' * 64}")
    print(f"  Best val loss {best_val_loss:.4f} @ epoch {best_epoch}")
    print(f"  Checkpoint saved → {ckpt_path}")
    print(f"{'═' * 64}\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Train ST-GCN for 7-state action recognition")
    p.add_argument("--data-dir",     default="data/training_data") # تم تعديل المسار ليتوافق مع الداتا بتاعتك
    p.add_argument("--out-dir",      default="checkpoints")
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch-size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--kt",           type=int,   default=9, help="Temporal kernel size")
    p.add_argument("--dropout",      type=float, default=0.3) # تم التعديل لـ 0.3
    p.add_argument("--patience",     type=int,   default=15,
                   help="Early-stopping patience (epochs)")
    p.add_argument("--lr-patience",  type=int,   default=6,
                   help="ReduceLROnPlateau patience (epochs)")
    p.add_argument("--min-delta",    type=float, default=1e-4,
                   help="Minimum val-loss improvement to reset patience")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()