"""
model/dataset.py
════════════════
PyTorch Dataset + DataLoader factory لبيانات التدريب.

كل sample:
  X       : (32, 34) float32 — normalized keypoint sequence
  y_ex    : int64            — exercise class index
  y_state : int64            — rep state (0=ready, 1=down, 2=up)
  y_form  : float32          — form quality score [0, 1]

Data augmentation (training only):
  • Gaussian noise على الـ keypoints (σ=0.02)
  • Random horizontal flip (σ=0.5) — mirrors left/right joints
  • Random temporal jitter: random start within ±2 frames
  • Random scale: multiply by Uniform(0.9, 1.1)

Fix applied (v2):
  FIX-1  Hardcoded Dataset Crash
         np.bincount(..., minlength=10) hard-coded the original exercise count.
         With 15 exercises this silently dropped the last 5 classes from the
         weight vector, causing an IndexError in get_sampler() and an incorrect
         distribution printout.
         Fix: minlength=len(Exercise) — always tracks the actual catalogue size.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from exercises import Exercise


# ── COCO-17 horizontal flip pairs ────────────────────────────────────────────
FLIP_PAIRS = [
    (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)
]

def _make_flip_indices():
    """Build index arrays for horizontal flip on flattened (34,) keypoints."""
    idx = list(range(34))
    for l, r in FLIP_PAIRS:
        idx[2*l],   idx[2*r]   = 2*r,   2*l
        idx[2*l+1], idx[2*r+1] = 2*r+1, 2*l+1
    return idx

FLIP_IDX = _make_flip_indices()

# FIX-1: derive exercise count from the single source of truth
_N_EXERCISES = len(Exercise)


# ═══════════════════════════════════════════════════════════════════════════════

class FitnessDataset(Dataset):
    """
    PyTorch Dataset for fitness exercise sequences.

    Parameters
    ----------
    npz_path  : path to train.npz / val.npz / test.npz
    augment   : apply data augmentation (training only)
    noise_std : Gaussian noise std on keypoints
    """

    def __init__(
        self,
        npz_path:  str,
        augment:   bool  = False,
        noise_std: float = 0.02,
    ) -> None:
        data = np.load(npz_path)

        self.X       = data["X"].astype(np.float32)       # (N, T, 34)
        self.y_ex    = data["y_ex"].astype(np.int64)      # (N,)
        self.y_state = data["y_state"].astype(np.int64)   # (N,)
        self.y_form  = data["y_form"].astype(np.float32)  # (N,)

        self.augment   = augment
        self.noise_std = noise_std
        self.T         = self.X.shape[1]   # sequence length

        # FIX-1: minlength must equal the number of exercise classes.
        # Using the hard-coded value 10 silently truncated the weight vector
        # once exercises 11-15 were added, causing IndexError in get_sampler()
        # and wrong per-class weights for the first 10 classes.
        ex_counts     = np.bincount(self.y_ex, minlength=_N_EXERCISES)
        ex_counts     = np.maximum(ex_counts, 1)
        self._weights = 1.0 / ex_counts[self.y_ex]

        print(f"[Dataset] Loaded {npz_path}  N={len(self)}  T={self.T}")
        self._print_distribution()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x       = self.X[idx].copy()        # (T, 34)
        y_ex    = self.y_ex[idx]
        y_state = self.y_state[idx]
        y_form  = self.y_form[idx]

        if self.augment:
            x = self._augment(x)

        return (
            torch.tensor(x,       dtype=torch.float32),
            torch.tensor(y_ex,    dtype=torch.long),
            torch.tensor(y_state, dtype=torch.long),
            torch.tensor(y_form,  dtype=torch.float32),
        )

    # ── Augmentation ─────────────────────────────────────────────────────────

    def _augment(self, x: np.ndarray) -> np.ndarray:
        """Apply random augmentations to a (T, 34) sequence."""

        # 1. Gaussian noise
        noise     = np.random.normal(0, self.noise_std, x.shape).astype(np.float32)
        zero_mask = (x == 0)
        x         = x + noise
        x[zero_mask] = 0.0

        # 2. Random horizontal flip (50% chance)
        if np.random.random() < 0.5:
            x_flip        = x[:, FLIP_IDX]
            x_flip[:, 0::2] *= -1
            x = x_flip

        # 3. Random scale (±10%)
        scale   = np.random.uniform(0.9, 1.1)
        nonzero = (x != 0)
        x[nonzero] *= scale

        # 4. Temporal jitter: randomly skip 1-2 frames at start
        jitter = np.random.randint(0, 3)
        if jitter > 0 and len(x) > jitter:
            x = np.concatenate([x[jitter:], x[-jitter:]], axis=0)

        return x.astype(np.float32)

    # ── Info ──────────────────────────────────────────────────────────────────

    def _print_distribution(self) -> None:
        # FIX-1: minlength=_N_EXERCISES so all 15 classes always appear
        counts = np.bincount(self.y_ex, minlength=_N_EXERCISES)
        print("  Exercise distribution:")
        for i, ex in enumerate(Exercise):
            bar = "█" * min(counts[i] // 10, 40)
            print(f"    {ex.value:<22s}: {counts[i]:5d}  {bar}")

    def get_sampler(self) -> WeightedRandomSampler:
        """Return a sampler that balances exercise classes during training."""
        weights = torch.tensor(self._weights, dtype=torch.float32)
        return WeightedRandomSampler(weights, num_samples=len(self), replacement=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  DataLoader factory
# ═══════════════════════════════════════════════════════════════════════════════

def make_dataloaders(
    data_dir:    str,
    batch_size:  int  = 64,
    num_workers: int  = 4,
    balanced:    bool = True,
) -> Dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders from the processed NPZ files.

    Parameters
    ----------
    data_dir    : directory containing train.npz, val.npz, test.npz
    batch_size  : batch size for all splits
    num_workers : parallel workers for data loading
    balanced    : oversample minority exercises during training

    Returns
    -------
    dict with keys "train", "val", "test"
    """
    data_dir = Path(data_dir)
    loaders  = {}

    for split in ("train", "val", "test"):
        npz_path = data_dir / f"{split}.npz"
        if not npz_path.exists():
            print(f"[WARN] {npz_path} not found, skipping {split} split")
            continue

        is_train = (split == "train")
        dataset  = FitnessDataset(str(npz_path), augment=is_train)

        sampler = dataset.get_sampler() if (is_train and balanced) else None
        shuffle = is_train and (sampler is None)

        loaders[split] = DataLoader(
            dataset,
            batch_size  = batch_size,
            shuffle     = shuffle,
            sampler     = sampler,
            num_workers = num_workers,
            pin_memory  = True,
            drop_last   = is_train,
        )

    return loaders
