"""
scripts/train_dl_v1.py
══════════════════════
Deep Learning v1 — SkeletonNet MLP  (One Model Per Exercise)

Strategy
────────
Keep every proven element from RF v3 and replace only the classifier:
  ✔  Same 20-feature engineering  (8 angles + BBox_AR + NoseAnkle_Y + 10 deltas)
  ✔  Same VALID_STATES per-exercise noise filter
  ✔  Same video-wise train / val / test split (no data leakage)
  ✔  Same one-model-per-exercise design
  ✗  RandomForestClassifier  →  SkeletonNet (PyTorch MLP)

Why an MLP over ST-GCN here
────────────────────────────
ST-GCN trains on raw (C, T, V) tensors and must learn joint anatomy from
scratch — expensive in parameters and in the amount of data needed.
SkeletonNet receives 20 features that already encode geometry (angles) and
motion (deltas), so the network starts from a much higher-level representation
and stays small (~6 K parameters) — appropriate for ~100 videos of training.

SkeletonNet Architecture
─────────────────────────
  Input  (20)
    → Linear(64)  → BatchNorm1d(64)  → ReLU  → Dropout(0.3)
    → Linear(32)  → BatchNorm1d(32)  → ReLU
    → Linear(num_classes)
  ≈ 20×64 + 64×32 + 32×N  ≈ 4 500 – 5 000 trainable parameters per exercise

Training details
────────────────
  • Adam (lr=1e-3), optional ReduceLROnPlateau
  • CrossEntropyLoss with inverse-frequency class weights
  • Early stopping on validation loss (patience configurable)
  • Input features standardised (zero-mean / unit-variance) per exercise
    using statistics computed on the training split ONLY
  • Validation split carved out from the training videos (20 % of train vids)
    so the held-out test set is never touched during training

Feature vector layout (20 features — identical to v3)
───────────────────────────────────────────────────────
  [0]  L_Knee         Left knee angle (°)
  [1]  R_Knee         Right knee angle (°)
  [2]  L_Hip          Left hip angle (°)
  [3]  R_Hip          Right hip angle (°)
  [4]  L_Elbow        Left elbow angle (°)
  [5]  R_Elbow        Right elbow angle (°)
  [6]  L_Shoulder     Left shoulder angle (°)
  [7]  R_Shoulder     Right shoulder angle (°)
  [8]  BBox_AR        Bounding-box aspect ratio (w/h, visible joints only)
  [9]  NoseAnkle_Y    Norm. nose-to-ankle vertical distance (÷ bbox height)
  [10] Δ_L_Knee       L_Knee[t]     − L_Knee[t−5]
  [11] Δ_R_Knee       R_Knee[t]     − R_Knee[t−5]
  [12] Δ_L_Hip        L_Hip[t]      − L_Hip[t−5]
  [13] Δ_R_Hip        R_Hip[t]      − R_Hip[t−5]
  [14] Δ_L_Elbow      L_Elbow[t]    − L_Elbow[t−5]
  [15] Δ_R_Elbow      R_Elbow[t]    − R_Elbow[t−5]
  [16] Δ_L_Shoulder   L_Shoulder[t] − L_Shoulder[t−5]
  [17] Δ_R_Shoulder   R_Shoulder[t] − R_Shoulder[t−5]
  [18] Δ_BBox_AR      BBox_AR[t]    − BBox_AR[t−5]
  [19] Δ_NoseAnkle_Y  NoseAnkle_Y[t]− NoseAnkle_Y[t−5]

Outputs
───────
  checkpoints/dl_model_{exercise}.pt   — {model_state, scaler_mean, scaler_std,
                                          classes, best_val_acc, n_features}
  checkpoints/dl_features.json         — full metadata + per-exercise results
  stdout                               — epoch log, classification report,
                                         confusion matrix (mirrors v3 format)

Usage
─────
  python scripts/train_dl_v1.py
  python scripts/train_dl_v1.py \\
      --processed-dir data/processed \\
      --labeled-dir   data/labeled   \\
      --out-dir       checkpoints    \\
      --test-size     0.20           \\
      --val-size      0.20           \\
      --epochs        100            \\
      --batch-size    256            \\
      --lr            1e-3           \\
      --patience      15             \\
      --dropout       0.3            \\
      --seed          42
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ══════════════════════════════════════════════════════════════════════════════
#  Constants shared with build_dataset.py / train_rf_v3.py
# ══════════════════════════════════════════════════════════════════════════════

STATE_NAMES: List[str] = [
    "neutral",   # 0
    "down",      # 1
    "up",        # 2
    "plank",     # 3
    "jump",      # 4
    "extended",  # 5
    "flexed",    # 6
]
N_STATES: int = len(STATE_NAMES)
STATE_TO_IDX: Dict[str, int] = {n: i for i, n in enumerate(STATE_NAMES)}

# ── Per-exercise valid-state whitelists (identical to v3) ─────────────────────
# Frames whose state label is NOT in the whitelist are dropped before training.
# Exercises absent from this dict keep all frames (safe default for new ones).
VALID_STATES: Dict[str, List[str]] = {
    "situp":           ["neutral", "down", "up"],
    "pushup":          ["neutral", "down", "up", "plank"],
    "squat":           ["neutral", "down", "up"],
    "wall_sit":        ["neutral", "down"],
    "lateral_raise":   ["neutral", "down", "up", "extended"],
    "tricep_dip":      ["down", "up", "extended"],
    # Uncomment / extend as you collect more data:
    # "bicep_curl":    ["neutral", "extended", "flexed"],
    # "lunge":         ["neutral", "down"],
    # "burpee":        ["neutral", "down", "plank", "up", "jump"],
    # "plank":         ["plank"],
    # "high_knees":    ["neutral", "flexed"],
    # "jumping_jack":  ["neutral", "extended"],
    # "shoulder_press":["neutral", "down", "extended"],
    # "mountain_climber":["plank", "flexed"],
    # "pullup":        ["neutral", "up"],
}

# Pre-computed frozensets for O(1) frame-level filtering
VALID_IDX: Dict[str, FrozenSet[int]] = {
    ex: frozenset(STATE_TO_IDX[s] for s in states if s in STATE_TO_IDX)
    for ex, states in VALID_STATES.items()
}

DELTA_LAG: int = 5  # frames; ≈167 ms at 30 fps, ≈333 ms after skip=2

# ── Angle triplet definitions ─────────────────────────────────────────────────
# (feature_name, point_a_idx, vertex_idx, point_c_idx)
# Angle is measured AT the vertex using vectors: vertex→a and vertex→c
ANGLE_DEFS: List[Tuple[str, int, int, int]] = [
    ("L_Knee",     11, 13, 15),  # hip → knee ← ankle
    ("R_Knee",     12, 14, 16),
    ("L_Hip",       5, 11, 13),  # shoulder → hip ← knee
    ("R_Hip",       6, 12, 14),
    ("L_Elbow",     5,  7,  9),  # shoulder → elbow ← wrist
    ("R_Elbow",     6,  8, 10),
    ("L_Shoulder", 11,  5,  7),  # hip → shoulder ← elbow
    ("R_Shoulder", 12,  6,  8),
]
N_ANGLES: int = len(ANGLE_DEFS)       # 8
SPATIAL_NAMES: List[str] = ["BBox_AR", "NoseAnkle_Y"]
N_SPATIAL: int = len(SPATIAL_NAMES)   # 2
N_BASE: int = N_ANGLES + N_SPATIAL    # 10

BASE_NAMES:     List[str] = [d[0] for d in ANGLE_DEFS] + SPATIAL_NAMES
DELTA_NAMES:    List[str] = [f"Δ_{n}" for n in BASE_NAMES]
ALL_FEAT_NAMES: List[str] = BASE_NAMES + DELTA_NAMES
N_FEATURES:     int       = len(ALL_FEAT_NAMES)   # 20


# ══════════════════════════════════════════════════════════════════════════════
#  Feature engineering  (copied verbatim from train_rf_v3.py — DO NOT change)
# ══════════════════════════════════════════════════════════════════════════════

def _is_occluded(pt: np.ndarray) -> bool:
    """True when the keypoint is zeroed out (not detected)."""
    return float(pt[0]) == 0.0 and float(pt[1]) == 0.0


def _angle_at_vertex(a: np.ndarray, v: np.ndarray, c: np.ndarray) -> float:
    """
    Interior angle (degrees) at vertex v in the triplet (a, v, c).
    Returns 0.0 for any occluded point or a degenerate (zero-length) vector.
    """
    if _is_occluded(a) or _is_occluded(v) or _is_occluded(c):
        return 0.0
    ba = (a - v).astype(np.float64)
    bc = (c - v).astype(np.float64)
    na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if na < 1e-6 or nc < 1e-6:
        return 0.0
    cos_a = float(np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def _bbox_aspect_ratio(keypoints: np.ndarray) -> float:
    """
    Bounding-box aspect ratio (width / height) over visible joints only.
    ~0.3 when standing upright; ~5–15 when lying flat.
    Returns 0.0 when fewer than 2 joints are visible.
    """
    visible = np.array(
        [kp for kp in keypoints if not _is_occluded(kp)],
        dtype=np.float32,
    )
    if len(visible) < 2:
        return 0.0
    w = float(visible[:, 0].max() - visible[:, 0].min())
    h = float(visible[:, 1].max() - visible[:, 1].min())
    return float(w / max(h, 1e-6))


def _norm_nose_ankle_y(keypoints: np.ndarray) -> float:
    """
    Normalised nose-to-ankle vertical distance (÷ bbox height).
    ~1.0 when fully upright; ~0.0–0.1 when lying flat.
    Returns 0.0 if nose or both ankles are occluded, or bbox height ≈ 0.
    """
    nose    = keypoints[0]
    l_ankle = keypoints[15]
    r_ankle = keypoints[16]

    valid_ankles = [a for a in [l_ankle, r_ankle] if not _is_occluded(a)]
    if _is_occluded(nose) or not valid_ankles:
        return 0.0

    ankle_y = float(np.mean([a[1] for a in valid_ankles]))
    raw_y   = ankle_y - float(nose[1])

    visible = np.array(
        [kp for kp in keypoints if not _is_occluded(kp)],
        dtype=np.float32,
    )
    if len(visible) < 2:
        return 0.0
    bbox_h = float(visible[:, 1].max() - visible[:, 1].min())
    return float(raw_y / max(bbox_h, 1e-6))


def calculate_features(keypoints: np.ndarray) -> np.ndarray:
    """
    Extract the 10-element base feature vector from one (17, 2) frame.
      [0:8]  8 joint angles (degrees)
      [8]    bounding-box aspect ratio
      [9]    normalised nose-to-ankle Y distance
    Occluded joints → 0.0.  dtype: float32.
    """
    feat = np.zeros(N_BASE, dtype=np.float32)
    for i, (_, a, v, c) in enumerate(ANGLE_DEFS):
        feat[i] = _angle_at_vertex(keypoints[a], keypoints[v], keypoints[c])
    feat[N_ANGLES]     = _bbox_aspect_ratio(keypoints)
    feat[N_ANGLES + 1] = _norm_nose_ankle_y(keypoints)
    return feat


def compute_delta_features(
    base_seq: np.ndarray,   # (N, N_BASE)
    lag:      int = DELTA_LAG,
) -> np.ndarray:
    """
    Temporal velocity: base_seq[t] − base_seq[max(t−lag, 0)].
    For t < lag the reference is frame 0 (avoids a spurious all-zero prefix).
    Returns (N, N_BASE) float32.
    """
    N      = len(base_seq)
    deltas = np.zeros_like(base_seq, dtype=np.float32)
    for t in range(N):
        deltas[t] = base_seq[t] - base_seq[max(t - lag, 0)]
    return deltas


def build_feature_matrix(kp_seq: np.ndarray) -> np.ndarray:
    """
    Full 20-feature matrix for one video clip.
    kp_seq : (N, 17, 2)
    Returns (N, 20) float32 = [base_features | delta_features].
    Delta references use the ORIGINAL temporal order so that the filter
    applied afterwards does not corrupt the look-back window.
    """
    base   = np.vstack([calculate_features(kp_seq[f]) for f in range(len(kp_seq))])
    deltas = compute_delta_features(base, lag=DELTA_LAG)
    return np.hstack([base, deltas]).astype(np.float32)


def get_valid_mask(rep_states: np.ndarray, ex_name: str) -> np.ndarray:
    """
    Boolean mask (N,) — True for frames whose label is in the whitelist.
    If ex_name is absent from VALID_IDX, all frames are kept.
    """
    valid_set = VALID_IDX.get(ex_name)
    if valid_set is None:
        return np.ones(len(rep_states), dtype=bool)
    return np.isin(rep_states, sorted(valid_set))


# ══════════════════════════════════════════════════════════════════════════════
#  Data loading  (video-grouped; identical logic to train_rf_v3.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_data_by_exercise(
    processed_dir: Path,
    labeled_dir:   Path,
    verbose:       bool = True,
) -> Dict[str, Dict]:
    """
    Scan labeled_dir for every exercise sub-folder.  For each video:
      1. Load raw keypoints from processed_dir.
      2. Build the 20-feature matrix on the FULL temporal sequence.
      3. Apply the per-exercise VALID_STATES whitelist.
      4. Accumulate into per-exercise buffers.

    Returns
    ───────
    dict keyed by exercise name:
        "X"         : np.ndarray (N, 20)  float32
        "y"         : np.ndarray (N,)     int32
        "video_ids" : list[str]  length N  (one entry per frame)
    """
    data:          Dict[str, Dict[str, list]] = {}
    total_vids    = 0
    total_dropped = 0
    skipped       = 0

    for ex_dir in sorted(labeled_dir.iterdir()):
        if not ex_dir.is_dir():
            continue
        ex_name = ex_dir.name

        for label_npz in sorted(ex_dir.glob("*.npz")):
            video_id = label_npz.stem
            vid_key  = f"{ex_name}/{video_id}"
            proc_npz = processed_dir / ex_name / f"{video_id}.npz"

            if not proc_npz.exists():
                if verbose:
                    print(f"  [WARN] missing processed: {vid_key}")
                skipped += 1
                continue

            try:
                proc  = np.load(proc_npz,  allow_pickle=True)
                label = np.load(label_npz, allow_pickle=True)

                kp         = proc["keypoints"].astype(np.float32)   # (N, 17, 2)
                rep_states = label["rep_states"].astype(np.int32)   # (N,)
                if ex_name == "situp":
                    rep_states[rep_states == 6] = 2  # دمج flexed مع up
                    rep_states[rep_states == 5] = 0  # دمج extended مع neutral
                N          = min(len(kp), len(rep_states))
                kp         = kp[:N]
                rep_states = rep_states[:N]

                # Clip any accidentally out-of-range state indices
                bad = (rep_states < 0) | (rep_states >= N_STATES)
                if bad.any():
                    if verbose:
                        print(f"  [WARN] {vid_key}: {bad.sum()} "
                              f"out-of-range labels → clipped to 0")
                    rep_states[bad] = 0

                # Build features on the full temporal sequence FIRST so that
                # delta references at frame t look back into the original clip.
                X_full = build_feature_matrix(kp)   # (N, 20)

                # Apply per-exercise whitelist AFTER feature computation
                mask      = get_valid_mask(rep_states, ex_name)
                n_dropped = int((~mask).sum())
                total_dropped += n_dropped

                X_vid = X_full[mask]
                y_vid = rep_states[mask]

                if len(y_vid) == 0:
                    if verbose:
                        print(f"  [SKIP] {vid_key}: 0 frames after filtering")
                    continue

                if ex_name not in data:
                    data[ex_name] = {"X": [], "y": [], "video_ids": []}
                data[ex_name]["X"].append(X_vid)
                data[ex_name]["y"].append(y_vid)
                data[ex_name]["video_ids"].extend([vid_key] * len(y_vid))

                total_vids += 1
                if verbose:
                    pct      = len(y_vid) / N * 100
                    drop_str = f"  ← dropped {n_dropped} noisy" if n_dropped else ""
                    print(f"  [load] {vid_key}: {len(y_vid)}/{N} frames "
                          f"({pct:.0f}% kept){drop_str}  "
                          f"states={np.unique(y_vid).tolist()}")

            except Exception as exc:
                if verbose:
                    print(f"  [WARN] Failed {vid_key}: {exc}")
                skipped += 1

    if not data:
        raise RuntimeError(
            "No data loaded. "
            "Check --processed-dir and --labeled-dir paths and "
            "confirm data/labeled/{exercise}/*.npz files exist."
        )

    # Stack per-video lists into single arrays
    result: Dict[str, Dict] = {}
    for ex_name, parts in data.items():
        result[ex_name] = {
            "X":         np.vstack(parts["X"]).astype(np.float32),
            "y":         np.concatenate(parts["y"]).astype(np.int32),
            "video_ids": parts["video_ids"],
        }

    if verbose:
        print(f"\n  Loaded {total_vids} videos across {len(result)} exercises  "
              f"|  dropped {total_dropped} noisy frames  |  skipped {skipped}")
        for ex, d in sorted(result.items()):
            counts    = np.bincount(d["y"], minlength=N_STATES)
            n_vids    = len(set(d["video_ids"]))
            state_str = "  ".join(
                f"{STATE_NAMES[i]}={counts[i]}"
                for i in range(N_STATES)
                if counts[i] > 0
            )
            print(f"    {ex:<22}: {d['X'].shape[0]:5d} frames "
                  f"({n_vids} vids) — {state_str}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Video-wise train / val / test split
# ══════════════════════════════════════════════════════════════════════════════

def video_wise_three_way_split(
    X:         np.ndarray,
    y:         np.ndarray,
    video_ids: List[str],
    test_size: float = 0.20,
    val_size:  float = 0.20,
    seed:      int   = 42,
) -> Tuple[np.ndarray, np.ndarray,
           np.ndarray, np.ndarray,
           np.ndarray, np.ndarray]:
    """
    Split by video ID so no frame from the same video crosses splits.
    val_size is expressed as a fraction of the FULL video list (not train only).

    Guarantees at least 1 video in test, at least 1 in val.
    Order: (X_tr, X_val, X_te, y_tr, y_val, y_te)
    """
    unique_vids = sorted(set(video_ids))
    rng         = np.random.default_rng(seed)
    rng.shuffle(unique_vids)

    n_total = len(unique_vids)
    n_test  = max(1, int(n_total * test_size))
    n_val   = max(1, int(n_total * val_size))
    # Guard: need at least 1 training video
    n_val   = min(n_val, n_total - n_test - 1)
    if n_val < 1:
        n_val = 1

    test_set  = set(unique_vids[-n_test:])
    val_set   = set(unique_vids[-(n_test + n_val): -n_test])

    vid_arr    = np.array(video_ids)
    train_mask = ~np.isin(vid_arr, list(test_set | val_set))
    val_mask   =  np.isin(vid_arr, list(val_set))
    test_mask  =  np.isin(vid_arr, list(test_set))

    return (
        X[train_mask], X[val_mask],  X[test_mask],
        y[train_mask], y[val_mask],  y[test_mask],
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Feature standardisation  (fit on train, apply to val + test)
# ══════════════════════════════════════════════════════════════════════════════

class StandardScaler:
    """
    Thin NumPy-based scaler: zero-mean, unit-variance per feature.
    Handles constant features (std ≈ 0) safely by leaving them unchanged.
    """

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_:  Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0)
        # Avoid division by zero for constant features (e.g. all-zero joint)
        self.std_  = np.where(self.std_ < 1e-8, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None, "Call fit() before transform()"
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ══════════════════════════════════════════════════════════════════════════════
#  PyTorch model
# ══════════════════════════════════════════════════════════════════════════════

class SkeletonNet(nn.Module):
    """
    Lightweight MLP for per-frame exercise state classification.

    Input  : (B, 20)  standardised feature vector
    Output : (B, num_classes)  raw logits

    Architecture
    ────────────
      Linear(20→64)  → BatchNorm1d(64)  → ReLU  → Dropout(p)
      Linear(64→32)  → BatchNorm1d(32)  → ReLU
      Linear(32→num_classes)

    Parameter count (num_classes = 7):
      20×64 + 64 + 64×2  (BN γ,β)  +  64×32 + 32 + 32×2  +  32×7 + 7  ≈ 4 800
    The model intentionally stays small to avoid overfitting on ~100 videos.
    """

    def __init__(self, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # ── Block 1 ───────────────────────────────────────────────────────
            nn.Linear(N_FEATURES, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            # ── Block 2 ───────────────────────────────────────────────────────
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            # ── Output head ───────────────────────────────────────────────────
            nn.Linear(32, num_classes),
        )
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
#  DataLoader factory
# ══════════════════════════════════════════════════════════════════════════════

def make_loader(
    X:          np.ndarray,
    y:          np.ndarray,
    batch_size: int,
    shuffle:    bool,
) -> DataLoader:
    """Wrap NumPy arrays in a TensorDataset and return a DataLoader."""
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y.astype(np.int64))
    ds  = TensorDataset(X_t, y_t)
    return DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = shuffle,
        drop_last   = False,
        pin_memory  = torch.cuda.is_available(),
        num_workers = 0,   # avoids multiprocessing issues on all platforms
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Class-weight computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_class_weights(
    y:          np.ndarray,
    classes:    List[int],
    device:     torch.device,
) -> torch.Tensor:
    """
    Inverse-frequency class weights clipped at 10× to avoid a single rare class
    dominating the gradient entirely.

    weights[i]  ∝  1 / count(class_i)   (normalised so mean weight ≈ 1)
    """
    counts = np.bincount(y, minlength=max(classes) + 1).astype(np.float32)
    # Only consider classes actually present in y
    weights = np.zeros(len(classes), dtype=np.float32)
    for local_i, cls_idx in enumerate(classes):
        weights[local_i] = 1.0 / max(counts[cls_idx], 1.0)

    # Normalise so the mean weight is 1.0 (keeps LR scale stable)
    weights = weights / weights.mean()
    # Cap extreme weights to prevent one very rare class from taking over
    weights = np.clip(weights, 0.0, 10.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ══════════════════════════════════════════════════════════════════════════════
#  Label remapping helpers
# ══════════════════════════════════════════════════════════════════════════════

def remap_labels(
    y: np.ndarray,
    classes: List[int],
) -> np.ndarray:
    """
    Map global state indices (0-6) to local indices (0..len(classes)-1).
    Only classes actually present in the exercise data are kept.
    """
    mapping = {c: i for i, c in enumerate(classes)}
    return np.array([mapping[v] for v in y], dtype=np.int64)


# ══════════════════════════════════════════════════════════════════════════════
#  Single-exercise training loop
# ══════════════════════════════════════════════════════════════════════════════

def train_one_exercise(
    ex_name:    str,
    X_tr:       np.ndarray,   # already standardised
    y_tr_local: np.ndarray,   # locally remapped labels
    X_val:      np.ndarray,
    y_val_local:np.ndarray,
    classes:    List[int],    # global state indices present in this exercise
    device:     torch.device,
    epochs:     int   = 100,
    batch_size: int   = 256,
    lr:         float = 1e-3,
    dropout:    float = 0.3,
    patience:   int   = 15,
    min_delta:  float = 1e-4,
    seed:       int   = 42,
) -> Tuple[SkeletonNet, Dict]:
    """
    Train SkeletonNet for one exercise.

    Returns
    ───────
    (trained_model, training_history_dict)
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    num_classes  = len(classes)
    model        = SkeletonNet(num_classes=num_classes, dropout=dropout).to(device)
    class_weights = compute_class_weights(y_tr_local, list(range(num_classes)), device)
    criterion    = nn.CrossEntropyLoss(weight=class_weights)
    optimizer    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler    = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(patience // 3, 3), min_lr=1e-6
    )

    train_loader = make_loader(X_tr,  y_tr_local,  batch_size, shuffle=True)
    val_loader   = make_loader(X_val, y_val_local, batch_size, shuffle=False)

    best_val_loss = math.inf
    best_val_acc  = 0.0
    best_state    = None
    no_improve    = 0
    history: Dict = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # ── Column header ─────────────────────────────────────────────────────────
    col = f"  {'Ep':>4}  {'TrLoss':>8}  {'TrAcc':>7}  {'VaLoss':>8}  {'VaAcc':>7}  {'LR':>8}"
    print(col)
    print("  " + "─" * (len(col) - 2))

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        tr_loss = tr_correct = tr_total = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits  = model(Xb)
            loss    = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss    += loss.item() * Xb.size(0)
            tr_correct += (logits.argmax(1) == yb).sum().item()
            tr_total   += Xb.size(0)

        tr_loss_avg = tr_loss / max(tr_total, 1)
        tr_acc      = tr_correct / max(tr_total, 1)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        va_loss = va_correct = va_total = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits  = model(Xb)
                loss    = criterion(logits, yb)
                va_loss    += loss.item() * Xb.size(0)
                va_correct += (logits.argmax(1) == yb).sum().item()
                va_total   += Xb.size(0)

        va_loss_avg = va_loss / max(va_total, 1)
        va_acc      = va_correct / max(va_total, 1)

        scheduler.step(va_loss_avg)
        lr_now = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tr_loss_avg)
        history["val_loss"].append(va_loss_avg)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        # ── Early stopping ────────────────────────────────────────────────────
        improved = va_loss_avg < best_val_loss - min_delta
        flag = ""
        if improved:
            best_val_loss = va_loss_avg
            best_val_acc  = va_acc
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
            flag = " ★"
        else:
            no_improve += 1

        # Print every epoch (compact single line)
        print(f"  {epoch:>4d}  {tr_loss_avg:>8.4f}  {tr_acc:>7.3f}  "
              f"{va_loss_avg:>8.4f}  {va_acc:>7.3f}  {lr_now:>8.2e}{flag}")

        if no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no val improvement for {patience} epochs)")
            break

    # Restore the best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    history["best_val_loss"] = float(best_val_loss)
    history["best_val_acc"]  = float(best_val_acc)
    history["epochs_run"]    = epoch
    return model, history


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation  (mirrors v3 output format)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_one_exercise(
    ex_name:     str,
    model:       SkeletonNet,
    X_test:      np.ndarray,   # standardised
    y_test_local:np.ndarray,   # locally remapped labels
    y_test_global:np.ndarray,  # original global indices (for display)
    classes:     List[int],
    device:      torch.device,
    batch_size:  int = 512,
) -> float:
    """
    Print classification report + confusion matrix, return test accuracy.
    Output format mirrors train_rf_v3.py so results are easy to compare.
    """
    model.eval()
    loader = make_loader(X_test, y_test_local, batch_size, shuffle=False)

    all_pred_local: List[int] = []
    for Xb, _ in loader:
        logits = model(Xb.to(device))
        all_pred_local.extend(logits.argmax(1).cpu().numpy().tolist())

    y_pred_local = np.array(all_pred_local, dtype=np.int64)

    # Map back to global state indices for human-readable labels
    local_to_global = {i: c for i, c in enumerate(classes)}
    y_pred_global   = np.array([local_to_global[p] for p in y_pred_local])

    present_global  = sorted(set(y_test_global.tolist()))
    labels          = [STATE_NAMES[i] for i in present_global]

    print(f"\n{'═'*64}")
    print(f"  EXERCISE: {ex_name.upper()}")
    print(f"  Test frames: {len(y_test_global):,}  |  States: {labels}")
    print(f"{'─'*64}")
    print(classification_report(
        y_test_global, y_pred_global,
        labels       = present_global,
        target_names = labels,
        zero_division= 0,
    ))

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_test_global, y_pred_global, labels=present_global)
    print("  Confusion matrix:")
    print(f"  {'':>12}", end="")
    for lab in labels:
        print(f"  {lab[:6]:>6}", end="")
    print()
    for i, row_lbl in enumerate(labels):
        print(f"  {row_lbl:>12}", end="")
        for j, val in enumerate(cm[i]):
            mark = "■" if i == j and val > 0 else " "
            print(f"  {val:>5}{mark}", end="")
        print()

    acc = float((y_pred_global == y_test_global).mean())
    print(f"\n  Test accuracy : {acc:.4f}  ({acc * 100:.1f}%)")

    # ── Per-class breakdown ───────────────────────────────────────────────────
    print("\n  Per-class accuracy:")
    for g_idx in present_global:
        mask   = y_test_global == g_idx
        if mask.sum() == 0:
            continue
        cls_acc = float((y_pred_global[mask] == y_test_global[mask]).mean())
        bar     = "█" * int(cls_acc * 30)
        print(f"    {STATE_NAMES[g_idx]:<12}: {cls_acc:.3f}  {bar}")

    return acc


# ══════════════════════════════════════════════════════════════════════════════
#  Export
# ══════════════════════════════════════════════════════════════════════════════

def save_model(
    ex_name:      str,
    model:        SkeletonNet,
    scaler:       StandardScaler,
    classes:      List[int],
    history:      Dict,
    out_dir:      Path,
) -> Path:
    """
    Save a single exercise model as a .pt file containing everything needed
    for inference without re-importing the scaler or class list separately.

    Saved dict keys
    ───────────────
    model_state     : OrderedDict — nn.Module state_dict()
    scaler_mean     : (20,) float32 — training-split feature means
    scaler_std      : (20,) float32 — training-split feature stds
    classes         : List[int]    — global state indices this model handles
    n_features      : int          — always 20
    num_classes     : int          — len(classes)
    best_val_acc    : float
    epochs_run      : int
    feature_names   : List[str]    — ALL_FEAT_NAMES (for introspection)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"dl_model_{ex_name}.pt"

    torch.save(
        {
            "model_state":  model.state_dict(),
            "scaler_mean":  scaler.mean_.astype(np.float32),
            "scaler_std":   scaler.std_.astype(np.float32),
            "classes":      classes,
            "n_features":   N_FEATURES,
            "num_classes":  len(classes),
            "best_val_acc": history["best_val_acc"],
            "epochs_run":   history["epochs_run"],
            "feature_names": ALL_FEAT_NAMES,
        },
        path,
    )
    return path


def save_metadata(
    per_ex_meta: Dict[str, Dict],
    out_dir:     Path,
) -> None:
    """
    Write dl_features.json — a self-contained inference recipe so any
    downstream code can reconstruct features without importing this script.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "dl_features.json"

    meta = {
        "version":     "dl_v1",
        "n_features":  N_FEATURES,
        "n_base":      N_BASE,
        "delta_lag":   DELTA_LAG,
        "feature_names":  ALL_FEAT_NAMES,
        "base_features":  BASE_NAMES,
        "delta_features": DELTA_NAMES,
        "angle_defs": [
            {"name": name, "point_a": a, "vertex": v, "point_c": c}
            for name, a, v, c in ANGLE_DEFS
        ],
        "spatial_features": {
            "BBox_AR": {
                "formula": "(max_x - min_x) / max(max_y - min_y, 1e-6)",
                "note":    "Visible joints only. ~0.3 upright, ~5-15 lying flat.",
            },
            "NoseAnkle_Y": {
                "formula": "(mean_ankle_y - nose_y) / max(bbox_height, 1e-6)",
                "note":    "~1.0 upright, ~0.0 lying flat. 0.0 if occluded.",
                "joints":  {"nose": 0, "left_ankle": 15, "right_ankle": 16},
            },
        },
        "valid_states": VALID_STATES,
        "state_names":  STATE_NAMES,
        "n_states":     N_STATES,
        "model_architecture": {
            "type":    "MLP / SkeletonNet",
            "layers":  [
                "Linear(20→64)", "BatchNorm1d(64)", "ReLU", "Dropout(0.3)",
                "Linear(64→32)", "BatchNorm1d(32)", "ReLU",
                "Linear(32→num_classes)",
            ],
        },
        "exercises":   per_ex_meta,
        "inference_note": (
            "1. Keep a deque of DELTA_LAG+1 raw keypoint frames (17,2).  "
            "2. feat_base  = calculate_features(kp[t]).  "
            "3. feat_delta = feat_base - calculate_features(kp[max(t-DELTA_LAG,0)]).  "
            "4. feat_vec   = np.concatenate([feat_base, feat_delta])  # (20,).  "
            "5. feat_norm  = (feat_vec - scaler_mean) / scaler_std.  "
            "6. logits     = model(torch.tensor(feat_norm).unsqueeze(0)).  "
            "7. state_local= logits.argmax(dim=1).item().  "
            "8. state_name = STATE_NAMES[classes[state_local]]."
        ),
    }

    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\n[Save]  Metadata → {meta_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── CLI ───────────────────────────────────────────────────────────────────
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "DL v1 — SkeletonNet MLP\n"
            "Same 20-feature engineering as RF v3, one PyTorch MLP per exercise."
        ),
    )
    p.add_argument("--processed-dir", default="data/processed",
                   help="Directory of per-video processed .npz files")
    p.add_argument("--labeled-dir",   default="data/labeled",
                   help="Directory of per-video labeled .npz files")
    p.add_argument("--out-dir",       default="checkpoints",
                   help="Output directory for model .pt files and metadata JSON")
    p.add_argument("--test-size",  type=float, default=0.20,
                   help="Fraction of videos held out as test  (default: 0.20)")
    p.add_argument("--val-size",   type=float, default=0.20,
                   help="Fraction of videos used for val / early stopping  (default: 0.20)")
    p.add_argument("--epochs",     type=int,   default=100,
                   help="Maximum training epochs per exercise  (default: 100)")
    p.add_argument("--batch-size", type=int,   default=256,
                   help="Mini-batch size  (default: 256)")
    p.add_argument("--lr",         type=float, default=1e-3,
                   help="Adam learning rate  (default: 1e-3)")
    p.add_argument("--dropout",    type=float, default=0.3,
                   help="Dropout probability in SkeletonNet  (default: 0.3)")
    p.add_argument("--patience",   type=int,   default=15,
                   help="Early-stopping patience in epochs  (default: 15)")
    p.add_argument("--seed",       type=int,   default=42,
                   help="Random seed for reproducibility  (default: 42)")
    p.add_argument("--device",     default="auto",
                   help="PyTorch device: auto | cpu | cuda | mps  (default: auto)")
    p.add_argument("--quiet",      action="store_true",
                   help="Suppress per-video loading log")
    args = p.parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # ── Banner ────────────────────────────────────────────────────────────────
    print("\n" + "═" * 64)
    print("  DL v1 — SkeletonNet  (Per-Exercise MLP)")
    print("═" * 64)
    print(f"  Architecture   : Input(20)→64→32→num_classes  (~4 800 params)")
    print(f"  Features/frame : {N_FEATURES}  "
          f"({N_BASE} base: 8 angles + 2 spatial + {N_BASE} deltas@t−{DELTA_LAG})")
    print(f"  Exercises with whitelist filter: "
          f"{', '.join(sorted(VALID_STATES.keys()))}")
    print(f"  Device         : {device}")
    print(f"  processed_dir  : {args.processed_dir}")
    print(f"  labeled_dir    : {args.labeled_dir}")
    print(f"  test_size      : {args.test_size:.0%}  (video-wise per exercise)")
    print(f"  val_size       : {args.val_size:.0%}   (video-wise per exercise)")
    print(f"  epochs         : {args.epochs}   patience={args.patience}")
    print(f"  batch_size     : {args.batch_size}   lr={args.lr}   "
          f"dropout={args.dropout}")
    print("═" * 64 + "\n")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("[Step 1]  Loading and filtering data …\n")
    exercise_data = load_data_by_exercise(
        Path(args.processed_dir),
        Path(args.labeled_dir),
        verbose=not args.quiet,
    )

    # ── 2. Per-exercise training loop ─────────────────────────────────────────
    print("\n[Step 2]  Training one SkeletonNet per exercise …")
    per_ex_meta:   Dict[str, Dict] = {}
    summary_rows:  List[Tuple]     = []

    for ex_name in sorted(exercise_data.keys()):
        d         = exercise_data[ex_name]
        X, y, vids = d["X"], d["y"], d["video_ids"]
        n_videos  = len(set(vids))
        classes   = sorted(set(y.tolist()))   # global state indices for this exercise

        print(f"\n{'─'*64}")
        print(f"  [{ex_name.upper()}]  "
              f"{X.shape[0]:,} frames  |  {n_videos} videos  |  "
              f"classes: {[STATE_NAMES[c] for c in classes]}")

        # Need at least 3 videos: 1 test + 1 val + 1 train
        if n_videos < 3:
            print(f"  [SKIP]  Only {n_videos} video(s) — need ≥3 "
                  f"(1 test + 1 val + 1 train).")
            continue

        # ── Split ─────────────────────────────────────────────────────────────
        X_tr, X_val, X_te, y_tr, y_val, y_te = video_wise_three_way_split(
            X, y, vids,
            test_size = args.test_size,
            val_size  = args.val_size,
            seed      = args.seed,
        )

        if len(X_te) == 0 or len(X_val) == 0:
            print("  [SKIP]  Empty val or test set after split.")
            continue

        print(f"  Train: {len(X_tr):,} frames  |  "
              f"Val: {len(X_val):,} frames  |  "
              f"Test: {len(X_te):,} frames")

        # ── Standardise ───────────────────────────────────────────────────────
        # Fit the scaler on TRAINING frames only — this is critical.
        # Val and test receive the same transform so they stay unseen.
        scaler  = StandardScaler()
        X_tr_s  = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        X_te_s  = scaler.transform(X_te)

        # ── Remap labels to contiguous local indices ───────────────────────────
        # CrossEntropyLoss requires targets in [0, num_classes).
        # E.g. if an exercise only has states {0, 1, 2} we get local {0,1,2}.
        # If it only has {1, 5} we get local {0, 1}.
        y_tr_l  = remap_labels(y_tr,  classes).astype(np.int64)
        y_val_l = remap_labels(y_val, classes).astype(np.int64)
        y_te_l  = remap_labels(y_te,  classes).astype(np.int64)

        # ── Train ─────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        model, history = train_one_exercise(
            ex_name     = ex_name,
            X_tr        = X_tr_s,
            y_tr_local  = y_tr_l,
            X_val       = X_val_s,
            y_val_local = y_val_l,
            classes     = classes,
            device      = device,
            epochs      = args.epochs,
            batch_size  = args.batch_size,
            lr          = args.lr,
            dropout     = args.dropout,
            patience    = args.patience,
            seed        = args.seed,
        )
        elapsed = time.perf_counter() - t0
        print(f"\n  Training time: {elapsed:.1f}s  |  "
              f"Best val acc: {history['best_val_acc']:.3f}  |  "
              f"Params: {model.count_parameters():,}")

        # ── Evaluate on held-out test set ─────────────────────────────────────
        test_acc = evaluate_one_exercise(
            ex_name       = ex_name,
            model         = model,
            X_test        = X_te_s,
            y_test_local  = y_te_l,
            y_test_global = y_te,
            classes       = classes,
            device        = device,
            batch_size    = args.batch_size,
        )

        # ── Save model ────────────────────────────────────────────────────────
        ckpt_path = save_model(
            ex_name = ex_name,
            model   = model,
            scaler  = scaler,
            classes = classes,
            history = history,
            out_dir = Path(args.out_dir),
        )
        print(f"  Model saved → {ckpt_path}")

        per_ex_meta[ex_name] = {
            "n_train":        int(len(X_tr)),
            "n_val":          int(len(X_val)),
            "n_test":         int(len(X_te)),
            "n_videos":       n_videos,
            "states_seen":    [STATE_NAMES[c] for c in classes],
            "valid_states":   VALID_STATES.get(ex_name, "all"),
            "best_val_acc":   round(history["best_val_acc"], 4),
            "test_accuracy":  round(test_acc, 4),
            "epochs_run":     history["epochs_run"],
            "checkpoint":     str(Path(args.out_dir) / f"dl_model_{ex_name}.pt"),
        }
        summary_rows.append((ex_name, len(X_tr), len(X_te), test_acc))

    # ── 3. Summary table ──────────────────────────────────────────────────────
    print(f"\n{'═'*64}")
    print("  FINAL SUMMARY")
    print(f"  {'Exercise':<22}  {'Train':>6}  {'Test':>6}  {'Accuracy':>9}")
    print(f"  {'─'*22}  {'─'*6}  {'─'*6}  {'─'*9}")
    total_test = total_correct = 0
    for ex_name, n_tr, n_te, acc in summary_rows:
        bar = "█" * int(acc * 20)
        print(f"  {ex_name:<22}  {n_tr:>6,}  {n_te:>6,}  "
              f"{acc*100:>7.1f}%  {bar}")
        total_test    += n_te
        total_correct += int(round(acc * n_te))

    if total_test > 0:
        overall = total_correct / total_test
        bar_o   = "█" * int(overall * 20)
        print(f"  {'─'*22}  {'─'*6}  {'─'*6}  {'─'*9}")
        print(f"  {'OVERALL (frame-weighted)':<22}  {'':>6}  "
              f"{total_test:>6,}  {overall*100:>7.1f}%  {bar_o}")
    print(f"{'═'*64}")

    if not per_ex_meta:
        print("\n[ERROR]  No models trained. "
              "Check --processed-dir / --labeled-dir paths.")
        sys.exit(1)

    # ── 4. Save shared metadata ───────────────────────────────────────────────
    save_metadata(per_ex_meta, Path(args.out_dir))
    print("\n[DL v1]  Done.\n")


if __name__ == "__main__":
    main()
