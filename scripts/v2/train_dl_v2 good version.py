"""
scripts/train_dl_v2.py
═══════════════════════
Deep Learning v2 — Per-Exercise Micro-Models with Quality-Weighted Training

What's new vs v1
────────────────
  • Per-exercise ISOLATED state vocabulary (sq_standing, sq_bottom, …) loaded
    directly from each labeled .npz — no more shared 7-state global vocab.
  • Quality-as-sample-weight: each frame's CrossEntropy loss is scaled by the
    human quality rating of the rep it belongs to. NO regression head.
  • On-the-fly augmentation: keypoint jitter + frame-dropout deltas.
  • Same proven 20-feature engineering and checkpoint format as v1, so the
    existing dl_engine.py loads these checkpoints unchanged.

Architecture (single head — exactly as specified)
───────────────────────────────────────────────────
  Input(20)
    → Linear(64) → BatchNorm1d → ReLU → Dropout(0.3)
    → Linear(32) → BatchNorm1d → ReLU
    → Linear(num_classes)

Quality-as-sample-weight mechanism
───────────────────────────────────
  Frame inside rep i        → weight = (rep_qualities[i] + 1) / 10.0
                              (quality 9 → 1.0,  quality 0 → 0.1)
  Frame outside any rep      → weight = 0.5  (default; mostly neutral states)

  Final per-sample loss weight = class_weight[label] × sample_weight[frame]
    • class_weight handles imbalance (long standing vs brief bottom)
    • sample_weight handles form quality (learn more from clean reps)

Data layout
───────────
  data/processed/<exercise>/<video_id>.npz   →  keypoints (N,17,2), scores (N,17)
  data/labeled_v2/<exercise>/<video_id>.npz  →  frame_states (N,), state_names,
                                                rep_starts, rep_ends, rep_qualities

  (Keypoints are read from processed/; if a labeled file also contains a
   "keypoints" array, that is used instead — both layouts are supported.)

Output
──────
  checkpoints_v2/dl_model_<exercise>.pt   — model_state, scaler_mean, scaler_std,
                                            classes, state_names, n_features=20,
                                            num_classes, best_val_acc, …

Usage
─────
  python scripts/train_dl_v2.py --exercise squat
  python scripts/train_dl_v2.py                      # all exercises found
  python scripts/train_dl_v2.py --exercise squat --epochs 150 --no-augment
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ══════════════════════════════════════════════════════════════════════════════
#  Feature engineering  (copied verbatim from train_dl_v1.py — DO NOT change)
# ══════════════════════════════════════════════════════════════════════════════

DELTA_LAG: int = 5   # frames; MUST match dl_engine.py inference constant

ANGLE_DEFS: List[Tuple[str, int, int, int]] = [
    ("L_Knee",     11, 13, 15),   # hip → knee ← ankle
    ("R_Knee",     12, 14, 16),
    ("L_Hip",       5, 11, 13),   # shoulder → hip ← knee
    ("R_Hip",       6, 12, 14),
    ("L_Elbow",     5,  7,  9),   # shoulder → elbow ← wrist
    ("R_Elbow",     6,  8, 10),
    ("L_Shoulder", 11,  5,  7),   # hip → shoulder ← elbow
    ("R_Shoulder", 12,  6,  8),
]
N_ANGLES:   int = len(ANGLE_DEFS)       # 8
SPATIAL_NAMES: List[str] = ["BBox_AR", "NoseAnkle_Y"]
N_BASE:     int = N_ANGLES + len(SPATIAL_NAMES)   # 10
BASE_NAMES:     List[str] = [d[0] for d in ANGLE_DEFS] + SPATIAL_NAMES
DELTA_NAMES:    List[str] = [f"Δ_{n}" for n in BASE_NAMES]
ALL_FEAT_NAMES: List[str] = BASE_NAMES + DELTA_NAMES
N_FEATURES:     int       = len(ALL_FEAT_NAMES)   # 20


def _is_occluded(pt: np.ndarray) -> bool:
    """True when the keypoint is zeroed out (not detected)."""
    return float(pt[0]) == 0.0 and float(pt[1]) == 0.0


def _angle_at_vertex(a: np.ndarray, v: np.ndarray, c: np.ndarray) -> float:
    """Interior angle (degrees) at vertex v in triplet (a, v, c)."""
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
    """Bounding-box aspect ratio (w/h) over visible joints only."""
    visible = np.array(
        [kp for kp in keypoints if not _is_occluded(kp)], dtype=np.float32,
    )
    if len(visible) < 2:
        return 0.0
    w = float(visible[:, 0].max() - visible[:, 0].min())
    h = float(visible[:, 1].max() - visible[:, 1].min())
    return float(w / max(h, 1e-6))


def _norm_nose_ankle_y(keypoints: np.ndarray) -> float:
    """Normalised nose-to-ankle vertical distance (÷ bbox height)."""
    nose, l_ankle, r_ankle = keypoints[0], keypoints[15], keypoints[16]
    valid_ankles = [a for a in [l_ankle, r_ankle] if not _is_occluded(a)]
    if _is_occluded(nose) or not valid_ankles:
        return 0.0
    ankle_y = float(np.mean([a[1] for a in valid_ankles]))
    raw_y   = ankle_y - float(nose[1])
    visible = np.array(
        [kp for kp in keypoints if not _is_occluded(kp)], dtype=np.float32,
    )
    if len(visible) < 2:
        return 0.0
    bbox_h = float(visible[:, 1].max() - visible[:, 1].min())
    return float(raw_y / max(bbox_h, 1e-6))


def calculate_features(keypoints: np.ndarray) -> np.ndarray:
    """Extract the 10-element base feature vector from one (17, 2) frame."""
    feat = np.zeros(N_BASE, dtype=np.float32)
    for i, (_, a, v, c) in enumerate(ANGLE_DEFS):
        feat[i] = _angle_at_vertex(keypoints[a], keypoints[v], keypoints[c])
    feat[N_ANGLES]     = _bbox_aspect_ratio(keypoints)
    feat[N_ANGLES + 1] = _norm_nose_ankle_y(keypoints)
    return feat


def build_base_matrix(kp_seq: np.ndarray) -> np.ndarray:
    """(N,17,2) → (N,10) base feature matrix (no deltas)."""
    return np.vstack(
        [calculate_features(kp_seq[f]) for f in range(len(kp_seq))]
    ).astype(np.float32)


def deltas_from_base(
    base_seq: np.ndarray,        # (N, 10)
    lag: int = DELTA_LAG,
    dropout_refs: Optional[np.ndarray] = None,  # (N,) custom reference indices
) -> np.ndarray:
    """
    Temporal velocity deltas: base[t] − base[ref(t)].

    Normal mode    : ref(t) = max(t − lag, 0)            (matches inference)
    Frame-dropout  : ref(t) = dropout_refs[t]            (augmentation only)
    """
    N = len(base_seq)
    deltas = np.zeros_like(base_seq, dtype=np.float32)
    if dropout_refs is None:
        for t in range(N):
            deltas[t] = base_seq[t] - base_seq[max(t - lag, 0)]
    else:
        for t in range(N):
            deltas[t] = base_seq[t] - base_seq[int(dropout_refs[t])]
    return deltas


def full_features_from_base(base_seq: np.ndarray) -> np.ndarray:
    """(N,10) base → (N,20) full vector using the standard lag (inference path)."""
    d = deltas_from_base(base_seq, lag=DELTA_LAG)
    return np.hstack([base_seq, d]).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  Per-frame sample-weight construction from rep quality
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_OUTSIDE_WEIGHT: float = 0.5   # frames not inside any rep


def build_sample_weights(
    n_frames:      int,
    rep_starts:    np.ndarray,
    rep_ends:      np.ndarray,
    rep_qualities: np.ndarray,
    outside_weight: float = DEFAULT_OUTSIDE_WEIGHT,
) -> np.ndarray:
    """
    Map each frame to a quality-derived loss weight.

    Frame inside rep i  →  (rep_qualities[i] + 1) / 10.0
    Frame outside reps  →  outside_weight (default 0.5)

    Overlapping reps: the LAST rep covering a frame wins (rare; reps shouldn't
    overlap, but we stay deterministic).
    """
    w = np.full(n_frames, outside_weight, dtype=np.float32)
    for s, e, q in zip(rep_starts, rep_ends, rep_qualities):
        s_i = int(max(0, s))
        e_i = int(min(n_frames - 1, e))
        if e_i < s_i:
            continue
        w[s_i:e_i + 1] = (float(q) + 1.0) / 10.0
    return w


# ══════════════════════════════════════════════════════════════════════════════
#  State merge map
#
#  Problem: some state pairs are visually identical at the frame level.
#  The model cannot distinguish them from a single frame without temporal
#  context, so training with both as separate classes introduces noise.
#
#  Solution: merge them into one combined state at load time.
#  The .npz files on disk are NEVER modified — the merge happens in memory
#  only.  To undo a merge, delete the entry from the dict below.
#
#  Rule for merging: merge states that share the same body geometry.
#    sq_standing  ≈  sq_ascending   (both = upright, knees straight)
#    pu_plank     ≈  pu_ascending   (both = arms extended, body rigid)
#    su_lying     ≈  su_descending  (both = flat / near-flat on back)
#    td_top       ≈  td_ascending   (both = arms straight at top)
#    sp_rack      ≈  sp_lowering    (both = hands at shoulder height)
#    bc_extended  ≈  bc_lowering    (both = arms down, curl released)
#    lr_down      ≈  lr_lowering    (both = arms at sides)
#    lu_standing  ≈  lu_returning   (both = upright, weight centred)
#    pu_hanging   ≈  pu_lowering    (both = dead hang, arms extended)
#
#  Timed exercises (plank, wall_sit) have only one state — no merge needed.
#  high_knees, mountain_climber, jumping_jack, burpee — states are physically
#  distinct at the frame level — no merge needed.
# ══════════════════════════════════════════════════════════════════════════════

MERGE_MAP: Dict[str, Dict[str, str]] = {
    "squat": {
        "sq_standing":   "sq_up",
        "sq_ascending":  "sq_up",
        # sq_descending and sq_bottom stay as-is
    },
    "pushup": {
        "pu_plank":      "pu_up",
        "pu_ascending":  "pu_up",
    },
    "situp": {
        "su_lying":      "su_down",
        "su_descending": "su_down",
    },
    "pullup": {
        "pu_hanging":    "pu_down",
        "pu_lowering":   "pu_down",
    },
    "tricep_dip": {
        "td_top":        "td_up",
        "td_ascending":  "td_up",
    },
    "shoulder_press": {
        "sp_rack":       "sp_down",
        "sp_lowering":   "sp_down",
    },
    "bicep_curl": {
        "bc_extended":   "bc_down",
        "bc_lowering":   "bc_down",
    },
    "lateral_raise": {
        "lr_down":       "lr_down",   # keep name, merge lowering into it
        "lr_lowering":   "lr_down",
    },
    "lunge": {
        "lu_standing":   "lu_up",
        "lu_returning":  "lu_up",
    },
}


def apply_merge(
    raw_names:    List[str],
    raw_states:   np.ndarray,
    exercise:     str,
) -> Tuple[np.ndarray, List[str]]:
    """
    Apply MERGE_MAP for one exercise.

    Parameters
    ──────────
    raw_names  : state names as stored in the .npz  (e.g. ['sq_standing', ...])
    raw_states : (N,) int64 frame state indices into raw_names
    exercise   : exercise name string

    Returns
    ───────
    (merged_states, merged_names)
      merged_states : (N,) int64 — new indices into merged_names
      merged_names  : list[str]  — deduplicated, order-preserving merged vocab
    """
    merge = MERGE_MAP.get(exercise, {})
    if not merge:
        return raw_states, raw_names

    # Build the merged name for every original name
    mapped = [merge.get(n, n) for n in raw_names]

    # Build merged_names: deduplicate while preserving first-occurrence order
    seen: List[str] = []
    for n in mapped:
        if n not in seen:
            seen.append(n)
    merged_names = seen

    # Build old_local_idx → new_local_idx lookup
    idx_map = {
        old_i: merged_names.index(mapped[old_i])
        for old_i in range(len(raw_names))
    }

    merged_states = np.array(
        [idx_map[int(s)] for s in raw_states], dtype=np.int64
    )
    return merged_states, merged_names


# ══════════════════════════════════════════════════════════════════════════════
#  Data loading — per-exercise, video-grouped, isolated vocab
# ══════════════════════════════════════════════════════════════════════════════

class VideoData:
    """Everything loaded from one labeled video, before feature extraction."""
    def __init__(
        self,
        vid_key:    str,
        keypoints:  np.ndarray,   # (N, 17, 2)
        states:     np.ndarray,   # (N,) local state indices
        weights:    np.ndarray,   # (N,) per-frame sample weights
    ) -> None:
        self.vid_key   = vid_key
        self.keypoints = keypoints
        self.states    = states
        self.weights   = weights
        self.n_frames  = len(keypoints)


def load_exercise_videos(
    exercise:      str,
    processed_dir: Path,
    labeled_dir:   Path,
    verbose:       bool = True,
) -> Tuple[List[VideoData], List[str]]:
    """
    Load every labeled video for one exercise.

    Returns
    ───────
    (videos, state_names)
      videos      : list of VideoData (keypoints + per-frame state + weight)
      state_names : list[str] — the isolated vocabulary for this exercise,
                    taken from the first video's state_names array

    State indices are assumed to be local (0..num_classes-1) as written by
    label_tool_v2.py.  All videos of one exercise must share state_names.
    """
    ex_label_dir = labeled_dir / exercise
    if not ex_label_dir.is_dir():
        raise FileNotFoundError(f"No labeled dir for exercise: {ex_label_dir}")

    videos:      List[VideoData] = []
    state_names: Optional[List[str]] = None

    for label_npz in sorted(ex_label_dir.glob("*.npz")):
        video_id = label_npz.stem
        vid_key  = f"{exercise}/{video_id}"

        try:
            label = np.load(label_npz, allow_pickle=True)

            # Keypoints: prefer labeled file, else read from processed/
            if "keypoints" in label:
                kp = label["keypoints"].astype(np.float32)
            else:
                proc_npz = processed_dir / exercise / f"{video_id}.npz"
                if not proc_npz.exists():
                    if verbose:
                        print(f"  [WARN] missing processed keypoints: {vid_key}")
                    continue
                proc = np.load(proc_npz, allow_pickle=True)
                kp   = proc["keypoints"].astype(np.float32)

            states = label["frame_states"].astype(np.int64)        # (N,) local idx
            names  = [str(s) for s in label["state_names"]]        # isolated vocab

            # ── Apply state merge (in-memory only, .npz unchanged) ────────────
            states, names = apply_merge(names, states, exercise)

            # Validate consistent vocabulary across videos
            if state_names is None:
                state_names = names
            elif names != state_names:
                if verbose:
                    print(f"  [WARN] {vid_key}: state_names differ from first "
                          f"video — skipping.\n"
                          f"         first={state_names}\n         this ={names}")
                continue

            # Rep arrays (may be empty for timed exercises)
            rep_starts    = label["rep_starts"].astype(np.int64) \
                if "rep_starts" in label else np.array([], dtype=np.int64)
            rep_ends      = label["rep_ends"].astype(np.int64) \
                if "rep_ends" in label else np.array([], dtype=np.int64)
            rep_qualities = label["rep_qualities"].astype(np.int64) \
                if "rep_qualities" in label else np.array([], dtype=np.int64)

            # Align lengths defensively
            N = min(len(kp), len(states))
            kp, states = kp[:N], states[:N]

            # Clip out-of-range state indices
            n_classes_local = len(state_names)
            bad = (states < 0) | (states >= n_classes_local)
            if bad.any():
                if verbose:
                    print(f"  [WARN] {vid_key}: {int(bad.sum())} out-of-range "
                          f"state indices → clipped to 0")
                states[bad] = 0

            weights = build_sample_weights(
                N, rep_starts, rep_ends, rep_qualities
            )

            videos.append(VideoData(vid_key, kp, states, weights))

            if verbose:
                uniq = np.unique(states).tolist()
                print(f"  [load] {vid_key}: {N} frames  "
                      f"{len(rep_starts)} reps  states={uniq}")

        except Exception as exc:
            if verbose:
                print(f"  [WARN] failed {vid_key}: {exc}")

    if not videos:
        raise RuntimeError(f"No usable videos loaded for {exercise}")
    if state_names is None:
        raise RuntimeError(f"No state_names found for {exercise}")

    return videos, state_names


# ══════════════════════════════════════════════════════════════════════════════
#  Video-wise split
# ══════════════════════════════════════════════════════════════════════════════

def split_videos(
    videos:    List[VideoData],
    test_size: float = 0.20,
    val_size:  float = 0.20,
    seed:      int   = 42,
) -> Tuple[List[VideoData], List[VideoData], List[VideoData]]:
    """
    Split by whole video so no frame leaks across train/val/test.
    Guarantees at least 1 video in test and 1 in val when ≥3 videos exist.
    """
    idx = np.arange(len(videos))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n = len(videos)
    n_test = max(1, int(n * test_size)) if n >= 3 else 0
    n_val  = max(1, int(n * val_size))  if n >= 3 else 0
    if n >= 3:
        n_val = min(n_val, n - n_test - 1)
        if n_val < 1:
            n_val = 1

    test_idx  = set(idx[:n_test].tolist())
    val_idx   = set(idx[n_test:n_test + n_val].tolist())

    train, val, test = [], [], []
    for i, v in enumerate(videos):
        if i in test_idx:
            test.append(v)
        elif i in val_idx:
            val.append(v)
        else:
            train.append(v)
    return train, val, test


# ══════════════════════════════════════════════════════════════════════════════
#  StandardScaler (NumPy)
# ══════════════════════════════════════════════════════════════════════════════

class StandardScaler:
    """Zero-mean / unit-variance per feature; safe for constant features."""

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_:  Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0)
        self.std_  = np.where(self.std_ < 1e-8, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean_) / self.std_).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  Datasets
# ══════════════════════════════════════════════════════════════════════════════

class FrameDataset(Dataset):
    """
    Static (non-augmented) frame dataset for VALIDATION and TEST.

    Pre-computes the full 20-feature matrix for every frame once, applies the
    scaler, and serves (features, label, sample_weight) tuples.
    """

    def __init__(
        self,
        videos: List[VideoData],
        scaler: StandardScaler,
    ) -> None:
        feats, labels, weights = [], [], []
        for v in videos:
            base = build_base_matrix(v.keypoints)        # (N,10)
            full = full_features_from_base(base)         # (N,20)
            feats.append(full)
            labels.append(v.states)
            weights.append(v.weights)

        self.X = scaler.transform(np.vstack(feats))      # (ΣN, 20)
        self.y = np.concatenate(labels).astype(np.int64) # (ΣN,)
        self.w = np.concatenate(weights).astype(np.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return (
            torch.from_numpy(self.X[i]),
            int(self.y[i]),
            float(self.w[i]),
        )


class AugmentedFrameDataset(Dataset):
    """
    On-the-fly augmented dataset for TRAINING.

    Each __getitem__ returns ONE frame, but augmentation operates per VIDEO so
    that deltas remain temporally coherent.  To keep this efficient we cache,
    per epoch-ish access, the augmented feature matrix per video and rebuild it
    only when the cache is invalidated by set_epoch().

    Augmentations
    ─────────────
      1. Keypoint jitter — Gaussian noise (sigma px) added to every visible
         keypoint BEFORE feature extraction.  Occluded (0,0) joints stay 0.
      2. Frame-dropout deltas — 30% of the time, the delta reference index for
         a frame is pulled back by an extra 1-2 frames to simulate camera
         stutter.  This perturbs only the delta half of the vector.

    The scaler (fit on clean train features) is applied after augmentation.
    """

    def __init__(
        self,
        videos:        List[VideoData],
        scaler:        StandardScaler,
        jitter_sigma:  float = 2.0,
        dropout_prob:  float = 0.30,
        augment:       bool  = True,
        seed:          int   = 0,
    ) -> None:
        self.videos       = videos
        self.scaler       = scaler
        self.jitter_sigma = jitter_sigma
        self.dropout_prob = dropout_prob
        self.augment      = augment
        self._rng         = np.random.default_rng(seed)

        # Frame index map: global frame i → (video_idx, local_frame)
        self._index: List[Tuple[int, int]] = []
        for vi, v in enumerate(videos):
            for fi in range(v.n_frames):
                self._index.append((vi, fi))

        # Per-video cache of the current epoch's augmented (N,20) matrix
        self._cache: List[Optional[np.ndarray]] = [None] * len(videos)
        self._build_all()

    # ── cache control ──────────────────────────────────────────────────────────

    def set_epoch(self, epoch: int) -> None:
        """Re-roll augmentation for a new epoch."""
        self._rng = np.random.default_rng(1000 + epoch)
        self._build_all()

    def _build_all(self) -> None:
        for vi, v in enumerate(self.videos):
            self._cache[vi] = self._build_video(v)

    # ── augmentation core ───────────────────────────────────────────────────────

    def _build_video(self, v: VideoData) -> np.ndarray:
        kp = v.keypoints
        if self.augment and self.jitter_sigma > 0:
            kp = self._jitter(kp)

        base = build_base_matrix(kp)             # (N,10)

        if self.augment and self.dropout_prob > 0:
            refs = self._dropout_refs(v.n_frames)
            d    = deltas_from_base(base, lag=DELTA_LAG, dropout_refs=refs)
        else:
            d    = deltas_from_base(base, lag=DELTA_LAG)

        full = np.hstack([base, d]).astype(np.float32)
        return self.scaler.transform(full)

    def _jitter(self, kp: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to visible keypoints only."""
        noisy = kp.copy()
        # visible mask: any joint not exactly (0,0)
        vis = ~((kp[:, :, 0] == 0.0) & (kp[:, :, 1] == 0.0))   # (N,17)
        noise = self._rng.normal(0.0, self.jitter_sigma, size=kp.shape).astype(np.float32)
        noise[~vis] = 0.0
        noisy += noise
        return noisy

    def _dropout_refs(self, n: int) -> np.ndarray:
        """
        Per-frame delta reference index.
        Baseline ref(t) = max(t-LAG, 0).  With prob dropout_prob, pull the
        reference back an extra 1-2 frames (clamped to 0) to simulate a stutter.
        """
        refs = np.maximum(np.arange(n) - DELTA_LAG, 0)
        for t in range(n):
            if self._rng.random() < self.dropout_prob:
                extra = int(self._rng.integers(1, 3))   # 1 or 2
                refs[t] = max(refs[t] - extra, 0)
        return refs

    # ── Dataset protocol ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int):
        vi, fi = self._index[i]
        X_v = self._cache[vi]
        v   = self.videos[vi]
        return (
            torch.from_numpy(X_v[fi]),
            int(v.states[fi]),
            float(v.weights[fi]),
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Model — single-head MLP (exactly as specified)
# ══════════════════════════════════════════════════════════════════════════════

class SkeletonNet(nn.Module):
    """Input(20)→64→BN→ReLU→Drop→32→BN→ReLU→num_classes."""

    def __init__(self, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
#  Class weights
# ══════════════════════════════════════════════════════════════════════════════

def compute_class_weights(
    y:           np.ndarray,
    num_classes: int,
    device:      torch.device,
) -> torch.Tensor:
    """
    Balanced inverse-frequency weights, normalised to mean 1.0, capped at 10×.
    Mirrors sklearn's compute_class_weight('balanced') with a safety cap.
    """
    counts  = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts  = np.where(counts < 1.0, 1.0, counts)
    weights = counts.sum() / (num_classes * counts)   # balanced formula
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.0, 10.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ══════════════════════════════════════════════════════════════════════════════
#  Train one exercise
# ══════════════════════════════════════════════════════════════════════════════

def train_one_exercise(
    exercise:    str,
    videos:      List[VideoData],
    state_names: List[str],
    device:      torch.device,
    args:        argparse.Namespace,
) -> Optional[Dict]:
    """Full pipeline for one exercise. Returns a result dict or None on skip."""
    num_classes = len(state_names)

    if len(videos) < 3:
        print(f"  [SKIP] {exercise}: only {len(videos)} video(s), need ≥3")
        return None

    # ── Split (video-wise) ────────────────────────────────────────────────────
    train_v, val_v, test_v = split_videos(
        videos, test_size=args.test_size, val_size=args.val_size, seed=args.seed
    )
    print(f"  Split: {len(train_v)} train / {len(val_v)} val / {len(test_v)} test videos")

    # ── Fit scaler on CLEAN (un-augmented) training features only ─────────────
    train_base = [build_base_matrix(v.keypoints) for v in train_v]
    train_full = np.vstack([full_features_from_base(b) for b in train_base])
    scaler = StandardScaler().fit(train_full)

    # ── Datasets / loaders ─────────────────────────────────────────────────────
    train_ds = AugmentedFrameDataset(
        train_v, scaler,
        jitter_sigma = args.jitter_sigma,
        dropout_prob = args.dropout_prob,
        augment      = not args.no_augment,
        seed         = args.seed,
    )
    val_ds  = FrameDataset(val_v,  scaler)
    test_ds = FrameDataset(test_v, scaler)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False)

    # ── Class weights from clean training labels ──────────────────────────────
    y_train = np.concatenate([v.states for v in train_v]).astype(np.int64)
    class_w = compute_class_weights(y_train, num_classes, device)

    # ── Model / optim / loss ───────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    model     = SkeletonNet(num_classes, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=max(args.patience // 3, 3), min_lr=1e-6,
    )
    # reduction='none' so we can multiply by per-sample weights ourselves
    criterion = nn.CrossEntropyLoss(weight=class_w, reduction="none")

    best_val_loss = math.inf
    best_val_acc  = 0.0
    best_state    = None
    no_improve    = 0

    print(f"  Model: {model.count_parameters():,} params | "
          f"classes={state_names}")
    header = f"  {'Ep':>4}  {'TrLoss':>8}  {'TrAcc':>7}  {'VaLoss':>8}  {'VaAcc':>7}  {'LR':>8}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    epoch = 0
    for epoch in range(1, args.epochs + 1):
        if not args.no_augment:
            train_ds.set_epoch(epoch)   # re-roll augmentation

        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        tr_loss = tr_correct = tr_total = 0
        for Xb, yb, wb in train_loader:
            Xb = Xb.to(device); yb = yb.to(device); wb = wb.to(device)
            logits   = model(Xb)
            per_samp = criterion(logits, yb)          # (B,) class-weighted CE
            loss     = (per_samp * wb).sum() / (wb.sum() + 1e-8)  # weight by quality
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss    += float(loss.item()) * Xb.size(0)
            tr_correct += int((logits.argmax(1) == yb).sum().item())
            tr_total   += Xb.size(0)
        tr_loss_avg = tr_loss / max(tr_total, 1)
        tr_acc      = tr_correct / max(tr_total, 1)

        # ── Validate (no augmentation, no sample weighting in the metric) ───────
        model.eval()
        va_loss = va_correct = va_total = 0
        with torch.no_grad():
            for Xb, yb, wb in val_loader:
                Xb = Xb.to(device); yb = yb.to(device); wb = wb.to(device)
                logits   = model(Xb)
                per_samp = criterion(logits, yb)
                loss     = (per_samp * wb).sum() / (wb.sum() + 1e-8)
                va_loss    += float(loss.item()) * Xb.size(0)
                va_correct += int((logits.argmax(1) == yb).sum().item())
                va_total   += Xb.size(0)
        va_loss_avg = va_loss / max(va_total, 1)
        va_acc      = va_correct / max(va_total, 1)

        scheduler.step(va_loss_avg)
        lr_now = optimizer.param_groups[0]["lr"]

        improved = va_loss_avg < best_val_loss - 1e-4
        flag = ""
        if improved:
            best_val_loss = va_loss_avg
            best_val_acc  = va_acc
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
            flag = " ★"
        else:
            no_improve += 1

        print(f"  {epoch:>4d}  {tr_loss_avg:>8.4f}  {tr_acc:>7.3f}  "
              f"{va_loss_avg:>8.4f}  {va_acc:>7.3f}  {lr_now:>8.2e}{flag}")

        if no_improve >= args.patience:
            print(f"  Early stop @ epoch {epoch} (no val improvement {args.patience} ep)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # ── Test evaluation ────────────────────────────────────────────────────────
    test_acc = evaluate(model, test_loader, state_names, device, exercise)

    # ── Save checkpoint (v1-compatible format) ─────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"dl_model_{exercise}.pt"
    torch.save(
        {
            "model_state":   model.state_dict(),
            "scaler_mean":   scaler.mean_.astype(np.float32),
            "scaler_std":    scaler.std_.astype(np.float32),
            "classes":       list(range(num_classes)),   # local indices 0..N-1
            "state_names":   state_names,                 # isolated vocab (NEW)
            "n_features":    N_FEATURES,
            "num_classes":   num_classes,
            "best_val_acc":  float(best_val_acc),
            "test_acc":      float(test_acc),
            "epochs_run":    epoch,
            "feature_names": ALL_FEAT_NAMES,
            "version":       "dl_v2",
        },
        ckpt_path,
    )
    print(f"  Saved → {ckpt_path}")

    return {
        "exercise":     exercise,
        "state_names":  state_names,
        "n_train_vids": len(train_v),
        "n_test_vids":  len(test_v),
        "best_val_acc": round(best_val_acc, 4),
        "test_acc":     round(test_acc, 4),
        "epochs_run":   epoch,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
    model:       SkeletonNet,
    loader:      DataLoader,
    state_names: List[str],
    device:      torch.device,
    exercise:    str,
) -> float:
    """Print classification report + confusion matrix; return accuracy."""
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    for Xb, yb, _ in loader:
        logits = model(Xb.to(device))
        y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
        y_true.extend(yb.numpy().tolist())

    if not y_true:
        print("  [WARN] empty test set")
        return 0.0

    y_true_a = np.array(y_true)
    y_pred_a = np.array(y_pred)
    present  = sorted(set(y_true_a.tolist()) | set(y_pred_a.tolist()))
    labels   = [state_names[i] for i in present]

    print(f"\n  ── TEST REPORT: {exercise} ──")
    print(classification_report(
        y_true_a, y_pred_a, labels=present,
        target_names=labels, zero_division=0,
    ))

    cm = confusion_matrix(y_true_a, y_pred_a, labels=present)
    print("  Confusion matrix (rows=true, cols=pred):")
    print(f"  {'':>14}", end="")
    for lab in labels:
        print(f"{lab[:8]:>10}", end="")
    print()
    for i, lab in enumerate(labels):
        print(f"  {lab:>14}", end="")
        for j in range(len(labels)):
            mark = "*" if i == j else " "
            print(f"{cm[i, j]:>9}{mark}", end="")
        print()

    acc = float((y_pred_a == y_true_a).mean())
    print(f"\n  Test accuracy: {acc:.4f} ({acc*100:.1f}%)")
    return acc


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="DL v2 — per-exercise micro-models with quality sample weighting",
    )
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--labeled-dir",   default="data/labeled_v2")
    p.add_argument("--out-dir",       default="checkpoints_v2")
    p.add_argument("--exercise",      default=None,
                   help="Train only this exercise (default: all found)")
    p.add_argument("--test-size",  type=float, default=0.20)
    p.add_argument("--val-size",   type=float, default=0.20)
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch-size", type=int,   default=256)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--dropout",    type=float, default=0.3)
    p.add_argument("--patience",   type=int,   default=15)
    p.add_argument("--jitter-sigma", type=float, default=2.0,
                   help="Gaussian keypoint jitter std in pixels (0 = off)")
    p.add_argument("--dropout-prob", type=float, default=0.30,
                   help="Per-frame probability of delta frame-dropout aug")
    p.add_argument("--no-augment", action="store_true",
                   help="Disable all augmentation (jitter + frame dropout)")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--device",     default="auto")
    args = p.parse_args()

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    labeled_dir = Path(args.labeled_dir)
    if not labeled_dir.exists():
        print(f"[ERROR] labeled-dir not found: {labeled_dir}")
        sys.exit(1)

    # Which exercises?
    if args.exercise:
        exercises = [args.exercise]
    else:
        exercises = sorted(d.name for d in labeled_dir.iterdir() if d.is_dir())

    print("\n" + "═" * 64)
    print("  DL v2 — Per-Exercise Micro-Models  (quality-weighted)")
    print("═" * 64)
    print(f"  Device       : {device}")
    print(f"  Labeled dir  : {labeled_dir}")
    print(f"  Output dir   : {args.out_dir}")
    print(f"  Augment      : {'OFF' if args.no_augment else f'jitter σ={args.jitter_sigma}px, dropout p={args.dropout_prob}'}")
    print(f"  Exercises    : {exercises}")
    print("═" * 64)

    results: List[Dict] = []
    for ex in exercises:
        print(f"\n{'─'*64}\n  EXERCISE: {ex.upper()}\n{'─'*64}")
        try:
            videos, state_names = load_exercise_videos(
                ex, Path(args.processed_dir), labeled_dir, verbose=True
            )
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"  [SKIP] {ex}: {exc}")
            continue

        res = train_one_exercise(ex, videos, state_names, device, args)
        if res:
            results.append(res)

    # ── Summary ────────────────────────────────────────────────────────────────
    if results:
        print(f"\n{'═'*64}\n  SUMMARY\n{'═'*64}")
        print(f"  {'Exercise':<20}{'Classes':>8}{'ValAcc':>9}{'TestAcc':>9}")
        print(f"  {'─'*20}{'─'*8}{'─'*9}{'─'*9}")
        for r in results:
            print(f"  {r['exercise']:<20}{len(r['state_names']):>8}"
                  f"{r['best_val_acc']*100:>8.1f}%{r['test_acc']*100:>8.1f}%")

        # Write a small JSON index of trained models
        out_dir = Path(args.out_dir)
        with open(out_dir / "training_summary_v2.json", "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"\n  Summary → {out_dir / 'training_summary_v2.json'}")
    else:
        print("\n[ERROR] No models trained.")
        sys.exit(1)

    print("\n[DL v2] Done.\n")


if __name__ == "__main__":
    main()
