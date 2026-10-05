"""
scripts/inference_dl.py
═══════════════════════
Real-Time SkeletonNet Inference  —  MediaPipe Pose + PyTorch MLP

Overview
────────
Loads a per-exercise SkeletonNet checkpoint produced by train_dl_v1.py, opens
a webcam (or video file), runs MediaPipe Pose on every frame, computes the
exact same 20-feature vector used during training, and displays the predicted
exercise state and rep count in a live OpenCV window.

Everything from train_dl_v1.py is reproduced verbatim:
  • Feature definitions (ANGLE_DEFS, DELTA_LAG, N_BASE, N_FEATURES)
  • _is_occluded / _angle_at_vertex / _bbox_aspect_ratio / _norm_nose_ankle_y
  • calculate_features() — produces the 10-element base vector
  • StandardScaler transform (mean / std loaded from .pt file)
  • Local→global class index remapping

The only new pieces are:
  • MediaPipe Pose → COCO-17 keypoint extraction
  • Per-frame deque for delta computation
  • Mode-vote smoother (deque of last N raw predictions)
  • Dwell-gated rep counter FSM (mirrors counter.py logic)
  • OpenCV HUD rendering

MediaPipe → COCO-17 index mapping  (verified against MP BlazePose 33):
  COCO[ 0] nose          → MP[ 0]    COCO[ 9] left_wrist   → MP[15]
  COCO[ 5] left_shoulder → MP[11]    COCO[10] right_wrist  → MP[16]
  COCO[ 6] right_shoulder→ MP[12]    COCO[11] left_hip     → MP[23]
  COCO[ 7] left_elbow    → MP[13]    COCO[12] right_hip    → MP[24]
  COCO[ 8] right_elbow   → MP[14]    COCO[13] left_knee    → MP[25]
                                      COCO[14] right_knee   → MP[26]
                                      COCO[15] left_ankle   → MP[27]
                                      COCO[16] right_ankle  → MP[28]

Indices not in this table (COCO 1-4) are eyes/ears, which are not used by any
ANGLE_DEF or spatial feature, so they are left as (0, 0) — treated as occluded.

Controls
────────
  [q] / [ESC]  — quit
  [r]          — reset rep counter and feature deque
  [p]          — pause / resume

Usage
─────
  python scripts/inference_dl.py --exercise squat
  python scripts/inference_dl.py --exercise pushup --checkpoint-dir checkpoints
  python scripts/inference_dl.py --exercise squat  --source data/test.mp4
  python scripts/inference_dl.py --exercise squat  --source 1  --flip
  python scripts/inference_dl.py --exercise squat  --device cpu --smooth-k 7
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════════
#  Global state vocabulary
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


# ══════════════════════════════════════════════════════════════════════════════
#  Rep sequences (from counter.py REP_SEQUENCE — kept in sync)
#
#  Each tuple is the ordered list of states that must be observed in sequence
#  to increment the rep counter by 1.  For timed exercises (plank, wall_sit)
#  the sequence is None and hold-seconds are counted instead.
# ══════════════════════════════════════════════════════════════════════════════

REP_SEQUENCES: Dict[str, Optional[List[str]]] = {
    "squat":            ["down", "up"],
    "pushup":           ["down", "up"],
    "pullup":           ["up", "neutral"],
    "situp":            ["up", "neutral"],
    "lunge":            ["down", "neutral"],
    "tricep_dip":       ["down", "up"],
    "shoulder_press":   ["down", "extended"],
    "bicep_curl":       ["extended", "flexed"],
    "jumping_jack":     ["extended", "neutral"],
    "lateral_raise":    ["extended", "neutral"],
    "high_knees":       ["flexed", "neutral"],
    "mountain_climber": ["flexed", "neutral"],
    "burpee":           ["down", "plank", "down", "up", "jump", "neutral"],
    # Timed holds — no rep sequence
    "plank":            None,
    "wall_sit":         None,
}

# States during which the hold timer runs (timed exercises only)
HOLD_TARGET: Dict[str, str] = {
    "plank":     "plank",
    "wall_sit":  "down",
}


# ══════════════════════════════════════════════════════════════════════════════
#  MediaPipe → COCO-17 index table
#
#  MediaPipe BlazePose has 33 landmarks.  We only need 13 of them.
#  Key   = COCO-17 index used by ANGLE_DEFS / spatial features
#  Value = corresponding MediaPipe landmark index
# ══════════════════════════════════════════════════════════════════════════════

# fmt: off
MP_TO_COCO17: Dict[int, int] = {
    #  COCO_idx : MP_idx
    0:   0,   # nose          → MP 0
    # COCO 1-4 (eyes/ears) not used by any feature — left as (0,0)
    5:  11,   # left_shoulder → MP 11
    6:  12,   # right_shoulder→ MP 12
    7:  13,   # left_elbow    → MP 13
    8:  14,   # right_elbow   → MP 14
    9:  15,   # left_wrist    → MP 15
    10: 16,   # right_wrist   → MP 16
    11: 23,   # left_hip      → MP 23
    12: 24,   # right_hip     → MP 24
    13: 25,   # left_knee     → MP 25
    14: 26,   # right_knee    → MP 26
    15: 27,   # left_ankle    → MP 27
    16: 28,   # right_ankle   → MP 28
}
# fmt: on

# Visibility threshold below which a landmark is treated as occluded (→ 0, 0)
_MP_VIS_THRESHOLD: float = 0.5

# MediaPipe skeleton edges for drawing — pairs of COCO-17 indices
# (only the joints we actually track, for clean rendering)
_COCO_SKELETON_EDGES: List[Tuple[int, int]] = [
    (5, 6),    # shoulder bar
    (5, 7),    (7, 9),    # left arm
    (6, 8),    (8, 10),   # right arm
    (5, 11),   (6, 12),   # torso sides
    (11, 12),             # hip bar
    (11, 13),  (13, 15),  # left leg
    (12, 14),  (14, 16),  # right leg
]


# ══════════════════════════════════════════════════════════════════════════════
#  Feature engineering  (exact copy from train_dl_v1.py — DO NOT modify)
# ══════════════════════════════════════════════════════════════════════════════

DELTA_LAG: int = 5   # frames; must match training

ANGLE_DEFS: List[Tuple[str, int, int, int]] = [
    ("L_Knee",      11, 13, 15),   # hip → knee ← ankle
    ("R_Knee",      12, 14, 16),
    ("L_Hip",        5, 11, 13),   # shoulder → hip ← knee
    ("R_Hip",        6, 12, 14),
    ("L_Elbow",      5,  7,  9),   # shoulder → elbow ← wrist
    ("R_Elbow",      6,  8, 10),
    ("L_Shoulder",  11,  5,  7),   # hip → shoulder ← elbow
    ("R_Shoulder",  12,  6,  8),
]
N_ANGLES:   int = len(ANGLE_DEFS)   # 8
N_BASE:     int = N_ANGLES + 2      # 10  (+ BBox_AR + NoseAnkle_Y)
N_FEATURES: int = N_BASE * 2        # 20  (base + deltas)


def _is_occluded(pt: np.ndarray) -> bool:
    """True when the keypoint is the (0, 0) sentinel (not detected)."""
    return float(pt[0]) == 0.0 and float(pt[1]) == 0.0


def _angle_at_vertex(a: np.ndarray, v: np.ndarray, c: np.ndarray) -> float:
    """
    Interior angle in degrees at vertex v in triplet (a, v, c).
    Returns 0.0 for any occluded point or degenerate (zero-length) vector.
    Identical to train_dl_v1.py — must stay in sync.
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
    ~1.0 when fully upright; ~0.0 when lying flat.
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
    Extract the 10-element base feature vector from one (17, 2) COCO frame.
      [0:8]  8 joint angles (degrees)
      [8]    bounding-box aspect ratio
      [9]    normalised nose-to-ankle Y distance
    dtype: float32.
    """
    feat = np.zeros(N_BASE, dtype=np.float32)
    for i, (_, a, v, c) in enumerate(ANGLE_DEFS):
        feat[i] = _angle_at_vertex(keypoints[a], keypoints[v], keypoints[c])
    feat[N_ANGLES]     = _bbox_aspect_ratio(keypoints)
    feat[N_ANGLES + 1] = _norm_nose_ankle_y(keypoints)
    return feat


# ══════════════════════════════════════════════════════════════════════════════
#  SkeletonNet  (must match train_dl_v1.py exactly)
# ══════════════════════════════════════════════════════════════════════════════

class SkeletonNet(nn.Module):
    """
    MLP: Input(20) → Linear(64) → BN → ReLU → Dropout → Linear(32) → BN → ReLU
         → Linear(num_classes)
    Architecture is fixed — any change here breaks checkpoint compatibility.
    """

    def __init__(self, num_classes: int, dropout: float = 0.0) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
#  Checkpoint loader
# ══════════════════════════════════════════════════════════════════════════════

class ExerciseModel:
    """
    Wraps a loaded SkeletonNet checkpoint.  Handles:
      • SkeletonNet reconstruction from saved num_classes
      • StandardScaler (mean/std arrays from checkpoint)
      • Local→global class index mapping
      • Single-frame inference
    """

    def __init__(self, checkpoint_path: Path, device: torch.device) -> None:
        self.device = device

        ckpt = torch.load(str(checkpoint_path), map_location=device,
                          weights_only=False)

        # ── Validate checkpoint keys ──────────────────────────────────────────
        required = {"model_state", "scaler_mean", "scaler_std",
                    "classes", "n_features", "num_classes"}
        missing  = required - ckpt.keys()
        if missing:
            raise KeyError(
                f"Checkpoint is missing required keys: {missing}\n"
                f"  Found: {list(ckpt.keys())}\n"
                f"  Was this checkpoint saved by train_dl_v1.py?"
            )
        if ckpt["n_features"] != N_FEATURES:
            raise ValueError(
                f"Checkpoint was trained with n_features={ckpt['n_features']} "
                f"but this script expects {N_FEATURES}.  "
                f"Rebuild the checkpoint with train_dl_v1.py."
            )

        # ── Scaler parameters ─────────────────────────────────────────────────
        self.scaler_mean: np.ndarray = ckpt["scaler_mean"].astype(np.float32)
        self.scaler_std:  np.ndarray = ckpt["scaler_std"].astype(np.float32)

        # ── Class index mapping ───────────────────────────────────────────────
        # classes is a list of GLOBAL state indices handled by this model, e.g.
        # [0, 1, 2] for squat or [1, 2, 5] for tricep_dip.
        # local index 0 → classes[0], local 1 → classes[1], etc.
        self.classes:       List[int] = list(ckpt["classes"])
        self.num_classes:   int       = len(self.classes)
        self.local_to_global: Dict[int, int] = {
            i: g for i, g in enumerate(self.classes)
        }

        # ── Model ─────────────────────────────────────────────────────────────
        # Use dropout=0.0 at inference time (BN is in eval mode so it uses
        # running stats instead of batch stats — correct for single samples).
        self.model = SkeletonNet(
            num_classes = self.num_classes,
            dropout     = 0.0,
        ).to(device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.best_val_acc: float = float(ckpt.get("best_val_acc", 0.0))
        self.epochs_run:   int   = int(ckpt.get("epochs_run",   0))

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(self, feat_vec_raw: np.ndarray) -> Tuple[int, str, float]:
        """
        Run a single inference pass.

        Parameters
        ----------
        feat_vec_raw : (20,) float32  — raw (un-standardised) feature vector

        Returns
        -------
        global_idx  : int   — index into STATE_NAMES (0-6)
        state_name  : str   — e.g. "down"
        confidence  : float — softmax probability of the winning class [0,1]
        """
        # Standardise using training-split statistics
        feat_norm = (feat_vec_raw - self.scaler_mean) / self.scaler_std
        x = torch.tensor(feat_norm, dtype=torch.float32,
                         device=self.device).unsqueeze(0)  # (1, 20)

        with torch.no_grad():
            logits = self.model(x)                            # (1, num_classes)
            probs  = torch.softmax(logits, dim=-1)[0]         # (num_classes,)

        local_idx  = int(probs.argmax().item())
        confidence = float(probs[local_idx].item())
        global_idx = self.local_to_global[local_idx]
        return global_idx, STATE_NAMES[global_idx], confidence


# ══════════════════════════════════════════════════════════════════════════════
#  Temporal feature deque
#
#  Maintains a sliding window of base feature vectors so that at time t we can
#  instantly compute:
#      feat_delta = feat_base[t]  −  feat_base[t − DELTA_LAG]
#  The deque holds exactly DELTA_LAG+1 entries.  While it is filling up (fewer
#  than DELTA_LAG+1 frames seen) the delta is computed against the oldest
#  available frame rather than returning zeros — same convention as training.
# ══════════════════════════════════════════════════════════════════════════════

class FeatureDeque:
    """Ring buffer of base feature vectors for delta computation."""

    def __init__(self, delta_lag: int = DELTA_LAG) -> None:
        self._lag:  int = delta_lag
        # maxlen = lag + 1  →  index [-1] is current, index [0] is t−lag
        self._buf: Deque[np.ndarray] = collections.deque(maxlen=delta_lag + 1)

    def push_and_compute(self, base: np.ndarray) -> np.ndarray:
        """
        Append the new base vector and return the full 20-feature vector.

        Parameters
        ----------
        base : (N_BASE,) float32  — output of calculate_features()

        Returns
        -------
        (N_FEATURES,) float32  =  [base | base - ref]
        where ref is the base vector DELTA_LAG frames ago (or the oldest
        available frame if the buffer is still filling up).
        """
        self._buf.append(base.copy())
        # Reference frame: oldest in the buffer (index 0)
        ref   = self._buf[0]      # (N_BASE,)
        delta = base - ref        # (N_BASE,)
        return np.concatenate([base, delta], axis=0).astype(np.float32)

    def reset(self) -> None:
        self._buf.clear()

    @property
    def filled(self) -> bool:
        """True once the buffer has accumulated DELTA_LAG+1 entries."""
        return len(self._buf) > self._lag


# ══════════════════════════════════════════════════════════════════════════════
#  Mode-vote smoother
#
#  Holds the last K raw state predictions and returns the most frequent one.
#  Ties are broken by recency (we count from newest to oldest and return the
#  first state whose count matches the maximum) — this biases toward the most
#  recently seen state, which is the right heuristic at decision boundaries.
# ══════════════════════════════════════════════════════════════════════════════

class ModeVoteSmoother:
    """Temporal smoothing via majority vote over a sliding window."""

    def __init__(self, window: int = 5) -> None:
        self._win: int = window
        self._buf: Deque[int] = collections.deque(maxlen=window)

    def push(self, raw_idx: int) -> int:
        """Add a raw prediction and return the smoothed (mode) index."""
        self._buf.append(raw_idx)
        counts: Dict[int, int] = {}
        for v in self._buf:
            counts[v] = counts.get(v, 0) + 1
        best_count = max(counts.values())
        # Walk newest→oldest to break ties by recency
        for v in reversed(self._buf):
            if counts[v] == best_count:
                return v
        return raw_idx  # unreachable

    def reset(self) -> None:
        self._buf.clear()


# ══════════════════════════════════════════════════════════════════════════════
#  Dwell-gated rep counter FSM
#
#  A state must be observed for MIN_DWELL consecutive smoothed frames before
#  it is "committed" and allowed to advance the sequence cursor.  This prevents
#  a single noisy frame from incrementing the count.
#
#  Mirrors the core logic of counter.py RepCounter without the full asyncio
#  infrastructure, since we need a simpler synchronous version here.
# ══════════════════════════════════════════════════════════════════════════════

# Frames a state must persist before being committed (≈100 ms at 30 fps)
_DWELL_FRAMES: int = 3
# Frames the cursor is locked after a rep is counted (≈500 ms at 30 fps)
_COOLDOWN_FRAMES: int = 15


class RepCounterFSM:
    """
    Sequence-cursor rep counter driven by committed (dwell-gated) state strings.

    For timed exercises (REP_SEQUENCES value is None) it instead accumulates
    hold_seconds while the target state is maintained.
    """

    def __init__(self, exercise: str) -> None:
        self.exercise  = exercise
        self._seq      = REP_SEQUENCES.get(exercise)
        self._is_timed = (self._seq is None)
        self._hold_tgt = HOLD_TARGET.get(exercise, "")

        self.reps:         int   = 0
        self.hold_seconds: float = 0.0

        self._cursor:       int   = 0
        self._dwell_cand:   str   = "neutral"
        self._dwell_cnt:    int   = 0
        self._committed:    str   = "neutral"
        self._cooldown:     int   = 0   # frames remaining in post-rep lock
        self._hold_start_t: float = 0.0

    def update(self, smoothed_state: str) -> None:
        """Feed one smoothed state string and update reps / hold_seconds."""
        # ── Dwell gate ────────────────────────────────────────────────────────
        if smoothed_state == self._dwell_cand:
            self._dwell_cnt += 1
        else:
            self._dwell_cand = smoothed_state
            self._dwell_cnt  = 1
            return   # not yet dwelled

        if self._dwell_cnt < _DWELL_FRAMES:
            return   # still building consensus

        self._committed = smoothed_state   # state is now committed

        if self._is_timed:
            self._update_hold(smoothed_state)
        else:
            self._advance_sequence(smoothed_state)

    def reset(self) -> None:
        self.reps         = 0
        self.hold_seconds = 0.0
        self._cursor      = 0
        self._dwell_cand  = "neutral"
        self._dwell_cnt   = 0
        self._committed   = "neutral"
        self._cooldown    = 0
        self._hold_start_t = 0.0

    # ── Internals ──────────────────────────────────────────────────────────────

    def _advance_sequence(self, state: str) -> None:
        if self._seq is None:
            return

        # Count down cooldown
        if self._cooldown > 0:
            self._cooldown -= 1
            return

        expected = self._seq[self._cursor]

        if state == expected:
            self._cursor += 1
            if self._cursor >= len(self._seq):
                # Completed the full sequence → one rep
                self.reps     += 1
                self._cursor   = 0
                self._cooldown = _COOLDOWN_FRAMES
        elif state == self._seq[0] and self._cursor > 0:
            # Soft recovery: user restarted — reset to position 1
            self._cursor = 1

    def _update_hold(self, state: str) -> None:
        now = time.monotonic()
        if state == self._hold_tgt:
            if self._hold_start_t == 0.0:
                self._hold_start_t = now
            else:
                self.hold_seconds = now - self._hold_start_t
        else:
            self._hold_start_t = 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  MediaPipe keypoint extraction
# ══════════════════════════════════════════════════════════════════════════════

def mp_results_to_coco17(
    results,          # mediapipe Pose results object
    frame_w: int,
    frame_h: int,
) -> Optional[np.ndarray]:
    """
    Convert MediaPipe Pose landmark results to a (17, 2) COCO-17 float32 array
    in pixel coordinates.

    Returns None when no pose was detected.
    Landmarks with visibility < _MP_VIS_THRESHOLD are set to (0.0, 0.0) so
    the _is_occluded() sentinel works exactly as during training.
    """
    if results.pose_landmarks is None:
        return None

    lm = results.pose_landmarks.landmark
    kp = np.zeros((17, 2), dtype=np.float32)  # all (0,0) = occluded by default

    for coco_idx, mp_idx in MP_TO_COCO17.items():
        pt = lm[mp_idx]
        if pt.visibility >= _MP_VIS_THRESHOLD:
            kp[coco_idx, 0] = pt.x * frame_w
            kp[coco_idx, 1] = pt.y * frame_h
        # else: leave as (0.0, 0.0) → occluded

    return kp


# ══════════════════════════════════════════════════════════════════════════════
#  OpenCV HUD rendering
# ══════════════════════════════════════════════════════════════════════════════

# Colour palette (BGR)
_C = {
    "neutral":  (160, 160, 160),
    "down":     (60,  80, 220),
    "up":       (60, 200,  60),
    "plank":    (200, 180,  40),
    "jump":     (180,  60, 220),
    "extended": (40,  200, 240),
    "flexed":   (40,  100, 240),
    "white":    (255, 255, 255),
    "black":    (10,   10,  10),
    "red":      (40,   40, 220),
    "green":    (40,  200,  60),
    "bg":       (20,   20,  20),
    "skeleton": (0,   220, 180),
    "joint":    (0,   240, 255),
    "joint_occ":(60,   60,  60),
}

_FONT       = cv2.FONT_HERSHEY_DUPLEX
_FONT_SM    = cv2.FONT_HERSHEY_SIMPLEX
_STATE_BAR_H = 68   # height of top-bar HUD in pixels


def draw_skeleton(
    frame:     np.ndarray,
    kp:        np.ndarray,   # (17, 2)
    state_col: Tuple[int, int, int],
) -> None:
    """Draw COCO-17 skeleton edges and joint dots on the frame in-place."""
    h, w = frame.shape[:2]

    # Edges
    for a, b in _COCO_SKELETON_EDGES:
        pa = (int(kp[a, 0]), int(kp[a, 1]))
        pb = (int(kp[b, 0]), int(kp[b, 1]))
        if pa == (0, 0) or pb == (0, 0):
            continue
        if not (0 <= pa[0] < w and 0 <= pa[1] < h and
                0 <= pb[0] < w and 0 <= pb[1] < h):
            continue
        cv2.line(frame, pa, pb, state_col, 2, cv2.LINE_AA)

    # Joints
    for coco_idx in range(17):
        x, y = int(kp[coco_idx, 0]), int(kp[coco_idx, 1])
        if x == 0 and y == 0:
            continue
        visible = coco_idx in MP_TO_COCO17
        dot_col = _C["joint"] if visible else _C["joint_occ"]
        cv2.circle(frame, (x, y), 5, dot_col, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 5, _C["black"], 1, cv2.LINE_AA)


def draw_top_bar(
    frame:        np.ndarray,
    exercise:     str,
    state_name:   str,
    confidence:   float,
    reps:         int,
    hold_seconds: float,
    is_timed:     bool,
    fps:          float,
    is_warming:   bool,
    is_paused:    bool,
) -> None:
    """
    Render the top-bar HUD:
      LEFT   — exercise name
      CENTRE — reps (or hold timer)
      RIGHT  — state badge + FPS
      BOTTOM — confidence bar + controls hint
    """
    h, w = frame.shape[:2]

    # Semi-transparent dark bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, _STATE_BAR_H), _C["bg"], -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    # Border line
    state_col = _C.get(state_name, _C["white"])
    cv2.rectangle(frame, (0, _STATE_BAR_H - 2), (w, _STATE_BAR_H),
                  state_col, -1)

    ex_display = exercise.replace("_", " ").upper()

    # ── LEFT: exercise name ───────────────────────────────────────────────────
    cv2.putText(frame, ex_display, (14, 44),
                _FONT, 0.72, _C["white"], 1, cv2.LINE_AA)

    # ── CENTRE: rep/hold counter ──────────────────────────────────────────────
    if is_paused:
        count_txt = "PAUSED"
        c_col     = (0, 180, 255)
    elif is_warming:
        count_txt = "WARMING UP..."
        c_col     = (160, 160, 160)
    elif is_timed:
        count_txt = f"HOLD  {hold_seconds:.1f}s"
        c_col     = _C["white"]
    else:
        count_txt = f"REPS  {reps}"
        c_col     = _C["white"]

    (cw, _), _ = cv2.getTextSize(count_txt, _FONT, 1.0, 2)
    cx = (w - cw) // 2
    cv2.putText(frame, count_txt, (cx, 50), _FONT, 1.0, c_col, 2, cv2.LINE_AA)

    # ── RIGHT: state badge + FPS ──────────────────────────────────────────────
    state_display = state_name.upper()
    (sw, _), _ = cv2.getTextSize(state_display, _FONT_SM, 0.65, 2)
    sx = w - sw - 14
    cv2.putText(frame, state_display, (sx, 36),
                _FONT_SM, 0.65, state_col, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS {fps:.0f}", (w - 75, 58),
                _FONT_SM, 0.42, (130, 130, 130), 1, cv2.LINE_AA)

    # ── Confidence bar (thin strip at bottom of HUD) ──────────────────────────
    bar_w   = w - 28
    bar_x   = 14
    bar_y   = _STATE_BAR_H - 10
    bar_h   = 5
    fill_w  = int(confidence * bar_w)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (50, 50, 50), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                  state_col, -1)


def draw_bottom_hint(frame: np.ndarray) -> None:
    """Draw keyboard controls hint at the very bottom of the frame."""
    h, w = frame.shape[:2]
    hint = "[Q/ESC] Quit   [R] Reset   [P] Pause"
    (hw, hh), _ = cv2.getTextSize(hint, _FONT_SM, 0.48, 1)
    x = (w - hw) // 2
    y = h - 12
    # shadow
    cv2.putText(frame, hint, (x + 1, y + 1),
                _FONT_SM, 0.48, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, hint, (x, y),
                _FONT_SM, 0.48, (180, 180, 180), 1, cv2.LINE_AA)


def draw_debug_panel(
    frame:       np.ndarray,
    feat_vec:    np.ndarray,   # (20,) raw
    raw_state:   str,
    smoothed:    str,
    cursor:      int,
    seq:         Optional[List[str]],
) -> None:
    """
    Optional semi-transparent debug panel (top-right corner).
    Shows raw prediction, smoothed prediction, sequence cursor,
    and top-3 feature values.
    """
    h, w = frame.shape[:2]
    px, py, pw, ph = w - 230, _STATE_BAR_H + 8, 222, 160

    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), (60, 60, 60), 1)

    def put(text: str, dy: int, col=(180, 180, 180), scale=0.38):
        cv2.putText(frame, text, (px + 6, py + dy),
                    _FONT_SM, scale, col, 1, cv2.LINE_AA)

    put("── DEBUG ──", 16, (0, 200, 255), 0.40)
    raw_col  = _C.get(raw_state, _C["white"])
    smo_col  = _C.get(smoothed,  _C["white"])
    put(f"Raw   : {raw_state}", 34, raw_col)
    put(f"Smooth: {smoothed}", 52, smo_col)

    if seq:
        seq_str = " → ".join(
            f"[{s}]" if i == cursor else s
            for i, s in enumerate(seq)
        )
        # wrap long sequences
        put(f"Cursor: {cursor}/{len(seq)}", 70)
        put(seq_str[:35], 86, (140, 140, 140))
    else:
        put("TIMED EXERCISE", 70, (140, 140, 140))

    # Show 4 most informative base features
    FEAT_LABELS = [
        "L_Knee", "R_Knee", "L_Hip", "R_Hip",
        "L_Elbow", "R_Elbow", "L_Shoulder", "R_Shoulder",
        "BBox_AR", "NoseAnkle_Y",
    ]
    put("── Features ──", 104, (0, 200, 255), 0.38)
    for row, (label, val) in enumerate(
        zip(FEAT_LABELS[:4], feat_vec[:4])
    ):
        put(f"{label:<12}: {val:6.1f}°", 120 + row * 12)


# ══════════════════════════════════════════════════════════════════════════════
#  FPS counter
# ══════════════════════════════════════════════════════════════════════════════

class FPSCounter:
    """Rolling-window frames-per-second estimate."""

    def __init__(self, window: int = 30) -> None:
        self._ts: Deque[float] = collections.deque(maxlen=window)

    def tick(self) -> float:
        self._ts.append(time.perf_counter())
        if len(self._ts) < 2:
            return 0.0
        return (len(self._ts) - 1) / (self._ts[-1] - self._ts[0])


# ══════════════════════════════════════════════════════════════════════════════
#  Main inference loop
# ══════════════════════════════════════════════════════════════════════════════

def run(args: argparse.Namespace) -> None:
    """Open the video source and run the inference loop."""

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

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt_dir  = Path(args.checkpoint_dir)
    ckpt_path = ckpt_dir / f"dl_model_{args.exercise}.pt"

    if not ckpt_path.exists():
        # Try to give a helpful list of available exercises
        available = sorted(p.stem.replace("dl_model_", "")
                           for p in ckpt_dir.glob("dl_model_*.pt"))
        msg = (f"\n[ERROR]  Checkpoint not found: {ckpt_path}\n"
               f"  Available exercises in {ckpt_dir}: "
               f"{available if available else '(none found)'}\n"
               f"  Train first:  python scripts/train_dl_v1.py")
        print(msg, file=sys.stderr)
        sys.exit(1)

    print(f"[DL Inference]  Loading checkpoint: {ckpt_path}")
    ex_model = ExerciseModel(ckpt_path, device)
    print(f"  Exercise   : {args.exercise}")
    print(f"  Classes    : {[STATE_NAMES[c] for c in ex_model.classes]}")
    print(f"  Val acc    : {ex_model.best_val_acc:.3f}")
    print(f"  Params     : "
          f"{sum(p.numel() for p in ex_model.model.parameters()):,}")

    # ── Optional: load dl_features.json for validation ────────────────────────
    meta_path = ckpt_dir / "dl_features.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        if meta.get("delta_lag") != DELTA_LAG:
            print(f"[WARN]  dl_features.json delta_lag={meta['delta_lag']} "
                  f"but inference uses DELTA_LAG={DELTA_LAG}. "
                  f"Rebuild checkpoint with train_dl_v1.py.")
        print(f"  Metadata   : {meta_path}  (v{meta.get('version','?')})")

    # ── Video capture ─────────────────────────────────────────────────────────
    src_str = args.source
    src     = int(src_str) if src_str.isdigit() else src_str
    cap     = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR]  Cannot open video source: {src_str}", file=sys.stderr)
        sys.exit(1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # reduce latency for live webcam

    
    # ── MediaPipe Pose ─────────────────────────────────────────────────────────
    try:
        import mediapipe as mp
        # استدعاء مباشر ومضمون للموديول
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        mp_pose = mp.solutions.pose
    except Exception as e:
        print(f"[ERROR]  MediaPipe init failed: {e}", file=sys.stderr)
        sys.exit(1)

    pose = mp_pose.Pose(
        model_complexity            = args.mp_complexity,
        smooth_landmarks            = True,
        enable_segmentation         = False,
        min_detection_confidence    = 0.5,
        min_tracking_confidence     = 0.5,
    )

    # ── Stateful objects ───────────────────────────────────────────────────────
    feat_deque  = FeatureDeque(delta_lag=DELTA_LAG)
    smoother    = ModeVoteSmoother(window=args.smooth_k)
    counter     = RepCounterFSM(args.exercise)
    fps_counter = FPSCounter(window=30)

    is_timed  = args.exercise in HOLD_TARGET
    is_paused = False

    # Current display state (updated each frame; seed with "neutral")
    raw_state      = "neutral"
    smoothed_state = "neutral"
    confidence     = 0.0
    last_feat_vec  = np.zeros(N_FEATURES, dtype=np.float32)

    win_title = f"SkeletonNet  —  {args.exercise.replace('_', ' ').title()}"
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)

    print(f"\n[DL Inference]  Running — press [Q] or [ESC] to quit\n")

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            # End of video file or camera disconnected
            break

        if args.flip:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        fps  = fps_counter.tick()

        if not is_paused:
            # ── MediaPipe inference ───────────────────────────────────────────
            # MediaPipe expects RGB
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            kp      = mp_results_to_coco17(results, w, h)

            if kp is not None:
                # ── Feature extraction ────────────────────────────────────────
                base_feat    = calculate_features(kp)      # (10,)
                feat_vec_raw = feat_deque.push_and_compute(base_feat)  # (20,)
                last_feat_vec = feat_vec_raw

                # ── Model prediction ──────────────────────────────────────────
                raw_idx, raw_state, confidence = ex_model.predict(feat_vec_raw)

                # ── Smoothing ─────────────────────────────────────────────────
                smooth_idx     = smoother.push(raw_idx)
                smoothed_state = STATE_NAMES[smooth_idx]

                # ── Rep counting ──────────────────────────────────────────────
                counter.update(smoothed_state)

                # ── Skeleton overlay ──────────────────────────────────────────
                state_col = _C.get(smoothed_state, _C["white"])
                draw_skeleton(frame, kp, state_col)

        # ── HUD ───────────────────────────────────────────────────────────────
        draw_top_bar(
            frame        = frame,
            exercise     = args.exercise,
            state_name   = smoothed_state,
            confidence   = confidence,
            reps         = counter.reps,
            hold_seconds = counter.hold_seconds,
            is_timed     = is_timed,
            fps          = fps,
            is_warming   = not feat_deque.filled,
            is_paused    = is_paused,
        )

        if args.debug and kp is not None:
            draw_debug_panel(
                frame      = frame,
                feat_vec   = last_feat_vec,
                raw_state  = raw_state,
                smoothed   = smoothed_state,
                cursor     = counter._cursor,
                seq        = REP_SEQUENCES.get(args.exercise),
            )

        draw_bottom_hint(frame)

        cv2.imshow(win_title, frame)

        # ── Key handling ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):   # Q or ESC
            break
        elif key in (ord("r"), ord("R")):
            counter.reset()
            feat_deque.reset()
            smoother.reset()
            raw_state      = "neutral"
            smoothed_state = "neutral"
            confidence     = 0.0
            print(f"  [Reset]  Reps and feature buffer cleared.")
        elif key in (ord("p"), ord("P")):
            is_paused = not is_paused
            print(f"  [{'Paused' if is_paused else 'Resumed'}]")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    pose.close()
    cap.release()
    cv2.destroyAllWindows()

    print("\n[DL Inference]  Session complete")
    if is_timed:
        print(f"  Hold time : {counter.hold_seconds:.1f}s")
    else:
        print(f"  Total reps: {counter.reps}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Real-time SkeletonNet inference — MediaPipe Pose + PyTorch MLP\n\n"
            "Examples\n"
            "────────\n"
            "  python scripts/inference_dl.py --exercise squat\n"
            "  python scripts/inference_dl.py --exercise pushup --flip\n"
            "  python scripts/inference_dl.py --exercise squat "
            "--source data/test.mp4\n"
            "  python scripts/inference_dl.py --exercise situp  "
            "--debug --smooth-k 7\n"
        ),
    )
    p.add_argument(
        "--exercise", required=True,
        help="Exercise name, e.g. squat, pushup, situp.  "
             "A checkpoint 'dl_model_<exercise>.pt' must exist in "
             "--checkpoint-dir.",
    )
    p.add_argument(
        "--checkpoint-dir", default="checkpoints",
        help="Directory containing dl_model_<exercise>.pt files "
             "and dl_features.json.  (default: checkpoints)",
    )
    p.add_argument(
        "--source", default="0",
        help="Video source: camera index (0, 1, …) or path to a video file.  "
             "(default: 0)",
    )
    p.add_argument(
        "--flip", action="store_true",
        help="Horizontally flip each frame (mirror mode for selfie cameras).",
    )
    p.add_argument(
        "--smooth-k", type=int, default=5, metavar="K",
        help="Mode-vote smoother window in frames.  "
             "Larger K = more stable but more lag.  (default: 5)",
    )
    p.add_argument(
        "--mp-complexity", type=int, default=1, choices=[0, 1, 2],
        help="MediaPipe model_complexity: 0=fast/less accurate, "
             "1=balanced (default), 2=slow/most accurate.",
    )
    p.add_argument(
        "--device", default="auto",
        help="PyTorch device for model inference: auto | cpu | cuda | mps.  "
             "MediaPipe always runs on CPU.  (default: auto)",
    )
    p.add_argument(
        "--debug", action="store_true",
        help="Overlay a debug panel showing raw prediction, smoothed state, "
             "sequence cursor, and top feature values.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    # Normalise exercise name
    args.exercise = args.exercise.lower().strip()

    if args.exercise not in REP_SEQUENCES:
        known = sorted(REP_SEQUENCES.keys())
        print(
            f"[WARN]  '{args.exercise}' is not in the built-in REP_SEQUENCES "
            f"table.\n"
            f"  Known exercises: {known}\n"
            f"  Continuing — rep counting will be disabled for unknown exercises.",
        )

    # Validate smooth-k
    if args.smooth_k < 1:
        print("[WARN]  --smooth-k must be >= 1.  Defaulting to 1.")
        args.smooth_k = 1

    run(args)


if __name__ == "__main__":
    main()
