"""
scripts/inference_dl_v2.py
═══════════════════════════
Real-Time SkeletonNet v2 Inference  —  MediaPipe Pose + PyTorch MLP

What changed from v1
────────────────────
  • Dynamic state vocabulary — no hardcoded global STATE_NAMES list.
    Each checkpoint carries its own isolated vocab under ckpt["state_names"]
    (e.g. ["sq_up", "sq_descending", "sq_bottom"] for squats).
    ExerciseModel reads and owns this list; nothing outside the class needs
    to know what states exist for a given exercise.

  • ExerciseModel.predict() returns (local_idx, state_name, confidence).
    local_idx is an index into the model's own state_names list — there is
    no global-index remapping because that concept no longer exists.

  • REP_SEQUENCES uses the new v2 isolated state names.
    Only exercises whose checkpoints exist in checkpoints_v2/ are active;
    the rest are commented out until those models are trained.

  • Color palette extended with named colors for every v2 state token so
    the HUD always renders a meaningful color even for unseen exercise states
    (falls back to white only for truly unknown tokens).

  • Checkpoint directory defaults to checkpoints_v2/ (pass
    --checkpoint-dir checkpoints to use v1 models).

  • --checkpoint-dir flag renamed from the v1 default for clarity;
    all other CLI flags are identical.

Feature pipeline (unchanged from v1 — must stay bit-identical to training)
───────────────────────────────────────────────────────────────────────────
  MediaPipe 33 landmarks → COCO-17 subset (13 joints used)
  → calculate_features()  →  (10,) base vector
  → FeatureDeque.push()   →  (20,) [base | delta]  (lag=5)
  → StandardScaler        →  normalised (20,)
  → SkeletonNet forward   →  softmax → state_name

Controls
────────
  [q] / [ESC]  quit
  [r]          reset rep counter + feature deque
  [p]          pause / resume

Usage
─────
  python scripts/inference_dl_v2.py --exercise squat
  python scripts/inference_dl_v2.py --exercise squat --flip --debug
  python scripts/inference_dl_v2.py --exercise squat --source data/test.mp4
  python scripts/inference_dl_v2.py --exercise squat --smooth-k 9
  python scripts/inference_dl_v2.py --exercise squat --checkpoint-dir checkpoints_v2
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════════
#  V2 rep sequences  (isolated per-exercise state names)
#
#  Only exercises with trained v2 checkpoints are active.
#  To add a new exercise: train it, then uncomment / add its entry here.
#
#  Timed exercises → None (hold_seconds accumulate instead of reps).
#  HOLD_TARGET maps timed exercise → the state name that keeps the timer
#  running (must match the single state in that exercise's checkpoint).
# ══════════════════════════════════════════════════════════════════════════════

REP_SEQUENCES: Dict[str, Optional[List[str]]] = {
    # ── Active (v2 checkpoints trained) ──────────────────────────────────────
    "squat":          ["sq_descending", "sq_bottom", "sq_up"],

    # ── Pending (uncomment when checkpoint is trained) ────────────────────────
    # "pushup":         ["pu_descending", "pu_bottom", "pu_up"],
    # "situp":          ["su_ascending",  "su_top",    "su_down"],
    # "pullup":         ["pu_pulling",    "pu_top",    "pu_down"],
    # "lunge":          ["lu_stepping",   "lu_bottom", "lu_up"],
    # "bicep_curl":     ["bc_curling",    "bc_top",    "bc_down"],
    # "tricep_dip":     ["td_descending", "td_bottom", "td_up"],
    # "lateral_raise":  ["lr_raising",    "lr_top",    "lr_down"],
    # "shoulder_press": ["sp_pressing",   "sp_top",    "sp_down"],
    # "jumping_jack":   ["jj_airborne",   "jj_wide",   "jj_standing"],
    # "high_knees":     ["hk_knee_up",    "hk_knee_down"],
    # "mountain_climber":["mc_knee_in",   "mc_knee_flexed"],
    # "burpee":         ["bu_squat_down", "bu_plank",  "bu_squat_up", "bu_jump", "bu_standing"],
    # ── Timed holds ───────────────────────────────────────────────────────────
    # "plank":          None,
    # "wall_sit":       None,
}

HOLD_TARGET: Dict[str, str] = {
    "plank":    "pk_hold",
    "wall_sit": "ws_hold",
}


# ══════════════════════════════════════════════════════════════════════════════
#  Colour palette  (BGR)
#
#  Covers every v2 state token so _C.get(state_name, _C["white"]) always
#  returns a meaningful colour for any trained exercise.
#  Naming convention: state tokens use underscore separators.
# ══════════════════════════════════════════════════════════════════════════════

_C: Dict[str, Tuple[int, int, int]] = {
    # ── UI chrome ─────────────────────────────────────────────────────────────
    "white":     (255, 255, 255),
    "black":     ( 10,  10,  10),
    "bg":        ( 20,  20,  20),
    "joint":     (  0, 240, 255),
    "joint_occ": ( 60,  60,  60),

    # ── Squat ─────────────────────────────────────────────────────────────────
    "sq_up":          (180, 220, 180),   # pale green  — upright / standing
    "sq_descending":  ( 80, 140, 255),   # blue        — going down
    "sq_bottom":      ( 40, 220,  40),   # bright green — full depth

    # ── Pushup ────────────────────────────────────────────────────────────────
    "pu_up":          (180, 220, 180),
    "pu_descending":  ( 80, 140, 255),
    "pu_bottom":      ( 40, 220,  40),

    # ── Situp ─────────────────────────────────────────────────────────────────
    "su_down":        (180, 180, 180),
    "su_ascending":   ( 80, 140, 255),
    "su_top":         ( 40, 220,  40),

    # ── Pullup ────────────────────────────────────────────────────────────────
    "pu_down":        (180, 180, 180),
    "pu_pulling":     ( 80, 140, 255),
    "pu_top":         ( 40, 220,  40),

    # ── Lunge ─────────────────────────────────────────────────────────────────
    "lu_up":          (180, 220, 180),
    "lu_stepping":    ( 80, 140, 255),
    "lu_bottom":      ( 40, 220,  40),

    # ── Bicep curl ────────────────────────────────────────────────────────────
    "bc_down":        (180, 180, 180),
    "bc_curling":     ( 80, 140, 255),
    "bc_top":         ( 40, 220,  40),

    # ── Tricep dip ────────────────────────────────────────────────────────────
    "td_up":          (180, 220, 180),
    "td_descending":  ( 80, 140, 255),
    "td_bottom":      ( 40, 220,  40),

    # ── Lateral raise ─────────────────────────────────────────────────────────
    "lr_down":        (180, 180, 180),
    "lr_raising":     ( 80, 140, 255),
    "lr_top":         ( 40, 220,  40),

    # ── Shoulder press ────────────────────────────────────────────────────────
    "sp_down":        (180, 180, 180),
    "sp_pressing":    ( 80, 140, 255),
    "sp_top":         ( 40, 220,  40),

    # ── Jumping jack ──────────────────────────────────────────────────────────
    "jj_standing":    (180, 180, 180),
    "jj_airborne":    (200,  80, 220),   # magenta — airborne
    "jj_wide":        ( 40, 220,  40),
    "jj_closing":     (255, 160,  40),

    # ── High knees ────────────────────────────────────────────────────────────
    "hk_standing":    (180, 180, 180),
    "hk_knee_up":     ( 40, 220,  40),
    "hk_knee_down":   ( 80, 140, 255),

    # ── Mountain climber ──────────────────────────────────────────────────────
    "mc_plank":       (200, 180,  40),   # gold — hold position
    "mc_knee_in":     ( 80, 140, 255),
    "mc_knee_flexed": ( 40, 220,  40),

    # ── Plank / wall sit (timed) ──────────────────────────────────────────────
    "pk_hold":        (200, 180,  40),
    "ws_hold":        (200, 180,  40),

    # ── Burpee ────────────────────────────────────────────────────────────────
    "bu_standing":    (180, 180, 180),
    "bu_squat_down":  ( 80, 140, 255),
    "bu_plank":       (200, 180,  40),
    "bu_squat_up":    (255, 160,  40),
    "bu_jump":        (200,  80, 220),
}

_FONT        = cv2.FONT_HERSHEY_DUPLEX
_FONT_SM     = cv2.FONT_HERSHEY_SIMPLEX
_STATE_BAR_H = 68


# ══════════════════════════════════════════════════════════════════════════════
#  MediaPipe → COCO-17 index table  (unchanged from v1)
# ══════════════════════════════════════════════════════════════════════════════

MP_TO_COCO17: Dict[int, int] = {
    0:   0,   # nose
    5:  11,   # left_shoulder
    6:  12,   # right_shoulder
    7:  13,   # left_elbow
    8:  14,   # right_elbow
    9:  15,   # left_wrist
    10: 16,   # right_wrist
    11: 23,   # left_hip
    12: 24,   # right_hip
    13: 25,   # left_knee
    14: 26,   # right_knee
    15: 27,   # left_ankle
    16: 28,   # right_ankle
}

_MP_VIS_THRESHOLD: float = 0.5

_COCO_SKELETON_EDGES: List[Tuple[int, int]] = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


# ══════════════════════════════════════════════════════════════════════════════
#  Feature engineering  (bit-identical to train_dl_v2.py — DO NOT modify)
# ══════════════════════════════════════════════════════════════════════════════

DELTA_LAG:  int = 5
ANGLE_DEFS: List[Tuple[str, int, int, int]] = [
    ("L_Knee",     11, 13, 15),
    ("R_Knee",     12, 14, 16),
    ("L_Hip",       5, 11, 13),
    ("R_Hip",       6, 12, 14),
    ("L_Elbow",     5,  7,  9),
    ("R_Elbow",     6,  8, 10),
    ("L_Shoulder", 11,  5,  7),
    ("R_Shoulder", 12,  6,  8),
]
N_ANGLES:   int = len(ANGLE_DEFS)   # 8
N_BASE:     int = N_ANGLES + 2      # 10
N_FEATURES: int = N_BASE * 2        # 20


def _is_occluded(pt: np.ndarray) -> bool:
    return float(pt[0]) == 0.0 and float(pt[1]) == 0.0


def _angle_at_vertex(a: np.ndarray, v: np.ndarray, c: np.ndarray) -> float:
    if _is_occluded(a) or _is_occluded(v) or _is_occluded(c):
        return 0.0
    ba = (a - v).astype(np.float64)
    bc = (c - v).astype(np.float64)
    na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if na < 1e-6 or nc < 1e-6:
        return 0.0
    return float(np.degrees(np.arccos(
        float(np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0))
    )))


def _bbox_aspect_ratio(keypoints: np.ndarray) -> float:
    vis = np.array([kp for kp in keypoints if not _is_occluded(kp)], dtype=np.float32)
    if len(vis) < 2:
        return 0.0
    return float((vis[:, 0].max() - vis[:, 0].min()) /
                 max(vis[:, 1].max() - vis[:, 1].min(), 1e-6))


def _norm_nose_ankle_y(keypoints: np.ndarray) -> float:
    nose = keypoints[0]
    valid_ankles = [a for a in [keypoints[15], keypoints[16]] if not _is_occluded(a)]
    if _is_occluded(nose) or not valid_ankles:
        return 0.0
    ankle_y = float(np.mean([a[1] for a in valid_ankles]))
    vis = np.array([kp for kp in keypoints if not _is_occluded(kp)], dtype=np.float32)
    if len(vis) < 2:
        return 0.0
    return float((ankle_y - float(nose[1])) /
                 max(vis[:, 1].max() - vis[:, 1].min(), 1e-6))


def calculate_features(keypoints: np.ndarray) -> np.ndarray:
    """(17,2) COCO keypoints → (10,) base feature vector."""
    feat = np.zeros(N_BASE, dtype=np.float32)
    for i, (_, a, v, c) in enumerate(ANGLE_DEFS):
        feat[i] = _angle_at_vertex(keypoints[a], keypoints[v], keypoints[c])
    feat[N_ANGLES]     = _bbox_aspect_ratio(keypoints)
    feat[N_ANGLES + 1] = _norm_nose_ankle_y(keypoints)
    return feat


# ══════════════════════════════════════════════════════════════════════════════
#  SkeletonNet  (architecture must match train_dl_v2.py exactly)
# ══════════════════════════════════════════════════════════════════════════════

class SkeletonNet(nn.Module):
    """Input(20)→64→BN→ReLU→Dropout→32→BN→ReLU→num_classes."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),          # disabled at inference; kept for weight compat
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
#  ExerciseModel  — v2 checkpoint loader
#
#  Key changes from v1
#  ───────────────────
#  • Reads self.state_names from ckpt["state_names"] (isolated v2 vocab).
#  • predict() returns (local_idx, state_name, confidence) where state_name
#    comes directly from self.state_names — no global index table needed.
#  • Validates "version" == "dl_v2" to catch accidentally loading a v1 ckpt.
# ══════════════════════════════════════════════════════════════════════════════

class ExerciseModel:
    """Loads a v2 SkeletonNet checkpoint and wraps single-frame inference."""

    def __init__(self, checkpoint_path: Path, device: torch.device) -> None:
        self.device = device

        ckpt = torch.load(str(checkpoint_path), map_location=device,
                          weights_only=False)

        # ── Validate ──────────────────────────────────────────────────────────
        required = {"model_state", "scaler_mean", "scaler_std",
                    "state_names", "n_features", "num_classes"}
        missing  = required - ckpt.keys()
        if missing:
            raise KeyError(
                f"Checkpoint missing keys {missing}.\n"
                f"  Found: {list(ckpt.keys())}\n"
                f"  Is this a v2 checkpoint from train_dl_v2.py?"
            )
        if ckpt.get("version") != "dl_v2":
            ver = ckpt.get("version", "unknown")
            print(f"  [WARN]  Checkpoint version='{ver}' — expected 'dl_v2'.\n"
                  f"          This may be a v1 checkpoint. Proceeding with caution.")
        if int(ckpt["n_features"]) != N_FEATURES:
            raise ValueError(
                f"Checkpoint n_features={ckpt['n_features']} but "
                f"script expects {N_FEATURES}. Rebuild with train_dl_v2.py."
            )

        # ── Scaler ────────────────────────────────────────────────────────────
        self._mean: np.ndarray = ckpt["scaler_mean"].astype(np.float32)
        self._std:  np.ndarray = ckpt["scaler_std"].astype(np.float32)

        # ── Isolated vocabulary  (THE v2 change) ─────────────────────────────
        # state_names is the source of truth — no global index remapping.
        self.state_names: List[str] = [str(s) for s in ckpt["state_names"]]
        self.num_classes: int       = int(ckpt["num_classes"])

        if len(self.state_names) != self.num_classes:
            raise ValueError(
                f"state_names length ({len(self.state_names)}) != "
                f"num_classes ({self.num_classes})"
            )

        # ── Model ─────────────────────────────────────────────────────────────
        self.model = SkeletonNet(self.num_classes).to(device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.best_val_acc: float = float(ckpt.get("best_val_acc", 0.0))
        self.test_acc:     float = float(ckpt.get("test_acc",     0.0))
        self.epochs_run:   int   = int(ckpt.get("epochs_run",     0))

    def predict(self, feat_raw: np.ndarray) -> Tuple[int, str, float]:
        """
        Standardise → forward pass → argmax.

        Returns
        ───────
        local_idx  : int   — index into self.state_names
        state_name : str   — e.g. "sq_bottom"
        confidence : float — softmax probability [0, 1]
        """
        feat_norm = (feat_raw - self._mean) / self._std
        x = torch.tensor(feat_norm, dtype=torch.float32,
                         device=self.device).unsqueeze(0)   # (1, 20)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=-1)[0]  # (num_classes,)
        local_idx  = int(probs.argmax().item())
        confidence = float(probs[local_idx].item())
        return local_idx, self.state_names[local_idx], confidence


# ══════════════════════════════════════════════════════════════════════════════
#  FeatureDeque  (unchanged from v1)
# ══════════════════════════════════════════════════════════════════════════════

class FeatureDeque:
    """Ring buffer of base feature vectors for temporal delta computation."""

    def __init__(self, delta_lag: int = DELTA_LAG) -> None:
        self._lag = delta_lag
        self._buf: Deque[np.ndarray] = collections.deque(maxlen=delta_lag + 1)

    def push_and_compute(self, base: np.ndarray) -> np.ndarray:
        """Append base vector; return (20,) [base | base − oldest_in_buffer]."""
        self._buf.append(base.copy())
        delta = base - self._buf[0]
        return np.concatenate([base, delta]).astype(np.float32)

    def reset(self) -> None:
        self._buf.clear()

    @property
    def filled(self) -> bool:
        return len(self._buf) > self._lag


# ══════════════════════════════════════════════════════════════════════════════
#  ModeVoteSmoother  (unchanged from v1)
# ══════════════════════════════════════════════════════════════════════════════

class ModeVoteSmoother:
    """Majority-vote flicker suppression over a sliding window of predictions."""

    def __init__(self, window: int = 5) -> None:
        self._buf: Deque[int] = collections.deque(maxlen=window)

    def push(self, raw_idx: int) -> int:
        self._buf.append(raw_idx)
        counts: Dict[int, int] = {}
        for v in self._buf:
            counts[v] = counts.get(v, 0) + 1
        best = max(counts.values())
        for v in reversed(self._buf):   # newest-first tie-break
            if counts[v] == best:
                return v
        return raw_idx

    def reset(self) -> None:
        self._buf.clear()


# ══════════════════════════════════════════════════════════════════════════════
#  RepCounterFSM  (v2: works on isolated state name strings)
#
#  Dwell gate:  state must persist _DWELL_FRAMES consecutive smoothed frames
#               before being committed to the sequence cursor.
#  Cooldown:    cursor locked for _COOLDOWN_FRAMES after a rep is counted.
#
#  The FSM is exercise-agnostic — it only knows the sequence of state name
#  strings loaded from REP_SEQUENCES.  Adding a new exercise = one new entry
#  in REP_SEQUENCES above; no FSM code changes needed.
# ══════════════════════════════════════════════════════════════════════════════

_DWELL_FRAMES:    int = 10   # ≈333 ms at 30 fps
_COOLDOWN_FRAMES: int = 45   # ≈1.5 s  at 30 fps


class RepCounterFSM:

    def __init__(self, exercise: str) -> None:
        self.exercise  = exercise
        self._seq      = REP_SEQUENCES.get(exercise)
        self._is_timed = (self._seq is None)
        self._hold_tgt = HOLD_TARGET.get(exercise, "")

        self.reps:         int   = 0
        self.hold_seconds: float = 0.0

        self._cursor:     int   = 0
        self._dwell_cand: str   = ""
        self._dwell_cnt:  int   = 0
        self._cooldown:   int   = 0
        self._hold_start: float = 0.0

    def update(self, state: str) -> None:
        # ── Dwell gate ────────────────────────────────────────────────────────
        if state == self._dwell_cand:
            self._dwell_cnt += 1
        else:
            self._dwell_cand = state
            self._dwell_cnt  = 1
            return
        if self._dwell_cnt < _DWELL_FRAMES:
            return

        # State is committed
        if self._is_timed:
            self._update_hold(state)
        else:
            self._advance(state)

    def reset(self) -> None:
        self.reps         = 0
        self.hold_seconds = 0.0
        self._cursor      = 0
        self._dwell_cand  = ""
        self._dwell_cnt   = 0
        self._cooldown    = 0
        self._hold_start  = 0.0

    def _advance(self, state: str) -> None:
        if self._seq is None:
            return
        if self._cooldown > 0:
            self._cooldown -= 1
            return
        expected = self._seq[self._cursor]
        if state == expected:
            self._cursor += 1
            if self._cursor >= len(self._seq):
                self.reps     += 1
                self._cursor   = 0
                self._cooldown = _COOLDOWN_FRAMES
        elif state == self._seq[0] and self._cursor > 0:
            self._cursor = 1   # soft recovery: user restarted

    def _update_hold(self, state: str) -> None:
        now = time.monotonic()
        if state == self._hold_tgt:
            if self._hold_start == 0.0:
                self._hold_start = now
            else:
                self.hold_seconds = now - self._hold_start
        else:
            self._hold_start = 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  FPS counter  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class FPSCounter:
    def __init__(self, window: int = 30) -> None:
        self._ts: Deque[float] = collections.deque(maxlen=window)

    def tick(self) -> float:
        self._ts.append(time.perf_counter())
        if len(self._ts) < 2:
            return 0.0
        return (len(self._ts) - 1) / (self._ts[-1] - self._ts[0])


# ══════════════════════════════════════════════════════════════════════════════
#  MediaPipe keypoint extraction  (unchanged from v1)
# ══════════════════════════════════════════════════════════════════════════════

def mp_results_to_coco17(results, w: int, h: int) -> Optional[np.ndarray]:
    """MediaPipe Pose results → (17,2) COCO-17 pixel-coord array, or None."""
    if results.pose_landmarks is None:
        return None
    lm = results.pose_landmarks.landmark
    kp = np.zeros((17, 2), dtype=np.float32)
    for coco_idx, mp_idx in MP_TO_COCO17.items():
        pt = lm[mp_idx]
        if pt.visibility >= _MP_VIS_THRESHOLD:
            kp[coco_idx, 0] = pt.x * w
            kp[coco_idx, 1] = pt.y * h
    return kp


# ══════════════════════════════════════════════════════════════════════════════
#  HUD rendering
# ══════════════════════════════════════════════════════════════════════════════

def draw_skeleton(frame: np.ndarray, kp: np.ndarray,
                  state_col: Tuple[int, int, int]) -> None:
    h, w = frame.shape[:2]
    for a, b in _COCO_SKELETON_EDGES:
        pa = (int(kp[a, 0]), int(kp[a, 1]))
        pb = (int(kp[b, 0]), int(kp[b, 1]))
        if pa == (0, 0) or pb == (0, 0):
            continue
        if not (0 <= pa[0] < w and 0 <= pa[1] < h and
                0 <= pb[0] < w and 0 <= pb[1] < h):
            continue
        cv2.line(frame, pa, pb, state_col, 2, cv2.LINE_AA)
    for ci in range(17):
        x, y = int(kp[ci, 0]), int(kp[ci, 1])
        if x == 0 and y == 0:
            continue
        dot = _C["joint"] if ci in MP_TO_COCO17 else _C["joint_occ"]
        cv2.circle(frame, (x, y), 5, dot,        -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 5, _C["black"],  1, cv2.LINE_AA)


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
    h, w    = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, _STATE_BAR_H), _C["bg"], -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    state_col = _C.get(state_name, _C["white"])
    cv2.rectangle(frame, (0, _STATE_BAR_H - 2), (w, _STATE_BAR_H), state_col, -1)

    # LEFT — exercise name
    cv2.putText(frame, exercise.replace("_", " ").upper(),
                (14, 44), _FONT, 0.72, _C["white"], 1, cv2.LINE_AA)

    # CENTRE — rep / hold counter
    if is_paused:
        txt, col = "PAUSED",       (0, 180, 255)
    elif is_warming:
        txt, col = "WARMING UP…",  (160, 160, 160)
    elif is_timed:
        txt, col = f"HOLD  {hold_seconds:.1f}s", _C["white"]
    else:
        txt, col = f"REPS  {reps}",  _C["white"]
    (cw, _), _ = cv2.getTextSize(txt, _FONT, 1.0, 2)
    cv2.putText(frame, txt, ((w - cw) // 2, 50), _FONT, 1.0, col, 2, cv2.LINE_AA)

    # RIGHT — state badge + FPS
    badge = state_name.upper()
    (sw, _), _ = cv2.getTextSize(badge, _FONT_SM, 0.65, 2)
    cv2.putText(frame, badge, (w - sw - 14, 36),
                _FONT_SM, 0.65, state_col, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS {fps:.0f}", (w - 75, 58),
                _FONT_SM, 0.42, (130, 130, 130), 1, cv2.LINE_AA)

    # Confidence bar
    bx, by, bh = 14, _STATE_BAR_H - 10, 5
    bw = w - 28
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh),        (50, 50, 50),  -1)
    cv2.rectangle(frame, (bx, by), (bx + int(confidence * bw), by + bh), state_col, -1)


def draw_bottom_hint(frame: np.ndarray) -> None:
    h, w  = frame.shape[:2]
    hint  = "[Q/ESC] Quit   [R] Reset   [P] Pause"
    (hw, _), _ = cv2.getTextSize(hint, _FONT_SM, 0.48, 1)
    x, y = (w - hw) // 2, h - 12
    cv2.putText(frame, hint, (x + 1, y + 1), _FONT_SM, 0.48, (0,   0,   0), 2, cv2.LINE_AA)
    cv2.putText(frame, hint, (x,     y    ), _FONT_SM, 0.48, (180, 180, 180), 1, cv2.LINE_AA)


def draw_debug_panel(
    frame:      np.ndarray,
    feat_vec:   np.ndarray,
    raw_state:  str,
    smoothed:   str,
    cursor:     int,
    seq:        Optional[List[str]],
) -> None:
    h, w = frame.shape[:2]
    px, py, pw, ph = w - 240, _STATE_BAR_H + 8, 232, 168

    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), (60, 60, 60), 1)

    def put(txt: str, dy: int, col=(180, 180, 180), scale=0.38):
        cv2.putText(frame, txt, (px + 6, py + dy),
                    _FONT_SM, scale, col, 1, cv2.LINE_AA)

    put("── DEBUG ──",         16, (0, 200, 255), 0.40)
    put(f"Raw   : {raw_state}", 34, _C.get(raw_state,  _C["white"]))
    put(f"Smooth: {smoothed}",  52, _C.get(smoothed,   _C["white"]))

    if seq:
        seq_str = " → ".join(f"[{s}]" if i == cursor else s
                             for i, s in enumerate(seq))
        put(f"Cursor: {cursor}/{len(seq)}", 70)
        put(seq_str[:36], 86, (140, 140, 140))
    else:
        put("TIMED EXERCISE", 70, (140, 140, 140))

    BASE_LABELS = ["L_Knee", "R_Knee", "L_Hip", "R_Hip",
                   "L_Elbow", "R_Elbow", "L_Shoulder", "R_Shoulder",
                   "BBox_AR", "NoseAnkle_Y"]
    put("── Features ──", 104, (0, 200, 255), 0.38)
    for row, (label, val) in enumerate(zip(BASE_LABELS[:4], feat_vec[:4])):
        put(f"{label:<12}: {val:6.1f}", 120 + row * 12)


# ══════════════════════════════════════════════════════════════════════════════
#  Main inference loop
# ══════════════════════════════════════════════════════════════════════════════

def run(args: argparse.Namespace) -> None:

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt_dir  = Path(args.checkpoint_dir)
    ckpt_path = ckpt_dir / f"dl_model_{args.exercise}.pt"

    if not ckpt_path.exists():
        available = sorted(p.stem.replace("dl_model_", "")
                           for p in ckpt_dir.glob("dl_model_*.pt"))
        print(
            f"\n[ERROR]  Checkpoint not found: {ckpt_path}\n"
            f"  Available in {ckpt_dir}: "
            f"{available if available else '(none)'}\n"
            f"  Train first:  python scripts/train_dl_v2.py --exercise {args.exercise}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[v2 Inference]  Loading: {ckpt_path}")
    ex_model = ExerciseModel(ckpt_path, device)
    print(f"  Exercise    : {args.exercise}")
    print(f"  States      : {ex_model.state_names}")
    print(f"  Val acc     : {ex_model.best_val_acc:.3f}  "
          f"Test acc: {ex_model.test_acc:.3f}")
    print(f"  Epochs      : {ex_model.epochs_run}")
    print(f"  Params      : {sum(p.numel() for p in ex_model.model.parameters()):,}")

    # Warn if exercise not in REP_SEQUENCES
    if args.exercise not in REP_SEQUENCES:
        print(f"  [WARN] '{args.exercise}' not in REP_SEQUENCES — rep counting disabled.")

    # ── Video capture ─────────────────────────────────────────────────────────
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR]  Cannot open: {args.source}", file=sys.stderr)
        sys.exit(1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # ── YOLO + RTMPose + Kalman (same pipeline as main.py) ────────────────────
    from detector import PersonDetector
    from pose     import PoseEstimator
    from utils    import KalmanKeypoints
    from tracker  import PersonTracker

    detector  = PersonDetector(model_name=args.yolo_model,
                               conf_threshold=args.det_conf)
    pose_est  = PoseEstimator()
    kalman    = KalmanKeypoints()
    tracker   = PersonTracker(iou_threshold=0.35, max_lost_frames=10)
    frame_ctr = 0

    # ── Stateful pipeline objects ─────────────────────────────────────────────
    feat_deque  = FeatureDeque(DELTA_LAG)
    smoother    = ModeVoteSmoother(args.smooth_k)
    counter     = RepCounterFSM(args.exercise)
    fps_counter = FPSCounter(30)

    is_timed  = args.exercise in HOLD_TARGET
    is_paused = False

    first_state    = ex_model.state_names[0]
    raw_state      = first_state
    smoothed_state = first_state
    confidence     = 0.0
    last_feat_vec  = np.zeros(N_FEATURES, dtype=np.float32)
    kp: Optional[np.ndarray] = None

    win = f"SkeletonNet v2  —  {args.exercise.replace('_', ' ').title()}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print(f"\n[v2 Inference]  YOLO+RTMPose — [Q/ESC]=quit  [R]=reset  [P]=pause\n")

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if args.flip:
            frame = cv2.flip(frame, 1)

        h, w   = frame.shape[:2]
        fps    = fps_counter.tick()
        frame_ctr += 1

        if not is_paused:
            # Detection every N frames; tracker coasts between
            run_det = (frame_ctr % args.detect_interval == 0
                       or tracker.state_str == "searching")
            if run_det:
                try:
                    bboxes = detector.detect(frame)
                except Exception:
                    bboxes = []
                track = tracker.update(bboxes)
            else:
                track = tracker.coast()

            kp = None
            if track.person_present:
                try:
                    result = pose_est.estimate_with_scores(frame, track.bbox)
                    if result is not None:
                        kp_raw, _sc = result
                        kp = kalman.smooth(kp_raw)
                except Exception:
                    pass

            if kp is not None:
                base_feat     = calculate_features(kp)
                feat_vec_raw  = feat_deque.push_and_compute(base_feat)
                last_feat_vec = feat_vec_raw

                # ── Model prediction ──────────────────────────────────────────
                raw_idx, raw_state, confidence = ex_model.predict(feat_vec_raw)

                # ── Smoothing ─────────────────────────────────────────────────
                smooth_idx     = smoother.push(raw_idx)
                smoothed_state = ex_model.state_names[smooth_idx]

                # ── Rep counting ──────────────────────────────────────────────
                counter.update(smoothed_state)

                # ── Skeleton overlay ──────────────────────────────────────────
                draw_skeleton(frame, kp, _C.get(smoothed_state, _C["white"]))

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

        if args.debug:
            draw_debug_panel(
                frame     = frame,
                feat_vec  = last_feat_vec,
                raw_state = raw_state,
                smoothed  = smoothed_state,
                cursor    = counter._cursor,
                seq       = REP_SEQUENCES.get(args.exercise),
            )

        draw_bottom_hint(frame)
        cv2.imshow(win, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        elif key in (ord("r"), ord("R")):
            counter.reset()
            feat_deque.reset()
            smoother.reset()
            raw_state      = first_state
            smoothed_state = first_state
            confidence     = 0.0
            print("  [Reset]  Counter and feature buffer cleared.")
        elif key in (ord("p"), ord("P")):
            is_paused = not is_paused
            print(f"  [{'Paused' if is_paused else 'Resumed'}]")

    cap.release()
    cv2.destroyAllWindows()

    print("\n[v2 Inference]  Session complete")
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
            "SkeletonNet v2 real-time inference — MediaPipe + PyTorch MLP\n\n"
            "Examples\n"
            "────────\n"
            "  python scripts/inference_dl_v2.py --exercise squat\n"
            "  python scripts/inference_dl_v2.py --exercise squat --flip --debug\n"
            "  python scripts/inference_dl_v2.py --exercise squat --source data/test.mp4\n"
            "  python scripts/inference_dl_v2.py --exercise squat --smooth-k 9\n"
        ),
    )
    p.add_argument("--exercise",        required=True,
                   help="Exercise name (must match trained checkpoint filename).")
    p.add_argument("--checkpoint-dir",  default="checkpoints_v2",
                   help="Directory containing dl_model_<exercise>.pt files. "
                        "(default: checkpoints_v2)")
    p.add_argument("--source",          default="0",
                   help="Camera index or path to video file. (default: 0)")
    p.add_argument("--flip",            action="store_true",
                   help="Flip frame horizontally (mirror mode).")
    p.add_argument("--smooth-k",        type=int, default=9, metavar="K",
                   help="Mode-vote smoother window in frames. (default: 9)")
    p.add_argument("--yolo-model",      default="yolov8n.pt",
                   help="YOLO model name. (default: yolov8n.pt)")
    p.add_argument("--det-conf",        type=float, default=0.5,
                   help="YOLO detection confidence threshold. (default: 0.5)")
    p.add_argument("--detect-interval", type=int, default=3,
                   help="Run YOLO every N frames; tracker coasts between. "
                        "(default: 3)")
    p.add_argument("--device",          default="auto",
                   help="PyTorch device: auto|cpu|cuda|mps. (default: auto)")
    p.add_argument("--debug",           action="store_true",
                   help="Show debug overlay with raw/smoothed state and feature values.")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    args.exercise = args.exercise.lower().strip()
    if args.smooth_k < 1:
        args.smooth_k = 1
    run(args)


if __name__ == "__main__":
    main()
