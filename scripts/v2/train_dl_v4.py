"""
scripts/train_dl_v3.py
═══════════════════════
Deep Learning v3 — Sliding-Window MLP  (per-exercise micro-models)
 
Why v3
──────
v2 fed the model ONE frame at a time (10 base + 10 delta = 20 features).
That single-frame view cannot distinguish states that look identical in a
snapshot but differ in motion history — e.g. sq_ascending vs sq_standing,
which is exactly where v2 confused the most (see the confusion matrix).
 
v3 fixes this WITHOUT an LSTM and WITHOUT new labels.  Instead of one frame
the model sees a SLIDING WINDOW of the last W frames, flattened into one
vector.  The window itself carries the temporal trend, so the model can tell
"I have been descending for 8 frames" from "I have been standing still".
 
  v2 input : (20,)            = 10 base + 10 delta  (1 frame)
  v3 input : (W × 10,)        = W frames × 10 base features
 
Default W = 12 frames ≈ 0.4 s at 30 fps.  No hand-built deltas — the window
makes them redundant (the model can learn any delta it wants across the W
frames internally).
 
Everything else is inherited from v2
────────────────────────────────────
  ✔  Same 10 base features (8 angles + BBox_AR + NoseAnkle_Y)
  ✔  Same per-exercise isolated state vocabulary
  ✔  Same MERGE_MAP (sq_standing+sq_ascending → sq_up, etc.)
  ✔  Same quality-as-sample-weight mechanism
  ✔  Same video-wise train/val/test split
  ✔  Same on-the-fly keypoint jitter augmentation
  ✔  Same balanced class weights
 
Window construction (critical detail)
───────────────────────────────────────
For frame t the window is base[t-W+1 : t+1].  For t < W-1 we LEFT-PAD by
repeating base[0] so every frame yields a full-length window.  The label for
the window is the state of its LAST frame (frame t) — we are classifying
"what state are we in NOW, given the recent past".
 
This padding convention must match inference exactly, so it is implemented
once in build_window_matrix() and reused by the live engine.
 
Checkpoint format (v3)
──────────────────────
  model_state    : MLP weights
  scaler_mean    : (W*10,) float32   — fit on flattened training windows
  scaler_std     : (W*10,) float32
  state_names    : list[str]         — isolated merged vocab
  window_size    : int               — W (inference must use the same)
  n_base         : int = 10
  n_features     : int = W*10         — flattened input dim
  num_classes    : int
  version        : "dl_v3"
 
Usage
─────
  python scripts/train_dl_v3.py --exercise squat
  python scripts/train_dl_v3.py --exercise squat --window 12 --epochs 150
  python scripts/train_dl_v3.py                       # all exercises found
"""
 
from __future__ import annotations
 
import argparse
import copy
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
#  Base feature engineering  (verbatim from train_dl_v2.py — DO NOT change)
# ══════════════════════════════════════════════════════════════════════════════
 
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
N_ANGLES:   int = len(ANGLE_DEFS)               # 8
SPATIAL_NAMES: List[str] = ["BBox_AR", "NoseAnkle_Y"]
N_BASE:     int = N_ANGLES + len(SPATIAL_NAMES)  # 10
BASE_NAMES: List[str] = [d[0] for d in ANGLE_DEFS] + SPATIAL_NAMES
 
 
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
    cos_a = float(np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))
 
 
def _bbox_aspect_ratio(keypoints: np.ndarray) -> float:
    visible = np.array([kp for kp in keypoints if not _is_occluded(kp)],
                       dtype=np.float32)
    if len(visible) < 2:
        return 0.0
    w = float(visible[:, 0].max() - visible[:, 0].min())
    h = float(visible[:, 1].max() - visible[:, 1].min())
    return float(w / max(h, 1e-6))
 
 
def _norm_nose_ankle_y(keypoints: np.ndarray) -> float:
    nose, l_ankle, r_ankle = keypoints[0], keypoints[15], keypoints[16]
    valid_ankles = [a for a in [l_ankle, r_ankle] if not _is_occluded(a)]
    if _is_occluded(nose) or not valid_ankles:
        return 0.0
    ankle_y = float(np.mean([a[1] for a in valid_ankles]))
    raw_y   = ankle_y - float(nose[1])
    visible = np.array([kp for kp in keypoints if not _is_occluded(kp)],
                       dtype=np.float32)
    if len(visible) < 2:
        return 0.0
    bbox_h = float(visible[:, 1].max() - visible[:, 1].min())
    return float(raw_y / max(bbox_h, 1e-6))
 
 
def calculate_features(keypoints: np.ndarray) -> np.ndarray:
    """(17,2) → (10,) base feature vector."""
    feat = np.zeros(N_BASE, dtype=np.float32)
    for i, (_, a, v, c) in enumerate(ANGLE_DEFS):
        feat[i] = _angle_at_vertex(keypoints[a], keypoints[v], keypoints[c])
    feat[N_ANGLES]     = _bbox_aspect_ratio(keypoints)
    feat[N_ANGLES + 1] = _norm_nose_ankle_y(keypoints)
    return feat
 
 
def build_base_matrix(kp_seq: np.ndarray) -> np.ndarray:
    """(N,17,2) → (N,10) base feature matrix."""
    return np.vstack(
        [calculate_features(kp_seq[f]) for f in range(len(kp_seq))]
    ).astype(np.float32)
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Sliding-window construction  (the heart of v3)
# ══════════════════════════════════════════════════════════════════════════════
 
def build_window_matrix(base_seq: np.ndarray, window: int,
                        use_slope: bool = True) -> np.ndarray:
    """
    Turn an (N, 10) base matrix into a sliding-window matrix.
 
    For frame t the window is base[t-window+1 : t+1] flattened, oldest-first:
        [ base[t-W+1] | base[t-W+2] | ... | base[t] ]
    Frames before index 0 are LEFT-PADDED by repeating base[0].
 
    SLOPE-FEAT (v4): if use_slope, append 10 extra features per row:
        slope = (newest_frame - oldest_frame_in_window) / (window - 1)
    i.e. the mean per-frame velocity of every base feature across the window.
    Velocity is the signal that separates "descending" from "ascending" at the
    same joint angle — handing it to the MLP explicitly instead of asking it to
    learn subtraction from a tiny dataset.  For left-padded early frames the
    oldest slot repeats base[0], so slope ≈ 0 at video start, identical at train
    and inference time.  The live engine MUST mirror this layout exactly
    (see dl_engine.DLEngine.update / _WindowDeque).
 
    Returns (N, window*10 [+10]) float32.
    """
    N = len(base_seq)
    out = np.zeros((N, window * N_BASE), dtype=np.float32)
    for t in range(N):
        for w in range(window):
            src = t - (window - 1) + w     # oldest-first
            src = max(src, 0)              # left-pad with frame 0
            out[t, w * N_BASE:(w + 1) * N_BASE] = base_seq[src]
    if use_slope:
        denom = float(max(window - 1, 1))
        slope = (out[:, -N_BASE:] - out[:, :N_BASE]) / denom
        out   = np.hstack([out, slope]).astype(np.float32)
    return out
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Quality-as-sample-weight  (verbatim from v2)
# ══════════════════════════════════════════════════════════════════════════════
 
DEFAULT_OUTSIDE_WEIGHT: float = 0.5
 
 
def build_sample_weights(
    n_frames:      int,
    rep_starts:    np.ndarray,
    rep_ends:      np.ndarray,
    rep_qualities: np.ndarray,
    outside_weight: float = DEFAULT_OUTSIDE_WEIGHT,
) -> np.ndarray:
    w = np.full(n_frames, outside_weight, dtype=np.float32)
    for s, e, q in zip(rep_starts, rep_ends, rep_qualities):
        s_i = int(max(0, s))
        e_i = int(min(n_frames - 1, e))
        if e_i < s_i:
            continue
        w[s_i:e_i + 1] = (float(q) + 1.0) / 10.0
    return w
 
 
def apply_boundary_downweight(
    weights: np.ndarray,
    states:  np.ndarray,
    radius:  int   = 3,
    factor:  float = 0.3,
) -> np.ndarray:
    """
    BOUNDARY-FIX (v4): down-weight frames near label transitions.
 
    Hand-labeled state boundaries are arbitrary by a few frames — nobody can say
    objectively WHICH frame sq_up ends and sq_descending begins.  Those frames
    carry contradictory supervision across videos and were a dominant error
    source (sq_descending recall 0.25, su_ascending 0.49).  Every frame within
    `radius` of ANY label change gets its sample weight × `factor`: the model
    still sees them but is no longer punished hard for disagreeing with an
    arbitrary boundary.  This also keeps Focal Loss from obsessing over the
    noisiest (boundary) frames, since those are exactly the "hard" examples
    focal would otherwise up-weight.
    """
    if radius <= 0 or factor >= 1.0 or len(states) < 2:
        return weights
    w = weights.copy()
    change_pts = np.flatnonzero(states[1:] != states[:-1]) + 1
    for c in change_pts:
        lo = max(0, int(c) - radius)
        hi = min(len(w), int(c) + radius)
        w[lo:hi] *= factor
    return w
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  State merge map  (verbatim from v2)
# ══════════════════════════════════════════════════════════════════════════════
 
MERGE_MAP: Dict[str, Dict[str, str]] = {
    "squat":          {"sq_standing": "sq_up", "sq_ascending": "sq_up"},
    "pushup":         {"pu_plank": "pu_up", "pu_ascending": "pu_up"},
    "situp":          {"su_lying": "su_down", "su_descending": "su_down"},
    "pullup":         {"pu_hanging": "pu_down", "pu_lowering": "pu_down"},
    "tricep_dip":     {"td_top": "td_up", "td_ascending": "td_up"},
    "shoulder_press": {"sp_rack": "sp_down", "sp_lowering": "sp_down"},
    "bicep_curl":     {"bc_extended": "bc_down", "bc_lowering": "bc_down"},
    "lateral_raise":  {"lr_down": "lr_down", "lr_lowering": "lr_down"},
    "lunge":          {"lu_standing": "lu_up", "lu_returning": "lu_up"},
}
 
 
def apply_merge(
    raw_names:  List[str],
    raw_states: np.ndarray,
    exercise:   str,
) -> Tuple[np.ndarray, List[str]]:
    """Apply MERGE_MAP in memory; return (merged_states, merged_names)."""
    merge = MERGE_MAP.get(exercise, {})
    if not merge:
        return raw_states, raw_names
    mapped = [merge.get(n, n) for n in raw_names]
    seen: List[str] = []
    for n in mapped:
        if n not in seen:
            seen.append(n)
    merged_names = seen
    idx_map = {old_i: merged_names.index(mapped[old_i])
               for old_i in range(len(raw_names))}
    merged_states = np.array([idx_map[int(s)] for s in raw_states],
                             dtype=np.int64)
    return merged_states, merged_names
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ══════════════════════════════════════════════════════════════════════════════
 
class VideoData:
    def __init__(self, vid_key, keypoints, states, weights, n_reps=0):
        self.vid_key   = vid_key
        self.keypoints = keypoints     # (N,17,2)
        self.states    = states        # (N,) merged local indices
        self.weights   = weights       # (N,)
        self.n_frames  = len(keypoints)
        self.n_reps    = n_reps        # labeled rep count (for FSM simulation)
 
 
def load_exercise_videos(
    exercise:      str,
    processed_dir: Path,
    labeled_dir:   Path,
    verbose:       bool = True,
    boundary_radius: int   = 3,
    boundary_factor: float = 0.3,
) -> Tuple[List[VideoData], List[str]]:
    ex_label_dir = labeled_dir / exercise
    if not ex_label_dir.is_dir():
        raise FileNotFoundError(f"No labeled dir: {ex_label_dir}")
 
    videos: List[VideoData] = []
    state_names: Optional[List[str]] = None
 
    for label_npz in sorted(ex_label_dir.glob("*.npz")):
        video_id = label_npz.stem
        vid_key  = f"{exercise}/{video_id}"
        try:
            label = np.load(label_npz, allow_pickle=True)
            if "keypoints" in label:
                kp = label["keypoints"].astype(np.float32)
            else:
                proc_npz = processed_dir / exercise / f"{video_id}.npz"
                if not proc_npz.exists():
                    if verbose:
                        print(f"  [WARN] missing processed: {vid_key}")
                    continue
                kp = np.load(proc_npz, allow_pickle=True)["keypoints"].astype(np.float32)
 
            states = label["frame_states"].astype(np.int64)
            names  = [str(s) for s in label["state_names"]]
            states, names = apply_merge(names, states, exercise)
 
            if state_names is None:
                state_names = names
            elif names != state_names:
                if verbose:
                    print(f"  [WARN] {vid_key}: vocab differs — skipping")
                continue
 
            rep_starts    = label["rep_starts"].astype(np.int64)    if "rep_starts"    in label else np.array([], np.int64)
            rep_ends      = label["rep_ends"].astype(np.int64)      if "rep_ends"      in label else np.array([], np.int64)
            rep_qualities = label["rep_qualities"].astype(np.int64) if "rep_qualities" in label else np.array([], np.int64)
 
            N = min(len(kp), len(states))
            kp, states = kp[:N], states[:N]
            ncls = len(state_names)
            bad = (states < 0) | (states >= ncls)
            if bad.any():
                states[bad] = 0
            weights = build_sample_weights(N, rep_starts, rep_ends, rep_qualities)
            weights = apply_boundary_downweight(
                weights, states, boundary_radius, boundary_factor)
 
            videos.append(VideoData(vid_key, kp, states, weights,
                                    n_reps=len(rep_starts)))
            if verbose:
                print(f"  [load] {vid_key}: {N} frames  "
                      f"{len(rep_starts)} reps  states={np.unique(states).tolist()}")
        except Exception as exc:
            if verbose:
                print(f"  [WARN] failed {vid_key}: {exc}")
 
    if not videos:
        raise RuntimeError(f"No usable videos for {exercise}")
    return videos, state_names
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Video-wise split  (verbatim from v2)
# ══════════════════════════════════════════════════════════════════════════════
 
def split_videos(videos, test_size=0.2, val_size=0.2, seed=42):
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
    test_idx = set(idx[:n_test].tolist())
    val_idx  = set(idx[n_test:n_test + n_val].tolist())
    train, val, test = [], [], []
    for i, v in enumerate(videos):
        if i in test_idx:   test.append(v)
        elif i in val_idx:  val.append(v)
        else:               train.append(v)
    return train, val, test
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Scaler
# ══════════════════════════════════════════════════════════════════════════════
 
class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_  = None
 
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0)
        self.std_  = np.where(self.std_ < 1e-8, 1.0, self.std_)
        return self
 
    def transform(self, X):
        return ((X - self.mean_) / self.std_).astype(np.float32)
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Datasets
# ══════════════════════════════════════════════════════════════════════════════
 
class WindowDataset(Dataset):
    """Static (non-augmented) windowed dataset for VAL and TEST."""
 
    def __init__(self, videos, scaler, window, use_slope=True):
        feats, labels, weights = [], [], []
        for v in videos:
            base = build_base_matrix(v.keypoints)
            win  = build_window_matrix(base, window, use_slope)
            feats.append(win)
            labels.append(v.states)
            weights.append(v.weights)
        self.X = scaler.transform(np.vstack(feats))
        self.y = np.concatenate(labels).astype(np.int64)
        self.w = np.concatenate(weights).astype(np.float32)
 
    def __len__(self):
        return len(self.y)
 
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), int(self.y[i]), float(self.w[i])
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Keypoint-space augmentation helpers (v4.1)
#  Pure functions on (N,17,2) keypoints.  Occluded joints (0,0) are a sentinel
#  and are preserved (never rotated / scaled / interpolated into fake positions).
# ══════════════════════════════════════════════════════════════════════════════

def _visible_mask(kp: np.ndarray) -> np.ndarray:
    """(N,17,2) -> (N,17) bool: True where joint is NOT the (0,0) sentinel."""
    return ~((kp[:, :, 0] == 0.0) & (kp[:, :, 1] == 0.0))


def _centroid(kp: np.ndarray, vis: np.ndarray) -> np.ndarray:
    """(N,17,2),(N,17) -> (N,2) mean of visible joints per frame (safe if none)."""
    cnt = vis.sum(axis=1, keepdims=True).clip(min=1).astype(np.float32)   # (N,1)
    return (kp * vis[..., None]).sum(axis=1) / cnt                        # (N,2)


def aug_rotate(kp: np.ndarray, deg: float) -> np.ndarray:
    """Rotate every visible joint by `deg` about its frame centroid (camera tilt)."""
    vis = _visible_mask(kp)
    cen = _centroid(kp, vis)
    th  = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    d = kp - cen[:, None, :]
    x, y = d[..., 0], d[..., 1]
    out = np.stack([cen[:, None, 0] + (x * c - y * s),
                    cen[:, None, 1] + (x * s + y * c)], axis=-1).astype(np.float32)
    out[~vis] = 0.0
    return out


def aug_scale(kp: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Independently scale x by `sx` and y by `sy` about the frame centroid."""
    vis = _visible_mask(kp)
    cen = _centroid(kp, vis)
    d = kp - cen[:, None, :]
    out = np.stack([cen[:, None, 0] + d[..., 0] * sx,
                    cen[:, None, 1] + d[..., 1] * sy], axis=-1).astype(np.float32)
    out[~vis] = 0.0
    return out


def resample_sequence(kp: np.ndarray, states: np.ndarray, weights: np.ndarray,
                      rate: float):
    """
    TEMPORAL SPEED-WARP (train-time only).  Resample a clip to a new length:
        rate < 1  -> slower -> MORE  frames (stretches the rep; adds transition frames)
        rate > 1  -> faster -> FEWER frames
    Keypoints are linearly interpolated along time; a resampled frame-joint is
    marked occluded (0,0) if EITHER source neighbour was occluded (so we never
    invent a position between a real point and the sentinel).  Labels are taken
    nearest-neighbour (categorical); weights linearly — so the warped copy stays
    correctly supervised.  The window/slope layout is rebuilt afterwards, so the
    model still sees the exact same geometry it will see at inference time.

    Returns (kp_new (M,17,2), states_new (M,), weights_new (M,)).
    """
    N = len(kp)
    if N < 2 or rate <= 0:
        return kp, states, weights
    new_len = max(2, int(round(N / rate)))
    src  = np.linspace(0.0, N - 1, new_len)
    lo   = np.floor(src).astype(np.int64)
    hi   = np.minimum(lo + 1, N - 1)
    frac = (src - lo).astype(np.float32)

    kp_new = (kp[lo] * (1.0 - frac)[:, None, None]
              + kp[hi] * frac[:, None, None]).astype(np.float32)
    occ = (((kp[lo, :, 0] == 0.0) & (kp[lo, :, 1] == 0.0)) |
           ((kp[hi, :, 0] == 0.0) & (kp[hi, :, 1] == 0.0)))
    kp_new[occ] = 0.0

    nn          = np.clip(np.round(src).astype(np.int64), 0, N - 1)
    states_new  = states[nn].astype(np.int64)
    weights_new = (weights[lo] * (1.0 - frac)
                   + weights[hi] * frac).astype(np.float32)
    return kp_new, states_new, weights_new


class AugmentedWindowDataset(Dataset):
    """
    Training dataset with on-the-fly keypoint augmentation, re-rolled every epoch
    via set_epoch().  Frame-dropout is NOT used — the window already provides
    temporal robustness, and dropping frames inside a window would corrupt the
    very trend we want the model to learn.

    v4.1 augmentation stack (all applied to raw keypoints BEFORE feature
    extraction, re-sampled every epoch):

      1. SPEED-WARP  (NEW, the high-value one) — resample the whole clip at a
         rate in [1-speed_warp, 1+speed_warp].  Because the window encodes the
         temporal trend and the slope feature IS the per-frame velocity, this
         teaches "descending slow" and "descending fast" are the SAME state and
         multiplies the frames spent in the under-represented fast transition
         phases (sq_descending / pu_bottom / su_top ...).  states resampled
         nearest-neighbour, weights linearly, so a warped copy stays labeled.
      2. JITTER   — gaussian keypoint noise (detector wobble / sensor noise).
      3. ROTATION (NEW) — small in-plane rotation about the body centroid
         (camera tilt); perturbs angles & bbox AR realistically.
      4. SCALE    (NEW) — independent x/y scaling about the centroid
         (focal length / aspect ratio / body proportions).
      5. HFLIP    — horizontal mirror (fixed 2nd slot; doubles the data).

    Spatial transforms never change length or labels, so they run AFTER the
    speed-warp on the already-resampled (kp, states, weights) triple.  Each
    (video, slot) caches its own resampled windows + labels, and the flat index
    is rebuilt every epoch because warped lengths change.
    """

    # ── COCO-17 left<->right joint swap pairs for horizontal flip ────────────
    # Valid for any bilaterally-symmetric exercise.
    # COCO-17: L_eye=1,R_eye=2  L_ear=3,R_ear=4  L_shoulder=5,R_shoulder=6
    #          L_elbow=7,R_elbow=8  L_wrist=9,R_wrist=10
    #          L_hip=11,R_hip=12  L_knee=13,R_knee=14  L_ankle=15,R_ankle=16
    _FLIP_PAIRS: List[Tuple[int, int]] = [
        (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)
    ]

    def __init__(self, videos, scaler, window,
                 jitter_sigma=2.0, augment=True, flip=True, seed=0,
                 use_slope=True, rot_deg=12.0, scale_jitter=0.08,
                 speed_warp=0.20):
        self.videos       = videos
        self.scaler       = scaler
        self.window       = window
        self.jitter_sigma = float(jitter_sigma)
        self.augment      = augment
        self.flip         = flip
        self.use_slope    = use_slope
        self.rot_deg      = float(rot_deg)
        self.scale_jitter = float(scale_jitter)
        self.speed_warp   = float(speed_warp)
        self._rng         = np.random.default_rng(seed)

        self._n_slots = 2 if (self.augment and self.flip) else 1
        # cache[vi][slot] = (windows (Ns,n_feat), states (Ns,), weights (Ns,))
        self._cache: List[List[Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]] = [
            [None] * self._n_slots for _ in videos
        ]
        # Flat index: (video_idx, slot, frame_idx) — rebuilt each epoch because
        # speed-warp changes the per-slot length.
        self._index: List[Tuple[int, int, int]] = []
        self._build_all()

    def set_epoch(self, epoch):
        self._rng = np.random.default_rng(1000 + epoch)
        self._build_all()

    def _build_all(self):
        self._index = []
        for vi, v in enumerate(self.videos):
            for slot in range(self._n_slots):
                flipped = (slot == 1)
                win, st, wt = self._build_slot(v, flipped)
                self._cache[vi][slot] = (win, st, wt)
                for fi in range(len(st)):
                    self._index.append((vi, slot, fi))

    def _build_slot(self, v, flipped: bool):
        kp      = v.keypoints
        states  = v.states
        weights = v.weights
        if self.augment:
            # 1) speed-warp first (changes length + labels)
            if self.speed_warp > 0:
                rate = float(self._rng.uniform(max(0.1, 1.0 - self.speed_warp),
                                               1.0 + self.speed_warp))
                kp, states, weights = resample_sequence(kp, states, weights, rate)
            # 2) jitter  3) rotate  4) scale  (length / labels unchanged)
            if self.jitter_sigma > 0:
                kp = self._jitter(kp)
            if self.rot_deg > 0:
                deg = float(self._rng.uniform(-self.rot_deg, self.rot_deg))
                kp  = aug_rotate(kp, deg)
            if self.scale_jitter > 0:
                sx = float(self._rng.uniform(1.0 - self.scale_jitter, 1.0 + self.scale_jitter))
                sy = float(self._rng.uniform(1.0 - self.scale_jitter, 1.0 + self.scale_jitter))
                kp = aug_scale(kp, sx, sy)
        if flipped:
            kp = self._hflip(kp)
        base = build_base_matrix(kp)
        win  = build_window_matrix(base, self.window, self.use_slope)
        return (self.scaler.transform(win),
                states.astype(np.int64),
                weights.astype(np.float32))

    def _jitter(self, kp):
        noisy = kp.copy()
        vis = ~((kp[:, :, 0] == 0.0) & (kp[:, :, 1] == 0.0))
        noise = self._rng.normal(0.0, self.jitter_sigma,
                                 size=kp.shape).astype(np.float32)
        noise[~vis] = 0.0
        return noisy + noise

    def _hflip(self, kp: np.ndarray) -> np.ndarray:
        """
        Horizontal mirror of (N,17,2) keypoints.  Features depend only on
        RELATIVE positions (angles, aspect ratio, vertical distance), so we
        simply NEGATE x then swap L<->R joint pairs.  Occluded (0,0) stays (0,0).
        """
        flipped = kp.copy()
        flipped[:, :, 0] = -flipped[:, :, 0]
        for l, r in self._FLIP_PAIRS:
            flipped[:, [l, r]] = flipped[:, [r, l]]
        return flipped

    def __len__(self):
        return len(self._index)

    def __getitem__(self, i):
        vi, slot, fi = self._index[i]
        win, st, wt = self._cache[vi][slot]
        return (torch.from_numpy(win[fi]),
                int(st[fi]),
                float(wt[fi]))

 
# ══════════════════════════════════════════════════════════════════════════════
#  Model — Sliding-Window MLP
# ══════════════════════════════════════════════════════════════════════════════
 
class WindowNet(nn.Module):
    """
    MLP over a flattened window.
      Input(W*10) → Linear(128) → BN → ReLU → Dropout
                  → Linear(64)  → BN → ReLU → Dropout
                  → Linear(num_classes)
 
    Wider than v2 (128/64 vs 64/32) because the input is larger (W*10 vs 20),
    but still small enough for ~1-2k windows per exercise.
    """
 
    def __init__(self, n_features: int, num_classes: int,
                 dropout: float = 0.4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
 
    def forward(self, x):
        return self.net(x)
 
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Class weights  (verbatim from v2)
# ══════════════════════════════════════════════════════════════════════════════
 
def compute_class_weights(y, num_classes, device):
    counts  = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts  = np.where(counts < 1.0, 1.0, counts)
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.0, 10.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Focal Loss  (v4 — applied to ALL exercises)
#
#  FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
#  gamma>0 down-weights easy, well-classified frames (the long stationary
#  states) so gradient focus shifts to the hard transition frames.  alpha_t is
#  the per-class inverse-frequency weight.  reduction='none' so the quality
#  sample-weights still multiply in exactly like CrossEntropy did.
#  Paired with boundary down-weighting so focal targets genuinely-hard frames,
#  not mislabeled boundary frames.
# ══════════════════════════════════════════════════════════════════════════════
 
class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 1.5) -> None:
        super().__init__()
        self.alpha = alpha          # (num_classes,) on device
        self.gamma = float(gamma)
 
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce    = nn.functional.cross_entropy(
                    logits, targets, weight=None, reduction="none")   # (B,)
        pt    = torch.exp(-ce)
        alpha = self.alpha[targets]
        return alpha * (1.0 - pt) ** self.gamma * ce                  # (B,)
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Rep anchors  (KEEP IN SYNC with dl_engine._REP_ANCHORS)  &  window defaults
# ══════════════════════════════════════════════════════════════════════════════
 
REP_ANCHORS: Dict[str, Tuple[str, str]] = {   # exercise -> (open, closed)
    "squat":          ("sq_up",       "sq_bottom"),
    "pushup":         ("pu_up",       "pu_bottom"),
    "situp":          ("su_down",     "su_top"),
    "lateral_raise":  ("lr_down",     "lr_top"),
    "tricep_dip":     ("td_up",       "td_bottom"),
    # ── now wired up (open = resting posture, closed = peak) ──
    "pullup":         ("pu_down",     "pu_top"),
    "bicep_curl":     ("bc_down",     "bc_top"),
    "shoulder_press": ("sp_down",     "sp_top"),
    "lunge":          ("lu_up",       "lu_bottom"),
    "jumping_jack":   ("jj_standing", "jj_wide"),
}
HOLD_TARGET: Dict[str, str] = {"plank": "pk_hold", "wall_sit": "ws_hold"}

# ══════════════════════════════════════════════════════════════════════════════
#  PER-EXERCISE TRAINING CONFIG  (baked-in "best" knobs — no flags needed)
#
#  Run `python train_dl_v4.py`            → trains every exercise with its config
#  Run `python train_dl_v4.py --exercise squat`  → just squat, with its config
#  Any flag you DO pass on the CLI overrides the config for that run.
#  Keys map 1:1 to the resolved args in effective_args():
#      window, epochs, patience, focal_gamma, sim_smooth_k, require_closed
#  (window omitted -> 0 = auto/data-driven via resolve_window + DEFAULT_WINDOWS)
# ══════════════════════════════════════════════════════════════════════════════

PER_EXERCISE_TRAIN: Dict[str, Dict] = {
    # ── rep exercises: form-gated counting, focal gamma=1.5 ──
    "squat":          dict(window=13, epochs=150, patience=25, focal_gamma=1.5,
                           sim_smooth_k=5, require_closed=True),
    "pushup":         dict(epochs=150, patience=25, focal_gamma=1.5,
                           sim_smooth_k=5, require_closed=True),
    "situp":          dict(epochs=150, patience=25, focal_gamma=1.5,
                           sim_smooth_k=5, require_closed=True),
    "lunge":          dict(epochs=150, patience=25, focal_gamma=1.5,
                           sim_smooth_k=5, require_closed=True),
    "bicep_curl":     dict(epochs=150, patience=25, focal_gamma=1.5,
                           sim_smooth_k=5, require_closed=True),
    "lateral_raise":  dict(epochs=150, patience=25, focal_gamma=1.5,
                           sim_smooth_k=5, require_closed=True),
    "shoulder_press": dict(epochs=150, patience=25, focal_gamma=1.5,
                           sim_smooth_k=5, require_closed=True),
    "jumping_jack":   dict(epochs=150, patience=25, focal_gamma=1.5,
                           sim_smooth_k=5, require_closed=True),
    # ── harder rep exercises: longer train + stronger focal gamma=2.0 ──
    "pullup":         dict(epochs=180, patience=30, focal_gamma=2.0,
                           sim_smooth_k=5, require_closed=True),
    "tricep_dip":     dict(epochs=180, patience=30, focal_gamma=2.0,
                           sim_smooth_k=5, require_closed=True),
    # ── timed holds: no reps, duration-gated, gentle focal gamma=1.0, longer smoothing ──
    "plank":          dict(epochs=120, patience=20, focal_gamma=1.0,
                           sim_smooth_k=7, require_closed=False),
    "wall_sit":       dict(epochs=120, patience=20, focal_gamma=1.0,
                           sim_smooth_k=7, require_closed=False),
}


def effective_args(exercise: str, args):
    """Merge CLI flags with PER_EXERCISE_TRAIN.

    A flag the user passed explicitly (non-None) always wins; otherwise the
    per-exercise default is used, falling back to a global default.  This is
    what lets you run the script with no flags (or just --exercise X) and still
    get the right epochs / patience / focal-gamma / window / smoothing /
    rep-gating per exercise.
    """
    eff = copy.copy(args)
    cfg = PER_EXERCISE_TRAIN.get(exercise, {})

    def pick(attr, glob):
        cur = getattr(args, attr)
        return cur if cur is not None else cfg.get(attr, glob)

    eff.epochs         = pick("epochs",         120)
    eff.patience       = pick("patience",       20)
    eff.focal_gamma    = pick("focal_gamma",    1.5)
    eff.sim_smooth_k   = pick("sim_smooth_k",   5)
    eff.window         = pick("window",         0)       # 0 -> resolve_window auto
    eff.require_closed = pick("require_closed", False)
    return eff
 
# Per-exercise default window (frames @ ~30 fps).  Rule of thumb:
# W ≈ ¼ of a median rep, clamped to [7,13] — captures one phase with margin
# without blurring across a whole rep.  Fast/short exercises → small window.
DEFAULT_WINDOWS: Dict[str, int] = {
    "wall_sit": 8, "plank": 8,
    "jumping_jack": 8, "high_knees": 8, "mountain_climber": 8,
    "squat": 10, "lunge": 10, "lateral_raise": 10,
    "pushup": 10, "bicep_curl": 10, "shoulder_press": 10, "situp": 10,
    "tricep_dip": 12, "pullup": 12,
    "burpee": 14,
}
 
 
def median_rep_frames(videos: List["VideoData"]) -> Optional[float]:
    """Median rep duration (frames) inferred from per-video reps & length."""
    spans = []
    for v in videos:
        if v.n_reps and v.n_reps > 0:
            spans.append(v.n_frames / v.n_reps)   # rough: frames per rep
    if not spans:
        return None
    return float(np.median(spans))
 
 
def resolve_window(exercise: str, videos: List["VideoData"],
                   requested: int) -> int:
    """requested>0 → honour it.  ==0 → DEFAULT_WINDOWS, else data-driven /4."""
    suggested = DEFAULT_WINDOWS.get(exercise)
    mrf = median_rep_frames(videos)
    data_w = int(np.clip(round((mrf or 40) / 4.0), 7, 13))
    if requested and requested > 0:
        chosen = requested
    elif suggested is not None:
        chosen = suggested
    else:
        chosen = data_w
    if mrf is not None:
        print(f"  [window] chosen={chosen}  (median rep ≈ {mrf:.0f} frames → "
              f"data-suggested {data_w}; default {suggested})")
    else:
        print(f"  [window] chosen={chosen}  (timed/no-rep exercise; default {suggested})")
    return chosen
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Label audit  (timed exercises — REPORT ONLY, never alters data)
# ══════════════════════════════════════════════════════════════════════════════
 
def audit_timed_labels(exercise: str, videos: List["VideoData"],
                       state_names: List[str]) -> None:
    """wall_sit hold/neutral is separable by knee angle; plank by bbox AR.  A
    label that contradicts the geometry is almost certainly a labeling slip.
    Prints a per-video disagreement count so noisy videos can be re-labeled."""
    tgt = HOLD_TARGET.get(exercise)
    if tgt is None or tgt not in state_names:
        return
    hold_idx = state_names.index(tgt)
    print(f"  ── LABEL AUDIT ({exercise}) ──")
    total = 0
    for v in videos:
        base = build_base_matrix(v.keypoints)
        flags = 0
        for t in range(v.n_frames):
            is_hold = int(v.states[t]) == hold_idx
            if exercise == "wall_sit":
                knees = [a for a in (base[t, 0], base[t, 1]) if a > 0.0]
                if not knees:
                    continue
                knee = float(np.mean(knees))
                if is_hold and knee >= 145.0:
                    flags += 1
                elif (not is_hold) and 0.0 < knee <= 120.0:
                    flags += 1
            else:  # plank
                ar = float(base[t, N_ANGLES])
                if ar <= 0.0:
                    continue
                if is_hold and ar < 0.75:
                    flags += 1
                elif (not is_hold) and ar > 1.30:
                    flags += 1
        total += flags
        if flags:
            print(f"    [AUDIT] {v.vid_key}: {flags} frames "
                  f"({flags / max(v.n_frames,1)*100:.0f}%) contradict geometry")
    if total == 0:
        print("    labels consistent with geometry ✓")
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Anchor rep-counter SIMULATION  (MIRRORS dl_engine._RepCounterFSM — keep sync)
#
#  Frame accuracy is the wrong metric: the product counts reps / times holds.
#  This replays each test video through the same ModeVote smoother + anchor
#  hysteresis the live engine uses, and reports predicted vs labeled reps.
# ══════════════════════════════════════════════════════════════════════════════
 
_SIM_OPEN_DWELL   = 5    # must match dl_engine._OPEN_DWELL
_SIM_AWAY_DWELL   = 2    # must match dl_engine._AWAY_DWELL
_SIM_COOLDOWN     = 18   # must match dl_engine._COOLDOWN_FRAMES
_SIM_AWAY_MIN     = 8    # must match dl_engine._AWAY_MIN_NOCLOSED
 
 
def _mode_vote(raw: List[int], k: int) -> List[int]:
    buf: List[int] = []
    out: List[int] = []
    for r in raw:
        buf.append(r)
        if len(buf) > k:
            buf.pop(0)
        counts: Dict[int, int] = {}
        for x in buf:
            counts[x] = counts.get(x, 0) + 1
        best = max(counts.values())
        pick = r
        for x in reversed(buf):
            if counts[x] == best:
                pick = x
                break
        out.append(pick)
    return out
 
 
def _anchor_count(state_seq: List[str], open_a: str, closed_a: str,
                  open_dwell=_SIM_OPEN_DWELL, away_dwell=_SIM_AWAY_DWELL,
                  cooldown_frames=_SIM_COOLDOWN, away_min=_SIM_AWAY_MIN,
                  require_closed=False) -> int:
    """Exact replica of dl_engine._RepCounterFSM._advance (rep path)."""
    reps = 0
    last_zone = -1; zone_run = 0
    armed = False; away = False; away_len = 0; saw_closed = False; cooldown = 0
    for s in state_seq:
        if cooldown > 0:
            cooldown -= 1
        zone = 0 if s == open_a else (2 if s == closed_a else 1)
        if zone == last_zone:
            zone_run += 1
        else:
            last_zone = zone; zone_run = 1
        if not armed:
            if zone == 0 and zone_run >= open_dwell:
                armed = True; away = False
            continue
        if not away:
            if zone != 0 and zone_run >= away_dwell:
                away = True; away_len = zone_run; saw_closed = (zone == 2)
        else:
            if zone != 0:   # count only non-OPEN frames (mirrors engine fix)
                away_len += 1
            if zone == 2:
                saw_closed = True
            if zone == 0 and zone_run >= open_dwell:
                valid = saw_closed or (not require_closed and away_len >= away_min)
                if valid and cooldown == 0:
                    reps += 1; cooldown = cooldown_frames
                away = False; saw_closed = False; away_len = 0
    return reps
 
 
@torch.no_grad()
def fsm_sim_report(model, scaler, test_videos, window, use_slope,
                   state_names, device, exercise, smooth_k=5,
                   require_closed=False) -> None:
    if not test_videos:
        return
    model.eval()
    is_timed = exercise in HOLD_TARGET
    anchors  = REP_ANCHORS.get(exercise)
    print(f"\n  ── FSM SIMULATION (what the user actually sees) ──")
    if not is_timed and anchors is None:
        print(f"    [note] no anchors defined for '{exercise}' — add to REP_ANCHORS")
        return
    tot_pred = tot_true = 0
    for v in test_videos:
        base = build_base_matrix(v.keypoints)
        win  = scaler.transform(build_window_matrix(base, window, use_slope))
        logits = model(torch.tensor(win, dtype=torch.float32, device=device))
        raw    = logits.argmax(1).cpu().numpy().tolist()
        acc    = float((np.array(raw) == v.states[:len(raw)]).mean())
        smoothed = [state_names[i] for i in _mode_vote(raw, smooth_k)]
 
        if is_timed:
            tgt       = HOLD_TARGET[exercise]
            hold_pred = sum(1 for s in smoothed if s == tgt)
            hold_true = int((v.states == state_names.index(tgt)).sum())
            print(f"    {v.vid_key}: frame-acc {acc*100:.0f}% | "
                  f"hold {hold_pred}/{hold_true} ({hold_pred/max(hold_true,1)*100:.0f}%)")
        else:
            reps = _anchor_count(smoothed, anchors[0], anchors[1],
                                 require_closed=require_closed)
            tot_pred += reps; tot_true += v.n_reps
            mark = "✓" if reps == v.n_reps else "✗"
            print(f"    {v.vid_key}: frame-acc {acc*100:.0f}% | "
                  f"reps {reps}/{v.n_reps}  {mark}")
    if not is_timed and tot_true:
        err = abs(tot_pred - tot_true) / tot_true * 100
        print(f"    ── total reps {tot_pred}/{tot_true}  "
              f"(|error| {err:.0f}% across test videos) ──")
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Train one exercise
# ══════════════════════════════════════════════════════════════════════════════
 
def train_one_exercise(exercise, videos, state_names, device, args):
    num_classes = len(state_names)
 
    if len(videos) < 3:
        print(f"  [SKIP] {exercise}: only {len(videos)} videos")
        return None
 
    # LABEL AUDIT for timed exercises (report only).
    if exercise in HOLD_TARGET:
        audit_timed_labels(exercise, videos, state_names)
 
    window     = resolve_window(exercise, videos, args.window)
    use_slope  = not args.no_slope
    n_features = window * N_BASE + (N_BASE if use_slope else 0)
 
    train_v, val_v, test_v = split_videos(
        videos, args.test_size, args.val_size, args.seed)
    print(f"  Split: {len(train_v)} train / {len(val_v)} val / {len(test_v)} test")
 
    # Fit scaler on clean training windows
    train_wins = []
    for v in train_v:
        base = build_base_matrix(v.keypoints)
        train_wins.append(build_window_matrix(base, window, use_slope))
    scaler = StandardScaler().fit(np.vstack(train_wins))
 
    train_ds = AugmentedWindowDataset(
        train_v, scaler, window,
        jitter_sigma=args.jitter_sigma,
        augment=not args.no_augment,
        flip=not args.no_flip,
        seed=args.seed,
        use_slope=use_slope,
        rot_deg=args.rot_deg,
        scale_jitter=args.scale_jitter,
        speed_warp=args.speed_warp)
    val_ds  = WindowDataset(val_v,  scaler, window, use_slope)
    test_ds = WindowDataset(test_v, scaler, window, use_slope)
 
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False)
 
    y_train = np.concatenate([v.states for v in train_v]).astype(np.int64)
    class_w = compute_class_weights(y_train, num_classes, device)
 
    # TRANSITION-BOOST (experimental, default off): up-weight the transition
    # class (the state that is neither open nor closed anchor).  We saw a naive
    # neutral boost HURT wall_sit earlier, so this is opt-in via --transition-boost.
    if args.transition_boost != 1.0 and exercise in REP_ANCHORS:
        o, c = REP_ANCHORS[exercise]
        for i, nm in enumerate(state_names):
            if nm not in (o, c):
                class_w[i] = class_w[i] * float(args.transition_boost)
        print(f"  [transition-boost] ×{args.transition_boost:.1f} on transition class")
 
    torch.manual_seed(args.seed)
    model     = WindowNet(n_features, num_classes, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=max(args.patience // 3, 3), min_lr=1e-6)
 
    # FOCAL LOSS (v4): applied to ALL exercises.  gamma focuses gradient on hard
    # (transition) frames; alpha = inverse-frequency class weights.
    if args.focal_gamma > 0:
        criterion = FocalLoss(alpha=class_w, gamma=args.focal_gamma)
        print(f"  [loss] FocalLoss γ={args.focal_gamma}  (alpha=class weights)")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_w, reduction="none")
        print(f"  [loss] CrossEntropy (focal disabled)")
 
    best_val_loss = math.inf
    best_val_acc  = 0.0
    best_state    = None
    no_improve    = 0
 
    print(f"  Model: {model.count_parameters():,} params | "
          f"window={window} ({n_features} feat) | classes={state_names}")
    header = f"  {'Ep':>4}  {'TrLoss':>8}  {'TrAcc':>7}  {'VaLoss':>8}  {'VaAcc':>7}  {'LR':>8}"
    print(header)
    print("  " + "─" * (len(header) - 2))
 
    epoch = 0
    for epoch in range(1, args.epochs + 1):
        if not args.no_augment:
            train_ds.set_epoch(epoch)
 
        model.train()
        tr_loss = tr_correct = tr_total = 0
        for Xb, yb, wb in train_loader:
            # BatchNorm needs >1 sample in train mode.  Speed-warp makes the
            # per-epoch window count vary, so the last mini-batch can land on
            # exactly 1 window — skip it (loses <=1 window/epoch, reshuffled
            # next epoch) rather than crashing _verify_batch_size.
            if Xb.size(0) <= 1:
                continue
            Xb, yb, wb = Xb.to(device), yb.to(device), wb.to(device)
            logits   = model(Xb)
            per_samp = criterion(logits, yb)
            loss     = (per_samp * wb).sum() / (wb.sum() + 1e-8)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss    += float(loss.item()) * Xb.size(0)
            tr_correct += int((logits.argmax(1) == yb).sum().item())
            tr_total   += Xb.size(0)
        tr_loss_avg = tr_loss / max(tr_total, 1)
        tr_acc      = tr_correct / max(tr_total, 1)
 
        model.eval()
        va_loss = va_correct = va_total = 0
        with torch.no_grad():
            for Xb, yb, wb in val_loader:
                Xb, yb, wb = Xb.to(device), yb.to(device), wb.to(device)
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
            print(f"  Early stop @ epoch {epoch}")
            break
 
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
 
    test_acc = evaluate(model, test_loader, state_names, device, exercise)
 
    # FSM SIMULATION — the metric that matches the product (reps / hold time).
    fsm_sim_report(model, scaler, test_v, window, use_slope,
                   state_names, device, exercise,
                   smooth_k=args.sim_smooth_k,
                   require_closed=args.require_closed)
 
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"dl_model_{exercise}.pt"
    torch.save({
        "model_state":  model.state_dict(),
        "scaler_mean":  scaler.mean_.astype(np.float32),
        "scaler_std":   scaler.std_.astype(np.float32),
        "state_names":  state_names,
        "window_size":  window,
        "n_base":       N_BASE,
        "n_features":   n_features,
        "use_slope":    use_slope,
        "num_classes":  num_classes,
        "best_val_acc": float(best_val_acc),
        "test_acc":     float(test_acc),
        "epochs_run":   epoch,
        "version":      "dl_v4",
    }, ckpt_path)
    print(f"  Saved → {ckpt_path}")
 
    return {
        "exercise":     exercise,
        "state_names":  state_names,
        "window_size":  window,
        "best_val_acc": round(best_val_acc, 4),
        "test_acc":     round(test_acc, 4),
        "epochs_run":   epoch,
    }
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation
# ══════════════════════════════════════════════════════════════════════════════
 
@torch.no_grad()
def evaluate(model, loader, state_names, device, exercise):
    model.eval()
    y_true, y_pred = [], []
    for Xb, yb, _ in loader:
        logits = model(Xb.to(device))
        y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
        y_true.extend(yb.numpy().tolist())
    if not y_true:
        print("  [WARN] empty test set")
        return 0.0
    y_true_a, y_pred_a = np.array(y_true), np.array(y_pred)
    present = sorted(set(y_true_a.tolist()) | set(y_pred_a.tolist()))
    labels  = [state_names[i] for i in present]
 
    print(f"\n  ── TEST REPORT: {exercise} ──")
    print(classification_report(y_true_a, y_pred_a, labels=present,
                                target_names=labels, zero_division=0))
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
 
def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="DL v3 — Sliding-Window MLP per-exercise micro-models")
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--labeled-dir",   default="data/labeled_v2")
    p.add_argument("--out-dir",       default="checkpoints_v3")
    p.add_argument("--exercise",      default=None)
    p.add_argument("--window",        type=int,   default=None,
                   help="Sliding window in frames. Omit = per-exercise config "
                        "(PER_EXERCISE_TRAIN); 0 = auto/data-driven; a number "
                        "forces it for every exercise.")
    p.add_argument("--test-size",     type=float, default=0.20)
    p.add_argument("--val-size",      type=float, default=0.20)
    p.add_argument("--epochs",        type=int,   default=None,
                   help="Override epochs for every exercise. "
                        "Omit = per-exercise config (else 120).")
    p.add_argument("--batch-size",    type=int,   default=256)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--dropout",       type=float, default=0.4)
    p.add_argument("--patience",      type=int,   default=None,
                   help="Override early-stop patience. "
                        "Omit = per-exercise config (else 20).")
    p.add_argument("--jitter-sigma",  type=float, default=2.0)
    p.add_argument("--no-augment",    action="store_true")
    p.add_argument("--no-flip",       action="store_true",
                   help="Disable horizontal flip augmentation (default: enabled)")
    p.add_argument("--rot-deg",       type=float, default=12.0,
                   help="Augment: max in-plane rotation in degrees (0=off). "
                        "Simulates camera tilt.")
    p.add_argument("--scale-jitter",  type=float, default=0.08,
                   help="Augment: max anisotropic x/y scale jitter (0=off). "
                        "Simulates focal length / body proportions.")
    p.add_argument("--speed-warp",    type=float, default=0.20,
                   help="Augment: max temporal speed-warp fraction; rate sampled "
                        "in [1-w, 1+w] (0=off). The high-value one for small data.")
    # ── v4 knobs ───────────────────────────────────────────────────────────────
    p.add_argument("--no-slope",      action="store_true",
                   help="Disable the 10 slope/velocity features (default: enabled)")
    p.add_argument("--focal-gamma",   type=float, default=None,
                   help="Focal-loss gamma. Omit = per-exercise config "
                        "(1.5 reps / 2.0 hard / 1.0 holds). 0 = plain CE.")
    p.add_argument("--boundary-radius", type=int, default=3,
                   help="Frames around each label transition to down-weight")
    p.add_argument("--boundary-factor", type=float, default=0.3,
                   help="Sample-weight multiplier for boundary frames (1.0=off)")
    p.add_argument("--transition-boost", type=float, default=1.0,
                   help="Experimental: extra class weight on the transition "
                        "state (1.0=off; >1 up-weights). Use with care.")
    p.add_argument("--sim-smooth-k",  type=int, default=None,
                   help="ModeVote window used in the FSM simulation (match the "
                        "engine). Omit = per-exercise config (5 reps / 7 holds).")
    p.add_argument("--require-closed", dest="require_closed",
                   action="store_true", default=None,
                   help="FSM sim: only count a rep if the closed anchor was hit "
                        "(stricter, form-gated).")
    p.add_argument("--duration-gated", dest="require_closed",
                   action="store_false",
                   help="FSM sim: count on a long-enough excursion even without "
                        "the closed anchor. Omit both = per-exercise config "
                        "(rep exercises form-gated, holds duration-gated).")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--device",        default="auto")
    args = p.parse_args()
 
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
 
    exercises = ([args.exercise] if args.exercise
                 else sorted(d.name for d in labeled_dir.iterdir() if d.is_dir()))
 
    print("\n" + "═" * 64)
    print("  DL v4 — Sliding-Window MLP  (anchor-FSM + slope + focal)")
    print("═" * 64)
    print(f"  Device   : {device}")
    if args.window is None:
        win_txt = "per-exercise config"
    elif args.window == 0:
        win_txt = "auto (data-driven)"
    else:
        win_txt = f"{args.window} frames (forced)"
    if args.focal_gamma is None:
        fg_txt = "per-exercise config"
    elif args.focal_gamma <= 0:
        fg_txt = "CrossEntropy"
    else:
        fg_txt = f"Focal gamma={args.focal_gamma}"
    print(f"  Window   : {win_txt}   slope={'off' if args.no_slope else 'on'}")
    print(f"  Loss     : {fg_txt}"
          f"   boundary-dw=x{args.boundary_factor}@+/-{args.boundary_radius}")
    if args.no_augment:
        aug_txt = "OFF"
    else:
        aug_txt = (f"jitter sigma={args.jitter_sigma}px  hflip={not args.no_flip}  "
                   f"rot=+/-{args.rot_deg}deg  scale=+/-{args.scale_jitter}  "
                   f"speed=+/-{args.speed_warp}")
    print(f"  Augment  : {aug_txt}")
    print(f"  Exercises: {exercises}")
    print("═" * 64)
 
    results = []
    for ex in exercises:
        print(f"\n{'─'*64}\n  EXERCISE: {ex.upper()}\n{'─'*64}")
        try:
            videos, state_names = load_exercise_videos(
                ex, Path(args.processed_dir), labeled_dir, verbose=True,
                boundary_radius=args.boundary_radius,
                boundary_factor=args.boundary_factor)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"  [SKIP] {ex}: {exc}")
            continue
        res = train_one_exercise(ex, videos, state_names, device,
                                 effective_args(ex, args))
        if res:
            results.append(res)
 
    if results:
        print(f"\n{'═'*64}\n  SUMMARY\n{'═'*64}")
        print(f"  {'Exercise':<20}{'Win':>5}{'Classes':>9}{'ValAcc':>9}{'TestAcc':>9}")
        print(f"  {'─'*20}{'─'*5}{'─'*9}{'─'*9}{'─'*9}")
        for r in results:
            print(f"  {r['exercise']:<20}{r['window_size']:>5}"
                  f"{len(r['state_names']):>9}"
                  f"{r['best_val_acc']*100:>8.1f}%{r['test_acc']*100:>8.1f}%")
        out_dir = Path(args.out_dir)
        with open(out_dir / "training_summary_v4.json", "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"\n  Summary → {out_dir / 'training_summary_v4.json'}")
    else:
        print("\n[ERROR] No models trained.")
        sys.exit(1)
    print("\n[DL v4] Done.\n")
 
 
if __name__ == "__main__":
    main()