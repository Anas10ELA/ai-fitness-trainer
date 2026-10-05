"""
scripts/train_rf_v3.py
══════════════════════
Random Forest v3 — Strict State Filtering + Spatial Feature Enrichment

Improvements over v2 (Squat 94% / Pushup 62% / Situp 28%)
────────────────────────────────────────────────────────────

  Root cause 1: Situp noise
    Labeling sessions accidentally tagged resting/transition frames with
    states like 'jump', 'plank', or 'extended' — states that are physically
    impossible in a situp.  These ~15–25% of noisy frames polluted the
    training data and created spurious decision boundaries.

  Root cause 2: Floor-exercise foreshortening
    When a person is lying down (pushup/situp), the camera captures a nearly
    horizontal silhouette.  Joint angles computed from 2D projected coordinates
    suffer from severe foreshortening: a knee at 90° in 3D may look like 150°
    on camera because the limb is partly pointing towards the lens.
    8 angles alone cannot disambiguate upright poses from horizontal ones.

Fixes applied in v3
────────────────────
  Fix 1 — VALID_STATES per-exercise whitelist (noise removal)
    Any frame whose rep_state label is NOT in the exercise's whitelist is
    silently dropped before features are computed.  The filter runs at the
    raw label level, BEFORE the delta computation, so the temporal sequence
    is not broken — valid frames still have correct δ(t-5) references.

    The whitelist is defined conservatively (only states that actually occur
    during normal execution of that exercise) and falls back to keeping ALL
    frames for exercises not listed, so future exercises work without changes.

  Fix 2 — Two spatial features added (10 base features total)
    Feature 9  — Bounding Box Aspect Ratio
        (max_x − min_x) / max(max_y − min_y, 1e-6)
        Uses only visible (non-zero) keypoints.
        Standing upright: ~0.3   (taller than wide)
        Lying flat:       ~5–15  (wider than tall)
        This is the most direct "gravity-awareness" signal available from
        2D coordinates: a lying person has a wide, flat bounding box.

    Feature 10 — Normalized Nose-to-Ankle Y Distance
        (mean_ankle_y − nose_y) / bounding_box_height
        Normalized so the value is ~1.0 when fully upright (full body height
        separates nose from ankles) and ~0.1 when lying flat (nose and ankles
        are at nearly the same height on the image).
        Falls back to 0.0 if nose or both ankles are occluded.

  Fix 3 — Temporal deltas applied to all 10 base features (20 total)
    Both spatial features also benefit from δ encoding:
    • Δ_BBox_AR  rising sharply → person is tilting toward floor (pushup descent)
    • Δ_NoseAnkle  dropping toward 0 → person is lowering into horizontal

Feature vector layout (20 features)
─────────────────────────────────────
  Index  Name               Description
  ─────  ─────────────────  ──────────────────────────────────────────────────
  0      L_Knee             Left knee angle, degrees
  1      R_Knee             Right knee angle
  2      L_Hip              Left hip angle
  3      R_Hip              Right hip angle
  4      L_Elbow            Left elbow angle
  5      R_Elbow            Right elbow angle
  6      L_Shoulder         Left shoulder angle
  7      R_Shoulder         Right shoulder angle
  8      BBox_AR            Bounding box aspect ratio (w/h, visible joints only)
  9      NoseAnkle_Y        Norm. nose-to-ankle Y distance (÷ bbox height)
  10     Δ_L_Knee           L_Knee[t] − L_Knee[t−5]
  11     Δ_R_Knee           R_Knee[t] − R_Knee[t−5]
  12     Δ_L_Hip            …
  13     Δ_R_Hip
  14     Δ_L_Elbow
  15     Δ_R_Elbow
  16     Δ_L_Shoulder
  17     Δ_R_Shoulder
  18     Δ_BBox_AR          BBox_AR[t] − BBox_AR[t−5]  (+ = tilting to floor)
  19     Δ_NoseAnkle_Y      NoseAnkle_Y[t] − NoseAnkle_Y[t−5]  (− = lying down)

Output
──────
    checkpoints/rf_models_v3.pkl     — {exercise_name: RandomForestClassifier}
    checkpoints/rf_v3_features.json  — feature names, valid-states map, metadata
    Printed: per-exercise classification report + confusion matrix

Usage
─────
    python scripts/train_rf_v3.py
    python scripts/train_rf_v3.py \\
        --processed-dir data/processed \\
        --labeled-dir   data/labeled   \\
        --out-dir       checkpoints    \\
        --test-size     0.20           \\
        --n-estimators  300            \\
        --seed          42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ── State vocabulary (must match build_dataset.py) ───────────────────────────

STATE_NAMES: List[str] = [
    "neutral",    # 0
    "down",       # 1
    "up",         # 2
    "plank",      # 3
    "jump",       # 4
    "extended",   # 5
    "flexed",     # 6
]
N_STATES:    int            = len(STATE_NAMES)
STATE_TO_IDX: Dict[str, int] = {n: i for i, n in enumerate(STATE_NAMES)}


# ── Per-exercise valid-state whitelists ───────────────────────────────────────
#
# Only states in this list are PHYSICALLY POSSIBLE for the given exercise.
# Frames whose labels fall outside the whitelist are dropped during loading.
#
# Rules for building the whitelist:
#   • Include every state the labeler should legitimately use.
#   • Exclude states that arise purely from labeling errors (e.g. 'jump' in situp).
#   • Exercises not listed here → no filter → all frames kept (safe default).

VALID_STATES: Dict[str, List[str]] = {
    "situp":          ["neutral", "down", "up", "flexed"],
    "pushup":         ["neutral", "down", "up", "plank"],
    "squat":          ["neutral", "down", "up"],
    "wall_sit":       ["neutral", "down"],
    "lateral_raise":  ["neutral", "extended"],
    "tricep_dip":     ["down", "up", "extended"],
    # Exercises below have no filter — add entries here as needed:
    # "bicep_curl":   ["neutral", "extended", "flexed"],
    # "lunge":        ["neutral", "down"],
    # "burpee":       ["neutral", "down", "plank", "up", "jump"],
    # "plank":        ["plank"],
    # "high_knees":   ["neutral", "flexed"],
    # "jumping_jack": ["neutral", "extended"],
    # "shoulder_press":["neutral", "down", "extended"],
    # "mountain_climber":["plank", "flexed"],
    # "pullup":       ["neutral", "up"],
}

# Pre-compute frozenset of valid indices for O(1) lookup
VALID_IDX: Dict[str, FrozenSet[int]] = {
    ex: frozenset(STATE_TO_IDX[s] for s in states if s in STATE_TO_IDX)
    for ex, states in VALID_STATES.items()
}


# ── Delta lag ─────────────────────────────────────────────────────────────────

DELTA_LAG: int = 5    # frames; ~167 ms at 30 fps


# ── Angle definitions ─────────────────────────────────────────────────────────
#
# Each entry: (feature_name, point_a_idx, vertex_idx, point_c_idx)
# Angle is computed AT the vertex using rays: vertex→a and vertex→c

ANGLE_DEFS: List[Tuple[str, int, int, int]] = [
    ("L_Knee",      11,  13,  15),   # hip → knee ← ankle
    ("R_Knee",      12,  14,  16),
    ("L_Hip",        5,  11,  13),   # shoulder → hip ← knee
    ("R_Hip",        6,  12,  14),
    ("L_Elbow",      5,   7,   9),   # shoulder → elbow ← wrist
    ("R_Elbow",      6,   8,  10),
    ("L_Shoulder",  11,   5,   7),   # hip → shoulder ← elbow
    ("R_Shoulder",  12,   6,   8),
]
N_ANGLES: int = len(ANGLE_DEFS)     # 8

# Spatial feature names (appended after angles)
SPATIAL_NAMES: List[str] = ["BBox_AR", "NoseAnkle_Y"]
N_SPATIAL: int = len(SPATIAL_NAMES)  # 2

N_BASE: int = N_ANGLES + N_SPATIAL   # 10

BASE_NAMES:     List[str] = [d[0] for d in ANGLE_DEFS] + SPATIAL_NAMES
DELTA_NAMES:    List[str] = [f"Δ_{n}" for n in BASE_NAMES]
ALL_FEAT_NAMES: List[str] = BASE_NAMES + DELTA_NAMES
N_FEATURES:     int       = len(ALL_FEAT_NAMES)   # 20


# ═══════════════════════════════════════════════════════════════════════════════
#  Angle helpers  (identical to v1/v2)
# ═══════════════════════════════════════════════════════════════════════════════

def _is_occluded(pt: np.ndarray) -> bool:
    """True if keypoint is zeroed out (undetected)."""
    return float(pt[0]) == 0.0 and float(pt[1]) == 0.0


def _angle_at_vertex(a: np.ndarray, v: np.ndarray, c: np.ndarray) -> float:
    """
    Angle at vertex v in triplet (a, v, c), in degrees.
    Returns 0.0 for any occluded point or degenerate vector.
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


# ═══════════════════════════════════════════════════════════════════════════════
#  Spatial feature helpers  (new in v3)
# ═══════════════════════════════════════════════════════════════════════════════

def _bbox_aspect_ratio(keypoints: np.ndarray) -> float:
    """
    Bounding box aspect ratio = width / height, computed over visible joints only.

    Standing upright:  ~0.3  (body is taller than wide)
    Lying flat:        ~5–15 (body is much wider than tall in image coords)
    All joints occluded or single point: returns 0.0.
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
    Vertical (Y) distance from nose to mean ankle, normalized by bbox height.

    In image coordinates Y increases downward, so for an upright person:
        ankle_y  >  nose_y  →  positive distance ≈ 1.0
    For a person lying flat:
        ankle_y  ≈  nose_y  →  distance ≈ 0.0–0.1

    Returns 0.0 if:
        • Nose (index 0) is occluded
        • Both ankles (indices 15, 16) are occluded
        • Bounding box height is degenerate (<1e-6 px after normalization)
    """
    nose    = keypoints[0]
    l_ankle = keypoints[15]
    r_ankle = keypoints[16]

    valid_ankles = [a for a in [l_ankle, r_ankle] if not _is_occluded(a)]
    if _is_occluded(nose) or not valid_ankles:
        return 0.0

    ankle_y = float(np.mean([a[1] for a in valid_ankles]))
    raw_y   = ankle_y - float(nose[1])   # positive when upright

    # Normalize by bbox height computed over visible joints
    visible = np.array(
        [kp for kp in keypoints if not _is_occluded(kp)],
        dtype=np.float32,
    )
    if len(visible) < 2:
        return 0.0

    bbox_h = float(visible[:, 1].max() - visible[:, 1].min())
    return float(raw_y / max(bbox_h, 1e-6))


# ═══════════════════════════════════════════════════════════════════════════════
#  10-feature base extractor  (new in v3)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_features(keypoints: np.ndarray) -> np.ndarray:
    """
    Extract the 10-element base feature vector for a single (17, 2) frame.

    Layout
    ------
      [0:8]  8 joint angles (degrees) — same as v1/v2
      [8]    bounding box aspect ratio
      [9]    normalized nose-to-ankle Y distance

    Occluded joints → 0.0.  All values are float32.
    """
    feat = np.zeros(N_BASE, dtype=np.float32)

    # Angles (0–7)
    for i, (_, a, v, c) in enumerate(ANGLE_DEFS):
        feat[i] = _angle_at_vertex(keypoints[a], keypoints[v], keypoints[c])

    # Spatial (8–9)
    feat[N_ANGLES]     = _bbox_aspect_ratio(keypoints)
    feat[N_ANGLES + 1] = _norm_nose_ankle_y(keypoints)

    return feat


# ═══════════════════════════════════════════════════════════════════════════════
#  Delta features  (generalised from v2; now operates on N_BASE=10 columns)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_delta_features(
    base_seq: np.ndarray,   # (N, N_BASE)
    lag:      int = DELTA_LAG,
) -> np.ndarray:
    """
    Temporal velocity: base_seq[t] − base_seq[max(t−lag, 0)].

    For t < lag the reference is frame 0, giving an honest "how much has
    changed since the start" signal rather than a spurious zero.

    Returns (N, N_BASE) float32.
    """
    N = len(base_seq)
    deltas = np.zeros_like(base_seq, dtype=np.float32)
    for t in range(N):
        ref        = max(t - lag, 0)
        deltas[t]  = base_seq[t] - base_seq[ref]
    return deltas


def build_feature_matrix(kp_seq: np.ndarray) -> np.ndarray:
    """
    Full 20-feature matrix for one video: [base_features | delta_features].

    Parameters
    ----------
    kp_seq : (N, 17, 2)

    Returns
    -------
    np.ndarray  (N, 20)  float32
        Columns 0–9   : base features (8 angles + BBox_AR + NoseAnkle_Y)
        Columns 10–19 : delta features (same 10, each = feat[t] − feat[t−5])
    """
    base   = np.vstack([calculate_features(kp_seq[f]) for f in range(len(kp_seq))])
    deltas = compute_delta_features(base)
    return np.hstack([base, deltas]).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  State filtering
# ═══════════════════════════════════════════════════════════════════════════════

def get_valid_mask(rep_states: np.ndarray, ex_name: str) -> np.ndarray:
    """
    Boolean mask: True for frames whose label is in VALID_IDX[ex_name].

    If ex_name is not in VALID_IDX, returns all-True (no frames dropped).

    Parameters
    ----------
    rep_states : (N,)  int32 array of state indices
    ex_name    : str   exercise folder name

    Returns
    -------
    np.ndarray  (N,)  bool
    """
    valid_set = VALID_IDX.get(ex_name)
    if valid_set is None:
        return np.ones(len(rep_states), dtype=bool)
    return np.isin(rep_states, sorted(valid_set))


# ═══════════════════════════════════════════════════════════════════════════════
#  Data loading — grouped by exercise, with state filtering
# ═══════════════════════════════════════════════════════════════════════════════

def load_data_by_exercise(
    processed_dir: Path,
    labeled_dir:   Path,
    verbose:       bool = True,
) -> Dict[str, Dict]:
    """
    Load all videos, apply per-exercise state filtering, build 20-feature
    matrices, and return data grouped by exercise.

    The filtering happens BEFORE feature computation at the frame level, but
    AFTER the full base-feature sequence is computed for the video so that
    delta references (t−5) are drawn from the ORIGINAL temporal sequence,
    not a gap-riddled post-filter one.  Only the final row selection changes.

    Returns
    -------
    dict keyed by exercise_name:
        {
            "X":         (N_valid_frames, 20)  float32
            "y":         (N_valid_frames,)     int32
            "video_ids": list[str]             length N_valid_frames
        }
    """
    data:       Dict[str, Dict[str, list]] = {}
    skipped    = 0
    total_vids = 0
    total_dropped = 0

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

                N          = min(len(kp), len(rep_states))
                kp         = kp[:N]
                rep_states = rep_states[:N]

                # ── Global validity check ────────────────────────────────────
                out_of_range = (rep_states < 0) | (rep_states >= N_STATES)
                if out_of_range.any():
                    if verbose:
                        print(f"  [WARN] {vid_key}: {out_of_range.sum()} "
                              f"out-of-range labels → clipped to 0")
                    rep_states[out_of_range] = 0

                # ── Build full feature matrix BEFORE filtering ───────────────
                # Delta features require an unbroken temporal sequence; we
                # compute on all N frames first, then drop invalid rows.
                X_full = build_feature_matrix(kp)   # (N, 20)

                # ── Per-exercise state whitelist filter ──────────────────────
                valid_mask    = get_valid_mask(rep_states, ex_name)
                n_dropped     = int((~valid_mask).sum())
                total_dropped += n_dropped

                X_video    = X_full[valid_mask]        # (N_valid, 20)
                y_video    = rep_states[valid_mask]    # (N_valid,)

                if len(y_video) == 0:
                    if verbose:
                        print(f"  [SKIP] {vid_key}: 0 frames after filtering "
                              f"(all {N} frames were invalid states)")
                    continue

                # ── Accumulate ───────────────────────────────────────────────
                if ex_name not in data:
                    data[ex_name] = {"X": [], "y": [], "video_ids": []}
                data[ex_name]["X"].append(X_video)
                data[ex_name]["y"].append(y_video)
                data[ex_name]["video_ids"].extend([vid_key] * len(y_video))

                total_vids += 1
                if verbose:
                    kept_pct = len(y_video) / N * 100
                    drop_str = (f"  ← dropped {n_dropped} noisy"
                                if n_dropped else "")
                    print(f"  [load] {vid_key}: {len(y_video)}/{N} frames "
                          f"({kept_pct:.0f}% kept){drop_str}  "
                          f"states={np.unique(y_video).tolist()}")

            except Exception as exc:
                if verbose:
                    print(f"  [WARN] Failed {vid_key}: {exc}")
                skipped += 1
                continue

    if not data:
        raise RuntimeError(
            "No data loaded. Check --processed-dir and --labeled-dir paths."
        )

    # Consolidate
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


# ═══════════════════════════════════════════════════════════════════════════════
#  Video-wise train/test split  (identical to v2)
# ═══════════════════════════════════════════════════════════════════════════════

def video_wise_split(
    X:         np.ndarray,
    y:         np.ndarray,
    video_ids: List[str],
    test_size: float = 0.20,
    seed:      int   = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split by video ID — no frame from the same video appears in both splits.
    Guarantees at least 1 test video regardless of test_size rounding.
    """
    unique_vids = sorted(set(video_ids))
    rng         = np.random.default_rng(seed)
    rng.shuffle(unique_vids)

    n_test   = max(1, int(len(unique_vids) * test_size))
    test_set = set(unique_vids[-n_test:])

    vid_arr    = np.array(video_ids)
    train_mask = ~np.isin(vid_arr, list(test_set))
    test_mask  =  np.isin(vid_arr, list(test_set))

    return (
        X[train_mask], X[test_mask],
        y[train_mask], y[test_mask],
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-exercise training
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_exercise(
    ex_name:          str,
    X_train:          np.ndarray,
    y_train:          np.ndarray,
    n_estimators:     int = 300,
    min_samples_leaf: int = 2,
    seed:             int = 42,
) -> RandomForestClassifier:
    """
    Train a single RandomForestClassifier for one exercise.

    class_weight='balanced' handles within-exercise state imbalance
    (e.g. most squat frames are neutral/down; very few are up or jump).
    """
    clf = RandomForestClassifier(
        n_estimators     = n_estimators,
        min_samples_leaf = min_samples_leaf,
        class_weight     = "balanced",
        random_state     = seed,
        n_jobs           = -1,
    )
    clf.fit(X_train, y_train)
    return clf


# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_one_exercise(
    ex_name: str,
    clf:     RandomForestClassifier,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
) -> float:
    """Print classification report, confusion matrix, top-5 features.
    Returns overall accuracy (float)."""
    y_pred  = clf.predict(X_test)
    present = sorted(set(y_test.tolist()))
    labels  = [STATE_NAMES[i] for i in present]

    print(f"\n{'═'*64}")
    print(f"  EXERCISE: {ex_name.upper()}")
    print(f"  Test frames: {len(y_test):,}  |  States: {labels}")
    print(f"{'─'*64}")
    print(classification_report(
        y_test, y_pred,
        labels       = present,
        target_names = labels,
        zero_division= 0,
    ))

    # ── Confusion matrix ─────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred, labels=present)
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

    acc = float((y_pred == y_test).mean())
    print(f"\n  Accuracy: {acc:.4f}  ({acc*100:.1f}%)")

    # ── Top-5 feature importances ─────────────────────────────────────────────
    imp   = clf.feature_importances_
    order = np.argsort(imp)[::-1][:5]
    print(f"\n  Top-5 features:")
    for rank, idx in enumerate(order, 1):
        bar = "█" * int(imp[idx] * 150)
        print(f"    {rank}. {ALL_FEAT_NAMES[idx]:<18}  {imp[idx]:.4f}  {bar}")

    return acc


# ═══════════════════════════════════════════════════════════════════════════════
#  Export
# ═══════════════════════════════════════════════════════════════════════════════

def save_models(
    models_dict: Dict[str, RandomForestClassifier],
    per_ex_meta: Dict[str, Dict],
    out_dir:     Path,
) -> None:
    """
    Save the dict of models to rf_models_v3.pkl and write rf_v3_features.json.

    Inference recipe (from the JSON):
        1. Maintain a deque of DELTA_LAG+1 raw keypoint frames (17, 2).
        2. For frame t: feat_base = calculate_features(kp[t])
        3. feat_delta = feat_base − calculate_features(kp[max(t−DELTA_LAG, 0)])
        4. feat_vec = np.concatenate([feat_base, feat_delta])   # shape (20,)
        5. pred = models_dict[exercise_name].predict([feat_vec])
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    models_path = out_dir / "rf_models_v3.pkl"
    meta_path   = out_dir / "rf_v3_features.json"

    joblib.dump(models_dict, models_path)

    meta = {
        "version":        "v3",
        "n_features":     N_FEATURES,
        "n_base":         N_BASE,
        "feature_names":  ALL_FEAT_NAMES,
        "base_features":  BASE_NAMES,
        "delta_features": DELTA_NAMES,
        "delta_lag":      DELTA_LAG,
        "angle_defs": [
            {"name": name, "point_a": a, "vertex": v, "point_c": c}
            for name, a, v, c in ANGLE_DEFS
        ],
        "spatial_features": {
            "BBox_AR": {
                "formula": "(max_x - min_x) / max(max_y - min_y, 1e-6)",
                "note":    "Visible joints only. ~0.3 upright, ~5-15 lying flat."
            },
            "NoseAnkle_Y": {
                "formula": "(mean_ankle_y - nose_y) / max(bbox_height, 1e-6)",
                "note":    "~1.0 upright, ~0.0 lying flat. 0.0 if occluded.",
                "joints":  {"nose": 0, "left_ankle": 15, "right_ankle": 16}
            },
        },
        "valid_states":  VALID_STATES,
        "state_names":   STATE_NAMES,
        "n_states":      N_STATES,
        "exercises":     per_ex_meta,
        "inference_note": (
            "1. Keep a deque of DELTA_LAG+1 keypoint frames (17,2). "
            "2. feat_base = calculate_features(kp[t]). "
            "3. feat_delta = feat_base - calculate_features(kp[max(t-DELTA_LAG,0)]). "
            "4. feat_vec = concat([feat_base, feat_delta]) → shape (20,). "
            "5. models_dict[exercise_name].predict([feat_vec])."
        ),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[Save]  Models dict ({len(models_dict)} exercises) → {models_path}")
    print(f"[Save]  Metadata                                  → {meta_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "RF v3: per-exercise models, state filtering, "
            "20 temporal features (10 base + 10 deltas)"
        )
    )
    p.add_argument("--processed-dir",    default="data/processed")
    p.add_argument("--labeled-dir",      default="data/labeled")
    p.add_argument("--out-dir",          default="checkpoints")
    p.add_argument("--test-size",        type=float, default=0.20,
                   help="Fraction of videos per exercise held out (default: 0.20)")
    p.add_argument("--n-estimators",     type=int,   default=300)
    p.add_argument("--min-samples-leaf", type=int,   default=2)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--quiet",            action="store_true")
    args = p.parse_args()

    verbose = not args.quiet

    print("\n[RF v3]  Per-Exercise RF  +  State Filtering  +  Spatial Features")
    print(f"  Features/frame : {N_FEATURES}  "
          f"({N_BASE} base: 8 angles + 2 spatial;  {N_BASE} deltas at t−{DELTA_LAG})")
    print(f"  Exercises with state filter: "
          f"{', '.join(sorted(VALID_STATES.keys()))}")
    print(f"  processed_dir  : {args.processed_dir}")
    print(f"  labeled_dir    : {args.labeled_dir}")
    print(f"  test_size      : {args.test_size:.0%}  (video-wise per exercise)")
    print(f"  n_estimators   : {args.n_estimators}\n")

    # ── 1. Load ───────────────────────────────────────────────────────────────
    exercise_data = load_data_by_exercise(
        Path(args.processed_dir),
        Path(args.labeled_dir),
        verbose=verbose,
    )

    # ── 2. Per-exercise loop ──────────────────────────────────────────────────
    models_dict:  Dict[str, RandomForestClassifier] = {}
    per_ex_meta:  Dict[str, Dict]                   = {}
    summary_rows: List[Tuple[str, int, int, float]] = []

    for ex_name in sorted(exercise_data.keys()):
        d          = exercise_data[ex_name]
        X, y, vids = d["X"], d["y"], d["video_ids"]
        n_videos   = len(set(vids))

        print(f"\n{'─'*64}")
        print(f"  [{ex_name}]  {X.shape[0]:,} frames  |  {n_videos} videos  "
              f"|  classes: {sorted(set(y.tolist()))}")

        if n_videos < 2:
            print(f"  [SKIP] Only {n_videos} video(s) — need ≥2 for split.")
            continue

        X_tr, X_te, y_tr, y_te = video_wise_split(
            X, y, vids, test_size=args.test_size, seed=args.seed
        )
        print(f"  Train: {len(X_tr):,} frames  |  Test: {len(X_te):,} frames")

        if len(X_te) == 0:
            print("  [SKIP] Empty test set.")
            continue

        clf = train_one_exercise(
            ex_name, X_tr, y_tr,
            n_estimators     = args.n_estimators,
            min_samples_leaf = args.min_samples_leaf,
            seed             = args.seed,
        )

        acc = evaluate_one_exercise(ex_name, clf, X_te, y_te)

        models_dict[ex_name] = clf
        per_ex_meta[ex_name] = {
            "n_train":       int(len(X_tr)),
            "n_test":        int(len(X_te)),
            "n_videos":      n_videos,
            "states_seen":   [STATE_NAMES[i] for i in sorted(set(y.tolist()))],
            "valid_states":  VALID_STATES.get(ex_name, "all"),
            "test_accuracy": round(acc, 4),
        }
        summary_rows.append((ex_name, len(X_tr), len(X_te), acc))

    # ── 3. Summary table ──────────────────────────────────────────────────────
    print(f"\n{'═'*64}")
    print("  SUMMARY")
    print(f"  {'Exercise':<22}  {'Train':>6}  {'Test':>6}  {'Accuracy':>9}")
    print(f"  {'─'*22}  {'─'*6}  {'─'*6}  {'─'*9}")
    total_test = total_correct = 0
    for ex_name, n_tr, n_te, acc in summary_rows:
        print(f"  {ex_name:<22}  {n_tr:>6,}  {n_te:>6,}  {acc*100:>8.1f}%")
        total_test    += n_te
        total_correct += int(acc * n_te)
    if total_test > 0:
        overall = total_correct / total_test
        print(f"  {'─'*22}  {'─'*6}  {'─'*6}  {'─'*9}")
        print(f"  {'OVERALL (weighted)':<22}  {'':>6}  {total_test:>6,}  "
              f"{overall*100:>8.1f}%")
    print(f"{'═'*64}")

    if not models_dict:
        print("\n[ERROR] No models trained — check data paths and video counts.")
        sys.exit(1)

    # ── 4. Export ─────────────────────────────────────────────────────────────
    save_models(models_dict, per_ex_meta, Path(args.out_dir))
    print("\n[RF v3]  Done.\n")


if __name__ == "__main__":
    main()
