"""
scripts/train_rf.py
═══════════════════
Classical ML approach: Feature Engineering + Random Forest
for skeleton-based rep-state classification (7 states).

Why Random Forest instead of ST-GCN on a small dataset?
────────────────────────────────────────────────────────
ST-GCN has ~339k–2.59M parameters and needs thousands of sequences to
generalise. With ~100 videos it overfits catastrophically (train 84% / val 39%).
A Random Forest trained on 8 geometric joint angles is a far better fit:
  • Tabular features are human-interpretable and robust to camera distance
  • RF is inherently regularised through bagging + feature sub-sampling
  • Training takes seconds; no GPU required
  • Works well with as few as a few hundred samples per class

Feature Engineering
───────────────────
For each frame (17, 2) COCO keypoint array we compute 8 joint angles (degrees):

  Joint           Triplet (point_a – vertex – point_c)   COCO indices
  ─────────────── ──────────────────────────────────────  ────────────
  Left  Knee      left_hip  – left_knee  – left_ankle     11, 13, 15
  Right Knee      right_hip – right_knee – right_ankle    12, 14, 16
  Left  Hip       left_shoulder  – left_hip  – left_knee   5, 11, 13
  Right Hip       right_shoulder – right_hip – right_knee  6, 12, 14
  Left  Elbow     left_shoulder  – left_elbow – left_wrist 5,  7,  9
  Right Elbow     right_shoulder – right_elbow– right_wrist 6,  8, 10
  Left  Shoulder  left_hip  – left_shoulder  – left_elbow 11,  5,  7
  Right Shoulder  right_hip – right_shoulder – right_elbow 12,  6,  8

Occluded joints (coordinate == [0, 0]) → angle set to 0.0 (sentinel).

Split strategy
──────────────
VIDEO-WISE split (not frame-wise).  This is critical: a frame-wise split
would leak frames from the same video into both train and test sets, inflating
accuracy by ~20–30% on small datasets.  Video-wise split gives a honest
estimate of generalisation to new subjects / recordings.

Usage
─────
    python scripts/train_rf.py
    python scripts/train_rf.py --processed-dir data/processed \\
                                --labeled-dir   data/labeled   \\
                                --out-dir       checkpoints    \\
                                --test-size     0.20           \\
                                --n-estimators  300            \\
                                --seed          42

Output
──────
    checkpoints/rf_model.pkl     — trained RandomForestClassifier (joblib)
    checkpoints/rf_features.json — feature names + label mapping
    Printed: classification report + confusion matrix
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Label maps (must match build_dataset.py) ──────────────────────────────────

STATE_NAMES: List[str] = [
    "neutral",    # 0
    "down",       # 1
    "up",         # 2
    "plank",      # 3
    "jump",       # 4
    "extended",   # 5
    "flexed",     # 6
]
N_STATES: int = len(STATE_NAMES)

# ── Feature definitions ───────────────────────────────────────────────────────

# Each entry: (feature_name, point_a_idx, vertex_idx, point_c_idx)
# Angle is computed at the vertex using the rays: vertex→a and vertex→c
ANGLE_DEFS: List[Tuple[str, int, int, int]] = [
    # Joint           a    vertex  c    description
    ("L_Knee",       11,   13,    15),  # hip – knee – ankle
    ("R_Knee",       12,   14,    16),
    ("L_Hip",         5,   11,    13),  # shoulder – hip – knee
    ("R_Hip",         6,   12,    14),
    ("L_Elbow",       5,    7,     9),  # shoulder – elbow – wrist
    ("R_Elbow",       6,    8,    10),
    ("L_Shoulder",   11,    5,     7),  # hip – shoulder – elbow
    ("R_Shoulder",   12,    6,     8),
]
FEATURE_NAMES: List[str] = [d[0] for d in ANGLE_DEFS]
N_FEATURES:    int        = len(ANGLE_DEFS)


# ═══════════════════════════════════════════════════════════════════════════════
#  Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════════

def _is_occluded(point: np.ndarray) -> bool:
    """Return True if the keypoint is zeroed out (occluded / undetected)."""
    return float(point[0]) == 0.0 and float(point[1]) == 0.0


def _angle_at_vertex(
    a: np.ndarray,   # (2,)  first arm of angle
    v: np.ndarray,   # (2,)  vertex
    c: np.ndarray,   # (2,)  second arm of angle
) -> float:
    """
    Compute the angle at vertex v in the triplet (a, v, c), in degrees.

    Returns 0.0 if any of the three points is occluded ([0,0]) or if the
    vectors are degenerate (zero-length, which would cause division by zero).
    The 0.0 sentinel is safe because a genuine straight-line pose gives 180°
    and a fully flexed joint gives ~30–90°; 0° does not naturally occur.
    """
    if _is_occluded(a) or _is_occluded(v) or _is_occluded(c):
        return 0.0

    ba = (a - v).astype(np.float64)
    bc = (c - v).astype(np.float64)

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0

    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))   # guard fp drift
    return float(np.degrees(np.arccos(cos_angle)))


def calculate_angles(keypoints: np.ndarray) -> np.ndarray:
    """
    Compute 8 joint angles from a single COCO-17 keypoint frame.

    Parameters
    ----------
    keypoints : (17, 2)  float array of (x, y) pixel or normalized coordinates
                         Zero-valued joints are treated as occluded.

    Returns
    -------
    np.ndarray  shape (8,)  dtype float32
        Angles in degrees: [L_Knee, R_Knee, L_Hip, R_Hip,
                            L_Elbow, R_Elbow, L_Shoulder, R_Shoulder]
        Occluded angles are 0.0.
    """
    angles = np.zeros(N_FEATURES, dtype=np.float32)
    for i, (_, a_idx, v_idx, c_idx) in enumerate(ANGLE_DEFS):
        angles[i] = _angle_at_vertex(
            keypoints[a_idx],
            keypoints[v_idx],
            keypoints[c_idx],
        )
    return angles


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_dataset(
    processed_dir: Path,
    labeled_dir:   Path,
    verbose:       bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Walk through all processed + labeled .npz pairs and build a tabular
    (X, y) dataset by computing 8 joint angles per frame.

    Returns
    -------
    X         : (N_frames, 8)  float32  — angle feature matrix
    y         : (N_frames,)    int32    — rep_state label per frame (0-6)
    video_ids : list of length N_frames — which video each frame came from
                (used for video-wise train/test split)
    """
    X_rows:    List[np.ndarray] = []
    y_rows:    List[np.ndarray] = []
    vid_rows:  List[str]        = []

    skipped    = 0
    total_vids = 0

    for ex_dir in sorted(labeled_dir.iterdir()):
        if not ex_dir.is_dir():
            continue
        ex_name = ex_dir.name

        for label_npz in sorted(ex_dir.glob("*.npz")):
            video_id  = label_npz.stem
            vid_key   = f"{ex_name}/{video_id}"
            proc_npz  = processed_dir / ex_name / f"{video_id}.npz"

            if not proc_npz.exists():
                if verbose:
                    print(f"  [WARN] missing processed: {vid_key}")
                skipped += 1
                continue

            try:
                proc  = np.load(proc_npz,  allow_pickle=True)
                label = np.load(label_npz, allow_pickle=True)

                kp          = proc["keypoints"].astype(np.float32)     # (N, 17, 2)
                rep_states  = label["rep_states"].astype(np.int32)     # (N,)

                # Align lengths
                N          = min(len(kp), len(rep_states))
                kp         = kp[:N]
                rep_states = rep_states[:N]

                # Validate state indices
                invalid = (rep_states < 0) | (rep_states >= N_STATES)
                if invalid.any():
                    if verbose:
                        print(f"  [WARN] {vid_key}: {invalid.sum()} invalid state "
                              f"indices — clipped to 0")
                    rep_states[invalid] = 0

                # ── Feature engineering: one row per frame ───────────────────
                angles = np.vstack([calculate_angles(kp[f]) for f in range(N)])
                # angles shape: (N, 8)

                X_rows.append(angles)
                y_rows.append(rep_states)
                vid_rows.extend([vid_key] * N)

                total_vids += 1
                if verbose:
                    print(f"  [load] {vid_key}: {N} frames, "
                          f"states={np.unique(rep_states).tolist()}")

            except Exception as exc:
                if verbose:
                    print(f"  [WARN] Failed {vid_key}: {exc}")
                skipped += 1
                continue

    if not X_rows:
        raise RuntimeError(
            "No data loaded. Check --processed-dir and --labeled-dir paths."
        )

    X = np.vstack(X_rows).astype(np.float32)   # (N_total, 8)
    y = np.concatenate(y_rows).astype(np.int32) # (N_total,)

    if verbose:
        print(f"\n  Loaded {total_vids} videos  |  "
              f"{X.shape[0]:,} frames  |  skipped {skipped}")
        state_counts = np.bincount(y, minlength=N_STATES)
        for i, (name, cnt) in enumerate(zip(STATE_NAMES, state_counts)):
            bar = "█" * (cnt * 30 // max(state_counts.max(), 1))
            print(f"    {name:<10} [{i}]: {cnt:5d}  {bar}")

    return X, y, vid_rows


# ═══════════════════════════════════════════════════════════════════════════════
#  Video-wise Train / Test Split
# ═══════════════════════════════════════════════════════════════════════════════

def video_wise_split(
    X:         np.ndarray,
    y:         np.ndarray,
    video_ids: List[str],
    test_size: float = 0.20,
    seed:      int   = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split by video, not by frame.

    Why this matters
    ────────────────
    A frame-wise split assigns individual frames to train/test randomly.
    Since consecutive frames in the same video look almost identical, the
    model would "memorise" the video rather than learn the motion pattern.
    Video-wise split guarantees every frame from a given recording is
    entirely in train OR entirely in test — giving an honest accuracy figure.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    unique_videos = sorted(set(video_ids))
    rng           = np.random.default_rng(seed)
    rng.shuffle(unique_videos)

    n_test    = max(1, int(len(unique_videos) * test_size))
    test_vids = set(unique_videos[-n_test:])
    train_vids= set(unique_videos[:-n_test])

    vid_arr = np.array(video_ids)
    train_mask = np.isin(vid_arr, list(train_vids))
    test_mask  = np.isin(vid_arr, list(test_vids))

    print(f"\n  Split: {train_mask.sum():,} train frames "
          f"({len(train_vids)} videos)  |  "
          f"{test_mask.sum():,} test frames "
          f"({len(test_vids)} videos)")

    return (
        X[train_mask], X[test_mask],
        y[train_mask], y[test_mask],
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Training & Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def train_random_forest(
    X_train:       np.ndarray,
    y_train:       np.ndarray,
    n_estimators:  int   = 300,
    max_depth:     Optional[int] = None,
    min_samples_leaf: int = 2,
    class_weight:  str   = "balanced",
    seed:          int   = 42,
    n_jobs:        int   = -1,
) -> RandomForestClassifier:
    """
    Train a Random Forest with sensible defaults for small skeletal datasets.

    Hyperparameter rationale
    ────────────────────────
    n_estimators=300     : More trees = lower variance; 300 is a good ceiling
                           before returns diminish; fast at inference.
    max_depth=None       : Unlimited depth — each tree will overfit its
                           bootstrap sample, which is the intended behaviour
                           in bagging. Depth limits hurt RF more than they help.
    min_samples_leaf=2   : Prevents single-sample leaves (most important
                           regulariser on very small datasets).
    class_weight=balanced: Compensates for state imbalance (neutral frames
                           typically outnumber jump or plank frames 10:1).
    """
    print(f"\n[RF] Training RandomForestClassifier  "
          f"n_estimators={n_estimators}  "
          f"min_samples_leaf={min_samples_leaf}  "
          f"class_weight={class_weight}")

    clf = RandomForestClassifier(
        n_estimators     = n_estimators,
        max_depth        = max_depth,
        min_samples_leaf = min_samples_leaf,
        class_weight     = class_weight,
        random_state     = seed,
        n_jobs           = n_jobs,
    )
    clf.fit(X_train, y_train)
    print(f"[RF] Training complete")
    return clf


def evaluate(
    clf:    RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Print classification report, confusion matrix, and feature importances."""
    y_pred = clf.predict(X_test)

    # ── Labels present in test set ───────────────────────────────────────────
    present = sorted(set(y_test.tolist()))
    labels  = [STATE_NAMES[i] for i in present]

    print(f"\n{'═'*64}")
    print("  CLASSIFICATION REPORT (test set)")
    print(f"{'─'*64}")
    print(classification_report(
        y_test, y_pred,
        labels      = present,
        target_names= labels,
        zero_division= 0,
    ))

    # ── Confusion matrix ─────────────────────────────────────────────────────
    cm      = confusion_matrix(y_test, y_pred, labels=present)
    max_val = cm.max()

    print("  CONFUSION MATRIX")
    print(f"  {'':>12}", end="")
    for lab in labels:
        print(f"  {lab[:6]:>6}", end="")
    print()

    for i, row_label in enumerate(labels):
        print(f"  {row_label:>12}", end="")
        for j, val in enumerate(cm[i]):
            # Shade correctly predicted cells
            mark = "■" if (i == j and val > 0) else " "
            print(f"  {val:>5}{mark}", end="")
        print()

    overall_acc = (y_pred == y_test).mean()
    print(f"\n  Overall accuracy : {overall_acc:.4f}  ({overall_acc*100:.1f}%)")
    print(f"{'═'*64}")

    # ── Feature importances ───────────────────────────────────────────────────
    importances = clf.feature_importances_
    order       = np.argsort(importances)[::-1]

    print("\n  FEATURE IMPORTANCES (mean decrease in impurity)")
    print(f"  {'Feature':<15}  {'Importance':>10}  Bar")
    print(f"  {'─'*15}  {'─'*10}  {'─'*30}")
    for idx in order:
        bar = "█" * int(importances[idx] * 200)
        print(f"  {FEATURE_NAMES[idx]:<15}  {importances[idx]:>10.4f}  {bar}")

    # ── OOB score (free validation estimate) ─────────────────────────────────
    if hasattr(clf, "oob_score_") and clf.oob_score_:
        print(f"\n  OOB (out-of-bag) accuracy : {clf.oob_score_:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Export
# ═══════════════════════════════════════════════════════════════════════════════

def save_model(
    clf:     RandomForestClassifier,
    out_dir: Path,
) -> None:
    """
    Save the trained model to out_dir/rf_model.pkl and a companion
    rf_features.json that documents what the model expects at inference time.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path   = out_dir / "rf_model.pkl"
    feature_path = out_dir / "rf_features.json"

    joblib.dump(clf, model_path)

    meta = {
        "feature_names":   FEATURE_NAMES,
        "n_features":      N_FEATURES,
        "angle_defs":      [
            {"name": name, "point_a": a, "vertex": v, "point_c": c}
            for name, a, v, c in ANGLE_DEFS
        ],
        "state_names":     STATE_NAMES,
        "n_states":        N_STATES,
        "n_estimators":    clf.n_estimators,
        "sklearn_version": __import__("sklearn").__version__,
        "note": (
            "Load with joblib.load(rf_model.pkl). "
            "Feed an (N, 8) float32 array of angles computed by calculate_angles(). "
            "Output is an (N,) int array of state indices (0-6)."
        ),
    }
    with open(feature_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[Save] Model → {model_path}")
    print(f"[Save] Metadata → {feature_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="Train a Random Forest rep-state classifier on COCO-17 joint angles"
    )
    p.add_argument("--processed-dir",    default="data/processed",
                   help="Root directory of per-exercise processed .npz files")
    p.add_argument("--labeled-dir",      default="data/labeled",
                   help="Root directory of per-exercise labeled .npz files")
    p.add_argument("--out-dir",          default="checkpoints",
                   help="Output directory for rf_model.pkl + rf_features.json")
    p.add_argument("--test-size",        type=float, default=0.20,
                   help="Fraction of VIDEOS held out for testing (default: 0.20)")
    p.add_argument("--n-estimators",     type=int,   default=300,
                   help="Number of RF trees (default: 300)")
    p.add_argument("--min-samples-leaf", type=int,   default=2,
                   help="RF min_samples_leaf regulariser (default: 2)")
    p.add_argument("--oob-score",        action="store_true", default=False,
                   help="Compute OOB score (slower; provides a free second estimate)")
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--quiet",            action="store_true")
    args = p.parse_args()

    verbose = not args.quiet

    print("\n[RF Pipeline] Random Forest rep-state classifier")
    print(f"  processed_dir : {args.processed_dir}")
    print(f"  labeled_dir   : {args.labeled_dir}")
    print(f"  test_size     : {args.test_size:.0%} (video-wise)")
    print(f"  n_estimators  : {args.n_estimators}")
    print()

    # ── 1. Load ──────────────────────────────────────────────────────────────
    X, y, video_ids = load_dataset(
        Path(args.processed_dir),
        Path(args.labeled_dir),
        verbose=verbose,
    )
    print(f"\n  Dataset: X={X.shape}  y={y.shape}  dtype={X.dtype}")

    # ── 2. Split (video-wise) ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = video_wise_split(
        X, y, video_ids,
        test_size = args.test_size,
        seed      = args.seed,
    )

    # ── 3. Train ─────────────────────────────────────────────────────────────
    clf = train_random_forest(
        X_train,
        y_train,
        n_estimators     = args.n_estimators,
        min_samples_leaf = args.min_samples_leaf,
        seed             = args.seed,
    )

    # Enable OOB score retroactively if requested
    if args.oob_score:
        clf.set_params(oob_score=True)
        clf.fit(X_train, y_train)

    # ── 4. Evaluate ──────────────────────────────────────────────────────────
    evaluate(clf, X_test, y_test)

    # ── 5. Export ────────────────────────────────────────────────────────────
    save_model(clf, Path(args.out_dir))

    print("\n[RF Pipeline] Done.\n")


if __name__ == "__main__":
    main()
