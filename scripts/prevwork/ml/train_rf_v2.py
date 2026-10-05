"""
scripts/train_rf_v2.py
══════════════════════
Random Forest v2 — Per-Exercise Models + Temporal Delta Features

Improvements over v1 (train_rf.py, 54% accuracy)
──────────────────────────────────────────────────
v1 had two fundamental flaws:

  Flaw 1 — No temporal context
    A single-frame feature vector cannot distinguish descent from ascent.
    A 90° knee angle looks identical whether the user is going *down* into
    a squat or coming *up* from one.  The model saw static snapshots and had
    no signal about the direction of motion.

  Flaw 2 — One global model for all exercises
    "Neutral" in a squat (both legs straight, hips over ankles) has similar
    joint angles to "Neutral" in a bicep curl (arms hanging at sides).
    "Flexed" in a curl (elbow ~45°) has similar hip/knee geometry to "Down"
    in a squat.  A single RF was forced to separate state×exercise combos
    that are genuinely indistinguishable from static angles alone.

Solutions
─────────
  Fix 1 — Temporal delta features (Δ at t−5)
    For each frame t we compute:
      • 8 base angles at frame t                    (current pose)
      • 8 delta angles = angles[t] − angles[t−5]   (velocity / direction)
    Total: 16 features per frame.

    The delta window is fixed at DELTA_LAG = 5 frames.  At 30 fps this covers
    ~167 ms — enough to tell "knee bending" from "knee straightening" without
    including so much history that the direction signal becomes stale.

    For the first 5 frames of a video (t < DELTA_LAG), the delta is computed
    against frame 0 (i.e. angles[t] − angles[0]), which is a small but
    non-zero value if the user has already started moving.  Using zeros would
    be a worse default — it implies no motion, which is only true at t=0.

  Fix 2 — One RF model per exercise
    Data is grouped by exercise folder name before any training.
    Each exercise gets its own RandomForestClassifier trained exclusively on
    its own frames.  This means:
      • The model can never confuse squat-neutral with curl-neutral, because
        they live in completely separate models.
      • Each model's output classes are exactly the states present in *that*
        exercise's training data — no phantom classes.

Feature vector layout (16 features)
─────────────────────────────────────
  Index  Name                Description
  ─────  ──────────────────  ─────────────────────────────────────────────────
  0      L_Knee              Left knee angle at frame t
  1      R_Knee              Right knee angle at frame t
  2      L_Hip               Left hip angle at frame t
  3      R_Hip               Right hip angle at frame t
  4      L_Elbow             Left elbow angle at frame t
  5      R_Elbow             Right elbow angle at frame t
  6      L_Shoulder          Left shoulder angle at frame t
  7      R_Shoulder          Right shoulder angle at frame t
  8      Δ_L_Knee            L_Knee[t] − L_Knee[t−5]
  9      Δ_R_Knee            R_Knee[t] − R_Knee[t−5]
  10     Δ_L_Hip             L_Hip[t] − L_Hip[t−5]
  11     Δ_R_Hip             R_Hip[t] − R_Hip[t−5]
  12     Δ_L_Elbow           L_Elbow[t] − L_Elbow[t−5]
  13     Δ_R_Elbow           R_Elbow[t] − R_Elbow[t−5]
  14     Δ_L_Shoulder        L_Shoulder[t] − L_Shoulder[t−5]
  15     Δ_R_Shoulder        R_Shoulder[t] − R_Shoulder[t−5]

  Negative delta → joint is bending (e.g. knee going from 180° → 90°)
  Positive delta → joint is extending (e.g. elbow going from 45° → 160°)

Output
──────
    checkpoints/rf_models_dict.pkl   — {exercise_name: RandomForestClassifier}
    checkpoints/rf_v2_features.json  — feature names, angle defs, per-exercise metadata
    Printed: per-exercise classification report + confusion matrix

Usage
─────
    python scripts/train_rf_v2.py
    python scripts/train_rf_v2.py --processed-dir data/processed \\
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
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ── Constants ──────────────────────────────────────────────────────────────────

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

# Number of frames to look back for delta features.
# At 30 fps, 5 frames ≈ 167 ms — enough to detect direction without staleness.
DELTA_LAG: int = 5

# Each entry: (feature_name, point_a_idx, vertex_idx, point_c_idx)
ANGLE_DEFS: List[Tuple[str, int, int, int]] = [
    ("L_Knee",      11,  13,  15),   # hip – knee – ankle
    ("R_Knee",      12,  14,  16),
    ("L_Hip",        5,  11,  13),   # shoulder – hip – knee
    ("R_Hip",        6,  12,  14),
    ("L_Elbow",      5,   7,   9),   # shoulder – elbow – wrist
    ("R_Elbow",      6,   8,  10),
    ("L_Shoulder",  11,   5,   7),   # hip – shoulder – elbow
    ("R_Shoulder",  12,   6,   8),
]

N_BASE:         int       = len(ANGLE_DEFS)
BASE_NAMES:     List[str] = [d[0] for d in ANGLE_DEFS]
DELTA_NAMES:    List[str] = [f"Δ_{n}" for n in BASE_NAMES]
ALL_FEAT_NAMES: List[str] = BASE_NAMES + DELTA_NAMES
N_FEATURES:     int       = len(ALL_FEAT_NAMES)   # 16


# ═══════════════════════════════════════════════════════════════════════════════
#  Angle calculation  (unchanged from v1)
# ═══════════════════════════════════════════════════════════════════════════════

def _is_occluded(pt: np.ndarray) -> bool:
    return float(pt[0]) == 0.0 and float(pt[1]) == 0.0


def _angle_at_vertex(a: np.ndarray, v: np.ndarray, c: np.ndarray) -> float:
    """
    Angle at vertex v formed by rays v→a and v→c, in degrees.
    Returns 0.0 for occluded or degenerate inputs.
    """
    if _is_occluded(a) or _is_occluded(v) or _is_occluded(c):
        return 0.0
    ba = (a - v).astype(np.float64)
    bc = (c - v).astype(np.float64)
    na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if na < 1e-6 or nc < 1e-6:
        return 0.0
    return float(np.degrees(np.arccos(float(np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0)))))


def calculate_angles(keypoints: np.ndarray) -> np.ndarray:
    """
    8 joint angles from one (17, 2) COCO keypoint frame.
    Returns float32 array of shape (8,).  Occluded joints → 0.0.
    """
    angles = np.zeros(N_BASE, dtype=np.float32)
    for i, (_, a, v, c) in enumerate(ANGLE_DEFS):
        angles[i] = _angle_at_vertex(keypoints[a], keypoints[v], keypoints[c])
    return angles


# ═══════════════════════════════════════════════════════════════════════════════
#  Temporal delta features
# ═══════════════════════════════════════════════════════════════════════════════

def compute_angle_sequence(kp_seq: np.ndarray) -> np.ndarray:
    """
    Compute the (N, 8) base-angle matrix for an entire video sequence.

    Parameters
    ----------
    kp_seq : (N, 17, 2)  raw keypoints for one video

    Returns
    -------
    np.ndarray  (N, 8)  float32
    """
    return np.vstack([calculate_angles(kp_seq[f]) for f in range(len(kp_seq))])


def compute_delta_features(angles_seq: np.ndarray, lag: int = DELTA_LAG) -> np.ndarray:
    """
    Compute the (N, 8) delta matrix: angles[t] − angles[max(t−lag, 0)].

    For t < lag the reference frame is frame 0, so the delta reflects how
    much the joint has moved since the very start of the recording rather
    than being a spurious zero.

    Parameters
    ----------
    angles_seq : (N, 8)  pre-computed base angles for one video
    lag        : int     look-back window in frames (default: 5)

    Returns
    -------
    np.ndarray  (N, 8)  float32  — signed velocity in degrees/lag-frames
    """
    N = len(angles_seq)
    deltas = np.zeros_like(angles_seq)          # (N, 8)
    for t in range(N):
        ref   = max(t - lag, 0)                  # clamp to frame 0 for early frames
        deltas[t] = angles_seq[t] - angles_seq[ref]
    return deltas.astype(np.float32)


def build_feature_matrix(kp_seq: np.ndarray) -> np.ndarray:
    """
    Full 16-feature matrix for one video: [base_angles | delta_angles].

    Parameters
    ----------
    kp_seq : (N, 17, 2)

    Returns
    -------
    np.ndarray  (N, 16)  float32
        Columns 0–7  : base angles at t
        Columns 8–15 : delta angles (angles[t] − angles[t−5])
    """
    base   = compute_angle_sequence(kp_seq)      # (N, 8)
    deltas = compute_delta_features(base)         # (N, 8)
    return np.hstack([base, deltas]).astype(np.float32)   # (N, 16)


# ═══════════════════════════════════════════════════════════════════════════════
#  Data loading — grouped by exercise
# ═══════════════════════════════════════════════════════════════════════════════

def load_data_by_exercise(
    processed_dir: Path,
    labeled_dir:   Path,
    verbose:       bool = True,
) -> Dict[str, Dict]:
    """
    Load all videos and group them by exercise name.

    Returns
    -------
    dict keyed by exercise_name, each value:
        {
            "X":         (N_frames, 16)  float32  — feature matrix
            "y":         (N_frames,)     int32    — rep_state labels
            "video_ids": list[str]       length N_frames — for video-wise split
        }
    """
    # ex_name → lists to accumulate across videos
    data: Dict[str, Dict[str, list]] = {}

    skipped    = 0
    total_vids = 0

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

                # Validate labels
                invalid = (rep_states < 0) | (rep_states >= N_STATES)
                if invalid.any():
                    if verbose:
                        print(f"  [WARN] {vid_key}: {invalid.sum()} invalid "
                              f"state indices — clipped to 0")
                    rep_states[invalid] = 0

                # ── 16-feature matrix for this video ────────────────────────
                X_video = build_feature_matrix(kp)   # (N, 16)

                # Accumulate per exercise
                if ex_name not in data:
                    data[ex_name] = {"X": [], "y": [], "video_ids": []}
                data[ex_name]["X"].append(X_video)
                data[ex_name]["y"].append(rep_states)
                data[ex_name]["video_ids"].extend([vid_key] * N)

                total_vids += 1
                if verbose:
                    print(f"  [load] {vid_key}: {N} frames  "
                          f"states={np.unique(rep_states).tolist()}")

            except Exception as exc:
                if verbose:
                    print(f"  [WARN] Failed {vid_key}: {exc}")
                skipped += 1
                continue

    if not data:
        raise RuntimeError(
            "No data loaded. Check --processed-dir and --labeled-dir paths."
        )

    # Consolidate lists → arrays
    result: Dict[str, Dict] = {}
    for ex_name, parts in data.items():
        result[ex_name] = {
            "X":         np.vstack(parts["X"]).astype(np.float32),
            "y":         np.concatenate(parts["y"]).astype(np.int32),
            "video_ids": parts["video_ids"],
        }

    if verbose:
        print(f"\n  Loaded {total_vids} videos across "
              f"{len(result)} exercises  |  skipped {skipped}")
        for ex, d in sorted(result.items()):
            counts = np.bincount(d["y"], minlength=N_STATES)
            state_strs = [f"{STATE_NAMES[i]}={counts[i]}" for i in range(N_STATES)
                          if counts[i] > 0]
            n_vids = len(set(d["video_ids"]))
            print(f"    {ex:<20}: {d['X'].shape[0]:5d} frames, "
                  f"{n_vids} videos — {', '.join(state_strs)}")

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Video-wise split  (same logic as v1, extracted as a clean utility)
# ═══════════════════════════════════════════════════════════════════════════════

def video_wise_split(
    X:         np.ndarray,
    y:         np.ndarray,
    video_ids: List[str],
    test_size: float = 0.20,
    seed:      int   = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split by video ID so no video's frames appear in both train and test.

    With a small dataset (e.g. 5 videos per exercise), ensures at least
    1 video is always held out for testing regardless of test_size rounding.
    """
    unique_vids = sorted(set(video_ids))
    rng         = np.random.default_rng(seed)
    rng.shuffle(unique_vids)

    n_test    = max(1, int(len(unique_vids) * test_size))
    test_set  = set(unique_vids[-n_test:])

    vid_arr    = np.array(video_ids)
    train_mask = ~np.isin(vid_arr, list(test_set))
    test_mask  =  np.isin(vid_arr, list(test_set))

    return (
        X[train_mask], X[test_mask],
        y[train_mask], y[test_mask],
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-exercise training & evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_exercise(
    ex_name:          str,
    X_train:          np.ndarray,
    y_train:          np.ndarray,
    n_estimators:     int   = 300,
    min_samples_leaf: int   = 2,
    seed:             int   = 42,
) -> RandomForestClassifier:
    """
    Train a RandomForestClassifier for a single exercise.

    class_weight="balanced" handles state imbalance within the exercise
    (e.g. most squat frames are "neutral" or "down", very few are "jump").
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


def evaluate_one_exercise(
    ex_name: str,
    clf:     RandomForestClassifier,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
) -> float:
    """
    Print a per-exercise classification report + visual confusion matrix.
    Returns overall accuracy for the summary table.
    """
    y_pred   = clf.predict(X_test)
    present  = sorted(set(y_test.tolist()))
    labels   = [STATE_NAMES[i] for i in present]

    print(f"\n{'═'*64}")
    print(f"  EXERCISE: {ex_name.upper()}")
    print(f"  Test frames: {len(y_test)}  |  "
          f"States: {labels}")
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
    # Header row
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
    print(f"\n  Accuracy: {acc:.4f} ({acc*100:.1f}%)")

    # ── Top-5 most important features for this exercise ───────────────────────
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
    models_dict:  Dict[str, RandomForestClassifier],
    per_ex_meta:  Dict[str, Dict],
    out_dir:      Path,
) -> None:
    """
    Save the dict of trained models and a companion JSON metadata file.

    The metadata documents what feature vector to pass at inference:
    build_feature_matrix(kp_seq) on the current video window, then take the
    last row as the single-frame feature vector for real-time prediction.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    models_path  = out_dir / "rf_models_dict.pkl"
    meta_path    = out_dir / "rf_v2_features.json"

    joblib.dump(models_dict, models_path)

    meta = {
        "version":       "v2",
        "n_features":    N_FEATURES,
        "feature_names": ALL_FEAT_NAMES,
        "base_features": BASE_NAMES,
        "delta_features":DELTA_NAMES,
        "delta_lag":     DELTA_LAG,
        "angle_defs": [
            {"name": name, "point_a": a, "vertex": v, "point_c": c}
            for name, a, v, c in ANGLE_DEFS
        ],
        "state_names":   STATE_NAMES,
        "n_states":      N_STATES,
        "exercises":     per_ex_meta,
        "inference_note": (
            "At runtime: maintain a rolling deque of the last DELTA_LAG+1 "
            "normalised keypoint frames. For each new frame t, call "
            "calculate_angles(kp[t]) for base angles and subtract "
            "calculate_angles(kp[max(t-DELTA_LAG, 0)]) for deltas. "
            "Concatenate → 16-feature vector. "
            "Call models_dict[exercise_name].predict([feat_vec])."
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
            "Train per-exercise RF classifiers with 16 temporal features "
            "(8 angles + 8 deltas at t−5)"
        )
    )
    p.add_argument("--processed-dir",    default="data/processed")
    p.add_argument("--labeled-dir",      default="data/labeled")
    p.add_argument("--out-dir",          default="checkpoints")
    p.add_argument("--test-size",        type=float, default=0.20,
                   help="Fraction of videos held out per exercise (default: 0.20)")
    p.add_argument("--n-estimators",     type=int,   default=300)
    p.add_argument("--min-samples-leaf", type=int,   default=2)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--quiet",            action="store_true")
    args = p.parse_args()

    verbose = not args.quiet

    print("\n[RF v2]  Per-Exercise Random Forest  +  Temporal Δ Features")
    print(f"  Features per frame : {N_FEATURES} "
          f"(8 base angles + 8 deltas at t−{DELTA_LAG})")
    print(f"  processed_dir      : {args.processed_dir}")
    print(f"  labeled_dir        : {args.labeled_dir}")
    print(f"  test_size          : {args.test_size:.0%} (video-wise per exercise)")
    print(f"  n_estimators       : {args.n_estimators}\n")

    # ── 1. Load grouped by exercise ───────────────────────────────────────────
    exercise_data = load_data_by_exercise(
        Path(args.processed_dir),
        Path(args.labeled_dir),
        verbose=verbose,
    )

    # ── 2. Per-exercise: split → train → evaluate ─────────────────────────────
    models_dict: Dict[str, RandomForestClassifier] = {}
    per_ex_meta: Dict[str, Dict]                   = {}
    summary_rows: List[Tuple[str, int, int, float]] = []

    for ex_name in sorted(exercise_data.keys()):
        d          = exercise_data[ex_name]
        X, y, vids = d["X"], d["y"], d["video_ids"]
        n_videos   = len(set(vids))

        print(f"\n{'─'*64}")
        print(f"  [{ex_name}]  {X.shape[0]:,} frames  |  {n_videos} videos")

        if n_videos < 2:
            print(f"  [SKIP] Only {n_videos} video(s) — need at least 2 for a split.")
            continue

        # Video-wise split
        X_tr, X_te, y_tr, y_te = video_wise_split(
            X, y, vids, test_size=args.test_size, seed=args.seed
        )
        print(f"  Train: {len(X_tr):,} frames  |  Test: {len(X_te):,} frames")

        # Skip if test set has no samples
        if len(X_te) == 0:
            print(f"  [SKIP] Empty test set.")
            continue

        # Train
        clf = train_one_exercise(
            ex_name,
            X_tr, y_tr,
            n_estimators     = args.n_estimators,
            min_samples_leaf = args.min_samples_leaf,
            seed             = args.seed,
        )

        # Evaluate
        acc = evaluate_one_exercise(ex_name, clf, X_te, y_te)

        models_dict[ex_name] = clf
        per_ex_meta[ex_name] = {
            "n_train":      int(len(X_tr)),
            "n_test":       int(len(X_te)),
            "n_videos":     n_videos,
            "states_seen":  [STATE_NAMES[i] for i in sorted(set(y.tolist()))],
            "test_accuracy": round(acc, 4),
        }
        summary_rows.append((ex_name, len(X_tr), len(X_te), acc))

    # ── 3. Summary table ─────────────────────────────────────────────────────
    print(f"\n{'═'*64}")
    print("  SUMMARY")
    print(f"  {'Exercise':<20}  {'Train':>6}  {'Test':>6}  {'Accuracy':>9}")
    print(f"  {'─'*20}  {'─'*6}  {'─'*6}  {'─'*9}")
    total_test = total_correct = 0
    for ex_name, n_tr, n_te, acc in summary_rows:
        print(f"  {ex_name:<20}  {n_tr:>6,}  {n_te:>6,}  {acc*100:>8.1f}%")
        total_test    += n_te
        total_correct += int(acc * n_te)
    if total_test > 0:
        overall = total_correct / total_test
        print(f"  {'─'*20}  {'─'*6}  {'─'*6}  {'─'*9}")
        print(f"  {'OVERALL (weighted)':<20}  {'':>6}  {total_test:>6,}  "
              f"{overall*100:>8.1f}%")
    print(f"{'═'*64}")

    if not models_dict:
        print("\n[ERROR] No models trained — check data paths and video counts.")
        sys.exit(1)

    # ── 4. Export ─────────────────────────────────────────────────────────────
    save_models(models_dict, per_ex_meta, Path(args.out_dir))
    print("\n[RF v2]  Done.\n")


if __name__ == "__main__":
    main()
