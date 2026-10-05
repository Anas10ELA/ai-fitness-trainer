"""
scripts/build_dataset.py
════════════════════════
ST-GCN Data Pipeline — COCO-17 Skeleton

Builds training-ready tensors from labeled keypoint sequences.

Pipeline
--------
1. Load every processed .npz  +  its matching labeled .npz
2. Normalize keypoints        (hip-center translation + shoulder-hip scale)
3. Sliding window             (default T=60, stride=16)
4. Reshape to ST-GCN format   (C=2, T, V=17, M=1)
5. Build the COCO-17 Adjacency Matrix A  (normalized, with self-loops)
6. Train / Val / Test split   (70 / 15 / 15 by video)
7. Save:
     data/processed/train.npz
     data/processed/val.npz
     data/processed/test.npz
     data/processed/graph_A.npy   ← shared adjacency matrix

Tensor format
-------------
Each sample X has shape  (C=2, T, V=17, M=1)  stored as float32.
The saved .npz arrays therefore have shape  (N_samples, 2, T, 17, 1).

Labels per sample
-----------------
  y_ex    int64   exercise class index
  y_state int64   state index (0-6, see STATE_TO_IDX)
  y_form  float32 mean form score of window

State vocabulary (7 states)
-----------------------------
  0 neutral   — resting / start position
  1 down      — squat bottom / hip hinge / low point
  2 up        — standing / lockout / top position
  3 plank     — horizontal isometric hold
  4 jump      — airborne / explosive phase
  5 extended  — full limb extension (overhead press lockout, leg extension)
  6 flexed    — peak concentric contraction (bicep top, leg curl top)

Adjacency Matrix
----------------
Edges follow the natural COCO-17 skeleton connectivity (16 edges).
Self-loops (I) are added before symmetric degree normalisation:
    Â = A + I
    D̂_ii = Σ_j Â_ij
    A_norm = D̂^{-½}  Â  D̂^{-½}

The resulting 17×17 float32 matrix is saved to graph_A.npy and can be
loaded directly as a fixed graph in ST-GCN's SpatialGraphConv layer.

Usage
-----
    python scripts/build_dataset.py
    python scripts/build_dataset.py --window 60 --stride 16
    python scripts/build_dataset.py --window 30 --stride 8 --seed 0
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from exercises import Exercise

# ── Label indices ──────────────────────────────────────────────────────────────

EXERCISE_TO_IDX: Dict[str, int] = {e.value: i for i, e in enumerate(Exercise)}
IDX_TO_EXERCISE: Dict[int, str] = {i: e.value for i, e in enumerate(Exercise)}

STATE_TO_IDX: Dict[str, int] = {
    "neutral":  0,
    "down":     1,
    "up":       2,
    "plank":    3,
    "jump":     4,
    "extended": 5,
    "flexed":   6,
}
IDX_TO_STATE: Dict[int, str] = {v: k for k, v in STATE_TO_IDX.items()}

N_KEYPOINTS  = 17
N_COORDS     = 2          # x, y
N_PERSONS    = 1          # M dimension (single-person videos)
N_EXERCISES  = len(Exercise)
N_STATES     = len(STATE_TO_IDX)


# ═══════════════════════════════════════════════════════════════════════════════
#  Adjacency Matrix  (COCO-17, symmetric, degree-normalized)
# ═══════════════════════════════════════════════════════════════════════════════

# COCO-17 keypoint indices:
#   0  nose          1  left_eye       2  right_eye
#   3  left_ear      4  right_ear      5  left_shoulder
#   6  right_shoulder  7  left_elbow   8  right_elbow
#   9  left_wrist   10  right_wrist   11  left_hip
#  12  right_hip    13  left_knee     14  right_knee
#  15  left_ankle   16  right_ankle

COCO_EDGES: List[Tuple[int, int]] = [
    # Head
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Shoulder bar
    (5, 6),
    # Left arm
    (5, 7), (7, 9),
    # Right arm
    (6, 8), (8, 10),
    # Torso sides
    (5, 11), (6, 12),
    # Hip bar
    (11, 12),
    # Left leg
    (11, 13), (13, 15),
    # Right leg
    (12, 14), (14, 16),
]


def build_adjacency_matrix(
    n_joints: int = N_KEYPOINTS,
    edges:    List[Tuple[int, int]] = COCO_EDGES,
) -> np.ndarray:
    """
    Build the symmetric, degree-normalised adjacency matrix for the COCO-17 graph.

    Steps
    -----
    1. Construct raw adjacency A from edge list  (symmetric)
    2. Add self-loops:  Â = A + I
    3. Compute degree matrix D̂  where  D̂_ii = Σ_j Â_ij
    4. Normalise:  A_norm = D̂^{-½} · Â · D̂^{-½}

    Returns
    -------
    np.ndarray  shape (17, 17)  dtype float32
        Drop-in graph for ST-GCN's SpatialGraphConv.
    """
    A = np.zeros((n_joints, n_joints), dtype=np.float32)

    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0   # symmetric

    # Self-loops
    A_hat = A + np.eye(n_joints, dtype=np.float32)

    # Degree matrix and its inverse square-root
    degree   = A_hat.sum(axis=1)                          # (17,)
    d_inv_sq = np.where(degree > 0, degree ** -0.5, 0.0)  # (17,)
    D_inv_sq = np.diag(d_inv_sq)                          # (17,17)

    # Symmetric normalisation
    A_norm = D_inv_sq @ A_hat @ D_inv_sq

    return A_norm.astype(np.float32)


# ── Pre-build once; reused for every sample ───────────────────────────────────
GRAPH_A: np.ndarray = build_adjacency_matrix()


# ═══════════════════════════════════════════════════════════════════════════════
#  Normalization  (hip-center + shoulder-hip scale)
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_sequence(
    kp_seq:         np.ndarray,    # (N, 17, 2)
    sc_seq:         np.ndarray,    # (N, 17)
    conf_threshold: float = 0.25,
) -> np.ndarray:
    """
    Normalize keypoints to be translation- and scale-invariant.

    • Translation: subtract hip midpoint (joints 11 & 12) each frame
    • Scale:       divide by shoulder-to-hip distance (joints 5/6 → 11/12)
    • Confidence:  zero out joints below `conf_threshold`

    This is the *only* normalisation step in the pipeline.
    The same logic runs at inference time inside FitnessInferenceEngine.

    Returns
    -------
    np.ndarray  shape (N, 17, 2)  float32
    """
    kp = kp_seq.copy().astype(np.float32)

    if sc_seq is not None:
        mask       = sc_seq < conf_threshold   # (N, 17) bool
        kp[mask]   = 0.0

    N   = len(kp)
    out = np.zeros_like(kp)

    for i in range(N):
        frame = kp[i]   # (17, 2)

        # Hip midpoint as origin (joints 11=left_hip, 12=right_hip)
        lh, rh = frame[11], frame[12]
        valid_hips = [p for p in [lh, rh] if not (p[0] == 0 and p[1] == 0)]
        if not valid_hips:
            out[i] = frame   # fallback: keep raw
            continue
        hip_center = np.mean(valid_hips, axis=0)   # (2,)

        # Scale: shoulder-to-hip distance (joints 5=L_shoulder, 6=R_shoulder)
        ls, rs = frame[5], frame[6]
        valid_sh = [p for p in [ls, rs] if not (p[0] == 0 and p[1] == 0)]
        scale = 1.0
        if valid_sh:
            shoulder_center = np.mean(valid_sh, axis=0)
            scale = float(np.linalg.norm(shoulder_center - hip_center))
            scale = max(scale, 1e-6)

        nonzero           = ~((frame[:, 0] == 0) & (frame[:, 1] == 0))
        centered          = frame.copy()
        centered[nonzero] = (frame[nonzero] - hip_center) / scale
        out[i]            = centered

    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  Sliding window → ST-GCN tensor  (C, T, V, M)
# ═══════════════════════════════════════════════════════════════════════════════

def sliding_window_stgcn(
    kp_norm:      np.ndarray,   # (N, 17, 2)  normalized
    rep_states:   np.ndarray,   # (N,)        int  0-6
    form_scores:  np.ndarray,   # (N,)        float
    exercise_idx: int,
    window:       int = 60,
    stride:       int = 16,
) -> Tuple[List[np.ndarray], List[int], List[int], List[float]]:
    """
    Slide a fixed-length window over the sequence and produce ST-GCN tensors.

    For each window of length T:
      • X       shape (C=2, T, V=17, M=1)  float32
                — the canonical ST-GCN input tensor
      • y_ex    exercise class index  (int)
      • y_state rep_state of the CENTER frame  (int, 0-6)
      • y_form  mean form_score of all T frames  (float)

    Tensor construction
    -------------------
    window_kp  : (T, V, C)  — raw window slice
    transposed : (C, T, V)  — swap axes so channel is first
    expanded   : (C, T, V, M=1)  — add person dimension

    Returns (X_list, y_ex_list, y_state_list, y_form_list)
    """
    N = len(kp_norm)
    X_list, y_ex_list, y_state_list, y_form_list = [], [], [], []

    if N < window:
        return X_list, y_ex_list, y_state_list, y_form_list

    for start in range(0, N - window + 1, stride):
        end = start + window
        mid = start + window // 2   # center frame index

        # ── (T, V, C) slice ─────────────────────────────────────────────────
        window_kp = kp_norm[start:end]               # (T, V=17, C=2)

        # ── (C, T, V) — put channel first ───────────────────────────────────
        # np.transpose maps axis 0→1, 1→2, 2→0  i.e.  (T,V,C) → (C,T,V)
        stgcn_tensor = window_kp.transpose(2, 0, 1)  # (C=2, T, V=17)

        # ── (C, T, V, M=1) — add person/body dimension ──────────────────────
        stgcn_tensor = stgcn_tensor[:, :, :, np.newaxis]  # (2, T, 17, 1)

        center_state = int(rep_states[mid])
        window_form  = float(np.mean(form_scores[start:end]))

        X_list.append(stgcn_tensor.astype(np.float32))
        y_ex_list.append(exercise_idx)
        y_state_list.append(center_state)
        y_form_list.append(window_form)

    return X_list, y_ex_list, y_state_list, y_form_list


# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset builder
# ═══════════════════════════════════════════════════════════════════════════════

def build(args) -> None:
    proc_dir    = Path(args.processed_dir)
    labeled_dir = Path(args.labeled_dir)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    window = args.window
    stride = args.stride
    seed   = args.seed
    random.seed(seed)
    np.random.seed(seed)

    # ── Save the shared adjacency matrix once ─────────────────────────────────
    graph_path = out_dir / "graph_A.npy"
    np.save(graph_path, GRAPH_A)
    print(f"\n[Graph]  Adjacency matrix saved → {graph_path}  shape={GRAPH_A.shape}")

    # ── Collect labeled videos ────────────────────────────────────────────────
    records = []   # (ex_name, video_id, proc_npz, label_npz)

    for ex_dir in sorted(labeled_dir.iterdir()):
        if not ex_dir.is_dir():
            continue
        ex_name = ex_dir.name
        if ex_name not in EXERCISE_TO_IDX:
            print(f"[WARN] Unknown exercise folder: {ex_name}")
            continue

        for label_npz in sorted(ex_dir.glob("*.npz")):
            video_id = label_npz.stem
            proc_npz = proc_dir / ex_name / f"{video_id}.npz"
            if not proc_npz.exists():
                print(f"[WARN] Missing processed file: {ex_name}/{video_id}")
                continue
            records.append((ex_name, video_id, proc_npz, label_npz))

    if not records:
        print("[ERROR] No labeled videos found. Run label_tool.py first.")
        sys.exit(1)

    print(f"[Build]  {len(records)} labeled videos  |  "
          f"window={window}  stride={stride}  seed={seed}")
    print(f"         Tensor format per sample: (C=2, T={window}, V=17, M=1)\n")

    # ── Split by video (stratified by exercise where possible) ───────────────
    random.shuffle(records)
    n_train = int(len(records) * 0.70)
    n_val   = int(len(records) * 0.15)
    splits: Dict[str, list] = {
        "train": records[:n_train],
        "val":   records[n_train: n_train + n_val],
        "test":  records[n_train + n_val:],
    }
    for k, v in splits.items():
        print(f"  {k:5s}: {len(v):3d} videos")

    # ── Process each split ────────────────────────────────────────────────────
    for split_name, split_records in splits.items():
        X_all: List[np.ndarray] = []
        y_ex_all:    List[int]   = []
        y_state_all: List[int]   = []
        y_form_all:  List[float] = []

        for ex_name, video_id, proc_npz, label_npz in split_records:
            try:
                proc  = np.load(proc_npz,  allow_pickle=True)
                label = np.load(label_npz, allow_pickle=True)

                kp          = proc["keypoints"].astype(np.float32)     # (N, 17, 2)
                sc          = proc["scores"].astype(np.float32)         # (N, 17)
                rep_states  = label["rep_states"].astype(np.int32)      # (N,)
                form_scores = label["form_scores"].astype(np.float32)   # (N,)

                # Align frame counts across all arrays
                N           = min(len(kp), len(rep_states), len(form_scores))
                kp          = kp[:N]
                sc          = sc[:N]
                rep_states  = rep_states[:N]
                form_scores = form_scores[:N]

                # Validate state values
                invalid = np.where((rep_states < 0) | (rep_states >= N_STATES))[0]
                if len(invalid) > 0:
                    print(f"  [WARN] {ex_name}/{video_id}: "
                          f"{len(invalid)} frames have out-of-range state indices → clipped to 0")
                    rep_states[invalid] = 0

                # Normalize
                kp_norm = normalize_sequence(kp, sc)

                # Sliding window → ST-GCN tensors
                ex_idx = EXERCISE_TO_IDX[ex_name]
                X, y_ex, y_state, y_form = sliding_window_stgcn(
                    kp_norm, rep_states, form_scores, ex_idx, window, stride
                )

                X_all.extend(X)
                y_ex_all.extend(y_ex)
                y_state_all.extend(y_state)
                y_form_all.extend(y_form)

                print(f"  [{split_name}] {ex_name}/{video_id}: "
                      f"{len(X)} windows  tensor=({N_COORDS},{window},{N_KEYPOINTS},{N_PERSONS})")

            except Exception as exc:
                print(f"  [WARN] Failed {ex_name}/{video_id}: {exc}")
                continue

        if not X_all:
            print(f"  [WARN] No samples generated for split '{split_name}'")
            continue

        # Stack: (N_samples, C=2, T, V=17, M=1)
        X_arr       = np.stack(X_all, axis=0)               # (N, 2, T, 17, 1)
        y_ex_arr    = np.array(y_ex_all,    dtype=np.int64)
        y_state_arr = np.array(y_state_all, dtype=np.int64)
        y_form_arr  = np.array(y_form_all,  dtype=np.float32)

        out_path = out_dir / f"{split_name}.npz"
        np.savez_compressed(
            out_path,
            X       = X_arr,
            y_ex    = y_ex_arr,
            y_state = y_state_arr,
            y_form  = y_form_arr,
        )
        print(
            f"\n  ✓ {split_name}.npz\n"
            f"      X        shape={X_arr.shape}   dtype={X_arr.dtype}\n"
            f"      y_ex     shape={y_ex_arr.shape}  classes={np.unique(y_ex_arr).tolist()}\n"
            f"      y_state  shape={y_state_arr.shape}  states={np.unique(y_state_arr).tolist()}\n"
            f"      y_form   shape={y_form_arr.shape}  "
            f"min={y_form_arr.min():.2f}  max={y_form_arr.max():.2f}\n"
            f"      → saved to {out_path}"
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═' * 58}")
    print(f"  ST-GCN Dataset build complete")
    print(f"  Window={window}  Stride={stride}  Seed={seed}")
    print(f"  Tensor format : (N, C=2, T={window}, V=17, M=1)")
    print(f"  States        : {N_STATES}  {list(STATE_TO_IDX.keys())}")
    print(f"  Exercises     : {N_EXERCISES}  {list(EXERCISE_TO_IDX.keys())}")
    print(f"  Adjacency A   : {graph_path}")
    print(f"{'═' * 58}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ST-GCN training dataset from labeled COCO-17 keypoints"
    )
    parser.add_argument("--processed-dir", default="data/processed",
                        help="Directory containing per-exercise processed .npz files")
    parser.add_argument("--labeled-dir",   default="data/labeled",
                        help="Directory containing per-exercise labeled .npz files")
    parser.add_argument("--out-dir",       default="data/processed",
                        help="Output directory for train/val/test splits + graph_A.npy")
    parser.add_argument("--window", type=int, default=60,
                        help="Temporal window length T  (default: 60 frames)")
    parser.add_argument("--stride", type=int, default=16,
                        help="Sliding window stride  (default: 16 frames)")
    parser.add_argument("--seed",   type=int, default=42,
                        help="Random seed for reproducible splits")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
