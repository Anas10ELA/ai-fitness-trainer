"""
scripts/extract_keypoints.py
════════════════════════════
المرحلة الأولى: استخراج الـ keypoints من الفيديوهات

المدخل:   data/raw/{exercise}/{video}.mp4
المخرج:   data/processed/{exercise}/{video}.npz
          keypoints (N,17,2)  scores (N,17)  frame_ids (N,)

Strategy
────────
  • Locked bbox: YOLO runs once on the first 30 frames, locks onto the largest
    person (closest to camera = the one exercising), then RTMPose uses that
    fixed bbox for every frame.  YOLO never runs again → no jumping to
    background people, ~3× faster.

  • Kalman smoothing: raw RTMPose keypoints are passed through the SAME
    KalmanKeypoints filter main.py uses at live inference, in temporal order,
    called identically (smooth(kp), no scores).  This guarantees the .npz files
    used for labeling and training carry the exact smoothed signal the model
    sees at runtime — eliminating the train/serve jitter mismatch.

Usage
─────
  python scripts/extract_keypoints.py --overwrite
  python scripts/extract_keypoints.py --skip 1 --overwrite
  python scripts/extract_keypoints.py --no-smooth        # raw (not recommended)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from exercises import Exercise
from detector  import PersonDetector
from pose      import PoseEstimator
from utils     import select_largest_bbox, KalmanKeypoints

EXERCISE_NAMES = [e.value for e in Exercise]
SUPPORTED_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


# ══════════════════════════════════════════════════════════════════════════════
#  Extractor
# ══════════════════════════════════════════════════════════════════════════════

class KeypointExtractor:
    """
    Locked-bbox + Kalman smoothing.

    CRITICAL — train/inference parity
    ──────────────────────────────────
    Keypoints pass through the SAME KalmanKeypoints filter main.py applies
    live, in temporal order, called the same way (smooth(kp), no scores).
    A fresh filter is created per video (it is stateful and must reset at
    each video boundary).  Keypoints stay in original full-frame pixel coords.
    """

    def __init__(self, det_conf: float = 0.45, skip: int = 1,
                 smooth: bool = True) -> None:
        print("[Extractor] Loading YOLOv8 …")
        self.detector = PersonDetector(conf_threshold=det_conf)
        print("[Extractor] Loading RTMPose …")
        self.pose = PoseEstimator()
        self.skip   = skip
        self.smooth = smooth

    def extract(self, video_path: str) -> Optional[dict]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  [WARN] Cannot open: {video_path}")
            return None

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        print(f"  Frames: {total}  FPS: {fps:.1f}  Skip: {self.skip}")

        # ── Step 1: Lock on the target person once ────────────────────────────
        locked_bbox = self._find_target_bbox(cap, max_search_frames=30)
        if locked_bbox is None:
            print("  [WARN] Could not detect a person in first 30 frames.")
            cap.release()
            return None

        x1, y1, x2, y2 = locked_bbox
        print(f"  Locked bbox: ({x1:.0f},{y1:.0f})→({x2:.0f},{y2:.0f})")

        # ── Step 2: Restart from frame 0 ──────────────────────────────────────
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Fresh Kalman for THIS video (stateful — resets per video).
        # Called exactly as main.py: smooth(kp), keypoints only, no scores.
        kalman = KalmanKeypoints() if self.smooth else None

        keypoints_list: List[np.ndarray] = []
        scores_list:    List[np.ndarray] = []
        frame_ids_list: List[int]        = []

        frame_idx = 0
        processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.skip == 0:
                result = self.pose.estimate_with_scores(frame, locked_bbox)

                if result is not None:
                    kp, sc = result

                    # Kalman smoothing — identical to live inference path.
                    # main.py: kp = self._smoother.smooth(kp)
                    if kalman is not None:
                        kp = kalman.smooth(kp)

                    keypoints_list.append(kp)
                    scores_list.append(sc)
                    frame_ids_list.append(frame_idx)
                    processed += 1

                    if processed % 50 == 0:
                        print(f"  Processed {processed} frames …", end="\r")

            frame_idx += 1

        cap.release()
        tag = "Kalman ON" if self.smooth else "Kalman OFF (raw)"
        print(f"  ✓ Extracted {processed} valid frames from {frame_idx} total  [{tag}]")

        if processed < 10:
            print(f"  [WARN] Too few frames ({processed}), skipping.")
            return None

        return {
            "keypoints": np.array(keypoints_list,  dtype=np.float32),
            "scores":    np.array(scores_list,      dtype=np.float32),
            "frame_ids": np.array(frame_ids_list,   dtype=np.int32),
        }

    def _find_target_bbox(self, cap, max_search_frames: int = 30) -> Optional[tuple]:
        """Scan first frames, return largest bbox = person doing the exercise."""
        best_bbox = None
        best_area = 0.0
        for _ in range(max_search_frames):
            ret, frame = cap.read()
            if not ret:
                break
            bboxes = self.detector.detect(frame)
            if not bboxes:
                continue
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox[:4]
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    best_bbox = (x1, y1, x2, y2)
        return best_bbox


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset scanner
# ══════════════════════════════════════════════════════════════════════════════

def scan_videos(data_dir: Path) -> List[Tuple[str, str, Path]]:
    entries = []
    for exercise_dir in sorted(data_dir.iterdir()):
        if not exercise_dir.is_dir():
            continue
        ex_name = exercise_dir.name.lower()
        if ex_name not in EXERCISE_NAMES:
            print(f"[WARN] Unknown exercise folder: {ex_name}  (skipping)")
            continue
        for video_file in sorted(exercise_dir.iterdir()):
            if video_file.suffix.lower() in SUPPORTED_EXTS:
                entries.append((ex_name, video_file.stem, video_file))
    return entries


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",  default="data/raw")
    parser.add_argument("--out-dir",   default="data/processed")
    parser.add_argument("--skip",      type=int,   default=1,
                        help="Process every N-th frame. Default 1 = all frames "
                             "(use 2 only for >60fps footage).")
    parser.add_argument("--det-conf",  type=float, default=0.45)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-smooth", action="store_true",
                        help="Disable Kalman smoothing (store raw keypoints). "
                             "NOT recommended — breaks train/inference parity.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)

    if not data_dir.exists():
        print(f"[ERROR] data-dir not found: {data_dir}")
        sys.exit(1)

    videos = scan_videos(data_dir)
    if not videos:
        print(f"[ERROR] No videos found under {data_dir}")
        sys.exit(1)

    print(f"\n[INFO] Found {len(videos)} videos")
    print(f"[INFO] Output → {out_dir}")
    print(f"[INFO] Smoothing: {'ON (Kalman, matches live)' if not args.no_smooth else 'OFF (raw)'}\n")

    extractor = KeypointExtractor(
        det_conf=args.det_conf, skip=args.skip, smooth=not args.no_smooth
    )
    stats = {"ok": 0, "skip": 0, "fail": 0}

    for ex_name, video_id, video_path in videos:
        out_ex_dir = out_dir / ex_name
        out_ex_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_ex_dir / f"{video_id}.npz"

        if out_file.exists() and not args.overwrite:
            print(f"[SKIP] {ex_name}/{video_id}")
            stats["skip"] += 1
            continue

        print(f"\n[EXTRACT] {ex_name} / {video_id}")
        result = extractor.extract(str(video_path))
        if result is None:
            stats["fail"] += 1
            continue

        np.savez_compressed(
            out_file,
            keypoints = result["keypoints"],
            scores    = result["scores"],
            frame_ids = result["frame_ids"],
            exercise  = np.array(ex_name),
            video_id  = np.array(video_id),
        )
        print(f"  Saved → {out_file}  ({len(result['frame_ids'])} frames)")
        stats["ok"] += 1

    print(f"\n{'═'*50}")
    print(f"  Done.  OK:{stats['ok']}  Skipped:{stats['skip']}  Failed:{stats['fail']}")
    print(f"{'═'*50}\n")


if __name__ == "__main__":
    main()
