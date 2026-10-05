"""
scripts/extract_keypoints.py
════════════════════════════
المرحلة الأولى: استخراج الـ keypoints من الفيديوهات

المدخل:
    data/raw/{exercise_name}/{video_id}.mp4

المخرج:
    data/processed/{exercise_name}/{video_id}.npz
    كل ملف فيه:
        keypoints  : (N_frames, 17, 2)  — pixel coords
        scores     : (N_frames, 17)     — confidence per keypoint
        frame_ids  : (N_frames,)        — original frame indices
        exercise   : str
        video_id   : str

Usage:
    python scripts/extract_keypoints.py --data-dir data/raw --out-dir data/processed
    python scripts/extract_keypoints.py --data-dir data/raw --out-dir data/processed --skip 2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

# ── Add project root to path ──────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from exercises import Exercise
from detector import PersonDetector
from pose import PoseEstimator
from utils import select_largest_bbox


# ── Constants ─────────────────────────────────────────────────────────────────

EXERCISE_NAMES = [e.value for e in Exercise]
SUPPORTED_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


# ═══════════════════════════════════════════════════════════════════════════════
#  Core extraction
# ═══════════════════════════════════════════════════════════════════════════════

class KeypointExtractor:
    """
    Runs YOLOv8 + RTMPose on every frame of a video and returns
    the keypoints array for that video.
    """

    def __init__(self, det_conf: float = 0.45, skip: int = 1) -> None:
        print("[Extractor] Loading YOLOv8 …")
        self.detector = PersonDetector(conf_threshold=det_conf)

        print("[Extractor] Loading RTMPose …")
        self.pose = PoseEstimator()

        self.skip = skip   # process every N-th frame (1 = all frames)

    def extract(self, video_path: str) -> Optional[dict]:
        """
        Extract keypoints from a video file.

        Returns
        -------
        dict with keys: keypoints (N,17,2), scores (N,17), frame_ids (N,)
        or None if extraction failed.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  [WARN] Cannot open: {video_path}")
            return None

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        print(f"  Frames: {total}  FPS: {fps:.1f}  Skip: {self.skip}")

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
                kp, sc = self._process_frame(frame)
                if kp is not None:
                    keypoints_list.append(kp)
                    scores_list.append(sc)
                    frame_ids_list.append(frame_idx)
                    processed += 1

                    if processed % 50 == 0:
                        print(f"  Processed {processed} frames …", end="\r")

            frame_idx += 1

        cap.release()
        print(f"  ✓ Extracted {processed} valid frames from {frame_idx} total")

        if processed < 10:
            print(f"  [WARN] Too few frames ({processed}), skipping.")
            return None

        return {
            "keypoints":  np.array(keypoints_list,  dtype=np.float32),  # (N,17,2)
            "scores":     np.array(scores_list,      dtype=np.float32),  # (N,17)
            "frame_ids":  np.array(frame_ids_list,   dtype=np.int32),    # (N,)
        }

    def _process_frame(
        self, frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Run detector + pose on one frame. Returns (keypoints, scores) or (None, None)."""
        bboxes = self.detector.detect(frame)
        if not bboxes:
            return None, None

        bbox = select_largest_bbox(bboxes)
        result = self.pose.estimate_with_scores(frame, bbox)

        if result is None:
            return None, None

        kp, sc = result   # (17,2), (17,)
        return kp, sc


# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset scanner
# ═══════════════════════════════════════════════════════════════════════════════

def scan_videos(data_dir: Path) -> List[Tuple[str, str, Path]]:
    """
    Scan data/raw/ for videos organised as data/raw/{exercise}/{video}.mp4

    Returns list of (exercise_name, video_id, video_path)
    """
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
                video_id = video_file.stem
                entries.append((ex_name, video_id, video_file))
    return entries


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract keypoints from fitness videos")
    parser.add_argument("--data-dir",  default="data/raw",       help="Root of raw videos")
    parser.add_argument("--out-dir",   default="data/processed",  help="Output directory")
    parser.add_argument("--skip",      type=int, default=2,        help="Process every N-th frame")
    parser.add_argument("--det-conf",  type=float, default=0.45,   help="Detection confidence")
    parser.add_argument("--overwrite", action="store_true",        help="Overwrite existing files")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)

    if not data_dir.exists():
        print(f"[ERROR] data-dir not found: {data_dir}")
        sys.exit(1)

    videos = scan_videos(data_dir)
    if not videos:
        print(f"[ERROR] No videos found under {data_dir}/{{exercise}}/{{video}}.mp4")
        sys.exit(1)

    print(f"\n[INFO] Found {len(videos)} videos across {len(set(v[0] for v in videos))} exercises")
    print(f"[INFO] Output  → {out_dir}\n")

    extractor = KeypointExtractor(det_conf=args.det_conf, skip=args.skip)

    stats = {"ok": 0, "skip": 0, "fail": 0}

    for ex_name, video_id, video_path in videos:
        out_ex_dir = out_dir / ex_name
        out_ex_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_ex_dir / f"{video_id}.npz"

        if out_file.exists() and not args.overwrite:
            print(f"[SKIP] {ex_name}/{video_id}.npz (already exists)")
            stats["skip"] += 1
            continue

        print(f"\n[EXTRACT] {ex_name} / {video_id}")
        result = extractor.extract(str(video_path))

        if result is None:
            stats["fail"] += 1
            continue

        # Save with metadata
        np.savez_compressed(
            out_file,
            keypoints = result["keypoints"],   # (N,17,2)
            scores    = result["scores"],       # (N,17)
            frame_ids = result["frame_ids"],    # (N,)
            exercise  = np.array(ex_name),
            video_id  = np.array(video_id),
        )
        n = len(result["frame_ids"])
        print(f"  Saved → {out_file}  ({n} frames)")
        stats["ok"] += 1

    print(f"\n{'═'*50}")
    print(f"  Done.  OK:{stats['ok']}  Skipped:{stats['skip']}  Failed:{stats['fail']}")
    print(f"{'═'*50}\n")


if __name__ == "__main__":
    main()
