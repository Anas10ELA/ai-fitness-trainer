"""
pose.py — RTMPose Keypoint Estimation Module  [UPDATED v5 — Production Bug Fix]

Changes in v5 (PROD-FIX — RTMPose feet/lower-leg clipping):
  POSEFIX-1  Asymmetric bbox padding. RTMPose's MMPoseInferencer internally
             warps the bbox to a fixed 256x192 (4:3 portrait) input. A tall,
             tight standing-person bbox loses its ankles/feet in that affine
             warp. We now pad generously and ASYMMETRICALLY, with extra room
             below the feet (_PAD_RATIO_BOTTOM=0.35) and above the head, plus
             ±25% width. See _pad_bbox_for_pose().
  POSEFIX-2  Lowered default keypoint score_threshold 0.3 → 0.15 so genuinely
             valid lower-body keypoints (ankles/knees often land at 0.15-0.3)
             are no longer artificially zeroed out.

Changes in v4 (Step 9):
  PROD-7  All print() calls replaced with log.info() / log.warning() for
          consistent logging through the Python logging system.

All v3 (Hybrid Pipeline) changes preserved:
  PIPELINE-1  No redundant crop allocation.
  PIPELINE-2  Fixed bounding box padding bug.
"""

from __future__ import annotations

import logging
import os
import time
import urllib.request
from typing import Optional, Tuple

import cv2
import numpy as np

from detector import PersonDetector, BBox

log = logging.getLogger(__name__)

# ── Keypoint indices ──────────────────────────────────────────────────────────
KP = {
    "nose":           0,
    "left_eye":       1,  "right_eye":      2,
    "left_ear":       3,  "right_ear":      4,
    "left_shoulder":  5,  "right_shoulder": 6,
    "left_elbow":     7,  "right_elbow":    8,
    "left_wrist":     9,  "right_wrist":    10,
    "left_hip":       11, "right_hip":      12,
    "left_knee":      13, "right_knee":     14,
    "left_ankle":     15, "right_ankle":    16,
}

Keypoints = np.ndarray  # shape (17, 2) — (x, y) in frame-space pixel coords


# ── Caffemodel download helper ────────────────────────────────────────────────

WEIGHT_MIRRORS = [
    "https://huggingface.co/camenduru/openpose/resolve/main/models/pose/coco/pose_iter_440000.caffemodel",
    "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel",
]

_last_progress_log = 0.0


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    global _last_progress_log
    downloaded = block_num * block_size
    now = time.time()
    # PROD-7: throttle to one log line per second to avoid flooding
    if total_size > 0 and (now - _last_progress_log) >= 1.0:
        pct = min(downloaded / total_size * 100, 100)
        mb  = downloaded / 1_048_576
        log.info("  Download: %.1f%%  %.1f MB", pct, mb)
        _last_progress_log = now


def download_weights(weights_path: str, retries: int = 3) -> None:
    if os.path.exists(weights_path):
        return
    log.info("Downloading caffemodel (~200 MB) …")
    os.makedirs(os.path.dirname(weights_path) or ".", exist_ok=True)
    tmp_path = weights_path + ".part"
    for url in WEIGHT_MIRRORS:
        log.info("  Trying: %s", url)
        for attempt in range(retries):
            try:
                urllib.request.urlretrieve(url, tmp_path, reporthook=_progress)
                os.rename(tmp_path, weights_path)
                log.info("Download complete: %s", weights_path)
                return
            except Exception as exc:
                log.warning(
                    "  Attempt %d/%d failed: %s", attempt + 1, retries, exc
                )
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
    raise RuntimeError("[Pose] All mirrors failed.")


# ── PoseEstimator ─────────────────────────────────────────────────────────────

class PoseEstimator:
    _DEFAULT_CONFIG   = "rtmpose-s_8xb256-420e_coco-256x192"

    # ── PADDING TUNING (POSEFIX-1) ────────────────────────────────────────────
    # RTMPose warps the bbox to a fixed 256x192 (4:3) input. A tall, tight
    # standing-person box loses its ankles/feet in that warp. We pad generously,
    # with EXTRA padding on the bottom where the feet are most often clipped.
    # Over-padding is safe: RTMPose returns keypoints in absolute frame
    # coordinates regardless of crop size.
    _PAD_RATIO_X:      float = 0.25   # ±25% width   (was 0.15 symmetric)
    _PAD_RATIO_TOP:    float = 0.20   # +20% above head
    _PAD_RATIO_BOTTOM: float = 0.35   # +35% below feet  (the key fix)

    def __init__(
        self,
        config:          str        = "auto",
        checkpoint:      str        = "auto",
        device:          str | None = None,
        score_threshold: float      = 0.15,   # POSEFIX-2: was 0.3 (hid lower-body kp)
    ) -> None:
        self.score_threshold = score_threshold
        self._inferencer     = None
        self._dnn_net        = None

        if device is None:
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda:0"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "cpu"
                else:
                    device = "cpu"
            except ImportError:
                device = "cpu"

        self.device = device
        self._init_backend(config, checkpoint)

    def _init_backend(self, config: str, checkpoint: str) -> None:
        self._init_mmpose(config, checkpoint)

    def _init_mmpose(self, config: str, checkpoint: str) -> None:
        from mmpose.apis import MMPoseInferencer  # type: ignore
        if config == "auto":
            config = self._DEFAULT_CONFIG
        if checkpoint == "auto":
            checkpoint = "auto"
        log.info("PoseEstimator: loading RTMPose config=%s device=%s", config, self.device)
        self._inferencer = MMPoseInferencer(
            pose2d         = config,
            pose2d_weights = None if checkpoint == "auto" else checkpoint,
            device         = self.device,
            show_progress  = False,
        )
        log.info("PoseEstimator: RTMPose ready")

    def _init_opencv_dnn(self) -> None:
        model_dir    = ".pose_models"
        os.makedirs(model_dir, exist_ok=True)
        proto_path   = os.path.join(model_dir, "pose_deploy_linevec.prototxt")
        weights_path = os.path.join(model_dir, "pose_iter_440000.caffemodel")
        BASE_URL = (
            "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/"
            "openpose/master/models/pose/coco/"
        )
        if not os.path.exists(proto_path):
            log.info("PoseEstimator: downloading prototxt …")
            urllib.request.urlretrieve(BASE_URL + "pose_deploy_linevec.prototxt", proto_path)
        download_weights(weights_path)
        self._dnn_net = cv2.dnn.readNetFromCaffe(proto_path, weights_path)
        if self.device.startswith("cuda"):
            self._dnn_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self._dnn_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        log.info("PoseEstimator: OpenCV DNN pose model ready")

    def estimate(
        self,
        frame:     np.ndarray,
        bbox:      BBox,
        pad_ratio: float = 0.25,
    ) -> Optional[Keypoints]:
        result = self.estimate_with_scores(frame, bbox, pad_ratio)
        return result[0] if result is not None else None

    def estimate_with_scores(
        self,
        frame:     np.ndarray,
        bbox:      BBox,
        pad_ratio: float = 0.25,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        # POSEFIX-1: asymmetric padding so RTMPose's affine warp keeps the feet.
        # pad_ratio is retained for API compatibility but the asymmetric helper
        # below takes precedence (it gives the lower body extra room).
        padded_bbox = self._pad_bbox_for_pose(bbox, frame.shape)

        if self._inferencer is not None:
            return self._estimate_mmpose_with_scores(frame, padded_bbox)
        elif self._dnn_net is not None:
            x1, y1, x2, y2 = padded_bbox
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            kp = self._estimate_dnn(crop, x1, y1)
            if kp is None:
                return None
            return kp, np.ones(17, dtype=np.float32) * 0.5
        return None

    def _pad_bbox_for_pose(
        self,
        bbox:        BBox,
        frame_shape: Tuple[int, int],
    ) -> BBox:
        """
        POSEFIX-1: Asymmetric bbox expansion tuned for top-down pose estimation.

        Math (horizontal center preserved; vertically biased downward to keep
        feet inside RTMPose's 4:3 affine warp):

            bw = x2 - x1,  bh = y2 - y1
            x1' = x1 - bw * PAD_X
            x2' = x2 + bw * PAD_X
            y1' = y1 - bh * PAD_TOP
            y2' = y2 + bh * PAD_BOTTOM       <-- extra room for feet/ankles

        Result is clipped to frame bounds. RTMPose returns keypoints in absolute
        frame coordinates regardless of the (larger) crop, so over-padding never
        shifts the output coordinates — it only gives the model spatial context.
        """
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)

        px     = int(bw * self._PAD_RATIO_X)
        py_top = int(bh * self._PAD_RATIO_TOP)
        py_bot = int(bh * self._PAD_RATIO_BOTTOM)

        nx1 = max(0, x1 - px)
        ny1 = max(0, y1 - py_top)
        nx2 = min(w, x2 + px)
        ny2 = min(h, y2 + py_bot)
        return (nx1, ny1, nx2, ny2)

    def _estimate_mmpose_with_scores(
        self, frame: np.ndarray, bbox: BBox
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        x1, y1, x2, y2 = bbox
        try:
            result_gen = self._inferencer(
                frame, bboxes=[[x1, y1, x2, y2]],
                return_datasamples=False, show=False,
            )
            results = next(result_gen)
        except (StopIteration, Exception):
            return None

        if not results or not results.get("predictions"):
            return None
        preds = results["predictions"][0]
        if not preds:
            return None

        pred         = preds[0]
        keypoints_xy = np.array(pred["keypoints"],       dtype=np.float32)
        scores       = np.array(pred["keypoint_scores"], dtype=np.float32)
        # POSEFIX-2: only zero genuinely missing points (below the lowered 0.15
        # threshold), so feet/ankles at 0.15-0.3 confidence now survive.
        keypoints_xy[scores < self.score_threshold] = 0.0
        return keypoints_xy, scores

    def _estimate_dnn(
        self, crop: np.ndarray, offset_x: int, offset_y: int
    ) -> Optional[Keypoints]:
        INPUT_H, INPUT_W = 368, 368
        blob = cv2.dnn.blobFromImage(
            crop, 1.0 / 255, (INPUT_W, INPUT_H), (0, 0, 0), swapRB=False, crop=False
        )
        self._dnn_net.setInput(blob)
        output    = self._dnn_net.forward()
        ch, out_h, out_w = output.shape[1], output.shape[2], output.shape[3]
        crop_h, crop_w   = crop.shape[:2]
        OPENPOSE_TO_COCO17 = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
        keypoints = np.zeros((17, 2), dtype=np.float32)
        for coco17_idx, op_idx in enumerate(OPENPOSE_TO_COCO17):
            if op_idx >= ch:
                continue
            heatmap  = output[0, op_idx]
            _, conf, _, point = cv2.minMaxLoc(heatmap)
            if conf < self.score_threshold:
                keypoints[coco17_idx] = (0.0, 0.0)
            else:
                kx = point[0] * crop_w / out_w + offset_x
                ky = point[1] * crop_h / out_h + offset_y
                keypoints[coco17_idx] = (kx, ky)
        return keypoints


def get_keypoint(keypoints: Keypoints, name: str) -> Tuple[float, float]:
    return tuple(keypoints[KP[name]])
