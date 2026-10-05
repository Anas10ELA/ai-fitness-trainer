"""
detector.py — YOLOv8 Person Detection Module  [UPDATED v3 — Production Cleanup]

Changes in v3 (Step 9):
  PROD-7  Replaced print() calls with log.info() for consistent logging.
          All startup messages now go through the Python logging system
          instead of bypassing it via stdout.

All v2 (Performance) changes are preserved:
  PERF-1  FP16 inference on GPU.
  PERF-2  Vectorized bounding box extraction.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError(
        "ultralytics is not installed. Run: pip install ultralytics"
    ) from e

log = logging.getLogger(__name__)

PERSON_CLASS_ID = 0
BBox = Tuple[int, int, int, int]


class PersonDetector:
    def __init__(
        self,
        model_name:      str   = "yolov8n.pt",
        conf_threshold:  float = 0.45,
        iou_threshold:   float = 0.45,
        input_size:      int   = 640,
        device: str | None     = None,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.input_size     = input_size

        log.info("PersonDetector: loading model: %s", model_name)
        self.model = YOLO(model_name)

        if device is None:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        log.info("PersonDetector: using device: %s", self.device)

    def detect(self, frame: np.ndarray) -> List[BBox]:
        use_half = self.device in ("cuda", "mps")

        results = self.model.predict(
            source    = frame,
            imgsz     = self.input_size,
            conf      = self.conf_threshold,
            iou       = self.iou_threshold,
            classes   = [PERSON_CLASS_ID],
            device    = self.device,
            half      = use_half,
            verbose   = False,
        )

        bboxes: List[BBox] = []
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            xyxy   = results[0].boxes.xyxy.cpu().numpy().astype(int)
            bboxes = [(b[0], b[1], b[2], b[3]) for b in xyxy]

        if bboxes:
            bboxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)

        return bboxes

    @staticmethod
    def bbox_area(bbox: BBox) -> int:
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @staticmethod
    def pad_bbox(
        bbox:        BBox,
        frame_shape: Tuple[int, int],
        pad_ratio:   float = 0.1,
    ) -> BBox:
        h, w    = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh  = x2 - x1, y2 - y1
        px, py  = int(bw * pad_ratio), int(bh * pad_ratio)
        return (
            max(0, x1 - px),
            max(0, y1 - py),
            min(w, x2 + px),
            min(h, y2 + py),
        )
