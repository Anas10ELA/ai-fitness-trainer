"""
tracker.py — Persistent Single-Target IoU Tracker  [UPDATED v2 — Hybrid Pipeline]

Changes in v2:
  HYBRID-1  Added coast() method. When YOLO is skipped (to save GPU), the tracker
            coasts on the previous target instead of registering it as a lost frame.
            This is critical for the Hybrid Pipeline architecture to maintain
            identity between detection intervals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int]


# ── Configuration ─────────────────────────────────────────────────────────────
IOU_MATCH_THRESHOLD: float = 0.35
MAX_LOST_FRAMES: int = 10
IOU_LOCK_THRESHOLD: float = 0.0
BBOX_SMOOTH_ALPHA: float = 0.75

# ── State ─────────────────────────────────────────────────────────────────────
class _TrackState(Enum):
    SEARCHING  = auto()
    TRACKING   = auto()
    RECOVERING = auto()

# ── IoU geometry ──────────────────────────────────────────────────────────────
def _iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter   = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def _box_area(b: BBox) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

def _smooth_bbox(prev: BBox, curr: BBox, alpha: float) -> BBox:
    return (
        int(alpha * curr[0] + (1 - alpha) * prev[0]),
        int(alpha * curr[1] + (1 - alpha) * prev[1]),
        int(alpha * curr[2] + (1 - alpha) * prev[2]),
        int(alpha * curr[3] + (1 - alpha) * prev[3]),
    )

# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrackResult:
    bbox:           Optional[BBox]
    person_present: bool
    match_iou:      float
    state:          str
    track_id:       int

# ═══════════════════════════════════════════════════════════════════════════════

class PersonTracker:
    def __init__(
        self,
        iou_threshold:     float = IOU_MATCH_THRESHOLD,
        max_lost_frames:   int   = MAX_LOST_FRAMES,
        bbox_smooth_alpha: float = BBOX_SMOOTH_ALPHA,
    ) -> None:
        self._iou_thr   = iou_threshold
        self._max_lost  = max_lost_frames
        self._alpha     = bbox_smooth_alpha
        self._state:     _TrackState     = _TrackState.SEARCHING
        self._target:    Optional[BBox]  = None
        self._raw_target: Optional[BBox] = None
        self._track_id:          int = 0
        self._frames_tracked:    int = 0
        self._frames_lost:       int = 0
        self._total_id_switches: int = 0
        self._total_frames:      int = 0
        self._total_tracked:     int = 0

    def update(self, detections: List[BBox]) -> TrackResult:
        self._total_frames += 1
        if self._state == _TrackState.SEARCHING:
            return self._handle_searching(detections)
        elif self._state == _TrackState.TRACKING:
            return self._handle_tracking(detections)
        else:
            return self._handle_recovering(detections)

    def coast(self) -> TrackResult:
        """
        HYBRID-1: Coast the tracker when YOLO detection is skipped to save GPU.
        Holds the previous bounding box without penalizing tracking stability.
        """
        self._total_frames += 1
        if self._state in (_TrackState.TRACKING, _TrackState.RECOVERING):
            self._frames_tracked += 1
            self._total_tracked += 1
            return TrackResult(
                bbox=self._target, person_present=True,
                match_iou=1.0, state=self.state_str, track_id=self._track_id,
            )
        else:
            return TrackResult(
                bbox=None, person_present=False,
                match_iou=0.0, state="searching", track_id=self._track_id,
            )

    def reset(self) -> None:
        self._state      = _TrackState.SEARCHING
        self._target     = None
        self._raw_target = None
        self._frames_tracked = 0
        self._frames_lost    = 0
        log.debug("PersonTracker: reset (track_id was %d)", self._track_id)

    @property
    def track_id(self) -> int:
        return self._track_id

    @property
    def frames_tracked(self) -> int:
        return self._frames_tracked

    @property
    def frames_lost(self) -> int:
        return self._frames_lost

    @property
    def total_id_switches(self) -> int:
        return self._total_id_switches

    @property
    def stability_ratio(self) -> float:
        if self._total_frames == 0:
            return 0.0
        return self._total_tracked / self._total_frames

    @property
    def state_str(self) -> str:
        return self._state.name.lower()

    def metrics_dict(self) -> dict:
        return {
            "track_id":       self._track_id,
            "state":          self.state_str,
            "frames_tracked": self._frames_tracked,
            "frames_lost":    self._frames_lost,
            "id_switches":    self._total_id_switches,
            "stability":      round(self.stability_ratio, 3),
            "total_frames":   self._total_frames,
        }

    def _handle_searching(self, detections: List[BBox]) -> TrackResult:
        if not detections:
            return TrackResult(
                bbox=None, person_present=False,
                match_iou=0.0, state="searching", track_id=self._track_id,
            )
        best = max(detections, key=_box_area)
        self._acquire(best)
        log.debug("PersonTracker: acquired target track_id=%d bbox=%s", self._track_id, best)
        return TrackResult(
            bbox=self._target, person_present=True,
            match_iou=1.0, state="tracking", track_id=self._track_id,
        )

    def _handle_tracking(self, detections: List[BBox]) -> TrackResult:
        if not detections:
            return self._enter_recovering()
        best_det, best_iou = self._best_match(detections)
        if best_iou >= self._iou_thr:
            self._raw_target = best_det
            self._target     = _smooth_bbox(self._target, best_det, self._alpha)
            self._frames_tracked += 1
            self._frames_lost     = 0
            self._total_tracked  += 1
            return TrackResult(
                bbox=self._target, person_present=True,
                match_iou=round(best_iou, 3), state="tracking", track_id=self._track_id,
            )
        else:
            return self._enter_recovering()

    def _handle_recovering(self, detections: List[BBox]) -> TrackResult:
        self._frames_lost += 1
        if detections:
            best_det, best_iou = self._best_match(detections)
            if best_iou >= self._iou_thr:
                self._state      = _TrackState.TRACKING
                self._raw_target = best_det
                self._target     = _smooth_bbox(self._target, best_det, self._alpha)
                self._frames_tracked += 1
                self._frames_lost     = 0
                self._total_tracked  += 1
                return TrackResult(
                    bbox=self._target, person_present=True,
                    match_iou=round(best_iou, 3), state="tracking", track_id=self._track_id,
                )

        if self._frames_lost >= self._max_lost:
            if detections:
                best = max(detections, key=_box_area)
                self._acquire(best)
                return TrackResult(
                    bbox=self._target, person_present=True,
                    match_iou=1.0, state="tracking", track_id=self._track_id,
                )
            else:
                self._state      = _TrackState.SEARCHING
                self._target     = None
                self._raw_target = None
                self._frames_tracked = 0
                return TrackResult(
                    bbox=None, person_present=False,
                    match_iou=0.0, state="searching", track_id=self._track_id,
                )

        return TrackResult(
            bbox=self._target, person_present=True,
            match_iou=0.0, state="recovering", track_id=self._track_id,
        )

    def _enter_recovering(self) -> TrackResult:
        self._state = _TrackState.RECOVERING
        self._frames_tracked = 0
        self._frames_lost += 1
        return TrackResult(
            bbox=self._target, person_present=True,
            match_iou=0.0, state="recovering", track_id=self._track_id,
        )

    def _acquire(self, bbox: BBox) -> None:
        if self._target is not None:
            self._total_id_switches += 1
        self._track_id  += 1
        self._state      = _TrackState.TRACKING
        self._target     = bbox
        self._raw_target = bbox
        self._frames_tracked = 1
        self._frames_lost    = 0
        self._total_tracked += 1

    def _best_match(self, detections: List[BBox]) -> Tuple[BBox, float]:
        best_det  = detections[0]
        best_iou  = 0.0
        for det in detections:
            score = _iou(self._target, det)
            if score > best_iou:
                best_iou = score
                best_det = det
        return best_det, best_iou

def draw_tracker_overlay(frame: np.ndarray, result: TrackResult, metrics: dict) -> None:
    import cv2
    h, w = frame.shape[:2]
    x = w - 210
    STATE_COLORS = {
        "tracking":  (60, 220, 60),
        "recovering": (0, 200, 255),
        "searching":  (60, 60, 220),
    }
    col = STATE_COLORS.get(result.state, (180, 180, 180))
    cv2.rectangle(frame, (x - 5, 100), (w - 5, 230), (15, 15, 15), -1)

    def put(text, y, color=(200, 200, 200), scale=0.42):
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    put("Tracker (Hybrid Mode)",                  105, (0, 200, 255), 0.48)
    put(f"State : {result.state.upper()}",        124, col)
    put(f"ID    : {result.track_id}",             142, (200, 200, 200))
    put(f"IoU   : {result.match_iou:.2f}",        160, (200, 200, 200))
    put(f"Stable: {metrics['stability']*100:.0f}%", 178, (200, 200, 200))
    put(f"Lost  : {metrics['frames_lost']}fr",   196, (200, 200, 200))
    put(f"Swaps : {metrics['id_switches']}",      214, (200, 200, 200))

    if result.bbox is not None:
        x1, y1, x2, y2 = result.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
        label = f"ID:{result.track_id} {result.state[:3].upper()}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), col, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1, cv2.LINE_AA)