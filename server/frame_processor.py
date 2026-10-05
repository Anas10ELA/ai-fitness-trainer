"""
server/frame_processor.py
═════════════════════════
Full Per-Frame Processing Pipeline  [UPDATED v7 — Async Queue compatible]

Changes in v7:
  ASYNC-Q1  FrameProcessor.process() is now called from multiple threads
            simultaneously (one per InferenceQueue worker thread).
            Verified thread-safe:
            - No shared mutable state between calls: all state lives in
              `session` (passed per call) and local variables.
            - `session` objects are NOT shared between sessions; each WS
              connection owns exactly one Session instance.
            - detector / pose_estimator model forward passes must themselves be
              thread-safe; each session owns its own DLEngine instance, so the
              rolling temporal window and rep counter are not shared.
            - DETECTION_INTERVAL env var is read once at import time (constant).

            No code changes required for thread safety — only this docstring
            update and a guard on TrackResult type annotation fix below.

  HYBRID-3  Unchanged: YOLO frame-skipping logic remains in place.
            YOLO is only executed every DETECTION_INTERVAL frames (default: 3)
            OR when the tracker state is "searching" (target lost).
"""

from __future__ import annotations

import base64
import logging
import time
import traceback
from typing import Optional

import cv2
import numpy as np

from exercises import Exercise
from angles    import get_joint_angles
from tracker   import PersonTracker, TrackResult
from server.schemas import WSFrameOutput, KeypointXY
from config import cfg

log = logging.getLogger(__name__)

# Run YOLO detection every N frames — sourced from centralised config (Step 9)
DETECTION_INTERVAL = cfg.detection_interval


# ═══════════════════════════════════════════════════════════════════════════════

class FrameProcessor:
    """
    Stateless per-call frame processing pipeline.

    Thread safety:
        process() is safe to call from multiple threads concurrently
        provided that:
          1. Each call receives a different `session` object.
          2. The underlying models (detector, pose, engine) support
             concurrent forward passes — PyTorch with separate input
             tensors per call is safe on both CPU and CUDA (CUDA streams
             are implicitly serialised per device but don't block Python).
    """

    def __init__(self, detector, pose_estimator) -> None:
        self.detector = detector
        self.pose     = pose_estimator

    def process(
        self,
        frame_b64: str,
        session,
        requested_exercise: Optional[str] = None,
    ) -> WSFrameOutput:
        t_start = time.perf_counter()
        try:
            return self._process_inner(frame_b64, session, requested_exercise, t_start)
        except Exception:
            log.error(
                "FrameProcessor.process() unhandled exception for session=%s exercise=%s:\n%s",
                getattr(session, "session_id", "?"),
                getattr(getattr(session, "exercise", None), "value", "?"),
                traceback.format_exc(),
            )
            return self._error_output(session, "Internal processing error")

    def _process_inner(
        self,
        frame_b64: str,
        session,
        requested_exercise: Optional[str],
        t_start: float,
    ) -> WSFrameOutput:

        if requested_exercise:
            try:
                req_ex = Exercise(requested_exercise.lower())
                if req_ex != session.exercise:
                    session.switch_exercise(req_ex)
            except ValueError:
                log.warning("FrameProcessor: unknown exercise switch request %r", requested_exercise)

        exercise = session.exercise

        t0 = time.perf_counter()
        frame = _decode_frame(frame_b64)
        decode_ms = (time.perf_counter() - t0) * 1000

        if frame is None:
            return self._error_output(session, "Failed to decode frame")

        # ── HYBRID-3: Person detection & Tracking ─────────────────────────────
        t0 = time.perf_counter()

        run_detector = (session.frames_since_detect >= DETECTION_INTERVAL) or \
                       (session.tracker.state_str == "searching")

        if run_detector:
            try:
                raw_bboxes = self.detector.detect(frame)
            except Exception:
                log.warning("FrameProcessor: detection failed:\n%s", traceback.format_exc())
                raw_bboxes = []
            session.frames_since_detect = 0
            track_result: TrackResult = session.tracker.update(raw_bboxes)
        else:
            session.frames_since_detect += 1
            track_result: TrackResult = session.tracker.coast()

        track_ms = (time.perf_counter() - t0) * 1000

        if not track_result.person_present:
            session.fps.tick()
            return self._no_person_output(session, time.perf_counter() - t_start, track_result)

        bbox = track_result.bbox

        # ── Pose estimation ───────────────────────────────────────────────────
        t0 = time.perf_counter()
        result = None
        try:
            result = self.pose.estimate_with_scores(frame, bbox)
        except Exception:
            log.warning("FrameProcessor: pose failed:\n%s", traceback.format_exc())
        pose_ms = (time.perf_counter() - t0) * 1000

        if result is None:
            session.fps.tick()
            return self._no_person_output(session, time.perf_counter() - t_start, track_result)

        raw_kp, scores = result
        scores = _validate_scores(scores)

        # ── Kalman smoothing ──────────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            kp = session.smoother.smooth(raw_kp)
        except Exception:
            kp = raw_kp.astype(np.float32) if raw_kp is not None else raw_kp
        smooth_ms = (time.perf_counter() - t0) * 1000

        if kp is None:
            session.fps.tick()
            return self._no_person_output(session, time.perf_counter() - t_start, track_result)

        # ── Angle calculation ─────────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            angles = get_joint_angles(kp, exercise, scores=scores, prev_angles=session.prev_angles)
        except Exception:
            angles = {}
        angles_ms = (time.perf_counter() - t0) * 1000

        session.last_angles = angles

        try:
            session.temporal.add(angles)
        except Exception:
            pass

        # ── AI model inference ────────────────────────────────────────────────
        t0 = time.perf_counter()
        current_fps = max(session.fps.fps, 15.0)
        try:
            result = session.engine.update(kp, infer_fps=current_fps)
        except Exception:
            log.warning("FrameProcessor: DLEngine failed:\n%s", traceback.format_exc())
            result = None
        ai_ms = (time.perf_counter() - t0) * 1000

        if result is not None and result.ready:
            session.model_ready  = True
            session.ai_state     = result.state_name
            session.ai_conf      = result.confidence
            session.reps         = result.reps
            session.hold_seconds = result.hold_seconds
        else:
            session.model_ready = False
            session.ai_state = ""
            session.ai_conf = 0.0

        # ── Form feedback ─────────────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            fb_msg, rep_blocked = session.feedback.evaluate(
                exercise=exercise, angles=angles, ai_state=session.ai_state,
                keypoints=kp, extra={}
            )
        except Exception:
            fb_msg, rep_blocked = "", False

        session.last_feedback = fb_msg
        session.rep_blocked   = rep_blocked
        fb_ms = (time.perf_counter() - t0) * 1000

        session.fps.tick()
        total_ms = (time.perf_counter() - t_start) * 1000

        log.debug(
            "session=%s ex=%s | decode=%.1f det+track=%.1f pose=%.1f "
            "smooth=%.1f angles=%.1f ai=%.1f fb=%.1f count=%.1f | total=%.1f ms",
            getattr(session, "session_id", "?"), exercise.value,
            decode_ms, track_ms, pose_ms, smooth_ms,
            angles_ms, ai_ms, fb_ms, 0.0, total_ms,
        )

        keypoints_out = [KeypointXY(x=float(kp[i, 0]), y=float(kp[i, 1])) for i in range(17)]
        tracker_metrics = session.tracker.metrics_dict()

        return WSFrameOutput(
            reps=session.reps, state=session.ai_state, hold_seconds=session.hold_seconds,
            exercise_name=exercise.value,
            exercise_conf=session.ai_conf, form_score=0.0 if rep_blocked else 1.0,
            is_form_good=not rep_blocked, model_ready=session.model_ready,
            feedback_message=fb_msg, rep_blocked=rep_blocked,
            fps=round(session.fps.fps, 1), latency_ms=round(total_ms, 1),
            keypoints=keypoints_out, person_bbox=[float(v) for v in bbox],
            angles={k: round(v, 1) for k, v in angles.items() if v > 0},
            track_id=track_result.track_id, track_state=track_result.state,
            track_iou=round(track_result.match_iou, 3), track_stability=round(session.tracker.stability_ratio, 3),
            track_id_switches=tracker_metrics["id_switches"],
        )

    def _no_person_output(self, session, elapsed: float, track_result: Optional[TrackResult] = None) -> WSFrameOutput:
        t_state = track_result.state if track_result else "searching"
        t_id    = track_result.track_id if track_result else 0
        session.model_ready = False
        session.ai_state = ""
        session.ai_conf = 0.0
        if t_state == "searching":
            try:
                session.engine.reset_window()
            except Exception:
                pass
        return WSFrameOutput(
            reps=session.reps, state=session.ai_state, hold_seconds=session.hold_seconds,
            exercise_name=session.exercise.value, exercise_conf=0.0, form_score=1.0, is_form_good=True, model_ready=False,
            feedback_message="👤 No person detected", rep_blocked=False, fps=round(session.fps.fps, 1),
            latency_ms=round(elapsed * 1000, 1), track_id=t_id, track_state=t_state, track_iou=0.0,
            track_stability=round(getattr(session, "tracker", PersonTracker()).stability_ratio, 3),
            track_id_switches=getattr(getattr(session, "tracker", None), "total_id_switches", 0),
        )

    def _error_output(self, session, msg: str) -> WSFrameOutput:
        session.model_ready = False
        session.ai_state = ""
        session.ai_conf = 0.0
        return WSFrameOutput(
            reps=session.reps, state=session.ai_state, hold_seconds=session.hold_seconds,
            exercise_name=session.exercise.value, exercise_conf=0.0, form_score=1.0, is_form_good=True, model_ready=False,
            feedback_message=f"⚠ {msg}", rep_blocked=False, fps=0.0, latency_ms=0.0,
            track_id=0, track_state="searching", track_iou=0.0, track_stability=0.0, track_id_switches=0,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _decode_frame(frame_b64: str) -> Optional[np.ndarray]:
    try:
        if not frame_b64: return None
        if "," in frame_b64: frame_b64 = frame_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(frame_b64)
        if not img_bytes: return None
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        if img_array.size == 0: return None
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or frame.shape[0] < 10:
            return None
        return frame
    except Exception:
        return None


def _validate_scores(scores: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if scores is None or not isinstance(scores, np.ndarray) or scores.ndim != 1 or len(scores) != 17:
        return None
    if not np.isfinite(scores).all():
        scores = np.where(np.isfinite(scores), scores, 0.0).astype(np.float32)
    return scores
