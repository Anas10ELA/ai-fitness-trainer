"""
main.py — AI Fitness Trainer  [v2 — SkeletonNet DL Engine]

Controls:
  WELCOME: [ENTER] → Menu  |  [ESC/Q] → Exit
  MENU:    Arrows/WASD → Navigate  |  [ENTER] → Start  |  [ESC] → Exit
  WORKOUT: [M] → Menu  |  [R] → Reset  |  [ESC] → Exit
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from config    import cfg
from exercises import Exercise, KEY_MAP, REGISTRY
from angles    import get_joint_angles
from feedback  import FeedbackEngine
from temporal  import TemporalValidator
from tracker   import PersonTracker, TrackResult, draw_tracker_overlay
from utils     import (
    DualFPSCounter, KalmanKeypoints, PipelineProfiler,
    print_keybind_legend, print_status,
    draw_skeleton, draw_bbox,
    draw_welcome_screen, draw_menu_screen, draw_top_hud,
)
from dl_engine import DLEngine, EngineResult
from rep_policy import resolve_rep_policy as _resolve_rep_policy
from voice_coach import speak_instruction_async

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI & I/O helpers
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--source",            default=cfg.video_source)
    p.add_argument("--exercise",          default="squat",
                                          choices=[e.value for e in Exercise])
    p.add_argument("--model",             default=cfg.yolo_model)
    p.add_argument("--det-conf",          type=float, default=cfg.det_conf)
    p.add_argument("--no-display",        action="store_true", default=cfg.no_display)
    p.add_argument("--save",              default=cfg.save_video or None)
    p.add_argument("--log-level",         default=cfg.log_level)
    p.add_argument("--show-tracking",     action="store_true", default=cfg.show_tracking)
    p.add_argument("--no-voice",          action="store_true",
                   help="Disable spoken pre-exercise instructions in the desktop app")
    p.add_argument("--iou-threshold",     type=float, default=cfg.tracker_iou_threshold)
    p.add_argument("--max-lost",          type=int,   default=cfg.tracker_max_lost)
    p.add_argument("--detect-interval",   type=int,   default=cfg.detection_interval)
    # ── DL engine (v4) ──────────────────────────────────────────────────────────
    p.add_argument("--dl-checkpoint-dir", default="checkpoints_v3",
                   help="Directory with dl_model_<exercise>.pt files  "
                        "(default: checkpoints_v3)")
    p.add_argument("--smooth-k",          type=int, default=5,
                   help="ModeVote smoother window in frames. MUST match the "
                        "train FSM-sim --sim-smooth-k so live counting equals "
                        "the reported evaluation  (default: 5)")
    p.add_argument("--require-closed",    default="auto",
                   choices=["auto", "on", "off"],
                   help="Rep-counting gate. 'auto' = per-exercise policy "
                        "(_REP_POLICY): require the peak posture for clear-peak "
                        "exercises (squat/lateral_raise/tricep_dip), duration-gate "
                        "the rest. 'on'/'off' force it globally for live A/B "
                        "testing  (default: auto)")
    return p.parse_args()


def _open_capture(source: str) -> cv2.VideoCapture:
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        log.critical("Cannot open video source: %s", source)
        sys.exit(1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _create_writer(cap: cv2.VideoCapture,
                   path: Optional[str]) -> Optional[cv2.VideoWriter]:
    if not path:
        return None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cv2.VideoWriter(path, fourcc, fps, (w, h))


# ══════════════════════════════════════════════════════════════════════════════
#  Asynchronous inference worker  (YOLO + RTMPose + Kalman — unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class _InferenceWorker:
    def __init__(
        self,
        detector, pose, smoother, profiler,
        iou_threshold:      float = 0.35,
        max_lost_frames:    int   = 10,
        detection_interval: int   = 3,
    ) -> None:
        self._det      = detector
        self._pose     = pose
        self._smoother = smoother
        self._profiler = profiler
        self._tracker  = PersonTracker(
            iou_threshold   = iou_threshold,
            max_lost_frames = max_lost_frames,
        )
        self._detection_interval  = detection_interval
        self._frames_since_detect = 0

        self._lock         = threading.Lock()
        self._evt          = threading.Event()
        self._pending      = None
        self._detected     = False
        self._track_result = None
        self._keypoints    = None
        self._scores       = None
        self._running      = False
        self._thread       = threading.Thread(target=self._loop, daemon=True)

    def start(self)  -> None:
        self._running = True
        self._thread.start()

    def stop(self)   -> None:
        self._running = False
        self._evt.set()
        self._thread.join(timeout=3)

    def submit(self, frame: np.ndarray) -> None:
        with self._lock:
            self._pending = frame
        self._evt.set()

    def result(self) -> Tuple:
        with self._lock:
            kp = self._keypoints.copy() if self._keypoints is not None else None
            sc = self._scores.copy()    if self._scores    is not None else None
            return self._detected, self._track_result, kp, sc

    def reset_tracker(self) -> None:
        self._tracker.reset()
        self._frames_since_detect = 0

    def tracker_metrics(self) -> dict:
        return self._tracker.metrics_dict()

    def _loop(self) -> None:
        while self._running:
            if not self._evt.wait(timeout=0.5):
                continue
            self._evt.clear()
            with self._lock:
                frame, self._pending = self._pending, None
            if frame is None:
                continue
            detected, track_result, kp, sc = self._infer(frame)
            with self._lock:
                self._detected     = detected
                self._track_result = track_result
                self._keypoints    = kp
                self._scores       = sc

    def _infer(self, frame: np.ndarray) -> Tuple:
        t0 = time.perf_counter()
        run_det = (self._frames_since_detect >= self._detection_interval
                   or self._tracker.state_str == "searching")
        if run_det:
            try:
                bboxes = self._det.detect(frame)
            except Exception:
                bboxes = []
            self._frames_since_detect = 0
            track_result = self._tracker.update(bboxes)
        else:
            self._frames_since_detect += 1
            track_result = self._tracker.coast()
        self._profiler.add("1. Detection", (time.perf_counter() - t0) * 1000)

        if not track_result.person_present:
            return False, track_result, None, None

        t1 = time.perf_counter()
        result = None
        try:
            result = self._pose.estimate_with_scores(frame, track_result.bbox)
        except Exception:
            pass
        self._profiler.add("2. Pose (RTMPose)", (time.perf_counter() - t1) * 1000)
        if result is None:
            return True, track_result, None, None

        kp, sc = result

        t2 = time.perf_counter()
        try:
            kp = self._smoother.smooth(kp)
        except Exception:
            pass
        self._profiler.add("3. Kalman Smooth", (time.perf_counter() - t2) * 1000)

        return True, track_result, kp, sc


# ══════════════════════════════════════════════════════════════════════════════
#  _CounterShim
#
#  The existing HUD (draw_top_hud) reads:
#    session.counter.reps
#    session.counter.state
#    session.counter.hold_seconds
#
#  We expose those three attributes by delegating to DLEngine._counter.
#  Nothing in the render code changes.
# ══════════════════════════════════════════════════════════════════════════════

class _CounterShim:
    __slots__ = ("_engine", "_ai_state")

    def __init__(self, engine: DLEngine, ai_state: str) -> None:
        self._engine   = engine
        self._ai_state = ai_state

    @property
    def reps(self) -> int:
        return self._engine._counter.reps

    @property
    def hold_seconds(self) -> float:
        return self._engine._counter.hold_seconds

    @property
    def state(self) -> str:
        return self._ai_state


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
#  Per-exercise rep-counting policy  (real-time)
#
#  The anchor counter (dl_engine._RepCounterFSM) can validate a rep two ways:
#    require_closed=True   → the rep MUST reach the contracted/peak posture
#                            (closed anchor).  Kills "anything-similar" false
#                            counts from shifting / partial motion, because
#                            incidental movement never passes through the peak.
#                            Safe ONLY when the closed-anchor recall is high.
#    require_closed=False  → duration-gated: a rep counts on a long-enough
#                            excursion away from rest (>= away_min NON-OPEN
#                            frames).  Used where the peak recall is too low for
#                            a hard gate; away_min is raised to demand a
#                            rep-like excursion rather than a twitch.
#
#  Decision threshold = closed-anchor recall on the v4 test set:
#      lr_top 0.96, sq_bottom 0.75, td_bottom 0.71  → peak-gate (>= 0.70)
#      pu_bottom 0.52, su_top 0.38                  → duration-gate (< 0.70)
#  Phase-2 temporal augmentation should lift pu_bottom/su_top recall; once they
#  clear ~0.70 they can be flipped to require_closed=True here.
#
#  away_min only matters when require_closed=False.  10 frames (~0.33 s @30fps)
#  sits safely under the fastest real away-phase (pushup ~18f rep → ~12-15f
#  away; situp ~21f rep → ~14-16f away) while rejecting short incidental motion.
# ══════════════════════════════════════════════════════════════════════════════

#  The concrete policy lives in rep_policy.py so desktop and API remain aligned.


# ══════════════════════════════════════════════════════════════════════════════
#  Session  (replaces old _Session; now owns the DLEngine directly)
# ══════════════════════════════════════════════════════════════════════════════

class _Session:
    def __init__(
        self,
        exercise:          Exercise,
        profiler:          PipelineProfiler,
        dl_checkpoint_dir: str = "checkpoints_v3",
        smooth_k:          int = 5,
        require_closed_mode: str = "auto",
    ) -> None:
        self.exercise  = exercise
        self.profiler  = profiler
        self._ckpt_dir = dl_checkpoint_dir
        self._smooth_k = smooth_k
        self._rc_mode  = require_closed_mode

        self.feedback  = FeedbackEngine()
        self.temporal  = TemporalValidator()
        self.angles:   Dict[str, float] = {}
        self.form_msg: str  = ""
        self.rep_blocked: bool = False
        self.prev_angles: Dict = {}

        # DL engine replaces the old ST-GCN + RepCounter combo
        rc, amin_sec = _resolve_rep_policy(exercise.value, require_closed_mode)
        log.info("Rep policy [%s]: require_closed=%s  away_min=%.2fs  smooth_k=%d",
                 exercise.value, rc, amin_sec, smooth_k)
        self.engine = DLEngine(
            exercise       = exercise.value,
            checkpoint_dir = dl_checkpoint_dir,
            smooth_window  = smooth_k,
            require_closed = rc,
            away_min_sec   = amin_sec,
        )

        # Last known AI output — used by HUD and _CounterShim
        self.ai_state:    str   = ""
        self.ai_conf:     float = 0.0
        self.model_ready: bool  = False

    # ── Convenience: let draw_top_hud read reps / state / hold ────────────────

    @property
    def counter(self) -> _CounterShim:
        return _CounterShim(self.engine, self.ai_state)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def switch(self, new_exercise: Exercise, worker: _InferenceWorker) -> None:
        self.exercise = new_exercise
        self.feedback = FeedbackEngine()
        self.temporal = TemporalValidator()
        self.prev_angles.clear()
        self.ai_state    = ""
        self.ai_conf     = 0.0
        self.model_ready = False
        rc, amin_sec = _resolve_rep_policy(new_exercise.value, self._rc_mode)
        log.info("Rep policy [%s]: require_closed=%s  away_min=%.2fs  smooth_k=%d",
                 new_exercise.value, rc, amin_sec, self._smooth_k)
        self.engine = DLEngine(
            exercise       = new_exercise.value,
            checkpoint_dir = self._ckpt_dir,
            smooth_window  = self._smooth_k,
            require_closed = rc,
            away_min_sec   = amin_sec,
        )
        worker.reset_tracker()

    def reset_reps(self, worker: _InferenceWorker) -> None:
        self.engine.reset_reps()
        self.feedback.reset()
        self.temporal.reset()
        self.prev_angles.clear()
        worker.reset_tracker()

    # ── Per-frame processing ──────────────────────────────────────────────────

    def process(self, kp: np.ndarray, sc: Optional[np.ndarray],
                fps: DualFPSCounter) -> None:
        fps.tick_infer()

        # ── Step 4: Geometry / angles ──────────────────────────────────────────
        t4 = time.perf_counter()
        self.angles = get_joint_angles(
            kp, self.exercise, scores=sc, prev_angles=self.prev_angles
        )
        self.temporal.add(self.angles)
        self.profiler.add("4. Geometry Math", (time.perf_counter() - t4) * 1000)

        # ── Step 5: DL engine ──────────────────────────────────────────────────
        t5 = time.perf_counter()
        result: EngineResult = self.engine.update(kp, infer_fps=fps.infer_fps)
        self.model_ready = result.ready
        if result.ready:
            self.ai_state = result.state_name
            self.ai_conf  = result.confidence
        self.profiler.add("5. DL Engine", (time.perf_counter() - t5) * 1000)

        # ── Step 6: Form feedback (geometry-based, unchanged) ─────────────────
        t6 = time.perf_counter()
        try:
            self.form_msg, rule_blocked = self.feedback.evaluate(
                self.exercise, self.angles, self.ai_state, kp,
                {"active_knee": getattr(self, "_active_knee", "left")},
            )
            self.rep_blocked = rule_blocked
        except Exception:
            pass
        self.profiler.add("6. FSM & Feedback", (time.perf_counter() - t6) * 1000)


# ══════════════════════════════════════════════════════════════════════════════
#  HUD helpers
# ══════════════════════════════════════════════════════════════════════════════

def _draw_tracked_bbox(frame: np.ndarray, bbox: Tuple,
                       track: TrackResult) -> None:
    col = {
        "tracking":   (60, 220,  60),
        "recovering": ( 0, 200, 255),
    }.get(track.state, (60, 60, 220))
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                  col, 2, cv2.LINE_AA)


def _draw_ai_overlay(frame: np.ndarray, session: _Session) -> None:
    """Small badge (top-right) showing the current isolated DL state."""
    h, w    = frame.shape[:2]
    state   = session.ai_state
    conf    = session.ai_conf
    badge_w = 230

    # Background
    cv2.rectangle(frame, (w - badge_w - 5, 8), (w - 5, 98),
                  (15, 15, 15), -1)
    cv2.rectangle(frame, (w - badge_w - 5, 8), (w - 5, 98),
                  (60, 60, 60), 1)

    # State name
    cv2.putText(frame, state or "–",
                (w - badge_w + 6, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 255, 180), 1,
                cv2.LINE_AA)

    # Confidence bar
    bx, by, bh = w - badge_w + 6, 60, 6
    bw = badge_w - 16
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (40, 40, 40), -1)
    fill = int(conf * bw)
    cv2.rectangle(frame, (bx, by), (bx + fill, by + bh), (80, 200, 80), -1)
    cv2.putText(frame, f"{conf*100:.0f}%",
                (w - badge_w + 6, 86),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1,
                cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  Main  (UI state machine — WELCOME / MENU / WORKOUT)
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level  = getattr(logging, args.log_level.upper(), logging.WARNING),
        format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream = sys.stderr,
    )

    profiler = PipelineProfiler()

    from detector import PersonDetector
    from pose     import PoseEstimator

    detector = PersonDetector(model_name=args.model, conf_threshold=args.det_conf)
    pose     = PoseEstimator()
    smoother = KalmanKeypoints()

    cap    = _open_capture(args.source)
    writer = _create_writer(cap, args.save)

    fps     = DualFPSCounter()
    session = _Session(
        exercise          = Exercise(args.exercise),
        profiler          = profiler,
        dl_checkpoint_dir = args.dl_checkpoint_dir,
        smooth_k          = args.smooth_k,
        require_closed_mode = args.require_closed,
    )
    worker = _InferenceWorker(
        detector, pose, smoother, profiler,
        iou_threshold      = args.iou_threshold,
        max_lost_frames    = args.max_lost,
        detection_interval = args.detect_interval,
    )
    worker.start()

    try:
        from shutdown import register_desktop_handlers
        register_desktop_handlers(worker, cap, writer)
    except Exception:
        pass

    last_kp = last_bbox = last_sc = last_track_result = None

    ui_state        = "WELCOME"
    exercises_list  = [e.value for e in Exercise]
    selected_ex_idx = (exercises_list.index(args.exercise)
                       if args.exercise in exercises_list else 0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            display = frame.copy()

            # ── WELCOME ───────────────────────────────────────────────────────
            if ui_state == "WELCOME":
                draw_welcome_screen(display)
                cv2.imshow("AI Fitness", display)
                key = cv2.waitKeyEx(30)
                if key in [27, ord('q'), ord('Q')]:
                    break
                elif key in [13, 10]:
                    ui_state = "MENU"

            # ── MENU ──────────────────────────────────────────────────────────
            elif ui_state == "MENU":
                draw_menu_screen(display, exercises_list, selected_ex_idx)
                cv2.imshow("AI Fitness", display)
                key = cv2.waitKeyEx(30)
                if key == 27:
                    break
                elif key == 8:
                    ui_state = "WELCOME"
                elif key in [2490368, 65362, ord('w'), ord('W')]:
                    if selected_ex_idx >= 3:
                        selected_ex_idx -= 3
                elif key in [2621440, 65364, ord('s'), ord('S')]:
                    if selected_ex_idx + 3 < len(exercises_list):
                        selected_ex_idx += 3
                elif key in [2424832, 65361, ord('a'), ord('A')]:
                    if selected_ex_idx % 3 > 0:
                        selected_ex_idx -= 1
                elif key in [2555904, 65363, ord('d'), ord('D')]:
                    if selected_ex_idx % 3 < 2 and selected_ex_idx + 1 < len(exercises_list):
                        selected_ex_idx += 1
                elif key in [13, 10]:
                    chosen = Exercise(exercises_list[selected_ex_idx])
                    session.switch(chosen, worker)
                    if cfg.voice_enabled and not args.no_voice:
                        speak_instruction_async(chosen, language=cfg.voice_language)
                    smoother.reset()
                    last_kp = last_bbox = last_sc = last_track_result = None
                    ui_state = "WORKOUT"

            # ── WORKOUT ───────────────────────────────────────────────────────
            elif ui_state == "WORKOUT":
                fps.tick_display()
                worker.submit(frame)

                detected, track_result_new, kp_new, sc_new = worker.result()

                # Tracker state from the PREVIOUS cycle (to detect rising edges).
                prev_track_state = (last_track_result.state
                                    if last_track_result is not None else None)

                if track_result_new:
                    last_track_result = track_result_new
                    last_bbox         = track_result_new.bbox

                new_track_state = (last_track_result.state
                                   if last_track_result is not None else None)

                # ── FIX 2 (tracker dropout) ───────────────────────────────────
                # On the rising edge into "searching", flush the engine's
                # temporal context so frames captured AFTER the gap aren't
                # concatenated with stale pre-gap frames (that discontinuity
                # spikes a false state transition → phantom rep). reset_window()
                # preserves reps/hold; no private-field juggling here.
                if new_track_state == "searching" and prev_track_state != "searching":
                    session.engine.reset_window()

                # ── FIX 1 (duplicate-frame bug) ───────────────────────────────
                # worker.result() returns a fresh .copy() each call, so the old
                # `id(kp_new) != id(last_kp)` guard was ALWAYS true and the engine
                # was fed the same pose at display FPS, inflating the FSM and
                # burning cooldowns. Compare CONTENT instead: advance the engine
                # only on a genuinely new pose (equal arrays = no new worker result).
                is_new_kp = kp_new is not None and (
                    last_kp is None or not np.array_equal(kp_new, last_kp)
                )
                if is_new_kp:
                    last_kp, last_sc = kp_new, sc_new
                    try:
                        session.process(kp_new, sc_new, fps)
                    except Exception:
                        pass
                elif (not detected and
                      (last_track_result is None or
                       last_track_result.state == "searching")):
                    last_kp = last_bbox = last_sc = None

                # ── Render CV elements ────────────────────────────────────────
                if last_bbox and last_track_result:
                    _draw_tracked_bbox(display, last_bbox, last_track_result)
                if last_kp is not None:
                    draw_skeleton(display, last_kp,
                                  form_bad=session.rep_blocked)

                # ── Top HUD ───────────────────────────────────────────────────
                is_timed = REGISTRY[session.exercise].is_timed
                draw_top_hud(
                    display,
                    fps.display_fps,
                    session.counter.reps,
                    session.counter.state,
                    session.form_msg,
                    session.rep_blocked,
                    session.exercise.value,
                    session.counter.hold_seconds,
                    is_timed,
                )

                # ── DL state badge ────────────────────────────────────────────
                if session.model_ready and session.engine.is_loaded:
                    _draw_ai_overlay(display, session)

                if args.show_tracking and last_track_result:
                    draw_tracker_overlay(display, last_track_result,
                                         worker.tracker_metrics())

                if writer:
                    writer.write(display)

                cv2.imshow("AI Fitness", display)
                key = cv2.waitKeyEx(1)
                if key == 27:
                    break
                elif key in [ord('m'), ord('M')]:
                    ui_state = "MENU"
                elif key in [ord('r'), ord('R')]:
                    session.reset_reps(worker)

    finally:
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()
        if writer:
            writer.release()

        log.info("═" * 55)
        log.info("  SESSION SUMMARY")
        log.info("  Total reps  : %d", session.counter.reps)
        log.info("  Infer FPS   : %.1f", fps.infer_fps)
        log.info("═" * 55)
        log.info("%s", profiler.summary())


if __name__ == "__main__":
    main()
