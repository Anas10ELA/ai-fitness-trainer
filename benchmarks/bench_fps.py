"""
benchmarks/bench_fps.py
═══════════════════════
Step 10 — FPS & Per-Stage Latency Benchmark (Single Session)

Measures the full server-side processing pipeline for a single session using
synthetic frames, producing a detailed per-stage latency breakdown.

What is measured
----------------
  • Frame decode          (base64 → numpy)
  • Detection + tracking  (YOLO every N frames, tracker coast in between)
  • Pose estimation       (RTMPose on tracked crop)
  • Kalman smoothing
  • Angle calculation
  • BiLSTM inference
  • Form feedback
  • Rep counting FSM
  • Total end-to-end latency
  • Effective inference FPS

Usage
-----
  # Quick run with defaults:
  python benchmarks/bench_fps.py

  # Custom run, save JSON:
  python benchmarks/bench_fps.py --n-warmup 20 --n-bench 500 --output results/fps.json

  # Headless mode (no model checkpoint needed):
  python benchmarks/bench_fps.py --no-model

Output
------
  Console: per-stage latency table (p50, p90, p99, min, max)
  JSON:    full results including raw per-frame timings
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bench_fps")


# ── Argument parsing ───────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-session FPS & latency benchmark")
    p.add_argument("--n-warmup",   default=10,  type=int,   help="Warmup frames (discarded)")
    p.add_argument("--n-bench",    default=200, type=int,   help="Benchmark frames")
    p.add_argument("--frame-size", default=480, type=int,   help="Synthetic frame height/width")
    p.add_argument("--yolo-model", default="yolov8n.pt",    help="YOLO model")
    p.add_argument("--det-conf",   default=0.45, type=float)
    p.add_argument("--checkpoint", default="model/checkpoints/best.pth")
    p.add_argument("--stats",      default="data/processed/normalizer_stats.json")
    p.add_argument("--no-model",   action="store_true",     help="Skip BiLSTM engine")
    p.add_argument("--output",     default="",              help="Save JSON report to path")
    p.add_argument("--exercise",   default="squat",         help="Exercise for the session")
    return p.parse_args()


# ── Synthetic frame generator ─────────────────────────────────────────────────

def _make_frame_b64(size: int) -> str:
    """Generate a synthetic BGR frame with a person-like silhouette, return as base64 JPEG."""
    frame = np.random.randint(50, 150, (size, size, 3), dtype=np.uint8)
    # Person silhouette
    cx, cy = size // 2, size // 2
    cv2.rectangle(frame, (cx - size//6, cy - size//3), (cx + size//6, cy + size//3), (180, 130, 100), -1)
    cv2.circle(frame,    (cx, cy - size//3 - size//12), size // 12, (200, 160, 120), -1)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode()


# ── Percentile helper ─────────────────────────────────────────────────────────

def _pct(data: List[float], p: int) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    return s[min(int(len(s) * p / 100), len(s) - 1)]


# ── Stage timing wrapper ──────────────────────────────────────────────────────

class _StageClock:
    def __init__(self) -> None:
        self._data: Dict[str, List[float]] = {}

    def record(self, stage: str, ms: float) -> None:
        self._data.setdefault(stage, []).append(ms)

    def stats(self, stage: str) -> Dict:
        d = self._data.get(stage, [])
        if not d:
            return {}
        return {
            "p50": round(_pct(d, 50), 2),
            "p90": round(_pct(d, 90), 2),
            "p99": round(_pct(d, 99), 2),
            "min": round(min(d), 2),
            "max": round(max(d), 2),
            "mean": round(sum(d) / len(d), 2),
            "n":   len(d),
        }

    def all_stages(self) -> List[str]:
        return list(self._data.keys())


# ── Main benchmark ────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> Dict:
    log.info("═" * 60)
    log.info("  bench_fps — Single-Session Latency Benchmark")
    log.info("═" * 60)
    log.info("  n_warmup=%d  n_bench=%d  frame_size=%d",
             args.n_warmup, args.n_bench, args.frame_size)

    # ── Build pipeline ────────────────────────────────────────────────────────
    log.info("Loading pipeline …")

    from detector        import PersonDetector
    from pose            import PoseEstimator
    from exercises       import Exercise
    from counter         import RepCounter
    from feedback        import FeedbackEngine
    from temporal        import TemporalValidator
    from tracker         import PersonTracker
    from utils           import KalmanKeypoints, FPSCounter
    from angles          import get_joint_angles
    import collections

    detector  = PersonDetector(model_name=args.yolo_model, conf_threshold=args.det_conf)
    pose      = PoseEstimator()
    smoother  = KalmanKeypoints()
    tracker   = PersonTracker()
    engine    = None

    if not args.no_model and Path(args.checkpoint).exists():
        from model.inference import FitnessInferenceEngine
        engine = FitnessInferenceEngine(
            checkpoint_path=args.checkpoint,
            stats_path=args.stats,
        )
        log.info("BiLSTM engine loaded")
    else:
        log.info("BiLSTM engine: disabled")

    exercise    = Exercise(args.exercise)
    counter     = RepCounter(exercise)
    feedback    = FeedbackEngine()
    temporal    = TemporalValidator()
    ai_buffer   = collections.deque(maxlen=32)
    fps_counter = FPSCounter(window=50)
    clock       = _StageClock()

    DETECT_INTERVAL = 3
    frames_since_detect = 0

    frame_b64 = _make_frame_b64(args.frame_size)
    total_latencies: List[float] = []

    def _run_one_frame(measure: bool = True) -> None:
        nonlocal frames_since_detect
        t_total = time.perf_counter()

        # ── Decode ────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        img_bytes = base64.b64decode(frame_b64)
        arr       = np.frombuffer(img_bytes, dtype=np.uint8)
        frame     = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        decode_ms = (time.perf_counter() - t0) * 1000

        # ── Detect + track ────────────────────────────────────────────────────
        t0 = time.perf_counter()
        run_det = (frames_since_detect >= DETECT_INTERVAL) or (tracker.state_str == "searching")
        if run_det:
            bboxes            = detector.detect(frame)
            frames_since_detect = 0
            track_result      = tracker.update(bboxes)
        else:
            frames_since_detect += 1
            track_result      = tracker.coast()
        det_ms = (time.perf_counter() - t0) * 1000

        if not track_result.person_present:
            if measure:
                clock.record("0. Decode",        decode_ms)
                clock.record("1. Det+Track",     det_ms)
                total_latencies.append((time.perf_counter() - t_total) * 1000)
            return

        bbox = track_result.bbox

        # ── Pose ──────────────────────────────────────────────────────────────
        t0     = time.perf_counter()
        result = pose.estimate_with_scores(frame, bbox)
        pose_ms = (time.perf_counter() - t0) * 1000

        if result is None:
            if measure:
                clock.record("0. Decode",    decode_ms)
                clock.record("1. Det+Track", det_ms)
                clock.record("2. Pose",      pose_ms)
                total_latencies.append((time.perf_counter() - t_total) * 1000)
            return

        raw_kp, scores = result

        # ── Kalman smooth ──────────────────────────────────────────────────────
        t0     = time.perf_counter()
        kp     = smoother.smooth(raw_kp)
        sm_ms  = (time.perf_counter() - t0) * 1000

        if kp is None:
            kp = raw_kp

        # ── Angles ────────────────────────────────────────────────────────────
        t0      = time.perf_counter()
        angles  = get_joint_angles(kp, exercise, scores=scores)
        ang_ms  = (time.perf_counter() - t0) * 1000
        temporal.add(angles)

        # ── BiLSTM ────────────────────────────────────────────────────────────
        t0     = time.perf_counter()
        if engine is not None:
            engine.push_frame(kp, scores, buffer=ai_buffer)
        ai_ms  = (time.perf_counter() - t0) * 1000

        # ── Feedback + FSM ────────────────────────────────────────────────────
        t0   = time.perf_counter()
        _, rep_blocked = feedback.evaluate(exercise, angles, counter.state, kp, {})
        counter.update(
            angles=angles, keypoints=kp, is_rep_valid=not rep_blocked,
            temporal=temporal, current_fps=max(fps_counter.fps, 15.0),
        )
        fsm_ms = (time.perf_counter() - t0) * 1000

        fps_counter.tick()
        total_ms = (time.perf_counter() - t_total) * 1000

        if measure:
            clock.record("0. Decode",        decode_ms)
            clock.record("1. Det+Track",     det_ms)
            clock.record("2. Pose",          pose_ms)
            clock.record("3. Kalman Smooth", sm_ms)
            clock.record("4. Angles",        ang_ms)
            clock.record("5. BiLSTM",        ai_ms)
            clock.record("6. FSM+Feedback",  fsm_ms)
            total_latencies.append(total_ms)

    # ── Warmup ────────────────────────────────────────────────────────────────
    log.info("Warming up (%d frames) …", args.n_warmup)
    for _ in range(args.n_warmup):
        _run_one_frame(measure=False)
    tracker.reset()
    smoother.reset()
    frames_since_detect = 0

    # ── Benchmark ─────────────────────────────────────────────────────────────
    log.info("Benchmarking (%d frames) …", args.n_bench)
    t_bench_start = time.perf_counter()
    for i in range(args.n_bench):
        _run_one_frame(measure=True)
        if (i + 1) % 50 == 0:
            log.info("  %d/%d  running p50=%.1f ms",
                     i + 1, args.n_bench, _pct(total_latencies, 50))

    bench_elapsed = time.perf_counter() - t_bench_start
    effective_fps = args.n_bench / bench_elapsed

    # ── Report ────────────────────────────────────────────────────────────────
    stages = clock.all_stages()
    report = {
        "benchmark": "bench_fps",
        "config": {
            "n_warmup":   args.n_warmup,
            "n_bench":    args.n_bench,
            "frame_size": args.frame_size,
            "exercise":   args.exercise,
            "model":      args.yolo_model,
            "bilstm":     "enabled" if engine else "disabled",
        },
        "effective_fps": round(effective_fps, 1),
        "total_latency": {
            "p50": round(_pct(total_latencies, 50), 2),
            "p90": round(_pct(total_latencies, 90), 2),
            "p99": round(_pct(total_latencies, 99), 2),
            "min": round(min(total_latencies) if total_latencies else 0, 2),
            "max": round(max(total_latencies) if total_latencies else 0, 2),
        },
        "stages": {s: clock.stats(s) for s in stages},
    }

    _print_report(report)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        log.info("Report saved → %s", args.output)

    return report


def _print_report(report: Dict) -> None:
    log.info("")
    log.info("─" * 60)
    log.info("  RESULTS")
    log.info("─" * 60)
    log.info("  Effective FPS : %.1f", report["effective_fps"])
    tl = report["total_latency"]
    log.info("  Total latency : p50=%.1fms  p90=%.1fms  p99=%.1fms",
             tl["p50"], tl["p90"], tl["p99"])
    log.info("")
    log.info("  %-18s  %7s  %7s  %7s  %7s  %7s",
             "Stage", "p50 ms", "p90 ms", "p99 ms", "min ms", "max ms")
    log.info("  " + "─" * 60)
    for stage, s in report["stages"].items():
        if s:
            log.info("  %-18s  %7.2f  %7.2f  %7.2f  %7.2f  %7.2f",
                     stage, s["p50"], s["p90"], s["p99"], s["min"], s["max"])
    log.info("─" * 60)


if __name__ == "__main__":
    args = _parse()
    run(args)
