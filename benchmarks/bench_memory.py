"""
benchmarks/bench_memory.py
══════════════════════════
Step 10 — Memory Stability Benchmark

Monitors RAM and GPU memory usage over a long simulated session to detect:
  • Memory leaks in session state, angle history, or Kalman buffers
  • Python object accumulation (unbounded deques, growing dicts)
  • GPU memory fragmentation from repeated inference
  • Steady-state vs peak memory delta

Strategy
--------
  1. Initialise the full pipeline (detector, pose, Kalman, FSM, BiLSTM).
  2. Process N_FRAMES synthetic frames in a tight loop.
  3. Sample RSS (RAM) and GPU memory every SAMPLE_INTERVAL frames.
  4. After completion, fit a linear regression to the memory trace.
     A slope > LEAK_THRESHOLD_MB_PER_1K indicates a potential leak.
  5. Report: baseline, peak, final, growth_per_1k_frames, verdict.

Usage
-----
  python benchmarks/bench_memory.py
  python benchmarks/bench_memory.py --n-frames 5000 --output results/memory.json

  # Check GPU memory too:
  python benchmarks/bench_memory.py --gpu

Requirements
------------
  pip install psutil
  pip install gputil    (optional, for GPU memory)
"""

from __future__ import annotations

import argparse
import base64
import collections
import gc
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bench_memory")


# ── Argument parsing ───────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Memory stability benchmark")
    p.add_argument("--n-frames",      default=2000,  type=int,   help="Total frames to process")
    p.add_argument("--sample-every",  default=100,   type=int,   help="Sample memory every N frames")
    p.add_argument("--frame-size",    default=320,   type=int)
    p.add_argument("--yolo-model",    default="yolov8n.pt")
    p.add_argument("--det-conf",      default=0.45,  type=float)
    p.add_argument("--checkpoint",    default="model/checkpoints/best.pth")
    p.add_argument("--stats",         default="data/processed/normalizer_stats.json")
    p.add_argument("--no-model",      action="store_true")
    p.add_argument("--gpu",           action="store_true",        help="Also monitor GPU memory")
    p.add_argument("--exercise",      default="squat")
    p.add_argument("--output",        default="")
    # Leak detection: flag if memory grows more than this per 1000 frames (MB)
    p.add_argument("--leak-threshold-mb", default=5.0, type=float,
                   help="MB growth per 1000 frames above which we flag a potential leak")
    return p.parse_args()


# ── Memory sampling ───────────────────────────────────────────────────────────

def _rss_mb() -> float:
    try:
        import psutil
        import os
        return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    except ImportError:
        return 0.0


def _gpu_mb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(0) / 1024 ** 2
        return 0.0
    except ImportError:
        return 0.0


# ── Linear regression for leak detection ─────────────────────────────────────

def _linear_slope(xs: List[float], ys: List[float]) -> float:
    """Return slope (y per unit x) via least-squares."""
    n = len(xs)
    if n < 2:
        return 0.0
    sx  = sum(xs)
    sy  = sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sxx = sum(x * x for x in xs)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return 0.0
    return (n * sxy - sx * sy) / denom


# ── Synthetic frame ───────────────────────────────────────────────────────────

def _make_frame_b64(size: int) -> str:
    frame = np.random.randint(40, 150, (size, size, 3), dtype=np.uint8)
    cx    = size // 2
    cv2.rectangle(frame, (cx - size//6, size//5), (cx + size//6, 4*size//5), (170, 120, 90), -1)
    cv2.circle(frame, (cx, size//6), size//12, (200, 160, 120), -1)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf.tobytes()).decode()


# ── Main benchmark ────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> Dict:
    log.info("═" * 60)
    log.info("  bench_memory — Memory Stability Benchmark")
    log.info("═" * 60)
    log.info("  n_frames=%d  sample_every=%d  frame_size=%d",
             args.n_frames, args.sample_every, args.frame_size)

    try:
        import psutil
    except ImportError:
        log.error("psutil not installed. pip install psutil")
        sys.exit(1)

    # ── Build pipeline ────────────────────────────────────────────────────────
    from detector  import PersonDetector
    from pose      import PoseEstimator
    from exercises import Exercise
    from counter   import RepCounter
    from feedback  import FeedbackEngine
    from temporal  import TemporalValidator
    from tracker   import PersonTracker
    from utils     import KalmanKeypoints, FPSCounter
    from angles    import get_joint_angles

    detector = PersonDetector(model_name=args.yolo_model, conf_threshold=args.det_conf)
    pose     = PoseEstimator()
    smoother = KalmanKeypoints()
    tracker  = PersonTracker()
    engine   = None

    if not args.no_model and Path(args.checkpoint).exists():
        from model.inference import FitnessInferenceEngine
        engine = FitnessInferenceEngine(
            checkpoint_path=args.checkpoint,
            stats_path=args.stats,
        )

    exercise  = Exercise(args.exercise)
    counter   = RepCounter(exercise)
    feedback  = FeedbackEngine()
    temporal  = TemporalValidator()
    ai_buffer = collections.deque(maxlen=32)
    fps_c     = FPSCounter(window=50)

    DETECT_INTERVAL     = 3
    frames_since_detect = 0
    frame_b64           = _make_frame_b64(args.frame_size)

    # ── Measurement arrays ────────────────────────────────────────────────────
    frame_indices: List[float] = []
    rss_samples:   List[float] = []
    gpu_samples:   List[float] = []

    # Force GC baseline
    gc.collect()
    baseline_rss = _rss_mb()
    baseline_gpu = _gpu_mb() if args.gpu else 0.0
    log.info("  Baseline RAM: %.1f MB  GPU: %.1f MB", baseline_rss, baseline_gpu)

    t0 = time.perf_counter()

    for frame_idx in range(args.n_frames):

        # ── Decode ────────────────────────────────────────────────────────────
        img_bytes = base64.b64decode(frame_b64)
        arr       = np.frombuffer(img_bytes, dtype=np.uint8)
        frame     = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        # ── Detect + track ────────────────────────────────────────────────────
        run_det = (frames_since_detect >= DETECT_INTERVAL) or (tracker.state_str == "searching")
        if run_det:
            bboxes              = detector.detect(frame)
            frames_since_detect = 0
            track_result        = tracker.update(bboxes)
        else:
            frames_since_detect += 1
            track_result        = tracker.coast()

        if track_result.person_present:
            result = pose.estimate_with_scores(frame, track_result.bbox)
            if result is not None:
                raw_kp, scores = result
                kp = smoother.smooth(raw_kp) or raw_kp
                angles = get_joint_angles(kp, exercise, scores=scores)
                temporal.add(angles)
                if engine is not None:
                    engine.push_frame(kp, scores, buffer=ai_buffer)
                _, rep_blocked = feedback.evaluate(exercise, angles, counter.state, kp, {})
                counter.update(
                    angles=angles, keypoints=kp, is_rep_valid=not rep_blocked,
                    temporal=temporal, current_fps=max(fps_c.fps, 15.0),
                )
                fps_c.tick()

        # ── Sample memory ──────────────────────────────────────────────────────
        if frame_idx % args.sample_every == 0:
            gc.collect()  # collect before sampling for accuracy
            rss = _rss_mb()
            gpu = _gpu_mb() if args.gpu else 0.0
            frame_indices.append(float(frame_idx))
            rss_samples.append(rss)
            gpu_samples.append(gpu)

            elapsed = time.perf_counter() - t0
            fps_val = frame_idx / elapsed if elapsed > 0 else 0
            log.info("  [%5d/%d]  RAM=%.1f MB (+%.1f)  GPU=%.1f MB  fps=%.1f",
                     frame_idx, args.n_frames,
                     rss, rss - baseline_rss,
                     gpu, fps_val)

    elapsed_total = time.perf_counter() - t0
    effective_fps = args.n_frames / elapsed_total

    # ── Leak analysis ─────────────────────────────────────────────────────────
    rss_slope_per_1k = _linear_slope(frame_indices, rss_samples) * 1000
    gpu_slope_per_1k = _linear_slope(frame_indices, gpu_samples) * 1000 if args.gpu else 0.0

    peak_rss  = max(rss_samples) if rss_samples else 0.0
    final_rss = rss_samples[-1]  if rss_samples else 0.0
    peak_gpu  = max(gpu_samples) if gpu_samples else 0.0
    final_gpu = gpu_samples[-1]  if gpu_samples else 0.0

    rss_leak = rss_slope_per_1k > args.leak_threshold_mb
    gpu_leak = gpu_slope_per_1k > args.leak_threshold_mb and args.gpu

    # ── Report ────────────────────────────────────────────────────────────────
    log.info("")
    log.info("═" * 60)
    log.info("  RESULTS")
    log.info("═" * 60)
    log.info("  Effective FPS    : %.1f",   effective_fps)
    log.info("  RAM baseline     : %.1f MB", baseline_rss)
    log.info("  RAM peak         : %.1f MB", peak_rss)
    log.info("  RAM final        : %.1f MB", final_rss)
    log.info("  RAM growth/1k fr : %.2f MB  %s",
             rss_slope_per_1k, "⚠ POTENTIAL LEAK" if rss_leak else "✓ stable")
    if args.gpu:
        log.info("  GPU baseline     : %.1f MB", baseline_gpu)
        log.info("  GPU peak         : %.1f MB", peak_gpu)
        log.info("  GPU final        : %.1f MB", final_gpu)
        log.info("  GPU growth/1k fr : %.2f MB  %s",
                 gpu_slope_per_1k, "⚠ POTENTIAL LEAK" if gpu_leak else "✓ stable")

    verdict = "PASS"
    issues  = []
    if rss_leak:
        verdict = "WARN"
        issues.append(f"RAM growth {rss_slope_per_1k:.2f} MB/1k frames > threshold {args.leak_threshold_mb}")
    if gpu_leak:
        verdict = "WARN"
        issues.append(f"GPU growth {gpu_slope_per_1k:.2f} MB/1k frames > threshold {args.leak_threshold_mb}")

    log.info("  Verdict          : %s", verdict)
    for issue in issues:
        log.warning("  ⚠  %s", issue)
    log.info("═" * 60)

    report = {
        "benchmark": "bench_memory",
        "config": {
            "n_frames":          args.n_frames,
            "sample_every":      args.sample_every,
            "frame_size":        args.frame_size,
            "exercise":          args.exercise,
            "bilstm":            "enabled" if engine else "disabled",
            "leak_threshold_mb": args.leak_threshold_mb,
        },
        "ram": {
            "baseline_mb":        round(baseline_rss, 1),
            "peak_mb":            round(peak_rss, 1),
            "final_mb":           round(final_rss, 1),
            "growth_per_1k_mb":   round(rss_slope_per_1k, 3),
            "potential_leak":     rss_leak,
            "samples":            [round(x, 1) for x in rss_samples],
            "sample_frames":      [int(x) for x in frame_indices],
        },
        "gpu": {
            "baseline_mb":      round(baseline_gpu, 1),
            "peak_mb":          round(peak_gpu, 1),
            "final_mb":         round(final_gpu, 1),
            "growth_per_1k_mb": round(gpu_slope_per_1k, 3),
            "potential_leak":   gpu_leak,
        } if args.gpu else {"monitored": False},
        "performance": {
            "effective_fps": round(effective_fps, 1),
            "elapsed_s":     round(elapsed_total, 1),
        },
        "verdict":  verdict,
        "issues":   issues,
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        log.info("Report saved → %s", args.output)

    return report


if __name__ == "__main__":
    args = _parse()
    run(args)
