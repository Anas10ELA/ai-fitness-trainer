"""
tools/benchmark_inference.py
════════════════════════════
Step 8 — Inference Latency Benchmark: PyTorch vs ONNX vs TensorRT

Measures and compares per-stage inference latency across all three backends
for both the detector (YOLO) and pose estimator (RTMPose).

Usage
-----
  # Benchmark detector only (all available backends):
  python tools/benchmark_inference.py --component detector

  # Benchmark pose estimator only:
  python tools/benchmark_inference.py --component pose

  # Benchmark everything and save JSON report:
  python tools/benchmark_inference.py --output benchmark_report.json

  # Custom number of runs and frame size:
  python tools/benchmark_inference.py --n-warmup 20 --n-bench 200 --frame-size 720

  # Full pipeline benchmark (YOLO + RTMPose, sequential):
  python tools/benchmark_inference.py --component pipeline

Output
------
  Console: formatted latency table (p50, p90, p99, min, max, fps)
  JSON:    full raw results + summary for each (backend, component) pair

Report format example
---------------------
  {
    "system": { "gpu": "NVIDIA RTX 3080", "cuda": "12.1", "trt_version": "8.6.1" },
    "config": { "n_warmup": 10, "n_bench": 100, "frame_size": 640 },
    "results": [
      {
        "backend":    "pytorch",
        "component":  "detector",
        "p50_ms":     11.2,
        "p90_ms":     12.8,
        "p99_ms":     15.1,
        "min_ms":     10.1,
        "max_ms":     19.4,
        "mean_ms":    11.5,
        "fps":        86.9,
        "raw_ms":     [...]
      },
      ...
    ]
  }
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("benchmark")


# ── Argument parsing ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference latency benchmark")
    p.add_argument("--component",   default="all",
                   choices=["detector", "pose", "pipeline", "all"],
                   help="Which component to benchmark")
    p.add_argument("--backends",    nargs="+", default=["pytorch", "onnx", "trt"],
                   help="Backends to test: pytorch onnx trt")
    p.add_argument("--n-warmup",    default=10,   type=int, help="Warmup iterations")
    p.add_argument("--n-bench",     default=100,  type=int, help="Benchmark iterations")
    p.add_argument("--frame-size",  default=640,  type=int, help="Input frame width/height")
    p.add_argument("--yolo-pt",     default="yolov8n.pt",       help="YOLO PyTorch model")
    p.add_argument("--yolo-onnx",   default="models/yolo.onnx", help="YOLO ONNX model")
    p.add_argument("--yolo-trt",    default="models/yolo.engine",help="YOLO TRT engine")
    p.add_argument("--pose-onnx",   default="models/rtmpose.onnx",   help="Pose ONNX")
    p.add_argument("--pose-trt",    default="models/rtmpose.engine",  help="Pose TRT engine")
    p.add_argument("--output",      default="",   help="Save JSON report to this path")
    p.add_argument("--device",      default="cuda:0", help="Torch device")
    return p.parse_args()


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class BenchResult:
    backend:   str
    component: str
    raw_ms:    List[float] = field(default_factory=list)
    error:     str = ""

    @property
    def p50_ms(self) -> float:
        return _pct(self.raw_ms, 50)

    @property
    def p90_ms(self) -> float:
        return _pct(self.raw_ms, 90)

    @property
    def p99_ms(self) -> float:
        return _pct(self.raw_ms, 99)

    @property
    def min_ms(self) -> float:
        return min(self.raw_ms) if self.raw_ms else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.raw_ms) if self.raw_ms else 0.0

    @property
    def mean_ms(self) -> float:
        return sum(self.raw_ms) / len(self.raw_ms) if self.raw_ms else 0.0

    @property
    def fps(self) -> float:
        return (1000.0 / self.mean_ms) if self.mean_ms > 0 else 0.0

    def to_dict(self) -> Dict:
        d = {
            "backend":   self.backend,
            "component": self.component,
            "p50_ms":    round(self.p50_ms, 2),
            "p90_ms":    round(self.p90_ms, 2),
            "p99_ms":    round(self.p99_ms, 2),
            "min_ms":    round(self.min_ms, 2),
            "max_ms":    round(self.max_ms, 2),
            "mean_ms":   round(self.mean_ms, 2),
            "fps":       round(self.fps, 1),
        }
        if self.error:
            d["error"] = self.error
        return d


def _pct(data: List[float], pct: int) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = int(len(s) * pct / 100)
    return s[min(idx, len(s) - 1)]


# ── Benchmark runner ───────────────────────────────────────────────────────────

def _run_benchmark(
    fn: Callable,
    n_warmup: int,
    n_bench: int,
    label: str,
) -> List[float]:
    """
    Run fn() n_warmup times (discarded), then n_bench times (measured).
    Returns list of latencies in milliseconds.
    """
    log.info("  Warmup (%d) …", n_warmup)
    for _ in range(n_warmup):
        fn()

    log.info("  Benchmarking (%d) …", n_bench)
    latencies = []
    for i in range(n_bench):
        t0 = time.perf_counter()
        fn()
        latencies.append((time.perf_counter() - t0) * 1000)
        if (i + 1) % 25 == 0:
            log.info("    %d/%d  running p50=%.1f ms", i + 1, n_bench, _pct(latencies, 50))

    return latencies


# ── Detector benchmarks ────────────────────────────────────────────────────────

def _bench_detector_pytorch(
    frame: np.ndarray,
    model_name: str,
    n_warmup: int,
    n_bench: int,
) -> BenchResult:
    result = BenchResult(backend="pytorch", component="detector")
    log.info("[detector/pytorch] Loading %s …", model_name)
    try:
        from detector import PersonDetector
        det = PersonDetector(model_name=model_name)
        result.raw_ms = _run_benchmark(
            lambda: det.detect(frame), n_warmup, n_bench, "detector/pytorch"
        )
    except Exception as e:
        result.error = str(e)
        log.warning("[detector/pytorch] Failed: %s", e)
    return result


def _bench_detector_onnx(
    frame: np.ndarray,
    onnx_path: str,
    n_warmup: int,
    n_bench: int,
) -> BenchResult:
    result = BenchResult(backend="onnx", component="detector")
    if not os.path.exists(onnx_path):
        result.error = f"ONNX model not found: {onnx_path}"
        log.warning("[detector/onnx] %s", result.error)
        return result
    log.info("[detector/onnx] Loading %s …", onnx_path)
    try:
        from trt_detector import OnnxPersonDetector
        det = OnnxPersonDetector(onnx_path=onnx_path)
        result.raw_ms = _run_benchmark(
            lambda: det.detect(frame), n_warmup, n_bench, "detector/onnx"
        )
    except Exception as e:
        result.error = str(e)
        log.warning("[detector/onnx] Failed: %s", e)
    return result


def _bench_detector_trt(
    frame: np.ndarray,
    engine_path: str,
    n_warmup: int,
    n_bench: int,
) -> BenchResult:
    result = BenchResult(backend="trt", component="detector")
    if not os.path.exists(engine_path):
        result.error = f"TRT engine not found: {engine_path}"
        log.warning("[detector/trt] %s", result.error)
        return result
    log.info("[detector/trt] Loading %s …", engine_path)
    try:
        from trt_detector import TrtPersonDetector
        det = TrtPersonDetector(engine_path=engine_path)
        result.raw_ms = _run_benchmark(
            lambda: det.detect(frame), n_warmup, n_bench, "detector/trt"
        )
    except Exception as e:
        result.error = str(e)
        log.warning("[detector/trt] Failed: %s", e)
    return result


# ── Pose estimator benchmarks ──────────────────────────────────────────────────

def _bench_pose_pytorch(
    frame: np.ndarray,
    bbox: tuple,
    n_warmup: int,
    n_bench: int,
) -> BenchResult:
    result = BenchResult(backend="pytorch", component="pose")
    log.info("[pose/pytorch] Loading MMPose RTMPose …")
    try:
        from pose import PoseEstimator
        pose = PoseEstimator()
        result.raw_ms = _run_benchmark(
            lambda: pose.estimate_with_scores(frame, bbox),
            n_warmup, n_bench, "pose/pytorch",
        )
    except Exception as e:
        result.error = str(e)
        log.warning("[pose/pytorch] Failed: %s", e)
    return result


def _bench_pose_onnx(
    frame: np.ndarray,
    bbox: tuple,
    onnx_path: str,
    n_warmup: int,
    n_bench: int,
) -> BenchResult:
    result = BenchResult(backend="onnx", component="pose")
    if not os.path.exists(onnx_path):
        result.error = f"ONNX model not found: {onnx_path}"
        log.warning("[pose/onnx] %s", result.error)
        return result
    log.info("[pose/onnx] Loading %s …", onnx_path)
    try:
        from trt_pose import OnnxPoseEstimator
        pose = OnnxPoseEstimator(onnx_path=onnx_path)
        result.raw_ms = _run_benchmark(
            lambda: pose.estimate_with_scores(frame, bbox),
            n_warmup, n_bench, "pose/onnx",
        )
    except Exception as e:
        result.error = str(e)
        log.warning("[pose/onnx] Failed: %s", e)
    return result


def _bench_pose_trt(
    frame: np.ndarray,
    bbox: tuple,
    engine_path: str,
    n_warmup: int,
    n_bench: int,
) -> BenchResult:
    result = BenchResult(backend="trt", component="pose")
    if not os.path.exists(engine_path):
        result.error = f"TRT engine not found: {engine_path}"
        log.warning("[pose/trt] %s", result.error)
        return result
    log.info("[pose/trt] Loading %s …", engine_path)
    try:
        from trt_pose import TrtPoseEstimator
        pose = TrtPoseEstimator(engine_path=engine_path)
        result.raw_ms = _run_benchmark(
            lambda: pose.estimate_with_scores(frame, bbox),
            n_warmup, n_bench, "pose/trt",
        )
    except Exception as e:
        result.error = str(e)
        log.warning("[pose/trt] Failed: %s", e)
    return result


# ── Pipeline benchmark (YOLO + Pose sequential) ────────────────────────────────

def _bench_pipeline(
    frame: np.ndarray,
    args: argparse.Namespace,
    n_warmup: int,
    n_bench: int,
) -> List[BenchResult]:
    """Benchmark the full detection + pose pipeline for each backend pair."""
    results = []
    combos = [
        ("pytorch", "pytorch"),
        ("onnx",    "onnx"),
        ("trt",     "trt"),
    ]

    for det_be, pose_be in combos:
        label = f"pipeline/{det_be}"
        result = BenchResult(backend=det_be, component="pipeline")

        try:
            if det_be == "pytorch":
                from detector import PersonDetector
                det = PersonDetector(model_name=args.yolo_pt)
            elif det_be == "onnx":
                if not os.path.exists(args.yolo_onnx):
                    result.error = f"ONNX not found: {args.yolo_onnx}"
                    results.append(result)
                    continue
                from trt_detector import OnnxPersonDetector
                det = OnnxPersonDetector(onnx_path=args.yolo_onnx)
            else:
                if not os.path.exists(args.yolo_trt):
                    result.error = f"TRT not found: {args.yolo_trt}"
                    results.append(result)
                    continue
                from trt_detector import TrtPersonDetector
                det = TrtPersonDetector(engine_path=args.yolo_trt)

            if pose_be == "pytorch":
                from pose import PoseEstimator
                pose = PoseEstimator()
            elif pose_be == "onnx":
                if not os.path.exists(args.pose_onnx):
                    result.error = f"ONNX not found: {args.pose_onnx}"
                    results.append(result)
                    continue
                from trt_pose import OnnxPoseEstimator
                pose = OnnxPoseEstimator(onnx_path=args.pose_onnx)
            else:
                if not os.path.exists(args.pose_trt):
                    result.error = f"TRT not found: {args.pose_trt}"
                    results.append(result)
                    continue
                from trt_pose import TrtPoseEstimator
                pose = TrtPoseEstimator(engine_path=args.pose_trt)

            # Dummy bbox covering centre of frame
            h, w = frame.shape[:2]
            bbox = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)

            def _pipeline_once():
                bboxes = det.detect(frame)
                b = bboxes[0] if bboxes else bbox
                pose.estimate_with_scores(frame, b)

            log.info("[%s] Benchmarking pipeline …", label)
            result.raw_ms = _run_benchmark(_pipeline_once, n_warmup, n_bench, label)

        except Exception as e:
            result.error = str(e)
            log.warning("[%s] Failed: %s", label, e)

        results.append(result)

    return results


# ── Report generation ──────────────────────────────────────────────────────────

def _system_info() -> Dict:
    info: Dict = {}
    try:
        import torch
        info["pytorch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"]  = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
            info["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            )
    except ImportError:
        pass
    try:
        import tensorrt as trt
        info["tensorrt_version"] = trt.__version__
    except ImportError:
        info["tensorrt_version"] = "not installed"
    try:
        import onnxruntime as ort
        info["onnxruntime_version"] = ort.__version__
        info["onnxruntime_providers"] = ort.get_available_providers()
    except ImportError:
        info["onnxruntime_version"] = "not installed"
    return info


def _print_table(results: List[BenchResult]) -> None:
    print()
    print("┌─────────────┬──────────────┬───────────┬───────────┬───────────┬──────────┬──────────┐")
    print("│  backend    │  component   │   p50 ms  │   p90 ms  │   p99 ms  │  min ms  │  fps     │")
    print("├─────────────┼──────────────┼───────────┼───────────┼───────────┼──────────┼──────────┤")
    for r in results:
        if r.error:
            print(f"│  {r.backend:<11}│  {r.component:<12}│  {'ERROR':<9}│  {r.error[:9]:<9}│           │          │          │")
        else:
            print(
                f"│  {r.backend:<11}│  {r.component:<12}"
                f"│  {r.p50_ms:>7.2f}  │  {r.p90_ms:>7.2f}  │  {r.p99_ms:>7.2f}  "
                f"│  {r.min_ms:>6.2f}  │  {r.fps:>7.1f} │"
            )
    print("└─────────────┴──────────────┴───────────┴───────────┴───────────┴──────────┴──────────┘")
    print()

    # Speedup summary
    pytorch_det  = next((r for r in results if r.backend == "pytorch" and r.component == "detector" and not r.error), None)
    pytorch_pose = next((r for r in results if r.backend == "pytorch" and r.component == "pose"     and not r.error), None)

    for r in results:
        if r.error or r.backend == "pytorch":
            continue
        if r.component == "detector" and pytorch_det and pytorch_det.mean_ms > 0:
            speedup = pytorch_det.mean_ms / r.mean_ms
            print(f"  {r.backend} detector speedup vs PyTorch:  {speedup:.2f}×")
        if r.component == "pose" and pytorch_pose and pytorch_pose.mean_ms > 0:
            speedup = pytorch_pose.mean_ms / r.mean_ms
            print(f"  {r.backend} pose      speedup vs PyTorch:  {speedup:.2f}×")
    print()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    log.info("=" * 68)
    log.info("Step 8 — Inference Latency Benchmark")
    log.info("=" * 68)

    sys_info = _system_info()
    log.info("System: GPU=%s  CUDA=%s  TRT=%s  ORT=%s",
             sys_info.get("gpu", "N/A"),
             sys_info.get("cuda", "N/A"),
             sys_info.get("tensorrt_version", "N/A"),
             sys_info.get("onnxruntime_version", "N/A"))

    # Generate a realistic test frame (BGR, person-like)
    h = w = args.frame_size
    frame = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
    # Draw a rough person silhouette for realistic detection behaviour
    cv2.rectangle(frame, (w//3, h//6), (2*w//3, 5*h//6), (180, 120, 90), -1)
    cv2.circle(frame, (w//2, h//6), h//10, (200, 160, 120), -1)

    bbox = (w // 3, h // 6, 2 * w // 3, 5 * h // 6)
    results: List[BenchResult] = []

    # ── Detector benchmarks ────────────────────────────────────────────────────
    if args.component in ("detector", "all"):
        log.info("")
        log.info("── Detector benchmarks ──────────────────────────────────────────")
        if "pytorch" in args.backends:
            results.append(_bench_detector_pytorch(
                frame, args.yolo_pt, args.n_warmup, args.n_bench))
        if "onnx" in args.backends:
            results.append(_bench_detector_onnx(
                frame, args.yolo_onnx, args.n_warmup, args.n_bench))
        if "trt" in args.backends:
            results.append(_bench_detector_trt(
                frame, args.yolo_trt, args.n_warmup, args.n_bench))

    # ── Pose benchmarks ────────────────────────────────────────────────────────
    if args.component in ("pose", "all"):
        log.info("")
        log.info("── Pose estimator benchmarks ────────────────────────────────────")
        if "pytorch" in args.backends:
            results.append(_bench_pose_pytorch(
                frame, bbox, args.n_warmup, args.n_bench))
        if "onnx" in args.backends:
            results.append(_bench_pose_onnx(
                frame, bbox, args.pose_onnx, args.n_warmup, args.n_bench))
        if "trt" in args.backends:
            results.append(_bench_pose_trt(
                frame, bbox, args.pose_trt, args.n_warmup, args.n_bench))

    # ── Pipeline benchmarks ────────────────────────────────────────────────────
    if args.component in ("pipeline", "all"):
        log.info("")
        log.info("── Full pipeline benchmarks ──────────────────────────────────────")
        results.extend(_bench_pipeline(frame, args, args.n_warmup, args.n_bench))

    # ── Print results ──────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 68)
    log.info("RESULTS")
    log.info("=" * 68)
    _print_table(results)

    # ── Save JSON report ───────────────────────────────────────────────────────
    if args.output:
        report = {
            "system":  sys_info,
            "config": {
                "n_warmup":   args.n_warmup,
                "n_bench":    args.n_bench,
                "frame_size": args.frame_size,
                "component":  args.component,
                "backends":   args.backends,
            },
            "results": [r.to_dict() for r in results],
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        log.info("Report saved → %s", args.output)


if __name__ == "__main__":
    main()
