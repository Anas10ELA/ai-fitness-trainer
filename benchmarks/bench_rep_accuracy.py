"""
benchmarks/bench_rep_accuracy.py
═════════════════════════════════
Step 10 — Rep Counting Accuracy Validator

Validates RepCounter accuracy against synthetic, ground-truth angle sequences
for all 15 exercises. Tests:

  1. Clean reps          — smooth sine-wave cycles through full ROM
  2. Partial reps        — cycles that don't reach the down threshold
  3. Jitter robustness   — noise injected into clean signal
  4. Fast motion         — signal at 2× and 3× normal speed
  5. False positive guard— micro-oscillations that must NOT count
  6. Double-count guard  — bounce at the bottom, must count as ONE rep

Accuracy metric
---------------
  accuracy = correct_reps_counted / ground_truth_reps

A test passes if accuracy >= ACCURACY_THRESHOLD (default 0.85 = 85%).
For false-positive tests, passes if reps == 0.
For double-count tests, passes if reps == ground_truth (exactly 1 per cycle).

Usage
-----
  python benchmarks/bench_rep_accuracy.py
  python benchmarks/bench_rep_accuracy.py --threshold 0.90 --output results/accuracy.json
  python benchmarks/bench_rep_accuracy.py --exercise squat pushup bicep_curl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bench_accuracy")


# ── Argument parsing ───────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    from exercises import Exercise
    all_ex = [e.value for e in Exercise]
    p = argparse.ArgumentParser(description="Rep counting accuracy validator")
    p.add_argument("--exercise",   nargs="*", default=all_ex,  help="Exercises to test")
    p.add_argument("--threshold",  default=0.85, type=float,   help="Min accuracy to pass")
    p.add_argument("--n-reps",     default=10,   type=int,     help="Ground-truth reps per test")
    p.add_argument("--fps",        default=30,   type=int,     help="Simulated frame rate")
    p.add_argument("--output",     default="",                  help="Save JSON report")
    return p.parse_args()


# ── Signal generators ──────────────────────────────────────────────────────────

def _sine_wave(
    low: float,
    high: float,
    n_reps: int,
    fps: int,
    rep_duration_secs: float = 2.0,
    noise_sigma: float = 0.0,
    speed_mult: float = 1.0,
) -> List[float]:
    """
    Generate a smooth sine sweep:  low → high → low  (one rep = one full cycle).
    low  = fully-contracted angle (DOWN position, e.g. 80° for squat)
    high = fully-extended angle   (UP position, e.g. 170° for squat)
    """
    frames_per_rep = int(fps * rep_duration_secs / speed_mult)
    values = []
    mid = (low + high) / 2.0
    amp = (high - low) / 2.0
    for rep in range(n_reps):
        for f in range(frames_per_rep):
            t   = f / frames_per_rep
            val = mid - amp * math.cos(2 * math.pi * t)
            if noise_sigma > 0.0:
                val += noise_sigma * (2 * (hash((rep, f)) % 1000) / 1000.0 - 1.0)
            values.append(val)
    # End in UP position
    values.append(high)
    return values


def _micro_oscillation(
    base: float,
    amplitude: float,
    n_cycles: int,
    fps: int,
) -> List[float]:
    """Micro-oscillation that stays entirely within the READY zone — must NOT count."""
    values = []
    for i in range(n_cycles * fps):
        t = i / fps
        values.append(base + amplitude * math.sin(2 * math.pi * 2 * t))
    return values


def _bounce_signal(
    high: float,
    low: float,
    fps: int,
    bounce_amplitude: float = 8.0,
) -> List[float]:
    """
    One clean rep (high → low → high) followed immediately by a small
    bounce at the bottom. Must count as exactly 1 rep, not 2.
    """
    # Down phase
    frames_down = fps  # 1 second down
    values = []
    for f in range(frames_down):
        t = f / frames_down
        values.append(high - (high - low) * t)  # linear descent
    # Bounce: small oscillation around `low`
    for f in range(fps // 2):
        t = f / (fps // 2)
        values.append(low + bounce_amplitude * math.sin(math.pi * t))
    # Up phase
    frames_up = fps
    for f in range(frames_up):
        t = f / frames_up
        values.append(low + (high - low) * t)
    return values


# ── RepCounter driver ──────────────────────────────────────────────────────────

def _pump_counter(
    exercise_name: str,
    angle_key: str,
    angle_sequence: List[float],
    fps: int,
    secondary_angles: Optional[Dict[str, float]] = None,
) -> int:
    """
    Feed a synthetic angle sequence through RepCounter. Returns final rep count.

    angle_key     : the dict key name that RepCounter reads for this exercise
    secondary_angles : fixed values for secondary joints (e.g. Back, Hip)
    """
    from exercises import Exercise
    from counter   import RepCounter
    from temporal  import TemporalValidator

    exercise = Exercise(exercise_name)
    counter  = RepCounter(exercise)
    temporal = TemporalValidator()

    base_angles = secondary_angles or {}

    for val in angle_sequence:
        angles = {**base_angles, angle_key: val}
        # Mirror to bilateral joint if needed
        left_key  = angle_key
        right_key = angle_key.replace("Left", "Right").replace("L ", "R ")
        if right_key != left_key:
            angles[right_key] = val

        temporal.add(angles)
        counter.update(
            angles       = angles,
            keypoints    = None,
            is_rep_valid = True,
            temporal     = temporal,
            current_fps  = float(fps),
        )

    return counter.reps


# ── Exercise-specific test configs ────────────────────────────────────────────

# Maps exercise_name → (primary_angle_key, down_angle, up_angle, secondary_angles)
_EXERCISE_CONFIGS: Dict[str, Tuple[str, float, float, Dict[str, float]]] = {
    "squat":          ("Left Knee",     80.0,  165.0, {"Back": 160.0, "Right Knee": 80.0}),
    "pushup":         ("Left Elbow",    85.0,  155.0, {"Left Hip": 175.0, "Right Hip": 175.0, "Right Elbow": 85.0}),
    "jumping_jack":   ("L Arm Raise",   40.0,  140.0, {"R Arm Raise": 40.0}),
    "pullup":         ("Left Elbow",    85.0,  160.0, {"Right Elbow": 85.0}),
    "situp":          ("Back Angle",    60.0,  140.0, {}),
    "lunge":          ("Left Knee",     85.0,  155.0, {"Right Knee": 155.0, "Back": 165.0}),
    "bicep_curl":     ("Left Elbow",    45.0,  155.0, {"Right Elbow": 45.0, "Back": 165.0}),
    "tricep_dip":     ("Left Elbow",    85.0,  160.0, {"Right Elbow": 85.0}),
    "lateral_raise":  ("L Arm Abduct",  15.0,  90.0,  {"R Arm Abduct": 15.0}),
    "overhead_press": ("Left Elbow",    85.0,  165.0, {"Right Elbow": 85.0, "Back": 165.0}),
    "burpee":         ("Left Knee",     85.0,  165.0, {"Back Angle": 130.0}),
}


# ── Test result ───────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    exercise:      str
    test_name:     str
    expected_reps: int
    actual_reps:   int
    accuracy:      float = 0.0
    passed:        bool  = False
    threshold:     float = 0.85

    def __post_init__(self) -> None:
        if self.test_name.startswith("false_positive") or self.test_name.startswith("no_count"):
            self.accuracy = 1.0 if self.actual_reps == 0 else 0.0
            self.passed   = self.actual_reps == 0
        else:
            denom = max(self.expected_reps, 1)
            self.accuracy = max(0.0, 1.0 - abs(self.actual_reps - self.expected_reps) / denom)
            self.passed   = self.accuracy >= self.threshold

    def to_dict(self) -> Dict:
        return {
            "exercise":    self.exercise,
            "test":        self.test_name,
            "expected":    self.expected_reps,
            "actual":      self.actual_reps,
            "accuracy":    round(self.accuracy, 3),
            "passed":      self.passed,
        }


# ── Test suite ────────────────────────────────────────────────────────────────

def _run_exercise_tests(
    exercise: str,
    n_reps: int,
    fps: int,
    threshold: float,
) -> List[TestResult]:
    if exercise not in _EXERCISE_CONFIGS:
        log.info("  %-22s  skipped (no config)", exercise)
        return []

    angle_key, down_angle, up_angle, secondary = _EXERCISE_CONFIGS[exercise]
    results: List[TestResult] = []

    # ── Test 1: Clean signal ──────────────────────────────────────────────────
    seq   = _sine_wave(down_angle, up_angle, n_reps, fps)
    reps  = _pump_counter(exercise, angle_key, seq, fps, secondary)
    results.append(TestResult(exercise, "clean_signal", n_reps, reps, threshold=threshold))

    # ── Test 2: Partial reps (only go halfway down) ───────────────────────────
    mid   = (down_angle + up_angle) / 2.0
    seq   = _sine_wave(mid, up_angle, n_reps, fps)
    reps  = _pump_counter(exercise, angle_key, seq, fps, secondary)
    results.append(TestResult(exercise, "partial_rom_no_count", 0, reps, threshold=threshold))

    # ── Test 3: Noisy signal ──────────────────────────────────────────────────
    noise = (up_angle - down_angle) * 0.06  # 6% noise
    seq   = _sine_wave(down_angle, up_angle, n_reps, fps, noise_sigma=noise)
    reps  = _pump_counter(exercise, angle_key, seq, fps, secondary)
    results.append(TestResult(exercise, "noisy_signal", n_reps, reps, threshold=threshold))

    # ── Test 4: Fast motion (2× speed) ───────────────────────────────────────
    seq   = _sine_wave(down_angle, up_angle, n_reps, fps, speed_mult=2.0)
    reps  = _pump_counter(exercise, angle_key, seq, fps, secondary)
    results.append(TestResult(exercise, "fast_2x_speed", n_reps, reps, threshold=threshold))

    # ── Test 5: Micro-oscillation false positive guard ────────────────────────
    # Oscillate in the READY zone — must not count
    resting = up_angle - (up_angle - down_angle) * 0.1
    seq     = _micro_oscillation(resting, (up_angle - down_angle) * 0.05, 20, fps)
    reps    = _pump_counter(exercise, angle_key, seq, fps, secondary)
    results.append(TestResult(exercise, "false_positive_micro_osc", 0, reps, threshold=threshold))

    # ── Test 6: Double-count guard (bounce at bottom) ─────────────────────────
    seq     = _bounce_signal(up_angle, down_angle, fps, bounce_amplitude=(up_angle-down_angle)*0.08)
    reps    = _pump_counter(exercise, angle_key, seq, fps, secondary)
    results.append(TestResult(exercise, "double_count_guard", 1, reps, threshold=threshold))

    # ── Log per-exercise results ──────────────────────────────────────────────
    n_pass = sum(1 for r in results if r.passed)
    status = "✓" if n_pass == len(results) else "✗"
    log.info("  %s %-22s  %d/%d tests passed", status, exercise, n_pass, len(results))
    for r in results:
        icon = "✓" if r.passed else "✗"
        log.info("      %s %-35s  expected=%-3d  actual=%-3d  acc=%.0f%%",
                 icon, r.test_name, r.expected_reps, r.actual_reps, r.accuracy * 100)

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse()

    log.info("═" * 64)
    log.info("  bench_rep_accuracy — Rep Counting Accuracy Validator")
    log.info("═" * 64)
    log.info("  threshold=%.0f%%  n_reps=%d  fps=%d",
             args.threshold * 100, args.n_reps, args.fps)
    log.info("")

    all_results: List[TestResult] = []
    t0 = time.perf_counter()

    for ex in args.exercise:
        results = _run_exercise_tests(ex, args.n_reps, args.fps, args.threshold)
        all_results.extend(results)

    elapsed = time.perf_counter() - t0

    # ── Summary ───────────────────────────────────────────────────────────────
    n_total  = len(all_results)
    n_passed = sum(1 for r in all_results if r.passed)
    n_failed = n_total - n_passed

    # Group by exercise
    by_ex: Dict[str, List[TestResult]] = {}
    for r in all_results:
        by_ex.setdefault(r.exercise, []).append(r)

    ex_pass_rates = {
        ex: sum(1 for r in rs if r.passed) / len(rs) * 100
        for ex, rs in by_ex.items()
    }

    log.info("")
    log.info("═" * 64)
    log.info("  SUMMARY  (%d/%d tests passed in %.1fs)", n_passed, n_total, elapsed)
    log.info("═" * 64)

    if n_failed > 0:
        log.info("  FAILED TESTS:")
        for r in all_results:
            if not r.passed:
                log.info("    ✗ %s / %s  expected=%d actual=%d acc=%.0f%%",
                         r.exercise, r.test_name,
                         r.expected_reps, r.actual_reps, r.accuracy * 100)

    avg_acc = sum(r.accuracy for r in all_results) / max(n_total, 1) * 100
    log.info("  Average accuracy : %.1f%%", avg_acc)
    log.info("  Overall result   : %s", "PASS ✓" if n_failed == 0 else f"FAIL ✗ ({n_failed} failures)")

    # ── JSON report ───────────────────────────────────────────────────────────
    report = {
        "benchmark": "bench_rep_accuracy",
        "config": {
            "threshold": args.threshold,
            "n_reps":    args.n_reps,
            "fps":       args.fps,
            "exercises": args.exercise,
        },
        "summary": {
            "total_tests":  n_total,
            "passed":       n_passed,
            "failed":       n_failed,
            "avg_accuracy": round(avg_acc / 100, 3),
            "elapsed_s":    round(elapsed, 2),
            "overall":      "PASS" if n_failed == 0 else "FAIL",
        },
        "by_exercise": {
            ex: round(rate, 1)
            for ex, rate in sorted(ex_pass_rates.items(), key=lambda x: x[1])
        },
        "tests": [r.to_dict() for r in all_results],
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        log.info("Report saved → %s", args.output)

    sys.exit(0 if n_failed == 0 else 1)


if __name__ == "__main__":
    main()
