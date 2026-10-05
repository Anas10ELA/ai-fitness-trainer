"""
benchmarks/bench_multi_session.py
══════════════════════════════════
Step 10 — Concurrent Session Stress Test

Spawns N concurrent WebSocket sessions against a running server and measures:
  • Per-session throughput (frames/sec received)
  • Per-session latency (p50, p90, p99)
  • Server-side backpressure events (queue-full drops)
  • Tracker stability across sessions (no subject-switching)
  • Memory growth over the test window
  • Server metric deltas (queue depth, processed frames, drop rate)

The test sends real base64-encoded JPEG frames so the full pipeline fires —
detection, tracking, pose, FSM, rep counting.

Usage
-----
  # First start the server:
  uvicorn server.main_api:app --port 8000

  # Then run:
  python benchmarks/bench_multi_session.py

  # Custom parameters:
  python benchmarks/bench_multi_session.py \\
      --sessions 8 --fps 15 --duration 30 \\
      --server http://localhost:8000 \\
      --output results/multi_session.json

  # Ramp test (1 → N sessions, observe degradation):
  python benchmarks/bench_multi_session.py --ramp --max-sessions 16

Requirements
------------
  pip install websockets aiohttp
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bench_multi")


# ── Argument parsing ───────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-session concurrent stress test")
    p.add_argument("--server",      default="http://localhost:8000", help="Server base URL")
    p.add_argument("--sessions",    default=4,    type=int,  help="Concurrent sessions")
    p.add_argument("--fps",         default=15,   type=int,  help="Frames per second per session")
    p.add_argument("--duration",    default=20,   type=int,  help="Test duration in seconds")
    p.add_argument("--exercise",    default="squat")
    p.add_argument("--frame-size",  default=320,  type=int,  help="Synthetic frame size")
    p.add_argument("--output",      default="",              help="Save JSON report")
    p.add_argument("--ramp",        action="store_true",     help="Ramp from 1 to --sessions")
    p.add_argument("--max-sessions",default=8,   type=int,   help="Ramp target session count")
    return p.parse_args()


# ── Synthetic frame ───────────────────────────────────────────────────────────

def _make_frame_b64(size: int) -> str:
    frame = np.random.randint(40, 140, (size, size, 3), dtype=np.uint8)
    cx = size // 2
    cv2.rectangle(frame, (cx - size//6, size//5), (cx + size//6, 4*size//5), (170, 120, 90), -1)
    cv2.circle(frame, (cx, size//6), size//12, (200, 160, 120), -1)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf.tobytes()).decode()


# ── Per-session result ────────────────────────────────────────────────────────

@dataclass
class SessionResult:
    session_idx:     int
    session_id:      str = ""
    frames_sent:     int = 0
    frames_received: int = 0
    backpressure_events: int = 0
    errors:          int = 0
    latencies_ms:    List[float] = field(default_factory=list)
    final_reps:      int = 0
    track_stability: float = 0.0
    track_id_switches: int = 0
    error_messages:  List[str] = field(default_factory=list)

    @property
    def p50(self) -> float:
        return _pct(self.latencies_ms, 50)

    @property
    def p90(self) -> float:
        return _pct(self.latencies_ms, 90)

    @property
    def p99(self) -> float:
        return _pct(self.latencies_ms, 99)

    @property
    def throughput(self) -> float:
        return self.frames_received / max(len(self.latencies_ms), 1) * 1000 if self.latencies_ms else 0

    def to_dict(self) -> Dict:
        return {
            "session_idx":       self.session_idx,
            "session_id":        self.session_id,
            "frames_sent":       self.frames_sent,
            "frames_received":   self.frames_received,
            "backpressure":      self.backpressure_events,
            "errors":            self.errors,
            "p50_ms":            round(self.p50, 2),
            "p90_ms":            round(self.p90, 2),
            "p99_ms":            round(self.p99, 2),
            "final_reps":        self.final_reps,
            "track_stability":   self.track_stability,
            "track_id_switches": self.track_id_switches,
        }


def _pct(data: List[float], p: int) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    return s[min(int(len(s) * p / 100), len(s) - 1)]


# ── Server helpers ────────────────────────────────────────────────────────────

async def _create_session(
    http_base: str,
    exercise: str,
    idx: int,
) -> Optional[str]:
    try:
        import aiohttp
    except ImportError:
        log.error("aiohttp not installed. pip install aiohttp")
        sys.exit(1)
    async with aiohttp.ClientSession() as s:
        try:
            r = await s.post(
                f"{http_base}/session/start",
                json={"exercise": exercise},
                timeout=aiohttp.ClientTimeout(total=10),
            )
            data = await r.json()
            sid  = data.get("session_id", "")
            log.debug("Session %d created: %s", idx, sid)
            return sid
        except Exception as e:
            log.error("Session %d create failed: %s", idx, e)
            return None


async def _get_server_metrics(http_base: str) -> Dict:
    try:
        import aiohttp
        async with aiohttp.ClientSession() as s:
            r = await s.get(
                f"{http_base}/metrics",
                timeout=aiohttp.ClientTimeout(total=5),
            )
            return await r.json()
    except Exception:
        return {}


# ── Single session worker ─────────────────────────────────────────────────────

async def _run_session(
    ws_base: str,
    http_base: str,
    idx: int,
    exercise: str,
    fps: int,
    duration_secs: int,
    frame_b64: str,
) -> SessionResult:
    result = SessionResult(session_idx=idx)

    session_id = await _create_session(http_base, exercise, idx)
    if session_id is None:
        result.errors += 1
        result.error_messages.append("Failed to create session")
        return result

    result.session_id = session_id
    ws_url = f"{ws_base}/stream/{session_id}"

    interval = 1.0 / fps
    deadline  = time.perf_counter() + duration_secs

    try:
        import websockets
    except ImportError:
        log.error("websockets not installed. pip install websockets")
        sys.exit(1)

    try:
        async with websockets.connect(
            ws_url,
            ping_interval=20,
            ping_timeout=10,
            max_size=10 * 1024 * 1024,
        ) as ws:
            while time.perf_counter() < deadline:
                t_send = time.perf_counter()
                try:
                    await ws.send(json.dumps({
                        "frame_b64": frame_b64,
                        "exercise":  exercise,
                    }))
                    result.frames_sent += 1
                except Exception as e:
                    result.errors += 1
                    result.error_messages.append(f"send: {e}")
                    break

                try:
                    raw  = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    msg  = json.loads(raw)
                    t_rt = (time.perf_counter() - t_send) * 1000

                    msg_type = msg.get("type", "")
                    if msg_type == "backpressure":
                        result.backpressure_events += 1
                        # Back off for one interval to relieve pressure
                        await asyncio.sleep(interval * 2)
                        continue

                    if "error" in msg and "reps" not in msg:
                        result.errors += 1
                        result.error_messages.append(str(msg.get("error", "")))
                        continue

                    result.frames_received += 1
                    result.latencies_ms.append(t_rt)
                    result.final_reps        = msg.get("reps", 0)
                    result.track_stability   = msg.get("track_stability", 0.0)
                    result.track_id_switches = msg.get("track_id_switches", 0)

                except asyncio.TimeoutError:
                    result.errors += 1
                    result.error_messages.append("recv timeout")
                except Exception as e:
                    result.errors += 1
                    result.error_messages.append(f"recv: {e}")

                # Pace to target FPS
                elapsed = time.perf_counter() - t_send
                sleep   = max(0.0, interval - elapsed)
                if sleep > 0:
                    await asyncio.sleep(sleep)

    except Exception as e:
        result.errors += 1
        result.error_messages.append(f"ws connect: {e}")

    return result


# ── Main benchmark ────────────────────────────────────────────────────────────

async def _run_concurrent(
    args: argparse.Namespace,
    n_sessions: int,
) -> Dict:
    http_base = args.server.rstrip("/")
    ws_base   = http_base.replace("http://", "ws://").replace("https://", "wss://")

    log.info("Running %d concurrent sessions @ %d fps for %ds …",
             n_sessions, args.fps, args.duration)

    # Pre-bake one frame (all sessions share the same synthetic frame)
    frame_b64 = _make_frame_b64(args.frame_size)

    # Snapshot server metrics before test
    metrics_before = await _get_server_metrics(http_base)

    # Launch all sessions concurrently
    tasks = [
        _run_session(
            ws_base, http_base, i, args.exercise,
            args.fps, args.duration, frame_b64,
        )
        for i in range(n_sessions)
    ]
    t0      = time.perf_counter()
    results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - t0

    # Snapshot metrics after test
    metrics_after = await _get_server_metrics(http_base)

    # ── Aggregate ────────────────────────────────────────────────────────────
    all_latencies = [l for r in results for l in r.latencies_ms]
    total_sent     = sum(r.frames_sent     for r in results)
    total_received = sum(r.frames_received for r in results)
    total_bp       = sum(r.backpressure_events for r in results)
    total_errors   = sum(r.errors          for r in results)
    avg_stability  = (
        sum(r.track_stability for r in results if r.track_stability > 0) /
        max(sum(1 for r in results if r.track_stability > 0), 1)
    )
    id_switches    = sum(r.track_id_switches for r in results)

    throughput_total = total_received / elapsed

    report = {
        "benchmark": "bench_multi_session",
        "config": {
            "n_sessions":  n_sessions,
            "fps":         args.fps,
            "duration_s":  args.duration,
            "frame_size":  args.frame_size,
            "exercise":    args.exercise,
        },
        "aggregate": {
            "total_frames_sent":     total_sent,
            "total_frames_received": total_received,
            "delivery_rate_pct":     round(total_received / max(total_sent, 1) * 100, 1),
            "total_backpressure":    total_bp,
            "total_errors":          total_errors,
            "throughput_total_fps":  round(throughput_total, 1),
            "throughput_per_session_fps": round(throughput_total / n_sessions, 1),
            "p50_ms":                round(_pct(all_latencies, 50), 2),
            "p90_ms":                round(_pct(all_latencies, 90), 2),
            "p99_ms":                round(_pct(all_latencies, 99), 2),
            "avg_track_stability":   round(avg_stability, 3),
            "total_id_switches":     id_switches,
            "elapsed_s":             round(elapsed, 1),
        },
        "per_session": [r.to_dict() for r in results],
        "server_metrics_delta": _diff_metrics(metrics_before, metrics_after),
    }

    _print_aggregate(report["aggregate"], n_sessions)
    return report


def _diff_metrics(before: Dict, after: Dict) -> Dict:
    delta = {}
    for key in ("total_submitted", "total_processed", "total_dropped"):
        b = before.get(key, 0)
        a = after.get(key, 0)
        delta[key + "_delta"] = a - b
    return delta


def _print_aggregate(agg: Dict, n: int) -> None:
    log.info("  Sessions     : %d", n)
    log.info("  Delivered    : %.1f%%  (%d / %d frames)",
             agg["delivery_rate_pct"],
             agg["total_frames_received"],
             agg["total_frames_sent"])
    log.info("  Backpressure : %d events", agg["total_backpressure"])
    log.info("  Errors       : %d", agg["total_errors"])
    log.info("  Throughput   : %.1f fps total  (%.1f fps/session)",
             agg["throughput_total_fps"], agg["throughput_per_session_fps"])
    log.info("  Latency      : p50=%.1fms  p90=%.1fms  p99=%.1fms",
             agg["p50_ms"], agg["p90_ms"], agg["p99_ms"])
    log.info("  Track stable : %.1f%%  (id_switches=%d)",
             agg["avg_track_stability"] * 100, agg["total_id_switches"])


async def _run_ramp(args: argparse.Namespace) -> List[Dict]:
    """Run with 1, 2, 4, … sessions up to max_sessions and collect results."""
    counts  = []
    n = 1
    while n <= args.max_sessions:
        counts.append(n)
        n *= 2
    if counts[-1] != args.max_sessions:
        counts.append(args.max_sessions)

    ramp_results = []
    for n in counts:
        log.info("")
        log.info("══ Ramp: %d session(s) ══", n)
        r = await _run_concurrent(args, n)
        ramp_results.append(r)
        await asyncio.sleep(2)  # let the server breathe

    log.info("")
    log.info("═" * 56)
    log.info("  RAMP SUMMARY")
    log.info("  %-10s  %-10s  %-10s  %-10s  %-10s",
             "sessions", "fps/sess", "p50 ms", "p90 ms", "delivery%")
    log.info("  " + "─" * 54)
    for r in ramp_results:
        a = r["aggregate"]
        log.info("  %-10d  %-10.1f  %-10.1f  %-10.1f  %-10.1f%%",
                 r["config"]["n_sessions"],
                 a["throughput_per_session_fps"],
                 a["p50_ms"], a["p90_ms"],
                 a["delivery_rate_pct"])
    log.info("═" * 56)
    return ramp_results


def main() -> None:
    args = _parse()

    log.info("═" * 60)
    log.info("  bench_multi_session — Concurrent Session Stress Test")
    log.info("═" * 60)
    log.info("  server=%s  exercise=%s", args.server, args.exercise)

    if args.ramp:
        results = asyncio.run(_run_ramp(args))
        final_report = {"benchmark": "ramp", "ramp_results": results}
    else:
        result = asyncio.run(_run_concurrent(args, args.sessions))
        final_report = result

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(final_report, f, indent=2)
        log.info("Report saved → %s", args.output)


if __name__ == "__main__":
    main()
