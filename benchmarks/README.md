# benchmarks/

Step 10 — Benchmarking & Stress Testing suite for the AI Fitness Trainer.

---

## Files

| Script | What it measures |
|--------|-----------------|
| `bench_fps.py` | Single-session FPS and per-stage latency breakdown |
| `bench_multi_session.py` | Concurrent session throughput and backpressure stress test |
| `bench_rep_accuracy.py` | Rep counting accuracy against synthetic ground-truth signals |
| `bench_memory.py` | RAM and GPU memory stability over long sessions (leak detection) |
| `report_generator.py` | Aggregates all JSON results into a Markdown + JSON report |

The inference backend comparison (`tools/benchmark_inference.py`) is part of Step 8.

---

## Quick Start

```bash
# 1. Install benchmark dependencies
pip install psutil gputil websockets aiohttp

# 2. Run accuracy test (no server needed — pure Python)
python benchmarks/bench_rep_accuracy.py --output results/accuracy.json

# 3. Run memory stability test (no server needed)
python benchmarks/bench_memory.py --n-frames 2000 --output results/memory.json

# 4. Run FPS benchmark (no server needed)
python benchmarks/bench_fps.py --n-bench 200 --output results/fps.json

# 5. Start the server, then run the multi-session stress test
uvicorn server.main_api:app --port 8000 &
python benchmarks/bench_multi_session.py --sessions 4 --duration 30 --output results/multi_session.json

# 6. Run inference backend comparison (Step 8 tool)
python tools/benchmark_inference.py --output results/inference.json

# 7. Generate the unified report
python benchmarks/report_generator.py --results-dir results/ --output results/BENCHMARK_REPORT
```

---

## Detailed Usage

### bench_fps.py

Measures the full processing pipeline for a single session. No server required.

```bash
# Defaults (200 frames, 480px synthetic frames, squat exercise):
python benchmarks/bench_fps.py

# Custom:
python benchmarks/bench_fps.py \
    --n-bench 500 \
    --frame-size 720 \
    --exercise pushup \
    --output results/fps.json

# Without BiLSTM (faster, tests pure detection/pose/FSM path):
python benchmarks/bench_fps.py --no-model
```

**Output example:**
```
Stage              p50 ms  p90 ms  p99 ms  min ms  max ms
────────────────────────────────────────────────────────────
0. Decode            0.82    1.10    1.80    0.60    3.20
1. Det+Track         8.40   10.20   14.10    6.10   18.90
2. Pose             12.30   15.80   21.40    9.20   28.10
3. Kalman Smooth     0.06    0.09    0.14    0.04    0.20
4. Angles            0.12    0.15    0.22    0.09    0.40
5. BiLSTM            2.10    2.80    3.90    1.80    5.20
6. FSM+Feedback      0.08    0.11    0.18    0.06    0.30
────────────────────────────────────────────────────────────
Effective FPS: 38.4
```

---

### bench_multi_session.py

Sends concurrent WebSocket frames to a running server. Requires server to be up.

```bash
# 4 sessions at 15 fps for 20 seconds:
python benchmarks/bench_multi_session.py \
    --sessions 4 --fps 15 --duration 20 \
    --server http://localhost:8000

# Ramp test (auto-scales 1 → 8 sessions):
python benchmarks/bench_multi_session.py --ramp --max-sessions 8 --duration 15

# High load:
python benchmarks/bench_multi_session.py \
    --sessions 16 --fps 10 --duration 60 \
    --output results/multi_session.json
```

**Key metrics:**
- `delivery_rate_pct` — frames that got a full response (not dropped / backpressured)
- `throughput_per_session_fps` — effective fps received per client
- `avg_track_stability` — fraction of frames where the tracker had a confirmed target
- `total_id_switches` — number of times the tracker switched to a different person

---

### bench_rep_accuracy.py

No server required. Tests RepCounter with synthetic sine-wave angle signals.

```bash
# All 15 exercises with default 85% accuracy threshold:
python benchmarks/bench_rep_accuracy.py

# Stricter threshold:
python benchmarks/bench_rep_accuracy.py --threshold 0.90

# Specific exercises only:
python benchmarks/bench_rep_accuracy.py --exercise squat pushup bicep_curl

# Save results:
python benchmarks/bench_rep_accuracy.py --output results/accuracy.json
```

**Test types per exercise:**

| Test | Ground truth | What it validates |
|------|-------------|-------------------|
| `clean_signal` | N reps | Normal smooth reps |
| `partial_rom_no_count` | 0 reps | Half-ROM must NOT count |
| `noisy_signal` | N reps | 6% noise tolerance |
| `fast_2x_speed` | N reps | 2× speed tolerance |
| `false_positive_micro_osc` | 0 reps | Micro-movement must NOT count |
| `double_count_guard` | 1 rep | Bounce at bottom = 1 rep only |

Exit code 0 = all passed, 1 = failures.

---

### bench_memory.py

No server required. Runs N frames and monitors RSS over time.

```bash
# 2000 frames (default):
python benchmarks/bench_memory.py

# Long-running stability test:
python benchmarks/bench_memory.py --n-frames 10000 --sample-every 200

# Include GPU memory monitoring:
python benchmarks/bench_memory.py --gpu --output results/memory.json
```

**Leak detection:** fits a linear regression to the memory trace.
If growth > `--leak-threshold-mb` (default 5 MB) per 1000 frames, verdict is `WARN`.

---

### report_generator.py

Combines all JSON results into a unified report.

```bash
# After running all benchmarks:
python benchmarks/report_generator.py

# Custom paths:
python benchmarks/report_generator.py \
    --results-dir my_results/ \
    --output reports/v8_benchmark
```

Produces:
- `results/BENCHMARK_REPORT.md` — Human-readable Markdown
- `results/BENCHMARK_REPORT.json` — Machine-readable full report

---

## Expected Performance Targets

These are reference targets on an **RTX 3080, PyTorch backend, 480p input**:

| Metric | Target | Notes |
|--------|--------|-------|
| Single-session FPS | ≥ 25 | PyTorch; ≥ 40 with TRT |
| Total p99 latency | ≤ 100 ms | Single session |
| 4-session delivery rate | ≥ 90% | At 15 fps per client |
| 8-session delivery rate | ≥ 75% | At 10 fps per client |
| Rep accuracy (clean) | ≥ 90% | Smooth synthetic signal |
| Rep accuracy (noisy) | ≥ 85% | 6% noise |
| False positive rate | 0% | Micro-oscillation & partial ROM |
| RAM growth / 1k frames | ≤ 5 MB | Long session stability |
| GPU growth / 1k frames | ≤ 2 MB | After warmup |

---

## Tuning the server for benchmarks

```bash
# Lower detection interval = more accurate but slower
DETECTION_INTERVAL=2 uvicorn server.main_api:app

# More inference threads = higher multi-session throughput
QUEUE_INFERENCE_THREADS=8 uvicorn server.main_api:app

# TRT backend = significant speedup
DETECTOR_BACKEND=trt POSE_BACKEND=trt uvicorn server.main_api:app
```
