# AI Fitness Trainer

Real-time AI fitness coaching system using YOLOv8 person detection, RTMPose keypoint estimation, and exercise-specific DLEngine models for state prediction, rep counting, and form feedback gating.

## Portfolio highlights

- Processes webcam or video frames through person detection, tracking, pose
  estimation, and exercise-specific repetition-state prediction.
- Counts repetitions with a stabilized state machine and gives exercise-aware
  form cues using joint geometry and temporal checks.
- Supports 15 exercises, a desktop OpenCV experience, and a FastAPI/WebSocket
  service with session handling, metrics, and optional spoken instructions.
- Includes inference benchmarks, automated tests, Docker/Nginx deployment
  files, and optional ONNX/TensorRT backends.

This public source repository intentionally excludes trained checkpoints,
base weights, raw videos, processed training data, virtual environments,
runtime caches, and benchmark outputs. Obtain permission for your chosen
training data and model weights before adding them locally under `data/raw/`
and `checkpoints_v3/`. The YOLOv8 base weight is downloaded by Ultralytics
when needed rather than stored here.

## Architecture

```
ai_fitness_trainer/
├── main.py                  ← Desktop app (OpenCV window)
├── config.py                ← Centralised configuration (all env vars)
├── startup.py               ← Pre-flight checks & environment validation
├── shutdown.py              ← Graceful shutdown (SIGTERM / SIGINT)
│
├── detector.py              ← YOLOv8 person detection (PyTorch)
├── pose.py                  ← RTMPose keypoint estimation (MMPose)
├── trt_detector.py          ← TensorRT / ONNX detector drop-in (Step 8)
├── trt_pose.py              ← TensorRT / ONNX pose estimator drop-in (Step 8)
│
├── tracker.py               ← Persistent IoU-based person tracker (Step 2)
├── frame_processor.py       ← Hybrid pipeline: detect→track→pose (Step 3)
├── counter.py               ← Rep counting FSM with hysteresis (Step 6)
├── feedback.py              ← Form feedback engine (15 exercises)
├── angles.py                ← Joint angle geometry
├── temporal.py              ← Temporal motion validator
├── dl_engine.py             ← Sliding-window MLP inference + anchor-FSM counter
├── utils.py                 ← Kalman filter, FPS counters, drawing (Step 7)
├── exercises.py             ← Exercise registry (15 exercises)
├── voice_cues.py            ← Natural English pre-exercise scripts
├── voice_coach.py           ← TTS generation + cached audio URLs
├── debug_overlay.py         ← Debug visualisation overlays
│
├── server/                  ← FastAPI WebSocket server
│   ├── main_api.py          ← API routes, lifespan, WS endpoint
│   ├── inference_queue.py   ← Async producer-consumer queue (Step 5)
│   ├── session_manager.py   ← Session lifecycle management
│   ├── frame_processor.py   ← Per-frame processing pipeline
│   └── schemas.py           ← Pydantic request/response models
│
├── checkpoints_v3/          ← dl_model_<exercise>.pt DLEngine checkpoints
├── scripts/train_dl_v4.py   ← Current per-exercise sliding-window trainer
├── model/                   ← Archived/experimental model work
│
├── tools/                   ← Step 8: Export & benchmark tools
│   ├── export_yolo_onnx.py    ← YOLOv8 → ONNX / TRT FP16
│   ├── export_rtmpose_onnx.py ← RTMPose → ONNX / TRT FP16
│   └── benchmark_inference.py ← PyTorch vs ONNX vs TRT latency
│
├── benchmarks/              ← Step 10: Benchmarking & stress testing
│   ├── bench_fps.py           ← Single-session FPS + per-stage latency
│   ├── bench_multi_session.py ← Concurrent session stress test
│   ├── bench_rep_accuracy.py  ← Rep counting accuracy validator
│   ├── bench_memory.py        ← Memory stability / leak detection
│   ├── report_generator.py    ← Unified Markdown + JSON report
│   └── README.md              ← Full benchmark usage guide
│
├── scripts/                 ← Data pipeline scripts
│   ├── extract_keypoints.py   ← Extract keypoints from video
│   ├── build_dataset.py       ← Build train/val/test .npz splits
│   └── label_tool.py          ← Manual annotation tool
│
├── tests/                   ← Unit & integration tests
│   ├── test_counter.py        ← RepCounter FSM tests
│   ├── test_inference_queue.py← Async queue tests
│   └── test_kalman.py         ← Kalman filter tests
│
├── deployment/              ← Docker & Nginx
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── nginx.conf
│
├── models/                  ← TRT/ONNX engine files (gitignored)
├── data/processed/          ← Training data (gitignored)
├── requirements.txt
├── .env.example
├── setup.sh                 ← Linux/macOS one-click setup
└── setup.bat                ← Windows one-click setup
```

---

## Quick Start

### 1. Install

```bash
cp .env.example .env          # configure your environment
bash setup.sh                 # Linux / macOS
# OR
setup.bat                     # Windows
```

### 2. Desktop App

```bash
source .venv/bin/activate
python main.py                          # webcam, default exercise: squat
python main.py --exercise pushup        # specific exercise
python main.py --source video.mp4       # video file
python main.py --list-exercises         # list all 15 exercises
```

**Keyboard shortcuts in desktop app:**

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Reset rep count |
| `1–9` | Switch exercise (see `--list-exercises`) |

### 3. API Server

```bash
uvicorn server.main_api:app --host 0.0.0.0 --port 8000

# Or via Docker:
docker-compose -f deployment/docker-compose.yml up -d
```

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health + active backends |
| `/session/start` | POST | Create a new session |
| `/session/end/{id}` | POST | End session, get summary |
| `/stream/{session_id}` | WebSocket | Real-time frame streaming |
| `/voice/instructions/{exercise}` | GET | Natural English setup instructions + optional audio URL |
| `/voice/instructions/{exercise}/audio` | GET | Generate and return playable setup audio directly |
| `/voice/audio/{filename}` | GET | Serve cached voice-coach audio |
| `/exercises` | GET | List all 15 supported exercises |
| `/metrics` | GET | Queue and latency telemetry |
| `/backends` | GET | Active inference backend info |
| `/config` | GET | Full active configuration |
| `/docs` | GET | Swagger UI |

**WebSocket frame format:**
```json
{ "frame_b64": "<base64 JPEG>", "exercise": "squat" }
```

**Voice coach:**

The API returns a natural English pre-exercise coach script when a session starts.
With `VOICE_PROVIDER=auto`, the server uses OpenAI TTS when `OPENAI_API_KEY` is
set, otherwise it falls back to Windows Speech and returns a cached WAV. Clients
can play `voice_instruction.audio_url`, or call
`/voice/instructions/squat/audio` directly for a playable response.

Recommended defaults:

```bash
VOICE_PROVIDER=auto
VOICE_LANGUAGE=en-US
VOICE_TTS_MODEL=gpt-4o-mini-tts
VOICE_TTS_VOICE=verse
VOICE_TTS_FORMAT=mp3
```

---

## TensorRT / ONNX Acceleration (Step 8)

```bash
# Export YOLOv8 to TRT FP16:
python tools/export_yolo_onnx.py --model yolov8n.pt --output models/yolo.onnx --trt

# Export RTMPose to TRT FP16:
python tools/export_rtmpose_onnx.py --output models/rtmpose.onnx --trt

# Benchmark all backends:
python tools/benchmark_inference.py --output results/inference.json

# Server auto-selects TRT → ONNX → PyTorch:
DETECTOR_BACKEND=auto POSE_BACKEND=auto uvicorn server.main_api:app
```

---

## Training the DLEngine Models

```bash
# 1. Extract keypoints from labelled videos:
python scripts/extract_keypoints.py --video-dir data/raw/ --output data/processed/

# 2. Train one exercise:
python scripts/train_dl_v4.py --exercise squat

# Or train every configured exercise:
python scripts/train_dl_v4.py

# Checkpoints are saved as:
# checkpoints_v3/dl_model_<exercise>.pt
```

---

## Benchmarking (Step 10)

```bash
# Run all benchmarks (no server needed except multi-session):
python benchmarks/bench_rep_accuracy.py --output results/accuracy.json
python benchmarks/bench_memory.py        --output results/memory.json
python benchmarks/bench_fps.py           --output results/fps.json

# Start server first, then:
python benchmarks/bench_multi_session.py --ramp --max-sessions 8 --output results/multi_session.json

# Generate unified report:
python benchmarks/report_generator.py
# → results/BENCHMARK_REPORT.md
```

---

## Configuration

All settings are configured via environment variables. See `.env.example` for full documentation.

Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `DETECTOR_BACKEND` | `auto` | `trt` / `onnx` / `pytorch` / `auto` |
| `POSE_BACKEND` | `auto` | `trt` / `onnx` / `mmpose` / `auto` |
| `DETECTION_INTERVAL` | `3` | Run YOLO every N frames |
| `QUEUE_INFERENCE_THREADS` | `4` | Concurrent inference threads |
| `SESSION_TIMEOUT_SECS` | `300` | Session inactivity timeout |
| `VOICE_PROVIDER` | `auto` | `auto`, `openai`, `windows_sapi`, or `none` |
| `VOICE_LANGUAGE` | `en-US` | Natural English coach scripts |
| `VOICE_TTS_MODEL` | `gpt-4o-mini-tts` | OpenAI TTS model for generated audio |
| `VOICE_TTS_VOICE` | `verse` | OpenAI voice used for coach audio |

---

## Supported Exercises

Squat · Push-up · Jumping Jack · High Knees · Plank · Pull-up · Sit-up · Lunge · Mountain Climber · Burpee · Bicep Curl · Tricep Dip · Lateral Raise · Shoulder Press · Wall Sit

---

## 10-Step Implementation History

| Step | Feature |
|------|---------|
| 1 | Crash fixes & defensive programming |
| 2 | Persistent IoU-based person tracker |
| 3 | Hybrid pipeline (YOLO every N frames) |
| 4 | Performance optimisation (FP16, vectorised ops) |
| 5 | Async producer-consumer inference queue |
| 6 | Rep counting stabilisation (velocity, hysteresis, debounce) |
| 7 | Kalman filter vectorisation & Joseph-form stability |
| 8 | TensorRT / ONNX export & inference backends |
| 9 | Production cleanup (config, startup diagnostics, graceful shutdown) |
| 10 | Benchmarking & stress testing suite |

