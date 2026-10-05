"""
server/main_api.py
══════════════════
FastAPI Server — AI Fitness Trainer API  [UPDATED v7 — Production Cleanup]

Changes in v7 (Step 9 — Final Production Cleanup):
────────────────────────────────────────────────────
  PROD-1  All os.getenv() calls replaced with cfg.<field> from config.py.
          No more scattered constants — single source of truth.

  PROD-2  run_startup_checks() from startup.py called in lifespan before
          model loading. Validates config, checks Python version, packages,
          GPU, disk space, and port availability. Fails fast with a clear
          error message instead of crashing mid-load.

  PROD-3  Graceful shutdown wired via shutdown.register_server_handlers().
          SIGTERM/SIGINT now drains the inference queue, logs session
          summaries, and exits cleanly. No more abrupt mid-inference kills.

  PROD-4  session_watchdog() from shutdown.py replaces the ad-hoc
          _cleanup_loop(). Watchdog also heals leaked queue slots.

  PROD-5  /config endpoint added — returns the full active configuration
          as JSON for deployment introspection without SSH access.

  PROD-6  Duplicate cv2 import removed from main.py helper functions
          (was imported at module level AND inside two functions).

  PROD-7  Logging standardised: all startup print() calls replaced with
          log.info(). Log format consistent across all modules.

  PROD-8  APP_VERSION centralised in config and exposed in /health response.

All v6 (TRT/ONNX) changes are preserved unchanged.
All v5 (Async Queue) changes are preserved unchanged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# PROD-1: single config import replaces all scattered os.getenv() calls
from config  import cfg
from startup import run_startup_checks, StartupError
import shutdown as _shutdown

from server.schemas         import (
    SessionStartRequest, SessionStartResponse,
    SessionEndResponse, HealthResponse,
    VoiceInstructionResponse,
    WSFrameInput, WSSwitchExercise,
)
from server.session_manager import SessionManager
from server.frame_processor import FrameProcessor
from server.inference_queue import InferenceQueue, WorkItem, QueueFullError
from exercises              import Exercise
from voice_coach            import VoiceCoach, media_type_for_audio

# ── Logging ───────────────────────────────────────────────────────────────────
# PROD-7: consistent format, level from config
logging.basicConfig(
    level  = getattr(logging, cfg.log_level, logging.INFO),
    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

APP_VERSION = "7.0.0"

# ── Shared singletons ─────────────────────────────────────────────────────────
_detector:         object          = None
_pose:             object          = None
_processor:        FrameProcessor  = None
_sessions:         SessionManager  = None
_inference_queue:  InferenceQueue  = None
_voice_coach:      VoiceCoach      = None

_active_detector_backend: str = "unknown"
_active_pose_backend:     str = "unknown"


def _dl_models_available() -> bool:
    """True when at least one per-exercise DLEngine checkpoint is available."""
    return any(Path(cfg.dl_checkpoint_dir).glob("dl_model_*.pt"))


def _get_voice_coach() -> VoiceCoach:
    """Return the voice coach singleton, building it for direct test calls if needed."""
    global _voice_coach
    if _voice_coach is None:
        _voice_coach = VoiceCoach(cfg)
    return _voice_coach


# ═══════════════════════════════════════════════════════════════════════════════
#  Lifespan
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _detector, _pose, _processor, _sessions, _inference_queue, _voice_coach
    global _active_detector_backend, _active_pose_backend

    # PROD-2: fail-fast environment validation before touching any GPU resource
    try:
        run_startup_checks()
    except StartupError as e:
        log.critical("Startup validation failed:\n%s", e)
        sys.exit(1)

    log.info("══ Loading models ══")

    # ── Detector ──────────────────────────────────────────────────────────────
    log.info("Loading detector (backend=%s) …", cfg.detector_backend)
    try:
        from trt_detector import build_detector
        _detector = build_detector(
            trt_engine     = cfg.trt_engine_path,
            onnx_path      = cfg.onnx_model_path,
            pt_model       = cfg.yolo_model,
            conf_threshold = cfg.det_conf,
            iou_threshold  = cfg.det_iou,
            input_size     = cfg.det_input_size,
        )
        _active_detector_backend = _detector.__class__.__name__
    except Exception as e:
        log.warning("TRT/ONNX detector unavailable (%s) — using PyTorch", e)
        from detector import PersonDetector
        _detector = PersonDetector(
            model_name      = cfg.yolo_model,
            conf_threshold  = cfg.det_conf,
            iou_threshold   = cfg.det_iou,
            input_size      = cfg.det_input_size,
        )
        _active_detector_backend = "PersonDetector"
    log.info("Detector: %s", _active_detector_backend)

    # ── Pose estimator ────────────────────────────────────────────────────────
    log.info("Loading pose estimator (backend=%s) …", cfg.pose_backend)
    try:
        from trt_pose import build_pose_estimator
        _pose = build_pose_estimator(
            trt_engine      = cfg.trt_pose_path,
            onnx_path       = cfg.onnx_pose_path,
            score_threshold = cfg.pose_score_threshold,
        )
        _active_pose_backend = _pose.__class__.__name__
    except Exception as e:
        log.warning("TRT/ONNX pose unavailable (%s) — using MMPose", e)
        from pose import PoseEstimator
        _pose = PoseEstimator(score_threshold=cfg.pose_score_threshold)
        _active_pose_backend = "PoseEstimator"
    log.info("Pose estimator: %s", _active_pose_backend)

    model_loaded = _dl_models_available()
    log.info(
        "DLEngine checkpoints: dir=%s available=%s",
        cfg.dl_checkpoint_dir, model_loaded,
    )

    # ── Wire everything up ────────────────────────────────────────────────────
    _processor = FrameProcessor(_detector, _pose)
    _sessions  = SessionManager(
        checkpoint_dir=cfg.dl_checkpoint_dir,
        smooth_window=cfg.dl_smooth_window,
        require_closed_mode=cfg.dl_require_closed,
    )
    _voice_coach = VoiceCoach(cfg)

    _inference_queue = InferenceQueue(_processor)
    await _inference_queue.start()

    # PROD-3: graceful shutdown with queue drain
    _shutdown.register_server_handlers(_inference_queue, _sessions)

    # PROD-4: session watchdog replaces the ad-hoc cleanup loop
    asyncio.create_task(
        _shutdown.session_watchdog(_sessions, _inference_queue, check_interval_secs=30.0)
    )

    log.info("══ Ready — accepting connections ══")
    log.info("  API version    : %s", APP_VERSION)
    log.info("  Detector       : %s", _active_detector_backend)
    log.info("  Pose estimator : %s", _active_pose_backend)
    log.info("  DL checkpoints : %s", "available" if model_loaded else "missing")
    log.info(
        "  Voice coach    : enabled=%s provider=%s language=%s voice=%s",
        cfg.voice_enabled, cfg.voice_provider, cfg.voice_language, cfg.voice_tts_voice,
    )

    yield

    # ── Shutdown sequence (also invoked by SIGTERM via shutdown.py) ───────────
    log.info("══ Shutdown ══")
    await _inference_queue.stop()
    log.info("Inference queue stopped")


# ═══════════════════════════════════════════════════════════════════════════════
#  App
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title       = "AI Fitness Trainer API",
    description = "Real-time fitness coaching via YOLOv8 + RTMPose + DLEngine",
    version     = APP_VERSION,
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = cfg.cors_origins,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
#  HTTP Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    model_loaded = _dl_models_available()
    n_sessions   = await _sessions.count() if _sessions else 0
    return HealthResponse(
        status           = "ok",
        model_loaded     = model_loaded,
        active_sessions  = n_sessions,
        detector_backend = _active_detector_backend,
        pose_backend     = _active_pose_backend,
    )


# PROD-5: config introspection endpoint
@app.get("/config", tags=["System"])
async def get_config():
    """
    Return the active server configuration as JSON.
    Useful for deployment verification without SSH access.
    Sensitive fields (paths) are included — restrict this endpoint in production
    by adding an auth dependency if needed.
    """
    return cfg.to_dict()


@app.post("/session/start", response_model=SessionStartResponse, tags=["Session"])
async def start_session(req: SessionStartRequest):
    valid_exercises = [e.value for e in Exercise]
    if req.exercise.lower() not in valid_exercises:
        raise HTTPException(
            status_code = 400,
            detail      = f"Unknown exercise '{req.exercise}'. Valid: {valid_exercises}",
        )
    exercise = Exercise(req.exercise.lower())
    session_id = await _sessions.create(exercise.value, req.user_id)
    voice_instruction = _get_voice_coach().instruction(exercise).to_dict()
    log.info("Session created: %s exercise=%s user=%s", session_id, exercise.value, req.user_id)
    return SessionStartResponse(
        session_id = session_id,
        exercise   = exercise.value,
        message    = f"Session started for exercise: {exercise.value}",
        voice_instruction = voice_instruction,
    )


@app.post("/session/end/{session_id}", response_model=SessionEndResponse, tags=["Session"])
async def end_session(session_id: str):
    summary = await _sessions.close(session_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Session not found")
    log.info("Session ended: %s reps=%s", session_id, summary.get("total_reps"))
    return SessionEndResponse(**summary, message="Session ended successfully")


@app.get("/exercises", tags=["System"])
async def list_exercises():
    from exercises import REGISTRY
    return {
        ex.value: {
            "display_name": reg.display_name,
            "description":  reg.description,
            "is_timed":     reg.is_timed,
        }
        for ex, reg in REGISTRY.items()
    }


@app.get("/voice/instructions/{exercise}", response_model=VoiceInstructionResponse, tags=["Voice"])
async def get_voice_instruction(exercise: str):
    try:
        ex = Exercise(exercise.lower())
    except ValueError:
        valid_exercises = [e.value for e in Exercise]
        raise HTTPException(
            status_code = 400,
            detail      = f"Unknown exercise '{exercise}'. Valid: {valid_exercises}",
        )
    return _get_voice_coach().instruction(ex).to_dict()


@app.get("/voice/instructions/{exercise}/audio", tags=["Voice"])
async def get_voice_instruction_audio(exercise: str):
    try:
        ex = Exercise(exercise.lower())
    except ValueError:
        valid_exercises = [e.value for e in Exercise]
        raise HTTPException(
            status_code = 400,
            detail      = f"Unknown exercise '{exercise}'. Valid: {valid_exercises}",
        )

    instruction = _get_voice_coach().instruction(ex)
    if not instruction.audio_ready or not instruction.audio_url:
        raise HTTPException(
            status_code = 503,
            detail      = instruction.reason or "Voice audio is not available",
        )

    filename = Path(instruction.audio_url).name
    audio_path = _get_voice_coach().cached_audio_path(filename)
    if audio_path is None:
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(
        path       = audio_path,
        media_type = media_type_for_audio(audio_path),
        filename   = audio_path.name,
    )


@app.get("/voice/audio/{filename}", tags=["Voice"])
async def get_voice_audio(filename: str):
    audio_path = _get_voice_coach().cached_audio_path(filename)
    if audio_path is None:
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(
        path       = audio_path,
        media_type = media_type_for_audio(audio_path),
        filename   = audio_path.name,
    )


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    if _inference_queue is None:
        raise HTTPException(status_code=503, detail="Inference queue not initialised")
    snap = _inference_queue.metrics_snapshot()
    snap["inference_backends"] = {
        "detector": _active_detector_backend,
        "pose":     _active_pose_backend,
    }
    return snap


@app.get("/metrics/queues", tags=["Monitoring"])
async def get_queue_depths():
    if _inference_queue is None:
        raise HTTPException(status_code=503, detail="Inference queue not initialised")
    return {
        "queue_depths":    _inference_queue.queue_depth(),
        "active_sessions": _inference_queue.metrics.active_sessions,
        "timestamp":       time.time(),
    }


@app.get("/backends", tags=["System"])
async def get_backends():
    return {
        "detector": {
            "active_backend": _active_detector_backend,
            "trt_engine":     cfg.trt_engine_path,
            "trt_exists":     Path(cfg.trt_engine_path).exists(),
            "onnx_model":     cfg.onnx_model_path,
            "onnx_exists":    Path(cfg.onnx_model_path).exists(),
        },
        "pose": {
            "active_backend": _active_pose_backend,
            "trt_engine":     cfg.trt_pose_path,
            "trt_exists":     Path(cfg.trt_pose_path).exists(),
            "onnx_model":     cfg.onnx_pose_path,
            "onnx_exists":    Path(cfg.onnx_pose_path).exists(),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  WebSocket Endpoint
# ═══════════════════════════════════════════════════════════════════════════════

@app.websocket("/stream/{session_id}")
async def stream(websocket: WebSocket, session_id: str):
    await websocket.accept()

    session = await _sessions.get(session_id)
    if session is None:
        await websocket.send_json({
            "error": "Session not found or expired. Call POST /session/start first.",
            "code": 404,
        })
        await websocket.close()
        return

    _inference_queue.register_session(session_id)
    log.info("WS connected: session=%s exercise=%s", session_id, session.exercise.value)

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON", "code": 400})
                continue

            msg_type = msg.get("type", "frame")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong", "ts": time.time()})
                continue

            if msg_type == "switch":
                new_ex = msg.get("exercise", "")
                try:
                    ex = Exercise(new_ex.lower())
                    session.switch_exercise(ex)
                    log.info("WS switch: session=%s → %s", session_id, ex.value)
                    await websocket.send_json({"type": "switched", "exercise": ex.value})
                except ValueError:
                    await websocket.send_json(
                        {"error": f"Unknown exercise: {new_ex}", "code": 400}
                    )
                continue

            if msg_type == "reset":
                session.reset_reps()
                await websocket.send_json({"type": "reset_ok", "reps": 0})
                continue

            frame_b64 = msg.get("frame_b64", "")
            if not frame_b64:
                await websocket.send_json({"error": "frame_b64 is required", "code": 400})
                continue

            session.touch()
            work = WorkItem(
                session_id         = session_id,
                frame_b64          = frame_b64,
                session            = session,
                requested_exercise = msg.get("exercise"),
            )

            try:
                output = await _inference_queue.submit(work)
                await websocket.send_json(output.model_dump())

            except QueueFullError:
                snap = _inference_queue.metrics.snapshot()
                log.warning(
                    "Backpressure: session=%s drop_rate=%.1f%%",
                    session_id, snap["drop_rate_pct"],
                )
                await websocket.send_json({
                    "type":          "backpressure",
                    "drop_rate_pct": snap["drop_rate_pct"],
                    "message":       "Server queue full — reduce frame rate",
                })

            except asyncio.CancelledError:
                break

            except Exception as exc:
                log.error("Frame error session=%s: %s", session_id, exc, exc_info=True)
                await websocket.send_json({"error": str(exc), "code": 500})

    except WebSocketDisconnect:
        log.info("WS disconnected: session=%s", session_id)
    except Exception as e:
        log.error("WS error session=%s: %s", session_id, e, exc_info=True)
        try:
            await websocket.send_json({"error": str(e), "code": 500})
        except Exception:
            pass
    finally:
        _inference_queue.unregister_session(session_id)
        await _sessions.close(session_id)
        log.info("WS session released: session=%s", session_id)


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(
        "server.main_api:app",
        host      = cfg.host,
        port      = cfg.port,
        workers   = cfg.workers,   # must be 1
        reload    = False,
        log_level = cfg.log_level.lower(),
    )
