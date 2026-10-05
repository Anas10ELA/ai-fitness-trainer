"""
config.py
═════════
Step 9 — Centralised Configuration

Single source of truth for every tuneable constant in the system.
All values are loaded from environment variables with validated defaults.
Import this module everywhere instead of scattering os.getenv() calls.

Usage
-----
    from config import cfg

    # Access any value:
    cfg.yolo_model          # "yolov8n.pt"
    cfg.det_conf            # 0.45
    cfg.queue_batch_size    # 8

    # Validate on startup (raises ConfigError with a clear message on failure):
    cfg.validate()

    # Dump current config to log:
    cfg.log_summary()

Environment variables
---------------------
See each field's docstring below.  Every field maps 1:1 to an env var.
The env var name is the field name uppercased (e.g. DET_CONF, YOLO_MODEL).

Adding a new setting
--------------------
1. Add a field here with its default and type annotation.
2. Add the os.getenv() call in _load().
3. Add a validation rule in validate() if needed.
4. Use cfg.<field> everywhere instead of a hardcoded constant.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)


class ConfigError(ValueError):
    """Raised by cfg.validate() when a configuration constraint is violated."""


@dataclass
class AppConfig:
    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    """LOG_LEVEL — Python logging level: DEBUG, INFO, WARNING, ERROR."""

    # ── Model paths ───────────────────────────────────────────────────────────
    yolo_model: str = "yolov8n.pt"
    """YOLO_MODEL — Ultralytics model name or path, e.g. yolov8n.pt / yolov8s.pt."""

    checkpoint_path: str = "model/checkpoints/best.pth"
    """MODEL_CHECKPOINT — Legacy ST-GCN/BiLSTM checkpoint path."""

    normalizer_stats: str = "data/processed/normalizer_stats.json"
    """NORMALIZER_STATS — Legacy keypoint normalisation stats JSON path."""

    dl_checkpoint_dir: str = "checkpoints_v3"
    """DL_CHECKPOINT_DIR — Directory containing dl_model_<exercise>.pt files."""

    dl_smooth_window: int = 5
    """DL_SMOOTH_WINDOW — Mode-vote smoother window for DLEngine state output."""

    dl_require_closed: str = "auto"
    """DL_REQUIRE_CLOSED — auto | on | off rep peak-gating policy override."""

    # ── Voice coach ───────────────────────────────────────────────────────────
    voice_enabled: bool = True
    """VOICE_ENABLED — Include pre-exercise voice-coach instructions in API responses."""

    voice_provider: str = "auto"
    """VOICE_PROVIDER — auto | openai | windows_sapi | none."""

    voice_language: str = "en-US"
    """VOICE_LANGUAGE — Natural-language cue locale. Default is natural English."""

    voice_cache_dir: str = "voice_cache"
    """VOICE_CACHE_DIR — Directory where generated coach audio files are cached."""

    voice_tts_model: str = "gpt-4o-mini-tts"
    """VOICE_TTS_MODEL — OpenAI text-to-speech model for coach audio."""

    voice_tts_voice: str = "verse"
    """VOICE_TTS_VOICE — OpenAI voice name for coach audio."""

    voice_tts_format: str = "mp3"
    """VOICE_TTS_FORMAT — mp3 | opus | aac | flac | wav | pcm."""

    voice_tts_speed: float = 1.0
    """VOICE_TTS_SPEED — Playback speed multiplier, 0.25 to 4.0."""

    voice_tts_style: str = "Natural English fitness coach: calm, clear, concise, and encouraging."
    """VOICE_TTS_STYLE — Speaking style instructions for capable TTS models."""

    # ── TRT / ONNX backend paths ──────────────────────────────────────────────
    detector_backend: str = "auto"
    """DETECTOR_BACKEND — trt | onnx | pytorch | auto."""

    pose_backend: str = "auto"
    """POSE_BACKEND — trt | onnx | mmpose | auto."""

    trt_engine_path: str = "models/yolo.engine"
    """TRT_ENGINE_PATH — Path to TensorRT YOLOv8 FP16 engine file."""

    trt_pose_path: str = "models/rtmpose.engine"
    """TRT_POSE_PATH — Path to TensorRT RTMPose FP16 engine file."""

    onnx_model_path: str = "models/yolo.onnx"
    """ONNX_MODEL_PATH — Path to YOLOv8 ONNX model."""

    onnx_pose_path: str = "models/rtmpose.onnx"
    """ONNX_POSE_PATH — Path to RTMPose ONNX model."""

    # ── Detection / tracking ──────────────────────────────────────────────────
    det_conf: float = 0.45
    """DET_CONF — YOLOv8 detection confidence threshold."""

    det_iou: float = 0.45
    """DET_IOU — YOLOv8 NMS IoU threshold."""

    det_input_size: int = 640
    """DET_INPUT_SIZE — YOLOv8 inference image size (square)."""

    detection_interval: int = 3
    """DETECTION_INTERVAL — Run YOLO every N frames; tracker coasts in between."""

    tracker_iou_threshold: float = 0.35
    """TRACKER_IOU_THRESHOLD — Minimum IoU to maintain tracking identity."""

    tracker_max_lost: int = 10
    """TRACKER_MAX_LOST — Frames before tracker enters re-acquisition."""

    # ── Pose estimation ───────────────────────────────────────────────────────
    pose_score_threshold: float = 0.30
    """POSE_SCORE_THRESHOLD — Minimum keypoint confidence to accept."""

    # ── Inference engine ──────────────────────────────────────────────────────
    model_window: int = 32
    """MODEL_WINDOW — Legacy temporal model sequence length (frames)."""

    form_threshold: float = 0.60
    """FORM_THRESHOLD — Minimum form score to consider form good."""

    model_conf_threshold: float = 0.50
    """MODEL_CONF_THRESHOLD — Minimum exercise classification confidence."""

    model_kp_conf_threshold: float = 0.25
    """MODEL_KP_CONF_THRESHOLD — Keypoint confidence cutoff for normalisation."""

    # ── Async inference queue ─────────────────────────────────────────────────
    queue_max_per_session: int = 4
    """QUEUE_MAX_PER_SESSION — Per-session frame queue depth before backpressure."""

    queue_batch_size: int = 8
    """QUEUE_BATCH_SIZE — Maximum frames dispatched per batch."""

    queue_inference_threads: int = 4
    """QUEUE_INFERENCE_THREADS — ThreadPoolExecutor size."""

    queue_dispatch_interval: float = 0.005
    """QUEUE_DISPATCH_INTERVAL — Dispatcher poll interval in seconds."""

    # ── Session management ────────────────────────────────────────────────────
    session_timeout_secs: int = 300
    """SESSION_TIMEOUT_SECS — Inactive session expiry time."""

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    """HOST — FastAPI / uvicorn bind address."""

    port: int = 8000
    """PORT — FastAPI / uvicorn bind port."""

    workers: int = 1
    """WORKERS — uvicorn worker count. Must be 1 (asyncio queue is in-process)."""

    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    """CORS_ORIGINS — Comma-separated allowed origins, or * for all."""

    # ── Video source (desktop app only) ───────────────────────────────────────
    video_source: str = "0"
    """VIDEO_SOURCE — Camera index (0,1,…) or video file path."""

    show_tracking: bool = False
    """SHOW_TRACKING — Render tracker debug overlay in desktop app."""

    no_display: bool = False
    """NO_DISPLAY — Run desktop app headless (no cv2.imshow)."""

    save_video: str = ""
    """SAVE_VIDEO — If set, save annotated output to this path."""

    # ── Internal ──────────────────────────────────────────────────────────────
    _loaded: bool = field(default=False, repr=False, compare=False)

    # ─────────────────────────────────────────────────────────────────────────

    def _load(self) -> "AppConfig":
        """Read every value from environment variables, with typed coercion."""

        def _str(key: str, default: str) -> str:
            return os.getenv(key, default).strip()

        def _float(key: str, default: float) -> float:
            try:
                return float(os.getenv(key, str(default)))
            except ValueError:
                log.warning("Config: invalid float for %s — using default %s", key, default)
                return default

        def _int(key: str, default: int) -> int:
            try:
                return int(os.getenv(key, str(default)))
            except ValueError:
                log.warning("Config: invalid int for %s — using default %s", key, default)
                return default

        def _bool(key: str, default: bool) -> bool:
            val = os.getenv(key, str(default)).strip().lower()
            return val in ("1", "true", "yes", "on")

        def _strlist(key: str, default: List[str]) -> List[str]:
            raw = os.getenv(key, "")
            return [x.strip() for x in raw.split(",") if x.strip()] if raw else default

        self.log_level               = _str("LOG_LEVEL",                   self.log_level).upper()
        self.yolo_model              = _str("YOLO_MODEL",                   self.yolo_model)
        self.checkpoint_path         = _str("MODEL_CHECKPOINT",             self.checkpoint_path)
        self.normalizer_stats        = _str("NORMALIZER_STATS",             self.normalizer_stats)
        self.dl_checkpoint_dir       = _str("DL_CHECKPOINT_DIR",            self.dl_checkpoint_dir)
        self.dl_smooth_window        = _int("DL_SMOOTH_WINDOW",             self.dl_smooth_window)
        self.dl_require_closed       = _str("DL_REQUIRE_CLOSED",            self.dl_require_closed)
        self.voice_enabled           = _bool("VOICE_ENABLED",               self.voice_enabled)
        self.voice_provider          = _str("VOICE_PROVIDER",               self.voice_provider).lower()
        self.voice_language          = _str("VOICE_LANGUAGE",               self.voice_language)
        self.voice_cache_dir         = _str("VOICE_CACHE_DIR",              self.voice_cache_dir)
        self.voice_tts_model         = _str("VOICE_TTS_MODEL",              self.voice_tts_model)
        self.voice_tts_voice         = _str("VOICE_TTS_VOICE",              self.voice_tts_voice)
        self.voice_tts_format        = _str("VOICE_TTS_FORMAT",             self.voice_tts_format).lower()
        self.voice_tts_speed         = _float("VOICE_TTS_SPEED",            self.voice_tts_speed)
        self.voice_tts_style         = _str("VOICE_TTS_STYLE",              self.voice_tts_style)
        self.detector_backend        = _str("DETECTOR_BACKEND",             self.detector_backend)
        self.pose_backend            = _str("POSE_BACKEND",                 self.pose_backend)
        self.trt_engine_path         = _str("TRT_ENGINE_PATH",              self.trt_engine_path)
        self.trt_pose_path           = _str("TRT_POSE_PATH",                self.trt_pose_path)
        self.onnx_model_path         = _str("ONNX_MODEL_PATH",              self.onnx_model_path)
        self.onnx_pose_path          = _str("ONNX_POSE_PATH",               self.onnx_pose_path)
        self.det_conf                = _float("DET_CONF",                   self.det_conf)
        self.det_iou                 = _float("DET_IOU",                    self.det_iou)
        self.det_input_size          = _int("DET_INPUT_SIZE",               self.det_input_size)
        self.detection_interval      = _int("DETECTION_INTERVAL",           self.detection_interval)
        self.tracker_iou_threshold   = _float("TRACKER_IOU_THRESHOLD",      self.tracker_iou_threshold)
        self.tracker_max_lost        = _int("TRACKER_MAX_LOST",             self.tracker_max_lost)
        self.pose_score_threshold    = _float("POSE_SCORE_THRESHOLD",       self.pose_score_threshold)
        self.model_window            = _int("MODEL_WINDOW",                 self.model_window)
        self.form_threshold          = _float("FORM_THRESHOLD",             self.form_threshold)
        self.model_conf_threshold    = _float("MODEL_CONF_THRESHOLD",       self.model_conf_threshold)
        self.model_kp_conf_threshold = _float("MODEL_KP_CONF_THRESHOLD",    self.model_kp_conf_threshold)
        self.queue_max_per_session   = _int("QUEUE_MAX_PER_SESSION",        self.queue_max_per_session)
        self.queue_batch_size        = _int("QUEUE_BATCH_SIZE",             self.queue_batch_size)
        self.queue_inference_threads = _int("QUEUE_INFERENCE_THREADS",      self.queue_inference_threads)
        self.queue_dispatch_interval = _float("QUEUE_DISPATCH_INTERVAL",    self.queue_dispatch_interval)
        self.session_timeout_secs    = _int("SESSION_TIMEOUT_SECS",         self.session_timeout_secs)
        self.host                    = _str("HOST",                         self.host)
        self.port                    = _int("PORT",                         self.port)
        self.workers                 = _int("WORKERS",                      self.workers)
        self.cors_origins            = _strlist("CORS_ORIGINS",             self.cors_origins)
        self.video_source            = _str("VIDEO_SOURCE",                 self.video_source)
        self.show_tracking           = _bool("SHOW_TRACKING",               self.show_tracking)
        self.no_display              = _bool("NO_DISPLAY",                  self.no_display)
        self.save_video              = _str("SAVE_VIDEO",                   self.save_video)
        self._loaded = True
        return self

    def validate(self) -> None:
        """
        Check all constraints that would cause silent misbehaviour or crashes.
        Raises ConfigError with a descriptive message on the first violation.
        """
        errors: List[str] = []

        # Type / range checks
        if not (0.0 < self.det_conf < 1.0):
            errors.append(f"DET_CONF={self.det_conf} must be in (0, 1)")
        if not (0.0 < self.det_iou < 1.0):
            errors.append(f"DET_IOU={self.det_iou} must be in (0, 1)")
        if self.det_input_size not in (320, 416, 512, 640, 1280):
            errors.append(f"DET_INPUT_SIZE={self.det_input_size} must be one of 320/416/512/640/1280")
        if self.detection_interval < 1:
            errors.append(f"DETECTION_INTERVAL={self.detection_interval} must be >= 1")
        if not (0.0 < self.pose_score_threshold < 1.0):
            errors.append(f"POSE_SCORE_THRESHOLD={self.pose_score_threshold} must be in (0, 1)")
        if self.model_window < 4:
            errors.append(f"MODEL_WINDOW={self.model_window} must be >= 4")
        if self.dl_smooth_window < 1:
            errors.append(f"DL_SMOOTH_WINDOW={self.dl_smooth_window} must be >= 1")
        if self.dl_require_closed not in ("auto", "on", "off"):
            errors.append(f"DL_REQUIRE_CLOSED={self.dl_require_closed} must be auto/on/off")
        if self.voice_provider not in ("auto", "openai", "windows_sapi", "none"):
            errors.append(f"VOICE_PROVIDER={self.voice_provider} must be auto/openai/windows_sapi/none")
        if not self.voice_cache_dir:
            errors.append("VOICE_CACHE_DIR must not be empty")
        if self.voice_tts_format not in ("mp3", "opus", "aac", "flac", "wav", "pcm"):
            errors.append(f"VOICE_TTS_FORMAT={self.voice_tts_format} must be mp3/opus/aac/flac/wav/pcm")
        if not (0.25 <= self.voice_tts_speed <= 4.0):
            errors.append(f"VOICE_TTS_SPEED={self.voice_tts_speed} must be in [0.25, 4.0]")
        if not (0.0 < self.form_threshold < 1.0):
            errors.append(f"FORM_THRESHOLD={self.form_threshold} must be in (0, 1)")
        if self.queue_max_per_session < 1:
            errors.append(f"QUEUE_MAX_PER_SESSION={self.queue_max_per_session} must be >= 1")
        if self.queue_batch_size < 1:
            errors.append(f"QUEUE_BATCH_SIZE={self.queue_batch_size} must be >= 1")
        if self.queue_inference_threads < 1:
            errors.append(f"QUEUE_INFERENCE_THREADS={self.queue_inference_threads} must be >= 1")
        if self.queue_dispatch_interval <= 0:
            errors.append(f"QUEUE_DISPATCH_INTERVAL={self.queue_dispatch_interval} must be > 0")
        if self.session_timeout_secs < 10:
            errors.append(f"SESSION_TIMEOUT_SECS={self.session_timeout_secs} must be >= 10")
        if self.workers != 1:
            errors.append(
                f"WORKERS={self.workers}: must be 1. The async inference queue is "
                "in-process and cannot be shared across uvicorn workers. "
                "For horizontal scaling, run multiple processes behind a load balancer."
            )
        if self.port < 1 or self.port > 65535:
            errors.append(f"PORT={self.port} must be in [1, 65535]")
        if self.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            errors.append(f"LOG_LEVEL={self.log_level} must be one of DEBUG/INFO/WARNING/ERROR/CRITICAL")
        if self.detector_backend not in ("auto", "trt", "onnx", "pytorch"):
            errors.append(f"DETECTOR_BACKEND={self.detector_backend} must be auto/trt/onnx/pytorch")
        if self.pose_backend not in ("auto", "trt", "onnx", "mmpose"):
            errors.append(f"POSE_BACKEND={self.pose_backend} must be auto/trt/onnx/mmpose")

        if errors:
            msg = "Configuration errors:\n" + "\n".join(f"  • {e}" for e in errors)
            raise ConfigError(msg)

    def log_summary(self) -> None:
        """Emit all config values at INFO level. Call after validate()."""
        log.info("── Configuration ────────────────────────────────────")
        log.info("  log_level              = %s", self.log_level)
        log.info("  yolo_model             = %s", self.yolo_model)
        log.info("  checkpoint_path        = %s", self.checkpoint_path)
        log.info("  dl_checkpoint_dir      = %s", self.dl_checkpoint_dir)
        log.info("  dl_smooth_window       = %d", self.dl_smooth_window)
        log.info("  dl_require_closed      = %s", self.dl_require_closed)
        log.info("  voice_enabled          = %s", self.voice_enabled)
        log.info("  voice_provider         = %s", self.voice_provider)
        log.info("  voice_language         = %s", self.voice_language)
        log.info("  voice_cache_dir        = %s", self.voice_cache_dir)
        log.info("  voice_tts_model        = %s", self.voice_tts_model)
        log.info("  voice_tts_voice        = %s", self.voice_tts_voice)
        log.info("  detector_backend       = %s", self.detector_backend)
        log.info("  pose_backend           = %s", self.pose_backend)
        log.info("  det_conf               = %.2f", self.det_conf)
        log.info("  det_input_size         = %d", self.det_input_size)
        log.info("  detection_interval     = %d", self.detection_interval)
        log.info("  tracker_iou_threshold  = %.2f", self.tracker_iou_threshold)
        log.info("  tracker_max_lost       = %d", self.tracker_max_lost)
        log.info("  model_window           = %d", self.model_window)
        log.info("  form_threshold         = %.2f", self.form_threshold)
        log.info("  queue_max_per_session  = %d", self.queue_max_per_session)
        log.info("  queue_batch_size       = %d", self.queue_batch_size)
        log.info("  queue_inference_threads= %d", self.queue_inference_threads)
        log.info("  session_timeout_secs   = %d", self.session_timeout_secs)
        log.info("  host:port              = %s:%d", self.host, self.port)
        log.info("  trt_engine_path        = %s (exists=%s)", self.trt_engine_path, Path(self.trt_engine_path).exists())
        log.info("  onnx_model_path        = %s (exists=%s)", self.onnx_model_path, Path(self.onnx_model_path).exists())
        log.info("─────────────────────────────────────────────────────")

    def to_dict(self) -> dict:
        """Return all public config values as a plain dict (for /config endpoint)."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }


# ── Module-level singleton ────────────────────────────────────────────────────

cfg = AppConfig()._load()
