"""
startup.py
══════════
Step 9 — Startup Diagnostics & Environment Validation

Runs a comprehensive pre-flight check before the server accepts connections.
Call run_startup_checks() in the lifespan context manager before loading models.

Checks performed
----------------
  1. Python version (>= 3.10)
  2. Required packages installed and importable
  3. Config validation (delegates to cfg.validate())
  4. CUDA / GPU availability and VRAM
  5. Model file existence (checkpoint, stats, TRT/ONNX engines if configured)
  6. Write permissions on log/output directories
  7. Disk space (warn if < 2 GB free on model directory)
  8. Port availability (warn if configured port is already in use)

On any CRITICAL failure, raises StartupError with a clear diagnostic message.
Non-fatal issues are logged as WARNING so deployment proceeds with degraded mode.

Usage
-----
    from startup import run_startup_checks, StartupError
    try:
        run_startup_checks()
    except StartupError as e:
        log.critical("Startup failed: %s", e)
        sys.exit(1)
"""

from __future__ import annotations

import importlib
import logging
import os
import shutil
import socket
import sys
from pathlib import Path
from typing import List, Tuple

log = logging.getLogger(__name__)


class StartupError(RuntimeError):
    """Raised when a critical pre-flight check fails."""


# ── Required packages ─────────────────────────────────────────────────────────

_REQUIRED_PACKAGES: List[Tuple[str, str]] = [
    # (import_name, pip_name)
    ("numpy",        "numpy"),
    ("cv2",          "opencv-python"),
    ("torch",        "torch"),
    ("fastapi",      "fastapi"),
    ("uvicorn",      "uvicorn[standard]"),
    ("pydantic",     "pydantic"),
    ("ultralytics",  "ultralytics"),
]

_OPTIONAL_PACKAGES: List[Tuple[str, str, str]] = [
    # (import_name, pip_name, feature_description)
    ("mmpose",        "mmpose",          "RTMPose pose estimation"),
    ("onnxruntime",   "onnxruntime-gpu", "ONNX Runtime inference"),
    ("tensorrt",      "tensorrt",        "TensorRT FP16 inference"),
    ("pycuda",        "pycuda",          "TensorRT CUDA buffers"),
]

# Minimum Python version
_MIN_PYTHON = (3, 10)

# Minimum free disk space in bytes before warning
_MIN_FREE_DISK_BYTES = 2 * 1024 ** 3  # 2 GB

# Minimum CUDA VRAM in bytes before warning
_MIN_VRAM_BYTES = 3 * 1024 ** 3  # 3 GB


# ═══════════════════════════════════════════════════════════════════════════════

def run_startup_checks() -> None:
    """
    Run all pre-flight checks. Raises StartupError on critical failure.
    Logs warnings for non-critical issues and continues.
    """
    from config import cfg

    log.info("═" * 56)
    log.info("  AI Fitness Trainer — Startup Diagnostics")
    log.info("═" * 56)

    errors:   List[str] = []
    warnings: List[str] = []

    _check_python(errors)
    _check_packages(errors, warnings)
    _check_config(errors, cfg)
    _check_gpu(warnings)
    _check_model_files(warnings, cfg)
    _check_disk_space(warnings, cfg)
    _check_port(warnings, cfg)
    _check_write_permissions(warnings)

    # Report
    for w in warnings:
        log.warning("  ⚠  %s", w)

    if errors:
        log.critical("  ✗  %d critical error(s):", len(errors))
        for e in errors:
            log.critical("      • %s", e)
        raise StartupError(
            "Startup failed with critical errors:\n" +
            "\n".join(f"  • {e}" for e in errors)
        )

    log.info("  ✓  All startup checks passed (%d warning(s))", len(warnings))
    log.info("═" * 56)

    # Log full config summary
    cfg.log_summary()


# ── Individual checks ─────────────────────────────────────────────────────────

def _check_python(errors: List[str]) -> None:
    v = sys.version_info[:2]
    if v < _MIN_PYTHON:
        errors.append(
            f"Python {v[0]}.{v[1]} is too old. "
            f"Requires Python {_MIN_PYTHON[0]}.{_MIN_PYTHON[1]}+."
        )
        return
    log.info("  ✓  Python %d.%d", v[0], v[1])


def _check_packages(errors: List[str], warnings: List[str]) -> None:
    missing_required = []
    for import_name, pip_name in _REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "?")
            log.info("  ✓  %-20s  %s", import_name, ver)
        except ImportError:
            missing_required.append(pip_name)
            log.error("  ✗  %-20s  MISSING", import_name)

    if missing_required:
        errors.append(
            "Missing required packages. Install with:\n"
            f"  pip install {' '.join(missing_required)}"
        )

    for import_name, pip_name, feature in _OPTIONAL_PACKAGES:
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "?")
            log.info("  ✓  %-20s  %s  (optional: %s)", import_name, ver, feature)
        except ImportError:
            warnings.append(
                f"{import_name} not installed — {feature} unavailable. "
                f"pip install {pip_name}"
            )
            log.info("  -  %-20s  not installed (optional: %s)", import_name, feature)


def _check_config(errors: List[str], cfg) -> None:
    try:
        cfg.validate()
        log.info("  ✓  Configuration valid")
    except Exception as e:
        errors.append(str(e))


def _check_gpu(warnings: List[str]) -> None:
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory
            vram_gb = vram / 1024 ** 3
            log.info("  ✓  GPU: %s  VRAM: %.1f GB", device_name, vram_gb)
            if vram < _MIN_VRAM_BYTES:
                warnings.append(
                    f"GPU VRAM is low ({vram_gb:.1f} GB). "
                    "Recommend >= 3 GB for real-time inference. "
                    "Consider using ONNX backend with smaller models."
                )
        else:
            warnings.append(
                "No CUDA GPU detected — running on CPU. "
                "Real-time inference may be too slow for production use. "
                "Consider a CUDA-capable GPU or cloud GPU instance."
            )
            log.info("  -  No CUDA GPU — CPU inference only")
    except ImportError:
        warnings.append("torch not installed — cannot check GPU")


def _check_model_files(warnings: List[str], cfg) -> None:
    # DLEngine per-exercise checkpoints (optional per exercise).
    dl_dir = Path(cfg.dl_checkpoint_dir)
    dl_models = sorted(dl_dir.glob("dl_model_*.pt")) if dl_dir.exists() else []
    if not dl_models:
        warnings.append(
            f"No DLEngine checkpoints found in {cfg.dl_checkpoint_dir}. "
            "DL state prediction and rep counting will be disabled."
        )
        log.info("  -  DLEngine checkpoints: %s  (none found)", cfg.dl_checkpoint_dir)
    else:
        total_mb = sum(p.stat().st_size for p in dl_models) / 1024 ** 2
        log.info(
            "  ✓  DLEngine checkpoints: %s  (%d files, %.1f MB)",
            cfg.dl_checkpoint_dir, len(dl_models), total_mb,
        )

    # TRT / ONNX engines (optional — server falls back to PyTorch)
    for label, path in [
        ("TRT YOLO engine",    cfg.trt_engine_path),
        ("TRT Pose engine",    cfg.trt_pose_path),
        ("ONNX YOLO model",    cfg.onnx_model_path),
        ("ONNX Pose model",    cfg.onnx_pose_path),
    ]:
        if Path(path).exists():
            size_mb = Path(path).stat().st_size / 1024 ** 2
            log.info("  ✓  %-22s %s  (%.1f MB)", label + ":", path, size_mb)
        else:
            log.info("  -  %-22s %s  (not found — PyTorch fallback)", label + ":", path)


def _check_disk_space(warnings: List[str], cfg) -> None:
    try:
        # Check disk space near the active DLEngine checkpoint directory.
        check_path = Path(cfg.dl_checkpoint_dir)
        if not check_path.exists():
            check_path = Path(".")
        usage = shutil.disk_usage(check_path)
        free_gb = usage.free / 1024 ** 3
        if usage.free < _MIN_FREE_DISK_BYTES:
            warnings.append(
                f"Low disk space: {free_gb:.1f} GB free on {check_path}. "
                "Recommend at least 2 GB free for model files and logs."
            )
        else:
            log.info("  ✓  Disk space: %.1f GB free on %s", free_gb, check_path)
    except Exception as e:
        log.debug("Disk space check failed: %s", e)


def _check_port(warnings: List[str], cfg) -> None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            result = s.connect_ex(("127.0.0.1", cfg.port))
            if result == 0:
                warnings.append(
                    f"Port {cfg.port} appears to already be in use. "
                    "The server may fail to bind. "
                    f"Set PORT=<other> in your environment or kill the process using port {cfg.port}."
                )
            else:
                log.info("  ✓  Port %d is available", cfg.port)
    except Exception:
        pass  # Non-critical


def _check_write_permissions(warnings: List[str]) -> None:
    check_dirs = [".", "logs", "models"]
    for d in check_dirs:
        p = Path(d)
        if p.exists() and not os.access(p, os.W_OK):
            warnings.append(f"No write permission on directory: {p.resolve()}")
        elif p.exists():
            log.info("  ✓  Write access: %s", p.resolve())
