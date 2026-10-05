# ══════════════════════════════════════════════════════════════════════════════
# AI Fitness Trainer — Production Dockerfile  [v2 — Step 9 Updated]
#
# Build:
#   docker build -t fitness-ai:latest .
#
# Run (GPU):
#   docker run --gpus all -p 8000:8000 --env-file .env fitness-ai:latest
#
# Run (CPU — dev only):
#   docker run -p 8000:8000 --env-file .env fitness-ai:latest
# ══════════════════════════════════════════════════════════════════════════════

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 wget git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools>=69.5.1 wheel

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir \
    --find-links https://download.openmmlab.com/mmcv/dist/cu121/torch2.2/index.html \
    mmengine>=0.10.0 mmcv>=2.1.0 mmdet>=3.3.0 mmpose>=1.3.1

COPY . .
RUN mkdir -p model/checkpoints data/processed .pose_models models

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app LOG_LEVEL=INFO
ENV HOST=0.0.0.0 PORT=8000 WORKERS=1
ENV YOLO_MODEL=yolov8n.pt DET_CONF=0.45 DETECTION_INTERVAL=3
ENV DETECTOR_BACKEND=auto POSE_BACKEND=auto
ENV MODEL_CHECKPOINT=/app/model/checkpoints/best.pth
ENV NORMALIZER_STATS=/app/data/processed/normalizer_stats.json
ENV TRT_ENGINE_PATH=/app/models/yolo.engine
ENV TRT_POSE_PATH=/app/models/rtmpose.engine
ENV ONNX_MODEL_PATH=/app/models/yolo.onnx
ENV ONNX_POSE_PATH=/app/models/rtmpose.onnx
ENV QUEUE_INFERENCE_THREADS=4 SESSION_TIMEOUT_SECS=300

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD wget -qO- http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "server.main_api:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info"]
