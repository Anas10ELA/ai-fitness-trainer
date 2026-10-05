"""
trt_detector.py
═══════════════
Step 8 — TensorRT / ONNX Drop-in Replacement for PersonDetector

This module provides OnnxPersonDetector and TrtPersonDetector, both of which
implement the exact same public interface as detector.PersonDetector so that
frame_processor.py requires zero changes to use optimised inference.

Interface compatibility
-----------------------
  PersonDetector.detect(frame)     → List[BBox]
  PersonDetector.bbox_area(bbox)   → int
  PersonDetector.pad_bbox(...)     → BBox

Usage in main_api.py / main.py
-------------------------------
  # Was:
  from detector import PersonDetector
  detector = PersonDetector(model_name="yolov8n.pt")

  # Now (ONNX, CPU or GPU):
  from trt_detector import OnnxPersonDetector
  detector = OnnxPersonDetector(onnx_path="models/yolo.onnx")

  # Now (TensorRT FP16, GPU only):
  from trt_detector import TrtPersonDetector
  detector = TrtPersonDetector(engine_path="models/yolo.engine")

  # Auto-select (TRT if available, ONNX otherwise, PyTorch fallback):
  from trt_detector import build_detector
  detector = build_detector()

Output format
-------------
  YOLO8 ONNX output shape: (1, 84, 8400)  — (batch, xywh+80cls, anchors)
  This module decodes that tensor to List[(x1,y1,x2,y2)] filtered to person
  class (index 0) above conf_threshold, with NMS applied.

Performance
-----------
  PyTorch YOLOv8n  @640:  ~12 ms  (RTX 3080, FP16)
  ONNX   YOLOv8n  @640:  ~9  ms  (onnxruntime-gpu, CUDA EP)
  TRT    YOLOv8n  @640:  ~5  ms  (TRT FP16, RTX 3080)
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from detector import PersonDetector, BBox   # type: ignore

log = logging.getLogger(__name__)

PERSON_CLASS_ID = 0   # COCO class index for 'person'


# ═══════════════════════════════════════════════════════════════════════════════
#  Shared NMS / decode utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _decode_yolo_output(
    output: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
    orig_h: int,
    orig_w: int,
    model_size: int,
) -> List[BBox]:
    """
    Decode YOLOv8 ONNX output tensor → List[BBox].

    YOLOv8 ONNX output: (1, 84, 8400)
      - dim 1: [cx, cy, w, h, cls_0_conf, cls_1_conf, …, cls_79_conf]
      - dim 2: 8400 anchors at multiple scales

    Steps:
      1. Transpose to (8400, 84)
      2. Extract person-class confidence (index 4)
      3. Filter by conf_threshold
      4. Convert cx/cy/w/h → x1/y1/x2/y2 (scaled to orig frame)
      5. Apply NMS via cv2.dnn.NMSBoxes
    """
    # output: (1, 84, 8400) or (84, 8400)
    if output.ndim == 3:
        output = output[0]              # (84, 8400)
    predictions = output.T             # (8400, 84)

    # Column 4 is person confidence (class 0)
    person_scores = predictions[:, 4]
    mask = person_scores >= conf_threshold
    filtered = predictions[mask]       # (N, 84)

    if len(filtered) == 0:
        return []

    scores = filtered[:, 4]
    cx, cy, w, h = filtered[:, 0], filtered[:, 1], filtered[:, 2], filtered[:, 3]

    # Scale from model_size space to original frame space
    scale_x = orig_w / model_size
    scale_y = orig_h / model_size
    x1 = ((cx - w / 2) * scale_x).astype(int)
    y1 = ((cy - h / 2) * scale_y).astype(int)
    x2 = ((cx + w / 2) * scale_x).astype(int)
    y2 = ((cy + h / 2) * scale_y).astype(int)

    # Clip to frame bounds
    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w)
    y2 = np.clip(y2, 0, orig_h)

    # NMS
    boxes_cv  = [(int(x1[i]), int(y1[i]), int(x2[i]-x1[i]), int(y2[i]-y1[i])) for i in range(len(scores))]
    indices   = cv2.dnn.NMSBoxes(boxes_cv, scores.tolist(), conf_threshold, iou_threshold)
    if isinstance(indices, tuple):  # cv2 returns tuple when empty
        return []

    bboxes = [(x1[i], y1[i], x2[i], y2[i]) for i in indices.flatten()]
    bboxes.sort(key=lambda b: (b[2]-b[0]) * (b[3]-b[1]), reverse=True)
    return bboxes


def _preprocess_frame(frame: np.ndarray, model_size: int) -> np.ndarray:
    """
    Preprocess a BGR frame for YOLOv8 ONNX inference.

    Returns: float32 array (1, 3, model_size, model_size), values in [0, 1].
    Letterbox resize preserves aspect ratio with grey padding.
    """
    h, w = frame.shape[:2]
    scale = model_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized  = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((model_size, model_size, 3), 114, dtype=np.uint8)
    pad_y  = (model_size - nh) // 2
    pad_x  = (model_size - nw) // 2
    canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = resized

    # BGR → RGB, HWC → CHW, uint8 → float32 [0,1]
    rgb    = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return tensor[np.newaxis]   # (1, 3, H, W)


# ═══════════════════════════════════════════════════════════════════════════════
#  ONNX Runtime detector
# ═══════════════════════════════════════════════════════════════════════════════

class OnnxPersonDetector:
    """
    Drop-in replacement for PersonDetector using ONNX Runtime.

    Uses CUDA Execution Provider when available, falls back to CPU.
    ~30–50% faster than PyTorch eager on GPU due to lower framework overhead.
    """

    def __init__(
        self,
        onnx_path:      str   = "models/yolo.onnx",
        conf_threshold: float = 0.45,
        iou_threshold:  float = 0.45,
        input_size:     int   = 640,
        device:         Optional[str] = None,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.input_size     = input_size

        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime not installed.\n"
                "  GPU: pip install onnxruntime-gpu\n"
                "  CPU: pip install onnxruntime"
            )

        # Provider selection
        available = ort.get_available_providers()
        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
            log.warning("OnnxPersonDetector: CUDA EP not available, using CPU")

        self.device = "cuda" if "CUDAExecutionProvider" in providers else "cpu"

        log.info("OnnxPersonDetector: loading %s on %s", onnx_path, self.device)
        t0 = time.perf_counter()
        self._session = ort.InferenceSession(onnx_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        log.info(
            "OnnxPersonDetector: ready (%.1fs) input_name=%s",
            time.perf_counter() - t0, self._input_name,
        )

        # Warmup
        dummy = np.zeros((1, 3, input_size, input_size), dtype=np.float32)
        for _ in range(3):
            self._session.run(None, {self._input_name: dummy})
        log.info("OnnxPersonDetector: warmup complete")

    def detect(self, frame: np.ndarray) -> List[BBox]:
        orig_h, orig_w = frame.shape[:2]
        tensor = _preprocess_frame(frame, self.input_size)

        t0 = time.perf_counter()
        outputs = self._session.run(None, {self._input_name: tensor})
        latency_ms = (time.perf_counter() - t0) * 1000
        log.debug("OnnxDetector: %.1f ms", latency_ms)

        return _decode_yolo_output(
            outputs[0], self.conf_threshold, self.iou_threshold,
            orig_h, orig_w, self.input_size,
        )

    @staticmethod
    def bbox_area(bbox: BBox) -> int:
        return PersonDetector.bbox_area(bbox)

    @staticmethod
    def pad_bbox(bbox: BBox, frame_shape: Tuple[int, int], pad_ratio: float = 0.1) -> BBox:
        return PersonDetector.pad_bbox(bbox, frame_shape, pad_ratio)


# ═══════════════════════════════════════════════════════════════════════════════
#  TensorRT FP16 detector
# ═══════════════════════════════════════════════════════════════════════════════

class TrtPersonDetector:
    """
    Drop-in replacement for PersonDetector using TensorRT FP16.

    Requires:
      - CUDA GPU
      - tensorrt  (pip install tensorrt)
      - pycuda    (pip install pycuda)
      - A pre-built .engine file (from tools/export_yolo_onnx.py --trt)

    ~2–3× faster than PyTorch eager, ~1.5× faster than ONNX Runtime.
    """

    def __init__(
        self,
        engine_path:    str   = "models/yolo.engine",
        conf_threshold: float = 0.45,
        iou_threshold:  float = 0.45,
        input_size:     int   = 640,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.input_size     = input_size
        self.device         = "cuda"

        self._load_engine(engine_path)
        self._warmup()

    def _load_engine(self, engine_path: str) -> None:
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit   # noqa: F401
        except ImportError as e:
            raise ImportError(
                "TrtPersonDetector requires tensorrt and pycuda.\n"
                "  pip install tensorrt pycuda"
            ) from e

        import pycuda.driver as cuda
        import tensorrt as trt

        self._cuda  = cuda
        self._trt   = trt

        TRT_LOGGER  = trt.Logger(trt.Logger.ERROR)
        runtime     = trt.Runtime(TRT_LOGGER)

        log.info("TrtPersonDetector: loading engine %s", engine_path)
        with open(engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())

        self._context = self._engine.create_execution_context()
        self._stream  = cuda.Stream()

        # Pre-allocate I/O buffers
        self._bindings      = []
        self._host_inputs   = []
        self._device_inputs = []
        self._host_outputs  = []
        self._device_outputs = []
        self._output_shapes = []

        for i in range(self._engine.num_bindings):
            shape = self._engine.get_binding_shape(i)
            size  = int(np.prod(shape))
            dtype = trt.nptype(self._engine.get_binding_dtype(i))
            h_mem = cuda.pagelocked_empty(size, dtype)
            d_mem = cuda.mem_alloc(h_mem.nbytes)
            self._bindings.append(int(d_mem))
            if self._engine.binding_is_input(i):
                self._host_inputs.append(h_mem)
                self._device_inputs.append(d_mem)
            else:
                self._host_outputs.append(h_mem)
                self._device_outputs.append(d_mem)
                self._output_shapes.append(shape)

        log.info("TrtPersonDetector: engine loaded, %d bindings", self._engine.num_bindings)

    def _warmup(self, n: int = 5) -> None:
        dummy = np.zeros((1, 3, self.input_size, self.input_size), dtype=np.float32)
        for _ in range(n):
            self._infer(dummy)
        log.info("TrtPersonDetector: warmup complete (%d runs)", n)

    def _infer(self, tensor: np.ndarray) -> np.ndarray:
        """Run one inference pass. Returns raw output numpy array."""
        cuda  = self._cuda
        np.copyto(self._host_inputs[0], tensor.ravel())
        cuda.memcpy_htod_async(self._device_inputs[0], self._host_inputs[0], self._stream)
        self._context.execute_async_v2(
            bindings     = self._bindings,
            stream_handle = self._stream.handle,
        )
        cuda.memcpy_dtoh_async(self._host_outputs[0], self._device_outputs[0], self._stream)
        self._stream.synchronize()
        return self._host_outputs[0].reshape(self._output_shapes[0])

    def detect(self, frame: np.ndarray) -> List[BBox]:
        orig_h, orig_w = frame.shape[:2]
        tensor = _preprocess_frame(frame, self.input_size)

        t0 = time.perf_counter()
        output = self._infer(tensor)
        latency_ms = (time.perf_counter() - t0) * 1000
        log.debug("TrtDetector: %.1f ms", latency_ms)

        return _decode_yolo_output(
            output, self.conf_threshold, self.iou_threshold,
            orig_h, orig_w, self.input_size,
        )

    @staticmethod
    def bbox_area(bbox: BBox) -> int:
        return PersonDetector.bbox_area(bbox)

    @staticmethod
    def pad_bbox(bbox: BBox, frame_shape: Tuple[int, int], pad_ratio: float = 0.1) -> BBox:
        return PersonDetector.pad_bbox(bbox, frame_shape, pad_ratio)


# ═══════════════════════════════════════════════════════════════════════════════
#  Auto-selector factory
# ═══════════════════════════════════════════════════════════════════════════════

def build_detector(
    trt_engine:     Optional[str] = None,
    onnx_path:      Optional[str] = None,
    pt_model:       str           = "yolov8n.pt",
    conf_threshold: float         = 0.45,
    iou_threshold:  float         = 0.45,
    input_size:     int           = 640,
) -> "PersonDetector | OnnxPersonDetector | TrtPersonDetector":
    """
    Auto-select the best available detector backend:
      1. TrtPersonDetector  — if engine_path exists and tensorrt+pycuda installed
      2. OnnxPersonDetector — if onnx_path exists and onnxruntime installed
      3. PersonDetector     — PyTorch fallback (always available)

    Override detection by setting environment variables:
      DETECTOR_BACKEND = trt | onnx | pytorch
      TRT_ENGINE_PATH  = path/to/yolo.engine
      ONNX_MODEL_PATH  = path/to/yolo.onnx
    """
    backend_env = os.getenv("DETECTOR_BACKEND", "auto").lower()

    trt_path  = trt_engine  or os.getenv("TRT_ENGINE_PATH",  "models/yolo.engine")
    onnx_path = onnx_path   or os.getenv("ONNX_MODEL_PATH",  "models/yolo.onnx")

    common_kwargs = dict(
        conf_threshold = conf_threshold,
        iou_threshold  = iou_threshold,
        input_size     = input_size,
    )

    # --- TRT ---
    if backend_env in ("auto", "trt"):
        if os.path.exists(trt_path):
            try:
                d = TrtPersonDetector(engine_path=trt_path, **common_kwargs)
                log.info("build_detector: using TrtPersonDetector (%s)", trt_path)
                return d
            except Exception as e:
                log.warning("build_detector: TRT failed (%s) — trying ONNX", e)

    # --- ONNX ---
    if backend_env in ("auto", "onnx"):
        if os.path.exists(onnx_path):
            try:
                d = OnnxPersonDetector(onnx_path=onnx_path, **common_kwargs)
                log.info("build_detector: using OnnxPersonDetector (%s)", onnx_path)
                return d
            except Exception as e:
                log.warning("build_detector: ONNX failed (%s) — falling back to PyTorch", e)

    # --- PyTorch fallback ---
    log.info("build_detector: using PyTorch PersonDetector (%s)", pt_model)
    return PersonDetector(
        model_name      = pt_model,
        conf_threshold  = conf_threshold,
        iou_threshold   = iou_threshold,
        input_size      = input_size,
    )
