"""
trt_pose.py
═══════════
Step 8 — TensorRT / ONNX Drop-in Replacement for PoseEstimator

Provides OnnxPoseEstimator and TrtPoseEstimator, both implementing the same
public interface as pose.PoseEstimator so that frame_processor.py requires
zero changes.

Interface compatibility
-----------------------
  PoseEstimator.estimate(frame, bbox)               → Optional[Keypoints]
  PoseEstimator.estimate_with_scores(frame, bbox)   → Optional[tuple[Keypoints, scores]]

Usage
-----
  # Was:
  from pose import PoseEstimator
  pose = PoseEstimator()

  # Now (ONNX):
  from trt_pose import OnnxPoseEstimator
  pose = OnnxPoseEstimator(onnx_path="models/rtmpose.onnx")

  # Now (TRT FP16):
  from trt_pose import TrtPoseEstimator
  pose = TrtPoseEstimator(engine_path="models/rtmpose.engine")

  # Auto-select:
  from trt_pose import build_pose_estimator
  pose = build_pose_estimator()

RTMPose ONNX output
-------------------
  keypoints : (1, 17, 2)   — (x, y) in model-input-space coordinates
  scores    : (1, 17)      — per-keypoint confidence in [0, 1]

Preprocessing
-------------
  RTMPose expects a person crop resized to (input_h × input_w) and
  normalised with ImageNet mean/std. We replicate the MMPose pipeline:
    1. Crop + pad the detected person bbox
    2. Resize to (input_h, input_w)
    3. BGR → RGB, divide by 255
    4. Subtract mean=[0.485, 0.456, 0.406], divide by std=[0.229, 0.224, 0.225]
    5. HWC → CHW, add batch dim

Post-processing
---------------
  Keypoints from the model are in model-input-space (0..input_w, 0..input_h).
  We rescale them back to original frame coordinates using the crop offset
  and scale factor computed during preprocessing.

Performance
-----------
  PyTorch RTMPose-s  @256×192:  ~18 ms  (RTX 3080)
  ONNX   RTMPose-s  @256×192:  ~12 ms  (onnxruntime-gpu CUDA EP)
  TRT    RTMPose-s  @256×192:  ~7  ms  (TRT FP16, RTX 3080)
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from detector import PersonDetector, BBox   # type: ignore
from pose import Keypoints                  # type: ignore  (type alias only)

log = logging.getLogger(__name__)

# ImageNet normalisation constants
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_DEFAULT_SCORE_THRESHOLD = 0.3


# ═══════════════════════════════════════════════════════════════════════════════
#  Shared preprocessing / postprocessing
# ═══════════════════════════════════════════════════════════════════════════════

class _CropInfo:
    """Stores crop geometry to allow back-projection of keypoints to frame space."""
    __slots__ = ("x1", "y1", "scale_x", "scale_y", "input_h", "input_w")

    def __init__(
        self,
        x1: int, y1: int,
        crop_w: int, crop_h: int,
        input_h: int, input_w: int,
    ) -> None:
        self.x1      = x1
        self.y1      = y1
        self.scale_x = crop_w / input_w
        self.scale_y = crop_h / input_h
        self.input_h = input_h
        self.input_w = input_w


def _preprocess_pose(
    frame: np.ndarray,
    bbox: BBox,
    input_h: int = 256,
    input_w: int = 192,
    pad_ratio: float = 0.15,
) -> Tuple[np.ndarray, _CropInfo]:
    """
    Crop + resize + normalise a person region for RTMPose inference.

    Returns:
        tensor   : float32 (1, 3, input_h, input_w), ImageNet normalised
        crop_info: geometry for back-projecting keypoints to frame space
    """
    padded = PersonDetector.pad_bbox(bbox, frame.shape, pad_ratio)
    x1, y1, x2, y2 = padded
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

    crop   = frame[y1:y2, x1:x2]
    crop_h = max(y2 - y1, 1)
    crop_w = max(x2 - x1, 1)

    if crop.size == 0:
        # Fallback: return zero tensor
        zero = np.zeros((1, 3, input_h, input_w), dtype=np.float32)
        return zero, _CropInfo(x1, y1, crop_w, crop_h, input_h, input_w)

    resized = cv2.resize(crop, (input_w, input_h), interpolation=cv2.INTER_LINEAR)

    # BGR → RGB, [0,1], normalise
    rgb    = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normed = (rgb - _MEAN) / _STD                   # (H, W, 3)
    tensor = normed.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)

    return tensor, _CropInfo(x1, y1, crop_w, crop_h, input_h, input_w)


def _postprocess_keypoints(
    kp_model: np.ndarray,
    scores: np.ndarray,
    crop_info: _CropInfo,
    score_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map keypoints from model space back to original frame space.

    kp_model : (17, 2) in model-input coordinates
    Returns  : kp_frame (17, 2) in frame-pixel coordinates, scores (17,)
    """
    kp = kp_model.copy()

    # Scale from model input size → crop size → frame space
    kp[:, 0] = kp[:, 0] * crop_info.scale_x + crop_info.x1
    kp[:, 1] = kp[:, 1] * crop_info.scale_y + crop_info.y1

    # Zero out low-confidence keypoints
    low_conf = scores < score_threshold
    kp[low_conf] = 0.0

    return kp.astype(np.float32), scores.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  ONNX Runtime pose estimator
# ═══════════════════════════════════════════════════════════════════════════════

class OnnxPoseEstimator:
    """
    Drop-in for PoseEstimator using ONNX Runtime.

    Loads the ONNX model exported by tools/export_rtmpose_onnx.py.
    """

    def __init__(
        self,
        onnx_path:       str   = "models/rtmpose.onnx",
        input_h:         int   = 256,
        input_w:         int   = 192,
        score_threshold: float = _DEFAULT_SCORE_THRESHOLD,
        device:          Optional[str] = None,
    ) -> None:
        self.input_h         = input_h
        self.input_w         = input_w
        self.score_threshold = score_threshold

        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime not installed.\n"
                "  GPU: pip install onnxruntime-gpu\n"
                "  CPU: pip install onnxruntime"
            )

        available = ort.get_available_providers()
        if device == "cpu":
            providers = ["CPUExecutionProvider"]
        elif "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
            log.warning("OnnxPoseEstimator: CUDA EP not available, using CPU")

        self.device = "cuda" if "CUDAExecutionProvider" in providers else "cpu"

        log.info("OnnxPoseEstimator: loading %s on %s", onnx_path, self.device)
        t0 = time.perf_counter()
        self._session     = ort.InferenceSession(onnx_path, providers=providers)
        self._input_name  = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]
        log.info(
            "OnnxPoseEstimator: ready (%.1fs)  outputs=%s",
            time.perf_counter() - t0, self._output_names,
        )

        # Warmup
        dummy = np.zeros((1, 3, input_h, input_w), dtype=np.float32)
        for _ in range(3):
            self._session.run(None, {self._input_name: dummy})
        log.info("OnnxPoseEstimator: warmup complete")

    def estimate(
        self,
        frame: np.ndarray,
        bbox: BBox,
        pad_ratio: float = 0.15,
    ) -> Optional[Keypoints]:
        result = self.estimate_with_scores(frame, bbox, pad_ratio)
        if result is None:
            return None
        return result[0]

    def estimate_with_scores(
        self,
        frame: np.ndarray,
        bbox: BBox,
        pad_ratio: float = 0.15,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        try:
            tensor, crop_info = _preprocess_pose(frame, bbox, self.input_h, self.input_w, pad_ratio)
        except Exception as e:
            log.warning("OnnxPoseEstimator: preprocess failed: %s", e)
            return None

        try:
            t0 = time.perf_counter()
            outputs = self._session.run(None, {self._input_name: tensor})
            log.debug("OnnxPose: %.1f ms", (time.perf_counter() - t0) * 1000)
        except Exception as e:
            log.warning("OnnxPoseEstimator: inference failed: %s", e)
            return None

        # outputs[0]: (1,17,2) keypoints,  outputs[1]: (1,17) scores
        kp_model = outputs[0][0]    # (17, 2)
        scores   = outputs[1][0]    # (17,)

        kp_frame, scores_out = _postprocess_keypoints(
            kp_model, scores, crop_info, self.score_threshold
        )
        return kp_frame, scores_out


# ═══════════════════════════════════════════════════════════════════════════════
#  TensorRT FP16 pose estimator
# ═══════════════════════════════════════════════════════════════════════════════

class TrtPoseEstimator:
    """
    Drop-in for PoseEstimator using TensorRT FP16.

    Requires a .engine file built by tools/export_rtmpose_onnx.py --trt.
    Assumes outputs: binding 1 = keypoints (1,17,2), binding 2 = scores (1,17).
    """

    def __init__(
        self,
        engine_path:     str   = "models/rtmpose.engine",
        input_h:         int   = 256,
        input_w:         int   = 192,
        score_threshold: float = _DEFAULT_SCORE_THRESHOLD,
    ) -> None:
        self.input_h         = input_h
        self.input_w         = input_w
        self.score_threshold = score_threshold
        self.device          = "cuda"

        self._load_engine(engine_path)
        self._warmup()

    def _load_engine(self, engine_path: str) -> None:
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
        except ImportError as e:
            raise ImportError("TrtPoseEstimator requires tensorrt and pycuda.") from e

        import pycuda.driver as cuda
        import tensorrt as trt

        self._cuda   = cuda
        self._trt    = trt
        TRT_LOGGER   = trt.Logger(trt.Logger.ERROR)
        runtime      = trt.Runtime(TRT_LOGGER)

        log.info("TrtPoseEstimator: loading engine %s", engine_path)
        with open(engine_path, "rb") as f:
            self._engine  = runtime.deserialize_cuda_engine(f.read())
        self._context = self._engine.create_execution_context()
        self._stream  = cuda.Stream()

        # Pre-allocate I/O buffers (input + 2 outputs: keypoints, scores)
        self._bindings  = []
        self._h_bufs    = []
        self._d_bufs    = []
        self._out_shapes = []

        for i in range(self._engine.num_bindings):
            shape = self._engine.get_binding_shape(i)
            size  = int(np.prod(shape))
            dtype = trt.nptype(self._engine.get_binding_dtype(i))
            h_mem = cuda.pagelocked_empty(size, dtype)
            d_mem = cuda.mem_alloc(h_mem.nbytes)
            self._bindings.append(int(d_mem))
            self._h_bufs.append(h_mem)
            self._d_bufs.append(d_mem)
            if not self._engine.binding_is_input(i):
                self._out_shapes.append(shape)

        # Indices: input=0, kp_output=1, score_output=2
        self._n_inputs = sum(
            1 for i in range(self._engine.num_bindings)
            if self._engine.binding_is_input(i)
        )
        log.info("TrtPoseEstimator: engine loaded, %d bindings", self._engine.num_bindings)

    def _infer(
        self, tensor: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        cuda = self._cuda
        np.copyto(self._h_bufs[0], tensor.ravel())
        cuda.memcpy_htod_async(self._d_bufs[0], self._h_bufs[0], self._stream)
        self._context.execute_async_v2(
            bindings=self._bindings,
            stream_handle=self._stream.handle,
        )
        for i in range(self._n_inputs, self._engine.num_bindings):
            cuda.memcpy_dtoh_async(self._h_bufs[i], self._d_bufs[i], self._stream)
        self._stream.synchronize()

        # binding 1 → keypoints (1,17,2), binding 2 → scores (1,17)
        kp_idx = self._n_inputs
        sc_idx = self._n_inputs + 1
        kp = self._h_bufs[kp_idx].reshape(self._out_shapes[0])   # (1,17,2)
        sc = self._h_bufs[sc_idx].reshape(self._out_shapes[1])   # (1,17)
        return kp[0], sc[0]

    def _warmup(self, n: int = 5) -> None:
        dummy = np.zeros((1, 3, self.input_h, self.input_w), dtype=np.float32)
        for _ in range(n):
            self._infer(dummy)
        log.info("TrtPoseEstimator: warmup complete (%d runs)", n)

    def estimate(
        self,
        frame: np.ndarray,
        bbox: BBox,
        pad_ratio: float = 0.15,
    ) -> Optional[Keypoints]:
        result = self.estimate_with_scores(frame, bbox, pad_ratio)
        return result[0] if result is not None else None

    def estimate_with_scores(
        self,
        frame: np.ndarray,
        bbox: BBox,
        pad_ratio: float = 0.15,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        try:
            tensor, crop_info = _preprocess_pose(
                frame, bbox, self.input_h, self.input_w, pad_ratio
            )
        except Exception as e:
            log.warning("TrtPoseEstimator: preprocess failed: %s", e)
            return None

        try:
            t0 = time.perf_counter()
            kp_model, scores = self._infer(tensor.astype(np.float32))
            log.debug("TrtPose: %.1f ms", (time.perf_counter() - t0) * 1000)
        except Exception as e:
            log.warning("TrtPoseEstimator: inference failed: %s", e)
            return None

        kp_frame, scores_out = _postprocess_keypoints(
            kp_model, scores, crop_info, self.score_threshold
        )
        return kp_frame, scores_out


# ═══════════════════════════════════════════════════════════════════════════════
#  Auto-selector factory
# ═══════════════════════════════════════════════════════════════════════════════

def build_pose_estimator(
    trt_engine:      Optional[str] = None,
    onnx_path:       Optional[str] = None,
    input_h:         int           = 256,
    input_w:         int           = 192,
    score_threshold: float         = _DEFAULT_SCORE_THRESHOLD,
) -> "object":
    """
    Auto-select the best available pose estimator backend:
      1. TrtPoseEstimator   — if engine_path exists + tensorrt+pycuda installed
      2. OnnxPoseEstimator  — if onnx_path exists + onnxruntime installed
      3. PoseEstimator      — MMPose/DNN fallback (always available)

    Override with env vars:
      POSE_BACKEND     = trt | onnx | mmpose
      TRT_POSE_PATH    = path/to/rtmpose.engine
      ONNX_POSE_PATH   = path/to/rtmpose.onnx
    """
    from pose import PoseEstimator  # type: ignore

    backend_env = os.getenv("POSE_BACKEND", "auto").lower()
    trt_path    = trt_engine or os.getenv("TRT_POSE_PATH",  "models/rtmpose.engine")
    onnx_path   = onnx_path  or os.getenv("ONNX_POSE_PATH", "models/rtmpose.onnx")

    common = dict(input_h=input_h, input_w=input_w, score_threshold=score_threshold)

    # --- TRT ---
    if backend_env in ("auto", "trt"):
        if os.path.exists(trt_path):
            try:
                p = TrtPoseEstimator(engine_path=trt_path, **common)
                log.info("build_pose_estimator: using TrtPoseEstimator (%s)", trt_path)
                return p
            except Exception as e:
                log.warning("build_pose_estimator: TRT failed (%s) — trying ONNX", e)

    # --- ONNX ---
    if backend_env in ("auto", "onnx"):
        if os.path.exists(onnx_path):
            try:
                p = OnnxPoseEstimator(onnx_path=onnx_path, **common)
                log.info("build_pose_estimator: using OnnxPoseEstimator (%s)", onnx_path)
                return p
            except Exception as e:
                log.warning("build_pose_estimator: ONNX failed (%s) — falling back to MMPose", e)

    # --- MMPose fallback ---
    log.info("build_pose_estimator: using MMPose PoseEstimator")
    return PoseEstimator(score_threshold=score_threshold)
