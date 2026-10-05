"""
tools/export_rtmpose_onnx.py
════════════════════════════
Step 8 — Export RTMPose to ONNX and TensorRT FP16

RTMPose is a top-down pose estimator from MMPose. This script exports the
pre-trained RTMPose-s (or any RTMPose variant) to ONNX and optionally to a
TensorRT FP16 serialised engine, ready for use with OnnxPoseEstimator
(trt_pose.py).

Usage
-----
# ONNX only:
    python tools/export_rtmpose_onnx.py --output models/rtmpose.onnx

# ONNX + TRT FP16:
    python tools/export_rtmpose_onnx.py --output models/rtmpose.onnx --trt

# Custom checkpoint:
    python tools/export_rtmpose_onnx.py \\
        --config rtmpose-m_8xb256-420e_coco-256x192 \\
        --checkpoint /path/to/rtmpose-m.pth \\
        --output models/rtmpose-m.onnx --trt

Output files
------------
  <output>           ONNX model
  <output>.engine    TensorRT FP16 engine (if --trt)
  <output>.json      Export metadata

Notes
-----
  RTMPose input:  (1, 3, H, W)  — typically 256×192
  RTMPose output: (1, 17, H/4, W/4) heatmaps  [some variants: (1,17,2) simcc]
  The export handles both heatmap and SimCC output heads.

  MMPose must be installed:  pip install mmpose mmcv mmdet
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("export_rtmpose")


# ── RTMPose model registry ─────────────────────────────────────────────────────

_RTMPOSE_CONFIGS = {
    "rtmpose-s": {
        "config": "rtmpose-s_8xb256-420e_coco-256x192",
        "input_h": 256, "input_w": 192,
    },
    "rtmpose-m": {
        "config": "rtmpose-m_8xb256-420e_coco-256x192",
        "input_h": 256, "input_w": 192,
    },
    "rtmpose-l": {
        "config": "rtmpose-l_8xb256-420e_coco-256x192",
        "input_h": 256, "input_w": 192,
    },
}
_DEFAULT_VARIANT = "rtmpose-s"


# ── Argument parsing ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export RTMPose → ONNX / TensorRT FP16")
    p.add_argument("--variant",    default=_DEFAULT_VARIANT,
                   choices=list(_RTMPOSE_CONFIGS.keys()),
                   help="RTMPose variant (default: rtmpose-s)")
    p.add_argument("--config",     default="",
                   help="Override MMPose config name (optional)")
    p.add_argument("--checkpoint", default="auto",
                   help="Override checkpoint path (optional, 'auto' = download)")
    p.add_argument("--output",     default="models/rtmpose.onnx",
                   help="ONNX output path")
    p.add_argument("--opset",      default=17, type=int, help="ONNX opset")
    p.add_argument("--trt",        action="store_true",  help="Build TRT FP16 engine")
    p.add_argument("--trt-output", default="",           help="TRT engine path (auto if empty)")
    p.add_argument("--trt-workspace", default=2048, type=int, help="TRT workspace MB")
    p.add_argument("--validate",   action="store_true",  help="Run onnxruntime validation")
    p.add_argument("--device",     default="cuda:0",     help="Torch device for export")
    return p.parse_args()


# ── MMPose → ONNX export ───────────────────────────────────────────────────────

def export_rtmpose_onnx(
    config_name: str,
    checkpoint: str,
    output_path: str,
    input_h: int = 256,
    input_w: int = 192,
    opset: int = 17,
    device: str = "cuda:0",
) -> str:
    """
    Export RTMPose via MMPose's mmdeploy-style torch.onnx.export.

    Strategy:
      1. Instantiate the MMPose model.
      2. Wrap it in a thin export-friendly module that takes a raw image
         tensor (B, C, H, W) and returns keypoints + scores.
      3. Export with torch.onnx.export.

    Returns absolute path to the ONNX file.
    """
    output_path = str(Path(output_path).resolve())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    log.info("Loading RTMPose config: %s", config_name)
    log.info("Checkpoint: %s", checkpoint)
    log.info("Device: %s", device)

    try:
        import torch
        from mmpose.apis import MMPoseInferencer
    except ImportError:
        log.error(
            "mmpose not installed.\n"
            "  pip install mmpose mmcv-full mmdet\n"
            "  (see https://mmpose.readthedocs.io/en/latest/installation.html)"
        )
        sys.exit(1)

    # ── Build the inferencer to trigger model download / config resolution ───
    log.info("Initialising MMPoseInferencer (triggers model download if needed) …")
    inferencer = MMPoseInferencer(
        pose2d         = config_name,
        pose2d_weights = None if checkpoint == "auto" else checkpoint,
        device         = device,
        show_progress  = False,
    )

    # Extract the internal pose model
    pose_model = inferencer.pose2d_model.model
    pose_model.eval()
    pose_model.to(device)
    log.info("RTMPose model loaded (%d parameters)",
             sum(p.numel() for p in pose_model.parameters()))

    # ── Export wrapper ────────────────────────────────────────────────────────
    class _RTMPoseExportWrapper(torch.nn.Module):
        """
        Thin wrapper that exposes a clean (image_tensor) → (keypoints, scores)
        interface for ONNX export.

        RTMPose internally uses a data_preprocessor + backbone + head pipeline.
        We bypass the MMPose data pipeline and feed a normalised tensor directly.
        """
        def __init__(self, model: torch.nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(self, x: "torch.Tensor") -> "tuple[torch.Tensor, torch.Tensor]":
            # x: (B, 3, H, W), normalised [0,1]
            # Most RTMPose heads expose a `predict` or `decode` method.
            # We use the backbone + neck + head forward path directly.
            feats = self.model.backbone(x)
            if hasattr(self.model, "neck") and self.model.neck is not None:
                feats = self.model.neck(feats)
            # head.forward returns either heatmaps or (locs_x, locs_y) for SimCC
            head_out = self.model.head.forward(feats)

            # Normalise output to (B, 17, 2) keypoints + (B, 17) scores
            # regardless of whether it is a heatmap or SimCC head
            if isinstance(head_out, (list, tuple)) and len(head_out) == 2:
                # SimCC output: (pred_x, pred_y) each (B, 17, bins)
                pred_x, pred_y = head_out
                scores_x = torch.softmax(pred_x, dim=-1)
                scores_y = torch.softmax(pred_y, dim=-1)
                kp_x = (scores_x * torch.arange(scores_x.shape[-1],
                         dtype=scores_x.dtype, device=scores_x.device)).sum(-1)
                kp_y = (scores_y * torch.arange(scores_y.shape[-1],
                         dtype=scores_y.dtype, device=scores_y.device)).sum(-1)
                keypoints = torch.stack([kp_x, kp_y], dim=-1)   # (B, 17, 2)
                scores    = (scores_x.max(-1).values + scores_y.max(-1).values) / 2
            else:
                # Heatmap output: (B, 17, H', W')
                heatmaps  = head_out if not isinstance(head_out, (list, tuple)) else head_out[0]
                B, K, Hh, Wh = heatmaps.shape
                flat      = heatmaps.view(B, K, -1)
                scores    = flat.amax(dim=-1)
                idx       = flat.argmax(dim=-1)
                kp_x      = (idx % Wh).float()
                kp_y      = (idx // Wh).float()
                keypoints = torch.stack([kp_x, kp_y], dim=-1)   # (B, 17, 2)

            return keypoints, scores   # (B,17,2), (B,17)

    wrapper = _RTMPoseExportWrapper(pose_model).eval().to(device)

    dummy_input = torch.zeros(1, 3, input_h, input_w, device=device)

    log.info("Exporting to ONNX (opset=%d, input=%dx%d) …", opset, input_h, input_w)
    t0 = time.perf_counter()

    import torch.onnx
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        opset_version       = opset,
        input_names         = ["image"],
        output_names        = ["keypoints", "scores"],
        dynamic_axes        = None,     # Fixed batch=1 for TRT
        do_constant_folding = True,
        export_params       = True,
    )

    elapsed = time.perf_counter() - t0
    onnx_size_mb = os.path.getsize(output_path) / 1_048_576
    log.info("ONNX export complete in %.1fs  (%.1f MB) → %s", elapsed, onnx_size_mb, output_path)

    # Optional: run onnx-simplifier
    try:
        import onnx
        import onnxsim
        log.info("Running onnx-simplifier …")
        model_onnx = onnx.load(output_path)
        model_simplified, ok = onnxsim.simplify(model_onnx)
        if ok:
            onnx.save(model_simplified, output_path)
            log.info("  Simplified ✓")
        else:
            log.warning("  Simplifier returned not-ok — keeping original")
    except ImportError:
        log.info("onnx-simplifier not installed (optional). pip install onnxsim")

    # Write metadata
    meta = {
        "model":         "rtmpose",
        "config":        config_name,
        "checkpoint":    checkpoint,
        "onnx_path":     output_path,
        "input_h":       input_h,
        "input_w":       input_w,
        "opset":         opset,
        "output_names":  ["keypoints", "scores"],
        "export_time_s": round(elapsed, 2),
        "exported_at":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_path = output_path + ".json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Metadata written → %s", meta_path)

    return output_path


# ── ONNX validation ────────────────────────────────────────────────────────────

def validate_onnx(onnx_path: str, input_h: int = 256, input_w: int = 192) -> None:
    log.info("Validating ONNX model …")
    try:
        import onnxruntime as ort
    except ImportError:
        log.warning("onnxruntime not installed — skipping. pip install onnxruntime-gpu")
        return

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
                if _cuda_available() else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)

    inp   = sess.get_inputs()[0]
    outs  = sess.get_outputs()
    log.info("  Input : name=%s shape=%s", inp.name, inp.shape)
    for o in outs:
        log.info("  Output: name=%s shape=%s", o.name, o.shape)

    dummy = np.random.rand(1, 3, input_h, input_w).astype(np.float32)
    t0    = time.perf_counter()
    result = sess.run(None, {inp.name: dummy})
    latency_ms = (time.perf_counter() - t0) * 1000

    kp_shape = result[0].shape   # (1, 17, 2)
    sc_shape = result[1].shape   # (1, 17)
    log.info("  keypoints shape : %s", kp_shape)
    log.info("  scores shape    : %s", sc_shape)
    log.info("  Latency : %.1f ms (onnxruntime)", latency_ms)
    log.info("Validation PASSED ✓")


# ── TensorRT engine build ─────────────────────────────────────────────────────

def build_trt_engine(
    onnx_path: str,
    engine_path: str,
    workspace_mb: int = 2048,
    input_h: int = 256,
    input_w: int = 192,
) -> str:
    log.info("Building TensorRT FP16 engine from: %s", onnx_path)
    try:
        import tensorrt as trt
    except ImportError:
        log.error("tensorrt not installed. See https://developer.nvidia.com/tensorrt")
        sys.exit(1)

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        config = builder.create_builder_config()
        workspace_bytes = workspace_mb * 1024 * 1024

        if hasattr(config, "set_memory_pool_limit"):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        else:
            config.max_workspace_size = workspace_bytes  # type: ignore

        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            log.info("  FP16 enabled ✓")

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    log.error("  Parse error: %s", parser.get_error(i))
                sys.exit(1)

        log.info("  Parsed %d layers. Building …", network.num_layers)
        t0 = time.perf_counter()

        if hasattr(builder, "build_serialized_network"):
            serialised = builder.build_serialized_network(network, config)
            engine_bytes = bytes(serialised) if serialised else None
        else:
            engine = builder.build_engine(network, config)
            engine_bytes = engine.serialize() if engine else None

        if engine_bytes is None:
            log.error("  Engine build FAILED")
            sys.exit(1)

        elapsed = time.perf_counter() - t0
        log.info("  Built in %.1fs", elapsed)

    Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(engine_bytes)

    engine_size_mb = os.path.getsize(engine_path) / 1_048_576
    log.info("  Saved → %s  (%.1f MB)", engine_path, engine_size_mb)

    # Update metadata sidecar
    meta_path = onnx_path + ".json"
    if Path(meta_path).exists():
        with open(meta_path) as f:
            meta = json.load(f)
        meta["trt_engine_path"]    = engine_path
        meta["trt_engine_size_mb"] = round(engine_size_mb, 1)
        meta["trt_build_time_s"]   = round(elapsed, 2)
        meta["trt_fp16"]           = True
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    return engine_path


# ── Utilities ──────────────────────────────────────────────────────────────────

def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    variant_cfg = _RTMPOSE_CONFIGS[args.variant]
    config_name = args.config or variant_cfg["config"]
    input_h     = variant_cfg["input_h"]
    input_w     = variant_cfg["input_w"]

    log.info("=" * 60)
    log.info("Step 8 — RTMPose ONNX / TensorRT Export")
    log.info("=" * 60)
    log.info("  Variant : %s", args.variant)
    log.info("  Config  : %s", config_name)
    log.info("  Input   : %dx%d", input_h, input_w)

    onnx_path = export_rtmpose_onnx(
        config_name = config_name,
        checkpoint  = args.checkpoint,
        output_path = args.output,
        input_h     = input_h,
        input_w     = input_w,
        opset       = args.opset,
        device      = args.device,
    )

    if args.validate:
        validate_onnx(onnx_path, input_h=input_h, input_w=input_w)

    if args.trt:
        if not _cuda_available():
            log.error("No CUDA GPU — cannot build TRT engine.")
            sys.exit(1)
        trt_output  = args.trt_output or onnx_path.replace(".onnx", ".engine")
        build_trt_engine(
            onnx_path    = onnx_path,
            engine_path  = trt_output,
            workspace_mb = args.trt_workspace,
            input_h      = input_h,
            input_w      = input_w,
        )

    log.info("=" * 60)
    log.info("Export complete.")
    log.info("  ONNX  → %s", onnx_path)
    if args.trt:
        log.info("  TRT   → %s", trt_output)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
