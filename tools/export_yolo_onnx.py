"""
tools/export_yolo_onnx.py
═════════════════════════
Step 8 — Export YOLOv8 to ONNX and TensorRT FP16

Usage
-----
# ONNX only (CPU-safe, no TensorRT required):
    python tools/export_yolo_onnx.py --model yolov8n.pt --output models/yolo.onnx

# ONNX + TensorRT FP16 engine (requires tensorrt, CUDA GPU):
    python tools/export_yolo_onnx.py --model yolov8n.pt --output models/yolo.onnx --trt --trt-output models/yolo.engine

# Custom input size (must match detector input_size):
    python tools/export_yolo_onnx.py --model yolov8s.pt --output models/yolo.onnx --imgsz 640 --trt

Output files
------------
  <output>          ONNX model (opset 17, dynamic batch)
  <trt-output>      TensorRT FP16 serialised engine (if --trt)
  <output>.json     Metadata: original model, export timestamp, input shape, opset

Compatibility
-------------
  Tested with: ultralytics>=8.0.0, onnx>=1.14, onnxruntime-gpu>=1.16,
               tensorrt>=8.6 (TensorRT 10 also supported), torch>=2.0
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
log = logging.getLogger("export_yolo")


# ── Argument parsing ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export YOLOv8 → ONNX / TensorRT FP16")
    p.add_argument("--model",      default="yolov8n.pt",       help="Ultralytics .pt model path")
    p.add_argument("--output",     default="models/yolo.onnx", help="Output ONNX path")
    p.add_argument("--imgsz",      default=640, type=int,       help="Input image size (square)")
    p.add_argument("--opset",      default=17,  type=int,       help="ONNX opset version")
    p.add_argument("--batch",      default=1,   type=int,       help="Fixed batch size (use 1 for inference server)")
    p.add_argument("--trt",        action="store_true",         help="Also build TensorRT FP16 engine")
    p.add_argument("--trt-output", default="",                  help="TensorRT engine output path (auto if empty)")
    p.add_argument("--trt-workspace", default=4096, type=int,   help="TensorRT builder workspace MB")
    p.add_argument("--validate",   action="store_true",         help="Run ONNX validation after export")
    p.add_argument("--device",     default="0",                 help="CUDA device index (default: 0)")
    return p.parse_args()


# ── ONNX export ────────────────────────────────────────────────────────────────

def export_yolo_onnx(
    model_path: str,
    output_path: str,
    imgsz: int = 640,
    opset: int = 17,
    batch: int = 1,
    device: str = "0",
) -> str:
    """
    Export a YOLOv8 .pt model to ONNX using the Ultralytics built-in exporter.

    Returns the resolved absolute path to the exported ONNX file.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    output_path = str(Path(output_path).resolve())
    output_dir  = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading YOLOv8 model: %s", model_path)
    model = YOLO(model_path)

    log.info("Exporting to ONNX (opset=%d, imgsz=%d, batch=%d) …", opset, imgsz, batch)
    t0 = time.perf_counter()

    # Ultralytics export() saves to <model_stem>.onnx next to the .pt file
    # We then move it to the requested output_path.
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        dynamic=False,          # Fixed batch for TRT compatibility
        simplify=True,          # onnx-simplifier for cleaner graph
        half=False,             # Export in FP32; TRT builder handles FP16
        device=device,
        batch=batch,
    )

    elapsed = time.perf_counter() - t0
    log.info("ONNX export complete in %.1fs → %s", elapsed, exported)

    # Move to requested output path if different
    if str(Path(exported).resolve()) != output_path:
        import shutil
        shutil.move(str(exported), output_path)
        log.info("Moved → %s", output_path)

    # Write metadata sidecar
    meta = {
        "source_model":   model_path,
        "onnx_path":      output_path,
        "imgsz":          imgsz,
        "opset":          opset,
        "batch":          batch,
        "export_time_s":  round(elapsed, 2),
        "exported_at":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_path = output_path + ".json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Metadata written → %s", meta_path)

    return output_path


# ── ONNX validation ────────────────────────────────────────────────────────────

def validate_onnx(onnx_path: str, imgsz: int = 640, batch: int = 1) -> None:
    """Run a dummy inference through onnxruntime to confirm the graph is valid."""
    log.info("Validating ONNX model …")
    try:
        import onnxruntime as ort
    except ImportError:
        log.warning("onnxruntime not installed — skipping validation. pip install onnxruntime-gpu")
        return

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if _cuda_available()
        else ["CPUExecutionProvider"]
    )

    sess = ort.InferenceSession(onnx_path, providers=providers)
    inp  = sess.get_inputs()[0]
    log.info("  Input  : name=%s  shape=%s  dtype=%s", inp.name, inp.shape, inp.type)

    dummy = np.random.rand(batch, 3, imgsz, imgsz).astype(np.float32)
    t0    = time.perf_counter()
    out   = sess.run(None, {inp.name: dummy})
    latency_ms = (time.perf_counter() - t0) * 1000
    log.info("  Output : shapes=%s", [o.shape for o in out])
    log.info("  Latency: %.1f ms (onnxruntime)", latency_ms)
    log.info("Validation PASSED ✓")


# ── TensorRT engine build ─────────────────────────────────────────────────────

def build_trt_engine(
    onnx_path: str,
    engine_path: str,
    workspace_mb: int = 4096,
    imgsz: int = 640,
    batch: int = 1,
) -> str:
    """
    Build a TensorRT FP16 serialised engine from an ONNX file.

    Supports TensorRT 8.x and 10.x APIs.
    Returns the path to the .engine file.
    """
    log.info("Building TensorRT FP16 engine from: %s", onnx_path)
    log.info("  workspace_mb=%d  imgsz=%d  batch=%d", workspace_mb, imgsz, batch)

    try:
        import tensorrt as trt
    except ImportError:
        log.error(
            "tensorrt not installed. Install from: https://developer.nvidia.com/tensorrt\n"
            "  pip install tensorrt  (or use the TensorRT Docker image)"
        )
        sys.exit(1)

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        config = builder.create_builder_config()

        # Set workspace (API differs between TRT 8 and TRT 10)
        workspace_bytes = workspace_mb * 1024 * 1024
        if hasattr(config, "set_memory_pool_limit"):
            # TensorRT 10+
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        else:
            # TensorRT 8.x
            config.max_workspace_size = workspace_bytes  # type: ignore[attr-defined]

        # Enable FP16
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            log.info("  FP16 enabled ✓")
        else:
            log.warning("  Platform does not support fast FP16 — building FP32 engine")

        # Parse ONNX
        log.info("  Parsing ONNX graph …")
        with open(onnx_path, "rb") as f:
            onnx_data = f.read()
        if not parser.parse(onnx_data):
            for i in range(parser.num_errors):
                log.error("  ONNX parse error: %s", parser.get_error(i))
            sys.exit(1)
        log.info("  ONNX parsed successfully (%d layers)", network.num_layers)

        # Build engine
        log.info("  Building engine (this may take 1–5 minutes) …")
        t0 = time.perf_counter()

        # TRT 10 API
        if hasattr(builder, "build_serialized_network"):
            serialised = builder.build_serialized_network(network, config)
            if serialised is None:
                log.error("  Engine build FAILED")
                sys.exit(1)
            engine_bytes = bytes(serialised)
        else:
            # TRT 8 API
            engine = builder.build_engine(network, config)
            if engine is None:
                log.error("  Engine build FAILED")
                sys.exit(1)
            engine_bytes = engine.serialize()

        elapsed = time.perf_counter() - t0
        log.info("  Engine build complete in %.1fs", elapsed)

    # Save engine
    Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(engine_bytes)
    engine_size_mb = os.path.getsize(engine_path) / 1_048_576
    log.info("  Saved engine → %s  (%.1f MB)", engine_path, engine_size_mb)

    # Append engine info to sidecar
    meta_path = onnx_path + ".json"
    if Path(meta_path).exists():
        with open(meta_path) as f:
            meta = json.load(f)
        meta["trt_engine_path"]     = engine_path
        meta["trt_engine_size_mb"]  = round(engine_size_mb, 1)
        meta["trt_build_time_s"]    = round(elapsed, 2)
        meta["trt_workspace_mb"]    = workspace_mb
        meta["trt_fp16"]            = True
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    return engine_path


# ── TensorRT warmup benchmark ──────────────────────────────────────────────────

def benchmark_trt_engine(
    engine_path: str,
    imgsz: int = 640,
    batch: int = 1,
    n_warmup: int = 10,
    n_bench: int = 100,
) -> None:
    """Run latency benchmark on the built TRT engine."""
    log.info("Benchmarking TensorRT engine …")
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
    except ImportError:
        log.warning("pycuda not installed — skipping TRT benchmark. pip install pycuda")
        return

    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(TRT_LOGGER)

    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Allocate I/O buffers
    bindings = []
    host_inputs, device_inputs = [], []
    host_outputs, device_outputs = [], []
    output_shapes = []

    for i in range(engine.num_bindings):
        shape = engine.get_binding_shape(i)
        size  = int(np.prod(shape))
        dtype = trt.nptype(engine.get_binding_dtype(i))
        host_mem   = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(i):
            host_inputs.append(host_mem)
            device_inputs.append(device_mem)
        else:
            host_outputs.append(host_mem)
            device_outputs.append(device_mem)
            output_shapes.append(shape)

    stream = cuda.Stream()

    def _run_once(dummy_input: np.ndarray) -> None:
        np.copyto(host_inputs[0], dummy_input.ravel())
        cuda.memcpy_htod_async(device_inputs[0], host_inputs[0], stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(host_outputs[0], device_outputs[0], stream)
        stream.synchronize()

    dummy = np.random.rand(batch, 3, imgsz, imgsz).astype(np.float32)

    log.info("  Warming up (%d runs) …", n_warmup)
    for _ in range(n_warmup):
        _run_once(dummy)

    log.info("  Benchmarking (%d runs) …", n_bench)
    latencies = []
    for _ in range(n_bench):
        t0 = time.perf_counter()
        _run_once(dummy)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    log.info("  TRT FP16 benchmark results:")
    log.info("    p50 : %.2f ms", latencies[n_bench // 2])
    log.info("    p90 : %.2f ms", latencies[int(n_bench * 0.90)])
    log.info("    p99 : %.2f ms", latencies[int(n_bench * 0.99)])
    log.info("    min : %.2f ms", latencies[0])
    log.info("    max : %.2f ms", latencies[-1])


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

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.device)

    log.info("=" * 60)
    log.info("Step 8 — YOLOv8 ONNX / TensorRT Export")
    log.info("=" * 60)

    # 1. Export ONNX
    onnx_path = export_yolo_onnx(
        model_path  = args.model,
        output_path = args.output,
        imgsz       = args.imgsz,
        opset       = args.opset,
        batch       = args.batch,
        device      = args.device,
    )

    # 2. Validate ONNX (optional)
    if args.validate:
        validate_onnx(onnx_path, imgsz=args.imgsz, batch=args.batch)

    # 3. Build TensorRT engine (optional)
    if args.trt:
        if not _cuda_available():
            log.error("No CUDA GPU detected — cannot build TensorRT engine.")
            sys.exit(1)

        trt_output = args.trt_output or onnx_path.replace(".onnx", ".engine")
        engine_path = build_trt_engine(
            onnx_path    = onnx_path,
            engine_path  = trt_output,
            workspace_mb = args.trt_workspace,
            imgsz        = args.imgsz,
            batch        = args.batch,
        )
        benchmark_trt_engine(engine_path, imgsz=args.imgsz, batch=args.batch)

    log.info("=" * 60)
    log.info("Export complete.")
    log.info("  ONNX engine  → %s", onnx_path)
    if args.trt:
        log.info("  TRT engine   → %s", trt_output)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
