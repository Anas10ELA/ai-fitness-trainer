"""
model/inference.py
══════════════════
Production Inference Wrapper — ST-GCN Edition  [v4]

Changes from legacy BiLSTM version (v3):
  • Loads the STGCN model from scripts/train.py instead of model/architecture.py
  • Maintains a rolling temporal deque of T=60 normalized frames
  • Converts the buffer to (1, C=2, T, V=17, M=1) and runs a single forward pass
  • Returns one of 7 AI-predicted states instead of the old 3-state vocabulary
  • FPS-drop resilience: auto-pads short gaps; flushes stale windows
  • Mode-filters the last SMOOTH_K predictions to suppress single-frame flicker
  • Removed: normalizer_stats.json (Z-score path) — hip+shoulder normalization only
  • Removed: BiLSTM / build_model / AMP autocast (STGCN is fast enough on CPU)
"""

from __future__ import annotations

import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Import the STGCN class and constants from the training file.
# train.py uses  `if __name__ == "__main__": main()` so importing it is safe.
from model.train import N_COORDS, N_STATES, STGCN, STATE_NAMES  # noqa: E402

log = logging.getLogger(__name__)

# ── Inference constants ────────────────────────────────────────────────────────

SMOOTH_K:            int   = 5      # mode-filter window (frames)
PAD_MAX_FRAMES:      int   = 15     # max repeated frames injected on an FPS drop
PAD_MULTIPLIER:      float = 1.5    # gap > PAD_MULTIPLIER/fps → inject padding
STALE_WINDOW_SECS:   float = 2.0    # gap > this → flush buffer entirely
DEFAULT_TARGET_FPS:  float = 30.0   # assumed frame rate when not specified


# ═══════════════════════════════════════════════════════════════════════════════
#  Prediction result object
# ═══════════════════════════════════════════════════════════════════════════════

class ModelPrediction:
    """
    Immutable result returned by push_frame().

    Attributes
    ----------
    state_name  : one of the 7 state strings, or "neutral" when not ready
    state_idx   : integer index 0-6 (matches STATE_TO_IDX in build_dataset.py)
    state_conf  : confidence ∈ [0, 1] from mode-smoothed softmax
    raw_conf    : confidence of the single-frame logit (pre-smoothing)
    ready       : False until the buffer has T frames for the first time
    """

    __slots__ = ("state_name", "state_idx", "state_conf", "raw_conf", "ready")

    def __init__(
        self,
        state_name: str   = "neutral",
        state_idx:  int   = 0,
        state_conf: float = 0.0,
        raw_conf:   float = 0.0,
        ready:      bool  = False,
    ) -> None:
        self.state_name = state_name
        self.state_idx  = state_idx
        self.state_conf = state_conf
        self.raw_conf   = raw_conf
        self.ready      = ready

    def to_dict(self) -> Dict:
        return {
            "state_name":  self.state_name,
            "state_idx":   self.state_idx,
            "state_conf":  round(self.state_conf, 3),
            "raw_conf":    round(self.raw_conf, 3),
            "model_ready": self.ready,
        }

    def __repr__(self) -> str:
        if not self.ready:
            return "ModelPrediction(not_ready)"
        return (
            f"ModelPrediction(state={self.state_name!r}, "
            f"conf={self.state_conf:.2f}, ready=True)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Inference Engine
# ═══════════════════════════════════════════════════════════════════════════════

class FitnessInferenceEngine:
    """
    Stateful, single-person ST-GCN inference engine.

    Usage
    -----
        engine = FitnessInferenceEngine(
            checkpoint_path = "checkpoints/stgcn_best.pt",
            graph_path      = "data/processed/graph_A.npy",
        )
        while cap.isOpened():
            kp, sc = pose_estimator(frame)    # (17, 2), (17,)
            pred   = engine.push_frame(kp, sc)
            if pred.ready:
                do_something(pred.state_name)

    FPS-drop resilience
    -------------------
    push_frame() timestamps every call.  If the inter-frame gap exceeds
    PAD_MULTIPLIER / target_fps seconds, we inject copies of the last valid
    normalized frame (up to PAD_MAX_FRAMES) so the temporal window does not
    become dominated by a single stale frame.  If the gap exceeds
    STALE_WINDOW_SECS the buffer is flushed entirely and `ready` is reset to
    False — running inference on a zombie window yields nonsensical states.
    """

    def __init__(
        self,
        checkpoint_path:   str,
        graph_path:        str             = "data/processed/graph_A.npy",
        window:            int             = 60,
        target_fps:        float           = DEFAULT_TARGET_FPS,
        device:            str             = "auto",
        kp_conf_threshold: float           = 0.25,
    ) -> None:
        self.window            = window
        self.target_fps        = target_fps
        self.kp_conf_threshold = kp_conf_threshold

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Rolling buffer of normalized frames  (17, 2) each
        self._buffer: Deque[np.ndarray] = deque(maxlen=window)

        # Smoothing history of raw state indices
        self._pred_history: Deque[int] = deque(maxlen=SMOOTH_K)

        # FPS-drop / stale-window detection
        self._last_push_ts:  float            = 0.0
        self._last_norm_kp:  Optional[np.ndarray] = None

        # Model (populated by _load_model)
        self.model: Optional[torch.nn.Module] = None

        self._load_model(checkpoint_path, graph_path)

    # ── Public API ─────────────────────────────────────────────────────────────

    def push_frame(
        self,
        keypoints: np.ndarray,               # (17, 2)  pixel or normalized coords
        scores:    Optional[np.ndarray] = None,  # (17,) confidence per joint
    ) -> ModelPrediction:
        """
        Accept one pose frame, update the rolling buffer, and return the current
        AI-predicted state.  Returns ModelPrediction(ready=False) until the
        buffer has accumulated `window` frames.

        Thread safety: NOT thread-safe — call from a single producer thread.
        """
        now = time.monotonic()
        frame_interval = 1.0 / self.target_fps

        # ── FPS-drop handling ─────────────────────────────────────────────────
        if self._last_push_ts > 0.0:
            gap = now - self._last_push_ts

            if gap > STALE_WINDOW_SECS:
                # More than N seconds without a frame — the buffer is stale.
                # Flush so the next T fresh frames rebuild a clean window.
                log.debug(
                    "Buffer flushed: %.2fs gap exceeds stale threshold (%.2fs)",
                    gap, STALE_WINDOW_SECS,
                )
                self._buffer.clear()
                self._pred_history.clear()
                self._last_norm_kp = None

            elif gap > PAD_MULTIPLIER * frame_interval and self._last_norm_kp is not None:
                # Moderate FPS drop: inject repeated last frame to keep the
                # temporal window representative of the last known pose.
                n_pad = min(
                    int(gap * self.target_fps) - 1,
                    PAD_MAX_FRAMES,
                    self.window - len(self._buffer) - 1,  # don't overfill
                )
                for _ in range(max(0, n_pad)):
                    self._buffer.append(self._last_norm_kp)

        self._last_push_ts = now

        # ── Normalize and buffer current frame ───────────────────────────────
        norm_kp = self._normalize_frame(keypoints, scores)
        self._buffer.append(norm_kp)
        self._last_norm_kp = norm_kp

        if len(self._buffer) < self.window:
            return ModelPrediction(ready=False)

        return self._run_inference()

    def reset(self) -> None:
        """Flush the rolling buffer and prediction history (e.g. on exercise switch)."""
        self._buffer.clear()
        self._pred_history.clear()
        self._last_norm_kp = None
        self._last_push_ts = 0.0

    # ── Inference ──────────────────────────────────────────────────────────────

    def _run_inference(self) -> ModelPrediction:
        """
        Convert the current buffer to an ST-GCN tensor and run a forward pass.

        Buffer:  deque of T × (V=17, C=2) normalized frames
        Tensor:  (1, C=2, T, V=17, M=1)   — canonical ST-GCN input format

        The reshape mirrors build_dataset.py's sliding_window_stgcn() exactly:
            (T, V, C) → transpose(2,0,1) → (C, T, V) → [np.newaxis, ..., np.newaxis]
        """
        if self.model is None:
            return ModelPrediction(ready=False)

        # (T, V, C)  e.g. (60, 17, 2)
        frames = np.array(list(self._buffer), dtype=np.float32)

        # (C, T, V)  — channel first
        frames_cht = np.ascontiguousarray(frames.transpose(2, 0, 1))

        # (1, C, T, V, 1)  — add batch + person dims
        x = torch.from_numpy(
            frames_cht[np.newaxis, :, :, :, np.newaxis]
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(x)                         # (1, N_STATES)

        probs     = F.softmax(logits, dim=-1)[0].cpu()    # (N_STATES,)
        raw_idx   = int(probs.argmax().item())
        raw_conf  = float(probs[raw_idx].item())

        # Mode-smooth over the last SMOOTH_K predictions to reduce flicker
        smooth_idx, smooth_conf = self._smooth_prediction(raw_idx)

        return ModelPrediction(
            state_name = STATE_NAMES[smooth_idx],
            state_idx  = smooth_idx,
            state_conf = smooth_conf,
            raw_conf   = raw_conf,
            ready      = True,
        )

    def _smooth_prediction(self, raw_idx: int) -> Tuple[int, float]:
        """
        Mode-filter the last SMOOTH_K raw predictions.

        Returns the most frequently predicted state index and its frequency
        as a pseudo-confidence ∈ (0, 1].  On ties numpy's argmax picks the
        lower index — this consistently biases toward "neutral" (idx 0),
        which is the safest ambiguous state.
        """
        self._pred_history.append(raw_idx)
        hist   = np.array(self._pred_history, dtype=np.int32)
        counts = np.bincount(hist, minlength=N_STATES)
        mode   = int(counts.argmax())
        conf   = float(counts[mode]) / len(hist)
        return mode, conf

    # ── Normalization ──────────────────────────────────────────────────────────

    def _normalize_frame(
        self,
        kp:     np.ndarray,               # (17, 2)
        scores: Optional[np.ndarray],     # (17,) or None
    ) -> np.ndarray:
        """
        Per-frame hip-center translation + shoulder-hip scale normalization.

        This is the **only** normalization step — identical to
        normalize_sequence() in build_dataset.py (frame-wise equivalent).
        Low-confidence joints are zeroed before center/scale computation.
        """
        kp = kp.astype(np.float32).copy()

        if scores is not None:
            mask     = scores < self.kp_conf_threshold
            kp[mask] = 0.0

        # ── Hip center (joints 11=left_hip, 12=right_hip) ─────────────────────
        lh, rh = kp[11], kp[12]
        valid_hips = [p for p in [lh, rh] if not (p[0] == 0 and p[1] == 0)]
        if not valid_hips:
            return kp   # fallback: keep raw if hips are invisible
        hip_center = np.mean(valid_hips, axis=0)   # (2,)

        # ── Scale: shoulder-to-hip distance (joints 5, 6) ────────────────────
        ls, rs = kp[5], kp[6]
        valid_sh = [p for p in [ls, rs] if not (p[0] == 0 and p[1] == 0)]
        scale = 1.0
        if valid_sh:
            sh_center = np.mean(valid_sh, axis=0)
            scale     = float(np.linalg.norm(sh_center - hip_center))
            scale     = max(scale, 1e-6)

        nonzero        = ~((kp[:, 0] == 0) & (kp[:, 1] == 0))
        kp[nonzero]    = (kp[nonzero] - hip_center) / scale
        return kp

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_model(self, checkpoint_path: str, graph_path: str) -> None:
        """
        Load the STGCN model from a checkpoint saved by scripts/train.py.

        The checkpoint format (from train.py) is:
            {
                "epoch":       int,
                "model_state": OrderedDict,
                "val_loss":    float,
                "val_acc":     float,
                "args":        dict,     ← contains kt, dropout
                "state_names": list[str]
            }
        The adjacency matrix is loaded separately from graph_path.
        """
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            log.warning(
                "Checkpoint not found at %s — engine will return not_ready "
                "until a valid checkpoint is loaded.", checkpoint_path
            )
            return

        graph_np = Path(graph_path)
        if not graph_np.exists():
            raise FileNotFoundError(
                f"Adjacency matrix not found at {graph_path}. "
                "Run build_dataset.py first."
            )

        ckpt       = torch.load(str(ckpt_path), map_location=self.device,
                                weights_only=False)
        train_args = ckpt.get("args", {})
        A          = torch.from_numpy(
            np.load(str(graph_np)).astype(np.float32)
        ).to(self.device)

        self.model = STGCN(
            A            = A,
            in_channels  = N_COORDS,       # 2
            num_classes  = N_STATES,        # 7
            kt           = train_args.get("kt",      9),
            dropout      = 0.0,            # no dropout at inference time
        ).to(self.device)

        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        # Override window from checkpoint if it was saved
        saved_window = train_args.get("window")
        if saved_window and saved_window != self.window:
            log.info(
                "Overriding window from checkpoint: %d → %d",
                self.window, saved_window,
            )
            self.window = saved_window
            self._buffer = deque(maxlen=self.window)

        log.info(
            "ST-GCN loaded from %s  (epoch=%d  val_acc=%.3f  device=%s)",
            checkpoint_path,
            ckpt.get("epoch", -1),
            ckpt.get("val_acc", 0.0),
            self.device,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Module-level singleton  (convenience for simple single-session usage)
# ═══════════════════════════════════════════════════════════════════════════════

_engine_instance: Optional[FitnessInferenceEngine] = None


def get_engine(
    checkpoint_path: str = "checkpoints/stgcn_best.pt",
    graph_path:      str = "data/processed/graph_A.npy",
    window:          int = 60,
    target_fps:      float = DEFAULT_TARGET_FPS,
) -> FitnessInferenceEngine:
    """
    Return the module-level singleton engine, creating it on first call.

    Not appropriate for multi-exercise sessions — call the constructor directly
    and call engine.reset() between exercises.
    """
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = FitnessInferenceEngine(
            checkpoint_path = checkpoint_path,
            graph_path      = graph_path,
            window          = window,
            target_fps      = target_fps,
        )
    return _engine_instance
