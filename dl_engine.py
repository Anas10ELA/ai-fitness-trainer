"""
dl_engine.py  (v3 — Sliding-Window MLP)
════════════════════════════════════════
Drop-in replacement for the v2 engine.  Public API is IDENTICAL:

    from dl_engine import DLEngine, EngineResult
    engine = DLEngine("squat", checkpoint_dir="checkpoints_v3")
    result = engine.update(kalman_kp)   # (17,2) float32
    result.ready / .state_name / .confidence / .reps / .hold_seconds
    engine.reset() / engine.reset_reps() / engine.is_loaded

What changed from v2
────────────────────
  • The model now classifies a SLIDING WINDOW of the last W frames instead of
    a single 20-dim frame vector.  _WindowDeque holds the last W base-feature
    vectors and flattens them oldest-first into (W*10,) — the exact layout
    build_window_matrix() produces during training.

  • Warmup is shorter conceptually but we still left-pad: until W frames have
    been seen we repeat the oldest available frame, exactly like training.
    The engine reports ready=True as soon as ONE frame exists, because the
    left-pad makes a valid window from frame 0.  (If you prefer to suppress
    early predictions, set warmup_frames > 0.)

  • _ExerciseModel reads ckpt["window_size"] and builds _WindowNet accordingly.
    No hand-built deltas anywhere — the window is the temporal signal.

  • version check expects "dl_v3".

What changed in v3.2 (FPS-aware counting)
─────────────────────────────────────────
  • The FSM thresholds (open_dwell / away_dwell / cooldown / away_min) are now
    defined in SECONDS and converted to frame counts on the fly from the live
    inference fps.  update() takes an optional infer_fps so a rep means the same
    thing in wall-clock time regardless of how fast the machine runs.  At 30 fps
    the behavior is identical to the old fixed frame counts.

  • New reset_window(): flush the sliding window + smoother + in-flight FSM state
    after a tracker dropout WITHOUT wiping reps / hold time.

Everything else (smoother, FSM rep logic, hold logic) is unchanged.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Public result type
# ══════════════════════════════════════════════════════════════════════════════

class EngineResult(NamedTuple):
    ready:        bool
    state_name:   str
    confidence:   float
    reps:         int
    hold_seconds: float


_NOT_READY = EngineResult(
    ready=False, state_name="", confidence=0.0, reps=0, hold_seconds=0.0,
)


# ══════════════════════════════════════════════════════════════════════════════
#  V3 rep sequences  (isolated, merged vocab — same as v2 engine)
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  v4 rep ANCHORS  — the new counting basis (replaces strict sequence matching)
#
#  WHY THE CHANGE
#  ──────────────
#  v3 counted reps by matching the exact ordered sequence of ALL states
#  (descending → bottom → up).  That made counting hostage to the recall of the
#  hardest, shortest state — the transition — whose label boundaries are
#  humanly ambiguous (sq_descending recall 0.25, pu_bottom 0.10).  Worse, only
#  squat & pushup had a sequence defined at all; situp / lateral_raise /
#  tricep_dip had their sequences COMMENTED OUT, so they counted 0 reps live
#  regardless of how good the model was (tricep_dip: 93% frame-acc → 0 reps).
#
#  v4 counts on a 2-ANCHOR hysteresis instead.  Each rep-exercise is described
#  only by its two RELIABLE extreme states:
#       open_anchor  = the resting / extended posture (highest recall)
#       closed_anchor= the contracted / peak posture
#  A rep is one full excursion  open → (leave) → … → (return) open , validated
#  by either reaching the closed anchor OR a minimum away-duration.  The
#  transition state is now IRRELEVANT to counting — its duration and recall no
#  longer gate the rep — which is exactly the failure mode v3 suffered.  This
#  also makes per-state dwell unnecessary: short transitions can't be "filtered
#  out" of a count that doesn't depend on them.
#
#  Anchor names are the MERGED vocab produced by train (see MERGE_MAP), e.g.
#  tricep td_top+td_ascending→td_up, so the open anchor is "td_up".
# ══════════════════════════════════════════════════════════════════════════════

_REP_ANCHORS: Dict[str, Tuple[str, str]] = {   # exercise -> (open_anchor, closed_anchor)
    "squat":          ("sq_up",       "sq_bottom"),
    "pushup":         ("pu_up",       "pu_bottom"),
    "situp":          ("su_down",     "su_top"),
    "lateral_raise":  ("lr_down",     "lr_top"),
    "tricep_dip":     ("td_up",       "td_bottom"),
    # now wired up (open = resting posture, closed = peak).  MUST stay identical
    # to REP_ANCHORS in train_dl_v4.py so the FSM sim matches the live counter.
    "pullup":         ("pu_down",     "pu_top"),
    "bicep_curl":     ("bc_down",     "bc_top"),
    "shoulder_press": ("sp_down",     "sp_top"),
    "lunge":          ("lu_up",       "lu_bottom"),
    "jumping_jack":   ("jj_standing", "jj_wide"),
}

_HOLD_TARGET: Dict[str, str] = {
    "plank":    "pk_hold",
    "wall_sit": "ws_hold",
}


# ══════════════════════════════════════════════════════════════════════════════
#  Base feature engineering  (verbatim from train_dl_v3.py — DO NOT modify)
# ══════════════════════════════════════════════════════════════════════════════

_ANGLE_DEFS: List[Tuple[str, int, int, int]] = [
    ("L_Knee",     11, 13, 15),
    ("R_Knee",     12, 14, 16),
    ("L_Hip",       5, 11, 13),
    ("R_Hip",       6, 12, 14),
    ("L_Elbow",     5,  7,  9),
    ("R_Elbow",     6,  8, 10),
    ("L_Shoulder", 11,  5,  7),
    ("R_Shoulder", 12,  6,  8),
]
_N_ANGLES: int = len(_ANGLE_DEFS)   # 8
_N_BASE:   int = _N_ANGLES + 2      # 10


def _is_occluded(pt: np.ndarray) -> bool:
    return float(pt[0]) == 0.0 and float(pt[1]) == 0.0


def _angle_at_vertex(a: np.ndarray, v: np.ndarray, c: np.ndarray) -> float:
    if _is_occluded(a) or _is_occluded(v) or _is_occluded(c):
        return 0.0
    ba = (a - v).astype(np.float64)
    bc = (c - v).astype(np.float64)
    na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if na < 1e-6 or nc < 1e-6:
        return 0.0
    return float(np.degrees(np.arccos(
        float(np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0)))))


def _bbox_aspect_ratio(keypoints: np.ndarray) -> float:
    vis = np.array([kp for kp in keypoints if not _is_occluded(kp)], dtype=np.float32)
    if len(vis) < 2:
        return 0.0
    return float((vis[:, 0].max() - vis[:, 0].min()) /
                 max(vis[:, 1].max() - vis[:, 1].min(), 1e-6))


def _norm_nose_ankle_y(keypoints: np.ndarray) -> float:
    nose = keypoints[0]
    valid_ankles = [a for a in [keypoints[15], keypoints[16]] if not _is_occluded(a)]
    if _is_occluded(nose) or not valid_ankles:
        return 0.0
    ankle_y = float(np.mean([a[1] for a in valid_ankles]))
    vis = np.array([kp for kp in keypoints if not _is_occluded(kp)], dtype=np.float32)
    if len(vis) < 2:
        return 0.0
    return float((ankle_y - float(nose[1])) /
                 max(vis[:, 1].max() - vis[:, 1].min(), 1e-6))


def _calculate_features(keypoints: np.ndarray) -> np.ndarray:
    feat = np.zeros(_N_BASE, dtype=np.float32)
    for i, (_, a, v, c) in enumerate(_ANGLE_DEFS):
        feat[i] = _angle_at_vertex(keypoints[a], keypoints[v], keypoints[c])
    feat[_N_ANGLES]     = _bbox_aspect_ratio(keypoints)
    feat[_N_ANGLES + 1] = _norm_nose_ankle_y(keypoints)
    return feat


# ══════════════════════════════════════════════════════════════════════════════
#  WindowNet  (must match train_dl_v3.WindowNet exactly)
# ══════════════════════════════════════════════════════════════════════════════

class _WindowNet(nn.Module):
    def __init__(self, n_features: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
#  _WindowDeque — rolling window of base feature vectors
#
#  Holds the last `window` base vectors.  flatten() returns (window*10,) in the
#  exact oldest-first layout build_window_matrix() produces:
#      [ base[t-W+1] | ... | base[t-1] | base[t] ]
#  While fewer than `window` frames have been seen, the OLDEST available frame
#  is repeated to left-pad — identical to training's max(src, 0) convention.
# ══════════════════════════════════════════════════════════════════════════════

class _WindowDeque:
    def __init__(self, window: int) -> None:
        self._window = window
        self._buf: Deque[np.ndarray] = deque(maxlen=window)

    def push(self, base: np.ndarray) -> np.ndarray:
        self._buf.append(base.astype(np.float32).copy())
        n = len(self._buf)
        # Left-pad by repeating the oldest frame so the row is always full
        if n < self._window:
            pad = [self._buf[0]] * (self._window - n)
            frames = pad + list(self._buf)
        else:
            frames = list(self._buf)
        return np.concatenate(frames).astype(np.float32)   # (window*10,)

    @property
    def has_any(self) -> bool:
        return len(self._buf) > 0

    def reset(self) -> None:
        self._buf.clear()


# ══════════════════════════════════════════════════════════════════════════════
#  _ExerciseModel — v3 checkpoint loader
# ══════════════════════════════════════════════════════════════════════════════

class _ExerciseModel:
    def __init__(self, checkpoint_path: Path, device: torch.device) -> None:
        self.device = device
        ckpt = torch.load(str(checkpoint_path), map_location=device,
                          weights_only=False)

        required = {"model_state", "scaler_mean", "scaler_std",
                    "state_names", "window_size", "n_features", "num_classes"}
        missing = required - ckpt.keys()
        if missing:
            raise KeyError(
                f"Checkpoint missing keys {missing}. "
                f"Expected a v3 checkpoint from train_dl_v3.py. "
                f"Found: {list(ckpt.keys())}"
            )
        if ckpt.get("version") not in ("dl_v3", "dl_v4"):
            log.warning("Checkpoint version=%s, expected dl_v3/dl_v4.",
                        ckpt.get("version"))

        self.window:      int        = int(ckpt["window_size"])
        self.n_features:  int        = int(ckpt["n_features"])
        self.use_slope:   bool       = bool(ckpt.get("use_slope", False))
        self._mean:       np.ndarray = ckpt["scaler_mean"].astype(np.float32)
        self._std:        np.ndarray = ckpt["scaler_std"].astype(np.float32)
        self.state_names: List[str]  = [str(s) for s in ckpt["state_names"]]
        self.num_classes: int        = int(ckpt["num_classes"])

        expected = self.window * _N_BASE + (_N_BASE if self.use_slope else 0)
        if self.n_features != expected:
            raise ValueError(
                f"n_features={self.n_features} != expected {expected} "
                f"(window={self.window}, use_slope={self.use_slope}). "
                f"Engine and checkpoint are out of sync — retrain or update.")
        if len(self.state_names) != self.num_classes:
            raise ValueError("state_names length != num_classes")

        self._net = _WindowNet(self.n_features, self.num_classes).to(device)
        self._net.load_state_dict(ckpt["model_state"])
        self._net.eval()

        log.info("DLEngine v3: loaded %s  window=%d  states=%s  test_acc=%.3f",
                 checkpoint_path.name, self.window, self.state_names,
                 float(ckpt.get("test_acc", 0.0)))

    def predict(self, window_vec_raw: np.ndarray) -> Tuple[int, str, float]:
        """window_vec_raw: (window*10,) → (local_idx, state_name, confidence)."""
        feat_norm = (window_vec_raw - self._mean) / self._std
        x = torch.tensor(feat_norm, dtype=torch.float32,
                         device=self.device).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self._net(x), dim=-1)[0]
        local_idx  = int(probs.argmax().item())
        confidence = float(probs[local_idx].item())
        return local_idx, self.state_names[local_idx], confidence


# ══════════════════════════════════════════════════════════════════════════════
#  _ModeVoteSmoother  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class _ModeVoteSmoother:
    def __init__(self, window: int = 9) -> None:
        self._buf: Deque[int] = deque(maxlen=window)

    def push(self, raw_idx: int) -> int:
        self._buf.append(raw_idx)
        counts: Dict[int, int] = {}
        for v in self._buf:
            counts[v] = counts.get(v, 0) + 1
        best = max(counts.values())
        for v in reversed(self._buf):
            if counts[v] == best:
                return v
        return raw_idx

    def reset(self) -> None:
        self._buf.clear()


# ══════════════════════════════════════════════════════════════════════════════
#  _RepCounterFSM  (unchanged from v2 engine)
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  _RepCounterFSM  (v4 — anchor hysteresis)
#
#  Counts a rep on a confirmed excursion away from the OPEN anchor and back:
#     armed-at-open ──leave(≥AWAY_DWELL)──► away ──return(≥OPEN_DWELL)──► rep++
#  A rep is validated if EITHER the closed anchor was seen during the excursion
#  OR the excursion lasted ≥ AWAY_MIN frames (guards against open↔mid flicker).
#  COOLDOWN frames debounce double counts.  Transition-state duration/recall do
#  NOT affect the count — that was v3's core failure.
#
#  Defaults run on the smoothed state stream (ModeVote k≈5).  For very fast
#  exercises drop away_dwell to 1; for strict form-gated counting set
#  require_closed=True (rep only counts if the peak posture was actually hit).
# ══════════════════════════════════════════════════════════════════════════════

# ── FPS-AWARE THRESHOLDS ──────────────────────────────────────────────────────
# These were previously fixed frame counts tuned at 30 fps.  On hardware that
# runs the pipeline slower (or faster), a fixed frame count means a DIFFERENT
# wall-clock duration — which caused under-counting once the duplicate-frame bug
# was fixed and the engine began advancing at the true (lower) inference cadence.
#
# They are now expressed in SECONDS.  _RepCounterFSM converts each to a frame
# count ON THE FLY from the live inference fps passed to update(), so a rep means
# the same thing in real time at 12, 30, or 60 fps.  The frame equivalents shown
# are the original @30 fps constants, so default behavior at 30 fps is unchanged.
_REF_FPS: float = 30.0   # cadence the original frame-count thresholds were tuned at

_OPEN_DWELL_SEC:  float = 5  / _REF_FPS   # 0.167 s — confirm rest; rejects brief OPEN flicker (was 5 frames)
_AWAY_DWELL_SEC:  float = 2  / _REF_FPS   # 0.067 s — confirm a departure from OPEN          (was 2 frames)
_COOLDOWN_SEC:    float = 18 / _REF_FPS   # 0.600 s — debounce after a rep                   (was 18 frames)
_AWAY_MIN_SEC:    float = 8  / _REF_FPS   # 0.267 s — min excursion to count without a peak  (was 8 frames)

# ── PER-EXERCISE FSM TIMING (v4.1) ────────────────────────────────────────────
# Rep cadence varies a lot: a jumping-jack rep is ~0.5 s while a sit-up is ~2 s.
# A single global cooldown tuned for slow lifts (0.6 s) silently EATS reps of
# fast exercises — it outlasts a whole jumping-jack rep, so every 2nd rep is
# debounced away.  These per-exercise values are in SECONDS (fps-aware
# downstream).  Fields per exercise:
#     open_dwell, away_dwell, cooldown, away_min  (seconds)
#     require_closed (bool)   — count only if the peak/closed posture was hit
#     smooth         (int)    — ModeVote smoother window for this exercise
# Exercises absent here fall back to the global *_SEC defaults + require_closed
# False + smooth 5.  Slow, already-accurate lifts (pullup / shoulder_press /
# tricep_dip / lateral_raise) keep the original 0.167/0.067/0.60/0.267 timing.
# ── KEEP IN SYNC with _SIM_FSM_TIMING in train_dl_v4.py (frame-count mirror) ──
_FSM_TIMING: Dict[str, Dict[str, float]] = {
    # fast & cyclic — tiny cooldown / dwell, light smoothing
    "jumping_jack":   dict(open_dwell=0.07,  away_dwell=0.03,  cooldown=0.15, away_min=0.10,  require_closed=True,  smooth=3),
    # medium-fast
    "pushup":         dict(open_dwell=0.10,  away_dwell=0.05,  cooldown=0.33, away_min=0.30,  require_closed=True,  smooth=5),
    "bicep_curl":     dict(open_dwell=0.12,  away_dwell=0.05,  cooldown=0.45, away_min=0.30,  require_closed=True,  smooth=5),
    "lunge":          dict(open_dwell=0.12,  away_dwell=0.06,  cooldown=0.45, away_min=0.30,  require_closed=True,  smooth=5),
    "squat":          dict(open_dwell=0.13,  away_dwell=0.06,  cooldown=0.50, away_min=0.35,  require_closed=True,  smooth=5),
    # slow — peak (su_top) recall is low, so DURATION-gate instead of peak-gate
    "situp":          dict(open_dwell=0.15,  away_dwell=0.07,  cooldown=0.55, away_min=0.55,  require_closed=False, smooth=5),
    # slow lifts that already count well — keep original timing
    "pullup":         dict(open_dwell=0.167, away_dwell=0.067, cooldown=0.60, away_min=0.267, require_closed=True,  smooth=5),
    "shoulder_press": dict(open_dwell=0.167, away_dwell=0.067, cooldown=0.60, away_min=0.267, require_closed=True,  smooth=5),
    "tricep_dip":     dict(open_dwell=0.167, away_dwell=0.067, cooldown=0.60, away_min=0.267, require_closed=True,  smooth=5),
    "lateral_raise":  dict(open_dwell=0.167, away_dwell=0.067, cooldown=0.60, away_min=0.267, require_closed=True,  smooth=5),
    # timed holds — rep timing irrelevant; just use a longer smoother
    "plank":          dict(open_dwell=0.167, away_dwell=0.067, cooldown=0.60, away_min=0.267, require_closed=False, smooth=7),
    "wall_sit":       dict(open_dwell=0.167, away_dwell=0.067, cooldown=0.60, away_min=0.267, require_closed=False, smooth=7),
}


def _fsm_timing_for(exercise: str) -> Dict[str, float]:
    """Resolve per-exercise FSM timing (seconds) + require_closed + smoother
    window, falling back to the global defaults for unlisted exercises."""
    base: Dict[str, float] = dict(
        open_dwell=_OPEN_DWELL_SEC, away_dwell=_AWAY_DWELL_SEC,
        cooldown=_COOLDOWN_SEC, away_min=_AWAY_MIN_SEC,
        require_closed=False, smooth=5)
    base.update(_FSM_TIMING.get(exercise, {}))
    return base


_Z_OPEN, _Z_MID, _Z_CLOSED = 0, 1, 2


class _RepCounterFSM:
    def __init__(
        self,
        exercise:        str,
        state_names:     Optional[List[str]] = None,
        open_dwell_sec:  float = _OPEN_DWELL_SEC,
        away_dwell_sec:  float = _AWAY_DWELL_SEC,
        cooldown_sec:    float = _COOLDOWN_SEC,
        away_min_sec:    float = _AWAY_MIN_SEC,
        require_closed:  bool  = False,
        default_fps:     float = _REF_FPS,
    ) -> None:
        self._exercise   = exercise
        self._is_timed   = exercise in _HOLD_TARGET
        self._hold_tgt   = _HOLD_TARGET.get(exercise, "")

        # Resolve anchors against the actual (merged) vocab from the checkpoint.
        self._open: Optional[str] = None
        self._closed: Optional[str] = None
        anchors = _REP_ANCHORS.get(exercise)
        if anchors is not None and state_names:
            o, c = anchors
            if o in state_names and c in state_names:
                self._open, self._closed = o, c
            else:
                log.warning("DLEngine v4: anchors %s not in vocab %s for '%s' "
                            "— rep counting disabled for this exercise.",
                            anchors, state_names, exercise)

        # Thresholds are stored in SECONDS; the _open_dwell / _away_dwell /
        # _cooldown_frames / _away_min PROPERTIES convert them to frame counts
        # against the current fps so _advance() keeps using frame counters as before.
        self._open_dwell_sec  = max(0.0, float(open_dwell_sec))
        self._away_dwell_sec  = max(0.0, float(away_dwell_sec))
        self._cooldown_sec    = max(0.0, float(cooldown_sec))
        self._away_min_sec    = max(0.0, float(away_min_sec))
        self._require_closed  = require_closed

        # Live inference fps drives the seconds→frames conversion.  Seed it with
        # the reference cadence so counting is sane before the fps meter warms up
        # (DualFPSCounter.infer_fps reports 0.0 until it has ≥2 samples).
        self._fps             = max(1.0, float(default_fps))

        self.reps:         int   = 0
        self.hold_seconds: float = 0.0
        self._reset_dynamic()

    # ── seconds → frames, evaluated against the CURRENT inference fps ───────────
    @property
    def _open_dwell(self) -> int:
        return max(1, int(round(self._open_dwell_sec * self._fps)))

    @property
    def _away_dwell(self) -> int:
        return max(1, int(round(self._away_dwell_sec * self._fps)))

    @property
    def _cooldown_frames(self) -> int:
        return max(0, int(round(self._cooldown_sec * self._fps)))

    @property
    def _away_min(self) -> int:
        return max(1, int(round(self._away_min_sec * self._fps)))

    # ── public ───────────────────────────────────────────────────────────────
    def update(self, state: str, fps: Optional[float] = None) -> None:
        # Adopt the live inference fps (when valid) so the seconds-based
        # thresholds convert to the right frame counts for this hardware.
        if fps is not None and fps > 0.0:
            self._fps = float(fps)
        if self._is_timed:
            self._update_hold(state)
            return
        if self._open is None:          # no anchors → cannot count
            return
        self._advance(state)

    def reset(self) -> None:
        self.reps = 0
        self.hold_seconds = 0.0
        self._reset_dynamic()

    # ── internals ──────────────────────────────────────────────────────────────
    def _reset_dynamic(self, keep_hold: bool = False) -> None:
        self._last_zone = -1
        self._zone_run  = 0
        self._armed     = False
        self._away      = False
        self._away_len  = 0
        self._saw_closed = False
        self._cooldown  = 0
        if keep_hold:
            # reset_window() path (tracker dropout): pause the hold timer cleanly
            # instead of zeroing it — bank any in-progress hold into the
            # accumulator, then stop the clock so the gap itself isn't counted.
            # reps and hold_seconds are preserved.
            if self._hold_start != 0.0:
                self._hold_accum += time.monotonic() - self._hold_start
            self._hold_start = 0.0
        else:
            self._hold_start = 0.0
            self._hold_accum = 0.0

    def _zone_of(self, state: str) -> int:
        if state == self._open:
            return _Z_OPEN
        if state == self._closed:
            return _Z_CLOSED
        return _Z_MID

    def _advance(self, state: str) -> None:
        if self._cooldown > 0:
            self._cooldown -= 1

        zone = self._zone_of(state)
        if zone == self._last_zone:
            self._zone_run += 1
        else:
            self._last_zone = zone
            self._zone_run  = 1

        # Phase 0: wait for a confirmed OPEN before arming the counter.
        if not self._armed:
            if zone == _Z_OPEN and self._zone_run >= self._open_dwell:
                self._armed = True
                self._away  = False
            return

        if not self._away:
            # At rest; detect a confirmed departure from OPEN.
            if zone != _Z_OPEN and self._zone_run >= self._away_dwell:
                self._away       = True
                self._away_len   = self._zone_run
                self._saw_closed = (zone == _Z_CLOSED)
        else:
            # Away from OPEN; count only non-OPEN frames so that the return-O
            # dwell frames don't inflate the excursion length and trigger false
            # duration-gate hits on short noise bursts.
            if zone != _Z_OPEN:
                self._away_len += 1
            if zone == _Z_CLOSED:
                self._saw_closed = True
            if zone == _Z_OPEN and self._zone_run >= self._open_dwell:
                valid = self._saw_closed or (
                    not self._require_closed and self._away_len >= self._away_min)
                if valid and self._cooldown == 0:
                    self.reps     += 1
                    self._cooldown = self._cooldown_frames
                self._away       = False
                self._saw_closed = False
                self._away_len   = 0

    def _update_hold(self, state: str) -> None:
        """Cumulative hold timer (unchanged from v3.1): accumulates across brief
        flickers so a single misclassified frame doesn't reset the user's timer."""
        now = time.monotonic()
        if state == self._hold_tgt:
            if self._hold_start == 0.0:
                self._hold_start = now
            self.hold_seconds = self._hold_accum + (now - self._hold_start)
        else:
            if self._hold_start != 0.0:
                self._hold_accum += now - self._hold_start
                self._hold_start  = 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  DLEngine  (public API identical to v2)
# ══════════════════════════════════════════════════════════════════════════════

class DLEngine:
    def __init__(
        self,
        exercise:        str,
        checkpoint_dir:  str = "checkpoints_v3",
        smooth_window:   Optional[int]   = None,
        device:          str = "auto",
        warmup_frames:   int = 0,
        open_dwell_sec:  Optional[float] = None,
        away_dwell_sec:  Optional[float] = None,
        cooldown_sec:    Optional[float] = None,
        away_min_sec:    Optional[float] = None,
        require_closed:  Optional[bool]  = None,
        default_fps:     float = _REF_FPS,
    ) -> None:
        self.exercise = exercise.lower().strip()
        self._loaded  = False
        self._model: Optional[_ExerciseModel] = None
        self._warmup_frames = warmup_frames
        self._frames_seen   = 0

        if device == "auto":
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif (getattr(torch.backends, "mps", None) and
                  torch.backends.mps.is_available()):
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(device)

        ckpt_path = Path(checkpoint_dir) / f"dl_model_{self.exercise}.pt"
        if not ckpt_path.exists():
            available = sorted(p.stem.replace("dl_model_", "")
                               for p in Path(checkpoint_dir).glob("dl_model_*.pt"))
            log.warning("DLEngine v3: checkpoint not found: %s (available: %s)",
                        ckpt_path, available)
            self._window = 12
        else:
            try:
                self._model  = _ExerciseModel(ckpt_path, self._device)
                self._loaded = True
                self._window = self._model.window
            except Exception as exc:
                log.error("DLEngine v3: failed to load %s: %s", ckpt_path, exc)
                self._window = 12

        # Resolve per-exercise FSM timing / gating / smoothing.  An explicit
        # argument (non-None) always wins; otherwise use the per-exercise table
        # (which falls back to the global defaults).  This is what keeps a fast
        # exercise like jumping_jack from being eaten by a slow-lift cooldown.
        _t = _fsm_timing_for(self.exercise)
        if open_dwell_sec is None: open_dwell_sec = _t["open_dwell"]
        if away_dwell_sec is None: away_dwell_sec = _t["away_dwell"]
        if cooldown_sec   is None: cooldown_sec   = _t["cooldown"]
        if away_min_sec   is None: away_min_sec   = _t["away_min"]
        if require_closed is None: require_closed = bool(_t["require_closed"])
        if smooth_window  is None: smooth_window  = int(_t["smooth"])

        self._deque    = _WindowDeque(self._window)
        self._smoother = _ModeVoteSmoother(smooth_window)
        _names         = self._model.state_names if self._model else []
        self._counter  = _RepCounterFSM(
            self.exercise, state_names=_names,
            open_dwell_sec=open_dwell_sec, away_dwell_sec=away_dwell_sec,
            cooldown_sec=cooldown_sec, away_min_sec=away_min_sec,
            require_closed=require_closed, default_fps=default_fps)

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self, keypoints: np.ndarray,
               infer_fps: Optional[float] = None) -> EngineResult:
        if self._model is None:
            return _NOT_READY
        if keypoints is None:
            return _NOT_READY
        if not isinstance(keypoints, np.ndarray) or keypoints.shape != (17, 2):
            return _NOT_READY

        base_feat  = _calculate_features(keypoints)
        window_vec = self._deque.push(base_feat)
        # SLOPE-FEAT (v3.1): mirror build_window_matrix() exactly — mean
        # per-frame velocity across the window, appended after the flat window.
        if self._model.use_slope:
            denom = float(max(self._window - 1, 1))
            slope = (window_vec[-_N_BASE:] - window_vec[:_N_BASE]) / denom
            window_vec = np.concatenate([window_vec, slope]).astype(np.float32)
        self._frames_seen += 1

        # Optional warmup suppression (default 0 → ready immediately via left-pad)
        if self._frames_seen <= self._warmup_frames:
            return EngineResult(
                ready=False, state_name=self._model.state_names[0],
                confidence=0.0, reps=self._counter.reps,
                hold_seconds=self._counter.hold_seconds)

        try:
            raw_idx, _name, confidence = self._model.predict(window_vec)
        except Exception as exc:
            log.warning("DLEngine v3 predict failed: %s", exc)
            return _NOT_READY

        smooth_idx     = self._smoother.push(raw_idx)
        smoothed_state = self._model.state_names[smooth_idx]
        # Pass the live cadence so the FSM's seconds-based thresholds convert to
        # the correct frame counts for this machine.
        self._counter.update(smoothed_state, fps=infer_fps)

        return EngineResult(
            ready=True, state_name=smoothed_state, confidence=confidence,
            reps=self._counter.reps, hold_seconds=self._counter.hold_seconds)

    def reset(self) -> None:
        self._deque.reset()
        self._smoother.reset()
        self._counter.reset()
        self._frames_seen = 0

    def reset_reps(self) -> None:
        self._counter.reset()

    def reset_window(self) -> None:
        """Flush ONLY the temporal context — the sliding window, the ModeVote
        buffer, frames-seen, and the FSM's in-flight excursion state — WITHOUT
        zeroing the user's reps or hold time.

        Call this after a tracker dropout (e.g. tracker state == "searching") so
        that frames captured AFTER the gap aren't concatenated with stale pre-gap
        frames inside the window (that discontinuity produces a garbage slope and
        can spike a false state transition → phantom rep).  reps and hold_seconds
        are preserved; an in-progress hold is paused cleanly rather than reset.
        """
        self._deque.reset()
        self._smoother.reset()
        self._frames_seen = 0
        self._counter._reset_dynamic(keep_hold=True)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def state_names(self) -> List[str]:
        return self._model.state_names if self._model else []

    @property
    def window_size(self) -> int:
        return self._window
