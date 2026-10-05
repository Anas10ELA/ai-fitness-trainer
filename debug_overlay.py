"""
debug_overlay.py
════════════════
Step 6 Debugging Overlays — FSM State + Angle Trend Visualization

Provides three overlay renderers that can be composited onto a CV2 frame:

  1. FSMPanel       — current state, rep count, consec counters, cooldown,
                      EMA angles, amplitude range.  Rendered as a semi-
                      transparent panel in the top-right corner.

  2. AngleTrendPlot — mini rolling time-series chart of up to 3 joints,
                      rendered as a small plot in the bottom-left corner.
                      Shows raw and EMA lines with threshold markers.

  3. VelocityBar    — horizontal velocity bar for the primary joint,
                      coloured green (moving in expected direction) /
                      orange (wrong direction) / grey (stable).

All three are designed to be zero-allocation on the hot path:
  - AngleTrendPlot pre-allocates a fixed-size NumPy canvas once per
    instance and redraws in-place.
  - FSMPanel uses a pre-built colour lookup for state colours.

Usage:
    from debug_overlay import DebugOverlay
    overlay = DebugOverlay(primary_joint="Left Knee", thresholds=(100, 142))
    ...
    overlay.update(angles, counter.debug_snapshot())
    overlay.draw(frame)
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ── Config ────────────────────────────────────────────────────────────────────

HISTORY_LEN = 60          # frames of angle history to show
PLOT_W, PLOT_H = 240, 90  # pixels for the trend plot canvas
PANEL_W = 260             # width of FSM info panel

# State → BGR colour
_STATE_COLOUR = {
    "ready": (180, 180, 180),
    "down":  (60,  80,  220),
    "up":    (60, 200,  60),
}

_FONT      = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SM   = 0.38
_FONT_MED  = 0.45
_LINE_SM   = 1
_LINE_MED  = 1

# Colours (BGR)
_C_WHITE  = (255, 255, 255)
_C_GREY   = (140, 140, 140)
_C_YELLOW = (0,   200, 240)
_C_RED    = (60,   60, 220)
_C_GREEN  = (60,  200,  60)
_C_ORANGE = (30,  140, 240)
_C_BG     = (18,  18,   18)


# ═══════════════════════════════════════════════════════════════════════════════
#  AngleTrendPlot
# ═══════════════════════════════════════════════════════════════════════════════

class AngleTrendPlot:
    """
    Rolling time-series mini-chart for up to 3 joint angles.
    Shows:
      - Raw angle trace (dim line)
      - EMA-smoothed trace (bright line)
      - Horizontal threshold markers (dashed)
    """

    def __init__(
        self,
        joint:      str,
        thresholds: Tuple[float, float],   # (down_threshold, up_threshold)
        angle_range: Tuple[float, float] = (40.0, 200.0),
        colour:     Tuple[int, int, int]  = _C_GREEN,
        label:      str = "",
    ) -> None:
        self._joint      = joint
        self._lo, self._hi = thresholds
        self._amin, self._amax = angle_range
        self._colour     = colour
        self._label      = label or joint

        self._raw_hist:  deque = deque(maxlen=HISTORY_LEN)
        self._ema_hist:  deque = deque(maxlen=HISTORY_LEN)

        # Pre-allocate canvas
        self._canvas = np.zeros((PLOT_H, PLOT_W, 3), dtype=np.uint8)

    def push(self, raw: float, ema: float) -> None:
        if raw > 0:
            self._raw_hist.append(raw)
            self._ema_hist.append(ema)

    def draw(self, frame: np.ndarray, x: int, y: int) -> None:
        """Blit the plot at pixel position (x, y) on `frame`."""
        c = self._canvas
        c[:] = _C_BG

        H, W = PLOT_H - 4, PLOT_W - 4  # usable pixels

        def _to_px(angle: float) -> Tuple[int, int]:
            clamped = max(self._amin, min(self._amax, angle))
            py = int((1.0 - (clamped - self._amin) / (self._amax - self._amin)) * H) + 2
            return py

        # Threshold lines
        for thresh, col in ((self._lo, _C_RED), (self._hi, _C_GREEN)):
            ty = _to_px(thresh)
            for tx in range(2, W + 2, 6):
                cv2.line(c, (tx, ty), (tx + 3, ty), col, 1)

        # Axis label
        cv2.putText(c, self._label, (4, PLOT_H - 4), _FONT, 0.32, _C_GREY, 1, cv2.LINE_AA)

        # Raw trace (dim)
        raw_pts = list(self._raw_hist)
        if len(raw_pts) >= 2:
            xs = np.linspace(2, W + 2, len(raw_pts)).astype(int)
            ys = np.array([_to_px(v) for v in raw_pts])
            for i in range(len(xs) - 1):
                dim = tuple(int(c * 0.45) for c in self._colour)
                cv2.line(c, (xs[i], ys[i]), (xs[i+1], ys[i+1]), dim, 1, cv2.LINE_AA)

        # EMA trace (bright)
        ema_pts = list(self._ema_hist)
        if len(ema_pts) >= 2:
            xs = np.linspace(2, W + 2, len(ema_pts)).astype(int)
            ys = np.array([_to_px(v) for v in ema_pts])
            for i in range(len(xs) - 1):
                cv2.line(c, (xs[i], ys[i]), (xs[i+1], ys[i+1]), self._colour, 2, cv2.LINE_AA)

        # Border
        cv2.rectangle(c, (0, 0), (PLOT_W - 1, PLOT_H - 1), _C_GREY, 1)

        # Blit onto frame with bounds check
        fh, fw = frame.shape[:2]
        x2, y2 = min(x + PLOT_W, fw), min(y + PLOT_H, fh)
        pw, ph = x2 - x, y2 - y
        if pw > 0 and ph > 0:
            roi = frame[y:y2, x:x2]
            alpha_blend(roi, c[:ph, :pw], alpha=0.88)


# ═══════════════════════════════════════════════════════════════════════════════
#  VelocityBar
# ═══════════════════════════════════════════════════════════════════════════════

class VelocityBar:
    """
    Small horizontal bar showing angular velocity of the primary joint.
    Green = moving in expected direction, orange = opposite, grey = stable.
    """

    BAR_W = 140
    BAR_H = 14

    def __init__(self, max_vel: float = 15.0) -> None:
        self._max_vel = max_vel

    def draw(
        self,
        frame: np.ndarray,
        x: int, y: int,
        velocity: float,
        expected_dir: str,  # "falling" or "rising"
    ) -> None:
        fh, fw = frame.shape[:2]
        if x + self.BAR_W > fw or y + self.BAR_H > fh:
            return

        # Background
        cv2.rectangle(frame, (x, y), (x + self.BAR_W, y + self.BAR_H), (40, 40, 40), -1)
        cv2.rectangle(frame, (x, y), (x + self.BAR_W, y + self.BAR_H), _C_GREY, 1)

        # Bar fill
        fill = min(abs(velocity) / self._max_vel, 1.0)
        fill_w = int(fill * (self.BAR_W - 2))
        if fill_w > 0:
            correct = (
                (expected_dir == "falling" and velocity < -0.5) or
                (expected_dir == "rising"  and velocity >  0.5)
            )
            col = _C_GREEN if correct else (_C_ORANGE if abs(velocity) > 0.5 else _C_GREY)
            cv2.rectangle(frame, (x + 1, y + 1), (x + 1 + fill_w, y + self.BAR_H - 1), col, -1)

        # Label
        cv2.putText(
            frame, f"vel {velocity:+.1f}°/f",
            (x + 4, y + self.BAR_H - 3),
            _FONT, 0.30, _C_WHITE, 1, cv2.LINE_AA,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  FSMPanel
# ═══════════════════════════════════════════════════════════════════════════════

class FSMPanel:
    """
    Semi-transparent info panel showing FSM internals.

    Displays:
      STATE:  READY / DOWN / UP   (colour-coded)
      REPS:   N
      Cooldown bar
      Amplitude min/max/range
      EMA angles for up to 4 joints
      Consec counters for active keys
      Rep blocked indicator
    """

    LINE_H = 17  # pixels between lines

    def draw(self, frame: np.ndarray, x: int, y: int, snap: Dict) -> None:
        lines = self._build_lines(snap)
        h = self.LINE_H * (len(lines) + 1) + 8
        w = PANEL_W

        fh, fw = frame.shape[:2]
        x2, y2 = min(x + w, fw), min(y + h, fh)
        if x2 <= x or y2 <= y:
            return

        # Background overlay
        panel = frame[y:y2, x:x2].copy()
        bg    = np.full_like(panel, 20)
        cv2.addWeighted(bg, 0.65, panel, 0.35, 0, panel)
        frame[y:y2, x:x2] = panel

        # State header (colour-coded)
        state     = snap.get("state", "?")
        state_col = _STATE_COLOUR.get(state, _C_WHITE)
        cv2.putText(frame, f"FSM: {state.upper()}",
                    (x + 6, y + 16), _FONT, _FONT_MED, state_col, _LINE_MED, cv2.LINE_AA)

        # Body lines
        for i, (label, value, colour) in enumerate(lines):
            ty = y + 16 + self.LINE_H * (i + 1)
            cv2.putText(frame, f"{label}: {value}",
                        (x + 6, ty), _FONT, _FONT_SM, colour, _LINE_SM, cv2.LINE_AA)

        # Cooldown bar
        cd = snap.get("cooldown_remaining", 0.0)
        if cd > 0:
            bar_x, bar_y = x + 6, y + h - 10
            bar_w = PANEL_W - 12
            fill  = int(min(cd / 0.30, 1.0) * bar_w)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 5), (50, 50, 50), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + 5), _C_ORANGE, -1)
            cv2.putText(frame, f"cooldown {cd:.2f}s",
                        (bar_x, bar_y - 2), _FONT, 0.28, _C_ORANGE, 1, cv2.LINE_AA)

    def _build_lines(self, snap: Dict):
        lines = []

        reps = snap.get("reps", 0)
        lines.append(("REPS", str(reps), _C_YELLOW))

        amp_range = snap.get("amp_range", 0.0)
        amp_min   = snap.get("amp_min")
        amp_max   = snap.get("amp_max", 0.0)
        amp_col   = _C_GREEN if amp_range > 0 else _C_GREY
        lines.append((
            "AMP",
            f"{amp_range:.1f}° ({amp_min or '?'}→{amp_max:.1f})",
            amp_col,
        ))

        if snap.get("rep_blocked"):
            lines.append(("BLOCKED", "YES", _C_RED))

        hold = snap.get("hold_seconds", 0.0)
        if hold > 0:
            lines.append(("HOLD", f"{hold:.1f}s", _C_GREEN))

        ema = snap.get("ema", {})
        for k, v in list(ema.items())[:4]:
            lines.append((k[:14], f"{v}°", _C_WHITE))

        consec = snap.get("consec", {})
        active = [(k, v) for k, v in consec.items() if v > 0]
        for k, v in active[:4]:
            col = _C_GREEN if v >= 2 else _C_GREY
            lines.append((f"  {k[:12]}", str(v), col))

        arm_states = snap.get("arm_states", {})
        if arm_states:
            lines.append((
                "ARMS",
                f"L:{arm_states.get('left','?')[0].upper()} R:{arm_states.get('right','?')[0].upper()}",
                _C_WHITE,
            ))

        return lines


# ═══════════════════════════════════════════════════════════════════════════════
#  DebugOverlay (facade)
# ═══════════════════════════════════════════════════════════════════════════════

class DebugOverlay:
    """
    Composites all three debug layers (FSMPanel, AngleTrendPlot, VelocityBar)
    onto a CV2 BGR frame.

    Usage:
        overlay = DebugOverlay(
            primary_joint  = "Left Knee",
            thresholds     = (100.0, 142.0),
            angle_range    = (50.0, 180.0),
        )
        # per frame:
        overlay.update(angles_dict, counter.debug_snapshot(), ema_dict)
        overlay.draw(frame)
    """

    def __init__(
        self,
        primary_joint: str,
        thresholds:    Tuple[float, float],
        angle_range:   Tuple[float, float] = (40.0, 200.0),
        secondary_joints: Optional[List[str]] = None,
    ) -> None:
        self._primary_joint = primary_joint
        self._thresholds    = thresholds

        self._trend = AngleTrendPlot(
            joint       = primary_joint,
            thresholds  = thresholds,
            angle_range = angle_range,
            colour      = _C_GREEN,
        )
        self._panel   = FSMPanel()
        self._vel_bar = VelocityBar()

        # State for velocity computation
        self._prev_angle: Optional[float] = None
        self._velocity: float = 0.0

        # Current snapshot (set by update())
        self._snap:   Dict = {}
        self._angles: Dict = {}

    def update(
        self,
        angles:   Dict[str, float],
        snapshot: Dict,
        ema_dict: Optional[Dict[str, float]] = None,
    ) -> None:
        self._snap   = snapshot
        self._angles = angles

        raw = angles.get(self._primary_joint, 0.0)
        if raw > 0:
            ema_val = (ema_dict or {}).get(
                self._primary_joint,
                snapshot.get("ema", {}).get(
                    # guess EMA key from joint name
                    self._primary_joint.lower().replace(" ", "_")[:10],
                    raw,
                ),
            )
            if self._prev_angle is not None:
                self._velocity = ema_val - self._prev_angle
            self._prev_angle = ema_val
            self._trend.push(raw, ema_val)

    def draw(self, frame: np.ndarray) -> None:
        fh, fw = frame.shape[:2]

        # FSM Panel — top right
        self._panel.draw(frame, x=fw - PANEL_W - 6, y=6, snap=self._snap)

        # Trend plot — bottom left
        plot_x = 6
        plot_y = fh - PLOT_H - VelocityBar.BAR_H - 20
        self._trend.draw(frame, plot_x, plot_y)

        # Velocity bar — below the plot
        state  = self._snap.get("state", "ready")
        exp_dir = "falling" if state in ("ready", "up") else "rising"
        self._vel_bar.draw(
            frame, x=plot_x, y=plot_y + PLOT_H + 4,
            velocity=self._velocity, expected_dir=exp_dir,
        )


# ── Utility ───────────────────────────────────────────────────────────────────

def alpha_blend(dst: np.ndarray, src: np.ndarray, alpha: float = 0.75) -> None:
    """In-place alpha blend src onto dst (both same shape, BGR uint8)."""
    cv2.addWeighted(src, alpha, dst, 1.0 - alpha, 0, dst)


# ── Convenience: build a DebugOverlay from an exercise registry entry ─────────

def make_debug_overlay(exercise) -> DebugOverlay:
    """
    Factory that reads thresholds from the exercise registry to pre-configure
    the overlay correctly for each exercise.
    """
    from exercises import REGISTRY, Exercise

    reg  = REGISTRY.get(exercise)
    name = exercise.value if hasattr(exercise, "value") else str(exercise)

    # Primary joint and angle range per exercise
    JOINT_MAP = {
        "squat":            ("Left Knee",    (60.0, 180.0)),
        "pushup":           ("Left Elbow",   (60.0, 180.0)),
        "pullup":           ("Left Elbow",   (60.0, 180.0)),
        "bicep_curl":       ("Left Elbow",   (40.0, 170.0)),
        "tricep_dip":       ("Left Elbow",   (60.0, 180.0)),
        "overhead_press":   ("Left Elbow",   (60.0, 180.0)),
        "lateral_raise":    ("L Arm Abduct", (0.0,  120.0)),
        "jumping_jack":     ("L Arm Raise",  (0.0,  160.0)),
        "situp":            ("Back Angle",   (60.0, 180.0)),
        "lunge":            ("Left Knee",    (60.0, 180.0)),
        "burpee":           ("Back Angle",   (60.0, 180.0)),
        "mountain_climber": ("L Hip Angle",  (40.0, 160.0)),
        "plank":            ("Left Hip",     (120.0, 220.0)),
        "wall_sit":         ("Left Knee",    (40.0, 140.0)),
        "high_knees":       ("L Knee Height",(40.0, 160.0)),
    }
    joint, angle_range = JOINT_MAP.get(name, ("Left Knee", (40.0, 200.0)))

    down_t = reg.down_threshold if reg else 100.0
    up_t   = reg.up_threshold   if reg else 150.0

    return DebugOverlay(
        primary_joint = joint,
        thresholds    = (down_t, up_t),
        angle_range   = angle_range,
    )
