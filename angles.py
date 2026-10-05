"""
angles.py — Joint Angle Calculations  [UPDATED v7]

Fixes applied (v7 stabilization):
  CRASH-FIX-6  get_joint_angles() — input shape validation.
               If keypoints arrives with wrong shape (e.g. None, wrong dims,
               wrong number of keypoints), the function now logs a warning and
               returns an empty dict rather than crashing with IndexError.

  CRASH-FIX-7  _knee_height_ratio() — numpy norm on (0,0) subtraction.
               hip == ankle (both (0,0)) gives leg_len == 0 but the existing
               check `leg_len < 10.0` already guards this.  Added an explicit
               np.isfinite check on the result to guard against nan/inf that
               could propagate to the FSM.

  CRASH-FIX-8  _back_angle() — np.mean on a list of numpy arrays.
               np.mean([kp[5], kp[6]]) is safe, but if kp rows contain nan
               from upstream float arithmetic, smid/hmid can be nan.
               Added np.isfinite guard before calling calculate_angle.

  STABILITY-4  calculate_angle() — nan propagation guard.
               Added math.isfinite check on the final result.  If acos
               produces nan (should be impossible after clip, but floating
               point edge cases exist), returns 0.0 instead of propagating nan
               into the FSM.

  LOGGING      Module-level logger.  Invalid input is logged at WARNING; it is
               never silently swallowed.

All threshold values and exercise logic are unchanged from v6.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from exercises import Exercise

log = logging.getLogger(__name__)

Point = Tuple[float, float]

# ── COCO-17 indices ───────────────────────────────────────────────────────────
_NOSE       = 0
_L_SHOULDER = 5;  _R_SHOULDER = 6
_L_ELBOW    = 7;  _R_ELBOW    = 8
_L_WRIST    = 9;  _R_WRIST    = 10
_L_HIP      = 11; _R_HIP      = 12
_L_KNEE     = 13; _R_KNEE     = 14
_L_ANKLE    = 15; _R_ANKLE    = 16

MAX_JUMP_DEG = 35.0

# Expected keypoint array shape
_EXPECTED_SHAPE = (17, 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  Core geometry
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_angle(p1: Point, p2: Point, p3: Point) -> float:
    """Angle at vertex p2 via cosine rule. Returns 0.0 on missing kp (0,0)."""
    for p in (p1, p2, p3):
        if p[0] == 0.0 and p[1] == 0.0:
            return 0.0
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]], dtype=np.float64)
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]], dtype=np.float64)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    result = math.degrees(math.acos(cos_a))
    # STABILITY-4: guard against nan (edge case in float arithmetic)
    return result if math.isfinite(result) else 0.0


def angle_between_vectors(v1: Sequence[float], v2: Sequence[float]) -> float:
    a = np.array(v1, dtype=np.float64)
    b = np.array(v2, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    cos = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    result = math.degrees(math.acos(cos))
    return result if math.isfinite(result) else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Confidence filtering
# ═══════════════════════════════════════════════════════════════════════════════

def filter_keypoints_by_confidence(
    keypoints: np.ndarray,
    scores:    Optional[np.ndarray],
    threshold: float = 0.25,
) -> np.ndarray:
    if scores is None:
        return keypoints
    kp   = keypoints.copy()
    # Guard: scores must be 1-D and same length as keypoints
    if scores.ndim == 1 and len(scores) == len(kp):
        mask = scores < threshold
        kp[mask] = 0.0
    return kp


# ═══════════════════════════════════════════════════════════════════════════════
#  View angle detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_view_angle(kp: np.ndarray) -> str:
    """Returns 'front' | 'side_l' | 'side_r' | 'unknown'."""
    ls_ok = _visible(kp, _L_SHOULDER); rs_ok = _visible(kp, _R_SHOULDER)
    lh_ok = _visible(kp, _L_HIP);     rh_ok = _visible(kp, _R_HIP)

    if ls_ok and rs_ok and lh_ok and rh_ok:
        width  = abs(float(kp[_R_SHOULDER, 0]) - float(kp[_L_SHOULDER, 0]))
        head_y = float(kp[0, 1]) if _visible(kp, 0) else 0.0
        ank_y  = _avg_nz(float(kp[_L_ANKLE, 1]), float(kp[_R_ANKLE, 1]))
        body_h = max(ank_y - head_y, 100.0)
        if width / body_h < 0.08:
            return "side_l" if lh_ok else "side_r"
        return "front"

    if (ls_ok or lh_ok) and not (rs_ok or rh_ok):
        return "side_l"
    if (rs_ok or rh_ok) and not (ls_ok or lh_ok):
        return "side_r"
    return "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
#  Jump-spike guard
# ═══════════════════════════════════════════════════════════════════════════════

def _guarded(name: str, angle: float, prev_angles: Dict[str, float]) -> float:
    """
    Rejects per-frame angle spikes > MAX_JUMP_DEG.
    Also rejects nan/inf values that could corrupt the FSM.
    """
    # STABILITY-4: reject non-finite angles before spike-guard
    if not math.isfinite(angle):
        return 0.0
    prev = prev_angles.get(name, angle)
    if angle == 0.0:
        return 0.0
    if abs(angle - prev) > MAX_JUMP_DEG:
        return prev
    prev_angles[name] = angle
    return angle


def reset_angle_cache() -> None:
    """DEPRECATED — no-op shim."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
#  Exercise-specific angle extraction
# ═══════════════════════════════════════════════════════════════════════════════

def get_joint_angles(
    keypoints:   np.ndarray,
    exercise:    Exercise,
    scores:      Optional[np.ndarray] = None,
    prev_angles: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Compute only the joint angles relevant to *exercise*.
    Returns dict of {display_name: degrees}, or empty dict on invalid input.
    """
    # CRASH-FIX-6: validate input before any indexing
    if keypoints is None:
        log.warning("get_joint_angles: keypoints is None for exercise=%s", exercise.value)
        return {}
    if not isinstance(keypoints, np.ndarray):
        log.warning(
            "get_joint_angles: keypoints is %s, expected np.ndarray for exercise=%s",
            type(keypoints).__name__, exercise.value,
        )
        return {}
    if keypoints.shape != _EXPECTED_SHAPE:
        log.warning(
            "get_joint_angles: keypoints shape %s != expected %s for exercise=%s",
            keypoints.shape, _EXPECTED_SHAPE, exercise.value,
        )
        return {}

    if prev_angles is None:
        prev_angles = {}

    kp   = filter_keypoints_by_confidence(keypoints, scores)
    view = detect_view_angle(kp)
    angles: Dict[str, float] = {}

    def xy(i: int) -> Point:
        return (float(kp[i, 0]), float(kp[i, 1]))

    def ga(name: str, p1: int, p2: int, p3: int) -> float:
        return _guarded(name, calculate_angle(xy(p1), xy(p2), xy(p3)), prev_angles)

    def bilateral(
        name_l: str, name_r: str,
        p1l: int, p2l: int, p3l: int,
        p1r: int, p2r: int, p3r: int,
    ) -> Tuple[float, float]:
        al = ga(name_l, p1l, p2l, p3l)
        ar = ga(name_r, p1r, p2r, p3r)
        if view == "side_l":   return al, al
        if view == "side_r":   return ar, ar
        return al, ar

    # ── SQUAT ─────────────────────────────────────────────────────────────────
    if exercise == Exercise.SQUAT:
        lk, rk = bilateral("sq_knee_l", "sq_knee_r",
                           _L_HIP, _L_KNEE, _L_ANKLE, _R_HIP, _R_KNEE, _R_ANKLE)
        angles["Left Knee"]  = lk
        angles["Right Knee"] = rk
        angles["Back"]       = _guarded("sq_back", _back_angle(kp), prev_angles)

    # ── PUSH-UP ───────────────────────────────────────────────────────────────
    elif exercise == Exercise.PUSHUP:
        le, re = bilateral("pu_el_l", "pu_el_r",
                           _L_SHOULDER, _L_ELBOW, _L_WRIST,
                           _R_SHOULDER, _R_ELBOW, _R_WRIST)
        lh, rh = bilateral("pu_hip_l", "pu_hip_r",
                           _L_SHOULDER, _L_HIP, _L_KNEE,
                           _R_SHOULDER, _R_HIP, _R_KNEE)
        angles["Left Elbow"]  = le; angles["Right Elbow"] = re
        angles["Left Hip"]    = lh; angles["Right Hip"]   = rh

    # ── JUMPING JACK ──────────────────────────────────────────────────────────
    elif exercise == Exercise.JUMPING_JACK:
        angles["L Arm Raise"] = ga("jj_arm_l", _L_HIP, _L_SHOULDER, _L_WRIST)
        angles["R Arm Raise"] = ga("jj_arm_r", _R_HIP, _R_SHOULDER, _R_WRIST)

    # ── HIGH KNEES ────────────────────────────────────────────────────────────
    elif exercise == Exercise.HIGH_KNEES:
        angles["L Knee Height"] = ga("hk_l", _L_SHOULDER, _L_HIP, _L_KNEE)
        angles["R Knee Height"] = ga("hk_r", _R_SHOULDER, _R_HIP, _R_KNEE)
        angles["L Knee Ratio"]  = _knee_height_ratio(kp, "left")
        angles["R Knee Ratio"]  = _knee_height_ratio(kp, "right")

    # ── PLANK ─────────────────────────────────────────────────────────────────
    elif exercise == Exercise.PLANK:
        lh = ga("pl_hip_l", _L_SHOULDER, _L_HIP, _L_KNEE)
        rh = ga("pl_hip_r", _R_SHOULDER, _R_HIP, _R_KNEE)
        if view == "side_l":   lh = rh = lh
        elif view == "side_r": lh = rh = rh
        angles["Left Hip"]   = lh; angles["Right Hip"] = rh
        angles["Body Align"] = _avg_nz(lh, rh)

    # ── PULL-UP ───────────────────────────────────────────────────────────────
    elif exercise == Exercise.PULLUP:
        le, re = bilateral("pup_el_l", "pup_el_r",
                           _L_SHOULDER, _L_ELBOW, _L_WRIST,
                           _R_SHOULDER, _R_ELBOW, _R_WRIST)
        angles["Left Elbow"]  = le; angles["Right Elbow"] = re

    # ── SIT-UP ────────────────────────────────────────────────────────────────
    elif exercise == Exercise.SITUP:
        angles["Back Angle"] = _guarded("su_back", _back_angle(kp), prev_angles)

    # ── LUNGE ─────────────────────────────────────────────────────────────────
    elif exercise == Exercise.LUNGE:
        lk, rk = bilateral("lg_knee_l", "lg_knee_r",
                           _L_HIP, _L_KNEE, _L_ANKLE, _R_HIP, _R_KNEE, _R_ANKLE)
        angles["Left Knee"]  = lk; angles["Right Knee"] = rk
        angles["Back"]       = _guarded("lg_back", _back_angle(kp), prev_angles)

    # ── MOUNTAIN CLIMBER ──────────────────────────────────────────────────────
    elif exercise == Exercise.MOUNTAIN_CLIMBER:
        lh = ga("mc_hip_l", _L_SHOULDER, _L_HIP, _L_KNEE)
        rh = ga("mc_hip_r", _R_SHOULDER, _R_HIP, _R_KNEE)
        angles["L Hip Angle"] = lh; angles["R Hip Angle"] = rh
        angles["Body Align"]  = _avg_nz(lh, rh)

    # ── BURPEE ────────────────────────────────────────────────────────────────
    elif exercise == Exercise.BURPEE:
        lk, rk = bilateral("bur_knee_l", "bur_knee_r",
                           _L_HIP, _L_KNEE, _L_ANKLE, _R_HIP, _R_KNEE, _R_ANKLE)
        angles["Left Knee"]  = lk; angles["Right Knee"] = rk
        angles["Back Angle"] = _guarded("bur_back", _back_angle(kp), prev_angles)

    # ── BICEP CURL ────────────────────────────────────────────────────────────
    elif exercise == Exercise.BICEP_CURL:
        le, re = bilateral("bc_el_l", "bc_el_r",
                           _L_SHOULDER, _L_ELBOW, _L_WRIST,
                           _R_SHOULDER, _R_ELBOW, _R_WRIST)
        angles["Left Elbow"]  = le; angles["Right Elbow"] = re
        angles["Back"]        = _guarded("bc_back", _back_angle(kp), prev_angles)

    # ── TRICEP DIP ────────────────────────────────────────────────────────────
    elif exercise == Exercise.TRICEP_DIP:
        le, re = bilateral("td_el_l", "td_el_r",
                           _L_SHOULDER, _L_ELBOW, _L_WRIST,
                           _R_SHOULDER, _R_ELBOW, _R_WRIST)
        angles["Left Elbow"]  = le; angles["Right Elbow"] = re
        angles["Back"]        = _guarded("td_back", _back_angle(kp), prev_angles)

    # ── LATERAL RAISE ─────────────────────────────────────────────────────────
    elif exercise == Exercise.LATERAL_RAISE:
        la = ga("lr_abd_l", _L_HIP, _L_SHOULDER, _L_ELBOW)
        ra = ga("lr_abd_r", _R_HIP, _R_SHOULDER, _R_ELBOW)
        if view == "side_l":   la = ra = la
        elif view == "side_r": la = ra = ra
        angles["L Arm Abduct"] = la; angles["R Arm Abduct"] = ra

        ls, rs = bilateral("lr_str_l", "lr_str_r",
                           _L_SHOULDER, _L_ELBOW, _L_WRIST,
                           _R_SHOULDER, _R_ELBOW, _R_WRIST)
        angles["L Arm Straight"] = ls; angles["R Arm Straight"] = rs

    # ── OVERHEAD PRESS ────────────────────────────────────────────────────────
    elif exercise == Exercise.OVERHEAD_PRESS:
        le, re = bilateral("op_el_l", "op_el_r",
                           _L_SHOULDER, _L_ELBOW, _L_WRIST,
                           _R_SHOULDER, _R_ELBOW, _R_WRIST)
        angles["Left Elbow"]  = le; angles["Right Elbow"] = re
        angles["Back"]        = _guarded("op_back", _back_angle(kp), prev_angles)

    # ── WALL SIT ─────────────────────────────────────────────────────────────
    elif exercise == Exercise.WALL_SIT:
        lk, rk = bilateral("ws_knee_l", "ws_knee_r",
                           _L_HIP, _L_KNEE, _L_ANKLE, _R_HIP, _R_KNEE, _R_ANKLE)
        angles["Left Knee"]  = lk; angles["Right Knee"] = rk
        angles["Back"]       = _guarded("ws_back", _back_angle(kp), prev_angles)

    return angles


# ═══════════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _back_angle(kp: np.ndarray) -> float:
    """
    Torso inclination vs vertical.  ~180° = upright, ~90° = horizontal.
    Side-view aware: uses whichever shoulder/hip points are visible.
    """
    ls, rs = kp[_L_SHOULDER], kp[_R_SHOULDER]
    lh, rh = kp[_L_HIP],      kp[_R_HIP]
    s_pts = [p for p in (ls, rs) if not (p[0] == 0 and p[1] == 0)]
    h_pts = [p for p in (lh, rh) if not (p[0] == 0 and p[1] == 0)]
    if not s_pts or not h_pts:
        return 0.0
    smid = np.mean(s_pts, axis=0)
    hmid = np.mean(h_pts, axis=0)

    # CRASH-FIX-8: guard against nan in midpoints
    if not (np.isfinite(smid).all() and np.isfinite(hmid).all()):
        return 0.0

    return calculate_angle(
        (float(smid[0]), float(smid[1])),
        (float(hmid[0]), float(hmid[1])),
        (float(hmid[0]), float(hmid[1]) + 100.0),
    )


def _knee_height_ratio(kp: np.ndarray, side: str) -> float:
    """
    Normalised knee height ratio = (hip.y - knee.y) / leg_length.
    Positive = knee above hip.  Camera-distance invariant.
    """
    if side == "left":
        hip_i, knee_i, ankle_i = _L_HIP, _L_KNEE, _L_ANKLE
    else:
        hip_i, knee_i, ankle_i = _R_HIP, _R_KNEE, _R_ANKLE

    hip = kp[hip_i]; knee = kp[knee_i]; ankle = kp[ankle_i]
    for p in (hip, knee, ankle):
        if p[0] == 0 and p[1] == 0:
            return 0.0

    leg_len = float(np.linalg.norm(hip - ankle))
    if leg_len < 10.0:
        return 0.0

    result = (float(hip[1]) - float(knee[1])) / leg_len
    # CRASH-FIX-7: guard against nan/inf ratio
    return result if math.isfinite(result) else 0.0


def _visible(kp: np.ndarray, idx: int) -> bool:
    return not (kp[idx, 0] == 0.0 and kp[idx, 1] == 0.0)


def _avg_nz(*values: float) -> float:
    valid = [v for v in values if v > 0.0]
    return sum(valid) / len(valid) if valid else 0.0
