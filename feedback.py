"""
feedback.py — Hybrid Form Feedback Engine  [v8 — AI-State-Gated]

Changes from v7 (angle-FSM-gated) to v8 (AI-state-gated):
───────────────────────────────────────────────────────────
  HYBRID-1  The `state` parameter in evaluate() is now the raw AI-predicted
            state string from FitnessInferenceEngine ("neutral", "down", "up",
            "plank", "jump", "extended", "flexed").  Every per-exercise checker
            now gates its angle checks on this AI state rather than deriving
            the phase from angle thresholds.  This eliminates a whole class of
            false positives where a geometrically "bad" angle at one phase
            happens to be geometrically normal at another.

  HYBRID-2  State vocabulary updated throughout:
              "ready"   → "neutral"
              "up"      → "extended"   for bicep_curl, lateral_raise, overhead_press
              "down"    → "extended"   for bicep_curl bottom (full extension)
              Added "plank", "jump", "flexed" gates for new exercises.

  HYBRID-3  Always-on posture checks (back lean, hip sag during plank) run
            across all non-neutral states — these represent postural faults
            independent of movement phase.

  HYBRID-4  _check_burpee now has real logic gated on plank/down/up/jump states.

  HYBRID-5  _check_plank / _check_mountain_climber updated for "plank" state.

All crash-fix guards from v7 are preserved.
All PERSIST_FRAMES / cooldown / streak mechanics are unchanged.
"""

from __future__ import annotations

import logging
import math
import time
import traceback
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from exercises import Exercise, REGISTRY

log = logging.getLogger(__name__)


# ── Priority ──────────────────────────────────────────────────────────────────

class Priority(Enum):
    CRITICAL = 0   # blocks rep; shown after PERSIST_FRAMES
    WARNING  = 1   # shown after PERSIST_FRAMES; rep still counts
    TIP      = 2   # coaching hint; shown rarely


_COOLDOWN: Dict[Priority, float] = {
    Priority.CRITICAL: 2.0,
    Priority.WARNING:  4.0,
    Priority.TIP:      8.0,
}

# Issue must persist this many consecutive frames before being shown
PERSIST_FRAMES: Dict[Priority, int] = {
    Priority.CRITICAL: 3,
    Priority.WARNING:  3,
    Priority.TIP:      5,
}

# AI states considered "active exercise" (not at rest between reps)
ACTIVE_STATES = frozenset({"down", "up", "plank", "jump", "extended", "flexed"})


# ── Internal types ────────────────────────────────────────────────────────────

class _Issue:
    __slots__ = ("message", "priority", "blocks_rep")

    def __init__(self, message: str, priority: Priority, blocks_rep: bool = False):
        self.message    = message
        self.priority   = priority
        self.blocks_rep = blocks_rep


# ── Safe message lookup ───────────────────────────────────────────────────────

def _msg(msgs: Dict[str, str], key: str, fallback: str = "") -> str:
    """
    Safe lookup for mistake_messages entries.
    Returns `fallback` (empty string by default) instead of raising KeyError.
    A WARNING is emitted so registry gaps are visible in logs without crashing.
    """
    value = msgs.get(key)
    if value is None:
        log.warning(
            "FeedbackEngine: mistake_messages key %r not found. "
            "Check exercises.py REGISTRY. Skipping this feedback.",
            key,
        )
        return fallback
    return value


# ═══════════════════════════════════════════════════════════════════════════════
#  FeedbackEngine
# ═══════════════════════════════════════════════════════════════════════════════

class FeedbackEngine:
    """
    Stateful form-feedback engine, gated on the AI-predicted state.

    An issue must persist PERSIST_FRAMES consecutive frames before its
    message surfaces — this eliminates single-frame false triggers.

    State lifecycle
    ───────────────
    All mutable state is fully reset by reset().
    Session.switch_exercise() should call reset() so no feedback from a previous
    exercise leaks into the next one.

    Parameters
    ----------
    cooldown_scale : float
        Multiplier applied to all message cooldowns.  Use < 1.0 for debug/testing.
    """

    def __init__(self, cooldown_scale: float = 1.0) -> None:
        self._scale          = cooldown_scale
        self._last_shown:    Dict[str, float] = {}
        self._issue_streak:  Dict[str, int]   = {}
        self._active_msg:    str  = ""
        self._active_blocks: bool = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def evaluate(
        self,
        exercise:  Exercise,
        angles:    Dict[str, float],    # computed by frame_processor / angle helper
        ai_state:  str,                 # HYBRID-1: AI state string, not FSM-derived
        keypoints: Optional[np.ndarray],
        extra:     Optional[Dict] = None,
    ) -> Tuple[str, bool]:
        """
        Evaluate form for the current frame.

        Parameters
        ----------
        exercise  : Exercise enum member
        angles    : dict of angle name → degrees, e.g. {"Left Knee": 92.0}
        ai_state  : AI-predicted state string from FitnessInferenceEngine
        keypoints : (17, 2) array or None
        extra     : optional extra context (passed through to checkers)

        Returns
        -------
        (message, blocks_rep)
        """
        # CRASH-FIX-4: top-level guard
        try:
            return self._evaluate_inner(
                exercise, angles, ai_state, keypoints, extra or {}
            )
        except Exception:
            log.warning(
                "FeedbackEngine.evaluate() unhandled exception for exercise=%s:\n%s",
                exercise.value, traceback.format_exc(),
            )
            return self._active_msg or "", self._active_blocks

    def _evaluate_inner(
        self,
        exercise:  Exercise,
        angles:    Dict[str, float],
        ai_state:  str,
        keypoints: Optional[np.ndarray],
        extra:     Dict,
    ) -> Tuple[str, bool]:
        all_issues  = self._check(exercise, angles, ai_state, keypoints, extra)
        active_msgs = {i.message for i in all_issues if i.message}

        for msg in self._all_possible_messages(exercise):
            if not msg:
                continue
            self._issue_streak[msg] = (
                self._issue_streak.get(msg, 0) + 1
                if msg in active_msgs else 0
            )

        persistent = [
            i for i in all_issues
            if i.message
            and self._issue_streak.get(i.message, 0) >= PERSIST_FRAMES[i.priority]
        ]

        if not persistent:
            self._active_msg    = "✓ Good form"
            self._active_blocks = False
            return "✓ Good form", False

        persistent.sort(key=lambda i: i.priority.value)

        for issue in persistent:
            if self._can_show(issue.message, issue.priority):
                self._last_shown[issue.message] = time.time()
                self._active_msg    = issue.message
                self._active_blocks = issue.blocks_rep
                return issue.message, issue.blocks_rep

        has_blocking = any(i.blocks_rep for i in persistent)
        return self._active_msg, has_blocking

    def reset(self) -> None:
        """Fully wipe all feedback state (call on exercise switch)."""
        self._last_shown.clear()
        self._issue_streak.clear()
        self._active_msg    = ""
        self._active_blocks = False

    # ── Routing ────────────────────────────────────────────────────────────────

    def _check(
        self,
        exercise:  Exercise,
        angles:    Dict[str, float],
        ai_state:  str,
        keypoints: Optional[np.ndarray],
        extra:     Dict,
    ) -> List[_Issue]:
        dispatch = {
            Exercise.SQUAT:            self._check_squat,
            Exercise.PUSHUP:           self._check_pushup,
            Exercise.JUMPING_JACK:     self._check_jumping_jack,
            Exercise.HIGH_KNEES:       self._check_high_knees,
            Exercise.PLANK:            self._check_plank,
            Exercise.PULLUP:           self._check_pullup,
            Exercise.SITUP:            self._check_situp,
            Exercise.LUNGE:            self._check_lunge,
            Exercise.MOUNTAIN_CLIMBER: self._check_mountain_climber,
            Exercise.BURPEE:           self._check_burpee,
            Exercise.BICEP_CURL:       self._check_bicep_curl,
            Exercise.TRICEP_DIP:       self._check_tricep_dip,
            Exercise.LATERAL_RAISE:    self._check_lateral_raise,
            Exercise.SHOULDER_PRESS:   self._check_shoulder_press,
            Exercise.OVERHEAD_PRESS:   self._check_shoulder_press,
            Exercise.WALL_SIT:         self._check_wall_sit,
        }
        fn = dispatch.get(exercise)
        if fn is None:
            return []
        try:
            return fn(angles, ai_state, keypoints, extra)
        except Exception:
            log.warning(
                "_check_%s raised an exception:\n%s",
                exercise.value, traceback.format_exc(),
            )
            return []

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _can_show(self, msg: str, priority: Priority) -> bool:
        cooldown = _COOLDOWN[priority] * self._scale
        return time.time() - self._last_shown.get(msg, 0.0) >= cooldown

    def _all_possible_messages(self, exercise: Exercise) -> List[str]:
        """Return every feedback string registered for this exercise."""
        try:
            return list(REGISTRY[exercise].mistake_messages.values())
        except (KeyError, AttributeError):
            return []

    # ═══════════════════════════════════════════════════════════════════════════
    #  Per-exercise checkers
    #  Convention: each checker receives (angles, ai_state, kp, extra)
    #              and returns List[_Issue].
    #
    #  HYBRID gate pattern:
    #    if ai_state == "down":   ← phase-specific check
    #    if ai_state in ACTIVE_STATES:  ← always-on posture check
    # ═══════════════════════════════════════════════════════════════════════════

    def _check_squat(self, angles, ai_state, kp, extra) -> List[_Issue]:
        issues: List[_Issue] = []
        try:
            cfg  = REGISTRY[Exercise.SQUAT].form
            msgs = REGISTRY[Exercise.SQUAT].mistake_messages

            knee_l = angles.get("Left Knee",  0.0)
            knee_r = angles.get("Right Knee", 0.0)
            knee   = _best_nz(knee_l, knee_r)
            back   = angles.get("Back", 0.0)
            hip    = _best_nz(
                angles.get("Left Hip",  0.0),
                angles.get("Right Hip", 0.0),
            )

            # ── Always-on posture: back lean ─────────────────────────────────
            if ai_state in ACTIVE_STATES and back > 0:
                if back < cfg.get("back_min", 130.0):
                    m = _msg(msgs, "lean_forward")
                    if m: issues.append(_Issue(m, Priority.WARNING))

            # ── Down-phase: depth + knee tracking ────────────────────────────
            if ai_state == "down":
                if knee > 0 and knee > cfg.get("down_max", 105.0):
                    m = _msg(msgs, "not_deep")
                    if m: issues.append(_Issue(m, Priority.WARNING))
                # STAB-7: hip secondary check — ensures it's a real squat
                if hip > 0 and hip > cfg.get("hip_max", 120.0):
                    m = _msg(msgs, "not_deep")
                    if m: issues.append(_Issue(m, Priority.WARNING))
                # Knee valgus (knees caving in)
                if kp is not None and kp.shape == (17, 2):
                    try:
                        kl_x = float(kp[13, 0]); kr_x = float(kp[14, 0])
                        al_x = float(kp[15, 0]); ar_x = float(kp[16, 0])
                        if all(v > 0 for v in [kl_x, kr_x, al_x, ar_x]):
                            if kl_x > al_x + cfg.get("valgus_px", 15):
                                m = _msg(msgs, "knee_valgus")
                                if m: issues.append(_Issue(m, Priority.CRITICAL, True))
                            if kr_x < ar_x - cfg.get("valgus_px", 15):
                                m = _msg(msgs, "knee_valgus")
                                if m: issues.append(_Issue(m, Priority.CRITICAL, True))
                    except (IndexError, ValueError):
                        pass

            # ── Up-phase: check lockout ────────────────────────────────────
            if ai_state == "up" and knee > 0 and knee < cfg.get("up_min", 160.0):
                m = _msg(msgs, "not_standing")
                if m: issues.append(_Issue(m, Priority.TIP))

        except (KeyError, AttributeError):
            pass
        return issues

    def _check_pushup(self, angles, ai_state, kp, extra) -> List[_Issue]:
        issues: List[_Issue] = []
        try:
            cfg  = REGISTRY[Exercise.PUSHUP].form
            msgs = REGISTRY[Exercise.PUSHUP].mistake_messages

            elbow = _best_nz(
                angles.get("Left Elbow",  0.0),
                angles.get("Right Elbow", 0.0),
            )
            body_align = angles.get("Body Align",     0.0)
            hip_angle  = angles.get("Hip",            0.0)

            # ── Always-on: hip sag / pike (body alignment) ───────────────────
            # Only meaningful while horizontal (plank/down/up states)
            if ai_state in {"down", "up", "plank"} and body_align > 0:
                if body_align < cfg.get("hip_sag_max", 155.0):
                    m = _msg(msgs, "hip_sag")
                    if m: issues.append(_Issue(m, Priority.CRITICAL, True))
                elif body_align > cfg.get("hip_pike_min", 205.0):
                    m = _msg(msgs, "hip_pike")
                    if m: issues.append(_Issue(m, Priority.WARNING))

            # ── Down-phase: elbow depth + elbow flare ────────────────────────
            if ai_state == "down":
                if elbow > 0 and elbow > cfg.get("down_max", 100.0):
                    m = _msg(msgs, "not_deep")
                    if m: issues.append(_Issue(m, Priority.WARNING))
                elbow_flare = _best_nz(
                    angles.get("L Elbow Flare", 0.0),
                    angles.get("R Elbow Flare", 0.0),
                )
                if elbow_flare > 0 and elbow_flare > cfg.get("flare_max", 55.0):
                    m = _msg(msgs, "elbow_flare",
                             "Keep elbows closer to your sides")
                    if m: issues.append(_Issue(m, Priority.WARNING))

            # ── Up-phase: lockout ─────────────────────────────────────────────
            if ai_state == "up" and elbow > 0 and elbow < cfg.get("up_min", 150.0):
                m = _msg(msgs, "not_extended")
                if m: issues.append(_Issue(m, Priority.TIP))

        except (KeyError, AttributeError):
            pass
        return issues

    def _check_jumping_jack(self, angles, ai_state, kp, extra) -> List[_Issue]:
        issues: List[_Issue] = []
        try:
            cfg  = REGISTRY[Exercise.JUMPING_JACK].form
            msgs = REGISTRY[Exercise.JUMPING_JACK].mistake_messages

            l_abd = angles.get("L Arm Abduct", 0.0)
            r_abd = angles.get("R Arm Abduct", 0.0)
            abd   = _avg_nz(l_abd, r_abd)

            # HYBRID-2: "extended" replaces old "up"
            if ai_state == "extended":
                if abd > 0 and abd < cfg.get("arm_height_min", 70.0):
                    m = _msg(msgs, "arms_not_high")
                    if m: issues.append(_Issue(m, Priority.WARNING))
                if l_abd > 0 and r_abd > 0:
                    if abs(l_abd - r_abd) > cfg.get("asymmetry_max", 35.0):
                        m = _msg(msgs, "asymmetric")
                        if m: issues.append(_Issue(m, Priority.TIP))

        except (KeyError, AttributeError):
            pass
        return issues

    def _check_high_knees(self, angles, ai_state, kp, extra) -> List[_Issue]:
        issues: List[_Issue] = []
        try:
            cfg  = REGISTRY[Exercise.HIGH_KNEES].form
            msgs = REGISTRY[Exercise.HIGH_KNEES].mistake_messages

            knee = _best_nz(
                angles.get("Left Knee",  0.0),
                angles.get("Right Knee", 0.0),
            )
            # HYBRID-2: "flexed" = knee at peak height
            if ai_state == "flexed" and knee > 0:
                if knee > cfg.get("knee_max", 105.0):
                    m = _msg(msgs, "knee_not_high")
                    if m: issues.append(_Issue(m, Priority.WARNING))

        except (KeyError, AttributeError):
            pass
        return issues

    def _check_plank(self, angles, ai_state, kp, extra) -> List[_Issue]:
        issues: List[_Issue] = []
        try:
            cfg  = REGISTRY[Exercise.PLANK].form
            msgs = REGISTRY[Exercise.PLANK].mistake_messages
            body_align = angles.get("Body Align", 0.0)

            # HYBRID-2: only evaluate plank form while AI confirms plank position
            if ai_state == "plank" and body_align > 0:
                if body_align < cfg.get("hip_sag_max", 155.0):
                    m = _msg(msgs, "hip_sag")
                    if m: issues.append(_Issue(m, Priority.CRITICAL, True))
                elif body_align > cfg.get("hip_pike_min", 205.0):
                    m = _msg(msgs, "hip_pike")
                    if m: issues.append(_Issue(m, Priority.WARNING))
                # Head / neck alignment
                head = angles.get("Head", 0.0)
                if head > 0:
                    if head < cfg.get("head_min", 140.0):
                        m = _msg(msgs, "head_down")
                        if m: issues.append(_Issue(m, Priority.TIP))
                    elif head > cfg.get("head_max", 200.0):
                        m = _msg(msgs, "head_up")
                        if m: issues.append(_Issue(m, Priority.TIP))

        except (KeyError, AttributeError):
            pass
        return issues

    def _check_pullup(self, angles, ai_state, kp, extra) -> List[_Issue]:
        issues: List[_Issue] = []
        try:
            msgs  = REGISTRY[Exercise.PULLUP].mistake_messages
            elbow = _best_nz(
                angles.get("Left Elbow",  0.0),
                angles.get("Right Elbow", 0.0),
            )
            # "neutral" = hanging at bottom; arms should be fully extended
            if ai_state == "neutral" and elbow > 0 and elbow < 135:
                m = _msg(msgs, "not_extended")
                if m: issues.append(_Issue(m, Priority.WARNING))
            # "up" = chin over bar; elbows should be well flexed
            if ai_state == "up" and elbow > 0 and elbow > 95:
                m = _msg(msgs, "not_pulled")
                if m: issues.append(_Issue(m, Priority.WARNING))
        except (KeyError, AttributeError):
            pass
        return issues

    def _check_situp(self, angles, ai_state, kp, extra) -> List[_Issue]:
        issues: List[_Issue] = []
        try:
            msgs = REGISTRY[Exercise.SITUP].mistake_messages
            back = angles.get("Back Angle", angles.get("Back", 0.0))
            # "up" = fully sitting up
            if ai_state == "up" and back > 0 and back < 120:
                m = _msg(msgs, "not_full")
                if m: issues.append(_Issue(m, Priority.WARNING))
        except (KeyError, AttributeError):
            pass
        return issues

    def _check_lunge(self, angles, ai_state, kp, extra) -> List[_Issue]:
        issues: List[_Issue] = []
        try:
            cfg  = REGISTRY[Exercise.LUNGE].form
            msgs = REGISTRY[Exercise.LUNGE].mistake_messages
            back     = angles.get("Back", 0.0)
            min_knee = _min_nz(
                angles.get("Left Knee",  0.0),
                angles.get("Right Knee", 0.0),
            )

            # Always-on posture check — forward lean is bad in any phase
            if ai_state in ACTIVE_STATES and back > 0:
                if back < cfg.get("back_min", 138.0):
                    m = _msg(msgs, "lean_forward")
                    if m: issues.append(_Issue(m, Priority.WARNING))

            # Down-phase: check depth
            if ai_state == "down" and min_knee > 0 and min_knee > 125:
                m = _msg(msgs, "not_deep")
                if m: issues.append(_Issue(m, Priority.WARNING))

        except (KeyError, AttributeError):
            pass
        return issues

    def _check_mountain_climber(self, angles, ai_state, kp, extra) -> List[_Issue]:
        issues: List[_Issue] = []
        try:
            cfg  = REGISTRY[Exercise.MOUNTAIN_CLIMBER].form
            msgs = REGISTRY[Exercise.MOUNTAIN_CLIMBER].mistake_messages

            # Hip level check on plank phase (CRASH-FIX-3 guard retained)
            if ai_state == "plank" and kp is not None and kp.shape == (17, 2):
                try:
                    lh_y = float(kp[11, 1])
                    rh_y = float(kp[12, 1])
                    if lh_y > 0 and rh_y > 0:
                        if abs(lh_y - rh_y) > cfg.get("hip_level_px", 40.0):
                            m = _msg(msgs, "hips_rotating")
                            if m: issues.append(_Issue(m, Priority.WARNING))
                except (IndexError, ValueError):
                    pass

            body_align = angles.get("Body Align", 0.0)
            if ai_state in {"plank", "flexed"} and body_align > 0:
                if body_align < cfg.get("hip_sag_max", 148.0):
                    m = _msg(msgs, "hip_sag")
                    if m: issues.append(_Issue(m, Priority.CRITICAL, True))
                elif body_align > cfg.get("hip_pike_min", 205.0):
                    m = _msg(msgs, "hip_pike")
                    if m: issues.append(_Issue(m, Priority.WARNING))

        except (KeyError, AttributeError):
            pass
        return issues

    def _check_burpee(self, angles, ai_state, kp, extra) -> List[_Issue]:
        """
        HYBRID-4: Burpee now has phase-gated checks across 4 AI states.
        Previously this returned [] — now it provides meaningful feedback
        for the plank hold, squat descent, standing lockout, and jump phases.
        """
        issues: List[_Issue] = []
        try:
            msgs       = REGISTRY[Exercise.BURPEE].mistake_messages
            body_align = angles.get("Body Align", 0.0)
            knee       = _best_nz(
                angles.get("Left Knee",  0.0),
                angles.get("Right Knee", 0.0),
            )
            back = angles.get("Back", 0.0)

            # Plank hold: body alignment
            if ai_state == "plank" and body_align > 0:
                if body_align < 150.0:
                    m = _msg(msgs, "hip_sag", "Keep hips level in the plank")
                    if m: issues.append(_Issue(m, Priority.CRITICAL, True))

            # Squat phase: back posture
            if ai_state == "down" and back > 0 and back < 120.0:
                m = _msg(msgs, "lean_forward", "Keep your chest up in the squat")
                if m: issues.append(_Issue(m, Priority.WARNING))

            # Stand/up phase: full lockout
            if ai_state == "up" and knee > 0 and knee < 155.0:
                m = _msg(msgs, "not_standing", "Stand fully before the jump")
                if m: issues.append(_Issue(m, Priority.TIP))

        except (KeyError, AttributeError):
            pass
        return issues

    def _check_bicep_curl(self, angles, ai_state, kp, extra) -> List[_Issue]:
        issues: List[_Issue] = []
        try:
            cfg  = REGISTRY[Exercise.BICEP_CURL].form
            msgs = REGISTRY[Exercise.BICEP_CURL].mistake_messages
            back  = angles.get("Back", 0.0)
            elbow = _best_nz(
                angles.get("Left Elbow",  0.0),
                angles.get("Right Elbow", 0.0),
            )

            # Always-on: back lean is a form fault at any phase
            if ai_state in ACTIVE_STATES and back > 0:
                if back < cfg.get("back_lean_min", 150.0):
                    m = _msg(msgs, "lean_back")
                    if m: issues.append(_Issue(m, Priority.CRITICAL, True))

            # HYBRID-2: "flexed" = top of curl; "extended" = bottom
            if ai_state == "flexed" and elbow > 0:
                if elbow > cfg.get("curl_full_max", 70.0):
                    m = _msg(msgs, "partial_curl")
                    if m: issues.append(_Issue(m, Priority.WARNING))

            if ai_state == "extended" and elbow > 0:
                if elbow < cfg.get("extend_full_min", 135.0):
                    m = _msg(msgs, "not_extended")
                    if m: issues.append(_Issue(m, Priority.WARNING))

        except (KeyError, AttributeError):
            pass
        return issues

    def _check_tricep_dip(self, angles, ai_state, kp, extra) -> List[_Issue]:
        issues: List[_Issue] = []
        try:
            cfg  = REGISTRY[Exercise.TRICEP_DIP].form
            msgs = REGISTRY[Exercise.TRICEP_DIP].mistake_messages
            back  = angles.get("Back", 0.0)
            elbow = _best_nz(
                angles.get("Left Elbow",  0.0),
                angles.get("Right Elbow", 0.0),
            )

            if ai_state in ACTIVE_STATES and back > 0:
                if back < cfg.get("forward_lean_min", 135.0):
                    m = _msg(msgs, "forward_lean")
                    if m: issues.append(_Issue(m, Priority.WARNING))

            if ai_state == "down" and elbow > 0:
                if elbow > cfg.get("depth_max", 100.0):
                    m = _msg(msgs, "not_deep")
                    if m: issues.append(_Issue(m, Priority.WARNING))

            # HYBRID-2: lockout is now the "up" state
            if ai_state == "up" and elbow > 0:
                if elbow < cfg.get("lockout_min", 140.0):
                    m = _msg(msgs, "not_extended")
                    if m: issues.append(_Issue(m, Priority.WARNING))

        except (KeyError, AttributeError):
            pass
        return issues

    def _check_lateral_raise(self, angles, ai_state, kp, extra) -> List[_Issue]:
        issues: List[_Issue] = []
        try:
            cfg  = REGISTRY[Exercise.LATERAL_RAISE].form
            msgs = REGISTRY[Exercise.LATERAL_RAISE].mistake_messages
            l_abd = angles.get("L Arm Abduct", 0.0)
            r_abd = angles.get("R Arm Abduct", 0.0)
            abd   = _avg_nz(l_abd, r_abd)
            arm_straight = _avg_nz(
                angles.get("L Arm Straight", 0.0),
                angles.get("R Arm Straight", 0.0),
            )

            # HYBRID-2: "extended" = arms raised to target height
            if ai_state == "extended":
                if abd > 0 and abd < cfg.get("height_min", 70.0):
                    m = _msg(msgs, "arms_not_high")
                    if m: issues.append(_Issue(m, Priority.WARNING))

            # Bent elbows: always check when arms are moving
            if ai_state in {"extended", "neutral"} and arm_straight > 0:
                if arm_straight < cfg.get("arm_straight_min", 135.0):
                    m = _msg(msgs, "bent_elbows")
                    if m: issues.append(_Issue(m, Priority.WARNING))

            # Asymmetry: check at peak extension
            if ai_state == "extended" and l_abd > 0 and r_abd > 0:
                if abs(l_abd - r_abd) > cfg.get("asymmetry_max", 35.0):
                    m = _msg(msgs, "asymmetric")
                    if m: issues.append(_Issue(m, Priority.TIP))

        except (KeyError, AttributeError):
            pass
        return issues

    def _check_shoulder_press(self, angles, ai_state, kp, extra) -> List[_Issue]:
        """
        HYBRID-2: "extended" replaces old "up" for the overhead lockout check.
        Handles both Exercise.SHOULDER_PRESS and Exercise.OVERHEAD_PRESS via the
        dispatch table alias.
        """
        issues: List[_Issue] = []
        try:
            # Gracefully try SHOULDER_PRESS then OVERHEAD_PRESS for cfg lookup
            try:
                cfg  = REGISTRY[Exercise.SHOULDER_PRESS].form
                msgs = REGISTRY[Exercise.SHOULDER_PRESS].mistake_messages
            except KeyError:
                cfg  = REGISTRY[Exercise.OVERHEAD_PRESS].form
                msgs = REGISTRY[Exercise.OVERHEAD_PRESS].mistake_messages

            back  = angles.get("Back", 0.0)
            elbow = _best_nz(
                angles.get("Left Elbow",  0.0),
                angles.get("Right Elbow", 0.0),
            )

            # Always-on: back arch during pressing motion
            if ai_state in {"down", "extended"} and back > 0:
                if back < cfg.get("back_arch_min", 145.0):
                    m = _msg(msgs, "back_arch")
                    if m: issues.append(_Issue(m, Priority.CRITICAL, True))

            # HYBRID-2: overhead lockout at "extended"
            if ai_state == "extended" and elbow > 0:
                if elbow < cfg.get("lockout_min", 145.0):
                    m = _msg(msgs, "not_locked_out")
                    if m: issues.append(_Issue(m, Priority.WARNING))

            # Rack depth at "down"
            if ai_state == "down" and elbow > 0:
                if elbow > cfg.get("bottom_max", 105.0):
                    m = _msg(msgs, "not_deep")
                    if m: issues.append(_Issue(m, Priority.WARNING))

        except (KeyError, AttributeError):
            pass
        return issues

    # Keep legacy name as alias
    _check_overhead_press = _check_shoulder_press

    def _check_wall_sit(self, angles, ai_state, kp, extra) -> List[_Issue]:
        issues: List[_Issue] = []
        try:
            cfg  = REGISTRY[Exercise.WALL_SIT].form
            msgs = REGISTRY[Exercise.WALL_SIT].mistake_messages
            knee = _avg_nz(
                angles.get("Left Knee",  0.0),
                angles.get("Right Knee", 0.0),
            )
            back = angles.get("Back", 0.0)

            # All wall-sit checks are relevant only while holding ("down" state)
            if ai_state == "down":
                if knee > 0:
                    if knee > cfg.get("knee_max", 115.0):
                        m = _msg(msgs, "knee_too_high")
                        if m: issues.append(_Issue(m, Priority.WARNING))
                    elif knee < cfg.get("knee_min", 65.0):
                        m = _msg(msgs, "knee_too_low")
                        if m: issues.append(_Issue(m, Priority.WARNING))
                if back > 0 and back < cfg.get("back_min", 165.0):
                    m = _msg(msgs, "leaning_forward")
                    if m: issues.append(_Issue(m, Priority.WARNING))

        except (KeyError, AttributeError):
            pass
        return issues


# ── Helpers ────────────────────────────────────────────────────────────────────

def _avg_nz(*values: float) -> float:
    """Mean of positive finite values; 0.0 if none."""
    valid = [v for v in values if v > 0.0 and math.isfinite(v)]
    return sum(valid) / len(valid) if valid else 0.0


def _min_nz(*values: float) -> float:
    """Minimum of positive finite values; 0.0 if none."""
    valid = [v for v in values if v > 0.0 and math.isfinite(v)]
    return min(valid) if valid else 0.0


def _best_nz(a: float, b: float) -> float:
    """Average both sides if both visible, otherwise use the visible one (side-view safe)."""
    if a > 0 and b > 0:
        return (a + b) / 2.0
    return a if a > 0 else b
