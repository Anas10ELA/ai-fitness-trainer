"""
exercises.py — Exercise Registry & Definitions  [UPDATED v7]

Fixes applied (v7):
  CRASH-FIX-1  Missing "elbow_flare" key in PUSHUP mistake_messages.
               feedback.py _check_pushup() accesses msgs["elbow_flare"] when
               elbow > elbow_flare_max (90°) in the DOWN state.  Because the
               key did not exist, every single pushup session raised KeyError,
               crashing the desktop app and sending HTTP 500 to mobile clients.
               Fix: added "elbow_flare" to REGISTRY[Exercise.PUSHUP].mistake_messages.

All other values are unchanged from v6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


# ── Exercise catalogue ────────────────────────────────────────────────────────

class Exercise(Enum):
    SQUAT            = "squat"
    PUSHUP           = "pushup"
    JUMPING_JACK     = "jumping_jack"
    HIGH_KNEES       = "high_knees"
    PLANK            = "plank"
    PULLUP           = "pullup"
    SITUP            = "situp"
    LUNGE            = "lunge"
    MOUNTAIN_CLIMBER = "mountain_climber"
    BURPEE           = "burpee"
    BICEP_CURL       = "bicep_curl"
    TRICEP_DIP       = "tricep_dip"
    LATERAL_RAISE    = "lateral_raise"
    # PRESSFIX: Renamed from OVERHEAD_PRESS → SHOULDER_PRESS per UX feedback.
    # The legacy OVERHEAD_PRESS name is kept as an alias below for backward
    # compatibility with persisted session data, existing tests, and any
    # already-deployed mobile clients pinned to "overhead_press".
    SHOULDER_PRESS   = "shoulder_press"
    WALL_SIT         = "wall_sit"


# ── Backward-compatibility alias ──────────────────────────────────────────────
# Old code paths that reference Exercise.OVERHEAD_PRESS resolve to the new
# SHOULDER_PRESS member. Exercise("overhead_press") still works via the
# _missing_ hook below, so any persisted string also keeps working.
Exercise.OVERHEAD_PRESS = Exercise.SHOULDER_PRESS  # type: ignore[attr-defined]


def _exercise_missing(cls, value):
    """Allow legacy 'overhead_press' string to resolve to SHOULDER_PRESS."""
    if value == "overhead_press":
        return cls.SHOULDER_PRESS
    return None


# Attach _missing_ class method so Exercise("overhead_press") still resolves
Exercise._missing_ = classmethod(_exercise_missing)  # type: ignore[assignment]


# ── Keyboard shortcut → Exercise  (covers all 15) ────────────────────────────

KEY_MAP: Dict[str, Exercise] = {
    "1": Exercise.SQUAT,
    "2": Exercise.PUSHUP,
    "3": Exercise.JUMPING_JACK,
    "4": Exercise.HIGH_KNEES,
    "5": Exercise.PLANK,
    "6": Exercise.PULLUP,
    "7": Exercise.SITUP,
    "8": Exercise.LUNGE,
    "9": Exercise.MOUNTAIN_CLIMBER,
    "0": Exercise.BURPEE,
    "a": Exercise.BICEP_CURL,
    "b": Exercise.TRICEP_DIP,
    "c": Exercise.LATERAL_RAISE,
    "d": Exercise.SHOULDER_PRESS,
    "e": Exercise.WALL_SIT,
}


# ── Exercise definition dataclass ─────────────────────────────────────────────

@dataclass
class ExerciseDef:
    display_name:     str
    keyboard_key:     str
    description:      str
    is_timed:         bool = False
    down_threshold:   float = 90.0
    up_threshold:     float = 160.0
    form:             Dict[str, float] = field(default_factory=dict)
    display_joints:   List[str]        = field(default_factory=list)
    mistake_messages: Dict[str, str]   = field(default_factory=dict)


# ── Exercise registry ─────────────────────────────────────────────────────────

REGISTRY: Dict[Exercise, ExerciseDef] = {

    Exercise.SQUAT: ExerciseDef(
        display_name   = "Squat",
        keyboard_key   = "1",
        description    = "Bodyweight squat",
        down_threshold = 100.0,
        up_threshold   = 150.0,
        form = {
            "back_min":        145.0,
            "knee_cave_ratio":   0.60,
        },
        display_joints   = ["Left Knee", "Right Knee", "Back"],
        mistake_messages = {
            "lean_forward": "⚠ Chest up — don't lean forward!",
            "knees_cave":   "⚠ Knees caving in — drive them out!",
            "not_deep":     "⚠ Go deeper — aim for 90° at the knee",
        },
    ),

    Exercise.PUSHUP: ExerciseDef(
        display_name   = "Push-up",
        keyboard_key   = "2",
        description    = "Standard push-up",
        down_threshold = 100.0,
        up_threshold   = 145.0,
        form = {
            "hip_sag_max":     148.0,
            "hip_pike_min":    202.0,
            "elbow_flare_max":  90.0,
        },
        display_joints   = ["Left Elbow", "Right Elbow", "Left Hip"],
        mistake_messages = {
            "hip_sag":     "⚠ Hips drooping — brace your core!",
            "hip_pike":    "⚠ Hips too high — lower them!",
            "not_deep":    "⚠ Lower your chest closer to the floor!",
            "incomplete":  "⚠ Lock arms fully at the top!",
            # CRASH-FIX-1: this key was missing; feedback._check_pushup() uses it
            # when elbow > elbow_flare_max (90°) in the DOWN state, which fires
            # on virtually every rep → KeyError crashed desktop, 500 on server.
            "elbow_flare": "⚠ Keep elbows tucked — don't flare them out!",
        },
    ),

    Exercise.JUMPING_JACK: ExerciseDef(
        display_name   = "Jumping Jack",
        keyboard_key   = "3",
        description    = "Jumping jacks",
        down_threshold = 50.0,
        up_threshold   = 130.0,
        form = {
            "arms_min_up": 120.0,
        },
        display_joints   = ["L Arm Raise", "R Arm Raise"],
        mistake_messages = {
            "arms_not_raised": "⚠ Raise arms fully overhead!",
        },
    ),

    Exercise.HIGH_KNEES: ExerciseDef(
        display_name   = "High Knees",
        keyboard_key   = "4",
        description    = "High knees",
        down_threshold = 0.0,
        up_threshold   = 0.0,
        form = {
            "knee_height_ratio": 0.10,
        },
        display_joints   = ["L Knee Height", "R Knee Height"],
        mistake_messages = {
            "knees_low":      "⚠ Drive knees higher — above hip level!",
            "knees_not_high": "⚠ Drive knees higher — above hip level!",
        },
    ),

    Exercise.PLANK: ExerciseDef(
        display_name  = "Plank",
        keyboard_key  = "5",
        description   = "Plank hold — timed",
        is_timed      = True,
        form = {
            "hip_sag_max":  148.0,
            "hip_pike_min": 205.0,
        },
        display_joints   = ["Body Align", "Left Hip"],
        mistake_messages = {
            "hip_sag":  "⚠ Hips dropping — squeeze glutes!",
            "hip_pike": "⚠ Hips too high — straighten your back!",
        },
    ),

    Exercise.PULLUP: ExerciseDef(
        display_name   = "Pull-up",
        keyboard_key   = "6",
        description    = "Pull-up — inverted thresholds",
        down_threshold = 150.0,
        up_threshold   =  80.0,
        display_joints   = ["Left Elbow", "Right Elbow"],
        mistake_messages = {
            "not_extended": "⚠ Fully extend arms at the bottom!",
            "not_pulled":   "⚠ Pull chin above the bar!",
        },
    ),

    Exercise.SITUP: ExerciseDef(
        display_name   = "Sit-up",
        keyboard_key   = "7",
        description    = "Sit-up",
        down_threshold = 110.0,
        up_threshold   = 135.0,
        display_joints   = ["Back Angle"],
        mistake_messages = {
            "not_full":  "⚠ Come all the way up — don't half-rep!",
            "pull_neck": "⚠ Don't yank your neck — use your core!",
        },
    ),

    Exercise.LUNGE: ExerciseDef(
        display_name   = "Lunge",
        keyboard_key   = "8",
        description    = "Forward lunge",
        down_threshold = 105.0,
        up_threshold   = 148.0,
        form = {
            "back_min": 138.0,
        },
        display_joints   = ["Left Knee", "Right Knee", "Back"],
        mistake_messages = {
            "lean_forward": "⚠ Torso upright — don't lean forward!",
            "not_deep":     "⚠ Lower thigh to parallel — go deeper!",
        },
    ),

    Exercise.MOUNTAIN_CLIMBER: ExerciseDef(
        display_name   = "Mountain Climber",
        keyboard_key   = "9",
        description    = "Mountain climbers",
        down_threshold = 125.0,
        up_threshold   =  80.0,
        form = {
            "hip_level_px":  40.0,
            "hip_sag_max":  148.0,
            "hip_pike_min": 205.0,
        },
        display_joints   = ["L Hip Angle", "R Hip Angle"],
        mistake_messages = {
            "hips_rotating": "⚠ Keep hips level — don't rotate!",
            "hip_sag":       "⚠ Don't let hips sag — stay in plank!",
            "hip_pike":      "⚠ Lower your hips — stay in plank!",
        },
    ),

    Exercise.BURPEE: ExerciseDef(
        display_name   = "Burpee",
        keyboard_key   = "0",
        description    = "Burpee",
        down_threshold = 120.0,
        up_threshold   = 148.0,
        form = {
            "floor_knee_min": 125.0,
        },
        display_joints   = ["Left Knee", "Back Angle"],
        mistake_messages = {
            "no_extension": "⚠ Stand fully tall — jump at the top!",
            "not_flat":     "⚠ Get flat in plank at the bottom!",
        },
    ),

    # ── New 5 exercises ───────────────────────────────────────────────────────

    Exercise.BICEP_CURL: ExerciseDef(
        display_name   = "Bicep Curl",
        keyboard_key   = "a",
        description    = "Bicep curl — elbow flexion",
        down_threshold = 145.0,
        up_threshold   =  60.0,
        form = {
            "back_lean_min":    150.0,
            "curl_full_max":     70.0,
            "extend_full_min":  135.0,
        },
        display_joints   = ["Left Elbow", "Right Elbow"],
        mistake_messages = {
            "lean_back":    "⚠ Don't lean back — keep torso upright!",
            "partial_curl": "⚠ Curl fully — squeeze at the top!",
            "not_extended": "⚠ Fully extend arms at the bottom!",
        },
    ),

    Exercise.TRICEP_DIP: ExerciseDef(
        display_name   = "Tricep Dip",
        keyboard_key   = "b",
        description    = "Tricep dip — elbow extension",
        down_threshold = 95.0,
        up_threshold   = 145.0,
        form = {
            "depth_max":        100.0,
            "lockout_min":      140.0,
            "forward_lean_min": 135.0,
        },
        display_joints   = ["Left Elbow", "Right Elbow"],
        mistake_messages = {
            "not_deep":     "⚠ Dip lower — bend elbows more!",
            "not_extended": "⚠ Fully extend arms at the top!",
            "forward_lean": "⚠ Keep torso upright — less forward lean!",
        },
    ),

    Exercise.LATERAL_RAISE: ExerciseDef(
        display_name   = "Lateral Raise",
        keyboard_key   = "c",
        description    = "Lateral raise — arm abduction",
        down_threshold = 40.0,
        up_threshold   = 75.0,
        form = {
            "height_min":       70.0,
            "arm_straight_min": 135.0,
            "asymmetry_max":     35.0,
        },
        display_joints   = ["L Arm Abduct", "R Arm Abduct"],
        mistake_messages = {
            "arms_not_high": "⚠ Raise arms to shoulder height!",
            "bent_elbows":   "⚠ Keep arms straighter — slight bend only!",
            "asymmetric":    "⚠ Raise both arms evenly!",
        },
    ),

    Exercise.SHOULDER_PRESS: ExerciseDef(
        display_name   = "Shoulder Press",
        keyboard_key   = "d",
        description    = "Shoulder press — elbow extension overhead",
        # PRESSFIX: thresholds relaxed (was 100/148). Real users don't
        # bring the bar/dumbbell all the way to shoulder lock (105°) and
        # don't lock out elbows hard at the top.
        down_threshold = 110.0,
        up_threshold   = 135.0,
        form = {
            "back_arch_min": 145.0,
            "lockout_min":   135.0,
            "bottom_max":    115.0,
        },
        display_joints   = ["Left Elbow", "Right Elbow", "Back"],
        mistake_messages = {
            "back_arch":      "⚠ Don't arch your back — brace your core!",
            "not_locked_out": "⚠ Press arms fully overhead at the top!",
            "not_deep":       "⚠ Lower weight to shoulder level!",
        },
    ),

    Exercise.WALL_SIT: ExerciseDef(
        display_name  = "Wall Sit",
        keyboard_key  = "e",
        description   = "Wall sit — static hold, timed",
        is_timed      = True,
        form = {
            "knee_min":  65.0,
            "knee_max": 115.0,
            "back_min": 165.0,   # FIX-4 (v6): tightened from 148° → 165°
        },
        display_joints   = ["Left Knee", "Right Knee", "Back"],
        mistake_messages = {
            "knee_too_high":   "⚠ Slide down — knees should be at 90°!",
            "knee_too_low":    "⚠ Slide up slightly — too deep!",
            "leaning_forward": "⚠ Keep your back flat against the wall!",
        },
    ),
}


# ── Convenience helpers ───────────────────────────────────────────────────────

def list_exercises() -> str:
    lines = ["  Key | Exercise"]
    lines.append("  " + "-" * 46)
    for key, ex in KEY_MAP.items():
        d = REGISTRY[ex]
        timed = " [TIMED]" if d.is_timed else ""
        lines.append(f"   {key}  | {d.display_name:<22s}{timed}")
    return "\n".join(lines)
