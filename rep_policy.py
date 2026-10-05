"""
Shared rep-counting policy for the DLEngine anchor FSM.

Both the desktop app and the FastAPI server use these values so a given
exercise follows the same live counting rules on every surface.
"""

from __future__ import annotations

from typing import Dict, Tuple


# Values are seconds. DLEngine converts them to frames using live inference FPS.
REP_POLICY: Dict[str, Tuple[bool, float]] = {
    # exercise          require_closed  away_min_sec
    "squat":            (True,          0.27),
    "lateral_raise":    (True,          0.27),
    "tricep_dip":       (True,          0.27),
    "pushup":           (False,         0.33),
    "situp":            (False,         0.33),
    # Timed exercises ignore these values; DLEngine uses hold-timer mode.
    "wall_sit":         (False,         0.27),
    "plank":            (False,         0.27),
}

REP_POLICY_DEFAULT: Tuple[bool, float] = (False, 0.27)


def resolve_rep_policy(exercise: str, override: str = "auto") -> Tuple[bool, float]:
    """Return (require_closed, away_min_sec), honoring on/off override."""
    require_closed, away_min_sec = REP_POLICY.get(exercise, REP_POLICY_DEFAULT)
    if override == "on":
        require_closed = True
    elif override == "off":
        require_closed = False
    return require_closed, away_min_sec
