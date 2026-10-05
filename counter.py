"""
counter.py — AI State-Machine Rep Counter  [v9 — ST-GCN Pure State Machine]

Complete rewrite from v8.  All angle-based threshold logic has been REMOVED.
Rep counting is now driven EXCLUSIVELY by the 7-state sequence predicted by the
ST-GCN model, delivered as a string via RepCounter.update(ai_state).

Architecture
────────────
Each exercise defines a REP_SEQUENCE — an ordered list of states that must be
observed (in order) to constitute one repetition.  The state machine advances a
`cursor` through this list.  Timed-hold exercises (Plank, Wall Sit) have no
sequence; instead they accumulate hold_seconds while the target state persists.

Double-counting prevention (two independent guards)
────────────────────────────────────────────────────
  1. Dwell gate — a candidate state transition is only *committed* to the cursor
     after MIN_DWELL consecutive frames agree on it (≈100ms @ 30fps).  A single
     noisy frame can never advance or reset the sequence.

  2. Post-rep cooldown — after a rep is counted, the cursor is frozen for
     COOLDOWN_SECS.  The cursor resets only after the lock expires AND the model
     returns to the start state.  This prevents the deceleration tail of one rep
     (e.g. "up" overshoot at the top of a squat) from immediately satisfying the
     first step of the next rep's sequence.

Soft sequence recovery
──────────────────────
If the committed state matches seq[0] while cursor > 0 (user aborted mid-rep),
the cursor is reset to 1 ("we just saw seq[0] again") rather than 0.  This lets
the FSM handle genuine restarts without waiting for the full dwell-then-advance
cycle again.

7-State vocabulary (must match build_dataset.py STATE_TO_IDX):
    "neutral"   0 — resting / start position
    "down"      1 — squat bottom / hip hinge / low point
    "up"        2 — standing / lockout / top position
    "plank"     3 — horizontal isometric hold
    "jump"      4 — airborne / explosive phase
    "extended"  5 — full limb extension (press lockout, leg extension)
    "flexed"    6 — peak concentric contraction (curl top, leg curl top)
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

from exercises import Exercise  # project-local

log = logging.getLogger(__name__)


# ── Tuning constants ──────────────────────────────────────────────────────────

DWELL_WINDOW_SECS: float = 0.10   # seconds of agreement required to commit a state
COOLDOWN_SECS:     float = 0.50   # seconds after a rep before the next can begin
IDLE_TIMEOUT_SECS: float = 4.0    # seconds without any active state → soft reset


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-exercise state sequences
# ═══════════════════════════════════════════════════════════════════════════════

# Each sequence is the *minimum* ordered set of AI-predicted states that
# constitutes one full rep.  Intermediate frames (e.g. "neutral" passing
# through the top of a squat) are silently ignored by the cursor.
#
# None → timed-hold exercise; see HOLD_TARGET_STATE.

REP_SEQUENCE: Dict[Exercise, Optional[List[str]]] = {
    # ── Simple up/down cycles ─────────────────────────────────────────────────
    Exercise.SQUAT:            ["down", "up"],
    Exercise.PUSHUP:           ["down", "up"],
    Exercise.PULLUP:           ["up", "neutral"],
    Exercise.SITUP:            ["up", "neutral"],
    Exercise.LUNGE:            ["down", "neutral"],
    Exercise.TRICEP_DIP:       ["down", "up"],

    # ── Press / extended cycles ───────────────────────────────────────────────
    Exercise.SHOULDER_PRESS:   ["down", "extended"],
    Exercise.OVERHEAD_PRESS:   ["down", "extended"],   # alias kept for compat

    # ── Curl / flexed cycles ──────────────────────────────────────────────────
    Exercise.BICEP_CURL:       ["extended", "flexed"],

    # ── Abduction / raise cycles ──────────────────────────────────────────────
    Exercise.JUMPING_JACK:     ["extended", "neutral"],
    Exercise.LATERAL_RAISE:    ["extended", "neutral"],

    # ── Drive / knee-up cycles (each single drive = 1 rep) ───────────────────
    Exercise.HIGH_KNEES:       ["flexed", "neutral"],
    Exercise.MOUNTAIN_CLIMBER: ["flexed", "neutral"],

    # ── Multi-stage burpee ────────────────────────────────────────────────────
    # neutral → squat → plank position → squat → stand → jump → landing
    Exercise.BURPEE:           ["down", "plank", "down", "up", "jump", "neutral"],

    # ── Timed holds (no rep sequence) ────────────────────────────────────────
    Exercise.PLANK:            None,   # target state: "plank"
    Exercise.WALL_SIT:         None,   # target state: "down"
}

# State that must be maintained for timed-hold exercises
HOLD_TARGET_STATE: Dict[Exercise, str] = {
    Exercise.PLANK:    "plank",
    Exercise.WALL_SIT: "down",
}

# Exercises where hold_seconds is the primary metric (not rep count)
HOLD_EXERCISES = frozenset(HOLD_TARGET_STATE.keys())


# ═══════════════════════════════════════════════════════════════════════════════
#  RepCounter
# ═══════════════════════════════════════════════════════════════════════════════

class RepCounter:
    """
    Pure AI state-machine rep counter.

    Parameters
    ----------
    exercise : Exercise
        The exercise being performed.

    Public attributes (read-only)
    ────────────────────────────
    reps          : int   — confirmed rep count
    state         : str   — last committed AI state
    hold_seconds  : float — accumulated hold time (hold exercises only)
    """

    def __init__(self, exercise: Exercise) -> None:
        self.exercise      = exercise
        self.reps:    int  = 0
        self.state:   str  = "neutral"
        self.hold_seconds: float = 0.0

        self._sequence  = REP_SEQUENCE.get(exercise)
        self._is_hold   = exercise in HOLD_EXERCISES
        self._hold_target = HOLD_TARGET_STATE.get(exercise, "plank")

        self._reset_fsm()

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(
        self,
        ai_state:  str,
        fps:       float = 30.0,
        form_ok:   bool  = True,   # set False to block rep counting on bad form
    ) -> None:
        """
        Feed one AI-predicted state string into the counter.

        Parameters
        ----------
        ai_state : str
            The state string from FitnessInferenceEngine, e.g. "down".
        fps : float
            Current frame rate; used to compute the adaptive dwell threshold.
        form_ok : bool
            When False the rep counter still tracks sequence progress but will
            not increment reps.  Lets FeedbackEngine block a rep on bad form.
        """
        now = time.monotonic()

        # ── Idle timeout guard ────────────────────────────────────────────────
        if self._last_update_ts > 0:
            if now - self._last_update_ts > IDLE_TIMEOUT_SECS:
                log.debug("%s idle timeout — soft cursor reset", self.exercise.value)
                self._cursor         = 0
                self._dwell_count    = 0
                self._dwell_candidate = ai_state
                if self._is_hold:
                    self._hold_start  = 0.0
        self._last_update_ts = now

        # ── Dwell gate ────────────────────────────────────────────────────────
        # Only commit a state after MIN_DWELL consecutive frames agree on it.
        # This eliminates single-frame model blips from the sequence tracker.
        min_dwell = max(2, round(fps * DWELL_WINDOW_SECS))   # ≈ 3 @ 30fps

        if ai_state == self._dwell_candidate:
            self._dwell_count += 1
        else:
            self._dwell_candidate = ai_state
            self._dwell_count     = 1
            return   # not yet dwelled enough — do not commit

        if self._dwell_count < min_dwell:
            return   # still building consensus

        # ── Committed state ───────────────────────────────────────────────────
        committed      = ai_state
        self.state     = committed
        self._form_ok  = form_ok

        if self._is_hold:
            self._update_hold(committed, now)
        else:
            self._advance_sequence(committed, now, fps, form_ok)

    def reset(self) -> None:
        """Fully reset rep count, hold timer, and FSM state."""
        self.reps         = 0
        self.hold_seconds = 0.0
        self.state        = "neutral"
        self._reset_fsm()

    def debug_snapshot(self) -> Dict:
        """
        Return a dict suitable for a debug HUD overlay.
        All time values are in seconds.
        """
        cooldown_remaining = 0.0
        if self._locked and self._lock_until_ts > 0:
            cooldown_remaining = max(0.0, self._lock_until_ts - time.monotonic())

        return {
            "exercise":           self.exercise.value,
            "reps":               self.reps,
            "state":              self.state,
            "cursor":             self._cursor,
            "sequence":           self._sequence or [],
            "dwell_candidate":    self._dwell_candidate,
            "dwell_count":        self._dwell_count,
            "locked":             self._locked,
            "cooldown_remaining": round(cooldown_remaining, 2),
            "hold_seconds":       round(self.hold_seconds, 1),
            "is_hold_exercise":   self._is_hold,
        }

    # ── Internal FSM ──────────────────────────────────────────────────────────

    def _reset_fsm(self) -> None:
        """Initialise / re-initialise all FSM bookkeeping."""
        self._cursor:          int   = 0
        self._dwell_candidate: str   = "neutral"
        self._dwell_count:     int   = 0
        self._locked:          bool  = False
        self._lock_until_ts:   float = 0.0
        self._last_update_ts:  float = 0.0
        self._form_ok:         bool  = True
        # Hold exercise tracking
        self._hold_start:      float = 0.0   # monotonic ts when hold began

    # ── Rep-sequence advancement ───────────────────────────────────────────────

    def _advance_sequence(
        self,
        state:   str,
        now:     float,
        fps:     float,
        form_ok: bool,
    ) -> None:
        """
        Advance the sequence cursor on a committed state.

        Design principles:
          • The cursor only moves *forward* — never skips.
          • Soft recovery: if state == seq[0] and cursor > 0, cursor resets to 1
            (treats it as "user is restarting the rep from the top").
          • Any state not matching seq[cursor] and not equal to seq[0] is silently
            ignored — this lets transient "neutral" frames between down and up pass
            without disrupting the squat FSM.
          • Post-rep cooldown prevents the deceleration overshoot of one rep from
            immediately satisfying the first step of the next.
        """
        if self._sequence is None:
            return   # safety guard — should never be reached for hold exercises

        seq = self._sequence

        # ── Post-rep cooldown ─────────────────────────────────────────────────
        if self._locked:
            if now >= self._lock_until_ts:
                self._locked = False
                log.debug("%s cooldown expired", self.exercise.value)
            else:
                return   # still locked — ignore this frame

        # ── Cursor advancement ────────────────────────────────────────────────
        expected = seq[self._cursor]

        if state == expected:
            self._cursor += 1
            log.debug(
                "%s cursor %d→%d  (state=%r)",
                self.exercise.value, self._cursor - 1, self._cursor, state,
            )

            # ── Rep complete ──────────────────────────────────────────────────
            if self._cursor >= len(seq):
                if form_ok:
                    self.reps        += 1
                    self._last_rep_t  = now
                    log.info(
                        "Rep %d counted for %s",
                        self.reps, self.exercise.value,
                    )
                else:
                    log.debug(
                        "%s rep NOT counted — form_ok=False",
                        self.exercise.value,
                    )

                # Engage cooldown lock regardless of form_ok to prevent
                # double-counting even on blocked reps.
                self._cursor        = 0
                self._locked        = True
                self._lock_until_ts = now + COOLDOWN_SECS

        elif state == seq[0] and self._cursor > 0:
            # Soft recovery: user aborted mid-sequence and returned to the
            # starting state.  Reset cursor to 1 (seq[0] already satisfied).
            log.debug(
                "%s soft-recovery: saw seq[0]=%r at cursor=%d → reset to 1",
                self.exercise.value, seq[0], self._cursor,
            )
            self._cursor = 1

        # else: unrelated state — silently ignore, do not disturb cursor

    # ── Timed-hold tracking ────────────────────────────────────────────────────

    def _update_hold(self, state: str, now: float) -> None:
        """
        Accumulate hold_seconds while the target hold state is maintained.
        Resets hold_start when the state breaks.
        """
        if state == self._hold_target:
            if self._hold_start == 0.0:
                self._hold_start = now   # hold just started
            else:
                self.hold_seconds = now - self._hold_start
        else:
            # Hold broken — freeze hold_seconds at last value and clear timer.
            # hold_seconds is NOT reset here so the caller can display the achieved hold.
            self._hold_start = 0.0


# ── Module-level convenience factory ──────────────────────────────────────────

def make_counter(exercise_name: str) -> RepCounter:
    """
    Create a RepCounter by exercise name string (e.g. "squat").
    Raises ValueError for unknown exercise names.
    """
    try:
        ex = Exercise(exercise_name.lower())
    except ValueError:
        valid = [e.value for e in Exercise]
        raise ValueError(
            f"Unknown exercise {exercise_name!r}. Valid: {valid}"
        )
    return RepCounter(ex)
