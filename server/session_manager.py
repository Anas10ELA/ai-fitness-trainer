"""
server/session_manager.py
═════════════════════════
Session Management  [UPDATED v5 — Async Queue compatible]

Changes in v5:
  ASYNC-Q2  No changes to Session or SessionManager logic required.
            Session objects are not shared between concurrent calls:
            each WebSocket connection owns one Session, and the
            InferenceQueue ensures that only one WorkItem per session
            is executing in the thread pool at any time (per-session
            queue serialises calls for a single session while still
            allowing different sessions to run concurrently).

  HYBRID-2  Retained: `frames_since_detect` field tracks elapsed frames
            since last YOLO run. Resets on exercise switch / rep reset.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

from dl_engine import DLEngine
from exercises import Exercise
from feedback  import FeedbackEngine
from rep_policy import resolve_rep_policy
from temporal  import TemporalValidator
from tracker   import PersonTracker
from utils     import KalmanKeypoints, FPSCounter
from config    import cfg

log = logging.getLogger(__name__)

# ── Config — sourced from centralised config (Step 9) ─────────────────────────
SESSION_TIMEOUT_SECS = cfg.session_timeout_secs


# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Session:
    session_id:  str
    exercise:    Exercise
    user_id:     Optional[str]
    checkpoint_dir:       str = cfg.dl_checkpoint_dir
    smooth_window:        int = cfg.dl_smooth_window
    require_closed_mode:  str = cfg.dl_require_closed

    engine:   DLEngine          = field(init=False)
    feedback: FeedbackEngine    = field(init=False)
    temporal: TemporalValidator = field(init=False)
    smoother: KalmanKeypoints   = field(init=False)
    fps:      FPSCounter        = field(init=False)
    tracker:  PersonTracker     = field(init=False)

    prev_angles: Dict[str, float]  = field(init=False)

    # HYBRID-2: frames since last YOLO detection
    frames_since_detect: int = field(default=0)

    created_at:    float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    start_time:    float = field(default_factory=time.time)

    last_angles:   Dict  = field(default_factory=dict)
    last_feedback: str   = ""
    rep_blocked:   bool  = False
    ai_state:      str   = ""
    ai_conf:       float = 0.0
    model_ready:   bool  = False
    reps:          int   = 0
    hold_seconds:  float = 0.0

    def __post_init__(self) -> None:
        self.engine      = self._build_engine(self.exercise)
        self.feedback    = FeedbackEngine(cooldown_scale=1.0)
        self.temporal    = TemporalValidator()
        self.smoother    = KalmanKeypoints()
        self.fps         = FPSCounter(window=30)
        self.tracker     = PersonTracker()
        self.prev_angles = {}

    def _build_engine(self, exercise: Exercise) -> DLEngine:
        require_closed, away_min_sec = resolve_rep_policy(
            exercise.value, self.require_closed_mode
        )
        log.info(
            "Session DL policy [%s]: require_closed=%s away_min=%.2fs smooth_k=%d",
            exercise.value, require_closed, away_min_sec, self.smooth_window,
        )
        return DLEngine(
            exercise       = exercise.value,
            checkpoint_dir = self.checkpoint_dir,
            smooth_window  = self.smooth_window,
            require_closed = require_closed,
            away_min_sec   = away_min_sec,
        )

    def switch_exercise(self, new_exercise: Exercise) -> None:
        self.exercise  = new_exercise
        self.engine    = self._build_engine(new_exercise)
        self.feedback.reset()
        self.temporal.reset()
        self.smoother.reset()
        self.tracker.reset()
        self.prev_angles.clear()
        self.frames_since_detect = 0
        self.last_angles   = {}
        self.last_feedback = ""
        self.rep_blocked   = False
        self.ai_state      = ""
        self.ai_conf       = 0.0
        self.model_ready   = False
        self.reps          = 0
        self.hold_seconds  = 0.0

    def reset_reps(self) -> None:
        self.engine.reset()
        self.feedback.reset()
        self.temporal.reset()
        self.tracker.reset()
        self.prev_angles.clear()
        self.frames_since_detect = 0
        self.last_angles   = {}
        self.last_feedback = ""
        self.rep_blocked   = False
        self.ai_state      = ""
        self.ai_conf       = 0.0
        self.model_ready   = False
        self.reps          = 0
        self.hold_seconds  = 0.0

    def touch(self) -> None:
        self.last_activity = time.time()

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.last_activity) > SESSION_TIMEOUT_SECS

    @property
    def duration_secs(self) -> float:
        return time.time() - self.start_time

    def summary(self) -> Dict:
        return {
            "session_id":        self.session_id,
            "exercise":          self.exercise.value,
            "total_reps":        self.reps,
            "hold_seconds":      self.hold_seconds,
            "duration_secs":     round(self.duration_secs, 1),
            "track_id_switches": self.tracker.total_id_switches,
            "track_stability":   round(self.tracker.stability_ratio, 3),
        }


# ═══════════════════════════════════════════════════════════════════════════════

class SessionManager:
    def __init__(
        self,
        checkpoint_dir: str = cfg.dl_checkpoint_dir,
        smooth_window: int = cfg.dl_smooth_window,
        require_closed_mode: str = cfg.dl_require_closed,
    ) -> None:
        self._sessions: Dict[str, Session] = {}
        self._lock = asyncio.Lock()
        self._checkpoint_dir = checkpoint_dir
        self._smooth_window = smooth_window
        self._require_closed_mode = require_closed_mode

    async def create(self, exercise: str, user_id: Optional[str] = None) -> str:
        try:
            ex = Exercise(exercise.lower())
        except ValueError:
            ex = Exercise.SQUAT
        session_id = str(uuid.uuid4())
        session    = Session(
            session_id=session_id,
            exercise=ex,
            user_id=user_id,
            checkpoint_dir=self._checkpoint_dir,
            smooth_window=self._smooth_window,
            require_closed_mode=self._require_closed_mode,
        )
        async with self._lock:
            self._sessions[session_id] = session
        return session_id

    async def get(self, session_id: str) -> Optional[Session]:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if session.is_expired:
                del self._sessions[session_id]
                return None
            session.touch()
            return session

    async def close(self, session_id: str) -> Optional[Dict]:
        async with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return None
        return session.summary()

    async def count(self) -> int:
        async with self._lock:
            return len(self._sessions)

    async def cleanup_expired(self) -> int:
        async with self._lock:
            expired = [sid for sid, s in self._sessions.items() if s.is_expired]
            for sid in expired:
                del self._sessions[sid]
        return len(expired)
