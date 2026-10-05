"""
server/schemas.py
═════════════════
Pydantic models for API requests and responses.  [UPDATED v3 — TRT Backends]

Changes in v3 (Step 8):
  TRT-SCHEMA-1  HealthResponse now includes optional detector_backend and
                pose_backend fields so the /health endpoint surfaces which
                inference engine is active (TrtPersonDetector, OnnxPersonDetector,
                PersonDetector, TrtPoseEstimator, OnnxPoseEstimator, or PoseEstimator).

All v2 (Tracking) fields are preserved unchanged.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
#  HTTP Schemas
# ═══════════════════════════════════════════════════════════════════════════════

class SessionStartRequest(BaseModel):
    exercise: str = Field(default="squat", description="Exercise name")
    user_id:  Optional[str] = Field(default=None, description="Optional user identifier")


class VoiceInstructionResponse(BaseModel):
    exercise: str
    display_name: str
    language: str = Field("en-US", description="Instruction language/locale")
    provider: str = Field("openai", description="Configured voice provider")
    model: str = Field("", description="Text-to-speech model")
    voice: str = Field("", description="Text-to-speech voice")
    audio_format: str = Field("mp3", description="Cached audio format")
    text: str = Field(..., description="Natural spoken setup instruction")
    audio_url: Optional[str] = Field(None, description="Relative URL for cached/generated audio")
    audio_ready: bool = Field(False, description="True when audio_url points to playable audio")
    cached: bool = Field(False, description="True when the audio file already existed")
    reason: Optional[str] = Field(None, description="Why audio is unavailable, if text-only")


class SessionStartResponse(BaseModel):
    session_id: str
    exercise:   str
    message:    str = "Session started"
    voice_instruction: Optional[VoiceInstructionResponse] = None


class SessionEndResponse(BaseModel):
    session_id:         str
    total_reps:         int
    hold_seconds:       float
    exercise:           str
    duration_secs:      float
    message:            str   = "Session ended"
    track_id_switches:  int   = Field(0,   description="Total subject identity switches")
    track_stability:    float = Field(0.0, description="Fraction of frames with confirmed match")


class HealthResponse(BaseModel):
    status:           str  = "ok"
    model_loaded:     bool
    active_sessions:  int
    # TRT-SCHEMA-1: active inference backend names
    detector_backend: Optional[str] = Field(
        None, description="Active detector class: TrtPersonDetector | OnnxPersonDetector | PersonDetector"
    )
    pose_backend:     Optional[str] = Field(
        None, description="Active pose class: TrtPoseEstimator | OnnxPoseEstimator | PoseEstimator"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  WebSocket Schemas
# ═══════════════════════════════════════════════════════════════════════════════

class WSFrameInput(BaseModel):
    frame_b64: str   = Field(...,     description="Base64-encoded image frame")
    exercise:  str   = Field("squat", description="Current exercise")
    timestamp: Optional[float] = Field(None, description="Client timestamp (ms)")


class KeypointXY(BaseModel):
    x: float
    y: float


class WSFrameOutput(BaseModel):
    """
    Server sends this for every processed frame over WebSocket.
    The mobile client draws the skeleton and displays metrics from this data.
    """
    # Rep counting
    reps:         int   = Field(...,   description="Total reps counted")
    state:        str   = Field(...,   description="FSM state: ready/down/up")
    hold_seconds: float = Field(0.0,   description="Hold time (Plank/Wall Sit)")

    # AI predictions
    exercise_name: str   = Field(..., description="Active exercise name")
    exercise_conf: float = Field(..., description="DLEngine state confidence [0-1]")
    form_score:    float = Field(..., description="Rule-based form proxy [0-1]")
    is_form_good:  bool  = Field(..., description="form_score >= threshold")
    model_ready:   bool  = Field(..., description="False during warm-up buffer fill")

    # Feedback
    feedback_message: str  = Field("",    description="Human-readable form feedback")
    rep_blocked:      bool = Field(False, description="True if rep would not count")

    # Performance
    fps:          float = Field(..., description="Server-side inference FPS")
    latency_ms:   float = Field(..., description="Total processing latency in ms")

    # Skeleton
    keypoints:   Optional[List[KeypointXY]] = Field(
        None, description="17 COCO keypoints in pixel coords")
    person_bbox: Optional[List[float]] = Field(
        None, description="[x1,y1,x2,y2] tracked person bounding box")

    # Angles (debug)
    angles: Optional[Dict[str, float]] = Field(
        None, description="Joint angles in degrees")

    # Tracker metadata (v2)
    track_id:          int   = Field(0,           description="Current target track ID")
    track_state:       str   = Field("searching", description="Tracker state")
    track_iou:         float = Field(0.0,         description="IoU match confidence [0-1]")
    track_stability:   float = Field(0.0,         description="Stability ratio [0-1]")
    track_id_switches: int   = Field(0,           description="Total identity switches")


class WSErrorMessage(BaseModel):
    error:  str
    code:   int   = 400
    detail: Optional[str] = None


class WSSwitchExercise(BaseModel):
    type:     str = "switch"
    exercise: str
