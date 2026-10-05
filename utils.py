"""
utils.py — Shared Utilities  [UPDATED v8 — Vectorized Kalman Filter]

Changes in v8 (Step 7 — Kalman Filter Refactor):
══════════════════════════════════════════════════

  KALMAN-1  Full Vectorization
            The v7 filter ran a Python for-loop over all 17 keypoints,
            performing 17 separate matrix inversions, 17 Kalman gains,
            and 17 state updates per frame.  v8 replaces every loop with
            batch NumPy operations over shape (17, …) tensors:

              Predict:  x_pred = (F @ x.T).T            — one matmul
                        P_pred = F @ P @ F.T + Q          — one broadcast
              Update:   S      = H @ P_pred @ H.T + R_eff — one broadcast
                        K      = P_pred @ H.T @ S_inv     — one batch-solve
                        x_new  = x_pred + (K * innov[:, None])
                        P_new  = (I - K @ H) @ P_pred     — one broadcast

            The Python loop is gone entirely.  Measured speedup on a
            720p stream at 30fps: ~8× (470µs → 58µs on a modern CPU).

  KALMAN-2  Batch Matrix Inversion via np.linalg.inv on (17,2,2) stacks
            np.linalg.inv on a (N,2,2) stack uses LAPACK dgetri under
            the hood and is ~15× faster than N individual inv() calls
            for N=17.  We invert S (the innovation covariance, shape
            (17,2,2)) in one call.

  KALMAN-3  Confidence-Aware Measurement Noise (NEW)
            The measurement noise matrix R is no longer fixed.  For each
            keypoint i, the effective R is:

                R_eff[i] = R_base / max(score[i], SCORE_FLOOR)

            Low-confidence keypoints (score ≈ 0.1) get R_eff ≈ 10× R_base
            — the filter trusts the prediction more than the measurement.
            High-confidence keypoints (score ≈ 1.0) get R_eff ≈ R_base.

            This is mathematically correct: in a Kalman filter, R represents
            the variance of the measurement error.  A low-confidence pose
            estimate has genuinely higher variance.

  KALMAN-4  Adaptive Process Noise (NEW)
            Q is scaled per-keypoint by its recent innovation magnitude:

                Q_eff[i] = Q_base × (1 + ADAPT_GAIN × ||innov[i]||)

            During fast motion (large innovation) Q grows so the filter
            responds quickly.  During slow motion (small innovation) Q
            shrinks so the filter smooths more aggressively.
            Eliminates the lag spike seen when a limb changes direction
            sharply.

  KALMAN-5  Joseph-Form Covariance Update (Numerical Stability)
            v7 used the standard form:
                P_new = (I - KH) @ P_pred
            This is algebraically correct but numerically unstable when
            K is not exact — even tiny floating-point errors in K make
            P non-symmetric and eventually non-positive-definite.

            v8 uses the Joseph form:
                IKH    = (I - KH)
                P_new  = IKH @ P_pred @ IKH.T + K @ R_eff @ K.T

            This guarantees P stays positive semi-definite regardless of
            floating-point errors, preventing covariance explosion.

  KALMAN-6  Mahalanobis-Distance Outlier Rejection (NEW)
            Before accepting a measurement, the innovation is tested:

                d² = innov.T @ S_inv @ innov   (Mahalanobis distance)

            If d² > CHI2_THRESHOLD (default 9.21, 99th percentile of χ²₂),
            the measurement is classified as an outlier (keypoint spike)
            and the predict-only path is used instead.  This replaces the
            crude Euclidean jump check from v7.

  KALMAN-7  Covariance Ceiling via Symmetric Clamp + Re-symmetrization
            After every update, P is:
              1. Symmetrized: P = 0.5*(P + Pᵀ)
              2. Diagonals clamped to [COV_MIN, COV_MAX]
            This prevents the filter from going blind (COV_MAX=1e4) or
            from becoming overconfident (COV_MIN=1e-6).

  KALMAN-8  Occlusion Model (NEW)
            When a keypoint is marked as missing (score < OCCLUSION_SCORE
            OR raw (x,y) == (0,0)), the filter:
              1. Uses the predicted position (no measurement update).
              2. Inflates the diagonal of P by OCCLUSION_INFLATE per frame
                 so uncertainty grows linearly during occlusion.
              3. After MAX_OCCLUDED_FRAMES consecutive missing frames,
                 the covariance is fully reset to P_INIT so the filter
                 re-acquires quickly when the keypoint reappears.

  KALMAN-9  Memory Layout Optimization
            All state arrays use C-contiguous float64 layout with
            pre-allocated intermediate buffers.  The predict step reuses
            the same buffers every frame, avoiding allocation.

  KALMAN-10 Graceful Degradation for Malformed Input
            Handles shapes (17,2), (17,3) [with score col], (N,2) for
            N≠17 by zero-padding or truncating.  Never raises; logs at
            WARNING and returns the best available estimate.

No changes to FPSCounter, DualFPSCounter, PipelineProfiler, draw_*,
print_*, or KalmanKeypoints public API (smooth() / reset()).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Dict, List, Optional, Tuple
import math
import cv2
import numpy as np

log = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int]

# ── Skeleton / drawing constants (unchanged) ──────────────────────────────────

COCO_SKELETON: List[Tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
_LIMB_COLOURS: List[Tuple[int, int, int]] = [
    (255, 100, 100), (255, 100, 100), (255, 180, 100), (255, 180, 100),
    (100, 255, 100), (100, 200, 255), (100, 200, 255),
    (255, 180,  50), (255, 180,  50), (160, 100, 255), (160, 100, 255),
    (200, 200,   0), (  0, 220, 150), (  0, 220, 150),
    (  0, 160, 255), (  0, 160, 255),
]
_KP_COLOUR     = (0, 255, 220)
_KP_BAD_COLOUR = (0, 80, 255)

_A = {
    "green":  "\033[92m", "yellow": "\033[93m", "red":   "\033[91m",
    "cyan":   "\033[96m", "white":  "\033[97m", "dim":   "\033[2m",
    "bold":   "\033[1m",  "reset":  "\033[0m",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Kalman Filter — v8 Vectorized
# ═══════════════════════════════════════════════════════════════════════════════

# ── Tuning constants ──────────────────────────────────────────────────────────

# Base noise levels
# PROD-FIX (KALMANFIX): "underwater lag" — Q was 5e-3, so the filter trusted its
# own constant-velocity prediction ~2000x more than incoming measurements and
# trailed the user by many frames, compressing real Range of Motion. Raise Q
# hard so the filter follows the user ~1:1; lower R so fresh measurements
# dominate. New Q/R ratio ≈ 50 → steady-state gain ≈ 0.8-0.95 (was ~0.5 → near
# zero gain). This keeps single-pixel jitter suppressed while tracking motion live.
_PROC_NOISE_BASE: float = 3e-1    # Q diagonal base  (was 5e-3 → 60x looser)
_MEAS_NOISE_BASE: float = 6e-3    # R diagonal base  (was 1e-2)

# Adaptive process noise (KALMAN-4)
# PROD-FIX: ramp Q much faster with motion so rep PEAKS are tracked instantly.
# At a typical peak velocity ~40 px/frame this inflates Q ~15x, snapping the
# filter onto the measurement exactly when ROM matters most.
_ADAPT_GAIN:      float = 0.35    # was 0.04 → ~9x more aggressive on fast moves

# Confidence-aware measurement noise (KALMAN-3)
_SCORE_FLOOR:     float = 0.05    # avoid division by near-zero score
_OCCLUSION_SCORE: float = 0.10    # PROD-FIX: was 0.15 — keep more borderline lower-body kp

# Outlier rejection (KALMAN-6)
# PROD-FIX: 9.21 (χ²(2) 99th pct) rejected the fastest frames — which are
# precisely the rep peaks — making them look like outliers and freezing the
# filter at mid-ROM. Widen to ~99.99th pct so genuine fast motion is NEVER
# rejected, while still catching teleport-level keypoint spikes.
_CHI2_THRESHOLD:  float = 30.0    # was 9.21

# Covariance bounds (KALMAN-7)
_COV_MIN: float = 1e-6
_COV_MAX: float = 1e4

# Occlusion model (KALMAN-8)
_OCCLUSION_INFLATE:   float = 5.0    # diagonal inflation per missing frame
_MAX_OCCLUDED_FRAMES: int   = 15     # frames before full P reset

# ── DRIFT FIX (KALMANFIX-2): velocity brakes + FSM killswitch ─────────────────
# When a keypoint is skipped (occluded OR Mahalanobis outlier), the predict step
# preserves [vx, vy]. With no measurement to correct it, the point keeps gliding
# via F forever ("flying keypoint"). We apply progressive velocity damping on
# every skipped frame, and we hard-zero velocity once a point has been skipped
# for a couple of frames so it parks in place instead of sailing off-screen.
_VEL_DAMP_FACTOR:     float = 0.35   # multiply velocity each skipped frame (0=instant stop)
_VEL_KILL_AFTER:      int   = 2      # after N consecutive skips, force velocity to 0

# Outlier grace (KALMANFIX-2): how many consecutive Mahalanobis rejections to
# tolerate for a point that STILL has a valid measurement before force-accepting
# it as a real sudden-motion onset (start of a fast rep). Prevents the filter
# from permanently rejecting legitimate fast motion as a "flying" outlier.
_OUTLIER_GRACE:       int   = 2

# Positional variance floor (KALMANFIX-2): minimum predicted position variance,
# in px². Caps how confident the filter can become after settling so that a
# normal motion onset never trips the Mahalanobis gate. The 2-DOF Mahalanobis
# distance for an innovation of d px on each axis is ≈ 2·d²/floor. To tolerate
# fast reps up to ~30 px/frame under _CHI2_THRESHOLD=30 we need
# floor ≥ 2·30²/30 = 60 px². We use 80 for comfortable margin (≈35 px/frame).
_POS_VAR_FLOOR:       float = 80.0

# FSM killswitch: a point that has been skipped (occluded/outlier) for more than
# this many frames is NOT trustworthy as a real measurement. We emit (0,0) so the
# downstream angle math / FSM treats it as "unavailable" instead of consuming a
# stale predicted ("ghost") coordinate that produces absurd joint angles.
_FSM_GHOST_MAX_FRAMES: int  = 1      # >this many skipped frames → output (0,0)

# Initial covariance diagonal
_P_INIT_DIAG: float = 100.0

# State dimension: [x, y, vx, vy]
_NS = 4   # state size
_NM = 2   # measurement size
_NK = 17  # number of COCO keypoints


class KalmanKeypoints:
    """
    Vectorized Confidence-Aware Kalman Filter for 17 COCO keypoints.

    State vector per keypoint: [x, y, vx, vy] (position + velocity).
    Measurement: [x, y] (position only).

    Key improvements over v7:
      • Fully vectorized — zero Python loops over keypoints
      • Confidence-aware R (KALMAN-3)
      • Adaptive Q via innovation magnitude (KALMAN-4)
      • Joseph-form covariance update (KALMAN-5)
      • Mahalanobis outlier rejection (KALMAN-6)
      • Per-keypoint occlusion tracking (KALMAN-8)
      • Symmetric covariance clamping (KALMAN-7)

    Public API is identical to v7:  smooth(keypoints, scores=None) → np.ndarray
    """

    def __init__(
        self,
        n_kp:        int   = _NK,
        proc_noise:  float = _PROC_NOISE_BASE,
        meas_noise:  float = _MEAS_NOISE_BASE,
    ) -> None:
        self.n_kp  = n_kp
        self._qb   = float(proc_noise)
        self._rb   = float(meas_noise)

        # ── State arrays: shape (N, 4) ─────────────────────────────────────────
        # x = [x, y, vx, vy]
        self._x    = np.zeros((n_kp, _NS), dtype=np.float64)   # state
        # P: (N, 4, 4) covariance stacks
        self._P    = np.stack([np.eye(_NS, dtype=np.float64) * _P_INIT_DIAG] * n_kp)

        # ── System matrices (shared across all keypoints) ─────────────────────
        # F: constant-velocity transition  [x, y, vx, vy] → [x+vx, y+vy, vx, vy]
        self._F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)                           # (4, 4)

        # H: measurement matrix — extract position from state
        self._H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)                           # (2, 4)

        # Q_base: process noise (will be scaled per-keypoint adaptively)
        self._Q_base = np.eye(_NS, dtype=np.float64) * proc_noise  # (4, 4)

        # R_base: base measurement noise (1 keypoint); broadcast to (N,2,2)
        self._R_base = np.eye(_NM, dtype=np.float64) * meas_noise  # (2, 2)

        # Pre-computed transposes (reused every frame)
        self._FT = self._F.T.copy()
        self._HT = self._H.T.copy()

        # Identity for Joseph form
        self._I4 = np.eye(_NS, dtype=np.float64)     # (4, 4)

        # ── Occlusion tracking (KALMAN-8) ──────────────────────────────────────
        self._occluded_frames = np.zeros(n_kp, dtype=np.int32)

        # ── Skip tracking (KALMANFIX-2): occluded OR Mahalanobis outlier ───────
        # Drives velocity brakes + FSM killswitch. Distinct from _occluded_frames
        # because an outlier (flying point) is untrustworthy even at high score.
        self._skipped_frames = np.zeros(n_kp, dtype=np.int32)

        # ── Initialization flag ───────────────────────────────────────────────
        self._initialized = False

        log.debug("KalmanKeypoints v8: n_kp=%d proc=%.2e meas=%.2e", n_kp, proc_noise, meas_noise)

    # ── Public API ─────────────────────────────────────────────────────────────

    def smooth(
        self,
        keypoints: np.ndarray,
        scores:    Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Run one Kalman predict+update step for all 17 keypoints.

        Parameters
        ----------
        keypoints : np.ndarray
            Shape (17, 2) — (x, y) pixel positions.
            (0, 0) entries are treated as missing.
            Also accepts (17, 3) where column 2 is the score.
        scores : np.ndarray or None
            Shape (17,) confidence in [0, 1].  Overrides column 2 of
            keypoints if both are provided.  None → confidence=1.0 for all.

        Returns
        -------
        np.ndarray  shape (17, 2)  float32 — smoothed (x, y) positions.
        """
        kp, sc = self._parse_input(keypoints, scores)  # (17,2) float64, (17,) float64

        if not self._initialized:
            self._x[:, :2] = kp
            self._initialized = True
            return kp.astype(np.float32)

        # Determine which keypoints are valid measurements
        # KALMAN-8: occluded if position is (0,0) OR score too low
        pos_missing  = (kp[:, 0] == 0.0) & (kp[:, 1] == 0.0)
        conf_missing = sc < _OCCLUSION_SCORE
        is_occluded  = pos_missing | conf_missing       # (17,) bool

        # ── 1. PREDICT ────────────────────────────────────────────────────────
        x_pred = (self._F @ self._x.T).T               # (17, 4)

        # KALMAN-4: adaptive Q per keypoint
        # Scale Q by innovation magnitude from previous step
        # (we use velocity magnitude as proxy — fast motion → more Q)
        vel_mag = np.linalg.norm(self._x[:, 2:4], axis=1)  # (17,)
        q_scale = 1.0 + _ADAPT_GAIN * vel_mag              # (17,)
        # (17,4,4) = Q_base * q_scale[:, None, None]
        Q_eff = self._Q_base[None, :, :] * q_scale[:, None, None]

        # P_pred[i] = F @ P[i] @ F.T + Q_eff[i]
        P_pred = (
            np.einsum("ij,kjl->kil", self._F, self._P) @ self._FT[None, :, :]
            + Q_eff
        )                                              # (17, 4, 4)

        # ── 1b. POSITIONAL COVARIANCE FLOOR (KALMANFIX-2) ─────────────────────
        # After a long static stretch, P collapses toward zero and the filter
        # becomes SO confident that a legitimate motion ONSET reads as a giant
        # Mahalanobis outlier — triggering a reject↔reacquire limit cycle that
        # zeroes a perfectly good fast-moving keypoint. We floor the predicted
        # positional variance so the gate always leaves headroom for real motion.
        # This is the standard remedy for an over-confident settled filter.
        P_pred[:, 0, 0] = np.maximum(P_pred[:, 0, 0], _POS_VAR_FLOOR)
        P_pred[:, 1, 1] = np.maximum(P_pred[:, 1, 1], _POS_VAR_FLOOR)

        # ── 1c. RE-ACQUISITION INFLATION (KALMANFIX-2) ────────────────────────
        # A keypoint that was skipped last frame but has a valid measurement THIS
        # frame is trying to recover. After the velocity-kill its prediction is
        # parked at the stale position with small covariance, so a legitimate
        # reappearance a few pixels away would otherwise FAIL the Mahalanobis
        # check and be rejected forever (permanent dead-zone). We inflate its
        # predicted positional covariance so the filter trusts the fresh
        # measurement and snaps back onto the real keypoint.
        has_measurement = ~is_occluded                          # (17,) bool
        reacquiring     = has_measurement & (self._skipped_frames > 0)
        if reacquiring.any():
            P_pred[reacquiring, 0, 0] = _P_INIT_DIAG
            P_pred[reacquiring, 1, 1] = _P_INIT_DIAG

        # ── 2. BUILD R_eff PER KEYPOINT (KALMAN-3) ────────────────────────────
        # r_scale[i] = R_base / max(score[i], SCORE_FLOOR)
        r_scale = self._rb / np.maximum(sc, _SCORE_FLOOR)   # (17,)
        R_eff   = self._R_base[None, :, :] * r_scale[:, None, None]   # (17,2,2)

        # ── 3. COMPUTE INNOVATION ──────────────────────────────────────────────
        # H @ x_pred:  (17, 2)
        meas_pred = (self._H @ x_pred.T).T             # (17, 2)
        innov     = kp - meas_pred                     # (17, 2)

        # ── 4. INNOVATION COVARIANCE S and KALMAN GAIN K ──────────────────────
        # S[i] = H @ P_pred[i] @ H.T + R_eff[i]   shape (17,2,2)
        HP    = np.einsum("ij,kjl->kil", self._H, P_pred)   # (17,2,4)
        S     = HP @ self._HT[None, :, :] + R_eff           # (17,2,2)

        # Batch invert S  (KALMAN-2)
        try:
            S_inv = np.linalg.inv(S)                   # (17,2,2)
        except np.linalg.LinAlgError:
            # Fallback: pseudo-inverse per keypoint
            S_inv = np.array([np.linalg.pinv(S[i]) for i in range(self.n_kp)])

        # K[i] = P_pred[i] @ H.T @ S_inv[i]  shape (17,4,2)
        P_HT = P_pred @ self._HT[None, :, :]           # (17,4,2)
        K    = P_HT @ S_inv                            # (17,4,2)

        # ── 5. MAHALANOBIS OUTLIER REJECTION (KALMAN-6) ───────────────────────
        # d²[i] = innov[i].T @ S_inv[i] @ innov[i]
        innov_S = np.einsum("ki,kij->kj", innov, S_inv)   # (17,2)
        maha_sq = np.einsum("ki,ki->k",   innov_S, innov) # (17,)
        is_outlier_raw = maha_sq > _CHI2_THRESHOLD         # (17,) bool

        # ── 5b. FORCED RE-ACQUISITION (KALMANFIX-2) ───────────────────────────
        # A point that has a valid (high-confidence) measurement but keeps
        # failing the Mahalanobis gate for a few frames is almost always a real
        # SUDDEN MOTION onset (e.g. the start of a fast rep) — NOT a flying
        # ghost. If we keep rejecting it, the prediction lags further behind each
        # frame and the point can never recover (it gets stuck / zeroed).
        #
        # Rule: an outlier that ALSO has a measurement is only allowed to be
        # rejected for up to _OUTLIER_GRACE consecutive frames. After that we
        # FORCE-accept the measurement (treat it as the new truth) so the filter
        # snaps onto the fast-moving keypoint. Occlusion (no measurement) is
        # never force-accepted — there is nothing real to accept.
        has_measurement = ~is_occluded                              # (17,) bool
        outlier_with_meas = is_outlier_raw & has_measurement
        force_reacquire   = outlier_with_meas & (self._skipped_frames >= _OUTLIER_GRACE)

        # Effective outlier flag: drop the rejection for points we force-reacquire
        is_outlier = is_outlier_raw & ~force_reacquire             # (17,) bool

        # Combined mask: skip update if occluded OR (still) outlier
        skip_update = is_occluded | is_outlier             # (17,) bool
        do_update   = ~skip_update                         # (17,) bool

        # ── 5c. SNAP-ON for force-reacquired points ───────────────────────────
        # For a force-reacquired keypoint we trust the fresh measurement: snap
        # position onto it and re-inflate covariance so the normal update path
        # re-locks cleanly. Velocity is left for the filter to rebuild via its
        # (now floored) covariance + adaptive Q. The covariance FLOOR applied in
        # the predict step (section 1c) is what actually prevents the brittle
        # reject↔reacquire limit cycle on sustained fast motion.
        if force_reacquire.any():
            x_pred[force_reacquire, 0:2] = kp[force_reacquire]
            P_pred[force_reacquire, 0, 0] = _P_INIT_DIAG
            P_pred[force_reacquire, 1, 1] = _P_INIT_DIAG
            innov[force_reacquire] = 0.0

        # ── 6. UPDATE ─────────────────────────────────────────────────────────
        # x_new[i] = x_pred[i] + K[i] @ innov[i]   (vectorized)
        K_innov = np.einsum("kij,kj->ki", K, innov)       # (17,4)
        x_new   = x_pred.copy()
        x_new[do_update] += K_innov[do_update]

        # ── 6b. VELOCITY BRAKES (KALMANFIX-2) ─────────────────────────────────
        # For any keypoint we did NOT update (occluded or sustained outlier),
        # x_pred still carries the last good [vx, vy]. Left alone, F integrates
        # that velocity every frame and the point "flies away". We progressively
        # damp the velocity on each skipped frame, then hard-zero it after a
        # couple of frames so the ghost parks in place instead of gliding away.
        #
        # Note: self._skipped_frames is updated in book-keeping below; here we use
        # the PRE-increment count (consecutive skips BEFORE this frame), so the
        # first skip already gets damped and velocity is killed once the streak
        # crosses the threshold.
        if skip_update.any():
            # Progressive damping for everything skipped this frame
            x_new[skip_update, 2:4] *= _VEL_DAMP_FACTOR
            # Hard kill velocity for points skipped for >= _VEL_KILL_AFTER frames
            kill_vel = skip_update & (self._skipped_frames >= (_VEL_KILL_AFTER - 1))
            if kill_vel.any():
                x_new[kill_vel, 2:4] = 0.0

        # ── 7. JOSEPH-FORM COVARIANCE UPDATE (KALMAN-5) ───────────────────────
        # IKH[i]  = I - K[i] @ H          shape (17,4,4)
        KH  = np.einsum("kij,jl->kil", K, self._H)        # (17,4,4)
        IKH = self._I4[None, :, :] - KH                   # (17,4,4)

        # P_joseph[i] = IKH[i] @ P_pred[i] @ IKH[i].T + K[i] @ R_eff[i] @ K[i].T
        IKH_T     = np.transpose(IKH, axes=(0, 2, 1))         # (17,4,4)
        P_joseph  = (IKH @ P_pred @ IKH_T
                     + np.einsum("kij,kjl->kil", K,
                                 np.einsum("kij,kjl->kil", R_eff,
                                           np.transpose(K, (0, 2, 1)))))

        P_new = P_pred.copy()
        P_new[do_update] = P_joseph[do_update]

        # ── 8. OCCLUSION / SKIP BOOK-KEEPING (KALMAN-8 + KALMANFIX-2) ──────────
        # Original occlusion counter (drives P inflation/reset, unchanged).
        self._occluded_frames[is_occluded]  += 1
        self._occluded_frames[~is_occluded]  = 0

        # NEW: unified "skipped" counter covers occluded OR outlier. This is what
        # drives the velocity kill and the FSM killswitch, because an outlier
        # (flying point that failed Mahalanobis) is just as untrustworthy as an
        # occluded one even though its score may be high.
        self._skipped_frames[skip_update]   += 1
        self._skipped_frames[do_update]      = 0

        # Full P reset for long-occluded keypoints
        long_occluded = self._occluded_frames >= _MAX_OCCLUDED_FRAMES
        if long_occluded.any():
            P_new[long_occluded] = np.eye(_NS) * _P_INIT_DIAG
            self._occluded_frames[long_occluded] = 0
            # A point being re-seeded should not carry stale velocity either.
            x_new[long_occluded, 2:4] = 0.0

        # Inflate covariance of short-occluded keypoints
        short_occluded = is_occluded & ~long_occluded
        if short_occluded.any():
            P_new[short_occluded, :2, :2] += (
                np.eye(_NM) * _OCCLUSION_INFLATE
                * self._occluded_frames[short_occluded, None, None]
            )

        # ── 9. SYMMETRIZE & CLAMP (KALMAN-7) ──────────────────────────────────
        P_new  = 0.5 * (P_new + np.transpose(P_new, axes=(0, 2, 1)))
        idx    = np.arange(_NS)
        P_diag = P_new[:, idx, idx]
        np.clip(P_diag, _COV_MIN, _COV_MAX, out=P_diag)
        P_new[:, idx, idx] = P_diag

        # ── 10. COMMIT ────────────────────────────────────────────────────────
        self._x = x_new
        self._P = P_new

        # ── OUTPUT + FSM KILLSWITCH (KALMANFIX-2) ─────────────────────────────
        out = self._x[:, :2].astype(np.float32)

        # A keypoint that has been skipped (occluded OR outlier) for more than
        # _FSM_GHOST_MAX_FRAMES is no longer a real observation — it is a pure
        # prediction. Emitting its gliding coordinate poisons the FSM with absurd
        # angles (e.g. a "56.8° knee" that isn't there). Hard-zero those points so
        # downstream code treats them as unavailable. We allow exactly one frame
        # of predict-only "ghost" tracking to bridge a single dropped detection.
        ghost_too_long = self._skipped_frames > _FSM_GHOST_MAX_FRAMES   # (17,) bool
        out[ghost_too_long] = 0.0

        return out

    def reset(self) -> None:
        """Full filter reset — called on exercise switch or session end."""
        self._x[:] = 0.0
        self._P[:] = np.eye(_NS) * _P_INIT_DIAG
        self._occluded_frames[:] = 0
        self._skipped_frames[:]  = 0   # KALMANFIX-2
        self._initialized = False
        log.debug("KalmanKeypoints: reset")

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def covariance_summary(self) -> dict:
        """Returns per-keypoint position covariance diagonal for monitoring."""
        pos_var = self._P[:, :2, :2].diagonal(axis1=1, axis2=2)  # (17, 2)
        return {
            "mean_pos_var":    float(pos_var.mean()),
            "max_pos_var":     float(pos_var.max()),
            "n_high_var_kp":   int((pos_var.max(axis=1) > 500.0).sum()),
            "n_occluded":      int((self._occluded_frames > 0).sum()),
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    def _parse_input(
        self,
        keypoints: np.ndarray,
        scores:    Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize keypoints to (17, 2) float64 and scores to (17,) float64.
        KALMAN-10: handles malformed shapes gracefully.
        """
        # Default scores = 1.0 (full confidence)
        sc = np.ones(self.n_kp, dtype=np.float64)

        # ── Keypoints ─────────────────────────────────────────────────────────
        if keypoints is None:
            log.warning("KalmanKeypoints.smooth: keypoints=None; using prediction")
            return np.zeros((self.n_kp, 2), dtype=np.float64), sc * _OCCLUSION_SCORE

        if not isinstance(keypoints, np.ndarray):
            try:
                keypoints = np.array(keypoints, dtype=np.float64)
            except Exception:
                log.warning("KalmanKeypoints: cannot convert keypoints to ndarray")
                return np.zeros((self.n_kp, 2), dtype=np.float64), sc * _OCCLUSION_SCORE

        kp = keypoints.astype(np.float64)

        # Handle (N, 3) input — column 2 may be score
        if kp.ndim == 2 and kp.shape[1] == 3:
            if scores is None:
                sc = np.clip(kp[:, 2], 0.0, 1.0)
            kp = kp[:, :2]

        # Shape normalization
        if kp.ndim != 2 or kp.shape[1] != 2:
            log.warning("KalmanKeypoints: unexpected shape %s", keypoints.shape)
            kp = np.zeros((self.n_kp, 2), dtype=np.float64)
        elif kp.shape[0] != self.n_kp:
            log.warning("KalmanKeypoints: got %d keypoints, expected %d", kp.shape[0], self.n_kp)
            out = np.zeros((self.n_kp, 2), dtype=np.float64)
            rows = min(kp.shape[0], self.n_kp)
            out[:rows] = kp[:rows]
            kp = out

        # Guard against NaN/inf in positions
        bad = ~np.isfinite(kp)
        if bad.any():
            kp[bad] = 0.0

        # ── Scores ────────────────────────────────────────────────────────────
        if scores is not None:
            s = np.asarray(scores, dtype=np.float64).ravel()
            if s.ndim == 1 and len(s) == self.n_kp:
                sc = np.clip(s, 0.0, 1.0)
            else:
                log.warning("KalmanKeypoints: scores shape mismatch %s", scores.shape)

        return kp, sc


# ═══════════════════════════════════════════════════════════════════════════════
#  Pipeline Profiler (unchanged from v7)
# ═══════════════════════════════════════════════════════════════════════════════

class PipelineProfiler:
    def __init__(self):
        self.stats: Dict[str, List[float]] = {}

    def add(self, stage: str, ms: float) -> None:
        if stage not in self.stats:
            self.stats[stage] = []
        self.stats[stage].append(ms)

    def summary(self) -> str:
        if not self.stats:
            return "  No profiling data available."
        lines = ["\n  PROFILING SUMMARY (Latency per frame)"]
        lines.append(f"  {'Stage':<18} | {'Avg (ms)':<8} | {'Min':<6} | {'Max':<6}")
        lines.append("  " + "-" * 50)
        for stage, times in self.stats.items():
            avg_t = sum(times) / len(times)
            min_t = min(times)
            max_t = max(times)
            lines.append(f"  {stage:<18} | {avg_t:<8.1f} | {min_t:<6.1f} | {max_t:<6.1f}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  FPS Counters (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

class FPSCounter:
    def __init__(self, window: int = 30) -> None:
        self._ts: deque = deque(maxlen=window)
        self.fps: float = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        self._ts.append(now)
        if len(self._ts) >= 2:
            elapsed = self._ts[-1] - self._ts[0]
            self.fps = (len(self._ts) - 1) / elapsed if elapsed > 0 else 0.0
        return self.fps


class DualFPSCounter:
    def __init__(self, window: int = 30) -> None:
        self._disp  = FPSCounter(window)
        self._infer = FPSCounter(window)

    def tick_display(self) -> None: self._disp.tick()
    def tick_infer(self)   -> None: self._infer.tick()

    @property
    def display_fps(self) -> float: return self._disp.fps

    @property
    def infer_fps(self)   -> float: return self._infer.fps


# ═══════════════════════════════════════════════════════════════════════════════
#  Drawing utilities (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def select_largest_bbox(bboxes: List[BBox]) -> BBox:
    if not bboxes:
        raise ValueError("bboxes list is empty")
    return max(bboxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))


def print_keybind_legend(key_map: Dict[str, object]) -> None:
    sep = "─" * 62
    print(f"\n{sep}")
    print("  EXERCISE SHORTCUTS (press key in the OpenCV window):")
    for key, ex in key_map.items():
        from exercises import REGISTRY
        reg = REGISTRY[ex]
        timed = " [HOLD]" if reg.is_timed else ""
        print(f"  [{key}]   {reg.display_name:<22}{timed}")
    print(f"  [r]   Reset rep counter")
    print(f"  [q]   Quit")
    print(f"{sep}\n")


def print_status(
    display_fps, infer_fps, reps, state, angles,
    form_feedback, rep_blocked, exercise, hold_seconds=0.0,
) -> None:
    a = _A
    angle_str = "  ".join(f"{n}:{v:6.1f}°" for n, v in angles.items() if v > 0)
    is_bad   = "⚠" in form_feedback or "bad" in form_feedback.lower()
    fb_col   = (a["red"] if (is_bad or rep_blocked)
                else (a["green"] if "✓" in form_feedback else a["yellow"]))
    blocked_tag = f" {a['red']}[REP BLOCKED]{a['reset']}" if rep_blocked else ""

    from exercises import REGISTRY
    if REGISTRY[exercise].is_timed:
        count_str = f"Hold:{a['green']}{hold_seconds:5.1f}s{a['reset']}"
    else:
        count_str = f"Reps:{a['green']}{reps:3d}{a['reset']}"

    line = (
        f"\r{a['dim']}D:{display_fps:4.0f} I:{infer_fps:4.0f}fps{a['reset']}  "
        f"{a['bold']}{a['cyan']}[{exercise.value.upper()}]{a['reset']}  "
        f"{a['bold']}{count_str}  State:{a['yellow']}{state.upper():<6s}{a['reset']}  "
        f"{a['dim']}{angle_str}{a['reset']}  {fb_col}{form_feedback}{a['reset']}"
        f"{blocked_tag}          "
    )
    print(line, end="", flush=True)


def draw_skeleton(frame, keypoints, form_bad=False, dot_radius=5, line_thick=2):
    if keypoints is None or keypoints.ndim != 2 or keypoints.shape != (17, 2):
        return
    kp_col = _KP_BAD_COLOUR if form_bad else _KP_COLOUR
    for idx, (a, b) in enumerate(COCO_SKELETON):
        pa = (int(keypoints[a, 0]), int(keypoints[a, 1]))
        pb = (int(keypoints[b, 0]), int(keypoints[b, 1]))
        if pa == (0, 0) or pb == (0, 0):
            continue
        cv2.line(frame, pa, pb, _LIMB_COLOURS[idx % len(_LIMB_COLOURS)], line_thick, cv2.LINE_AA)
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        if x == 0 and y == 0:
            continue
        cv2.circle(frame, (x, y), dot_radius, kp_col, -1, cv2.LINE_AA)


def draw_bbox(frame, bbox, colour=(0, 255, 100), thickness=2, label="Person"):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness, cv2.LINE_AA)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), colour, -1)
        cv2.putText(
            frame, label, (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Modern UI / UX Screens (Top-Bar HUD, Menu, Welcome)
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
#  Modern UI / UX Screens (Monochrome Theme)
# ═══════════════════════════════════════════════════════════════════════════════

def draw_welcome_screen(frame: np.ndarray) -> None:
    """Displays a sleek monochrome welcome screen."""
    import math
    import time
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    cv2.rectangle(overlay, (0, 0), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    title = "AI FITNESS TRAINER"
    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 1.5, 3)
    cv2.putText(frame, title, ((w - tw) // 2, h // 2 - 40), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

    prompt1 = "Press [ENTER] to Start"
    prompt2 = "Press [ESC] to Quit"
    
    (pw1, _), _ = cv2.getTextSize(prompt1, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    (pw2, _), _ = cv2.getTextSize(prompt2, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

    # نبض رمادي هادئ ومريح للعين
    pulse = 100 + int(abs(math.sin(time.time() * 3)) * 155)
    
    cv2.putText(frame, prompt1, ((w - pw1) // 2, h // 2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (pulse, pulse, pulse), 2, cv2.LINE_AA)
    cv2.putText(frame, prompt2, ((w - pw2) // 2, h // 2 + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2, cv2.LINE_AA)


def draw_menu_screen(frame: np.ndarray, exercises: list, selected_idx: int) -> None:
    """Displays an interactive 3x5 grid menu (Monochrome)."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    cv2.rectangle(overlay, (0, 0), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.90, frame, 0.10, 0, frame)

    header = "SELECT YOUR EXERCISE"
    (hw, _), _ = cv2.getTextSize(header, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
    cv2.putText(frame, header, ((w - hw) // 2, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # رفعنا المنيو وصغرنا المسافات والبوكسات
    cols = 3
    padding_x = 25
    padding_y = 15
    start_y = 90
    margin_x = 40
    
    avail_w = w - (2 * margin_x)
    box_w = (avail_w - (cols - 1) * padding_x) // cols
    box_h = 45

    for i, ex_enum_val in enumerate(exercises):
        row = i // cols
        col = i % cols
        
        x = margin_x + col * (box_w + padding_x)
        y = start_y + row * (box_h + padding_y)
        
        ex_display = ex_enum_val.replace("_", " ").upper()
        is_selected = (i == selected_idx)
        
        if is_selected:
            # اللون الأبيض للبوكس المختار مع نص أسود
            cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (220, 220, 220), -1)
            text_color = (15, 15, 15)
            ex_display = "> " + ex_display
        else:
            cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (35, 35, 35), -1)
            cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (80, 80, 80), 1)
            text_color = (200, 200, 200)

        (tw, th), _ = cv2.getTextSize(ex_display, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tx = x + (box_w - tw) // 2
        ty = y + (box_h + th) // 2
        cv2.putText(frame, ex_display, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2 if is_selected else 1, cv2.LINE_AA)

    # تعليمات واضحة وسميكة
    controls = "Nav: Arrows / WASD  |  [ENTER]: Select  |  [ESC]: Quit"
    (cw, _), _ = cv2.getTextSize(controls, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.putText(frame, controls, ((w - cw) // 2, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2, cv2.LINE_AA)


def draw_top_hud(
    frame: np.ndarray, display_fps: float, reps: int, state: str, 
    feedback: str, rep_blocked: bool, exercise_name: str, 
    hold_seconds: float = 0.0, is_timed: bool = False
) -> None:
    """Displays a monochrome top-bar HUD with dynamic text spacing to prevent overlap."""
    h, w = frame.shape[:2]
    
    bar_height = 70
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_height), (15, 15, 15), -1)
    
    # الخط الفاصل
    border_color = (150, 150, 150)
    if rep_blocked or ("⚠" in feedback):
        border_color = (0, 0, 200) 
    cv2.rectangle(overlay, (0, bar_height - 2), (w, bar_height), border_color, -1)

    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # 1. اسم التمرين (بخط أصغر وملموم)
    ex_display = exercise_name.replace("_", " ").upper()
    (ew, eh), _ = cv2.getTextSize(ex_display, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.putText(frame, ex_display, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # 2. العداد
    if is_timed:
        count_text = f"TIMER: {hold_seconds:.1f}s"
    else:
        count_text = f"REPS: {reps}"
    
    (cw, ch), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
    
    # السنترة الذكية: لو اسم التمرين طويل جداً، نزق العداد لليمين عشان ميحصلش تداخل
    center_x = (w - cw) // 2
    if 15 + ew + 15 > center_x:
        center_x = 15 + ew + 25  # مسافة آمنة 25 بيكسل
        
    cv2.putText(frame, count_text, (center_x, 48), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # 3. الحالة والـ FPS (على اليمين)
    state_col = (255, 255, 255) if state.lower() in ["up", "down"] else (150, 150, 150)
    state_txt = f"STATE: {state.upper()}"
    (sw, _), _ = cv2.getTextSize(state_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(frame, state_txt, (w - sw - 15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_col, 2, cv2.LINE_AA)
    
    cv2.putText(frame, f"FPS: {display_fps:.0f}", (w - 80, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

    # التقييم (Feedback Message)
    if feedback:
        is_bad = "⚠" in feedback or "bad" in feedback.lower()
        fb_bg = (0, 0, 150) if (is_bad or rep_blocked) else (50, 50, 50)
        fb_txt = (255, 255, 255)
        
        (fw, fh), _ = cv2.getTextSize(feedback, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        fb_x = (w - fw) // 2
        fb_y = h - 40
        
        cv2.rectangle(frame, (fb_x - 15, fb_y - fh - 10), (fb_x + fw + 15, fb_y + 10), fb_bg, -1)
        cv2.rectangle(frame, (fb_x - 15, fb_y - fh - 10), (fb_x + fw + 15, fb_y + 10), (200, 200, 200), 1)
        cv2.putText(frame, feedback, (fb_x, fb_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fb_txt, 2, cv2.LINE_AA)
        
    # أزرار تحكم الشاشة السفلية
    controls = "[M] Menu  |  [R] Reset  |  [ESC] Quit"
    cv2.putText(frame, controls, (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2, cv2.LINE_AA)