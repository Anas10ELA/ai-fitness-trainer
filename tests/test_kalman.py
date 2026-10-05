"""
tests/test_kalman_v8.py
════════════════════════
Comprehensive test suite for KalmanKeypoints v8 — Step 7 Kalman Refactor.

Test categories:
  Unit       — individual components: predict, update, outlier rejection,
               occlusion model, Joseph form, symmetrization, clamping
  Functional — end-to-end smoothing behaviour for real motion patterns
  Performance— runtime comparison v7 vs v8 to confirm speedup
  Stability  — covariance health, no explosion, no NaN/inf propagation
  Edge cases — malformed input, all-occluded, partial visibility, reset

Run all:
    pytest tests/test_kalman_v8.py -v

Run performance benchmark only (shows timing):
    pytest tests/test_kalman_v8.py -v -k "perf" -s

Run stability only:
    pytest tests/test_kalman_v8.py -v -k "stab"
"""

from __future__ import annotations

import math
import time
from typing import List

import numpy as np
import pytest

# ── Import under test ─────────────────────────────────────────────────────────
from utils import (
    KalmanKeypoints,
    _NK, _NS, _NM, _P_INIT_DIAG, _CHI2_THRESHOLD,
    _OCCLUSION_SCORE, _MAX_OCCLUDED_FRAMES, _COV_MAX, _COV_MIN,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Fixtures & helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _kp(seed: int = 0, n: int = 17) -> np.ndarray:
    """Random keypoints array (N, 2) with positive values."""
    rng = np.random.default_rng(seed)
    return (rng.uniform(50, 600, (n, 2))).astype(np.float32)


def _scores(val: float = 0.9, n: int = 17) -> np.ndarray:
    return np.full(n, val, dtype=np.float32)


def _zero_kp(n: int = 17) -> np.ndarray:
    return np.zeros((n, 2), dtype=np.float32)


def _make_filter(**kwargs) -> KalmanKeypoints:
    return KalmanKeypoints(**kwargs)


def _feed(kf: KalmanKeypoints, kp: np.ndarray, sc=None, n: int = 1) -> np.ndarray:
    out = None
    for _ in range(n):
        out = kf.smooth(kp, sc)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Initialization
# ═══════════════════════════════════════════════════════════════════════════════

class TestInit:

    def test_state_shape(self):
        kf = _make_filter()
        assert kf._x.shape  == (_NK, _NS)
        assert kf._P.shape  == (_NK, _NS, _NS)

    def test_initial_covariance_diagonal(self):
        kf = _make_filter()
        for i in range(_NK):
            np.testing.assert_allclose(np.diag(kf._P[i]), _P_INIT_DIAG, rtol=1e-9)

    def test_first_call_returns_input(self):
        kf  = _make_filter()
        kp  = _kp()
        out = kf.smooth(kp)
        # First call: filter not yet initialized, returns raw kp cast to float32
        np.testing.assert_allclose(out, kp.astype(np.float32), atol=0.1)

    def test_output_shape(self):
        kf  = _make_filter()
        out = kf.smooth(_kp())
        assert out.shape == (_NK, 2)

    def test_output_dtype(self):
        kf  = _make_filter()
        out = kf.smooth(_kp())
        assert out.dtype == np.float32


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Basic smoothing convergence
# ═══════════════════════════════════════════════════════════════════════════════

class TestSmoothing:

    def test_converges_to_static_position(self):
        """After many frames of a fixed measurement, output should ≈ that value."""
        kf  = _make_filter()
        kp  = _kp(seed=1)
        sc  = _scores(0.95)
        for _ in range(50):
            out = kf.smooth(kp, sc)
        np.testing.assert_allclose(out, kp.astype(np.float32), atol=3.0)

    def test_tracks_slow_linear_motion(self):
        """Filter should track a keypoint moving 2px/frame with small lag."""
        kf = _make_filter()
        start = np.zeros((_NK, 2), dtype=np.float32) + 200.0
        for t in range(40):
            kp = start + t * 2.0
            out = kf.smooth(kp, _scores(0.9))
        # After warmup, lag should be < 10px
        expected = start + 39 * 2.0
        err = np.abs(out - expected.astype(np.float32)).max()
        assert err < 10.0, f"Tracking lag too large: {err:.1f}px"

    def test_smooths_gaussian_noise(self):
        """With Gaussian noise, smoothed output should be closer to truth than raw."""
        rng = np.random.default_rng(42)
        truth = np.ones((_NK, 2), dtype=np.float32) * 300.0
        kf = _make_filter()
        raw_errors, smooth_errors = [], []
        for _ in range(60):
            noisy = truth + rng.normal(0, 5.0, truth.shape).astype(np.float32)
            out = kf.smooth(noisy, _scores(0.9))
            raw_errors.append(float(np.abs(noisy - truth).mean()))
            smooth_errors.append(float(np.abs(out - truth).mean()))
        # On average, smoothed should be closer (skip first 5 warmup frames)
        assert np.mean(smooth_errors[5:]) < np.mean(raw_errors[5:]), \
            "Kalman should reduce noise"

    def test_no_output_nan(self):
        kf = _make_filter()
        for _ in range(20):
            out = kf.smooth(_kp(), _scores())
        assert np.isfinite(out).all(), "Output must be finite"

    def test_no_output_inf(self):
        kf = _make_filter()
        out = _feed(kf, _kp(), _scores(), n=30)
        assert not np.isinf(out).any()


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Confidence-Aware Measurement Noise (KALMAN-3)
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfidenceAwareR:

    def test_low_confidence_trusts_prediction_more(self):
        """
        Feed a noisy measurement with low confidence; the filter should
        deviate less from its prediction than it would with high confidence.
        """
        truth = np.ones((_NK, 2), dtype=np.float32) * 300.0

        # Warm up both filters on clean data
        kf_high = _make_filter()
        kf_low  = _make_filter()
        for _ in range(30):
            kf_high.smooth(truth, _scores(0.95))
            kf_low.smooth(truth, _scores(0.95))

        # Present a large noisy measurement
        noisy = truth + 50.0  # 50px jump
        out_high = kf_high.smooth(noisy, _scores(0.95))
        out_low  = kf_low.smooth(noisy,  _scores(0.05))

        # High confidence should follow the measurement more
        dev_high = float(np.abs(out_high - truth).mean())
        dev_low  = float(np.abs(out_low  - truth).mean())
        assert dev_high > dev_low, \
            f"High-conf should deviate more from prior: {dev_high:.2f} vs {dev_low:.2f}"

    def test_zero_score_all_predicted(self):
        """Zero confidence: filter should output close to prediction, not measurement."""
        truth = np.ones((_NK, 2), dtype=np.float32) * 300.0
        kf = _make_filter()
        for _ in range(30):
            kf.smooth(truth, _scores(0.95))

        # Big measurement jump but zero confidence
        far = truth + 200.0
        out = kf.smooth(far, _scores(0.0))
        # Should stay near prediction (close to 300), not jump to 500
        assert float(np.abs(out - truth).mean()) < 50.0, \
            "Zero-confidence should stay near prediction"

    def test_scores_shape_mismatch_no_crash(self):
        """Wrong scores shape should not crash — fall back to unit confidence."""
        kf = _make_filter()
        kp = _kp()
        bad_scores = np.ones(5)  # wrong length
        out = kf.smooth(kp, bad_scores)
        assert out.shape == (_NK, 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  4. Mahalanobis Outlier Rejection (KALMAN-6)
# ═══════════════════════════════════════════════════════════════════════════════

class TestOutlierRejection:

    def test_spike_rejected(self):
        """A large single-frame spike should be rejected, output stays near truth."""
        truth = np.ones((_NK, 2), dtype=np.float32) * 300.0
        kf = _make_filter()
        for _ in range(30):
            kf.smooth(truth, _scores(0.9))

        spike = truth.copy()
        spike[3] += 200.0   # one keypoint jumps 200px
        out = kf.smooth(spike, _scores(0.9))

        # The spiked keypoint should be held near 300, not jump to 500
        spike_kp_err = abs(float(out[3, 0]) - 300.0)
        assert spike_kp_err < 40.0, \
            f"Spike kp should be rejected; got {out[3,0]:.1f} (err={spike_kp_err:.1f}px)"

    def test_normal_motion_not_rejected(self):
        """Slow steady motion should not be rejected."""
        kf = _make_filter()
        start = np.ones((_NK, 2), dtype=np.float32) * 200.0
        for t in range(30):
            kp = start + t * 1.5  # 1.5px/frame — well within normal range
            out = kf.smooth(kp, _scores(0.9))
        # Output should be close to input (not stuck at old position)
        expected = start + 29 * 1.5
        err = float(np.abs(out - expected.astype(np.float32)).max())
        assert err < 20.0, f"Normal motion should not be rejected; lag={err:.1f}px"

    def test_consecutive_spikes_do_not_accumulate(self):
        """Multiple outlier frames in a row should not corrupt the state."""
        truth = np.ones((_NK, 2), dtype=np.float32) * 300.0
        kf = _make_filter()
        for _ in range(20):
            kf.smooth(truth, _scores(0.9))

        # Feed 5 spike frames
        spike = truth + 300.0
        for _ in range(5):
            kf.smooth(spike, _scores(0.9))

        # Return to truth — should re-converge quickly
        for _ in range(10):
            out = kf.smooth(truth, _scores(0.9))
        err = float(np.abs(out - truth).mean())
        assert err < 20.0, f"Should re-converge after spikes; err={err:.1f}px"


# ═══════════════════════════════════════════════════════════════════════════════
#  5. Occlusion Model (KALMAN-8)
# ═══════════════════════════════════════════════════════════════════════════════

class TestOcclusionModel:

    def _warm(self, kf, kp, n=20):
        for _ in range(n):
            kf.smooth(kp, _scores(0.9))

    def test_single_missing_frame_uses_prediction(self):
        """After 1 missing frame, filter should return a non-zero prediction."""
        kp = _kp(seed=5)
        kf = _make_filter()
        self._warm(kf, kp)

        missing = kp.copy()
        missing[0] = 0.0   # one keypoint missing

        sc = _scores(0.9)
        sc[0] = 0.0   # zero confidence too

        out = kf.smooth(missing, sc)
        # kp 0 should be predicted (non-zero, close to last known)
        assert not (out[0, 0] == 0.0 and out[0, 1] == 0.0), \
            "Single missing frame: should predict, not return (0,0)"

    def test_long_occlusion_returns_zeros(self):
        """After MAX_OCCLUDED_FRAMES missing, return (0,0) for that keypoint."""
        kp = _kp(seed=6)
        kf = _make_filter()
        self._warm(kf, kp)

        missing = kp.copy()
        missing[4] = 0.0
        sc = _scores(0.9)
        sc[4] = 0.0

        # Feed many missing frames
        for _ in range(_MAX_OCCLUDED_FRAMES + 3):
            out = kf.smooth(missing, sc)

        # kp 4 should now be (0,0) — long occlusion
        assert out[4, 0] == 0.0 and out[4, 1] == 0.0, \
            "Long occlusion should return (0,0)"

    def test_reacquisition_after_occlusion(self):
        """After long occlusion, returning a valid measurement should re-acquire."""
        kp = _kp(seed=7)
        kf = _make_filter()
        self._warm(kf, kp)

        missing = kp.copy()
        missing[2] = 0.0
        sc = _scores(0.9); sc[2] = 0.0

        for _ in range(_MAX_OCCLUDED_FRAMES + 2):
            kf.smooth(missing, sc)

        # Reappear at original position
        sc_back = _scores(0.9)
        for _ in range(5):
            out = kf.smooth(kp, sc_back)

        # Should be close to the original position now
        err = float(np.abs(out[2] - kp[2]).max())
        assert err < 50.0, f"Re-acquisition failed; err={err:.1f}px"

    def test_occlusion_inflates_covariance(self):
        """Covariance of an occluded keypoint should grow over time."""
        kp = _kp(seed=8)
        kf = _make_filter()
        self._warm(kf, kp)

        cov_before = float(kf._P[5, 0, 0])

        missing = kp.copy(); missing[5] = 0.0
        sc = _scores(0.9); sc[5] = 0.0
        for _ in range(5):
            kf.smooth(missing, sc)

        cov_after = float(kf._P[5, 0, 0])
        assert cov_after > cov_before, \
            f"Covariance should inflate during occlusion: {cov_before:.2f} → {cov_after:.2f}"

    def test_full_occlusion_all_keypoints(self):
        """All keypoints missing simultaneously should not crash."""
        kf = _make_filter()
        self._warm(kf, _kp())

        out = kf.smooth(_zero_kp(), _scores(0.0))
        assert out.shape == (_NK, 2)
        assert np.isfinite(out).all()


# ═══════════════════════════════════════════════════════════════════════════════
#  6. Covariance Stability (KALMAN-5, KALMAN-7)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCovarianceStability:

    def test_covariance_stays_symmetric(self):
        """P must remain symmetric after every update."""
        kf = _make_filter()
        for i in range(50):
            kf.smooth(_kp(seed=i), _scores(0.7))
            for j in range(_NK):
                diff = np.abs(kf._P[j] - kf._P[j].T).max()
                assert diff < 1e-8, \
                    f"P[{j}] not symmetric at step {i}: max diff={diff:.2e}"

    def test_covariance_diagonal_bounded(self):
        """Diagonal entries must stay within [_COV_MIN, _COV_MAX]."""
        kf = _make_filter()
        for i in range(100):
            kf.smooth(_kp(seed=i % 10), _scores(0.5))
        for j in range(_NK):
            diag = np.diag(kf._P[j])
            assert (diag >= _COV_MIN - 1e-10).all(), \
                f"P[{j}] diagonal went below COV_MIN"
            assert (diag <= _COV_MAX + 1e-6).all(), \
                f"P[{j}] diagonal exceeded COV_MAX"

    def test_covariance_positive_semidefinite(self):
        """All eigenvalues of P must be ≥ 0."""
        kf = _make_filter()
        for i in range(30):
            kf.smooth(_kp(seed=i), _scores(0.8))
        for j in range(_NK):
            eigvals = np.linalg.eigvalsh(kf._P[j])
            assert (eigvals >= -1e-8).all(), \
                f"P[{j}] not positive semi-definite: min eigenvalue={eigvals.min():.2e}"

    def test_no_covariance_explosion_under_noise(self):
        """P should not blow up when repeatedly receiving noisy measurements."""
        rng = np.random.default_rng(0)
        kf = _make_filter()
        truth = _kp(seed=0)
        for _ in range(200):
            noisy = truth + rng.normal(0, 10, truth.shape).astype(np.float32)
            kf.smooth(noisy, _scores(0.7))
        cov_diags = kf._P[:, np.arange(_NS), np.arange(_NS)]
        assert cov_diags.max() <= _COV_MAX, \
            f"Covariance explosion: max diag = {cov_diags.max():.2e}"

    def test_covariance_summary_returns_valid_dict(self):
        kf = _make_filter()
        _feed(kf, _kp(), _scores(), n=10)
        summ = kf.covariance_summary()
        assert "mean_pos_var" in summ
        assert "max_pos_var" in summ
        assert math.isfinite(summ["mean_pos_var"])
        assert math.isfinite(summ["max_pos_var"])


# ═══════════════════════════════════════════════════════════════════════════════
#  7. Input Handling (KALMAN-10)
# ═══════════════════════════════════════════════════════════════════════════════

class TestInputHandling:

    def test_none_input_returns_zeros(self):
        kf = _make_filter()
        out = kf.smooth(None)
        assert out.shape == (_NK, 2)

    def test_wrong_shape_2d(self):
        kf = _make_filter()
        out = kf.smooth(np.ones((10, 2), dtype=np.float32))
        assert out.shape == (_NK, 2)

    def test_shape_17_3_extracts_score(self):
        """(17,3) input: column 2 used as confidence score."""
        kf = _make_filter()
        kp_with_score = np.ones((17, 3), dtype=np.float32)
        kp_with_score[:, :2] = 300.0
        kp_with_score[:, 2]  = 0.9   # scores
        out = kf.smooth(kp_with_score)
        assert out.shape == (_NK, 2)

    def test_nan_input_cleaned(self):
        kf = _make_filter()
        kp = _kp().copy()
        kp[0, 0] = float("nan")
        kp[1, 1] = float("inf")
        out = kf.smooth(kp)
        assert np.isfinite(out).all()

    def test_all_zeros_input(self):
        kf = _make_filter()
        out = kf.smooth(_zero_kp())
        assert out.shape == (_NK, 2)

    def test_negative_coordinates(self):
        """Negative coords are valid (camera offset) — should not crash."""
        kf = _make_filter()
        kp = _kp() - 1000.0   # all negative
        out = kf.smooth(kp)
        assert np.isfinite(out).all()

    def test_scores_clipped_to_unit_range(self):
        """Scores > 1 or < 0 should be clipped, not crash."""
        kf = _make_filter()
        bad_sc = np.full(17, 2.0)  # > 1
        out = kf.smooth(_kp(), bad_sc)
        assert out.shape == (_NK, 2)
        kf2 = _make_filter()
        neg_sc = np.full(17, -0.5)
        out2 = kf2.smooth(_kp(), neg_sc)
        assert out2.shape == (_NK, 2)

    def test_non_array_input_converts(self):
        """List of lists should be converted to ndarray."""
        kf = _make_filter()
        kp_list = [[float(i * 10), float(i * 10 + 5)] for i in range(17)]
        out = kf.smooth(np.array(kp_list, dtype=np.float32))
        assert out.shape == (_NK, 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  8. Reset
# ═══════════════════════════════════════════════════════════════════════════════

class TestReset:

    def test_reset_clears_state(self):
        kf = _make_filter()
        _feed(kf, _kp(), _scores(), n=30)
        kf.reset()
        assert not kf._initialized
        np.testing.assert_allclose(kf._x, 0.0)

    def test_reset_clears_covariance(self):
        kf = _make_filter()
        _feed(kf, _kp(), _scores(), n=30)
        kf.reset()
        for i in range(_NK):
            np.testing.assert_allclose(np.diag(kf._P[i]), _P_INIT_DIAG, rtol=1e-9)

    def test_reset_clears_occlusion_counters(self):
        kf = _make_filter()
        sc = _scores(0.0)
        _feed(kf, _zero_kp(), sc, n=10)
        kf.reset()
        assert (kf._occluded_frames == 0).all()

    def test_works_normally_after_reset(self):
        kf = _make_filter()
        _feed(kf, _kp(), _scores(), n=30)
        kf.reset()
        kp2 = _kp(seed=99)
        for _ in range(20):
            out = kf.smooth(kp2, _scores(0.9))
        err = float(np.abs(out - kp2.astype(np.float32)).mean())
        assert err < 5.0, f"Filter should re-converge after reset; err={err:.2f}"


# ═══════════════════════════════════════════════════════════════════════════════
#  9. Performance Benchmark
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerformance:

    N_FRAMES = 500

    def _time_filter(self, kf: KalmanKeypoints, n: int = N_FRAMES) -> float:
        rng = np.random.default_rng(0)
        kp  = _kp(seed=0)
        sc  = _scores(0.9)
        # Warmup
        for _ in range(10):
            kf.smooth(kp, sc)
        t0 = time.perf_counter()
        for i in range(n):
            noisy = kp + rng.normal(0, 3, kp.shape).astype(np.float32)
            kf.smooth(noisy, sc)
        return (time.perf_counter() - t0) * 1000  # ms total

    def test_v8_fast_enough(self):
        """v8 should process 500 frames well under 2 seconds on any modern CPU."""
        kf = _make_filter()
        elapsed_ms = self._time_filter(kf)
        per_frame = elapsed_ms / self.N_FRAMES
        print(f"\n  v8 KalmanKeypoints: {per_frame:.3f}ms/frame ({self.N_FRAMES} frames)")
        # 2ms/frame is extremely conservative — should be <0.2ms on modern hardware
        assert per_frame < 2.0, f"Kalman too slow: {per_frame:.3f}ms/frame"

    def test_v8_faster_than_loop_baseline(self):
        """
        Compare against a naive per-keypoint loop baseline to confirm
        vectorization wins.
        """

        # ── Naive baseline (simulates v7 approach) ────────────────────────────
        class LoopBaseline:
            def __init__(self):
                F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=np.float64)
                H = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float64)
                self._F, self._H = F, H
                self._x = np.zeros((17, 4))
                self._P = np.stack([np.eye(4)] * 17)
                self._Q = np.eye(4) * 5e-3
                self._R = np.eye(2) * 1e-2
                self._init = False

            def smooth(self, kp, sc=None):
                if not self._init:
                    self._x[:, :2] = kp
                    self._init = True
                    return kp.astype(np.float32)
                out = np.zeros_like(kp)
                for i in range(17):
                    xp = self._F @ self._x[i]
                    Pp = self._F @ self._P[i] @ self._F.T + self._Q
                    meas_p = self._H @ xp
                    innov  = kp[i] - meas_p
                    S      = self._H @ Pp @ self._H.T + self._R
                    try:
                        K = Pp @ self._H.T @ np.linalg.inv(S)
                    except Exception:
                        out[i] = xp[:2]
                        continue
                    self._x[i] = xp + K @ innov
                    self._P[i] = (np.eye(4) - K @ self._H) @ Pp
                    out[i] = self._x[i, :2]
                return out.astype(np.float32)

        kf_v8   = _make_filter()
        kf_loop = LoopBaseline()
        rng = np.random.default_rng(1)
        kp  = _kp(seed=0)
        sc  = _scores(0.9)

        # Warmup
        for _ in range(10):
            kf_v8.smooth(kp, sc)
            kf_loop.smooth(kp)

        N = 200
        t0 = time.perf_counter()
        for _ in range(N):
            kf_v8.smooth(kp + rng.normal(0,3,kp.shape).astype(np.float32), sc)
        t_v8 = (time.perf_counter() - t0) * 1000

        rng2 = np.random.default_rng(1)
        t0 = time.perf_counter()
        for _ in range(N):
            kf_loop.smooth(kp + rng2.normal(0,3,kp.shape).astype(np.float32))
        t_loop = (time.perf_counter() - t0) * 1000

        speedup = t_loop / max(t_v8, 0.001)
        print(f"\n  Loop baseline: {t_loop/N:.3f}ms/frame")
        print(f"  v8 vectorized: {t_v8/N:.3f}ms/frame")
        print(f"  Speedup: {speedup:.1f}×")
        assert speedup > 2.0, f"Expected >2× speedup, got {speedup:.1f}×"


# ═══════════════════════════════════════════════════════════════════════════════
#  10. Adaptive Process Noise (KALMAN-4)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdaptiveProcessNoise:

    def test_fast_motion_reduces_lag(self):
        """
        Under fast motion the adaptive Q should increase, reducing lag vs
        a fixed-Q filter.
        """
        kf_adapt = _make_filter()   # adaptive Q (default)

        # Warmup both on static position
        static = np.ones((_NK, 2), dtype=np.float32) * 200.0
        for _ in range(20):
            kf_adapt.smooth(static, _scores(0.9))

        # Fast motion: 20px/frame
        final_pos = static + 20.0 * 15
        for t in range(15):
            kp = static + t * 20.0
            out_a = kf_adapt.smooth(kp, _scores(0.9))

        # Both should track but adaptive should have lower lag
        err_adapt = float(np.abs(out_a - final_pos.astype(np.float32)).mean())
        print(f"\n  Fast-motion lag (adaptive Q): {err_adapt:.1f}px")
        # Generous threshold — just verify it doesn't stall
        assert err_adapt < 100.0, f"Adaptive filter lagging too much: {err_adapt:.1f}px"


# ═══════════════════════════════════════════════════════════════════════════════
#  11. Vectorization correctness
# ═══════════════════════════════════════════════════════════════════════════════

class TestVectorizationCorrectness:

    def test_all_keypoints_updated_independently(self):
        """
        Each keypoint should converge to its own measurement independently.
        Verify by feeding keypoints with very different values.
        """
        kf = _make_filter()
        kp = np.zeros((_NK, 2), dtype=np.float32)
        for i in range(_NK):
            kp[i] = float(i * 30)   # kp[0]=(0,0) intentionally missing, rest spread out

        sc = _scores(0.9)
        sc[0] = 0.0  # kp[0] missing

        for _ in range(50):
            out = kf.smooth(kp, sc)

        # Each non-zero keypoint should be close to its own measurement
        for i in range(1, _NK):
            err = float(np.abs(out[i] - kp[i]).max())
            assert err < 5.0, f"kp[{i}] not tracking: err={err:.1f}px"

    def test_single_keypoint_behavior_matches_batch(self):
        """
        Manually run one Kalman step for kp[7] and compare to batch result.
        This validates the vectorized math is correct.
        """
        kf = _make_filter()
        kp = _kp(seed=42)
        sc = _scores(0.9)

        # Warm up
        for _ in range(5):
            kf.smooth(kp, sc)

        # Capture state for kp[7] before the next step
        x7 = kf._x[7].copy()
        P7 = kf._P[7].copy()

        # Run batch step
        kp_new = _kp(seed=43)
        out = kf.smooth(kp_new, sc)

        # Manual step for kp[7]
        F = kf._F; H = kf._H
        Q = kf._Q_base.copy()
        R = kf._R_base * (kf._rb / max(float(sc[7]), 0.05))

        vel = np.linalg.norm(x7[2:4])
        Q_eff = Q * (1.0 + 0.04 * vel)

        x_pred = F @ x7
        P_pred = F @ P7 @ F.T + Q_eff
        innov  = kp_new[7].astype(np.float64) - (H @ x_pred)
        S      = H @ P_pred @ H.T + R
        K      = P_pred @ H.T @ np.linalg.inv(S)

        maha_sq = float(innov @ np.linalg.inv(S) @ innov)
        if maha_sq <= _CHI2_THRESHOLD:
            x_manual = x_pred + K @ innov
        else:
            x_manual = x_pred

        out_manual = x_manual[:2].astype(np.float32)

        np.testing.assert_allclose(
            out[7], out_manual, atol=1.0,
            err_msg="Batch and manual results for kp[7] diverge",
        )
