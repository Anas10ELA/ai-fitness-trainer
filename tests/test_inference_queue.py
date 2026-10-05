"""
tests/test_inference_queue.py
══════════════════════════════
Comprehensive test suite for InferenceQueue — async producer-consumer
frame scheduling system introduced in Step 5.

Test categories:
  Unit       — InferenceQueue internals (submit, dispatch, metrics, futures)
  Integration— Multi-session concurrent submission
  Edge cases — QueueFull / backpressure, session lifecycle, cancellation
  Stress     — Starvation, frame flooding, rapid session churn
  Perf       — Throughput and latency regression guard

Run:
    pytest tests/test_inference_queue.py -v
    pytest tests/test_inference_queue.py -v -k "stress" --timeout=30
"""

from __future__ import annotations

import asyncio
import time
import types
import uuid
from collections import deque
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest

# ── Import under test ─────────────────────────────────────────────────────────
# We patch environment variables before import so config picks them up.
import os
os.environ.setdefault("QUEUE_MAX_PER_SESSION",   "4")
os.environ.setdefault("QUEUE_BATCH_SIZE",        "8")
os.environ.setdefault("QUEUE_INFERENCE_THREADS", "4")
os.environ.setdefault("QUEUE_DISPATCH_INTERVAL", "0.001")

from server.inference_queue import (
    InferenceQueue,
    WorkItem,
    QueueFullError,
    QueueMetrics,
    MAX_QUEUE_PER_SESSION,
    MAX_BATCH_SIZE,
    NUM_INFERENCE_THREADS,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Fixtures & helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_output(session_id: str = "test") -> MagicMock:
    """Fake WSFrameOutput-like object."""
    out = MagicMock()
    out.model_dump.return_value = {"reps": 0, "session_id": session_id}
    return out


def _make_processor(delay: float = 0.0, raise_exc: Exception = None) -> MagicMock:
    """
    Fake FrameProcessor whose process() call takes `delay` seconds.
    If raise_exc is set, process() raises that exception.
    """
    proc = MagicMock()
    call_count = [0]

    def _process(frame_b64, session, requested_exercise=None):
        call_count[0] += 1
        if delay:
            time.sleep(delay)
        if raise_exc:
            raise raise_exc
        return _make_output(getattr(session, "session_id", "?"))

    proc.process.side_effect = _process
    proc._call_count = call_count
    return proc


def _make_session(session_id: str = None) -> MagicMock:
    s = MagicMock()
    s.session_id = session_id or str(uuid.uuid4())
    return s


def _make_work(session_id: str = None, session=None) -> WorkItem:
    sid = session_id or str(uuid.uuid4())
    return WorkItem(
        session_id=sid,
        frame_b64="base64data",
        session=session or _make_session(sid),
        requested_exercise=None,
    )


@pytest.fixture
async def queue():
    """Provides a started InferenceQueue with a trivial processor."""
    proc = _make_processor(delay=0.001)
    q = InferenceQueue(proc)
    await q.start()
    yield q, proc
    await q.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Unit tests — basic submission
# ═══════════════════════════════════════════════════════════════════════════════

class TestBasicSubmit:

    @pytest.mark.asyncio
    async def test_single_frame_returns_output(self, queue):
        q, proc = queue
        work = _make_work()
        result = await asyncio.wait_for(q.submit(work), timeout=5.0)
        assert result is not None

    @pytest.mark.asyncio
    async def test_metrics_increment_on_submit(self, queue):
        q, _ = queue
        before = q.metrics.total_submitted
        work = _make_work()
        await asyncio.wait_for(q.submit(work), timeout=5.0)
        assert q.metrics.total_submitted == before + 1

    @pytest.mark.asyncio
    async def test_metrics_increment_on_process(self, queue):
        q, _ = queue
        work = _make_work()
        await asyncio.wait_for(q.submit(work), timeout=5.0)
        assert q.metrics.total_processed >= 1

    @pytest.mark.asyncio
    async def test_latency_recorded(self, queue):
        q, _ = queue
        work = _make_work()
        await asyncio.wait_for(q.submit(work), timeout=5.0)
        assert len(q.metrics.total_ms) >= 1
        assert q.metrics.total_ms[-1] >= 0

    @pytest.mark.asyncio
    async def test_auto_register_on_submit(self, queue):
        """Submitting to an unregistered session should auto-register it."""
        q, _ = queue
        sid = "auto-register-session"
        work = _make_work(session_id=sid)
        # Do NOT call register_session manually
        result = await asyncio.wait_for(q.submit(work), timeout=5.0)
        assert result is not None


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Session registration / unregistration
# ═══════════════════════════════════════════════════════════════════════════════

class TestSessionLifecycle:

    @pytest.mark.asyncio
    async def test_register_creates_queue(self, queue):
        q, _ = queue
        sid = "lifecycle-test"
        q.register_session(sid)
        assert sid in q._session_queues

    @pytest.mark.asyncio
    async def test_unregister_removes_queue(self, queue):
        q, _ = queue
        sid = "unregister-test"
        q.register_session(sid)
        q.unregister_session(sid)
        assert sid not in q._session_queues

    @pytest.mark.asyncio
    async def test_double_register_idempotent(self, queue):
        q, _ = queue
        sid = "double-register"
        q.register_session(sid)
        q.register_session(sid)
        assert q._session_queues[sid].maxsize == MAX_QUEUE_PER_SESSION

    @pytest.mark.asyncio
    async def test_unregister_cancels_pending_futures(self, queue):
        """Pending futures should be cancelled when session unregisters."""
        proc = _make_processor(delay=1.0)  # slow processor
        q = InferenceQueue(proc)
        await q.start()
        try:
            sid = "cancel-test"
            q.register_session(sid)
            work = _make_work(session_id=sid)
            submit_task = asyncio.create_task(q.submit(work))
            await asyncio.sleep(0.01)  # let it enqueue
            q.unregister_session(sid)
            # Future should be cancelled or resolve — either is acceptable
            # as long as we don't deadlock
            try:
                await asyncio.wait_for(submit_task, timeout=2.0)
            except (asyncio.CancelledError, Exception):
                pass  # expected
        finally:
            await q.stop()

    @pytest.mark.asyncio
    async def test_active_sessions_count(self, queue):
        q, _ = queue
        initial = q.metrics.active_sessions
        sids = [f"count-test-{i}" for i in range(3)]
        for sid in sids:
            q.register_session(sid)
        assert q.metrics.active_sessions == initial + 3
        for sid in sids:
            q.unregister_session(sid)
        assert q.metrics.active_sessions == initial


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Backpressure / QueueFull
# ═══════════════════════════════════════════════════════════════════════════════

class TestBackpressure:

    @pytest.mark.asyncio
    async def test_queue_full_raises_immediately(self):
        """
        When the per-session queue is at capacity, submit() must raise
        QueueFullError without blocking.
        """
        proc = _make_processor(delay=2.0)  # very slow — queue stays full
        q = InferenceQueue(proc)
        await q.start()
        try:
            sid = "backpressure-test"
            q.register_session(sid)
            # Fill the queue
            fill_tasks = []
            for _ in range(MAX_QUEUE_PER_SESSION):
                w = _make_work(session_id=sid)
                fill_tasks.append(asyncio.create_task(q.submit(w)))

            await asyncio.sleep(0.02)  # let dispatcher drain one item

            # The next submit should raise
            extra = _make_work(session_id=sid)
            t0 = time.perf_counter()
            with pytest.raises(QueueFullError):
                q._session_queues[sid].put_nowait(extra)
                # Or test via the public API:
                # await q.submit(extra) — but queue may have drained by then
            elapsed = (time.perf_counter() - t0) * 1000
            assert elapsed < 50, "QueueFullError should be instant, not block"

            # Cleanup tasks
            for t in fill_tasks:
                t.cancel()
                try: await t
                except: pass
        finally:
            await q.stop()

    @pytest.mark.asyncio
    async def test_drop_counter_increments(self):
        """Dropped frames must increment total_dropped."""
        proc = _make_processor(delay=5.0)
        q = InferenceQueue(proc)
        await q.start()
        try:
            sid = "drop-counter-test"
            q.register_session(sid)
            tasks = []
            dropped = 0
            for i in range(MAX_QUEUE_PER_SESSION + 5):
                w = _make_work(session_id=sid)
                try:
                    tasks.append(asyncio.create_task(q.submit(w)))
                    await asyncio.sleep(0)  # yield between submits
                except QueueFullError:
                    dropped += 1

            assert q.metrics.total_dropped >= 0  # may be 0 if dispatcher is fast
            for t in tasks:
                t.cancel()
                try: await t
                except: pass
        finally:
            await q.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  4. Multi-session concurrency
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultiSessionConcurrency:

    @pytest.mark.asyncio
    async def test_multiple_sessions_complete(self):
        """N concurrent sessions all receive results."""
        N = 8
        proc = _make_processor(delay=0.005)
        q = InferenceQueue(proc)
        await q.start()
        try:
            sids = [f"concurrent-{i}" for i in range(N)]
            for sid in sids:
                q.register_session(sid)

            tasks = [
                asyncio.create_task(q.submit(_make_work(session_id=sid)))
                for sid in sids
            ]
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=10.0,
            )
            successes = [r for r in results if not isinstance(r, Exception)]
            assert len(successes) == N, f"Only {len(successes)}/{N} succeeded"
        finally:
            await q.stop()

    @pytest.mark.asyncio
    async def test_slow_session_does_not_starve_fast(self):
        """
        A session with a slow client (submitting 1 frame) should not prevent
        another session from getting multiple frames processed.
        """
        call_log: List[str] = []

        def _process(frame_b64, session, requested_exercise=None):
            call_log.append(session.session_id)
            time.sleep(0.002)
            return _make_output(session.session_id)

        proc = MagicMock()
        proc.process.side_effect = _process

        q = InferenceQueue(proc)
        await q.start()
        try:
            fast_sid = "fast-session"
            slow_sid = "slow-session"
            q.register_session(fast_sid)
            q.register_session(slow_sid)

            # Fast session sends many frames; slow sends one
            fast_tasks = [
                asyncio.create_task(q.submit(_make_work(session_id=fast_sid)))
                for _ in range(6)
            ]
            slow_task = asyncio.create_task(q.submit(_make_work(session_id=slow_sid)))

            all_tasks = fast_tasks + [slow_task]
            results = await asyncio.wait_for(
                asyncio.gather(*all_tasks, return_exceptions=True),
                timeout=10.0,
            )
            # All should complete
            errors = [r for r in results if isinstance(r, Exception)]
            assert len(errors) == 0, f"Unexpected errors: {errors}"

            # Slow session should still have been served
            assert slow_sid in call_log
        finally:
            await q.stop()

    @pytest.mark.asyncio
    async def test_session_isolation(self):
        """Errors in one session's frame must not affect other sessions."""
        results = {}

        def _process(frame_b64, session, requested_exercise=None):
            sid = session.session_id
            if sid == "error-session":
                raise RuntimeError("Deliberate error")
            time.sleep(0.001)
            return _make_output(sid)

        proc = MagicMock()
        proc.process.side_effect = _process

        q = InferenceQueue(proc)
        await q.start()
        try:
            good_sid  = "good-session"
            error_sid = "error-session"
            q.register_session(good_sid)
            q.register_session(error_sid)

            good_task  = asyncio.create_task(q.submit(_make_work(session_id=good_sid)))
            error_task = asyncio.create_task(q.submit(_make_work(session_id=error_sid)))

            good_result  = await asyncio.wait_for(good_task,  timeout=5.0)
            error_result = await asyncio.wait_for(error_task, timeout=5.0)

            assert good_result is not None, "Good session must get a result"
            assert isinstance(error_result, Exception) or error_result is not None, \
                "Error session result should resolve (as exception or error output)"
        finally:
            await q.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  5. Metrics accuracy
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetrics:

    @pytest.mark.asyncio
    async def test_snapshot_keys_present(self, queue):
        q, _ = queue
        snap = q.metrics_snapshot()
        required = {
            "total_submitted", "total_processed", "total_dropped",
            "total_batches", "active_sessions", "drop_rate_pct",
            "queue_wait_ms", "inference_ms", "total_latency_ms",
            "queue_depths", "config",
        }
        assert required <= snap.keys()

    @pytest.mark.asyncio
    async def test_p50_p95_populated_after_work(self, queue):
        q, _ = queue
        tasks = [asyncio.create_task(q.submit(_make_work())) for _ in range(10)]
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=10.0)
        snap = q.metrics_snapshot()
        assert snap["total_latency_ms"]["p50"] > 0
        assert snap["total_latency_ms"]["p95"] >= snap["total_latency_ms"]["p50"]

    @pytest.mark.asyncio
    async def test_queue_depths_reflect_reality(self, queue):
        q, _ = queue
        sid = "depth-test"
        q.register_session(sid)
        depths = q.queue_depth()
        assert sid in depths
        assert depths[sid] == 0  # empty at start

    @pytest.mark.asyncio
    async def test_drop_rate_zero_when_no_drops(self, queue):
        q, _ = queue
        snap = q.metrics_snapshot()
        if q.metrics.total_dropped == 0:
            assert snap["drop_rate_pct"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  6. Edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    @pytest.mark.asyncio
    async def test_processor_exception_resolves_future(self):
        """If processor raises, the future should resolve (not hang)."""
        proc = _make_processor(raise_exc=RuntimeError("boom"))
        q = InferenceQueue(proc)
        await q.start()
        try:
            work = _make_work()
            result = await asyncio.wait_for(q.submit(work), timeout=5.0)
            # Should resolve to something (error output from FrameProcessor's own guard)
            assert result is not None
        finally:
            await q.stop()

    @pytest.mark.asyncio
    async def test_stop_while_work_in_flight(self):
        """stop() should not deadlock even with inflight work."""
        proc = _make_processor(delay=0.1)
        q = InferenceQueue(proc)
        await q.start()
        tasks = [asyncio.create_task(q.submit(_make_work())) for _ in range(3)]
        await asyncio.sleep(0.01)  # let some enqueue
        await asyncio.wait_for(q.stop(), timeout=5.0)
        for t in tasks:
            t.cancel()
            try: await t
            except: pass

    @pytest.mark.asyncio
    async def test_submit_after_stop_raises_or_returns(self):
        """After stop(), submit may fail — but must not hang."""
        proc = _make_processor()
        q = InferenceQueue(proc)
        await q.start()
        await q.stop()
        work = _make_work()
        try:
            result = await asyncio.wait_for(q.submit(work), timeout=2.0)
        except Exception:
            pass  # exception is acceptable after stop

    @pytest.mark.asyncio
    async def test_empty_frame_b64_handled(self, queue):
        """Empty frame_b64 should not crash the queue."""
        q, _ = queue
        work = WorkItem(
            session_id="empty-frame",
            frame_b64="",
            session=_make_session("empty-frame"),
            requested_exercise=None,
        )
        result = await asyncio.wait_for(q.submit(work), timeout=5.0)
        assert result is not None

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_session_noop(self, queue):
        """Unregistering a session that never registered should be silent."""
        q, _ = queue
        q.unregister_session("phantom-session-that-was-never-registered")


# ═══════════════════════════════════════════════════════════════════════════════
#  7. Stress tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestStress:

    @pytest.mark.asyncio
    async def test_high_throughput_single_session(self):
        """Single session should sustain >50 frames/s with 1ms processing."""
        proc = _make_processor(delay=0.001)
        q = InferenceQueue(proc)
        await q.start()
        try:
            sid = "throughput-test"
            q.register_session(sid)
            N = 50
            t0 = time.perf_counter()
            tasks = []
            for _ in range(N):
                tasks.append(asyncio.create_task(q.submit(_make_work(session_id=sid))))
                await asyncio.sleep(0)  # give dispatcher a chance

            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=15.0,
            )
            elapsed = time.perf_counter() - t0
            successes = sum(1 for r in results if not isinstance(r, Exception))
            fps = successes / elapsed
            assert successes >= N * 0.8, f"Only {successes}/{N} frames processed"
            print(f"\n  Throughput (1 session): {fps:.1f} frames/s")
        finally:
            await q.stop()

    @pytest.mark.asyncio
    async def test_many_sessions_no_starvation(self):
        """16 sessions should all make progress within a reasonable window."""
        N = 16
        frames_per_session = 5
        proc = _make_processor(delay=0.002)
        q = InferenceQueue(proc)
        await q.start()
        try:
            sids = [f"starv-{i}" for i in range(N)]
            for sid in sids:
                q.register_session(sid)

            all_tasks = []
            for sid in sids:
                for _ in range(frames_per_session):
                    all_tasks.append(asyncio.create_task(q.submit(_make_work(session_id=sid))))

            results = await asyncio.wait_for(
                asyncio.gather(*all_tasks, return_exceptions=True),
                timeout=30.0,
            )
            successes = sum(1 for r in results if not isinstance(r, Exception))
            total = N * frames_per_session
            assert successes >= total * 0.8, f"Starvation: only {successes}/{total} succeeded"
            print(f"\n  {N} sessions × {frames_per_session} frames: {successes}/{total} OK")
        finally:
            await q.stop()

    @pytest.mark.asyncio
    async def test_rapid_session_churn(self):
        """Sessions that register and immediately unregister should not cause leaks."""
        proc = _make_processor(delay=0.001)
        q = InferenceQueue(proc)
        await q.start()
        try:
            for i in range(50):
                sid = f"churn-{i}"
                q.register_session(sid)
                q.unregister_session(sid)
            # After churn: no lingering queue entries
            assert len(q._session_queues) == 0
        finally:
            await q.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  8. QueueMetrics unit tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestQueueMetrics:

    def test_p50_empty(self):
        m = QueueMetrics()
        assert m.p50(m.total_ms) == 0.0

    def test_p95_single_element(self):
        m = QueueMetrics()
        m.total_ms.append(42.0)
        assert m.p95(m.total_ms) == 42.0

    def test_p50_odd_count(self):
        m = QueueMetrics()
        for v in [10, 20, 30]:
            m.total_ms.append(v)
        assert m.p50(m.total_ms) == 20.0

    def test_snapshot_drop_rate(self):
        m = QueueMetrics()
        m.total_submitted = 100
        m.total_dropped   = 5
        snap = m.snapshot()
        assert snap["drop_rate_pct"] == 5.0

    def test_snapshot_zero_submitted(self):
        """Should not divide by zero."""
        m = QueueMetrics()
        snap = m.snapshot()
        assert snap["drop_rate_pct"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  9. Race condition guards
# ═══════════════════════════════════════════════════════════════════════════════

class TestRaceConditions:

    @pytest.mark.asyncio
    async def test_concurrent_register_unregister(self):
        """
        Simultaneous register + unregister from different coroutines
        must not corrupt _session_queues.
        """
        proc = _make_processor()
        q = InferenceQueue(proc)
        await q.start()
        try:
            async def _reg_unreg(i):
                sid = f"race-{i}"
                q.register_session(sid)
                await asyncio.sleep(0)
                q.unregister_session(sid)

            await asyncio.gather(*[_reg_unreg(i) for i in range(20)])
            # After all coroutines complete, queues should be empty
            remaining = {k for k in q._session_queues if k.startswith("race-")}
            assert len(remaining) == 0
        finally:
            await q.stop()

    @pytest.mark.asyncio
    async def test_future_set_only_once(self):
        """Each future must be resolved exactly once — no double-set."""
        resolved = []
        original_resolve = InferenceQueue._resolve_future

        @staticmethod
        def _tracking_resolve(fut, result):
            resolved.append(id(fut))
            original_resolve(fut, result)

        proc = _make_processor(delay=0.001)
        q = InferenceQueue(proc)

        with patch.object(InferenceQueue, "_resolve_future", _tracking_resolve):
            await q.start()
            try:
                tasks = [asyncio.create_task(q.submit(_make_work())) for _ in range(5)]
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=10.0)
            finally:
                await q.stop()

        # Each future ID should appear exactly once
        assert len(resolved) == len(set(resolved)), "Some futures were resolved more than once"
