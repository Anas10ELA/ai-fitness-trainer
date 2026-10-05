"""
server/inference_queue.py
═════════════════════════
Async Producer-Consumer Inference Queue  [NEW — Step 5]

Architecture Overview:
─────────────────────
  WebSocket handlers (producers)
       │  asyncio.Queue per session
       ▼
  InferenceQueue.submit()  ←── per-session bounded queue (backpressure)
       │
       ▼
  _dispatcher_loop()       ←── batches pending work items across sessions
       │
       ▼
  InferenceBatch           ←── list[WorkItem] grouped for one GPU pass
       │
       ▼
  _batch_worker_loop()     ←── ThreadPoolExecutor runs FrameProcessor
       │  asyncio.Future per work item
       ▼
  await result             ←── WebSocket handler gets WSFrameOutput

Key Design Decisions:
─────────────────────
  1. One asyncio.Queue per session (not one global queue) so a slow session
     cannot starve a fast one. Each per-session queue is bounded (default: 4
     frames) to apply backpressure at the client level.

  2. A single _dispatcher_loop() coroutine drains all per-session queues in
     round-robin order, packs up to MAX_BATCH_SIZE work items, and submits
     them as a batch to the thread pool. This is the *only* place GPU work is
     scheduled, so there is no locking needed at the session level.

  3. GPU inference is CPU-bound from asyncio's perspective (blocks a thread).
     We use a dedicated ThreadPoolExecutor with NUM_INFERENCE_THREADS workers.
     Each worker calls FrameProcessor.process() synchronously, which is safe
     because FrameProcessor has no shared mutable state between calls.

  4. Each WorkItem carries its own asyncio.Future. The WebSocket handler awaits
     this future, so back-pressure flows naturally: if the thread pool is busy,
     the future stays pending; if the per-session queue is full the producer
     gets a QueueFullError immediately and can drop the frame or tell the client
     to slow down.

  5. Metrics are maintained lock-free using asyncio (single-threaded event
     loop) for counters and a rolling deque for latency percentiles.

Environment Variables:
──────────────────────
  QUEUE_MAX_PER_SESSION   int  default 4    max queued frames per session
  QUEUE_BATCH_SIZE        int  default 8    max frames per GPU batch
  QUEUE_INFERENCE_THREADS int  default 4    thread-pool size
  QUEUE_DISPATCH_INTERVAL float default 0.005  dispatcher poll interval (s)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ── Config — sourced from centralised config (Step 9) ─────────────────────────

from config import cfg

MAX_QUEUE_PER_SESSION  = cfg.queue_max_per_session
MAX_BATCH_SIZE         = cfg.queue_batch_size
NUM_INFERENCE_THREADS  = cfg.queue_inference_threads
DISPATCH_INTERVAL      = cfg.queue_dispatch_interval

# Rolling window for latency percentiles
_LATENCY_WINDOW = 1000


# ═══════════════════════════════════════════════════════════════════════════════
#  Exceptions
# ═══════════════════════════════════════════════════════════════════════════════

class QueueFullError(Exception):
    """Raised when a session's per-frame queue has reached MAX_QUEUE_PER_SESSION."""


# ═══════════════════════════════════════════════════════════════════════════════
#  Data classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WorkItem:
    """One frame submitted by a WebSocket handler."""
    session_id:          str
    frame_b64:           str
    session:             Any                      # server.session_manager.Session
    requested_exercise:  Optional[str]
    enqueue_time:        float = field(default_factory=time.perf_counter)
    future:              "asyncio.Future[Any]" = field(default=None)   # set by InferenceQueue


@dataclass
class QueueMetrics:
    """Thread-safe (event-loop only) operational metrics."""
    total_submitted:   int = 0
    total_processed:   int = 0
    total_dropped:     int = 0      # frames dropped due to full queue
    total_batches:     int = 0
    active_sessions:   int = 0

    # Rolling latency deques (queue_wait_ms, inference_ms, total_ms)
    queue_wait_ms:   deque = field(default_factory=lambda: deque(maxlen=_LATENCY_WINDOW))
    inference_ms:    deque = field(default_factory=lambda: deque(maxlen=_LATENCY_WINDOW))
    total_ms:        deque = field(default_factory=lambda: deque(maxlen=_LATENCY_WINDOW))

    def p50(self, d: deque) -> float:
        if not d: return 0.0
        s = sorted(d)
        return s[len(s) // 2]

    def p95(self, d: deque) -> float:
        if not d: return 0.0
        s = sorted(d)
        return s[int(len(s) * 0.95)]

    def snapshot(self) -> Dict:
        return {
            "total_submitted":   self.total_submitted,
            "total_processed":   self.total_processed,
            "total_dropped":     self.total_dropped,
            "total_batches":     self.total_batches,
            "active_sessions":   self.active_sessions,
            "drop_rate_pct":     round(
                100 * self.total_dropped / max(self.total_submitted, 1), 2
            ),
            "queue_wait_ms": {
                "p50": round(self.p50(self.queue_wait_ms), 2),
                "p95": round(self.p95(self.queue_wait_ms), 2),
            },
            "inference_ms": {
                "p50": round(self.p50(self.inference_ms), 2),
                "p95": round(self.p95(self.inference_ms), 2),
            },
            "total_latency_ms": {
                "p50": round(self.p50(self.total_ms), 2),
                "p95": round(self.p95(self.total_ms), 2),
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  InferenceQueue
# ═══════════════════════════════════════════════════════════════════════════════

class InferenceQueue:
    """
    Central async producer-consumer queue for GPU inference.

    Lifecycle:
        queue = InferenceQueue(frame_processor)
        await queue.start()
        ...
        result = await queue.submit(work_item)
        ...
        await queue.stop()
    """

    def __init__(self, frame_processor) -> None:
        self._processor  = frame_processor
        self._executor   = ThreadPoolExecutor(
            max_workers  = NUM_INFERENCE_THREADS,
            thread_name_prefix = "inference",
        )

        # Per-session bounded queues  {session_id: asyncio.Queue}
        self._session_queues: Dict[str, asyncio.Queue] = {}

        # Ordered list of active session ids for round-robin dispatch
        self._session_order: List[str] = []

        # Metrics
        self.metrics = QueueMetrics()

        # Control
        self._running          = False
        self._dispatcher_task: Optional[asyncio.Task]  = None
        self._loop:            Optional[asyncio.AbstractEventLoop] = None

        log.info(
            "InferenceQueue init: max_per_session=%d batch=%d threads=%d",
            MAX_QUEUE_PER_SESSION, MAX_BATCH_SIZE, NUM_INFERENCE_THREADS,
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._loop    = asyncio.get_running_loop()
        self._running = True
        self._dispatcher_task = asyncio.create_task(
            self._dispatcher_loop(), name="inference-dispatcher"
        )
        log.info("InferenceQueue started")

    async def stop(self) -> None:
        self._running = False
        if self._dispatcher_task:
            self._dispatcher_task.cancel()
            try:
                await self._dispatcher_task
            except asyncio.CancelledError:
                pass
        self._executor.shutdown(wait=False)
        log.info("InferenceQueue stopped. Final metrics: %s", self.metrics.snapshot())

    # ── Session registration ──────────────────────────────────────────────────

    def register_session(self, session_id: str) -> None:
        """Call when a WebSocket session opens."""
        if session_id not in self._session_queues:
            self._session_queues[session_id] = asyncio.Queue(maxsize=MAX_QUEUE_PER_SESSION)
            self._session_order.append(session_id)
            self.metrics.active_sessions = len(self._session_queues)
            log.debug("InferenceQueue: registered session %s (total=%d)",
                      session_id, self.metrics.active_sessions)

    def unregister_session(self, session_id: str) -> None:
        """Call when a WebSocket session closes."""
        q = self._session_queues.pop(session_id, None)
        if q is not None:
            # Drain and cancel any pending futures to avoid leaks
            while not q.empty():
                try:
                    item: WorkItem = q.get_nowait()
                    if item.future and not item.future.done():
                        item.future.cancel()
                except asyncio.QueueEmpty:
                    break
        if session_id in self._session_order:
            self._session_order.remove(session_id)
        self.metrics.active_sessions = len(self._session_queues)
        log.debug("InferenceQueue: unregistered session %s (total=%d)",
                  session_id, self.metrics.active_sessions)

    # ── Producer API ──────────────────────────────────────────────────────────

    async def submit(self, item: WorkItem) -> Any:
        """
        Submit a frame for inference and await the result.

        Raises QueueFullError immediately (no blocking) when the per-session
        queue is at capacity — the WebSocket handler should drop the frame and
        optionally send a backpressure signal to the client.
        """
        self.metrics.total_submitted += 1

        q = self._session_queues.get(item.session_id)
        if q is None:
            # Session was never registered or already unregistered; auto-register
            self.register_session(item.session_id)
            q = self._session_queues[item.session_id]

        # Create a future the handler will await
        item.future = self._loop.create_future()

        try:
            q.put_nowait(item)
        except asyncio.QueueFull:
            self.metrics.total_dropped += 1
            item.future.cancel()
            raise QueueFullError(
                f"Session {item.session_id!r} queue full ({MAX_QUEUE_PER_SESSION} frames). "
                "Client is sending faster than the server can process."
            )

        # Await the future — resolved by the dispatcher when inference completes
        return await item.future

    # ── Dispatcher loop (async, single-threaded) ──────────────────────────────

    async def _dispatcher_loop(self) -> None:
        """
        Round-robin across all session queues, collect up to MAX_BATCH_SIZE
        items, and dispatch them to the thread pool.

        We intentionally do NOT await the batch futures here — we fire them
        with asyncio.ensure_future so the dispatcher stays free to collect the
        next batch immediately.
        """
        log.debug("InferenceQueue dispatcher started")
        while self._running:
            batch: List[WorkItem] = self._collect_batch()

            if not batch:
                # Nothing ready — yield control briefly
                await asyncio.sleep(DISPATCH_INTERVAL)
                continue

            self.metrics.total_batches += 1
            # Dispatch each work item to the thread pool concurrently
            asyncio.ensure_future(self._run_batch(batch))

            # Yield so WebSocket handlers can send back results
            await asyncio.sleep(0)

        log.debug("InferenceQueue dispatcher stopped")

    def _collect_batch(self) -> List[WorkItem]:
        """
        Drain up to MAX_BATCH_SIZE items from session queues in round-robin.
        Returns immediately (non-blocking); empty queues are skipped.
        """
        batch: List[WorkItem] = []
        if not self._session_order:
            return batch

        # Snapshot order to allow safe mutation during iteration
        order = list(self._session_order)
        idx   = 0
        while len(batch) < MAX_BATCH_SIZE and idx < len(order):
            sid = order[idx]
            q   = self._session_queues.get(sid)
            if q:
                try:
                    item = q.get_nowait()
                    batch.append(item)
                except asyncio.QueueEmpty:
                    pass
            idx += 1

        return batch

    # ── Batch execution (runs across thread pool) ─────────────────────────────

    async def _run_batch(self, batch: List[WorkItem]) -> None:
        """
        Submit all items in the batch to the thread pool concurrently.
        Each completes independently; results are set on item.future.
        """
        aws = [
            asyncio.get_running_loop().run_in_executor(
                self._executor,
                self._run_single,
                item,
            )
            for item in batch
        ]
        # gather — exceptions from individual items are caught inside _run_single
        await asyncio.gather(*aws, return_exceptions=True)

    def _run_single(self, item: WorkItem) -> None:
        """
        Synchronous inference for one work item.
        Runs in a ThreadPoolExecutor worker thread.
        Sets item.future with the result (or exception).
        MUST NOT touch any asyncio primitives directly; uses
        loop.call_soon_threadsafe to set the future safely.
        """
        t_queue_exit = time.perf_counter()
        queue_wait   = (t_queue_exit - item.enqueue_time) * 1000

        t_infer = time.perf_counter()
        try:
            output = self._processor.process(
                item.frame_b64,
                item.session,
                item.requested_exercise,
            )
            infer_ms  = (time.perf_counter() - t_infer) * 1000
            total_ms  = (time.perf_counter() - item.enqueue_time) * 1000

            # Record metrics (event-loop thread via call_soon_threadsafe)
            self._loop.call_soon_threadsafe(
                self._record_metrics, queue_wait, infer_ms, total_ms
            )
            # Resolve the future on the event loop
            self._loop.call_soon_threadsafe(
                self._resolve_future, item.future, output
            )
        except Exception as exc:
            log.error("InferenceQueue worker error for session=%s: %s",
                      item.session_id, exc, exc_info=True)
            self._loop.call_soon_threadsafe(
                self._reject_future, item.future, exc
            )

    # ── Future helpers (must be called on the event loop) ─────────────────────

    @staticmethod
    def _resolve_future(fut: "asyncio.Future", result: Any) -> None:
        if not fut.done():
            fut.set_result(result)

    @staticmethod
    def _reject_future(fut: "asyncio.Future", exc: Exception) -> None:
        if not fut.done():
            fut.set_exception(exc)

    def _record_metrics(self, queue_wait: float, infer_ms: float, total_ms: float) -> None:
        self.metrics.total_processed  += 1
        self.metrics.queue_wait_ms.append(queue_wait)
        self.metrics.inference_ms.append(infer_ms)
        self.metrics.total_ms.append(total_ms)

    # ── Monitoring ────────────────────────────────────────────────────────────

    def queue_depth(self) -> Dict[str, int]:
        """Return current depth of every per-session queue."""
        return {sid: q.qsize() for sid, q in self._session_queues.items()}

    def metrics_snapshot(self) -> Dict:
        snap = self.metrics.snapshot()
        snap["queue_depths"]      = self.queue_depth()
        snap["config"] = {
            "max_per_session":   MAX_QUEUE_PER_SESSION,
            "batch_size":        MAX_BATCH_SIZE,
            "inference_threads": NUM_INFERENCE_THREADS,
            "dispatch_interval": DISPATCH_INTERVAL,
        }
        return snap
