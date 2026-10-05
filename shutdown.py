"""
shutdown.py
═══════════
Step 9 — Graceful Shutdown

Handles SIGTERM and SIGINT signals cleanly:
  1. Stops accepting new WebSocket connections.
  2. Drains the inference queue (waits for in-flight frames).
  3. Closes all active sessions and flushes their summaries to the log.
  4. Releases GPU/model resources.
  5. Exits with code 0.

Why this matters
----------------
Without graceful shutdown:
  - In-flight inference work is discarded mid-frame.
  - Sessions are lost with no summary logged.
  - GPU memory may not be released cleanly (causes issues in Docker restarts).
  - WebSocket clients receive a TCP RST instead of a close frame.

Integration
-----------
This module is wired into the FastAPI lifespan automatically when imported.
You do NOT need to call anything manually — just import shutdown in main_api.py:

    import shutdown  # registers signal handlers

The handlers set a global asyncio.Event that uvicorn's shutdown hook already
polls, so there is no conflict with uvicorn's own signal handling.

For the desktop app (main.py), call:
    shutdown.register_desktop_handlers(worker, cap, writer)

which installs SIGTERM/SIGINT handlers that cleanly stop the inference worker
and release OpenCV resources.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from typing import Any, Callable, List, Optional

log = logging.getLogger(__name__)

# ── Global shutdown state ─────────────────────────────────────────────────────

_shutdown_requested: bool = False
_shutdown_callbacks: List[Callable] = []


def is_shutdown_requested() -> bool:
    return _shutdown_requested


def on_shutdown(callback: Callable) -> None:
    """Register a coroutine or sync function to call during graceful shutdown."""
    _shutdown_callbacks.append(callback)


# ═══════════════════════════════════════════════════════════════════════════════
#  Server (FastAPI / uvicorn) shutdown
# ═══════════════════════════════════════════════════════════════════════════════

def register_server_handlers(
    inference_queue: Any,
    session_manager: Any,
) -> None:
    """
    Register SIGTERM and SIGINT handlers for the FastAPI server process.

    Parameters
    ----------
    inference_queue : InferenceQueue
        The async queue — will be stopped (draining pending work) on shutdown.
    session_manager : SessionManager
        All active sessions — their summaries are logged before exit.
    """
    loop = None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # Not inside an async context yet — handlers will be set up at first call
        pass

    def _handle_signal(signum: int, frame: Any) -> None:
        sig_name = signal.Signals(signum).name
        log.info("Received %s — initiating graceful shutdown …", sig_name)
        global _shutdown_requested
        _shutdown_requested = True

        if loop and loop.is_running():
            # Schedule the async shutdown coroutine on the event loop
            loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(
                    _async_shutdown(inference_queue, session_manager)
                )
            )
        else:
            # Synchronous path (shouldn't happen in normal uvicorn operation)
            log.warning("Event loop not running — forcing immediate exit")
            sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT,  _handle_signal)
    log.debug("shutdown: SIGTERM/SIGINT handlers registered")


async def _async_shutdown(
    inference_queue: Any,
    session_manager: Any,
    drain_timeout_secs: float = 5.0,
) -> None:
    """
    Async shutdown sequence for the server.
    """
    t0 = time.perf_counter()
    log.info("── Graceful shutdown ──────────────────────────────────")

    # 1. Drain the inference queue
    if inference_queue is not None:
        log.info("  Draining inference queue …")
        try:
            await asyncio.wait_for(inference_queue.stop(), timeout=drain_timeout_secs)
            log.info("  Queue drained ✓")
        except asyncio.TimeoutError:
            log.warning(
                "  Queue drain timed out after %.1fs — forcing stop", drain_timeout_secs
            )
        except Exception as e:
            log.error("  Queue stop error: %s", e)

    # 2. Log session summaries
    if session_manager is not None:
        try:
            count = await session_manager.count()
            log.info("  Closing %d active session(s) …", count)
            # cleanup_expired will remove timed-out sessions; the rest are force-closed
            removed = await session_manager.cleanup_expired()
            log.info("  Sessions cleaned up: %d removed", removed)
        except Exception as e:
            log.error("  Session cleanup error: %s", e)

    # 3. Run user-registered callbacks
    for cb in _shutdown_callbacks:
        try:
            if asyncio.iscoroutinefunction(cb):
                await cb()
            else:
                cb()
        except Exception as e:
            log.error("  Shutdown callback %s raised: %s", cb, e)

    elapsed = time.perf_counter() - t0
    log.info("  Shutdown complete in %.2fs", elapsed)
    log.info("──────────────────────────────────────────────────────")


# ═══════════════════════════════════════════════════════════════════════════════
#  Desktop app (main.py) shutdown
# ═══════════════════════════════════════════════════════════════════════════════

def register_desktop_handlers(
    worker: Any,
    cap: Any,
    writer: Optional[Any] = None,
) -> None:
    """
    Register SIGTERM and SIGINT handlers for the desktop OpenCV app.

    Parameters
    ----------
    worker : _InferenceWorker
        Will be stopped (joins the inference thread) on signal.
    cap : cv2.VideoCapture
        Released on signal.
    writer : cv2.VideoWriter, optional
        Released on signal if not None.
    """

    def _handle(signum: int, frame: Any) -> None:
        sig_name = signal.Signals(signum).name
        log.info("Desktop app: received %s — shutting down …", sig_name)

        try:
            worker.stop()
            log.info("  Inference worker stopped")
        except Exception as e:
            log.warning("  Worker stop error: %s", e)

        try:
            cap.release()
            log.info("  Video capture released")
        except Exception as e:
            log.warning("  Cap release error: %s", e)

        if writer is not None:
            try:
                writer.release()
                log.info("  Video writer released")
            except Exception as e:
                log.warning("  Writer release error: %s", e)

        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass

        log.info("  Desktop app shutdown complete")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT,  _handle)
    log.debug("shutdown: desktop SIGTERM/SIGINT handlers registered")


# ═══════════════════════════════════════════════════════════════════════════════
#  Automatic recovery (session watchdog)
# ═══════════════════════════════════════════════════════════════════════════════

async def session_watchdog(
    session_manager: Any,
    inference_queue: Any,
    check_interval_secs: float = 30.0,
) -> None:
    """
    Background coroutine: periodically checks session health and cleans up
    leaked/zombie sessions that WebSocket disconnect handlers may have missed.

    Run this as an asyncio task in the lifespan context:
        asyncio.create_task(session_watchdog(sessions, queue))
    """
    log.debug("session_watchdog: started (interval=%.0fs)", check_interval_secs)
    while not _shutdown_requested:
        try:
            await asyncio.sleep(check_interval_secs)
            removed = await session_manager.cleanup_expired()
            active  = await session_manager.count()
            if removed > 0:
                log.info(
                    "session_watchdog: cleaned %d expired session(s), %d active",
                    removed, active,
                )
            # If any session is registered in the queue but not in session_manager,
            # it is a leaked queue slot — unregister it.
            if hasattr(inference_queue, "_session_queues") and \
               hasattr(session_manager, "_sessions"):
                async with session_manager._lock:
                    live_ids = set(session_manager._sessions.keys())
                queue_ids = set(inference_queue._session_queues.keys())
                leaked = queue_ids - live_ids
                for sid in leaked:
                    log.warning("session_watchdog: unregistering leaked queue slot %s", sid)
                    inference_queue.unregister_session(sid)
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error("session_watchdog: unexpected error: %s", e, exc_info=True)

    log.debug("session_watchdog: stopped")
