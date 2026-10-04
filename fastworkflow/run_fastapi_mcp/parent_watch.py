"""Parent-liveness watch for chatbot-spawned FastAPI servers.

When ``fastworkflow run_chatbot`` starts the FastAPI child with
``start_new_session=True``, SIGHUP/SIGKILL/OOM on the chatbot skip
``terminate_server``. This module lets the child notice the parent is gone
and shut itself down. Kept import-light so tests can exercise it without
loading uvicorn/fastapi.
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def parent_is_gone(parent_pid: int) -> bool:
    """True when *parent_pid* no longer appears to be our live parent."""
    if parent_pid <= 0:
        return True
    try:
        if os.getppid() != parent_pid:
            return True
    except OSError:
        return True
    try:
        os.kill(parent_pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    except OSError:
        return True
    return False


def watch_parent_until_gone(
    parent_pid: int,
    *,
    interval_s: float = 1.5,
    grace_s: float = 5.0,
    shutdown: Optional[Callable[[], None]] = None,
) -> None:
    """Block until the parent is gone, then request process shutdown.

    Default shutdown sends SIGTERM to this process (the same signal uvicorn
    treats as a graceful stop). If the process is still alive after *grace_s*,
    exits hard via ``os._exit``.
    """

    def _default_shutdown() -> None:
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except OSError:
            os._exit(1)

    stop = shutdown or _default_shutdown
    while True:
        if parent_is_gone(parent_pid):
            logger.warning(
                "parent pid %s is gone; shutting down FastAPI server",
                parent_pid,
            )
            try:
                stop()
            except Exception:
                os._exit(1)
            deadline = time.monotonic() + grace_s
            while time.monotonic() < deadline:
                time.sleep(0.1)
            os._exit(1)
        time.sleep(interval_s)


def start_parent_liveness_watch(
    parent_pid: Optional[int],
    *,
    interval_s: float = 1.5,
    grace_s: float = 5.0,
) -> Optional[threading.Thread]:
    """Start a daemon thread that exits the process when *parent_pid* dies.

    No-op when *parent_pid* is None (servers started outside the chatbot).
    """
    if parent_pid is None:
        return None
    thread = threading.Thread(
        target=watch_parent_until_gone,
        kwargs={
            "parent_pid": parent_pid,
            "interval_s": interval_s,
            "grace_s": grace_s,
        },
        name="fastapi-parent-watch",
        daemon=True,
    )
    thread.start()
    return thread
