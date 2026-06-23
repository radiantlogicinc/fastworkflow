"""
Turns-based async execution engine for the run_fastapi_mcp server.

A long operation must never live or die with an HTTP request/response cycle.
Every unit of work runs as a *turn execution* owned by an in-process
``TurnRegistry``. Each endpoint *submits* a turn and waits a short, bounded
window: if the turn finishes in time it is returned inline (feels synchronous);
otherwise the request returns while the execution keeps running, recoverable by
polling (Step 2).

See ``docs/fastworkflow_turns_async_execution_design.md`` for the full design.

Key invariants (do not break these):
  * **wait-or-defer, never wait-or-abort** — the request's wait window timing
    out must NEVER cancel the execution. The execution runs as its own
    ``asyncio.Task`` that the request merely *waits on* (via ``done_event``),
    so a request timeout cannot affect it.
  * **per-channel active-execution pointer** is the single source of truth for
    liveness + idempotency, and the basis for the 409 "busy" guard (NOT
    ``runtime.lock.locked()`` — the lock is released while a request defers and
    across ``AWAITING_USER``).
  * **persist before DONE** — conversation/suspended state is persisted inside
    the turn-completion path, under ``runtime.lock``, before ``exec_state=DONE``.
  * **construction-order contract** — ``TurnRegistry.start_or_get_active`` is the
    sole owner of ``TurnExecution`` creation and task launch: it builds the
    execution (mint ``turn_key`` + ``done_event``) and inserts the pointer
    BEFORE launching the task, so no waiter can ever observe a half-built
    execution.
"""


from __future__ import annotations

import asyncio
import contextlib
import enum
import hashlib
import json
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Optional

import fastworkflow
from fastworkflow.utils.logging import logger

from .conversation_store import extract_turns_from_history
from .utils import (
    collect_trace_events,
    save_conversation_incremental,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids any import cycle
    from .utils import ChannelRuntime, ChannelSessionManager


# Work functions are synchronous, blocking callables run in an executor thread.
# They return the public TurnOutput for the logical turn.
WorkFn = Callable[[], "fastworkflow.TurnOutput"]


class ExecState(str, enum.Enum):
    """Where the async work is (execution lifecycle).

    Orthogonal to ``TurnStatus`` (the turn outcome). ``DONE`` means a
    ``TurnOutput`` (or an error) is available; read the outcome from
    ``TurnExecution.result.status``. ``LOST`` is the in-process-only
    "process restarted, record gone" state (Step 1/2; Step 3 removes it).
    """

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    LOST = "lost"


_TERMINAL_STATES = (ExecState.DONE, ExecState.LOST)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def compute_idempotency_key(channel_id: str, kind: str, *args: Any) -> str:
    """Stable key deduping retried submissions of the same logical turn.

    Keyed on ``hash(channel_id + kind + normalized args)``. A client/proxy
    retry with the same args rejoins the SAME execution rather than spawning a
    duplicate (and duplicate LLM spend).
    """
    payload = json.dumps(
        {"channel_id": channel_id, "kind": kind, "args": args},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class TurnExecution:
    """One unit of async work owned by the registry."""

    turn_key: str
    channel_id: str
    kind: str
    idempotency_key: str
    exec_state: ExecState = ExecState.QUEUED
    result: Optional["fastworkflow.TurnOutput"] = None
    error: Optional[str] = None
    traces: list[dict[str, Any]] = field(default_factory=list)
    user_id: Optional[str] = None
    task: Optional[asyncio.Task] = None
    done_event: asyncio.Event = field(default_factory=asyncio.Event)
    created_at: datetime = field(default_factory=_now)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    ttl_expires_at: Optional[datetime] = None

    @property
    def is_terminal(self) -> bool:
        return self.exec_state in _TERMINAL_STATES


class ChannelBusyError(Exception):
    """Raised when a channel already has a *different* active execution.

    Carries the in-flight execution so the caller can decide how to respond
    (turn endpoints map this to HTTP 409).
    """

    def __init__(self, execution: TurnExecution):
        self.execution = execution
        super().__init__(
            f"channel {execution.channel_id} already has an active execution "
            f"({execution.turn_key})"
        )


class TurnRegistry:
    """In-process registry of turn executions, single-flight per channel."""

    def __init__(self) -> None:
        self._by_key: dict[str, TurnExecution] = {}
        # channel_id -> turn_key of the live (non-terminal) execution.
        self._active_by_channel: dict[str, str] = {}
        self._lock = asyncio.Lock()

    def _active_execution(self, channel_id: str) -> Optional[TurnExecution]:
        active_key = self._active_by_channel.get(channel_id)
        if not active_key:
            return None
        execn = self._by_key.get(active_key)
        return None if execn is None or execn.is_terminal else execn

    def has_active(self, channel_id: str) -> bool:
        """Is there a live (QUEUED/RUNNING) execution for this channel?

        This — NOT ``runtime.lock.locked()`` — is the basis for the 409 busy
        guard. Reading the pointer is atomic in a single event loop (no await),
        so callers can check-then-act without a separate primitive.
        """
        return self._active_execution(channel_id) is not None

    def active_turn_key(self, channel_id: str) -> Optional[str]:
        execn = self._active_execution(channel_id)
        return execn.turn_key if execn else None

    def get(self, turn_key: str) -> Optional[TurnExecution]:
        return self._by_key.get(turn_key)

    async def start_or_get_active(
        self,
        channel_id: str,
        *,
        kind: str,
        idempotency_key: str,
        run_turn: Callable[[TurnExecution], asyncio.Task],
        user_id: Optional[str] = None,
    ) -> TurnExecution:
        """Sole owner of TurnExecution creation + task launch.

        Construction-order contract (under the registry lock):
          1. If an active execution with a matching idempotency_key exists,
             return it (the retry rejoins the SAME execution). A mismatch is a
             ChannelBusyError.
          2. Otherwise build a fresh TurnExecution (mint turn_key, allocate
             done_event, QUEUED) and insert it into ``_by_key`` +
             ``_active_by_channel`` BEFORE launching any task.
          3. Only then call ``run_turn(execn)`` to create the asyncio.Task with
             the fully-built execution, and store it on ``execn.task``.

        This guarantees a concurrent waiter that observes the pointer always
        sees an execution with a valid ``done_event``.
        """
        async with self._lock:
            existing = self._active_execution(channel_id)
            if existing is not None:
                if existing.idempotency_key == idempotency_key:
                    return existing
                raise ChannelBusyError(existing)

            execn = TurnExecution(
                turn_key=fastworkflow.mint_turn_key(),
                channel_id=channel_id,
                kind=kind,
                idempotency_key=idempotency_key,
                user_id=user_id,
            )
            self._by_key[execn.turn_key] = execn
            self._active_by_channel[channel_id] = execn.turn_key
            # Launch the task only after the execution is fully built and the
            # pointer is in place (construction-order contract).
            execn.task = run_turn(execn)
            return execn

    async def clear_active(self, channel_id: str, turn_key: str) -> None:
        """Clear the active pointer if it still points at ``turn_key``.

        Called when an execution reaches a terminal state. The execution remains
        in ``_by_key`` for polling/recovery (TTL eviction is Step 2).
        """
        async with self._lock:
            if self._active_by_channel.get(channel_id) == turn_key:
                self._active_by_channel.pop(channel_id, None)

    def evict_terminal(self, now: Optional[datetime] = None) -> int:
        """TTL eviction of terminal (DONE/LOST) executions. Step 2 hook.

        Returns the number of evicted entries. ``ttl_expires_at`` is unset in
        Step 1, so this is a no-op until Step 2 wires TTLs; included so the API
        surface is stable.
        """
        now = now or _now()
        evict_keys = [
            key
            for key, execn in self._by_key.items()
            if execn.is_terminal
            and execn.ttl_expires_at is not None
            and execn.ttl_expires_at <= now
        ]
        for key in evict_keys:
            self._by_key.pop(key, None)
        return len(evict_keys)


def _persist_after_turn(
    session_manager: "ChannelSessionManager",
    runtime: "ChannelRuntime",
    result: Optional["fastworkflow.TurnOutput"],
) -> None:
    """Save or clear durable suspended state after a turn (keyed off ctx).

    ``ctx.awaiting_user`` is authoritative; ``TurnOutput.status`` is checked as a
    consistent secondary signal.
    """
    ctx = runtime.execution_context
    if awaiting := ctx.awaiting_user or (
        result is not None
        and result.status == fastworkflow.TurnStatus.AWAITING_USER
    ):
        session_manager.session_state_store.save(
            runtime.channel_id,
            ctx.serialize_state(channel_id=runtime.channel_id),
        )
    else:
        session_manager.session_state_store.clear(runtime.channel_id)


async def _run_turn(
    runtime: "ChannelRuntime",
    registry: TurnRegistry,
    execn: TurnExecution,
    work_fn: WorkFn,
    session_manager: "ChannelSessionManager",
) -> None:
    """The only place that touches ``ctx`` for a turn.

    Acquire ``runtime.lock`` per attempt, run the blocking ``work_fn`` in the
    executor, collect traces, run persistence BEFORE marking DONE, then set
    ``exec_state=DONE`` and fire ``done_event``. The lock is released (by exiting
    the ``async with``) on a terminal TurnStatus OR on AWAITING_USER — never held
    across suspension (the registry pointer, not the lock, carries the execution).
    """
    loop = asyncio.get_running_loop()
    try:
        async with runtime.lock:
            execn.exec_state = ExecState.RUNNING
            execn.started_at = _now()

            result = await loop.run_in_executor(None, work_fn)
            execn.result = result

            # Destructive trace drain (Step 1). Step 2 replaces this with a
            # non-destructive per-execution replay buffer.
            try:
                execn.traces = collect_trace_events(runtime, user_id=execn.user_id)
            except Exception as trace_exc:  # best-effort; never fail the turn
                logger.warning(
                    f"Failed to collect traces for turn {execn.turn_key}: {trace_exc}"
                )

            # Persist BEFORE DONE so a poller never sees "done" with unsaved state.
            save_conversation_incremental(
                runtime, extract_turns_from_history, logger
            )
            _persist_after_turn(session_manager, runtime, result)
    except Exception as exc:
        execn.error = str(exc)
        logger.error(
            f"Turn {execn.turn_key} (kind={execn.kind}, "
            f"channel={execn.channel_id}) failed: {exc}"
        )
        traceback.print_exc()
    finally:
        execn.finished_at = _now()
        execn.exec_state = ExecState.DONE
        await registry.clear_active(execn.channel_id, execn.turn_key)
        execn.done_event.set()


async def submit_turn(
    runtime: "ChannelRuntime",
    registry: TurnRegistry,
    work_fn: WorkFn,
    session_manager: "ChannelSessionManager",
    *,
    wait_seconds: float,
    kind: str,
    idempotency_key: str,
    user_id: Optional[str] = None,
) -> TurnExecution:
    """Submit a turn and wait a bounded window (wait-or-defer).

    Single-flight: a retry with the same ``idempotency_key`` rejoins the SAME
    execution. On wait-window timeout the request returns the (still-running)
    execution; the execution is owned by the registry and keeps running.

    Raises ``ChannelBusyError`` if the channel already has a *different* active
    execution.
    """
    # The REGISTRY owns TurnExecution creation and task launch. The factory
    # receives the fully-built execution, so there is no caller-side forward
    # reference and no half-built-execution race (see construction-order
    # contract above).
    execn = await registry.start_or_get_active(
        runtime.channel_id,
        kind=kind,
        idempotency_key=idempotency_key,
        user_id=user_id,
        run_turn=lambda execn: asyncio.create_task(
            _run_turn(runtime, registry, execn, work_fn, session_manager)
        ),
    )
    with contextlib.suppress(asyncio.TimeoutError):
        # shield: the request's wait window timing out must NEVER cancel the
        # execution. (The execution is a separate task anyway; this is defensive.)
        await asyncio.wait_for(
            asyncio.shield(execn.done_event.wait()), wait_seconds
        )
    return execn


# ---------------------------------------------------------------------------
# Response rendering helpers (keep response shapes in one place)
# ---------------------------------------------------------------------------

def _command_responses_for(execn: TurnExecution) -> list[dict[str, Any]]:
    """Backward-compatible ``command_responses`` for a finished turn.

    Per-surface semantics:
      * ``invoke_agent`` — the synthesized final agent answer text.
      * everything else (assistant / action) — the last command's responses,
        preserving artifacts.
    """
    result = execn.result
    if result is None:
        return []
    if execn.kind != "invoke_agent" and result.command_outputs:
        return [
            cr.model_dump(mode="json")
            for cr in result.command_outputs[-1].command_responses
        ]
    return [{"response": result.answer, "success": result.success}]


def render_turn_response(execn: TurnExecution) -> tuple[int, dict[str, Any]]:
    """Render a (status_code, body) for a turn endpoint.

    * Deferred (QUEUED/RUNNING) -> 202 {turn_key, exec_state:"running"}.
    * Done with error          -> 200 {..., error} (caller may raise 500).
    * Done with result         -> 200 {turn_key, exec_state, status, success,
                                        answer, command_responses, command_outputs,
                                        traces?}.
    """
    if not execn.is_terminal:
        return 202, {
            "turn_key": execn.turn_key,
            "exec_state": ExecState.RUNNING.value,
        }

    if execn.error is not None:
        return 200, {
            "turn_key": execn.turn_key,
            "exec_state": execn.exec_state.value,
            "error": execn.error,
        }

    result = execn.result
    body: dict[str, Any] = {
        "turn_key": execn.turn_key,
        "exec_state": execn.exec_state.value,
        "status": result.status.value if result else None,
        "success": result.success if result else False,
        "answer": result.answer if result else "",
        "command_responses": _command_responses_for(execn),
        "command_outputs": (
            [co.model_dump(mode="json") for co in result.command_outputs]
            if result
            else []
        ),
    }
    if execn.traces:
        body["traces"] = execn.traces
    return 200, body
