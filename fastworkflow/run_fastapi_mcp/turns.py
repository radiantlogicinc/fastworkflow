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
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Callable, Optional

import fastworkflow
from fastworkflow.state_serialization import StateEncodingError
from fastworkflow.utils.logging import logger
from fastworkflow.utils.react import NoSuspendedAgentStateError

from .checkpoint import (
    STARTUP_FAILED,
    STARTUP_SUCCEEDED,
    STARTUP_SUSPENDED,
)
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


# Kinds worth keeping after they finish. Only ``/initialize`` looks an execution
# up by turn_key afterwards, and only while the runtime is still live (a client
# re-polling its own startup turn) — every other endpoint consults the registry
# only while its execution is active. There is no GET /turns/{turn_key}, so a
# retained record is NOT recoverable once its runtime is gone.
COLLECTABLE_TERMINAL_KINDS = frozenset({"initialize_startup"})

# Retained terminal executions are request-sized: 20 records is ~17 MB at a
# 450 KB payload. The only consumer needs ~6 to cover the age window at the
# observed request rate, so this is roughly 3x margin. In a burst of 21 startup
# completions the count binds before the age window and the oldest is dropped
# seconds after finishing; that is accepted rather than sizing for peak bursts.
MAX_RETAINED_STARTUP_TURNS = 20
TURN_RETENTION_SECONDS = 300.0

# Where workflows read the caller's credential from. Documented in the server
# README, so the read contract is preserved — what changed is that it is only
# present while an accepted turn is running, and is never checkpointed.
CREDENTIAL_CONTEXT_KEY = "http_bearer_token"


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
    # When set, ``_turn_json_response`` uses this instead of 500. Used for
    # client-conflict failures that happen inside work_fn (e.g. resume desync)
    # where ChannelBusyError cannot fire at admission time.
    http_status_on_error: Optional[int] = None
    traces: list[dict[str, Any]] = field(default_factory=list)
    user_id: Optional[str] = None
    # Carried on the execution, never written to shared workflow state by the
    # dependency that authenticated the request. Two requests on one channel
    # would otherwise interleave: B writes its token, B is rejected with 409,
    # and A — still running — reads B's credential.
    http_bearer_token: Optional[str] = None
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


class AdmissionClosedError(Exception):
    """Raised when the registry has stopped admitting turns (shutdown).

    Closing admission has to be atomic with respect to submission, or a turn
    registered just after an empty drain scan would be shut down underneath.
    """


class TurnRegistry:
    """In-process registry of turn executions, single-flight per channel.

    Live executions are kept until they finish. Finished ones are kept only if
    something can still look them up, and then only within a bounded count and
    age — otherwise every completed request would retain its payload for the
    lifetime of the process.
    """

    def __init__(
        self,
        *,
        collectable_kinds: frozenset[str] = COLLECTABLE_TERMINAL_KINDS,
        max_retained_terminal: int = MAX_RETAINED_STARTUP_TURNS,
        retention_seconds: float = TURN_RETENTION_SECONDS,
    ) -> None:
        self._by_key: dict[str, TurnExecution] = {}
        # channel_id -> turn_key of the live (non-terminal) execution.
        self._active_by_channel: dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._collectable_kinds = collectable_kinds
        self._max_retained_terminal = max_retained_terminal
        self._retention_seconds = retention_seconds
        self._admission_closed = False

    async def close_admission(self) -> None:
        """Stop admitting new turns, atomically with respect to submission.

        Taken under the same lock as ``start_or_get_active``, so shutdown cannot
        scan an empty registry and then have a turn registered behind it.
        """
        async with self._lock:
            self._admission_closed = True

    @property
    def admission_closed(self) -> bool:
        return self._admission_closed

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
        http_bearer_token: Optional[str] = None,
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
            if self._admission_closed:
                raise AdmissionClosedError(
                    f"server is shutting down; no new turns for channel {channel_id}"
                )

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
                http_bearer_token=http_bearer_token,
            )
            self._by_key[execn.turn_key] = execn
            self._active_by_channel[channel_id] = execn.turn_key
            # Launch the task only after the execution is fully built and the
            # pointer is in place (construction-order contract).
            try:
                execn.task = run_turn(execn)
            except BaseException:
                # An execution that never launched will never reach a terminal
                # state, and retention must never evict a non-terminal record —
                # so it would be immortal. Undo the insertion instead.
                self._by_key.pop(execn.turn_key, None)
                if self._active_by_channel.get(channel_id) == execn.turn_key:
                    self._active_by_channel.pop(channel_id, None)
                raise
            return execn

    async def clear_active(self, channel_id: str, turn_key: str) -> None:
        """Retire a finished execution: clear its active pointer and bound retention.

        This is one step, not two: ``asyncio.Lock`` is not reentrant, so a
        separate "now prune" call that re-entered ``_lock`` would deadlock inside
        a turn's ``finally``. Under the single lock, drop the record unless its
        kind is still collectable, give a retained record its TTL once, then
        remove expired and overflow records.

        Contains no ``await`` and does no store I/O: this lock gates turn
        submission for every channel, not just this one.
        """
        async with self._lock:
            if self._active_by_channel.get(channel_id) == turn_key:
                self._active_by_channel.pop(channel_id, None)

            execn = self._by_key.get(turn_key)
            if execn is None or not execn.is_terminal:
                return

            if execn.kind not in self._collectable_kinds:
                # Nothing else to sweep for: this record is gone, and the count
                # cap is enforced on every completion of a kind that is retained.
                self._by_key.pop(turn_key, None)
                logger.debug(
                    f"Retired turn {turn_key} (kind={execn.kind}): nothing looks "
                    f"up a finished {execn.kind}"
                )
                return

            if execn.ttl_expires_at is None:
                execn.ttl_expires_at = (execn.finished_at or _now()) + timedelta(
                    seconds=self._retention_seconds
                )

            if evicted := self._evict_expired() + self._evict_overflow():
                logger.debug(
                    f"Evicted {evicted} retained terminal turn(s) after {turn_key}"
                )

    def _evict_expired(self, now: Optional[datetime] = None) -> int:
        """Drop retained terminal executions whose age window has passed."""
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

    def _evict_overflow(self) -> int:
        """Drop the oldest retained terminal executions above the count cap.

        Age cleanup is opportunistic — an expired record can sit in an idle
        process — so the count is the hard bound on finished records.
        """
        retained = sorted(
            (execn for execn in self._by_key.values() if execn.is_terminal),
            key=lambda execn: (execn.finished_at or execn.created_at, execn.turn_key),
        )
        overflow = len(retained) - self._max_retained_terminal
        if overflow <= 0:
            return 0
        for execn in retained[:overflow]:
            self._by_key.pop(execn.turn_key, None)
        return overflow

    def evict_terminal(self, now: Optional[datetime] = None) -> int:
        """TTL eviction of terminal (DONE/LOST) executions; returns how many went.

        Retirement sweeps on every completion, so nothing in the server calls
        this. It stays because the count cap — not the age window — is the hard
        bound, and an operator or test may want to age records out on demand.
        """
        return self._evict_expired(now)


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
    awaiting = ctx.awaiting_user or (
        result is not None
        and result.status == fastworkflow.TurnStatus.AWAITING_USER
    )
    # A session mid-parameter-extraction is not awaiting_user -- its turn
    # completed with an error asking for the missing values -- but it holds the
    # partially extracted parameters, and clearing here would drop them the
    # moment it is evicted between turns.
    if awaiting or ctx.has_open_command():
        try:
            state = ctx.serialize_state(channel_id=runtime.channel_id)
        except StateEncodingError as exc:
            # Not writable losslessly, so do not write. The runtime stays live
            # and keeps the state in memory; a lossy snapshot would be worse
            # than no snapshot because restore would trust it.
            logger.warning(
                f"Suspended state for channel_id {runtime.channel_id} is not "
                f"losslessly encodable, so it was not persisted: {exc}"
            )
            return
        session_manager.session_state_store.save(runtime.channel_id, state)
    else:
        session_manager.session_state_store.clear(runtime.channel_id)


@contextlib.contextmanager
def installed_credential(runtime: "ChannelRuntime", token: Optional[str]):
    """Put the request's credential in shared workflow state for this turn only.

    Workflows are documented to read ``workflow_context['http_bearer_token']``,
    so it has to be there while their commands run — but only then, and only for
    the turn that was actually admitted. Installing it at lookup time instead is
    what let a rejected request's token leak into a running one.
    """
    # The bound app workflow, not get_active_workflow(): that reads a stack which
    # is only populated while a command is executing, so installing through it
    # from the event loop — which is where the old updater ran — silently no-ops.
    workflow = runtime.execution_context.app_workflow
    context = workflow.context if workflow is not None else None
    if token is None or context is None:
        yield
        return

    had_previous = CREDENTIAL_CONTEXT_KEY in context
    previous = context.get(CREDENTIAL_CONTEXT_KEY)
    context[CREDENTIAL_CONTEXT_KEY] = token
    try:
        yield
    finally:
        if had_previous:
            context[CREDENTIAL_CONTEXT_KEY] = previous
        else:
            context.pop(CREDENTIAL_CONTEXT_KEY, None)


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

            with installed_credential(runtime, execn.http_bearer_token):
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
    except NoSuspendedAgentStateError as exc:
        execn.error = str(exc)
        execn.http_status_on_error = 409
        logger.warning(
            f"Turn {execn.turn_key} (kind={execn.kind}, "
            f"channel={execn.channel_id}) resume conflict: {exc}"
        )
    except Exception as exc:
        execn.error = str(exc)
        logger.error(
            f"Turn {execn.turn_key} (kind={execn.kind}, "
            f"channel={execn.channel_id}) failed: {exc}"
        )
        traceback.print_exc()
    finally:
        execn.finished_at = _now()
        # Before the outcome becomes observable: this is the path /initialize's
        # startup turn takes, so it is the one that has to make the fact durable.
        _commit_startup_outcome(session_manager, runtime, execn)
        execn.exec_state = ExecState.DONE
        await registry.clear_active(execn.channel_id, execn.turn_key)
        execn.done_event.set()


async def run_owned_turn(
    runtime: "ChannelRuntime",
    registry: TurnRegistry,
    execn: TurnExecution,
    work: Callable[[], Any],
    session_manager: "ChannelSessionManager",
) -> None:
    """Own an *async* unit of work as a registered turn.

    ``_run_turn`` runs a blocking callable in an executor and reports only at the
    end, which streaming cannot use: it has to emit while the work runs. This is
    the same lifecycle contract for work that awaits — the runtime lock is held
    for the attempt, conversation persistence completes before ``DONE``, and the
    execution is retired under the registry lock.

    The point of registering streaming here is ownership, not delivery. A client
    that times out or disconnects stops *reading*; this task keeps the registry
    pointer and the runtime lock until the work actually finishes, so nothing can
    snapshot or close the context underneath it.
    """
    try:
        async with runtime.lock:
            execn.exec_state = ExecState.RUNNING
            execn.started_at = _now()

            with installed_credential(runtime, execn.http_bearer_token):
                execn.result = await work()

            save_conversation_incremental(
                runtime, extract_turns_from_history, logger
            )
    except NoSuspendedAgentStateError as exc:
        execn.error = str(exc)
        execn.http_status_on_error = 409
        logger.warning(
            f"Turn {execn.turn_key} (kind={execn.kind}, "
            f"channel={execn.channel_id}) resume conflict: {exc}"
        )
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
        # After the registry lock is released, never inside it: trimming does
        # store I/O, and holding that lock across a write would block turn
        # submission on every channel, not just this one.
        await session_manager.trim_live_sessions()


def _commit_startup_outcome(
    session_manager: "ChannelSessionManager",
    runtime: "ChannelRuntime",
    execn: TurnExecution,
) -> None:
    """Make the startup outcome durable before it becomes observable.

    A startup that completes without mutating workflow.context would otherwise
    leave nothing behind: the digest is unchanged, no checkpoint is written, and
    a restart replays a startup that already ran. So this fact is committed on
    its own schedule rather than riding on whether the context happened to
    change, and it distinguishes succeeded from failed and suspended — a boolean
    cannot say "attempted and raised".
    """
    if execn.kind != "initialize_startup":
        return

    if execn.error is not None:
        runtime.startup_state = STARTUP_FAILED
    elif runtime.execution_context.awaiting_user:
        runtime.startup_state = STARTUP_SUSPENDED
    else:
        runtime.startup_state = STARTUP_SUCCEEDED
    runtime.startup_idempotency_key = execn.idempotency_key

    try:
        session_manager.commit_startup_state(runtime)
    except Exception as exc:  # never fail the turn on a metadata commit
        logger.warning(
            f"Could not commit startup state for channel_id {runtime.channel_id} "
            f"({type(exc).__name__}: {exc}); a restart may replay startup"
        )


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
    http_bearer_token: Optional[str] = None,
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
        http_bearer_token=http_bearer_token,
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

def render_turn_response(execn: TurnExecution) -> tuple[int, dict[str, Any]]:
    """Render a (status_code, body) for a turn endpoint.

    * Deferred (QUEUED/RUNNING) -> 202 {turn_key, exec_state:"running"}.
    * Done with error          -> 200 {..., error} (caller may raise 500).
    * Done with result         -> 200 {turn_key, exec_state, status,
                                        failure_reason, success, answer,
                                        command_outputs, traces?}.

    Everything but ``turn_key``, ``exec_state``, and ``traces`` is the public
    ``TurnOutput`` projection. ``exec_state`` is transport (where the *work*
    is), ``status``/``failure_reason``/``success`` are the turn outcome; a
    failed turn is still a 200. ``turn_key`` here is the EXECUTION's key — the
    handle a deferred 202 is polled with — not ``TurnOutput.turn_key``, which
    is the workflow's own logical-turn key.

    As of v3.0 the legacy top-level ``command_responses`` list is no longer
    emitted; clients should read ``answer`` and
    ``command_outputs[*].command_response``.
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
        "failure_reason": result.failure_reason if result else None,
        "success": result.success if result else False,
        "answer": result.answer if result else "",
        "command_outputs": (
            [co.model_dump(mode="json") for co in result.command_outputs]
            if result
            else []
        ),
    }
    if execn.traces:
        body["traces"] = execn.traces
    return 200, body
