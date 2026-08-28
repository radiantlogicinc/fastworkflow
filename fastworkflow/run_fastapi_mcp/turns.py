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
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

import fastworkflow
from fastworkflow.state_serialization import StateEncodingError
from fastworkflow.turn import TurnStatus
from fastworkflow.utils.logging import logger
from fastworkflow.utils.react import NoSuspendedAgentStateError

from .checkpoint import (
    STARTUP_FAILED,
    STARTUP_SUCCEEDED,
    STARTUP_SUSPENDED,
)
from fastworkflow.conversation_labeling import generate_topic_and_summary
from .utils import (
    collect_trace_events,
    trim_conversation_window,
    try_ensure_topic_and_summary,
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


# Kinds worth keeping after they finish. ``/initialize`` re-polls its startup
# turn by key while the runtime is live, so startup records are retained.
# Every other kind is dropped the moment it finishes: ``GET /turns/{turn_key}``
# serves those completed turns from the observability store instead (by the
# LOGICAL turn key — the 202 body carries it once known [R9]), so retaining the
# request-sized payload in memory would duplicate the durable record for no
# reader. The memory bound below is unchanged by the GET surface.
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

# Turn kinds whose completion is a chat exchange worth naming a conversation
# after. ``perform_action`` is a programmatic dispatch and
# ``initialize_startup`` is the workflow greeting itself; a title generated from
# either describes the client's wiring rather than anything the user said, and
# a channel that only ever initializes must cost no LLM call at all.
LABELABLE_TURN_KINDS = frozenset(
    {"invoke_agent", "invoke_agent_stream", "invoke_assistant"}
)

# Turn outcomes a label may be generated from. An allow-list, so a status added
# later spends no LLM call until somebody decides that it should.
#
# AWAITING_USER is absent, and that is the point of gating on status at all:
# persistence below runs unconditionally once ``work_fn`` returns, including on
# suspension, so a turn that stopped at ``ask_user`` has already recorded half an
# exchange. Labeling there titles a conversation from a clarifying question.
#
# FAILED is present because it is terminal. Nothing will complete that turn
# later, the exchange it did produce is durably recorded like any other, and
# skipping it would leave a conversation whose first turn failed with no title
# and no further trigger until its next refresh milestone.
LABELABLE_TURN_STATUSES = frozenset({TurnStatus.COMPLETED, TurnStatus.FAILED})


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
    # [R9] execution↔logical key identity. ``turn_key`` above is the EXECUTION
    # key (the registry's polling handle). ``logical_turn_key`` is the
    # workflow's own logical-turn key (``TurnOutput.turn_key`` — the
    # observability store's ``turns``/``spans`` key), recorded here once known
    # so deferred 202 bodies can carry it and GET /turns can resolve either.
    logical_turn_key: Optional[str] = None
    # ``ctx.current_turn_key`` snapshotted under ``runtime.lock`` before the
    # work began. ONCE THE SNAPSHOT EXISTS (exec_state RUNNING), a *different*
    # value observed later can only be the key ``_begin_turn`` minted for THIS
    # execution (single-flight guarantees no other turn runs on the channel).
    # While still QUEUED there is no snapshot and ``current_turn_key`` may hold
    # the previous turn's key — resolve_logical_turn_key guards on that.
    pre_turn_key: Optional[str] = None
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

    def get_by_key_or_logical(self, key: str) -> Optional[TurnExecution]:
        """Resolve *key* as an execution key OR a logical turn key [R9].

        ``GET /turns/{turn_key}`` accepts both: the execution key a deferred
        202 handed out, and the workflow's logical key (``TurnOutput.turn_key``)
        that the observability store is keyed by. The registry is bounded (one
        live execution per busy channel plus the retained startup records), so
        the logical-key fallback is a plain scan — no second index to keep
        consistent with eviction. No await, so reading is atomic in a single
        event loop, like ``has_active``.
        """
        execn = self._by_key.get(key)
        if execn is not None:
            return execn
        return next(
            (
                execn
                for execn in self._by_key.values()
                if execn.logical_turn_key == key
            ),
            None,
        )

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


async def _label_conversation_after_turn(
    runtime: "ChannelRuntime",
    execn: TurnExecution,
    turns_appended: int,
) -> None:
    """Fill this conversation's topic/summary, off the finished turn's critical path.

    WHERE THIS RUNS, AND WHY IT IS NOT NEXT TO THE INCREMENTAL SAVE (fix-dzs.6,
    R7). Callers invoke this as the very last act of the turn's own task: after
    ``exec_state=DONE``, after the registry pointer has been cleared, and after
    ``done_event`` is set. All three matter:

      * the shutdown drain does not wait on it. ``busy_channel_ids`` is the union
        of leases, the registry pointer and ``runtime.lock``; by this point the
        channel holds none of them, so an LLM round trip started here is outside
        the 30 s termination grace period that this epic exists to protect.
      * the request that submitted the turn is already free to answer. The first
        thing that can block below is an executor await, which hands the loop
        back, so the response is produced while the generation runs.
      * a concurrent request on this channel is not answered 409 for the length
        of a round trip, because ``_reject_if_busy`` reads the same cleared
        pointer.

    The placement the upstream issue asked for - beside the incremental save,
    inside ``runtime.lock``, before DONE - has none of those three properties.

    WHAT THAT COSTS, AND WHY IT IS SAFE ANYWAY. Once the pointer is cleared,
    nothing pins this runtime: ``trim_live_sessions`` may retire it and call
    ``execution_context.close()`` while the generation is still running. Pinning
    it instead - a session lease, or delaying the pointer clear - would put the
    channel straight back into the drain and undo the paragraph above. So the
    labeling path is built not to care: ``ensure_topic_and_summary`` reaches only
    the durable conversation record, through ``runtime.observability_store``,
    which is a handle to a SQLite file that retirement neither owns nor closes,
    and the conversation id it writes under is captured before it generates. A
    label landing after its session was retired is correct - the conversation is
    durable, the session was only a cache of it.

    That handle IS resolved off ``runtime.execution_context`` (the sink lives on
    the WEC), so it must be taken while the runtime is still live rather than
    re-read later. ``ensure_topic_and_summary`` reads it once, up front, for
    exactly this reason; do not move that read after an await.
    """
    if execn.kind not in LABELABLE_TURN_KINDS:
        return
    result = execn.result
    if execn.error is not None or result is None:
        # The work raised (or, defensively, reported no outcome at all), so
        # whatever reached the conversation is a fragment of an exchange that
        # never finished, and there is no status to gate on.
        return
    if result.status not in LABELABLE_TURN_STATUSES:
        return

    await try_ensure_topic_and_summary(
        runtime, generate_topic_and_summary, turns_appended=turns_appended
    )


def _turns_appended(runtime: "ChannelRuntime") -> int:
    """How many usable turns this attempt made durable — 0 or 1 (ruling I10).

    Replaces the count the incremental legacy save used to return. It reads the
    emit ack rather than the history length, so a turn whose record degraded to
    the queue does not advance the label schedule past a milestone that nothing
    can yet summarize.
    """
    ctx = runtime.execution_context
    if not ctx.last_turn_added_memory:
        return 0
    return 1 if ctx.last_turn_record_stored else 0


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
    # How many turns this attempt made durable, read by the labeling hook in the
    # finally to decide whether the conversation crossed a refresh milestone. A
    # turn whose work raised never reaches the save and leaves this at 0.
    turns_appended = 0
    try:
        async with runtime.lock:
            execn.exec_state = ExecState.RUNNING
            execn.started_at = _now()

            # [R9] record the execution↔logical mapping as early as knowable,
            # under the lock, before the worker thread exists (no race). A
            # resume continues the SAME logical turn (A30.2), so its key is
            # already current; a fresh turn's key is minted inside work_fn and
            # is picked up by resolve_logical_turn_key / the result below.
            ctx = runtime.execution_context
            execn.pre_turn_key = ctx.current_turn_key
            if ctx.awaiting_user:
                execn.logical_turn_key = ctx.current_turn_key

            with installed_credential(runtime, execn.http_bearer_token):
                result = await loop.run_in_executor(None, work_fn)
            execn.result = result
            if result is not None:
                execn.logical_turn_key = result.turn_key

            # Destructive trace drain (Step 1). Step 2 replaces this with a
            # non-destructive per-execution replay buffer.
            try:
                execn.traces = collect_trace_events(runtime, user_id=execn.user_id)
            except Exception as trace_exc:  # best-effort; never fail the turn
                logger.warning(
                    f"Failed to collect traces for turn {execn.turn_key}: {trace_exc}"
                )

            # The turn record was written synchronously inside work_fn's
            # finalize (Phase 7 §2.4), so by here it is already durable and a
            # poller can never see "done" with unsaved state. What is left is
            # windowing the in-memory history, which defers itself if that
            # write degraded to the queue (ruling I2).
            turns_appended = _turns_appended(runtime)
            trim_conversation_window(runtime, logger)
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
        # Last, and outside everything above: see _label_conversation_after_turn.
        await _label_conversation_after_turn(runtime, execn, turns_appended)


async def run_owned_turn(
    runtime: "ChannelRuntime",
    registry: TurnRegistry,
    execn: TurnExecution,
    work: Callable[[], Any],
    session_manager: "ChannelSessionManager",
    on_done: Optional[Callable[[], Awaitable[None]]] = None,
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

    A streaming turn is a chat turn, so it labels its conversation like any other
    (see _label_conversation_after_turn).

    ``on_done`` is awaited once the turn is observably finished and before that
    labeling, and exists so a caller can close out its own delivery first. The
    streaming endpoint passes the callback that flushes the end of its response
    body: without it, EOF waited on the label, so an MCP client reading the whole
    stream paid the generation in its tool call and a client with a short
    idle-read timeout could fail a turn whose answer it had already received. It
    is deliberately AFTER the error paths below, so a caller can still see
    ``execn.error`` when deciding what to deliver.
    """
    turns_appended = 0
    try:
        async with runtime.lock:
            execn.exec_state = ExecState.RUNNING
            execn.started_at = _now()

            # [R9] same execution↔logical capture as _run_turn (see there).
            ctx = runtime.execution_context
            execn.pre_turn_key = ctx.current_turn_key
            if ctx.awaiting_user:
                execn.logical_turn_key = ctx.current_turn_key

            with installed_credential(runtime, execn.http_bearer_token):
                execn.result = await work()
            if execn.result is not None:
                execn.logical_turn_key = execn.result.turn_key

            turns_appended = _turns_appended(runtime)
            trim_conversation_window(runtime, logger)
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
        if on_done is not None:
            # Before the trim and the label, both of which do I/O the caller's
            # client should not be waiting on. Its own try: a caller's delivery
            # failing must not skip this function's housekeeping, and an exception
            # escaping a finally would surface as a lost task exception rather
            # than anything actionable.
            try:
                await on_done()
            except Exception as exc:
                logger.warning(
                    f"on_done for turn {execn.turn_key} (kind={execn.kind}, "
                    f"channel={execn.channel_id}) failed "
                    f"({type(exc).__name__}: {exc})"
                )
        # After the registry lock is released, never inside it: trimming does
        # store I/O, and holding that lock across a write would block turn
        # submission on every channel, not just this one.
        await session_manager.trim_live_sessions()
        # And after the trim, not before: labeling can take an LLM round trip,
        # and delaying the trim by that would hold the session cache over its
        # target for the duration. Retirement cannot disturb the label anyway.
        await _label_conversation_after_turn(runtime, execn, turns_appended)


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


def resolve_logical_turn_key(
    execn: TurnExecution,
    runtime: Optional["ChannelRuntime"],
    registry: TurnRegistry,
) -> Optional[str]:
    """Best-effort capture of the execution's logical turn key [R9].

    Order of authority:
      1. already recorded on the execution (resume capture, or a finished run);
      2. the result's ``TurnOutput.turn_key`` (a completed/suspended turn);
      3. for an execution that is still THE active one for its channel, the
         runtime's ``current_turn_key`` — but only when it differs from the
         snapshot taken before the work began, i.e. only once ``_begin_turn``
         has minted this turn's key. Reading the attribute cross-thread is a
         GIL-atomic str read; the pre-work snapshot was taken under
         ``runtime.lock``, so a differing value cannot be a stale key.

    Never guesses: returns None while the logical key is genuinely unknowable
    (queued work that has not begun, or an errored run with no result).
    """
    if execn.logical_turn_key:
        return execn.logical_turn_key
    result = execn.result
    if result is not None:
        execn.logical_turn_key = result.turn_key
        return execn.logical_turn_key
    if execn.is_terminal or runtime is None:
        return None
    if registry.active_turn_key(execn.channel_id) != execn.turn_key:
        return None
    if execn.exec_state is not ExecState.RUNNING:
        # Queued: the pre-work snapshot has not been taken yet (it happens in
        # the same no-await block that sets RUNNING), so pre_turn_key is still
        # its unset default while current_turn_key may still hold the PREVIOUS
        # turn's key — the WEC never clears it between turns. Comparing now
        # would stamp that stale key onto this execution. Wait for the run.
        return None
    current = runtime.execution_context.current_turn_key
    if current and current != execn.pre_turn_key:
        execn.logical_turn_key = current
    return execn.logical_turn_key


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
    # [R9] a deferred 202 should hand the caller the logical key alongside the
    # execution key whenever it is already knowable (best-effort; see helper).
    resolve_logical_turn_key(execn, runtime, registry)
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

    Everything but ``turn_key``, ``exec_state``, ``logical_turn_key``, and
    ``traces`` is the public ``TurnOutput`` projection. ``exec_state`` is
    transport (where the *work* is), ``status``/``failure_reason``/``success``
    are the turn outcome; a failed turn is still a 200. ``turn_key`` here is
    the EXECUTION's key — the handle a deferred 202 is polled with — not
    ``TurnOutput.turn_key``, which is the workflow's own logical-turn key.
    [R9]: ``logical_turn_key`` carries that logical key alongside the
    execution key once it is known (always, once a result exists; on a 202
    only after the work has begun), so a deferred caller can later fetch the
    completed turn — and its trace — from the observability store, which is
    keyed by the logical key only.

    As of v3.0 the legacy top-level ``command_responses`` list is no longer
    emitted; clients should read ``answer`` and
    ``command_outputs[*].command_response``.
    """
    if not execn.is_terminal:
        body = {
            "turn_key": execn.turn_key,
            "exec_state": ExecState.RUNNING.value,
        }
        if execn.logical_turn_key:
            body["logical_turn_key"] = execn.logical_turn_key
        return 202, body

    if execn.error is not None:
        body = {
            "turn_key": execn.turn_key,
            "exec_state": execn.exec_state.value,
            "error": execn.error,
        }
        if execn.logical_turn_key:
            body["logical_turn_key"] = execn.logical_turn_key
        return 200, body

    result = execn.result
    body: dict[str, Any] = {
        "turn_key": execn.turn_key,
        "exec_state": execn.exec_state.value,
        "logical_turn_key": (
            result.turn_key if result else execn.logical_turn_key
        ),
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
