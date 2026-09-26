"""Process-local hot cache and the offload event log."""
from __future__ import annotations

import logging
import os
import sqlite3
import tempfile
import threading
from contextlib import closing
from typing import Any, Mapping, Optional

from fastworkflow.observation_offloading.archive import (
    RuntimeHandleArchive,
    RuntimeHandleScope,
    clear_live_raw,
    release_live_raw,
)
from fastworkflow.observability import store as observability_store

from fastworkflow import context_budget

#: The hot-cache cap at the reference context window. The effective cap is a
#: fraction of the model's window (``context_budget.OFFLOAD_HOT``).
HOT_HANDLE_MAX_BYTES = context_budget.REFERENCE_OFFLOAD_HOT_MAX_BYTES
HOT_HANDLE_MAX_BYTES_ENV = context_budget.OFFLOAD_HOT.override_env
#: How many diagnostic events the process keeps in memory. The in-process log is
#: a RING, not a ledger: the durable copy is the ``offload_events`` table of the
#: turn's observability database (``ObservabilityStore.offload_events``), and
#: what stays in memory is only what a live turn (or a test) reads back through
#: ``snapshot_events``. Unbounded it grew with lifetime traffic and held search
#: questions, reasoning and full answers long after the turns that produced them
#: had ended (ido-1ew). A scope's own events go the moment the scope is
#: reclaimed; this cap is the backstop for the events no scope owns and for a
#: single turn that talks more than the whole process should remember.
EVENT_BUFFER_MAX = 2000

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_handles: dict[str, dict[str, Any]] = {}
_archived: dict[str, dict[str, Any]] = {}
_search_answers: dict[str, int] = {}
#: ido-8ps.13. ``handle_key(scope, alias) -> context clause``, written by
#: ``CommandExecutor.invoke_command`` BEFORE the command runs and read when
#: the alias line is printed. Turn-scoped like everything else here.
_context_clauses: dict[str, str] = {}
_events: list[dict[str, Any]] = []
#: Database paths an event write has already failed against, so each failure
#: is logged once rather than once per event.
_event_write_failures: set[str] = set()
#: ``scope_id -> (scope, archive)``: how a recorded event, which names only its
#: scope id, finds the turn it belongs to and the database it is stored in.
#: The archive is ``None`` for a scope seen without one; its events then go to
#: ``durable_archive()``. Reads and writes are single dictionary operations,
#: which are atomic, so this is safe to touch while ``_lock`` is held.
_routes: dict[str, tuple[RuntimeHandleScope, Any]] = {}
_default_archive: Optional[RuntimeHandleArchive] = None
#: One archive object per database FILE (ido-pg2): two spellings of one path must be one
#: object, and a caller that holds only a store path must be able to reach the
#: durable subject rows in the same file without re-creating the schema on
#: every call.
_archives_by_path: dict[str, RuntimeHandleArchive] = {}
#: Database paths ``prune_once`` has already pruned in this process.
_pruned_paths: set[str] = set()
_default_scope = RuntimeHandleScope(
    store_identity=f"process-{os.getpid()}",
    channel_id=f"process-{os.getpid()}",
    experiment_id="unbound",
    task_id="unbound",
    attempt=0,
    turn_key=f"process-{os.getpid()}",
)


def hot_handle_max_bytes_from_env() -> int:
    """The hot-cache cap for this run. See ``fastworkflow.context_budget``."""
    return context_budget.offload_hot_max_bytes()


def archive_for_path(db_path: str) -> RuntimeHandleArchive:
    """The archive object for one observability database, created once per path.

    A caller holding only the database path reaches the evidence and subject
    tables here instead of re-opening the store per call.
    """
    key = os.path.abspath(os.path.expanduser(str(db_path)))
    with _lock:
        existing = _archives_by_path.get(key)
    if existing is not None:
        return existing
    created = RuntimeHandleArchive(key)
    with _lock:
        return _archives_by_path.setdefault(key, created)


def prune_once(db_path: str) -> bool:
    """Prune the observability database at *db_path*, at most once per process.

    Pruning is otherwise triggered only by a trace sink starting on the store,
    and the offload archive writes evidence whether or not a sink ever opened
    it: a context built with ``tracing.NoOpTraceSink()`` would grow its offload
    tables without bound. The agent builds a new archive object per agent, so
    the guard is per PATH, or every build would re-prune a large database on
    its first turn. Where a sink already pruned at startup this second pass
    finds nothing to delete. Returns whether this call ran the prune; a failure
    is logged and never raised, because an agent must still be built.
    """
    key = os.path.abspath(os.path.expanduser(str(db_path)))
    with _lock:
        if key in _pruned_paths:
            return False
        _pruned_paths.add(key)
    try:
        observability_store.ObservabilityStore(key).prune()
    except Exception as error:  # noqa: BLE001 - a prune must not stop an agent
        logger.warning(
            "could not prune observability database %s: %s: %s",
            key, type(error).__name__, error,
        )
    return True


def durable_archive(selected_archive: Any = None) -> Any:
    """Where this call's DURABLE subject records belong.

    The caller's archive when it named one; otherwise the archive the running
    agent writes its observations to, so a turn's subject metadata lands in the
    same database as the observations it describes; otherwise the process
    default, which is what every other write in this module falls back to.

    Never raises: durability of presentation metadata must not be able to fail a
    command, so an unresolvable archive is reported as ``None`` and the caller
    keeps working out of its process-local cache alone.
    """
    if selected_archive is not None:
        return selected_archive
    try:
        found = getattr(_current_agent(), "observation_archive", None)
        if found is not None:
            return found
    except Exception:  # noqa: BLE001 - no agent, no tracing host, no archive
        logger.debug("no agent archive for durable subject metadata", exc_info=True)
    try:
        return archive()
    except Exception:  # noqa: BLE001
        logger.debug("no default archive for durable subject metadata", exc_info=True)
        return None


def archive() -> RuntimeHandleArchive:
    """The process-default archive, for a caller with no archive of its own.

    ``build_tool_agent`` always passes the workflow's archive, beside its
    observability database; this per-process file is the fallback for a
    direct call from a command frame before an agent has been built.
    """
    global _default_archive
    if _default_archive is None:
        _default_archive = RuntimeHandleArchive(default_archive_path())
    return _default_archive


def default_archive_path() -> str:
    """Where the per-process fallback observability database lives.

    Named after the pid and in its own directory under the temp directory, so
    it is process-local by construction and the store's 0700 hardening applies
    to a directory this process created rather than to the temp directory
    itself. Spelled once, because ``clear_default_cold_records`` has to be able
    to reach the file whether or not this process has instantiated the archive
    object for it.
    """
    return os.path.join(
        tempfile.gettempdir(), f"fw-offload-{os.getpid()}", "observability.sqlite3"
    )


#: The table that exists so a turn can be read back after a restart
#: (``ido-dhw``). It is the only one whose rows a process-local reset has to
#: reach: everything else in the database is evidence a reset never owned.
COLD_RESTART_TABLES = ("offload_subjects",)


def clear_default_cold_records(*tables: str) -> None:
    """Empty the PROCESS-DEFAULT database's cold-restart tables.

    The durable half of a process-local reset. Subject clauses are read
    THROUGH to the database when the in-memory registry misses, so a reset
    that cleared only memory would be answered from disk by the very rows it
    meant to drop. Every caller that never named an archive of its own shares
    this one file AND one ``default_scope``, so those rows are exactly the
    state the reset owns.

    No archive a CALLER named is touched, and nothing but these tables is:
    a real observability database holds real turns, and this is not an
    erasure path.
    """
    wanted = tables or COLD_RESTART_TABLES
    path = default_archive_path()
    if not os.path.exists(path):
        return
    try:
        with closing(sqlite3.connect(path, timeout=30.0)) as conn:
            present = {
                str(row[0])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            for table in wanted:
                if table in present:
                    conn.execute(f'DELETE FROM "{table}"')
            conn.commit()
    except Exception:  # noqa: BLE001 - a reset must not fail on a temp file
        logger.debug("could not clear the default database's cold-restart records",
                     exc_info=True)


def scope_for_host(host: Any) -> RuntimeHandleScope:
    """The turn scope of the session this call is running under.

    ``build_tool_agent`` resolves the same scope for the ReAct loop; this is the
    same computation reached from a command's own frame, where the only handle
    on the session is the trace host ``CommandExecutor.invoke_command`` bound.
    Keeping one implementation matters: a scope computed two ways is two scopes
    the moment either changes, and a handle stored under one of them would be
    unreachable under the other.
    """
    from fastworkflow import state_paths, tracing

    claim = tracing.get_experiment_claim(host)
    channel_id = str(tracing.get_channel_id(host) or "unbound")
    turn_key = str(tracing.get_turn_key(host) or channel_id)
    sink = tracing.get_sink(host)
    sink_store = getattr(sink, "store", None)
    identity_value = getattr(sink_store, "store_identity", None)
    if callable(identity_value):
        identity_value = identity_value()
    getter = getattr(host, "get_active_workflow", None)
    active_workflow = getter() if callable(getter) else None
    workflow_path = str(getattr(active_workflow, "folderpath", "") or "")
    store_identity = str(
        identity_value
        or getattr(sink, "store_identity", None)
        or state_paths.observability_db(workflow_path)
    )
    return RuntimeHandleScope(
        store_identity=store_identity,
        channel_id=channel_id,
        experiment_id=str(claim.get("experiment_id") or "unbound"),
        task_id=str(claim.get("task_id") or "unbound"),
        attempt=int(claim.get("attempt") or 0),
        turn_key=turn_key,
    )


def handle_key(scope: RuntimeHandleScope, alias: str) -> str:
    _routes.setdefault(scope.scope_id, (scope, None))
    return f"{scope.scope_id}:{alias}"


def register_scope(scope: RuntimeHandleScope, archive: Any = None) -> None:
    """Remember where *scope*'s events are stored.

    Called by the turn runtime, which knows both the scope and its archive,
    and by the archivers for a scope they write to. A later call without an
    archive never forgets one an earlier call named.
    """
    if archive is None:
        _routes.setdefault(scope.scope_id, (scope, None))
    else:
        _routes[scope.scope_id] = (scope, archive)


def default_scope() -> RuntimeHandleScope:
    return _default_scope


# ---------------------------------------------------------------------------
# Identity: scope and the canonical execute alias
# ---------------------------------------------------------------------------


def _current_agent() -> Any:
    """The ReAct agent running this command, when there is one."""
    from fastworkflow import tracing

    host = tracing.current_host()
    if host is None:
        return None
    agent = getattr(host, "workflow_tool_agent", None)
    if agent is None:
        core = getattr(host, "_core", None)
        agent = getattr(core, "workflow_tool_agent", None)
    return agent


def current_scope() -> RuntimeHandleScope:
    """The scope a handle declared right now belongs to.

    The live agent's turn runtime first, when it has one: ``TurnRuntime`` is the
    component that binds a turn to its scope, so its answer is the scope this
    turn's observations are archived under, and a handle filed anywhere else
    would be a handle the same turn could not read back. The agent's own
    ``continuation_scope`` is the same answer for an agent built without a
    runtime, and is consulted next. Then the trace host (a command running
    outside the ReAct loop), then the process default.
    """
    from fastworkflow import tracing

    agent = _current_agent()
    runtime = getattr(agent, "turn_runtime", None)
    runtime_scope = getattr(runtime, "scope", None)
    if isinstance(runtime_scope, RuntimeHandleScope):
        return runtime_scope
    scope = getattr(agent, "continuation_scope", None)
    if isinstance(scope, RuntimeHandleScope):
        return scope
    host = tracing.current_host()
    if host is not None:
        try:
            return scope_for_host(host)
        except Exception:  # noqa: BLE001
            logger.debug("could not resolve a host scope", exc_info=True)
    return default_scope()


def current_execute_alias(agent: Any = None) -> Optional[str]:
    """The ``O`` alias of the execute step this command is running inside.

    ReAct writes ``tool_name_{idx}`` before it calls the tool and
    ``observation_{idx}`` after it returns, so during a command the in-flight
    step is the last one with no observation. Which step is in flight is read
    from ``current_trajectory``; what that step is CALLED is read from the
    agent's ``execute_ordinal_by_step`` ledger, which numbered it just before
    dispatch and is the same ledger ``annotate_execute_observations`` is given
    when the step completes.

    Counting ``current_trajectory`` instead is only right when the mirror holds
    the whole turn. A process that imported a suspension has an empty mirror and
    a resumed trajectory with N executes already in it, so the first command
    after the resume would declare and stamp O1 while its observation printed
    O(N+1) -- a collision against the real O1, or a printed handle nothing had
    declared. The ledger is restored with the suspension, so both sides read one
    number.

    The count stands in only for an agent with no ledger (a duck-typed host, a
    plain ReAct), where the mirror is the whole turn by construction.

    ``None`` when there is no agent step in flight: a direct user command, a
    non-execute tool, or offloading turned off. There is no agent-visible
    namespace in that case, so there is no alias to be wrong about.
    """
    agent = agent if agent is not None else _current_agent()
    trajectory = getattr(agent, "current_trajectory", None)
    if not isinstance(trajectory, Mapping) or not trajectory:
        return None
    indexes = [
        int(key.removeprefix("tool_name_"))
        for key in trajectory
        if key.startswith("tool_name_") and key.removeprefix("tool_name_").isdigit()
    ]
    if not indexes:
        return None
    latest = max(indexes)
    if str(trajectory.get(f"tool_name_{latest}") or "") != "execute_workflow_query":
        return None
    if f"observation_{latest}" in trajectory:
        # The step already completed; this call is not inside it.
        return None
    ledger = getattr(agent, "execute_ordinal_by_step", None)
    if isinstance(ledger, Mapping) and latest in ledger:
        ordinal = int(ledger[latest])
    else:
        ordinal = sum(
            1
            for index in indexes
            if str(trajectory.get(f"tool_name_{index}") or "")
            == "execute_workflow_query"
        )
    return f"O{ordinal}" if ordinal else None


def record_event(event: Mapping[str, Any]) -> None:
    """Append to the in-memory log and store a copy in the observability DB.

    The durable copy is a row of ``offload_events`` in the database that holds
    the event's turn, written through the same redaction as evidence
    (``RuntimeHandleArchive.persist_event``). It is best effort: offloading is
    an optimisation, so a locked, full or unwritable database must not abort,
    or noticeably delay, the agent step that produced the event. The write
    waits briefly for the lock, a failure drops the durable copy, and the first
    failure per database is logged.

    An event names its scope by id. One whose scope this process has never
    seen -- or has already released -- cannot be placed in a turn, so it is
    kept in memory only; an event with no scope id belongs to the current
    scope.
    """
    item = dict(event)
    with _lock:
        _events.append(item)
        # The ring closes here, inside the same lock that appended, so two
        # threads recording at once cannot both skip the trim. A list is kept
        # rather than a deque because callers read this buffer as a list.
        overflow = len(_events) - EVENT_BUFFER_MAX
        if overflow > 0:
            del _events[:overflow]
    _persist_event(item)


def _persist_event(item: dict[str, Any]) -> None:
    try:
        scope_id = item.get("scope_id")
        if scope_id:
            route = _routes.get(str(scope_id))
            if route is None:
                return
        else:
            scope = current_scope()
            route = _routes.get(scope.scope_id) or (scope, None)
        scope, routed = route
        store = routed if routed is not None else durable_archive()
        if store is None:
            return
    except Exception:  # noqa: BLE001 - an event must never fail a turn
        logger.debug("could not place an offload event", exc_info=True)
        return
    try:
        store.persist_event(scope, item)
    except Exception as error:  # noqa: BLE001 - an event must never fail a turn
        path = str(getattr(store, "db_path", ""))
        if path not in _event_write_failures:
            _event_write_failures.add(path)
            logger.warning(
                "observation offloading could not store an event in %s: %s: %s",
                path, type(error).__name__, error,
            )


def snapshot_events() -> list[dict[str, Any]]:
    with _lock:
        return list(_events)


def mark_archived(
    scope: RuntimeHandleScope, alias: str, *, text_sha256: str, inline: bool = True
) -> None:
    """Remember that this alias is durable, and whether it is still inline.

    The digest lets the eager archiver skip a re-write of text it already wrote
    in this process (compaction revisits every execute step at every step), and
    the ``inline`` flag lets a search event separate a miss on a handle that was
    never printed from a miss on an observation the agent could still read.
    """
    with _lock:
        _archived[handle_key(scope, alias)] = {
            "text_sha256": text_sha256,
            "inline": inline,
        }


def mark_offloaded(scope: RuntimeHandleScope, alias: str) -> None:
    """The trajectory now carries a label for this alias instead of its text."""
    key = handle_key(scope, alias)
    with _lock:
        entry = _archived.get(key)
        if entry is None:
            _archived[key] = {"text_sha256": None, "inline": False}
        else:
            entry["inline"] = False


def archived_digest(scope: RuntimeHandleScope, alias: str) -> Optional[str]:
    with _lock:
        entry = _archived.get(handle_key(scope, alias))
    return None if entry is None else entry["text_sha256"]


def observation_inline(scope: RuntimeHandleScope, alias: str) -> Optional[bool]:
    """True while the observation is inline, False once labelled, None if unknown."""
    with _lock:
        entry = _archived.get(handle_key(scope, alias))
    return None if entry is None else bool(entry["inline"])


def next_search_answer_sequence(scope: RuntimeHandleScope) -> int:
    """The next ordinal for an archived search answer in this scope.

    Only used to build a record key (``labels.search_answer_key``) when an
    answer had to be bounded, so repeated searches of the same observation each
    keep their own complete text. It is not an observation ordinal and never
    enters the agent-visible ``O`` namespace.
    """
    with _lock:
        _search_answers[scope.scope_id] = _search_answers.get(scope.scope_id, 0) + 1
        return _search_answers[scope.scope_id]


def record_context_clause(
    scope: RuntimeHandleScope,
    alias: str,
    clause: str,
    *,
    selected_archive: Any = None,
) -> None:
    """Remember the context an execute step's command RAN IN.

    Written at dispatch, from the context handle taken BEFORE the command
    executes, because that is the fact the observation is evidence about: a
    command that MOVES the context produced its output in the context it ran
    in, not in the one it entered. The alias line is printed later, in the
    ``on_step_complete`` hook, by which time the current context has already
    moved -- so the fact has to be carried, not recomputed.

    An empty clause is stored as an empty clause: "this ran at the root" is a
    fact, and it must not read as "nothing was captured".

    It is also written THROUGH to the archive, because this
    map is turn-scoped process memory and the subject of an observation has to
    outlive the process that saw it. The durable write is best effort on the
    same terms as everything else on this path -- an archive that cannot be
    written keeps the observation and loses only its cold-restart subject, and
    says so in the event log rather than failing the command.
    """
    text = str(clause or "")
    with _lock:
        _context_clauses[handle_key(scope, alias)] = text
    _write_subject(scope, alias, text, selected_archive)


def _write_subject(
    scope: RuntimeHandleScope, alias: str, clause: str, selected_archive: Any
) -> None:
    store = durable_archive(selected_archive)
    if store is None:
        return
    try:
        store.put_subject(scope, alias, clause)
    except Exception as error:  # noqa: BLE001 - metadata must never fail a turn
        record_event(
            {
                "kind": "subject_persist_refused",
                "scope_id": scope.scope_id,
                "alias": alias,
                "error": type(error).__name__,
            }
        )


def context_clause_of(
    scope: RuntimeHandleScope, alias: str, *, selected_archive: Any = None
) -> Optional[str]:
    """The recorded clause for *alias*, ``""`` at the root, None if unrecorded.

    Process memory first, then the archive. The second tier is
    what makes a subject survive a restart: a rehydrated label, a cross-context
    page stamp, the attribution check and observation search all read the
    subject through here, and in a process that only imported a suspension the
    map is empty while the rows are still on disk. A durable hit refills the map
    -- the bounded runtime cache is REBUILT from the durable record rather than
    kept a second way -- so the read is paid for once per alias per process.

    ``None`` still means UNRECORDED. Nothing here ever invents a subject.
    """
    key = handle_key(scope, alias)
    with _lock:
        if key in _context_clauses:
            return _context_clauses[key]
    store = durable_archive(selected_archive)
    if store is None:
        return None
    try:
        clause = store.get_subject(scope, alias)
    except Exception:  # noqa: BLE001 - an unreadable archive is an unrecorded one
        logger.debug("could not read the stored subject of %s", alias, exc_info=True)
        return None
    if clause is None:
        return None
    with _lock:
        _context_clauses.setdefault(key, clause)
    return clause


def forget_context_clause(
    scope: RuntimeHandleScope, alias: str, *, selected_archive: Any = None
) -> None:
    """Drop the clause recorded for *alias*, so it reads as UNRECORDED again.

    The dispatch-time stamp is a good default and a bad answer for a step whose
    real subject was declared somewhere else. If
    the declaring subject turns out to be unknown, "no subject recorded" is the
    truth and the context the agent happened to be standing in is not -- and
    "unrecorded" is a state every reader already handles, where a wrong clause
    is one every reader believes.

    The durable row goes with it: a correction that only reached
    process memory would be undone by the next restart, which is the failure
    mode this whole pair exists to prevent.
    """
    with _lock:
        _context_clauses.pop(handle_key(scope, alias), None)
    store = durable_archive(selected_archive)
    if store is None:
        return
    try:
        store.forget_subject(scope, alias)
    except Exception:  # noqa: BLE001 - metadata must never fail a turn
        logger.debug("could not drop the stored subject of %s", alias, exc_info=True)


def release_scope(scope: "RuntimeHandleScope | str") -> None:
    """Drop this component's process-local cache for one finished scope.

    Residency, never evidence: the archive keeps every row, so a scope
    reclaimed here is still readable from disk -- which is exactly what the
    cold-resume path already does in a process that never saw the turn at all.
    That includes the subject clauses dropped below: they are dropped from
    memory and no row is deleted, so a later read of a reclaimed scope
    rebuilds from the archive rather than answering "unrecorded". What a
    reclaimed scope reads from disk is the STORED text: the raw copies of its
    redacted observations are released here too, so after this point the
    process reads what any other process would.

    This is deliberately NOT a global reset. Everything the offloading runtime
    remembers is keyed by ``scope_id``, so one turn's state can be released
    while every other live turn in the process keeps its own. A reset that took
    the lot would invalidate the turns running beside this one.

    The caller decides what "finished" means, and only two places may: a turn
    that is over because the agent has bound the NEXT one
    (``StructuredContinuationReAct.bind_scope``), and a session that is over
    because its execution context was closed or evicted
    (``WorkflowExecutionContext.close``). Neither fires for a SUSPENDED turn,
    because a suspension is state that must outlive the process, not state to
    reclaim.

    An erasure reaches here too, through ``agent_runtime.reclaim_erased_scopes``,
    after the rows are already gone.
    """
    scope_id = scope if isinstance(scope, str) else scope.scope_id
    prefix = f"{scope_id}:"
    with _lock:
        for registry in (_handles, _archived, _context_clauses):
            for key in [key for key in registry if key.startswith(prefix)]:
                del registry[key]
        _search_answers.pop(scope_id, None)
        _routes.pop(scope_id, None)
        release_live_raw(scope_id)
        kept = [
            item for item in _events
            if str(item.get("scope_id") or "") != scope_id
        ]
        if len(kept) != len(_events):
            _events[:] = kept


def reset_observation_state() -> None:
    """Drop every process-local cache the offloading runtime holds.

    Result-handle and auto-navigation caches are reset by the runtime owner,
    which calls each component's reset hook. Stored SQLite rows are untouched:
    this resets residency, never evidence.

    The PROCESS-DEFAULT database's durable subject rows go too. That file is
    ``fw-offload-<pid>/observability.sqlite3`` in the temp directory --
    process-local by construction, named after this process, and shared by
    every caller that never passed an archive of its own, all of whom also
    share one ``default_scope``. Leaving that table behind would let a reset
    process read back the subjects of the state it just dropped. Nothing else
    in the file is touched, and no archive a CALLER named is touched at all:
    those are real observability databases holding real turns.
    """
    global _default_archive
    with _lock:
        _handles.clear()
        _archived.clear()
        _search_answers.clear()
        _context_clauses.clear()
        _events.clear()
        _event_write_failures.clear()
        _routes.clear()
        _default_archive = None
        _archives_by_path.clear()
        _pruned_paths.clear()
    clear_live_raw()
    clear_default_cold_records()


def reclaim_scope(scope: "RuntimeHandleScope | str") -> None:
    """Compatibility import; aggregate ownership lives in ``agent_runtime``."""
    from fastworkflow.agent_runtime import reclaim_scope as runtime_reclaim_scope

    runtime_reclaim_scope(scope)


def reset_runtime_state() -> None:
    """Compatibility import; aggregate ownership lives in ``agent_runtime``."""
    from fastworkflow.agent_runtime import reset_runtime_state as runtime_reset

    runtime_reset()


def stored_handles(scope: Optional[RuntimeHandleScope] = None) -> dict[str, dict[str, Any]]:
    selected = scope or _default_scope
    prefix = f"{selected.scope_id}:"
    with _lock:
        return {
            str(payload["alias"]): dict(payload)
            for key, payload in _handles.items()
            if key.startswith(prefix)
        }


def remember_handle(scope: RuntimeHandleScope, payload: dict[str, Any]) -> None:
    with _lock:
        _handles[handle_key(scope, str(payload["alias"]))] = payload


def hot_payload_bytes(scope: Optional[RuntimeHandleScope] = None) -> int:
    selected = scope or _default_scope
    prefix = f"{selected.scope_id}:"
    with _lock:
        return sum(
            len(str(value["text"]).encode("utf-8"))
            for key, value in _handles.items()
            if key.startswith(prefix)
        )


def clear_hot_handles(scope: Optional[RuntimeHandleScope] = None) -> None:
    selected = scope or _default_scope
    prefix = f"{selected.scope_id}:"
    with _lock:
        for key in [key for key in _handles if key.startswith(prefix)]:
            del _handles[key]


def evict_hot_handles(scope: RuntimeHandleScope, *, hot_handle_max_bytes: int) -> list[str]:
    evicted: list[str] = []
    prefix = f"{scope.scope_id}:"
    while hot_payload_bytes(scope) > hot_handle_max_bytes:
        with _lock:
            oldest_key = next((key for key in _handles if key.startswith(prefix)), None)
            if oldest_key is None:
                break
            payload = _handles.pop(oldest_key)
        alias = str(payload["alias"])
        evicted.append(alias)
        record_event(
            {
                "kind": "hot_evict",
                "scope_id": scope.scope_id,
                "alias": alias,
                "text_sha256": payload["text_sha256"],
                "hot_payload_bytes": hot_payload_bytes(scope),
            }
        )
    return evicted
