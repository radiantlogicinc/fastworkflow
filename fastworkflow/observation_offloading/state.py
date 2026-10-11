"""Process-local offload state and the offload event log."""
from __future__ import annotations

import logging
import os
import tempfile
import threading
from typing import Any, Mapping, Optional

from fastworkflow.observation_offloading.archive import (
    RuntimeHandleArchive,
    RuntimeHandleScope,
)
from fastworkflow.observability import store as observability_store

#: How many diagnostic events the process keeps in memory: a RING, not a ledger.
#: The durable copy is the ``offload_events`` table of the turn's observability
#: database.
EVENT_BUFFER_MAX = 2000

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_events: list[dict[str, Any]] = []
#: Database paths an event write has already failed against, so each failure
#: is logged once rather than once per event.
_event_write_failures: set[str] = set()
_default_archive: Optional[RuntimeHandleArchive] = None
#: One archive object per database FILE: two spellings of one path must be one
#: object, and a caller that holds only a store path must be able to reach the
#: durable rows in the same file without re-creating the schema on every call.
_archives_by_path: dict[str, RuntimeHandleArchive] = {}
#: Database paths ``prune_once`` has already pruned in this process.
_pruned_paths: set[str] = set()
_default_scope = RuntimeHandleScope(
    channel_id=f"process-{os.getpid()}",
    turn_key=f"process-{os.getpid()}",
)


def archive_for_path(db_path: str) -> RuntimeHandleArchive:
    """The archive object for one observability database, created once per path.

    A caller holding only the database path reaches the evidence tables here
    instead of re-opening the store per call.
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


def observability_db_path(host: Any) -> str:
    """The observability database of the workflow session *host* is running.

    The session's bound app workflow first, then its active workflow: the same
    database ``build_tool_agent`` opens for the turn's observations.
    """
    from fastworkflow import state_paths

    app_workflow = getattr(host, "app_workflow", None)
    getter = getattr(host, "get_active_workflow", None)
    active_workflow = getter() if callable(getter) else None
    workflow_path = (str(getattr(app_workflow, "folderpath", "") or "")
                     or str(getattr(active_workflow, "folderpath", "") or ""))
    return state_paths.observability_db(workflow_path)


def durable_archive(selected_archive: Any = None) -> Any:
    """Where this call's DURABLE records belong.

    The caller's archive when it named one; otherwise the observability database
    of the session the current command runs under, so a turn's records land
    beside the observations they describe; otherwise the process default.

    Never raises: durability of presentation metadata must not be able to fail a
    command, so an unresolvable archive is reported as ``None``.
    """
    if selected_archive is not None:
        return selected_archive
    try:
        from fastworkflow import tracing

        host = tracing.current_host()
        if host is not None:
            # A session's records never fall back to the process default: a
            # database that cannot be opened keeps them in memory only.
            return archive_for_path(observability_db_path(host))
    except Exception:  # noqa: BLE001 - an unopenable session database
        logger.debug("no session archive for durable records", exc_info=True)
        return None
    try:
        return archive()
    except Exception:  # noqa: BLE001
        logger.debug("no default archive for durable records", exc_info=True)
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
    itself.
    """
    return os.path.join(
        tempfile.gettempdir(), f"fw-offload-{os.getpid()}", "observability.sqlite3"
    )


def scope_for_host(host: Any) -> RuntimeHandleScope:
    """The turn scope of the session this call is running under.

    ``build_tool_agent`` resolves the same scope for the ReAct loop; a scope
    computed two ways would be two scopes the moment either changed.
    """
    from fastworkflow import tracing

    channel_id = str(tracing.get_channel_id(host) or "unbound")
    turn_key = str(tracing.get_turn_key(host) or channel_id)
    return RuntimeHandleScope(channel_id=channel_id, turn_key=turn_key)


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

    The turn scope of the session the command runs under, derived from the trace
    host; the process default outside a session.
    """
    from fastworkflow import tracing

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
    step is the last one with no observation. The alias is ``O{idx}``, the same
    index the step's observation is printed and archived under.

    ``None`` when there is no agent step in flight: a direct user command, a
    non-execute tool, or offloading turned off. There is no agent-visible
    namespace in that case, so there is no alias to be wrong about.
    """
    agent = agent if agent is not None else _current_agent()
    trajectory = getattr(agent, "trajectory", None)
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
    return f"O{latest}"


def record_event(
    event: Mapping[str, Any], *, scope: Optional[RuntimeHandleScope] = None,
    store: Any = None,
) -> None:
    """Append to the in-memory log and, when *scope* is given, store a copy.

    The durable copy is a row of ``offload_events`` in *store* (default: the
    session's database), written through the same redaction as evidence. It is
    best effort: a failure drops the durable copy and is logged once per
    database, and never fails the step that produced the event.
    """
    item = dict(event)
    with _lock:
        _events.append(item)
        overflow = len(_events) - EVENT_BUFFER_MAX
        if overflow > 0:
            del _events[:overflow]
    if scope is None:
        return
    target = store if store is not None else durable_archive()
    if target is None:
        return
    try:
        target.persist_event(scope, item)
    except Exception as error:  # noqa: BLE001 - an event must never fail a turn
        path = str(getattr(target, "db_path", ""))
        if path not in _event_write_failures:
            _event_write_failures.add(path)
            logger.warning(
                "observation offloading could not store an event in %s: %s: %s",
                path, type(error).__name__, error,
            )


def snapshot_events() -> list[dict[str, Any]]:
    with _lock:
        return list(_events)


def reset_observation_state() -> None:
    """Clear this module's process-local state (tests and standalone embedders).

    Stored rows are untouched.
    """
    global _default_archive
    with _lock:
        _events.clear()
        _event_write_failures.clear()
        _default_archive = None
        _archives_by_path.clear()
        _pruned_paths.clear()
