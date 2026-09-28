"""Composition and lifecycle owner for one workflow tool agent."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterable

if TYPE_CHECKING:
    from fastworkflow.observation_offloading.archive import RuntimeHandleScope


def build_turn_runtime(
    scope: RuntimeHandleScope, *, archive: Any = None
) -> "TurnRuntime":
    """Create the one valid runtime owner for a standalone agent."""
    from fastworkflow.observation_offloading import state

    selected_archive = archive if archive is not None else state.archive()
    return TurnRuntime(
        scope=scope, archive=selected_archive,
        observation_component=state, record_event=state.record_event,
    )


@dataclass
class TurnRuntime:
    """Bind one agent's scoped store and coordinate finished-scope cleanup.

    Suspension remains owned by the agent and WEC guards.  This object is called
    only after those guards decide a scope is finished, so it does not maintain
    a second lifecycle state.
    """

    scope: RuntimeHandleScope
    archive: Any
    observation_component: Any
    record_event: Callable[[dict[str, Any]], None]

    def __post_init__(self) -> None:
        self._register(self.scope)

    def _register(self, scope: RuntimeHandleScope) -> None:
        register = getattr(self.observation_component, "register_scope", None)
        if callable(register):
            register(scope, self.archive)

    def bind_scope(self, scope: RuntimeHandleScope) -> None:
        self.scope = scope
        self._register(scope)

    def emit(self, event: dict[str, Any]) -> None:
        self.record_event(event)

    def finish_scope(self, scope: RuntimeHandleScope) -> None:
        """Release a scope already proven finished by its caller.

        Evidence was redacted when it was written, so nothing on disk changes
        here: what goes is the process memory of the turn, including the raw
        copies of redacted observations its in-flight reads were served from.
        """
        self.release_scope(scope)

    def release_scope(self, scope: RuntimeHandleScope | str) -> None:
        """Ask the observation component to release the resources it owns."""
        scope_id = str(getattr(scope, "scope_id", scope))
        self.observation_component.release_scope(scope_id)


def reclaim_scope(scope: RuntimeHandleScope | str) -> None:
    """Released aggregate cleanup for callers without an agent runtime."""
    from fastworkflow.observation_offloading import state

    state.release_scope(str(getattr(scope, "scope_id", scope)))


def reclaim_erased_scopes(scope_ids: Iterable[str]) -> int:
    """Drop what this process still holds for scopes whose evidence was erased.

    Called by ``ObservabilityStore`` after an erasure or retention transaction
    has committed: the rows -- evidence, subjects and events alike -- are
    already gone, and this removes the hot handles, the raw in-flight copies,
    the context clauses, the event routes and the event-ring entries that
    could still serve them. Returns how many scopes were released. Never
    raises: a cache drop must not undo an erasure that already happened.

    The state module is imported here rather than at module scope because it
    imports the archive, which imports the observability store, which imports
    this module.
    """
    from fastworkflow.observation_offloading import state

    released = 0
    for scope_id in sorted({str(value) for value in scope_ids}):
        try:
            state.release_scope(scope_id)
        except Exception:  # noqa: BLE001 - a cache drop must not fail erasure
            state.logger.warning(
                "could not drop process caches for erased scope %s", scope_id,
                exc_info=True,
            )
            continue
        released += 1
    return released


def reset_runtime_state() -> None:
    """Released aggregate reset for tests and standalone integrations."""
    from fastworkflow.observation_offloading import state

    state.reset_observation_state()
