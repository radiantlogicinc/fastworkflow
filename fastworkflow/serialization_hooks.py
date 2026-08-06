"""Author-supplied serialization hooks for command-context objects.

A session is pinned — never evicted, so never bounded — while its workflow holds
a live command-context object, because those are ordinary Python instances that
no generic projection can reach. This module is the mechanism that unpins them.

The hooks live on the `Context` class a workflow author already writes at
`_commands/<Name>/_<Name>.py`, next to `get_parent` and `get_displayname`, and
are discovered the same way. They are optional, and *presence is the consent
signal*:

    no get_state at all        -> pin, and warn once per context class
    get_state -> EPHEMERAL     -> pin, and stay quiet: the author declared it
    get_state -> dict          -> evictable
    get_state -> None          -> pin AND warn: that is a bug, not a decision

See docs/fastworkflow_serialization_hooks_design.md for the rationale, including
why these are not `__getstate__`/`__setstate__` on the application class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import fastworkflow
from fastworkflow.utils import python_utils
from fastworkflow.utils.logging import logger

# The locator of a slot that *is* the anchor. Reserved, so no author locator can
# collide with it: an author locator is only ever consulted for a slot that
# differs from the anchor.
ANCHOR_LOCATOR = "."

SERIALIZATION_MODULE = "_serialization.py"

# Warn once per (workflow, context class). For a workflow that has not
# implemented the hook, pinned is the steady state, so an unthrottled warning
# would fire on every session and bury everything around it.
_warned: set[tuple[str, str]] = set()


class ProjectionOutcome(Enum):
    """Why a workflow's command contexts are, or are not, checkpointable."""

    SERIALIZABLE = "serializable"
    NO_CONTEXT = "no_context"
    DECLARED_EPHEMERAL = "declared_ephemeral"
    HOOK_ABSENT = "hook_absent"
    HOOK_FAILED = "hook_failed"
    LOCATOR_MISSING = "locator_missing"
    LOCATOR_UNSTABLE = "locator_unstable"


_QUIET_OUTCOMES = (
    ProjectionOutcome.SERIALIZABLE,
    ProjectionOutcome.NO_CONTEXT,
    ProjectionOutcome.DECLARED_EPHEMERAL,
)


@dataclass
class ContextProjection:
    """The durable projection of a workflow's command-context slots.

    ``anchor`` is the one object actually serialized; the other slots are stored
    as locators into it. Two independent snapshots of the same node would restore
    as two objects, and ``workflow.current_command_context`` would no longer be a
    node inside ``workflow.root_command_context``.
    """

    outcome: ProjectionOutcome
    reason: Optional[str] = None
    context_name: Optional[str] = None
    state: Optional[dict[str, Any]] = None
    state_version: Optional[int] = None
    anchor_is_root: bool = False
    current_locator: Optional[str] = None
    response_locator: Optional[str] = None

    @property
    def is_checkpointable(self) -> bool:
        return self.outcome in (
            ProjectionOutcome.SERIALIZABLE,
            ProjectionOutcome.NO_CONTEXT,
        )

    def as_record(self) -> dict[str, Any]:
        """The JSON-native form stored in a checkpoint's context section."""
        if self.outcome is ProjectionOutcome.NO_CONTEXT:
            return {}
        return {
            "context_name": self.context_name,
            "state": self.state,
            "state_version": self.state_version,
            "anchor_is_root": self.anchor_is_root,
            "current_locator": self.current_locator,
            "response_locator": self.response_locator,
        }


def get_context_hooks(workflow: "fastworkflow.Workflow", context_object: Any):
    """The author's ``Context`` class for a live object, or None.

    Keyed on the object's class name, which is the folder name under
    ``_commands/``. A workflow with no callback file for that context simply has
    no hooks, which is the ``HOOK_ABSENT`` branch rather than an error.
    """
    if context_object is None:
        return None
    context_model = fastworkflow.CommandContextModel.load(workflow.folderpath)
    return context_model.get_context_class(
        fastworkflow.Workflow.get_command_context_name(context_object),
        fastworkflow.ModuleType.CONTEXT_CLASS,
    )


def get_serialization_module(workflow: "fastworkflow.Workflow"):
    """The workflow's optional ``Serialization`` class from ``_commands/_serialization.py``.

    The file is underscore-prefixed, so the command scanner already ignores it
    and there is no collision with command discovery.
    """
    import os

    module_path = os.path.join(workflow.folderpath, "_commands", SERIALIZATION_MODULE)
    if not os.path.isfile(module_path):
        return None
    module = python_utils.get_module(module_path, workflow.folderpath)
    return getattr(module, "Serialization", None) if module else None


def project_command_contexts(workflow: "fastworkflow.Workflow") -> ContextProjection:
    """Project the workflow's command-context slots, or explain why it must pin.

    Serializes a single anchor — the root context if there is one, otherwise the
    current context, because a workflow may set only the current one — and stores
    the other slots as locators relative to it.
    """
    root = workflow.root_command_context
    current = workflow.current_command_context
    response = workflow.command_context_for_response_generation

    anchor = root if root is not None else current
    if anchor is None:
        return ContextProjection(outcome=ProjectionOutcome.NO_CONTEXT)

    context_name = fastworkflow.Workflow.get_command_context_name(anchor)
    hooks = get_context_hooks(workflow, anchor)

    if hooks is None or not hasattr(hooks, "get_state"):
        return _pinned(
            ProjectionOutcome.HOOK_ABSENT,
            context_name,
            f"context class {context_name!r} has no get_state hook",
        )

    try:
        state = hooks.get_state(anchor)
    except Exception as exc:
        return _pinned(
            ProjectionOutcome.HOOK_FAILED,
            context_name,
            f"{context_name}.get_state raised {type(exc).__name__}: {exc}",
        )

    if state is fastworkflow.EPHEMERAL:
        return ContextProjection(
            outcome=ProjectionOutcome.DECLARED_EPHEMERAL,
            context_name=context_name,
            reason=f"{context_name} declared its state ephemeral",
        )

    if not isinstance(state, dict):
        return _pinned(
            ProjectionOutcome.HOOK_FAILED,
            context_name,
            f"{context_name}.get_state returned {type(state).__name__}; return a "
            "dict, or fastworkflow.EPHEMERAL to decline deliberately",
        )

    locators: dict[str, Optional[str]] = {}
    for slot_name, slot in (("current", current), ("response", response)):
        locator, failure = _locate(hooks, anchor, slot, context_name, slot_name)
        if failure is not None:
            return failure
        locators[slot_name] = locator

    return ContextProjection(
        outcome=ProjectionOutcome.SERIALIZABLE,
        context_name=context_name,
        state=state,
        state_version=int(getattr(hooks, "state_version", 1)),
        anchor_is_root=root is not None,
        current_locator=locators["current"],
        response_locator=locators["response"],
    )


def restore_command_contexts(
    workflow: "fastworkflow.Workflow", record: dict[str, Any]
) -> bool:
    """Rebuild the command-context slots from ``ContextProjection.as_record()``.

    Returns False when there was nothing to restore. Raises whatever the author's
    ``from_state`` raises — including ``fastworkflow.UnsupportedStateVersion`` —
    so the caller can quarantine rather than apply a partly-understood record.
    """
    if not record or record.get("state") is None:
        return False

    context_name = record["context_name"]
    hooks = _hooks_by_name(workflow, context_name)
    if hooks is None or not hasattr(hooks, "from_state"):
        raise fastworkflow.UnsupportedStateVersion(
            int(record.get("state_version") or 1),
            f"context class {context_name!r} has no from_state hook to restore with",
        )

    anchor = hooks.from_state(
        record["state"],
        workflow,
        state_version=int(record.get("state_version") or 1),
    )
    restored_name = fastworkflow.Workflow.get_command_context_name(anchor)
    if restored_name != context_name:
        # Applying this would put the wrong class in the slot and every later
        # hook lookup would resolve to a different Context.
        raise fastworkflow.UnsupportedStateVersion(
            int(record.get("state_version") or 1),
            f"{context_name}.from_state returned a {restored_name}",
        )

    if record.get("anchor_is_root"):
        workflow.root_command_context = anchor
    else:
        workflow.current_command_context = anchor

    workflow.current_command_context = _resolve(
        hooks, anchor, record.get("current_locator")
    )
    workflow.command_context_for_response_generation = _resolve(
        hooks, anchor, record.get("response_locator")
    )
    return True


def warn_if_pinned(workflow: "fastworkflow.Workflow", projection: ContextProjection) -> None:
    """Tell the author once, at the moment the session actually becomes pinned.

    Warning at workflow load would fire for contexts a session never touches;
    warning per eviction attempt would fire forever.
    """
    if projection.outcome in _QUIET_OUTCOMES:
        return

    key = (workflow.folderpath, projection.context_name or "?")
    if key in _warned:
        return
    _warned.add(key)

    name = projection.context_name
    logger.warning(
        f"workflow {workflow.folderpath}: {projection.reason}, so sessions holding "
        f"it are pinned and will never be evicted. Add get_state/from_state to "
        f"_commands/{name}/_{name}.py, or return fastworkflow.EPHEMERAL to declare "
        "this deliberate and silence this warning."
    )


def reset_warnings() -> None:
    """Forget which workflows have been warned about (tests)."""
    _warned.clear()


def _pinned(
    outcome: ProjectionOutcome, context_name: str, reason: str
) -> ContextProjection:
    return ContextProjection(
        outcome=outcome, context_name=context_name, reason=reason
    )


def _hooks_by_name(workflow: "fastworkflow.Workflow", context_name: str):
    context_model = fastworkflow.CommandContextModel.load(workflow.folderpath)
    return context_model.get_context_class(
        context_name, fastworkflow.ModuleType.CONTEXT_CLASS
    )


def _locate(
    hooks: Any,
    anchor: Any,
    slot: Any,
    context_name: str,
    slot_name: str,
) -> tuple[Optional[str], Optional[ContextProjection]]:
    """Locator for a slot relative to the anchor, or the projection that pins."""
    if slot is None:
        return None, None
    if slot is anchor:
        return ANCHOR_LOCATOR, None

    if not hasattr(hooks, "get_locator") or not hasattr(hooks, "find_by_locator"):
        return None, _pinned(
            ProjectionOutcome.LOCATOR_MISSING,
            context_name,
            f"the {slot_name} context is not the anchor and {context_name} has no "
            "get_locator/find_by_locator pair to point into it",
        )

    try:
        locator = hooks.get_locator(slot)
        # A locator that does not round-trip would restore a *different* node
        # into the slot, which surfaces much later as quietly wrong behaviour.
        # One cheap check per eviction turns that into a pin.
        if hooks.find_by_locator(anchor, locator) is not slot:
            return None, _pinned(
                ProjectionOutcome.LOCATOR_UNSTABLE,
                context_name,
                f"{context_name}.find_by_locator did not round-trip the "
                f"{slot_name} context locator {locator!r}",
            )
    except Exception as exc:
        return None, _pinned(
            ProjectionOutcome.LOCATOR_UNSTABLE,
            context_name,
            f"{context_name} locator hooks raised {type(exc).__name__}: {exc}",
        )

    return locator, None


def _resolve(hooks: Any, anchor: Any, locator: Optional[str]) -> Any:
    if locator is None:
        return None
    if locator == ANCHOR_LOCATOR:
        return anchor
    return hooks.find_by_locator(anchor, locator)
