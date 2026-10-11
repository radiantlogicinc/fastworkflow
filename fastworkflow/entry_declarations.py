"""The entry-command declaration contract.

One canonical source for "which command enters this context": the context's own
callback class, read through :func:`declared_entry_commands`. Nothing in the
routing definition or the context model records the fact, and inferring it from
a command's NAME would bake one workflow's spelling conventions into the
framework.

The live callers read the declaration to name the entry command: the
foreign-context refusal guard in
``fastworkflow/_workflows/command_metadata_extraction/intent_detection.py``,
and the unavailable-command path in ``fastworkflow/context_navigation.py``.

Two more facts follow the same pattern. ``occupiable = False`` on a context's
callback class (read by :func:`declared_occupiable`) marks a mixin context a user
cannot enter. ``descends_to = "<Context>"`` and ``descend_parameter = "<param>"``
on a command's ``ResponseGenerator`` class (read by ``context_navigation``) say
a command enters a context, and only when that parameter is passed if one is
named. Absent attributes mean the
workflow declares nothing: the context is listed and the command has no
navigation effect.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


#: Class attributes a workflow's context callback class may declare to say which
#: command enters that context. THE canonical source for the fact: nothing in the
#: routing definition records which command sets the current context, and
#: inferring it from a command's NAME would bake one workflow's spelling
#: conventions into the framework. Recording the same fact a second time in the
#: context model file was considered and rejected: two sources for one fact is
#: one source plus a drift.
#:
#: Syntax: ``enter_command = "<command_name>"`` or
#: ``enter_command = "<command_name> <param>value-shaped hint</param>"``. Only the
#: leading command name is load-bearing; the rest is hint text shown to the agent.
#: ``enter_commands`` takes a list when a context has more than one entry command,
#: in which case dispatch declines and the hint names them all.
CONTEXT_ENTER_COMMAND_ATTRS = ("enter_command", "enter_commands")
# ---------------------------------------------------------------------------
# The declaration
# ---------------------------------------------------------------------------

def _declared_context_class(workflow_folderpath: str, context_name: str):
    """The context's callback class, or None when it has none."""
    import fastworkflow

    app_crd = fastworkflow.RoutingRegistry.get_definition(workflow_folderpath)
    return app_crd.context_model.get_context_class(
        context_name, fastworkflow.ModuleType.CONTEXT_CLASS
    )


def declared_entry_commands(workflow_folderpath: str, context_name: str) -> list[str]:
    """The ``enter_command`` declarations *context_name* carries, verbatim.

    Read off the context's own callback class, the one canonical source. Empty
    when the workflow declares nothing, when the context has no callback class,
    or when loading it fails -- a hint that names the context alone is worth more
    than a failed turn, so nothing here is allowed to raise.
    """
    try:
        context_class = _declared_context_class(workflow_folderpath, context_name)
        for attribute in CONTEXT_ENTER_COMMAND_ATTRS:
            value = getattr(context_class, attribute, None)
            if isinstance(value, str) and value.strip():
                return [value.strip()]
            if isinstance(value, (list, tuple)) and value:
                return [str(v).strip() for v in value if str(v).strip()]
    except Exception as exc:  # noqa: BLE001 - a hint must not fail a turn
        logger.debug(
            "no enter_command declaration readable for context %r: %r",
            context_name, exc,
        )
    return []


def declared_occupiable(workflow_folderpath: str, context_name: str) -> bool | None:
    """The ``occupiable`` declaration of *context_name*, or None when undeclared.

    ``occupiable = False`` on the context's callback class marks a mixin context a
    user cannot enter. None means the class does not say, and the context is
    listed. Never raises.
    """
    try:
        value = getattr(_declared_context_class(workflow_folderpath, context_name),
                        "occupiable", None)
    except Exception as exc:  # noqa: BLE001 - an undeclared fact must not fail a turn
        logger.debug(
            "no occupiable declaration readable for context %r: %r",
            context_name, exc,
        )
        return None
    return value if isinstance(value, bool) else None
