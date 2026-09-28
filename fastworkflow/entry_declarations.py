"""The entry-command declaration contract.

One canonical source for "which command enters this context": the context's own
callback class, read through :func:`declared_entry_commands`. Nothing in the
routing definition or the context model records the fact, and inferring it from
a command's NAME would bake one workflow's spelling conventions into the
framework.

The live caller is the foreign-context refusal guard in
``fastworkflow/_workflows/command_metadata_extraction/intent_detection.py``,
which reads the declaration to name the entry command in its refusal.
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

def declared_entry_commands(workflow_folderpath: str, context_name: str) -> list[str]:
    """The ``enter_command`` declarations *context_name* carries, verbatim.

    Read off the context's own callback class, the one canonical source. Empty
    when the workflow declares nothing, when the context has no callback class,
    or when loading it fails -- a hint that names the context alone is worth more
    than a failed turn, so nothing here is allowed to raise.
    """
    try:
        import fastworkflow

        app_crd = fastworkflow.RoutingRegistry.get_definition(workflow_folderpath)
        context_class = app_crd.context_model.get_context_class(
            context_name, fastworkflow.ModuleType.CONTEXT_CLASS
        )
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
