"""Navigation facts a workflow declares on its classes, and the retired entry-command declaration.

How a command enters a context is declared once, on the command: ``@descends_to("<Context>")``, or
``@descends_to("<Context>", when_parameter="<param>")``, on its ``ResponseGenerator`` class, read by
``context_navigation.collect_descend_edges`` (design 3.15). The context class used to declare it
too, as ``enter_command`` / ``enter_commands``. Those shipped, so a context class that still
declares either is a load error (:func:`check_retired_entry_declarations`), not a second source
for the same fact.

``occupiable = False`` on a context's callback class (read by :func:`declared_occupiable`) marks a
mixin context a user cannot enter. Absent attributes mean the workflow declares nothing: the
context is listed and the command has no navigation effect.
"""
from __future__ import annotations

import inspect
import logging
import os

logger = logging.getLogger(__name__)


#: The retired class attributes a context callback class may still carry. Declaring one is a load
#: error (design 3.15); nothing reads them.
CONTEXT_ENTER_COMMAND_ATTRS = ("enter_command", "enter_commands")


# ---------------------------------------------------------------------------
# The retired declaration
# ---------------------------------------------------------------------------

def check_retired_entry_declarations(workflow_folderpath: str, context_model) -> None:
    """Raise ValueError when a context class declares ``enter_command`` or ``enter_commands``.

    One line per declaration, naming the file, the context, and the command the declaration's
    leading token names, with the decorator to write instead: ``@descends_to`` on that command's
    ``ResponseGenerator``. Reads the context classes through *context_model* (the routing
    definition is still being built when this runs, so the routing registry is not consulted).
    """
    problems: list[str] = []
    for context_name in list(getattr(context_model, "_command_contexts", {})):
        context_class = context_model.get_context_class(context_name, _context_class_type())
        if context_class is None:
            continue
        for attribute in CONTEXT_ENTER_COMMAND_ATTRS:
            value = getattr(context_class, attribute, None)
            if value is None:
                continue
            declarations = list(value) if isinstance(value, (list, tuple)) else [value]
            for declaration in declarations:
                words = str(declaration).split()
                command = words[0] if words else "<command>"
                problems.append(
                    f"{_source_label(context_class, workflow_folderpath)}: "
                    f"Context.{attribute} is no longer read; declare @descends_to("
                    f"\"{context_name}\") on {command}'s ResponseGenerator"
                )
    if problems:
        raise ValueError("\n".join(problems))


def _context_class_type():
    import fastworkflow

    return fastworkflow.ModuleType.CONTEXT_CLASS


def _source_label(context_class, workflow_folderpath: str) -> str:
    """The context class's file, relative to the workflow folder, or its qualified name."""
    try:
        path = inspect.getsourcefile(context_class) or inspect.getfile(context_class)
    except (TypeError, OSError):
        return context_class.__qualname__
    if not path:
        return context_class.__qualname__
    return os.path.relpath(os.path.abspath(path), os.path.abspath(workflow_folderpath))


# ---------------------------------------------------------------------------
# The occupiable fact
# ---------------------------------------------------------------------------

def _declared_context_class(workflow_folderpath: str, context_name: str):
    """The context's callback class, or None when it has none."""
    import fastworkflow

    app_crd = fastworkflow.RoutingRegistry.get_definition(workflow_folderpath)
    return app_crd.context_model.get_context_class(
        context_name, fastworkflow.ModuleType.CONTEXT_CLASS
    )


def declared_occupiable(workflow_folderpath: str, context_name: str) -> bool | None:
    """The ``occupiable`` declaration of *context_name*, or None when undeclared.

    ``occupiable = False`` on the context's callback class marks a mixin context a user cannot
    enter. None means the class does not say, and the context is listed. Never raises.
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
