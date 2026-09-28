"""Which context instance a command ran in, in the workflow's own terms.

A listing produced by navigating into a context carries no identifier of the
instance it belongs to: ``list_permissions`` inside ``Account`` prints
``permission_uid  label`` rows, and the only thing tying them to Alan Cooper is
that the previous step entered his account. The link lives in the ORDER of the
commands, so any reader that is not allowed to use history --
``search_memory``, answer-time rehydration, the extract step, a human scrolling
a store -- cannot recover it. Measured on a real workflow, that missing link was
the cause of the large majority of rows a later reader could not resolve.

This module answers the one question the retrieval record was missing: *which
context instance*. It is deliberately generic. fastWorkflow has no framework
notion of a context instance's identity -- the current context is an arbitrary
application object whose only framework-visible fact is its class name -- so the
identity has to be DECLARED by the workflow, on the same context callback class
that already declares ``get_parent``, ``get_displayname`` and ``enter_command``:

* ``instance_label(command_context_object) -> str`` -- a classmethod (or any
  callable attribute) returning a short identity for this instance; or
* ``instance_label_attr = "uid"`` -- the name of an attribute to read off the
  context object, for a workflow that does not want to write a method.

A context that declares neither prints its NAME alone. Nothing here derives an
identifier from anything else: ``id()`` is a memory address, ``get_displayname``
is a display string a workflow may or may not derive from the instance, and a
guessed identifier that looks concrete is worse than an honest absence -- the
same reasoning ``tracing.context_handle`` gives for refusing to mint an
``instance_key``.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: A callable on the context callback class: ``instance_label(context_object)``.
INSTANCE_LABEL_HOOK = "instance_label"
#: The name of an attribute to read off the context object instead.
INSTANCE_LABEL_ATTR = "instance_label_attr"


def context_class_for(workflow: Any, context_name: str) -> Any:
    """The callback class declared for *context_name*, or None. Never raises.

    ``CommandContextModel`` is imported from its own module rather than read off
    the ``fastworkflow`` package, where the name is bound to ``None`` until
    ``fastworkflow.init()`` runs. Reading the package attribute would make the
    identity silently empty in every process that had not initialised -- which
    is the failure this module exists to refuse: an absent identity has to mean
    "the workflow declared none", never "the lookup was not wired up".
    """
    try:
        import fastworkflow
        from fastworkflow.command_context_model import CommandContextModel

        model = CommandContextModel.load(workflow.folderpath)
        return model.get_context_class(
            context_name, fastworkflow.ModuleType.CONTEXT_CLASS
        )
    except Exception:  # noqa: BLE001 - identity is presentation; never fail a turn
        logger.debug("no context class for %r", context_name, exc_info=True)
        return None


def declared_instance_label(context_class: Any, context_object: Any) -> str:
    """The instance identity *context_class* declares for *context_object*.

    ``""`` when nothing is declared, when the declaration returns nothing, or
    when it raises: an absent identity is printed as an absent identity.
    """
    if context_class is None or context_object is None:
        return ""
    hook = getattr(context_class, INSTANCE_LABEL_HOOK, None)
    if callable(hook):
        try:
            return str(hook(context_object) or "")
        except Exception:  # noqa: BLE001
            logger.debug("instance_label hook failed", exc_info=True)
            return ""
    attribute = getattr(context_class, INSTANCE_LABEL_ATTR, None)
    if isinstance(attribute, str) and attribute:
        try:
            return str(getattr(context_object, attribute, "") or "")
        except Exception:  # noqa: BLE001
            logger.debug("instance_label_attr read failed", exc_info=True)
    return ""


def context_identity(workflow: Any) -> tuple[str, str]:
    """``(context name, instance label)`` for the workflow's CURRENT context.

    ``("", "")`` at the root context, and whenever there is no workflow or no
    context to read: the root is the ambient state every turn starts in, so
    naming it on every observation would be noise, not identity.
    """
    try:
        if workflow is None:
            return ("", "")
        if getattr(workflow, "is_current_command_context_root", True):
            return ("", "")
        name = str(getattr(workflow, "current_command_context_name", "") or "")
        if not name:
            return ("", "")
        context_object = getattr(workflow, "current_command_context", None)
        return (name, declared_instance_label(
            context_class_for(workflow, name), context_object))
    except Exception:  # noqa: BLE001
        logger.debug("context identity unavailable", exc_info=True)
        return ("", "")


def context_clause_for(workflow: Any) -> str:
    """The printable ``<ContextName> <instance label>`` clause, or ``""``."""
    from fastworkflow.observation_offloading.labels import context_clause

    name, label = context_identity(workflow)
    return context_clause(name, label)


__all__ = [
    "INSTANCE_LABEL_ATTR",
    "INSTANCE_LABEL_HOOK",
    "context_class_for",
    "context_clause_for",
    "context_identity",
    "declared_instance_label",
]
