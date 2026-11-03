"""The ``approval`` annotation: a command class attribute that marks a command as needing the user's say.

A command's ``ResponseGenerator`` (or a planning command's ``PlanGenerator``) may declare::

    approval = "mandatory"    # the action cannot be undone (e.g. sending a critical email)
    approval = "delegatable"  # the user may delegate the decision

Absent means the agent decides, with no gate. This module only DECLARES and EXPOSES the annotation:
it is read the way ``descends_to`` is, validated when the routing definition loads, stored on the
command's metadata (so it lands in ``command_directory.json`` when present), and rendered as a short
marker in the planner and executor listings. Nothing here enforces anything; enforcement is
fix-3rsq.10.14. Owner guidance (2026-09-28): the annotation is not read/write gating; the two values
are the only annotations that exist.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional

import fastworkflow
from fastworkflow.command_routing import RoutingRegistry

logger = logging.getLogger(__name__)

#: The class attribute a command declares.
APPROVAL_ATTRIBUTE = "approval"
MANDATORY = "mandatory"
DELEGATABLE = "delegatable"
APPROVAL_VALUES = (MANDATORY, DELEGATABLE)

#: The short markers the listings append to a command's line. Unannotated commands get none.
APPROVAL_MARKERS = {
    MANDATORY: " (needs your approval before it runs)",
    DELEGATABLE: " (may need your approval)",
}


class ApprovalAnnotationError(ValueError):
    """A command declares an ``approval`` value that is not one of the allowed values, or not a string."""


def _allowed_text() -> str:
    return " or ".join(f"'{value}'" for value in APPROVAL_VALUES)


def check_approval_annotations(workflow_folderpath: str, routing) -> None:
    """Validate every command's ``approval`` declaration and store it on the command's metadata.

    Raises :class:`ApprovalAnnotationError` naming the command, its file and the allowed values when a
    declaration is an unknown value or not a string. Core commands carry no annotation and are skipped.
    A command whose class cannot be read is left unannotated here (its load error is raised elsewhere).
    The metadata is the snapshot: when an annotation changes, the command directory is re-saved so
    ``command_directory.json`` carries it.
    """
    problems: list[str] = []
    changed = False
    core = set(routing.command_directory.core_command_names)
    for command_name, metadata in routing.command_directory.map_command_2_metadata.items():
        if command_name in core:
            continue
        try:
            command_class = routing.get_command_class(
                command_name, fastworkflow.ModuleType.RESPONSE_GENERATION_INFERENCE)
        except Exception:  # noqa: BLE001 - an unreadable class is reported by its own load path
            logger.debug("approval declaration unread for %r", command_name, exc_info=True)
            command_class = None
        value = getattr(command_class, APPROVAL_ATTRIBUTE, None) if command_class is not None else None
        if value is not None and not isinstance(value, str):
            problems.append(
                f"{_source_label(metadata, workflow_folderpath)}: {command_name}'s "
                f"{APPROVAL_ATTRIBUTE} must be the string {_allowed_text()}, not {type(value).__name__}")
            continue
        if isinstance(value, str) and value not in APPROVAL_VALUES:
            problems.append(
                f"{_source_label(metadata, workflow_folderpath)}: {command_name}'s "
                f"{APPROVAL_ATTRIBUTE} is {value!r}; the allowed values are {_allowed_text()} "
                "(leave it out for the agent to decide, with no gate)")
            continue
        declared = value if isinstance(value, str) else None
        if metadata.approval != declared:
            metadata.approval = declared
            changed = True
    if problems:
        raise ApprovalAnnotationError("\n".join(problems))
    if changed:
        routing.command_directory.save()


def _source_label(metadata, workflow_folderpath: str) -> str:
    path = metadata.response_generation_module_path or ""
    if not path:
        return workflow_folderpath
    return os.path.relpath(os.path.abspath(path), os.path.abspath(workflow_folderpath))


def command_approval(workflow_folderpath: str, qualified_name: str) -> Optional[str]:
    """The ``approval`` a command declares: ``"mandatory"``, ``"delegatable"``, or None (agent discretion).

    None also when the workflow or the command cannot be read. A load error in an annotation is raised.
    """
    try:
        routing = RoutingRegistry.get_definition(workflow_folderpath)
    except ApprovalAnnotationError:
        raise
    except Exception:  # noqa: BLE001 - no routing definition means no annotation
        logger.debug("routing definition unread for approval of %r", qualified_name, exc_info=True)
        return None
    metadata = routing.command_directory.map_command_2_metadata.get(qualified_name)
    return metadata.approval if metadata is not None else None


def approval_marker(workflow_folderpath: str, qualified_name: str) -> str:
    """The listing marker for a command: ``""`` when it is unannotated, else its short marker."""
    value = command_approval(workflow_folderpath, qualified_name)
    return APPROVAL_MARKERS.get(value, "") if value else ""


_TEMPLATE_STEP_COMMAND = re.compile(r"[A-Za-z_][A-Za-z0-9_/]*")


def planning_steps_needing_approval(workflow_folderpath: str, planning_qualified_name: str) -> list[tuple[str, str]]:
    """The approval-annotated commands a planning command's template calls, as ``(command, approval)`` pairs.

    Each template step names the command it calls first (``find_member query=<member>``); a step's name is
    resolved to a registered command by its qualified name, or by its leaf name. Returned in template order,
    each command once. Empty when the planning command is unknown or none of its steps is annotated.
    """
    from fastworkflow import planning_commands

    command = planning_commands.planning_command(workflow_folderpath, planning_qualified_name)
    if command is None:
        return []
    routing = RoutingRegistry.get_definition(workflow_folderpath)
    directory = routing.command_directory.map_command_2_metadata
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for step in command.template:
        match = _TEMPLATE_STEP_COMMAND.match(step.strip())
        if match is None:
            continue
        called = match.group(0)
        target = called if called in directory else next(
            (name for name in directory if name.split("/")[-1] == called), None)
        if target is None or target in seen:
            continue
        seen.add(target)
        value = directory[target].approval
        if value:
            found.append((target, value))
    return found
