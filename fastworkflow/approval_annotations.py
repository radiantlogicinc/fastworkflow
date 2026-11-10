"""The approval annotation: a decorator on a command class that marks a command as needing the user's say.

A command's ``ResponseGenerator`` (or a planning command's ``PlanGenerator``) may declare::

    @user_approval_for_execution(UserApproval.REQUIRED_ALWAYS)            # cannot be undone (e.g. a critical email)
    @user_approval_for_execution(UserApproval.REQUIRED_UNTIL_DELEGATED)   # the user may delegate the decision

Absent means the agent decides, with no gate. The decorator and the enum live in
``fastworkflow.command_declarations``. This module VALIDATES and EXPOSES the annotation: it is read from
the class, checked when the routing definition loads, stored on the command's metadata (so it lands in
``command_directory.json`` as the enum's value), and rendered as a short marker in the planner and
executor listings. Nothing here enforces anything; the approval gate in ``approval_binding`` reads it.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional

import fastworkflow
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.command_declarations import CommandDeclarationError, UserApproval, declared_approval

logger = logging.getLogger(__name__)

#: The stored values (``CommandMetadata.approval``, ``command_directory.json``): the enum's values.
REQUIRED_ALWAYS = UserApproval.REQUIRED_ALWAYS.value
REQUIRED_UNTIL_DELEGATED = UserApproval.REQUIRED_UNTIL_DELEGATED.value
APPROVAL_VALUES = tuple(member.value for member in UserApproval)

#: The short markers the listings append to a command's line. Unannotated commands get none.
APPROVAL_MARKERS = {
    REQUIRED_ALWAYS: " (needs your approval before it runs)",
    REQUIRED_UNTIL_DELEGATED: " (may need your approval)",
}


class ApprovalAnnotationError(ValueError):
    """A command's approval declaration is not a ``UserApproval`` member (the decorator prevents this)."""


def check_approval_annotations(workflow_folderpath: str, routing) -> None:
    """Snapshot each command's ``@user_approval_for_execution`` value onto its metadata.

    The decorator itself rejects a non-``UserApproval`` argument at import, so a value that is not an
    enum member here is an internal error and raises :class:`ApprovalAnnotationError`. Retired spellings
    and hand-written private attributes are rejected by ``command_declarations.check_command_spellings``.
    Core commands carry no annotation and are skipped. A command whose class cannot be read is left
    unannotated here (its load error is raised elsewhere). The metadata is the snapshot: when an
    annotation changes, the command directory is re-saved so ``command_directory.json`` carries it.
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
        declared = declared_approval(command_class)
        if declared is None and command_class is not None and "__fw_approval__" in vars(command_class):
            problems.append(f"{_source_label(metadata, workflow_folderpath)}: {command_name}'s approval "
                            "is not a UserApproval member")
            continue
        stored = declared.value if declared is not None else None
        if metadata.approval != stored:
            metadata.approval = stored
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
    """The ``UserApproval`` value a command declares (``"required_always"``, ``"required_until_delegated"``),
    or None (agent discretion).

    None also when the workflow or the command cannot be read. A load error in an annotation is raised.
    """
    try:
        routing = RoutingRegistry.get_definition(workflow_folderpath)
    except (ApprovalAnnotationError, CommandDeclarationError):
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
