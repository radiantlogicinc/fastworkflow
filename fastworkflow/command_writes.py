"""The ``writes`` declaration: a command that changes the application's data says so (fix-fixv).

An execution command's ``ResponseGenerator`` declares::

    writes = True    # running it changes the application's data (adds a note, cancels an order)

Absent (or False) means the command only reads, or only moves the workflow between records. A
planning command does not declare it: it writes when one of its steps (directly, or through a planning
command it calls) runs a command that writes. A planning command whose steps reach no write is
read-only.

What the framework does with it. Inside a planning command's phase, the agent runs the steps of that
phase, and nothing else should change the application. So in a phase, a command that writes runs only
when one of the phase's steps names it:

- the phase's command listing leaves out the commands that write and that no step of the phase names
  (``unplanned_write_commands``);
- a call of such a command inside the phase is refused unrun, with a message that says why
  (``phase_write_refusal``, called by ``CommandExecutor`` before the approval gate).

Outside a planning command's phase (the turn's agent, a plain goal, a command the user typed) nothing
changes. The declaration is validated when the routing definition loads: it must be ``True`` or
``False``, and only an execution command may carry it.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import fastworkflow

logger = logging.getLogger(__name__)

#: The class attribute an execution command's ``ResponseGenerator`` declares.
WRITES_ATTRIBUTE = "writes"

#: The refusal the agent reads when a phase calls a write command its steps do not name.
UNPLANNED_WRITE_REFUSAL = (
    "{command} was not run: it changes the application's data, and no step of this part of the job "
    "runs it. Do only the steps given; this part of the job writes nothing that its steps do not name.")


class WritesDeclarationError(ValueError):
    """A command declares ``writes`` with a value that is not a bool, or on a planning command."""


def declares_writes(command_class: Any) -> bool:
    """Whether *command_class* (a ``ResponseGenerator``) declares ``writes = True``."""
    return getattr(command_class, WRITES_ATTRIBUTE, False) is True


def check_write_declarations(workflow_folderpath: str, routing) -> None:
    """Validate every command's ``writes`` declaration when the routing definition loads.

    Raises :class:`WritesDeclarationError` naming the command and its file when the value is not a
    bool, or when a planning command declares it (its steps say whether it writes). Core commands are
    skipped; a command whose class cannot be read is left to its own load path.
    """
    problems: list[str] = []
    core = set(routing.command_directory.core_command_names)
    for command_name, metadata in routing.command_directory.map_command_2_metadata.items():
        if command_name in core:
            continue
        try:
            command_class = routing.get_command_class(
                command_name, fastworkflow.ModuleType.RESPONSE_GENERATION_INFERENCE)
        except Exception:  # noqa: BLE001 - an unreadable class is reported by its own load path
            logger.debug("writes declaration unread for %r", command_name, exc_info=True)
            continue
        if command_class is None or WRITES_ATTRIBUTE not in vars(command_class):
            continue
        value = vars(command_class)[WRITES_ATTRIBUTE]
        where = _source_label(metadata, workflow_folderpath)
        if not isinstance(value, bool):
            problems.append(f"{where}: {command_name}'s {WRITES_ATTRIBUTE} must be True or False, "
                            f"not {type(value).__name__}")
        elif metadata.is_planning:
            problems.append(f"{where}: {command_name} is a planning command; it does not declare "
                            f"{WRITES_ATTRIBUTE}. It writes when its steps run a command that declares "
                            f"{WRITES_ATTRIBUTE} = True: declare it on that execution command")
    if problems:
        raise WritesDeclarationError("\n".join(problems))


def _source_label(metadata, workflow_folderpath: str) -> str:
    path = metadata.response_generation_module_path or ""
    if not path:
        return workflow_folderpath
    return os.path.relpath(os.path.abspath(path), os.path.abspath(workflow_folderpath))


def write_commands(workflow_folderpath: str) -> frozenset[str]:
    """The qualified names of the commands of *workflow_folderpath* that write: each execution command
    that declares ``writes = True``, and each planning command whose steps reach one (through the
    planning commands they call). Empty when the workflow cannot be read."""
    try:
        routing = fastworkflow.RoutingRegistry.get_definition(workflow_folderpath)
    except Exception:  # noqa: BLE001 - no routing definition: nothing is known to write
        logger.debug("routing definition unread for writes of %r", workflow_folderpath, exc_info=True)
        return frozenset()
    writes: set[str] = set()
    for command_name, metadata in routing.command_directory.map_command_2_metadata.items():
        if metadata.is_planning:
            continue
        try:
            command_class = routing.get_command_class(
                command_name, fastworkflow.ModuleType.RESPONSE_GENERATION_INFERENCE)
        except Exception:  # noqa: BLE001 - an unreadable class declares nothing
            continue
        if declares_writes(command_class):
            writes.add(command_name)
    if not writes:
        return frozenset()
    try:
        from fastworkflow import planning_commands

        callees = planning_commands.resolve_field_sources(workflow_folderpath).callees
    except Exception:  # noqa: BLE001 - planning commands unread: only the execution commands are known
        return frozenset(writes)
    changed = True
    while changed:   # a planning command writes when any command its steps call writes
        changed = False
        for name, called in callees.items():
            if name not in writes and called & writes:
                writes.add(name)
                changed = True
    return frozenset(writes)


def _host(chat_session: Any) -> Any:
    """The WorkflowExecutionContext behind *chat_session* (a ChatSession holds one as its core)."""
    core = getattr(chat_session, "_core", None)
    return core if core is not None else chat_session


def _plain(name: str) -> str:
    return str(name or "").strip().split("/")[-1].lower()


def phase_planned_commands(chat_session: Any) -> Optional[frozenset[str]]:
    """The plain names of the commands the steps of the planning-command phase now running name, or
    None when no planning command's phase is running (the turn's own agent, a typed command)."""
    host = _host(chat_session)
    try:
        state = getattr(host, "_phase_run", None)
    except Exception:  # noqa: BLE001 - a host without frames runs no phase
        return None
    if not isinstance(state, dict) or not state.get("plan"):
        return None
    from fastworkflow import planning_phases

    try:
        names = host._phase_command_names()
        return frozenset(planning_phases.planned_steps(state["plan"], **names))
    except Exception:  # noqa: BLE001 - steps that cannot be read name nothing for sure: no guard
        logger.debug("phase steps unread for the writes guard", exc_info=True)
        return None


def unplanned_write_commands(chat_session: Any, workflow_folderpath: str) -> frozenset[str]:
    """The commands that write and that no step of the running phase names, by qualified and by short
    name (the listings read either). Empty outside a planning command's phase."""
    planned = phase_planned_commands(chat_session)
    if planned is None:
        return frozenset()
    out: set[str] = set()
    for name in write_commands(workflow_folderpath):
        if _plain(name) not in planned:
            out.update({name, name.split("/")[-1]})
    return frozenset(out)


def phase_write_refusal(chat_session: Any, workflow_folderpath: str, command_name: str) -> Optional[str]:
    """The refusal for running *command_name* now, or None to run it: a command that writes, called
    inside a planning command's phase whose steps do not name it."""
    planned = phase_planned_commands(chat_session)
    if planned is None or _plain(command_name) in planned:
        return None
    if command_name not in write_commands(workflow_folderpath):
        return None
    return UNPLANNED_WRITE_REFUSAL.format(command=command_name)
