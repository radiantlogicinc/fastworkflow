"""Command class declarations: decorators on a command's ``ResponseGenerator`` (or ``PlanGenerator``).

A command states a fact about itself with a decorator, not a class attribute::

    @user_approval_for_execution(UserApproval.REQUIRED_ALWAYS)   # or REQUIRED_UNTIL_DELEGATED
    @writes                                                      # execution commands only
    @descends_to("Account", when_parameter="account_uid")        # when_parameter is optional
    class ResponseGenerator:
        ...

Each decorator stores its value on a private attribute (``__fw_approval__``, ``__fw_writes__``,
``__fw_descends_to__``) together with a mark that only the decorator sets. The loader
(:func:`check_command_spellings`) fails the load for three things: the retired public attributes
(``approval``, ``writes``, ``descends_to``, ``descend_parameter``), each naming its decorator, and a
private attribute written by hand without its decorator's mark.

This module imports nothing from the rest of fastworkflow, so the package and the planning module can
both re-export it.
"""
from __future__ import annotations

import enum
import logging
import os
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

_C = TypeVar("_C", bound=type)


class UserApproval(enum.Enum):
    """How a command needs the user's say before it runs (approval_annotations).

    ``REQUIRED_ALWAYS``: the user is asked every time. ``REQUIRED_UNTIL_DELEGATED``: the user is asked
    until they delegate the decision. A command with no ``@user_approval_for_execution`` is left to the
    agent, with no gate.
    """

    REQUIRED_ALWAYS = "required_always"
    REQUIRED_UNTIL_DELEGATED = "required_until_delegated"


class _DecoratorMark:
    """The value of a decorator's mark. Only this module creates one, so a hand-written mark is never it."""

    def __repr__(self) -> str:
        return "<set by a fastworkflow command decorator>"


_DECORATED = _DecoratorMark()

#: The private attributes the decorators store, and the mark each one is stored with.
APPROVAL_ATTRIBUTE = "__fw_approval__"
WRITES_ATTRIBUTE = "__fw_writes__"
DESCENDS_TO_ATTRIBUTE = "__fw_descends_to__"
_MARKS = {
    APPROVAL_ATTRIBUTE: "__fw_approval_decorated__",
    WRITES_ATTRIBUTE: "__fw_writes_decorated__",
    DESCENDS_TO_ATTRIBUTE: "__fw_descends_to_decorated__",
}
_DECORATOR_FOR = {
    APPROVAL_ATTRIBUTE: "@user_approval_for_execution(UserApproval.REQUIRED_ALWAYS)",
    WRITES_ATTRIBUTE: "@writes",
    DESCENDS_TO_ATTRIBUTE: '@descends_to("<Context>", when_parameter=...)',
}

#: The public class attributes the decorators replace. Declaring one is a load error.
RETIRED_ATTRIBUTES = ("approval", "writes", "descends_to", "descend_parameter")

_APPROVAL_BY_RETIRED_VALUE = {
    "mandatory": "REQUIRED_ALWAYS",
    "delegatable": "REQUIRED_UNTIL_DELEGATED",
}


class CommandDeclarationError(ValueError):
    """A command class spells a declaration the retired way, or writes a private attribute by hand."""


def _mark(command_class: _C, attribute: str, value: Any) -> _C:
    if not isinstance(command_class, type):
        raise TypeError(f"{_DECORATOR_FOR[attribute]} decorates a class (a ResponseGenerator or "
                        f"PlanGenerator), not {command_class!r}")
    setattr(command_class, attribute, value)
    setattr(command_class, _MARKS[attribute], _DECORATED)
    return command_class


def user_approval_for_execution(approval: UserApproval) -> Callable[[_C], _C]:
    """Gate a command's execution on the user's approval (``UserApproval.REQUIRED_ALWAYS`` or
    ``REQUIRED_UNTIL_DELEGATED``). Any other argument raises TypeError when the module is imported."""
    if not isinstance(approval, UserApproval):
        raise TypeError("@user_approval_for_execution takes a UserApproval member "
                        "(UserApproval.REQUIRED_ALWAYS or UserApproval.REQUIRED_UNTIL_DELEGATED), "
                        f"not {approval!r}")

    def decorate(command_class: _C) -> _C:
        return _mark(command_class, APPROVAL_ATTRIBUTE, approval)

    return decorate


def writes(command_class: _C) -> _C:
    """Declare that running an execution command's ``ResponseGenerator`` changes the application's data.

    A ``PlanGenerator`` cannot carry it: a planning command writes when a step of it runs a command that
    does. That is a load error, raised by :func:`fastworkflow.command_writes.check_write_declarations`.
    """
    return _mark(command_class, WRITES_ATTRIBUTE, True)


def descends_to(context: str, *, when_parameter: Optional[str] = None) -> Callable[[_C], _C]:
    """Declare the context a command enters. ``when_parameter`` names the input whose presence makes the
    command descend (the navigation edge is then conditional). Both arguments are checked when the module
    is imported."""
    if not isinstance(context, str) or not context.strip():
        raise TypeError(f"@descends_to takes the name of a context, not {context!r}")
    if when_parameter is not None and (not isinstance(when_parameter, str) or not when_parameter.strip()):
        raise TypeError(f"@descends_to(when_parameter=...) takes an input field name, not {when_parameter!r}")
    target = (context.strip(), when_parameter.strip() if when_parameter is not None else None)

    def decorate(command_class: _C) -> _C:
        return _mark(command_class, DESCENDS_TO_ATTRIBUTE, target)

    return decorate


# ---------------------------------------------------------------------------
# Readers (getattr: a subclass inherits its base's declaration, as the old attributes did)
# ---------------------------------------------------------------------------

def declared_approval(command_class: Any) -> Optional[UserApproval]:
    """The command's ``UserApproval``, or None when it declares none (agent discretion)."""
    value = getattr(command_class, APPROVAL_ATTRIBUTE, None) if command_class is not None else None
    return value if isinstance(value, UserApproval) else None


def declares_writes(command_class: Any) -> bool:
    """Whether *command_class* carries ``@writes``."""
    return command_class is not None and getattr(command_class, WRITES_ATTRIBUTE, False) is True


def declared_descend(command_class: Any) -> Optional[tuple[str, Optional[str]]]:
    """``(context, when_parameter)`` from ``@descends_to``, or None."""
    if command_class is None:
        return None
    value = getattr(command_class, DESCENDS_TO_ATTRIBUTE, None)
    return value if isinstance(value, tuple) else None


# ---------------------------------------------------------------------------
# The load check: retired spellings and hand-written private attributes
# ---------------------------------------------------------------------------

def _source_label(metadata, workflow_folderpath: str) -> str:
    path = metadata.response_generation_module_path or ""
    if not path:
        return workflow_folderpath
    return os.path.relpath(os.path.abspath(path), os.path.abspath(workflow_folderpath))


def _q(text: Any) -> str:
    return f'"{text}"'


def _retired_problems(command_name: str, command_class: type) -> list[str]:
    own = vars(command_class)
    problems: list[str] = []
    if "approval" in own:
        value = own["approval"]
        decorator_value = _APPROVAL_BY_RETIRED_VALUE.get(value) if isinstance(value, str) else None
        if decorator_value is not None:
            fix = f"write @user_approval_for_execution(UserApproval.{decorator_value})"
        else:
            fix = ("write @user_approval_for_execution(UserApproval.REQUIRED_ALWAYS) or "
                   "@user_approval_for_execution(UserApproval.REQUIRED_UNTIL_DELEGATED)")
        shown = _q(value) if isinstance(value, str) else str(value)
        problems.append(f"approval = {shown} is removed: {fix}")
    if "writes" in own:
        problems.append(f"writes = {own['writes']} is removed: write @writes on the ResponseGenerator "
                        "(execution commands only)")
    if "descends_to" in own or "descend_parameter" in own:
        target = own.get("descends_to")
        gate = own.get("descend_parameter")
        if isinstance(target, str):
            head = f"descends_to = {_q(target)} is removed"
        else:
            head = f"descend_parameter = {_q(gate)} is removed"
        args = [_q(target) if isinstance(target, str) else '"<Context>"']
        if isinstance(gate, str):
            args.append(f"when_parameter={_q(gate)}")
        problems.append(f"{head}: write @descends_to({', '.join(args)}) on the ResponseGenerator")
    return [f"{command_name}: {text}" for text in problems]


def _hand_written_problems(command_name: str, command_class: type) -> list[str]:
    own = vars(command_class)
    problems = []
    for attribute, mark in _MARKS.items():
        if attribute in own and own.get(mark) is not _DECORATED:
            problems.append(f"{command_name}: {attribute} is written by hand; declare it with "
                            f"{_DECORATOR_FOR[attribute]} instead")
    return problems


def check_command_spellings(workflow_folderpath: str, routing) -> None:
    """Fail the load for a retired spelling or a hand-written private attribute on any command class.

    Raises :class:`CommandDeclarationError` naming the file, the command and the decorator to write.
    Core commands are skipped; a command whose class cannot be read is left to its own load path.
    """
    import fastworkflow

    problems: list[str] = []
    core = set(routing.command_directory.core_command_names)
    for command_name, metadata in routing.command_directory.map_command_2_metadata.items():
        if command_name in core:
            continue
        try:
            command_class = routing.get_command_class(
                command_name, fastworkflow.ModuleType.RESPONSE_GENERATION_INFERENCE)
        except Exception:  # noqa: BLE001 - an unreadable class is reported by its own load path
            logger.debug("declarations unread for %r", command_name, exc_info=True)
            continue
        if command_class is None:
            continue
        where = _source_label(metadata, workflow_folderpath)
        for text in (*_retired_problems(command_name, command_class),
                     *_hand_written_problems(command_name, command_class)):
            problems.append(f"{where}: {text}")
    if problems:
        raise CommandDeclarationError("\n".join(problems))
