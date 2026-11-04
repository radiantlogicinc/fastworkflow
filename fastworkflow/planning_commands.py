r'''Planning commands as parameterized commands.

A planning command is one Python file, ``_commands/<name>.py``, written with the abstractions of a
command file: a ``Signature`` whose docstring describes the job (what the planning command matcher and
the text planner read) and whose pydantic ``Input`` declares typed parameters, and a
``class PlanGenerator`` (as a command has a ``ResponseGenerator``) whose methods write the
planning command's steps for one set of arguments (``fastworkflow.planning``). The planner fills the
parameters from the request (one typed dspy call), the arguments are checked, and the
``PlanGenerator`` expands them into the numbered plan. A planning command without parameters and
without phases gives fixed steps: a call of it runs those steps as its one phase, with no
planning call, and its steps are also the template the text planner reads for a request
it matches (see ``workflow_agent.load_planning_commands``); one without parameters that has phases is a
planning command call with no arguments (its phases, with no planning call).

A planning command sits in ``_commands`` as an execution command does: a root planning command is the file
``_commands/<name>.py`` whose name is the file stem, and a context planning command in
``_commands/<Context>/<name>.py`` is named ``"<Context>/<stem>"`` (as ``_commands/<Context>/``
names its commands); the file stem is a Python identifier (snake_case: ``show_record.py``).
A ``.py`` file or subfolder whose name starts with ``_`` or ``.`` is skipped (``_kinds.py``
is a helper the planning command files import with ``from ._kinds import FIND``, not a planning command).
``_commands/_split.py`` is not a planning command either: it configures the split stage (the
splitter's instructions, the identifier patterns of the split merge, the referent step's subject
hints; see ``workflow_agent.split_settings``). No other file is read.

The split settings file, every attribute optional::

    import re


    class Signature:
        """Split the user's request into the separate requests it bundles together. ..."""


    # Account numbers, as the commands take them: acct_4421
    identifier_patterns = [re.compile(r"\bacct_[0-9]+")]

    subject_hints = """Records in this workflow are accounts and their owners."""

The ``Signature`` docstring replaces the built-in splitter's instructions (``Signature.Input`` /
``Signature.Output`` may describe the splitter fields ``job_types``, ``user_query`` and
``subtasks``); a pattern that does not compile, or a file that does not import, fails the load.

A planning command file::

    from typing import Literal

    from pydantic import BaseModel, Field

    FIND = {"group": "find_group", "unit": "find_unit"}


    class Signature:
        """Show one named group or unit and its members. One job per record."""

        class Input(BaseModel):
            name: str = Field(description="the record's name",
                              json_schema_extra={"in_request": {"subject": True, "ignore": ["group", "unit"]}})
            kind: Literal["group", "unit"] = Field(
                description="what the request is about",
                json_schema_extra={"verify": {"group": "a group of people", "unit": "an organisation unit"}})
            person: str = Field(default="", description="one named person to check for; empty for everyone",
                                json_schema_extra={"in_request": "subject"})
            include_risk: bool = Field(default=False, description="the request asks how risky it is",
                                       json_schema_extra={"verify": True})
            count: int = Field(default=2, ge=0, le=20, description="how many members to open",
                               json_schema_extra={"in_request": True})

        plain_utterances = ["who is in the Payroll group", "is Ann Lee in Billing"]

        @staticmethod
        def validate_extracted_parameters(workflow, command, cmd_parameters, command_text=None):
            if cmd_parameters.include_risk and cmd_parameters.person:
                raise fastworkflow.Refused("the risk is assessed for the whole record, not for one person")


    class PlanGenerator:
        def members(self, input):
            out = [f"{FIND[input.kind]} query={input.name}"]
            out.append(f"list_members filter={input.person}" if input.person else "list_members")
            if input.include_risk:
                out.append("assess_risk")
            for n in range(1, input.count + 1):
                out.append(f"open member {n}")
            return out

- ``Signature`` (required) with a docstring (required; its whitespace is collapsed to one
  line). ``Input`` (a pydantic ``BaseModel``) declares the parameters in order; a planning command
  without parameters leaves it out or empty. A field's type is ``str``, ``bool``, ``int``
  or ``Literal[...]`` of text values (an enum). ``Field(default=...)`` makes it optional (no
  default: required; a ``bool`` always defaults to false, a ``default=True`` fails the
  load: the model sets it true only when the request asks for that part). A ``str``
  default is the value ``steps()`` gets when the request gives none, so ``"NOT_FOUND"``
  would be written into the steps: an argument the request may leave out defaults to
  ``""``, or is ``Optional[str] = None`` (the ``PlanGenerator``, the hook and the
  validators then get None for it; to the planner it is a str that defaults to empty).
  ``description``
  is what the planner and a ``verify`` check read; ``ge``/``le`` bounds are enforced by the
  argument screen (see ``check_arguments``). The planning command options go in ``json_schema_extra``
  (each optional; unknown keys fail the load):

  - ``in_request: "subject"`` (a ``str``): the argument names a record of the request (a person, a
    group, a product, a code). Besides being written in the request it must hold one of the
    request's subjects whole (``split_merge.request_subjects``, with the workflow's identifier
    patterns) or a name-like word (capitalised and not a common word, or holding a digit or
    ``_``) or be a name written in lower case (two or more uncommon words cutting no longer
    run of them: "ann lee"), and cut no subject: "I" of "Before I approve" and "Payroll" of "Payroll_Approver"
    are refused (an optional argument takes its default; a required one fails the call). One
    cut short, mid-word or at a space, at the start or end of exactly one subject becomes
    that subject ("Payroll_Admin" of "Payroll_Administrator", "Ann" of "Ann Lee").
    It implies ``in_request``. It replaces the retired ``subject: True`` option.
  - ``verify: True`` (a ``bool``, an enum, or a ``str`` with ``in_request: "subject"``): a classifier
    model decides the argument again from the request alone, when the planner path has one
    (``typed_planning_call``'s ``check_lm``); its answer replaces the planner's when it is at least
    ``VERIFY_THRESHOLD`` sure. It reads the field's ``description``, and an enum's meanings. An enum
    takes ``verify: {"group": "a group, people are its members", ...}`` instead: a dict giving each
    value's meaning, every key one of the enum's values (it replaces the retired ``means`` option).
    For a subject it chooses among the request's subjects (at most ``MAX_SUBJECT_OPTIONS``)
    or ``NO_SUBJECT`` ("none": the default, or a failed call for a required parameter); a
    request naming no subject is not checked.
  - ``in_request: True`` (a ``str``, an ``int`` or a ``list[str]``): a ``str`` must appear in the request
    exactly as the request writes it (a verbatim substring); an ``int`` must be a number
    the request writes (digits or the words zero to twenty outside the text of the other
    arguments); otherwise it takes the only one written, or its default when that is
    written, and with several others the call fails ("the first two" never runs as 3); a
    ``list[str]`` keeps each item the request writes and drops the others.
  - ``in_request: {"ignore": [words]}`` (a ``str``): words removed from either end of the value before
    the check (the cleanup the retired ``strip`` option did; it implies ``in_request``). A field with no
    check takes its cleanup in code: ``strip_words(text, words)`` in its validator or in
    ``Signature.validate_extracted_parameters``.
  - No check is implied by a field's type (design 2.46): a field without an option is not
    checked against the request, on either kind of command. A check option on a field of a
    type it does not apply to fails the load, naming the option and the type.

- ``PlanGenerator`` (required): its public methods are the planning command's phases, in order, each
  returning its steps as a list of str, one per step (empty ones are dropped); see
  ``fastworkflow.planning`` for the rules. The methods build text only: they must not call a
  backend or read state. A repeated part is a ``for`` loop over ``range()``, bounded by the
  int field's ``le``; steps shared by several planning commands go in a helper module
  (``_commands/_<name>.py``). Lookup tables are plain module-level dicts. A planning command whose one
  phase reads and fills no facts runs flat (one plan); one without parameters whose one
  method takes no ``input`` gives fixed steps.
- ``plain_utterances`` (optional, with or without params): example requests for the
  planning command, written the way people ask for that job. The planning command matcher shows the first
  ``workflow_agent.MATCHER_EXAMPLES`` of them after the description ("e.g. ..."); a planning command
  call also keeps them on its command (``PlanningCommandCommand.plain_utterances``). Write
  them generically: they are in every matcher prompt.
- ``template`` (optional): the template text of the planning command, shown to the planner in the
  planning command call's "expands to:" and used as the text planner's template when a planning command call
  fails. Without it the template is the steps rendered on placeholder inputs
  (``placeholder_template``): every combination of a ``<name>`` for a str (and "" when that
  is its default), true and false for a bool, the default of an int and each value of an
  enum (past ``MAX_GRID`` combinations, the defaults and each one or two changes of them,
  ``input_grid``, and a warning at load), merged into one list in which a step not every
  combination writes says when it is
  written (``(only when person is given)``) and the steps of an enum's values are joined
  (``<find_group or find_unit, by kind> query=<name>``). A repeated part shows only its
  steps for the int's default: declare ``template`` when the planner should see how it
  repeats ("repeat count times (n = 1 to count): open member <n>").
- Phases (with or without params; without, every call has the one plan of the empty
  ``Input``): a ``PlanGenerator`` method that reads an earlier phase's
  facts, or fills its own (its return annotation), makes the call run as phases: each its
  own bounded sub-plan, passing its filled facts model to later phases by parameter name
  (``fastworkflow.planning_phases``; the loader builds its ``Phase`` list,
  ``planning.planned``). At load the phases are built on the placeholder inputs: their
  parameters must bind (``_checked_plan``), their ``"from"`` must be well formed, and each
  method's template call must succeed (``PlanningCommandFormatError`` otherwise); every command step
  must name a workflow command and only parameters of its ``Signature.Input``
  (``check_step_commands``: the planning command is skipped at load, logged as an error, and
  ``fastworkflow command preview --check`` fails); a ``"from"`` naming a command or field the
  workflow lacks is a logged warning (``check_phase_sources``), not a load failure.
- ``validate_extracted_parameters`` (optional, the signature of a command's): called with
  the checked arguments as an ``Input`` (``_validate_arguments``); it returns None to accept, and
  ``raise fastworkflow.Refused(message, field=...)`` refuses the call (the planner gets the message and
  one more try, see the rules below); any other return is a call error; an exception
  fails it (the planner falls back), and a field it corrects in place is used. An argument that only means something when other arguments take some value is
  normalised here: the hook writes its default back (``if not cmd_parameters.include_risk:
  cmd_parameters.owner = ""``). ``placeholder_template`` runs the hook too, on each
  placeholder combination (with no workflow and an empty ``command_text``): a combination it
  changes or refuses is left out of the template (the planner never expands it), so the
  ``(only when ...)`` conditions follow the hook.
- Rules across fields (optional): pydantic validators on ``Input`` (``model_validator`` /
  ``field_validator``) that raise ``ValueError`` with a message saying what to fix ("record_a
  and record_b name the same record; record_b is the other one the request names"). They run
  on the arguments ``validate_extracted_parameters`` leaves (``check_input``), with the
  validation context ``{"request": <the request>, "workflow": <the workflow or None>}``
  (``info.context`` of a validator that takes ``info``). A refusal, and a hook's ``(False,
  message)``, is a ``PlanningCommandInputError``: the planner is given the message and fills the call
  once more (``typed_planning_call``); a second refusal falls back. So is a required argument
  the guards of ``check_arguments`` refuse (missing, invalid, not written in the request).
  They also run on the placeholder combinations (request ``""``): one they refuse is left out
  of the template.

The file is imported as a command file is (``python_utils.get_module`` under the workflow's
package, so package-relative imports work). A file that fails to import, or whose declaration
is malformed (a ``PlanningCommandFormatError`` naming the file), fails its load: the planning command is logged as
an error and skipped, the workflow's other planning commands still load, and ``fastworkflow command
preview`` fails naming it.

Any ``str`` argument that lies inside a match of one of the workflow's identifier patterns
(``identifier_patterns`` in ``_commands/_split.py``, e.g. ``\bacct_[0-9]+``) without being all of it becomes the
whole match ("4421" of "acct_4421"); inside several, it is refused.

The model's ``steps_before`` / ``steps_after`` lose any ``<...>`` placeholder (and the
``key=`` that holds one); a step holding a uid-shaped token the request does not write is
dropped, and so is a step whose commands outside the expansion no open part of the request
asks for, by a word of the command's name, catalog description or plain_utterances
(``filter_extra_steps``, ``command_texts``).

A planning command call's ``str`` and ``int`` arguments first go through the screen a command's
parameters go through (``utils.signatures.screen_parameter_values``: quotes and brackets
removed, placeholders and out-of-range numbers refused), and a str argument is looked up in
the request with Unicode dashes folded to "-" (see ``check_arguments``).
'''
from __future__ import annotations

import functools
import inspect
import json
import math
import os
import re
import sys
from collections import deque
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from types import UnionType
from typing import Any, Callable, Literal, Mapping, Optional, Sequence, Union, get_args, get_origin

from pydantic import BaseModel

from fastworkflow.command_context_model import CommandContextModel
from fastworkflow.command_directory import CommandDirectory, compute_commands_source_fingerprint
from fastworkflow.planning_command_json_schema_internal import (
    HAND_WRITTEN_AVAILABLE_FROM_KEY as _HAND_WRITTEN_AVAILABLE_FROM_KEY,
    PLANNING_COMMAND_JSON_SCHEMA_EXTRA_INTERNAL_KEYS as _PLANNING_COMMAND_JSON_SCHEMA_EXTRA_INTERNAL_KEYS,
)
from fastworkflow.utils.workflow_vocabulary import (
    KIND_WORD_WHY, WorkflowVocabulary, description_kind_options, is_kind_word, is_pronoun_phrase,
    load_workflow_vocabulary)
from fastworkflow.observation_offloading.facts import FactSourceError, fact_sources
from fastworkflow.observation_offloading.finish_check import (
    KIND_COMMAND,
    KIND_CONDITIONAL,
    KIND_SKIPPED,
    _CONDITION_RE,
    _NEGATION_RE,
    _outside_quoted_values,
    parse_plan,
    step_chain,
)

#: Like ``finish_check._CONDITION_RE``, but bare ``only`` in a note is not a condition word.
_NOTE_CONDITION_RE = re.compile(
    r"\b(?:if|only\s+(?:if|when)|when|whenever|unless|otherwise|optional(?:ly)?|in case|as needed|"
    r"where (?:needed|applicable)|else)\b", re.IGNORECASE)
from fastworkflow import planning as planning_api
from fastworkflow.planning_phases import (
    FULL_LIST_PHRASES,
    PhaseError,
    _PICK_NAMED_RE,
    active_phases,
    is_flat_plan,
    is_from_source,
    phase_steps,
    placeholder_steps,
)
from fastworkflow.utils import python_utils
from fastworkflow.utils.logging import logger
from fastworkflow.workflow_inheritance_model import (
    WorkflowInheritanceModel,
    WorkflowInheritanceModelError,
)

#: A ``verify`` check replaces the planner's argument only at or above this probability.
VERIFY_THRESHOLD = 0.65
NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}
#: The value a ``verify`` check of a ``subject`` parameter gives when the request names none.
NO_SUBJECT = "none"
#: A ``verify`` check of a ``subject`` parameter chooses among at most this many of the request's subjects.
MAX_SUBJECT_OPTIONS = 8
#: Names a parameter may not take (they are fields of the planner's typed call).
RESERVED_NAMES = frozenset({"user_query", "available_commands", "steps_before", "steps_after",
                            "reasoning"})

_NAME = r"[A-Za-z_][A-Za-z0-9_]*"
_COMMAND_LINE = re.compile(r"^- ([A-Za-z_][A-Za-z0-9_]*)\s*$", re.M)
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUMBER = re.compile(r"\b(\d+|" + "|".join(NUMBER_WORDS) + r")\b", re.I)
#: A 16+ hex-digit run with a digit in it, or a dashed uuid: an id only the run can know.
_UID = re.compile(r"\b(?=[0-9a-f-]*\d)(?:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|[0-9a-f]{16,})\b",
                  re.I)
_ARG_PLACEHOLDER = re.compile(r"\s*\b[A-Za-z_][\w.]*\s*[=:]\s*<[^<>]*>")
_BARE_PLACEHOLDER = re.compile(r"\s*<[^<>]*>")


class PlanningCommandFormatError(ValueError):
    """A malformed planning command file: its ``Signature`` or an ``Input`` field."""


class PlanningCommandCallError(ValueError):
    """A planning command call that cannot be used (bad argument, failed guard); the planner falls back."""


class PlanningCommandInputError(PlanningCommandCallError):
    """The arguments are refused: a pydantic validator of the planning command's ``Input`` raised,
    ``validate_extracted_parameters`` raised ``fastworkflow.Refused``, or a guard of
    ``check_arguments`` refused one of the planner's arguments (missing, not a valid value, not
    written in the request). The message says what to fix; ``typed_planning_call`` gives it to the
    planner for one more typed call. ``fields`` names the arguments a ``Refused(field=...)`` named:
    the question put to the user then asks for those arguments."""

    def __init__(self, message: str = "", fields=()):
        super().__init__(message)
        self.fields: tuple[str, ...] = tuple(fields)


#: The refusal of one argument: ``name=<value repr> <why>`` (see ``check_arguments``).
_ARGUMENT_REFUSAL = re.compile(r"^(?P<name>[A-Za-z_]\w*)=(?P<value>.*?) (?P<why>(?:is|names) .*)$")
_MISSING_ARGUMENT = re.compile(r"^missing required argument (?P<name>[A-Za-z_]\w*)$")


#: The refusals of a free-text value that ``refusal_question`` asks for without quoting the value:
#: a template token or an invented name, a pronoun the request does not tie to one subject, a
#: word of the workflow's own vocabulary (a kind of record, a field name, a declared choice).
PLACEHOLDER_WHY = "is a placeholder, not a value"
PRONOUN_WHY = "is a pronoun the request does not tie to a single subject"
#: Words of an invented example value ("John Doe"): a closed list, as PRONOUNS is.
PLACEHOLDER_WORDS = frozenset({
    "john", "jane", "doe", "smith", "foo", "bar", "baz", "example", "placeholder", "unknown",
    "unspecified", "tbd", "todo", "n/a", "none", "null"})
#: A template token written as is: brackets and braces, an upper-case ``NAME_1``, "unspecified_x".
_PLACEHOLDER_SHAPE = re.compile(r"[<>{}]|\b[A-Z][A-Z_]*_\d+\b|(?i:\bunspecified\w*|\bplaceholder\b)")


def _placeholder_like(text: str, folded_request: str) -> bool:
    """A template token or an invented example value. The example words count only when the
    request does not itself write the value (a real person named "John Doe" passes)."""
    if _PLACEHOLDER_SHAPE.search(text):
        return True
    words = re.findall(r"[a-z/]+", text.lower())
    return bool(words) and any(w in PLACEHOLDER_WORDS for w in words) and text.lower() not in folded_request


def _role_fits(request: str, value: str, p: "Param") -> bool:
    """Whether the request writes *p*'s role just before *value*: the input's name, as words, among
    the four words before the value's first occurrence in the request."""
    folded = _fold(request).lower()
    at = folded.find(_fold(value).lower())
    if at < 0:
        return False
    words = re.findall(r"[a-z0-9_]+", folded[:at].replace("_", " "))[-4:]
    role = p.name.lower().replace("_", " ")
    return re.search(rf"\b{re.escape(role)}s?\b", " ".join(words)) is not None


def _planning_commands_workflow_root(path: str) -> str | None:
    """The workflow folder of a planning command file: the parent of the ``_commands`` folder it sits in."""
    for parent in Path(path).parents:
        if parent.name == "_commands":
            return str(parent.parent)
    return None


def _vocabulary_for(command: "PlanningCommandCommand") -> WorkflowVocabulary | None:
    """The vocabulary of the workflow the planning command belongs to (the stored file when fresh, else
    derived in memory); None when the planning command has no workflow folder or the derivation fails."""
    root = _planning_commands_workflow_root(command.path) if command.path else None
    if root is None:
        return None
    try:
        return load_workflow_vocabulary(root)
    except Exception:  # noqa: BLE001 - the workflow's own code; the screens are skipped, not the call
        logger.warning(f"could not derive the vocabulary of {root}", exc_info=True)
        return None


def _kind_selector_options(p: Param, vocabulary: WorkflowVocabulary | None) -> frozenset[str]:
    """The kind words a kind-selector argument lists in its description (the same rule the
    command parameter screen applies, ``description_kind_options``). A planning command has no labeled
    examples of its own, so only its description can make it one."""
    if vocabulary is None:
        return frozenset()
    return description_kind_options(p.desc, vocabulary.kind_words)


def missing_argument_question(command: PlanningCommandCommand, name: str) -> str:
    """The question put to the user for a required argument the request does not give.

    The same words for every route that finds the argument missing: a planning command call's refusal
    (``refusal_question``), whatever its check found.
    """
    task = command.name.replace("_", " ")
    label = name.replace("_", " ")
    return (f"I can't run {task} yet: your request does not say which {label} to use. "
            f"Which {label} should I use?")


def refusal_question(command: PlanningCommandCommand, refusal: PlanningCommandInputError) -> str:
    """The question put to the user when a refused planning command call cannot be filled from the request.

    It names the recipe's input and says what is wrong with it, in plain words: the request
    does not say it, the value the planner took is not in the request, or it is not one of the
    allowed values. A refusal that names no input (a rule the planning command's own hook or Input raised)
    is put in its own words.
    """
    task = command.name.replace("_", " ")
    text = str(refusal)
    if refusal.fields:
        # A Refused(field=...) from the hook: the arguments it named are asked for, with its message
        labels = " and ".join(name.replace("_", " ") for name in refusal.fields)
        detail = re.sub(r"^validate_extracted_parameters: ", "", text)
        return f"I can't run {task} yet: {detail}. What should the {labels} be?"
    if match := _MISSING_ARGUMENT.match(text):
        return missing_argument_question(command, match["name"])
    match = _ARGUMENT_REFUSAL.match(text)
    param = next((p for p in command.params if match and match["name"] == p.name), None)
    if param is not None:
        label = param.name.replace("_", " ")
        value, why = match["value"], match["why"]
        if why in (PLACEHOLDER_WHY, PRONOUN_WHY, KIND_WORD_WHY):
            return missing_argument_question(command, param.name)
        if why.startswith("is also the value of "):
            other = why.removeprefix("is also the value of ").replace("_", " ")
            return (f"I can't run {task} yet: {value} is the {other} in your request, not the {label}. "
                    f"Which {label} should I use?")
        if why.startswith("is not written in the request"):
            wrong = f"the {label} {value} is not in your request"
        elif why.startswith("names no record of the request"):
            wrong = f"{value} does not match anything in your request"
        elif why.startswith("is not one of "):
            wrong = f"the {label} must be one of: {', '.join(param.values)}"
        elif why.startswith("is not an integer"):
            wrong = f"the {label} must be a whole number"
        elif why.startswith("is not true or false"):
            wrong = f"the {label} must be yes or no"
        else:
            wrong = f"the {label} {value} is not valid"
        return f"I can't run {task} yet: {wrong}. Which {label} should I use?"
    detail = re.sub(r"^(?:validate_extracted_parameters|the arguments break a rule of \S+): ", "", text)
    return f"I can't run {task} yet: {detail}. What should I change?"


def refusal_names_an_input(command: PlanningCommandCommand, refusal: PlanningCommandInputError) -> bool:
    """Whether a refused planning command call is something to ask the user: the refusal names one of
    the command's inputs (``Refused(..., field=...)`` from the hook, a missing argument, or a guard's
    refusal of one argument's value).

    A refusal that names no input (``Refused(message)`` without ``field``, or an ``Input`` rule) says
    the call does not fit the request: "owner is read only with kind='application'" is the choice of
    command, not a value the user can supply. The turn then plans the request with the text planner,
    as an unattended turn does, instead of putting the rule to the user.
    """
    if refusal.fields:
        return True
    text = str(refusal)
    if _MISSING_ARGUMENT.match(text):
        return True
    match = _ARGUMENT_REFUSAL.match(text)
    return match is not None and any(p.name == match["name"] for p in command.params)


# ---------------------------------------------------------------- the parsed planning command

@dataclass(frozen=True)
class Param:
    name: str
    type: str  # "str" | "bool" | "int" | "float" | "list[str]" | "enum"
    values: tuple[str, ...] = ()
    #: A list[str] default is a tuple (a Param is hashable); the others are the value.
    default: object = None
    required: bool = False
    desc: str = ""
    verify: bool = False
    #: ``verify: "alone"`` (a bool): the classifier alone decides the field from the request; the typed
    #: call and the parameter extractor never fill it, and it keeps its default when the classifier is
    #: not sure or there is no JEV_API_KEY (also sets ``verify``).
    alone: bool = False
    #: A verify check's {value: meaning} (an enum), in the order of ``values``.
    means: tuple[tuple[str, str], ...] = ()
    #: ``in_request: True`` or ``in_request: "subject"`` (which also sets ``subject``).
    in_request: bool = False
    subject: bool = False
    #: ``in_request: {"ignore": [words]}`` (a str): words removed from either end of the value before the check.
    ignore: tuple[str, ...] = ()

    def type_text(self) -> str:
        return "|".join(self.values) if self.type == "enum" else self.type


@dataclass(frozen=True)
class PlanningCommandCommand:
    """A planning command called with typed arguments: its signature and its ``steps()``. Every planning command
    file has one; a planning command without parameters or ``plan()`` (``fixed_steps``) has no
    parameters to fill and its one phase reads nothing."""

    name: str
    description: str
    params: tuple[Param, ...]
    path: str = field(default="", compare=False)
    #: The workflow's identifier patterns (``workflow_agent.identifier_patterns``), set at load.
    identifiers: tuple = field(default=(), compare=False)
    #: The planning command's pydantic ``Input``. Planning command-call arguments are screened through it like a
    #: command's parameters.
    input_model: type | None = field(default=None, compare=False)
    #: The planning command file's ``steps(input)``.
    steps_fn: Callable | None = field(default=None, compare=False)
    #: The template text (``Signature.template``, or ``steps()`` rendered on placeholder
    #: inputs).
    template: tuple[str, ...] = field(default=(), compare=False)
    #: ``Signature.validate_extracted_parameters``, or None.
    validate: Callable | None = field(default=None, compare=False)
    #: ``Signature.plain_utterances``: example requests for the planning command.
    plain_utterances: tuple[str, ...] = field(default=(), compare=False)
    #: The planning command file's ``plan(input) -> list[Phase]`` (``fastworkflow.planning_phases``), or None.
    plan_fn: Callable | None = field(default=None, compare=False)
    #: The phases' ``output`` models seen when ``plan()`` ran on the placeholder inputs at load.
    phase_outputs: tuple[type, ...] = field(default=(), compare=False)
    #: The planning command file's ``PlanGenerator`` class (``command_from_module`` only).
    plan_cls: type | None = field(default=None, compare=False)
    #: ``Signature.Output`` when the planning command declares one.
    output_model: type | None = field(default=None, compare=False)
    #: Whether ``PlanGenerator`` defines or inherits ``__result__``.
    uses_result: bool = field(default=False, compare=False)

    def param(self, name: str) -> Param | None:
        return next((p for p in self.params if p.name == name), None)

    def signature_text(self) -> str:
        """``name(p: type = default, ...)`` with the description, parameter docs and steps."""
        parts = []
        for p in self.params:
            s = f"{p.name}: {p.type_text()}"
            if not p.required:
                s += f" = {json.dumps(_json_value(p))}"
            parts.append(s)
        lines = [f"{self.name}({', '.join(parts)})", f"  {self.description}"]
        lines += [f"  - {p.name}: {p.desc}" for p in self.params]
        lines.append("  expands to:")
        lines += [f"    {i}. {s}" for i, s in enumerate(self.template, 1)]
        return "\n".join(lines)


# ---------------------------------------------------------------- parameters

def _check_keys(block: dict, allowed: set[str], what: str, where: str) -> None:
    unknown = sorted(set(block) - allowed)
    if unknown:
        raise PlanningCommandFormatError(f"{where}: {what} has unknown key(s) {', '.join(unknown)}")


def _parse_param(name: str, spec, where: str) -> Param:
    if not re.fullmatch(_NAME, name) or name in RESERVED_NAMES:
        raise PlanningCommandFormatError(f"{where}: bad parameter name {name!r}")
    if not isinstance(spec, dict):
        raise PlanningCommandFormatError(f"{where}: parameter {name!r} must be a block with a type")
    _check_keys(spec, {"type", "default", "desc", "verify", "in_request", "ignore"},
                f"parameter {name!r}", where)
    type_text = spec.get("type")
    if not isinstance(type_text, str):
        raise PlanningCommandFormatError(f"{where}: parameter {name!r} needs a type")
    values: tuple[str, ...] = ()
    m = re.fullmatch(r"enum\[(.*)\]", type_text.strip())
    if m:
        values = tuple(v.strip() for v in m.group(1).split(","))
        if not values or any(not re.fullmatch(r"[\w-]+", v) for v in values) or len(set(values)) != len(values):
            raise PlanningCommandFormatError(f"{where}: parameter {name!r} has a bad enum {type_text!r}")
        kind = "enum"
    elif type_text.strip() in ("str", "bool", "int", "float", "list[str]"):
        kind = type_text.strip()
    else:
        raise PlanningCommandFormatError(f"{where}: parameter {name!r} has unknown type {type_text!r} "
                               "(str, bool, int, float, list[str] or enum[...])")
    desc = spec.get("desc", "")
    if not isinstance(desc, str):
        raise PlanningCommandFormatError(f"{where}: parameter {name!r} desc must be text")
    raw_default = spec.get("default")
    if isinstance(raw_default, dict):
        raise PlanningCommandFormatError(f"{where}: parameter {name!r} default must be a value")
    required = raw_default is None and kind != "bool"
    default: object = None
    if kind == "bool":
        if raw_default not in (None, "true", "false"):
            raise PlanningCommandFormatError(f"{where}: bool parameter {name!r} default must be true or false")
        default = raw_default == "true"
    elif raw_default is not None:
        if kind == "int":
            if not re.fullmatch(r"-?\d+", raw_default):
                raise PlanningCommandFormatError(f"{where}: parameter {name!r} default {raw_default!r} is not an int")
            default = int(raw_default)
        elif kind == "float":
            default = _float_text(raw_default)
            if default is None:
                raise PlanningCommandFormatError(f"{where}: parameter {name!r} default {raw_default!r} is not a number")
        elif kind == "list[str]":
            default = _list_text(raw_default)
            if default is None:
                raise PlanningCommandFormatError(f"{where}: parameter {name!r} default {raw_default!r} is not a list of text")
        elif kind == "enum":
            if raw_default not in values:
                raise PlanningCommandFormatError(f"{where}: parameter {name!r} default {raw_default!r} is not one of its values")
            default = raw_default
        else:
            default = raw_default
    # in_request: True (a str, an int or a list[str]) or "subject" (a str; implies in_request)
    in_request_text = spec.get("in_request", "false")
    subject = in_request_text == "subject"
    if subject:
        if kind != "str":
            raise option_error(where, 'in_request: "subject"', name, kind, "a str")
    elif in_request_text not in ("true", "false"):
        raise PlanningCommandFormatError(f"{where}: parameter {name!r} in_request must be true, false or \"subject\"")
    in_request = subject or in_request_text == "true"
    ignore: tuple[str, ...] = ()
    if "ignore" in spec:
        if kind != "str":
            raise option_error(where, "in_request ignore", name, kind, "a str")
        ignore = tuple(spec["ignore"])
    if in_request and not subject and kind not in ("str", "int", "list[str]"):
        raise option_error(where, "in_request", name, kind, "a str, an int or a list[str]")
    # verify: True (a bool, an enum, or a subject str) or a {value: meaning} dict on an enum
    verify_spec = spec.get("verify", "false")
    means: tuple[tuple[str, str], ...] = ()
    alone = False
    if isinstance(verify_spec, dict):
        block = verify_spec
        if kind != "enum":
            raise option_error(where, "verify (a dict of meanings)", name, kind, "an enum")
        if not block:
            raise PlanningCommandFormatError(f"{where}: verify of {name!r} must be a dict of its enum values' meanings")
        for key, text in block.items():
            if key not in values or not isinstance(text, str):
                raise PlanningCommandFormatError(f"{where}: verify of {name!r}: {key!r} is not a value of it, or not text")
        means = tuple((v, block[v]) for v in values if v in block)
        verify = True
    elif verify_spec == "alone":
        if kind != "bool":
            raise option_error(where, 'verify "alone"', name, kind, "a bool")
        verify = alone = True
    else:
        verify = _flag({"verify": verify_spec}, "verify", name, where)
        if verify and kind not in ("bool", "enum") and not subject:
            raise option_error(where, "verify", name, kind,
                               'a bool, an enum or a str with in_request "subject"')
    return Param(name=name, type=kind, values=values, default=default, required=required,
                 desc=desc, verify=verify, alone=alone, means=means, in_request=in_request,
                 subject=subject, ignore=ignore)


def option_error(where: str, option: str, name: str, kind: str, takes: str) -> PlanningCommandFormatError:
    """The load error of a check option on a field of a type it does not apply to (design 2.46)."""
    return PlanningCommandFormatError(f"{where}: option {option} does not apply to field {name!r} of type "
                                      f"{kind} (it takes {takes})")


def _float_text(text) -> float | None:
    """*text* as a finite float, or None."""
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _list_text(value) -> tuple[str, ...] | None:
    """A list[str] default: a list, or the list written as JSON text; a tuple of its text, or None."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
        return None
    return tuple(value)


def _flag(spec: dict, key: str, name: str, where: str) -> bool:
    value = spec.get(key, "false")
    if value not in ("true", "false"):
        raise PlanningCommandFormatError(f"{where}: parameter {name!r} {key} must be true or false")
    return value == "true"


# ---------------------------------------------------------------- planning command files (_commands/<name>.py)

#: Keys a planning command file's ``Input`` field may set in ``json_schema_extra``.
PLANNING_COMMAND_FIELD_KEYS = frozenset({"verify", "in_request", "from", "available_from"})

#: The retired check options (the two-option form, design 2.46): a load error naming the new spelling.
RETIRED_OPTIONS = {
    "subject": 'option subject is retired: write in_request: "subject"',
    "means": "option means is retired: write verify: {value: meaning, ...}",
    "strip": 'option strip is retired: write in_request: {"ignore": [...]}',
}


def _retired_option_error(where: str, name: str, keys) -> None:
    """The load error of a retired check option among a field's ``json_schema_extra`` *keys*."""
    for key in RETIRED_OPTIONS:
        if key in keys:
            raise PlanningCommandFormatError(f"{where}: field {name!r}: {RETIRED_OPTIONS[key]}")


def _allowed_planning_command_field_json_schema_extra_keys() -> set[str]:
    return set(PLANNING_COMMAND_FIELD_KEYS) | set(_PLANNING_COMMAND_JSON_SCHEMA_EXTRA_INTERNAL_KEYS)


def _spec_from_field(name: str, info, where: str) -> dict:
    """The spec of one ``Input`` field (its type, default, description and planning command options, as
    text), for ``_parse_param``."""
    from pydantic_core import PydanticUndefined

    annotation = info.annotation
    optional = get_origin(annotation) in (Union, UnionType) and type(None) in get_args(annotation)
    if optional:
        if annotation != Optional[str] or info.default is not None:
            raise PlanningCommandFormatError(f"{where}: field {name!r}: Optional is for an argument the request may "
                                   "leave out, Optional[str] = None; give another type a default")
        annotation = str
    if annotation in (str, bool, int, float):
        type_text = annotation.__name__
    elif annotation == list[str]:
        type_text = "list[str]"
    elif get_origin(annotation) is Literal:
        values = get_args(annotation)
        if not values or not all(isinstance(v, str) for v in values):
            raise PlanningCommandFormatError(f"{where}: field {name!r}: a Literal of text values only")
        type_text = f"enum[{', '.join(values)}]"
    else:
        raise PlanningCommandFormatError(f"{where}: field {name!r} has type {annotation!r}; a planning command field is "
                               "str, bool, int, float, list[str] or Literal[...] of text values")
    spec: dict = {"type": type_text}
    if info.description:
        spec["desc"] = info.description
    default: object = PydanticUndefined
    if info.default_factory is not None:
        if type_text != "list[str]" or info.default_factory is not list:
            raise PlanningCommandFormatError(f"{where}: field {name!r}: give a plain default, not a default_factory")
        default = []
    elif info.default is not PydanticUndefined:
        default = info.default
    if optional:
        spec["default"] = ""
    elif default is not PydanticUndefined:
        if default is None or isinstance(default, dict) or (isinstance(default, (list, tuple)) and type_text != "list[str]"):
            raise PlanningCommandFormatError(f"{where}: field {name!r} default must be a value of its type "
                                   "(leave the default out for a required parameter)")
        if type_text == "list[str]":
            if _list_text(default) is None:
                raise PlanningCommandFormatError(f"{where}: field {name!r} default must be a list of text")
            spec["default"] = json.dumps(list(default))
        else:
            spec["default"] = ("true" if default else "false") if isinstance(default, bool) else str(default)
    extra = info.json_schema_extra
    if extra is None:
        return spec
    if not isinstance(extra, dict):
        raise PlanningCommandFormatError(f"{where}: field {name!r} json_schema_extra must be a dict")
    if "requires" in extra:
        raise PlanningCommandFormatError(f"{where}: field {name!r}: requires is not a planning command option; reset the "
                               "argument (or refuse the call) in Signature.validate_extracted_parameters")
    _retired_option_error(where, name, extra)
    _check_keys(extra, _allowed_planning_command_field_json_schema_extra_keys(),
                f"field {name!r} json_schema_extra", where)
    spec.update(check_option_spec(name, extra, where))
    return spec


def check_option_spec(name: str, options: dict, where: str) -> dict:
    """The ``_parse_param`` spec of a field's check options, read from its ``json_schema_extra``:
    ``in_request`` true, false, "subject", or a dict ``{"subject": True, "ignore": [words]}`` (each key
    optional; the dict implies ``in_request``); ``verify`` true, false or a dict of the enum's meanings
    (2.46). Text values, as ``_parse_param`` reads them."""
    spec: dict = {}
    if "in_request" in options:
        value = options["in_request"]
        if value == "subject":
            spec["in_request"] = "subject"
        elif isinstance(value, bool):
            spec["in_request"] = "true" if value else "false"
        elif isinstance(value, dict):
            unknown = sorted(set(value) - {"subject", "ignore"})
            if unknown:
                raise PlanningCommandFormatError(f"{where}: parameter {name!r} in_request has unknown key(s) "
                                                 f"{', '.join(unknown)}; its keys are subject and ignore")
            if not value:
                raise PlanningCommandFormatError(f"{where}: parameter {name!r} in_request {{}} is empty; give "
                                                 "subject, ignore, or in_request: true")
            subject = value.get("subject", False)
            if not isinstance(subject, bool):
                raise PlanningCommandFormatError(f"{where}: parameter {name!r} in_request subject must be true or false")
            spec["in_request"] = "subject" if subject else "true"
            if "ignore" in value:
                words = value["ignore"]
                if isinstance(words, str) or not isinstance(words, (list, tuple)) or \
                        not all(isinstance(w, str) for w in words):
                    raise PlanningCommandFormatError(f"{where}: parameter {name!r} in_request ignore must be a "
                                                     "list of words")
                spec["ignore"] = tuple(words)
        else:
            raise PlanningCommandFormatError(f"{where}: parameter {name!r} in_request must be true, false, "
                                             "\"subject\" or a dict")
    if "verify" in options:
        value = options["verify"]
        if isinstance(value, dict):
            spec["verify"] = value
        elif isinstance(value, bool):
            spec["verify"] = "true" if value else "false"
        elif value == "alone":
            spec["verify"] = "alone"
        else:
            raise PlanningCommandFormatError(f"{where}: parameter {name!r} verify must be true, false, \"alone\" "
                                             "or a dict of its enum values' meanings")
    return spec


@dataclass(frozen=True)
class PlanningCommandFile:
    """What a planning command file declares: its description, its parsed command, and its template
    text (``steps``: the steps the planner adapts, and for a fixed-steps planning command the steps it runs)."""

    description: str
    steps: tuple[str, ...]
    command: PlanningCommandCommand
    #: ``Signature.plain_utterances``: example requests for the planning command (with or without params).
    plain_utterances: tuple[str, ...] = ()
    #: The planning command file's path.
    path: str = ""


def fixed_steps(command: PlanningCommandCommand) -> bool:
    """Whether *command* is a planning command with fixed steps: no parameters and no ``plan()``. Its one
    phase reads nothing, so every call of it is the same plan (the phase runner runs it as
    one phase)."""
    return not command.params and command.plan_fn is None


def _absent_as_none(command: PlanningCommandCommand, args: dict) -> dict:
    """*args* as the planning command file's code reads them: an ``Optional[str] = None`` field's
    absent argument (``""``) is None."""
    fields = command.input_model.model_fields
    return {n: None if v == "" and n in fields and fields[n].default is None else v
            for n, v in args.items()}


def input_instance(command: PlanningCommandCommand, args: dict):
    """The planning command's ``Input`` for *args*, unvalidated, as ``steps()``, ``plan()`` and the hook
    get it (``_absent_as_none``)."""
    return command.input_model.model_construct(**_absent_as_none(command, args))


def _call_steps(fn: Callable, instance, where: str) -> list[str]:
    import inspect
    from fastworkflow.planning import Step

    takes_input = bool(inspect.signature(fn).parameters)
    lines = fn(instance) if takes_input else fn()
    if isinstance(lines, str) or not isinstance(lines, (list, tuple)) or \
            not all(isinstance(line, str) for line in lines):
        raise PlanningCommandCallError(f"{where}: steps() must return a list of str")
    return [
        (Step(line.strip(), line.helper) if isinstance(line, Step) else line.strip())
        for line in lines if line.strip()
    ]


#: A planning command's input combinations are each tried up to this many; past it, a bounded set of
#: them (``input_grid``). Shared by ``placeholder_template`` and the planning command preview.
MAX_GRID = 4096
#: Planning command files warned about once in this process (past ``MAX_GRID``).
_warned_grids: set[str] = set()


def input_grid(sizes: list[int], first: tuple | None = None) -> list[tuple]:
    """The combinations (index tuples) of inputs with *sizes* values each that are tried,
    *first* (default all 0) first: every one, in ``itertools.product`` order, up to
    ``MAX_GRID``; past it, *first*, each single change of it and each pair of changes of it
    (every two inputs' values together at least once, the other inputs as in *first*)."""
    import itertools

    first = tuple(0 for _ in sizes) if first is None else tuple(first)
    total = 1
    for size in sizes:
        total *= size
    if total <= MAX_GRID:
        grid = list(itertools.product(*[range(s) for s in sizes]))
        grid.remove(first)
        return [first] + grid
    changes = [[i for i in range(s) if i != first[k]] for k, s in enumerate(sizes)]
    out = [first]
    for k in range(len(sizes)):
        out += [first[:k] + (i,) + first[k + 1:] for i in changes[k]]
    for j, k in itertools.combinations(range(len(sizes)), 2):
        for a, b in itertools.product(changes[j], changes[k]):
            combo = list(first)
            combo[j], combo[k] = a, b
            out.append(tuple(combo))
    return out


def _template_values(p: Param) -> tuple:
    """The inputs ``placeholder_template`` gives one parameter; the first is its base input.

    A str reads ``<name>`` (and "" too when that is its default), a bool is true and false, an
    int takes its default (or 1 and 0 when that is 0), a float its default (or 1.0 and 0.0 when
    that is 0), a list[str] its default (or ``["<name>"]``: one value, never varied), an enum each
    value (its default first).
    """
    if p.type == "str":
        return (f"<{p.name}>",) if p.required or p.default != "" else (f"<{p.name}>", "")
    if p.type == "bool":
        return (True, False)
    if p.type == "int":
        return (p.default,) if p.default else (1, 0)
    if p.type == "float":
        return (p.default,) if p.default else (1.0, 0.0)
    if p.type == "list[str]":
        return (_json_value(p) if p.default else [f"<{p.name}>"],)
    if p.default is None:
        return tuple(p.values)
    return (p.default,) + tuple(v for v in p.values if v != p.default)


def _env_key(env: dict) -> tuple:
    """*env* as a hashable key (a list[str] argument is a list)."""
    return tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in env.items())


def _template_grid(command: PlanningCommandCommand) -> list[dict]:
    """The argument sets ``placeholder_template`` renders: the base first, then the other
    combinations ``input_grid`` tries (past ``MAX_GRID``, a bounded set, logged once per
    planning command file), leaving out a set the planning command's ``validate_extracted_parameters`` changes or
    refuses (``_validate_arguments`` never passes one on to ``steps()``; the set it changes to is
    another of the grid, or none)."""
    values = [_template_values(p) for p in command.params]
    sizes = [len(v) for v in values]
    total = 1
    for size in sizes:
        total *= size
    if total > MAX_GRID and command.path not in _warned_grids:
        from fastworkflow.utils.logging import logger

        _warned_grids.add(command.path)
        logger.warning(f"planning command {command.name!r}: {total} combinations of its inputs, past {MAX_GRID}: "
                       "its template is rendered from its defaults and each one or two changes of them")
    # past MAX_GRID the grid is centred on the defaults (a bool false, a str empty when that is
    # its default), the inputs most calls have
    first = None if total <= MAX_GRID else tuple(
        v.index(p.default) if not p.required and p.default in v and p.type != "int" else 0
        for p, v in zip(command.params, values))
    grid = [{p.name: v[i] for p, v, i in zip(command.params, values, combo)} for combo in input_grid(sizes, first)]
    out, seen = [], set()
    for env in grid:
        key = _env_key(env)
        if key in seen:
            continue
        try:
            if command.validate is not None and _run_validate(command, env, "", None) != env:
                continue
            check_input(command, env, "", None)
        except PlanningCommandCallError:
            continue
        seen.add(key)
        out.append(env)
    return out


def _merge_runs(runs: list[list[str]]) -> list[list]:
    """``[[line, {grid index, ...}], ...]``: the steps of every grid point in one order. Each
    run is aligned with the lines so far (a line a run writes twice is kept twice); a line
    only some runs write is placed where they write it."""
    import difflib

    merged: list[list] = []
    for i, lines in enumerate(runs):
        ops = difflib.SequenceMatcher(None, [m[0] for m in merged], lines, autojunk=False).get_opcodes()
        new: list[list] = []
        for tag, a0, a1, b0, b1 in ops:
            if tag == "equal":
                for k in range(a0, a1):
                    merged[k][1].add(i)
                new.extend(merged[a0:a1])
            else:
                new.extend(merged[a0:a1])
                new.extend([line, {i}] for line in lines[b0:b1])
        merged = new
    return merged


def _literal_words(p: Param, allowed: frozenset) -> str:
    if p.type == "bool":
        return p.name if True in allowed else f"not {p.name}"
    if p.type == "str":
        return f"{p.name} is empty" if "" in allowed else f"{p.name} is given"
    if p.type in ("int", "float"):
        return f"not {p.name}" if 0 in allowed else p.name
    shown = [v for v in p.values if v in allowed]
    if len(p.values) > 2 and len(shown) == len(p.values) - 1:
        return f"{p.name} is not {next(v for v in p.values if v not in allowed)}"
    return f"{p.name} is " + " or ".join(shown)


def _condition_text(command: PlanningCommandCommand, present: set, grid: list[dict]) -> str | None:
    """The words of a simple condition true exactly on the grid points ``present``: "" when
    that is all of them, else an ``and`` of up to three parameter tests or an ``or`` of up
    to three ``and``s of up to two; None when none fits."""
    return _Conditions(command, grid).text(present)


class _Conditions:
    """``_condition_text`` for the point sets of one grid. A set of points is a bit mask (bit
    i for grid point i), each parameter test (a set of its values) the mask of the points it
    holds on, and a text is kept by its mask. Tests are tried in ``_condition_text``'s order
    (a parameter's sets by size, then in its values' order), so the first that fits is the
    same; an ``and`` that already misses a wanted point, or an ``or`` whose remaining terms
    cannot cover them, is skipped with everything that extends it.

    Only tests that can win are made, from the values a parameter takes on the wanted points
    (``seen``). In an ``and`` every test holds on all of them, so it allows at least ``seen``,
    and the test allowing just ``seen`` comes first. In an ``or`` a winning test allows only
    values of ``seen`` (one more makes a later test), so each set of them is tried while
    ``seen`` has at most ``MAX_OR_VALUES`` values; past it, each one value alone."""

    MAX_OR_VALUES = 8

    def __init__(self, command: PlanningCommandCommand, grid: list[dict]):
        self.size = len(grid)
        self.full = (1 << self.size) - 1
        self.cache: dict[int, str | None] = {}
        index = {_env_key(env): i for i, env in enumerate(grid)}
        # per parameter: (i, j) for each grid point i and each point j that is i with only
        # that parameter's value changed
        self.pairs = [(p, [(i, j) for i, env in enumerate(grid) for v in _template_values(p)
                           if v != env[p.name] for j in [index.get(_env_key({**env, p.name: v}))]
                           if j is not None])
                      for p in command.params]
        # per parameter: (value, mask of the points with that value), in _template_values order
        self.values = {p.name: [(v, sum(1 << i for i, env in enumerate(grid) if env[p.name] == v))
                                for v in _template_values(p)]
                       for p in command.params}

    def text(self, present: set) -> str | None:
        want = 0
        for i in present:
            if 0 <= i < self.size:
                want |= 1 << i
        if want not in self.cache:
            self.cache[want] = self._text(want)
        return self.cache[want]

    def _text(self, want: int) -> str | None:
        import itertools

        if want == self.full:
            return ""
        relevant = [p for p, pairs in self.pairs   # whose value alone can change whether the line is written
                    if any((want >> i & 1) != (want >> j & 1) for i, j in pairs)]
        seen = {p.name: [(v, mask) for v, mask in self.values[p.name] if mask & want] for p in relevant}

        def test(p: Param, chosen) -> tuple:   # (p, allowed values, mask of the points it holds on)
            mask = 0
            for _, m in chosen:
                mask |= m
            return p, frozenset(v for v, _ in chosen), mask

        found = self._and([test(p, seen[p.name]) for p in relevant
                           if len(seen[p.name]) < len(self.values[p.name])], want)
        if found is not None:
            return found
        return self._or([test(p, chosen) for p in relevant
                         for k in range(1, min(len(seen[p.name]), len(self.values[p.name]) - 1) + 1)
                         if k == 1 or len(seen[p.name]) <= self.MAX_OR_VALUES
                         for chosen in itertools.combinations(seen[p.name], k)], want)

    @staticmethod
    def _words(literals: list[tuple], conj) -> str:
        return " and ".join(_literal_words(literals[k][0], literals[k][1]) for k in conj)

    def _and(self, literals: list[tuple], want: int) -> str | None:
        """The first ``and`` of up to three *literals* of different parameters true exactly on *want*."""
        names = [p.name for p, _, _ in literals]
        masks = [mask for _, _, mask in literals]
        n = len(literals)
        for a in range(n):
            if masks[a] == want:
                return self._words(literals, (a,))
        for a in range(n):
            for b in range(a + 1, n):
                if names[a] != names[b] and masks[a] & masks[b] == want:
                    return self._words(literals, (a, b))
        for a in range(n):
            if masks[a] & want != want:
                continue
            for b in range(a + 1, n):
                t = masks[a] & masks[b]
                if t & want != want or names[a] == names[b]:
                    continue
                for c in range(b + 1, n):
                    if t & masks[c] == want and names[c] != names[a] and names[c] != names[b]:
                        return self._words(literals, (a, b, c))
        return None

    def _or(self, literals: list[tuple], want: int) -> str | None:
        """The first ``or`` of two or three ``and``s of up to two *literals* true exactly on *want*."""
        names = [p.name for p, _, _ in literals]
        masks = [mask for _, _, mask in literals]
        n = len(literals)
        small: list[tuple] = []   # (literal indexes, mask): the ands of one or two that hold only within want
        for a in range(n):
            if not masks[a] & ~want:
                small.append(((a,), masks[a]))
        for a in range(n):
            for b in range(a + 1, n):
                t = masks[a] & masks[b]
                if names[a] != names[b] and not t & ~want:
                    small.append(((a, b), t))

        def joined(terms) -> str:
            return " or ".join(f"({self._words(literals, c)})" if len(c) > 1 else self._words(literals, c)
                               for c in terms)

        m = len(small)
        rest = [0] * (m + 1)   # rest[k]: the points the terms from k on hold on
        for k in range(m - 1, -1, -1):
            rest[k] = rest[k + 1] | small[k][1]
        for a in range(m):
            if small[a][1] | rest[a + 1] != want:
                continue
            for b in range(a + 1, m):
                if small[a][1] | small[b][1] == want:
                    return joined((small[a][0], small[b][0]))
        for a in range(m):
            if small[a][1] | rest[a + 1] != want:
                continue
            for b in range(a + 1, m):
                t = small[a][1] | small[b][1]
                if t | rest[b + 1] != want:
                    continue
                for c in range(b + 1, m):
                    if t | small[c][1] == want:
                        return joined((small[a][0], small[b][0], small[c][0]))
        return None


def _words(text: str) -> list[str]:
    """*text* cut into the parts ``_alternatives`` compares: after each space, ``=`` and
    ``(``, and before each ``)`` (``query=``, ``(``, ``identity_uid ``, ``)``), a
    placeholder (``<reach text>``) kept whole, so a varying part never starts inside a
    placeholder."""
    out: list[str] = []
    depth = 0
    for part in re.split(r"(?<=[ =(])|(?=\))", text):
        if not part:
            continue
        if depth and out:
            out[-1] += part
        else:
            out.append(part)
        depth = max(0, depth + part.count("<") - part.count(">"))
    return out


def _alternatives(texts: list[str], by: str, whole: bool = True) -> str | None:
    """One line for the lines ``texts`` (one per value of enum ``by``, in its order) that differ
    only in a middle part: ``<find_group or find_unit, by kind> query=<name>``; None when every
    one differs in more than a few words and in more than they share (different steps), and,
    without ``whole``, when they share no part (lines not at one place in their branches).
    Lines are compared by ``_words``; a middle keeps its brackets balanced and a ``key=``
    its value (the shared ``(``, ``)`` or value it needs is given back to it), so no merge
    cuts a bracket or parts a key from its value."""
    pieces = [_words(t) for t in texts]
    pre = 0
    while all(len(w) > pre for w in pieces) and len({w[pre] for w in pieces}) == 1:
        pre += 1
    suf = 0
    while all(len(w) - pre > suf for w in pieces) and len({w[-1 - suf] for w in pieces}) == 1:
        suf += 1
    raw = ["".join(w[pre:len(w) - suf]) for w in pieces]
    while any(m.count("(") != m.count(")") or m.endswith("=") for m in raw):
        if suf and any(m.count("(") > m.count(")") or m.endswith("=") for m in raw):
            suf -= 1   # a "(" needs its ")"; a "key=" keeps its value
        elif pre and any(m.count(")") > m.count("(") for m in raw):
            pre -= 1
        else:
            break
        raw = ["".join(w[pre:len(w) - suf]) for w in pieces]
    middles = list(dict.fromkeys(m.strip() for m in raw))
    if len(middles) == 1:
        return texts[0]
    if not whole and not pre and not suf:
        return None
    sizes = [len(m.split()) for m in middles]
    if min(sizes) > 3 and max(sizes) > pre + suf:
        return None
    gap = " " if suf and any(m.endswith(" ") for m in raw) else ""
    head, tail = "".join(pieces[0][:pre]), "".join(pieces[0][len(pieces[0]) - suf:])
    return head + "<" + " or ".join(m for m in middles if m) + f", by {by}>" + gap + tail


def _merge_enum_lines(p: Param, merged: list[list], grid: list[dict]) -> list[list]:
    """``merged`` with the branches of enum ``p`` joined where they line up: consecutive blocks
    of lines, each written for its own values of ``p`` alone (the other arguments alike), of
    one length; their n-th lines become one line with ``<a or b, by p>``."""
    own = [env[p.name] for env in grid]
    others = [tuple((k, v) for k, v in env.items() if k != p.name) for env in grid]

    info = []   # per line: (values of p it is written for, the other arguments it is written for)
    for _line, points in merged:
        values = frozenset(own[i] for i in points)
        rest = frozenset(others[i] for i in points)
        alone = {i for i in range(len(grid)) if own[i] in values and others[i] in rest} == points
        info.append((values, rest) if alone and len(values) < len(p.values) else None)

    def joinable(blocks: list[list[int]]) -> bool:
        sets = [info[b[0]][0] for b in blocks]
        return (sum(len(s) for s in sets) == len(frozenset().union(*sets))
                and len({len(b) for b in blocks}) == 1
                and all(len({info[b[n]][1] for b in blocks}) == 1 for n in range(len(blocks[0]))))

    out: list[list] = []
    i = 0
    while i < len(merged):
        if info[i] is None:
            out.append(merged[i])
            i += 1
            continue
        j = i
        while j < len(merged) and info[j] is not None:
            j += 1
        blocks: list[list[int]] = []
        for k in range(i, j):
            if blocks and info[blocks[-1][-1]][0] == info[k][0]:
                blocks[-1].append(k)
            else:
                blocks.append([k])
        # A block's place in its values' branches (None when its values differ in it): lines
        # that share no part join only at the same place ("check_a", the second line of
        # k0..k2's branches, is not the k3 branch's first line "find_k3"; fix-ajir.38).
        offset: dict[int, int | None] = {}   # by a block's first line
        for b in blocks:
            places = {sum(1 for k in range(i, b[0]) if v in info[k][0]) for v in info[b[0]][0]}
            offset[b[0]] = places.pop() if len(places) == 1 else None
        if joinable(blocks):   # the branches in the enum's order
            blocks.sort(key=lambda b: min(p.values.index(v) for v in info[b[0]][0]))
        start = 0
        while start < len(blocks):
            end = start + 1
            while end < len(blocks) and joinable(blocks[start:end + 1]):
                end += 1
            segment = blocks[start:end]
            if len(segment) == 1:
                out.extend(merged[k] for k in segment[0])
            else:
                segment.sort(key=lambda b: min(p.values.index(v) for v in info[b[0]][0]))
                aligned = len({offset[b[0]] for b in segment}) == 1 and offset[segment[0][0]] is not None
                for n in range(len(segment[0])):
                    members = [merged[b[n]] for b in segment]
                    text = _alternatives([m[0] for m in members], p.name, whole=aligned)
                    if text is not None:
                        out.append([text, set().union(*(m[1] for m in members))])
                    elif aligned:   # different steps, one line each
                        out.extend(members)
                    else:           # the lines at each place in their branches, joined apart
                        by_place: dict[int | None, list] = {}
                        for b, m in zip(segment, members):
                            by_place.setdefault(offset[b[0]], []).append(m)
                        for place, group in by_place.items():
                            text = _alternatives([m[0] for m in group], p.name) \
                                if len(group) > 1 and place is not None else None
                            if text is None:
                                out.extend(group)
                            else:
                                out.append([text, set().union(*(m[1] for m in group))])
            start = end
        i = j
    return out


def _merge_empty_str_lines(p: Param, merged: list[list], grid: list[dict]) -> list[list]:
    """``merged`` with each line written for an empty ``p`` joined to its neighbour written for
    ``<p>`` when it is that line with ``<p>`` left out (``assess_risk{owner=}`` of
    ``assess_risk{owner=<owner>}``): one step whose value may be empty."""
    shown = f"<{p.name}>"

    def written_for(points: set) -> set:
        return {grid[i][p.name] for i in points}

    out: list[list] = []
    for line, points in merged:
        if out and written_for(points) == {""} and written_for(out[-1][1]) == {shown} \
                and shown in out[-1][0] and out[-1][0].replace(shown, "") == line:
            out[-1] = [out[-1][0], out[-1][1] | points]
            continue
        out.append([line, points])
    return out


def placeholder_template(command: PlanningCommandCommand) -> list[str]:
    """A planning command file's template text when it declares none: ``steps()`` rendered on placeholder
    inputs.

    ``steps()`` runs on every combination of the inputs ``_template_values`` gives (a str
    reads ``<name>``, and "" when that is its default; a bool true and false; an int its
    default, or 1 and 0; a float its default, or 1.0 and 0.0; a list[str] one value, its default
    or ``["<name>"]``; an enum each value; past ``MAX_GRID`` combinations, those
    ``input_grid`` covers), and the runs are merged in order. A line not
    every run writes ends with ``(only when ...)``, the condition on the arguments under which
    it is written (``kind is group``, ``person is given``, ``not flag``); lines that differ only
    by an enum's value become one (``<find_group or find_unit, by kind> query=<name>``). A
    repeated part shows its steps for the int's default; declare ``Signature.template`` when
    the planner needs more than that.

    Raises PlanningCommandFormatError when ``steps()`` fails on the base inputs (the first of each, or
    the first combination ``validate_extracted_parameters`` leaves in).
    """
    from fastworkflow.planning import TemplateError

    grid, runs = [], []
    for env in _template_grid(command):
        try:
            lines = _call_steps(command.steps_fn, input_instance(command, env), command.path)
        except TemplateError as exc:
            raise PlanningCommandFormatError(f"{command.path}: {exc}") from None
        except Exception as exc:  # noqa: BLE001 - a set steps() refuses is left out of the template
            if not grid and not runs:
                raise PlanningCommandFormatError(f"{command.path}: steps() fails on placeholder inputs "
                                       f"({type(exc).__name__}: {exc}); declare Signature.template") from None
            continue
        grid.append(env)
        runs.append(lines)
    if not grid:
        raise PlanningCommandFormatError(f"{command.path}: validate_extracted_parameters changes or refuses every "
                               "placeholder combination; declare Signature.template")
    merged = _merge_runs(runs)
    for p in command.params:
        if p.type == "str" and len(_template_values(p)) > 1:
            merged = _merge_empty_str_lines(p, merged, grid)
    for p in command.params:
        if p.type == "enum":
            merged = _merge_enum_lines(p, merged, grid)
    conditions = _Conditions(command, grid)
    out = []
    for line, points in merged:
        condition = conditions.text(points)
        if condition is None:
            out.append(f"{line}{_SOME_ARGUMENTS}")
        elif condition and line.endswith(")") and _FACT_CONDITION in line:
            # the facts' condition (the planning command's PlanGenerator) and the inputs': one "only when"
            both = f"({condition})" if " or " in condition else condition
            out.append(f"{line[:-1]} and {both})")
        else:
            out.append(f"{line} (only when {condition})" if condition else line)
    return out


#: How a phase's template lines end when its method tested facts (``fastworkflow.planning``).
_FACT_CONDITION = " (only when "
#: How a template line ends when no condition on the arguments says when it is written.
_SOME_ARGUMENTS = " (only for some of the arguments)"


def _is_template_line(text: str) -> bool:
    """Whether *text* carries a condition only the planning command's TEMPLATE writes (``(only when
    find.uids)``, ``(only for some of the arguments)``): the planner copied a template line."""
    return _FACT_CONDITION in text or _SOME_ARGUMENTS in text


def _plan_generator_inherits_foreign(plan_cls: type, path: str) -> None:
    """Load error when *plan_cls* subclasses another command's ``PlanGenerator`` (3.4, 2.36)."""
    module = sys.modules.get(plan_cls.__module__)
    own = getattr(module, "PlanGenerator", None) if module is not None else None
    for base in plan_cls.__bases__:
        if not inspect.isclass(base) or base.__name__ != "PlanGenerator":
            continue
        if base is own:
            continue
        other_path = getattr(sys.modules.get(base.__module__, None), "__file__", "") or ""
        if os.path.basename(other_path).startswith("_"):
            continue
        other_name = _qualified_name_from_command_path(other_path)
        if other_name is None:
            continue
        raise PlanningCommandFormatError(
            f"{path}: PlanGenerator inherits from {other_name}'s PlanGenerator; to reuse its "
            f"whole job, call {other_name.split('/')[-1]} in a step; to share phases, move "
            "them to a base class in a _ module")


def _qualified_name_from_command_path(fpath: str) -> str | None:
    marker = f"{os.sep}_commands{os.sep}"
    if marker not in fpath:
        return None
    rel = fpath.split(marker, 1)[1]
    if rel.startswith("_") or f"{os.sep}_" in rel:
        return None
    return rel[:-3].replace(os.sep, "/")


def _result_method(plan_cls: type) -> tuple[Callable | None, type | None]:
    for klass in plan_cls.__mro__:
        if "__result__" in klass.__dict__:
            return klass.__dict__["__result__"], klass
    return None, None


def _phase_names_for_command(command: "PlanningCommandCommand") -> set[str]:
    names: set[str] = set()
    if command.plan_cls is not None:
        names.update(name for name, _ in planning_api.phase_methods(command.plan_cls))
    if command.plan_fn is None:
        return names
    for env in _template_grid(command):
        try:
            result = command.plan_fn(input_instance(command, env))
        except Exception:  # noqa: BLE001 - other grid points may still bind
            continue
        if is_flat_plan(result):
            continue
        try:
            phases = active_phases(result, command.path)
        except PhaseError:
            continue
        names.update(phase.name for phase in phases)
    return names


def _check_result_method(plan_cls: type, path: str, command: "PlanningCommandCommand") -> bool:
    result_fn, owner = _result_method(plan_cls)
    if result_fn is None:
        return False
    phase_names = _phase_names_for_command(command)
    for parameter in inspect.signature(result_fn).parameters.values():
        if parameter.name in ("self", "input"):
            continue
        if parameter.name not in phase_names:
            where = "__result__" if owner is plan_cls else f"{owner.__qualname__}.__result__"
            raise PlanningCommandFormatError(
                f"{path}: {where} takes {parameter.name!r}, which is not a phase of this command")
    return True


def command_from_module(name: str, module, path: str) -> PlanningCommandFile:
    """The planning command a planning command file (``_commands/<name>.py``, already imported) defines.

    Raises PlanningCommandFormatError for a malformed declaration (see the module doc).
    """
    import inspect

    from pydantic import BaseModel

    signature = getattr(module, "Signature", None)
    if not inspect.isclass(signature):
        raise PlanningCommandFormatError(f"{path}: a planning command file needs a class Signature")
    description = " ".join(inspect.cleandoc(signature.__doc__ or "").split())
    if not description:
        raise PlanningCommandFormatError(f"{path}: Signature needs a docstring (the planning command's description)")
    from fastworkflow import planning as planning_api
    from fastworkflow.planning_phases import PhaseError

    plan_cls = getattr(module, "PlanGenerator", None)
    if not inspect.isclass(plan_cls) or not planning_api.phase_methods(plan_cls):
        raise PlanningCommandFormatError(f"{path}: a planning command file needs a class PlanGenerator with at "
                               "least one phase method (fastworkflow.planning)")
    try:
        steps_fn, plan_fn = planning_api.planning_command_parts(plan_cls)
    except PhaseError as exc:
        raise PlanningCommandFormatError(f"{path}: {exc}") from None
    _plan_generator_inherits_foreign(plan_cls, path)
    return planning_command_from_parts(name, signature, path, steps_fn=steps_fn, plan_fn=plan_fn,
                            plan_cls=plan_cls)


def planning_command_from_parts(name: str, signature, path: str, *, steps_fn: Callable | None = None,
                     plan_fn: Callable | None = None,
                     plan_cls: type | None = None) -> PlanningCommandFile:
    """The planning command a ``Signature`` and its behaviour define: a flat ``steps_fn(input)`` or a
    ``plan_fn(input) -> list[Phase]`` (or both: a plan that may return no phases runs flat,
    from ``steps_fn``). ``command_from_module`` builds them from the file's ``PlanGenerator``
    class; the runtime reads only these. Raises PlanningCommandFormatError."""
    import inspect

    from pydantic import BaseModel

    description = " ".join(inspect.cleandoc(signature.__doc__ or "").split())
    if not description:
        raise PlanningCommandFormatError(f"{path}: Signature needs a docstring (the planning command's description)")
    own_steps = steps_fn is not None
    if steps_fn is None:
        if plan_fn is None:
            raise PlanningCommandFormatError(f"{path}: a planning command needs its steps or its phases")
        steps_fn = _concatenated_steps_fn(plan_fn)
    input_model = getattr(signature, "Input", None)
    if input_model is None:
        input_model = type(f"{signature.__name__}Input", (BaseModel,), {})
    if not (inspect.isclass(input_model) and issubclass(input_model, BaseModel)):
        raise PlanningCommandFormatError(f"{path}: Signature.Input must be a pydantic BaseModel")
    output_model = getattr(signature, "Output", None)
    declared_output = (inspect.isclass(output_model) and issubclass(output_model, BaseModel))
    # Signature.Output ``from_`` is checked at field-source resolution (3.3), not here, so a
    # planning command with that error still loads under the unified load policy.
    utterances = getattr(signature, "plain_utterances", ())
    if isinstance(utterances, str) or not all(isinstance(u, str) for u in utterances):
        raise PlanningCommandFormatError(f"{path}: plain_utterances must be a list of str")
    from fastworkflow.utils.signatures import reject_retired_db_lookup
    reject_retired_db_lookup(signature, input_model, path)
    validate = getattr(signature, "validate_extracted_parameters", None)
    if validate is not None and not callable(validate):
        raise PlanningCommandFormatError(f"{path}: validate_extracted_parameters must be a function")

    params = {n: _parse_param(n, _spec_from_field(n, info, path), path)
              for n, info in input_model.model_fields.items()}
    if not params and plan_fn is None:
        try:
            lines = _call_steps(steps_fn, input_model(), path)
        except PlanningCommandCallError as exc:
            raise PlanningCommandFormatError(str(exc)) from None
        if not lines:
            raise PlanningCommandFormatError(f"{path}: steps() returns no steps")
        _refuse_flat_say_that(path, name, lines)
        command = PlanningCommandCommand(name=name, description=description, params=(), path=path,
                               input_model=input_model, steps_fn=steps_fn, validate=validate,
                               plain_utterances=tuple(utterances), plan_cls=plan_cls,
                               template=tuple(lines))
        return PlanningCommandFile(description=description, steps=tuple(lines), command=command,
                         plain_utterances=tuple(utterances), path=path)

    command = PlanningCommandCommand(name=name, description=description, params=tuple(params.values()), path=path, input_model=input_model, steps_fn=steps_fn, validate=validate,
                           plain_utterances=tuple(utterances), plan_fn=plan_fn,
                           plan_cls=plan_cls,
                           output_model=output_model if declared_output else None)
    if plan_fn is not None:
        command = replace(command, phase_outputs=_checked_plan(command, own_steps=own_steps))
    else:
        for env in _template_grid(command):
            try:
                flat_lines = _call_steps(steps_fn, input_instance(command, env), path)
            except Exception:  # noqa: BLE001 - steps that fail here are the template check's to report
                continue
            _refuse_flat_say_that(path, name, flat_lines)
    uses_result = plan_cls is not None and _check_result_method(plan_cls, path, command)
    if declared_output and not uses_result:
        raise PlanningCommandFormatError(
            f"{path}: Signature.Output is declared, but nothing builds it; return it from "
            "__result__(self, input, ...)")
    command = replace(command, uses_result=uses_result)
    declared = getattr(signature, "template", None)
    if declared is not None:
        if isinstance(declared, str) or not all(isinstance(t, str) for t in declared):
            raise PlanningCommandFormatError(f"{path}: Signature.template must be a list of str")
        template = list(declared)   # as written: it is text for the planner
    else:
        template = placeholder_template(command)
    if not template:
        raise PlanningCommandFormatError(f"{path}: a planning command needs steps")
    return PlanningCommandFile(description=description, steps=tuple(template), command=replace(command, template=tuple(template)),
                     plain_utterances=tuple(utterances), path=path)


def _concatenated_steps_fn(plan_fn: Callable) -> Callable:
    """``steps(input)`` of a planning command file that defines only ``plan()``: the phases joined."""
    from fastworkflow.planning_phases import concatenated_steps

    def steps(input):
        return concatenated_steps(plan_fn(input))

    return steps


def _refuse_flat_say_that(path: str, name: str, lines) -> None:
    """A flat call whose answer is only 'say that' lines is refused as a last phase of them is (3.16):
    the answer is ``__result__``'s, and the load check can see the steps the call runs flat."""
    from fastworkflow import planning_phases

    if lines and planning_phases.answer_by_code("\n".join(lines)):
        raise PlanningCommandFormatError(planning_phases.say_that_last_message(path, name))


def _checked_plan(command: PlanningCommandCommand, own_steps: bool = True) -> tuple[type, ...]:
    """Run ``plan()`` on the placeholder inputs (``_template_grid``) and check what it returns.

    A flat result (``None`` or ``[]``, ``planning_phases.is_flat_plan``) is fine when the
    file defines its own ``steps()`` (*own_steps*), which that call then runs.
    Each other result must be a list of ``Phase`` whose parameters can be bound
    (``planning_phases.check_bindings``) and whose ``output`` models' ``"from"`` are
    well formed (``facts.fact_sources``); a parameter defaulting to None must bind (name an
    earlier phase with an output) in at least one of the results. Returns the output models seen. Whether the
    sources name real command fields is checked against the workflow later
    (``check_phase_sources``), as a warning. Raises PlanningCommandFormatError.
    """
    from fastworkflow.observation_offloading.facts import FactSourceError, fact_sources
    from fastworkflow.planning_phases import PhaseError, active_phases, check_bindings, is_flat_plan

    models: list[type] = []
    binds: dict[tuple[str, str, str], bool] = {}
    ran = 0
    for env in _template_grid(command):
        try:
            result = command.plan_fn(input_instance(command, env))
        except Exception as exc:  # noqa: BLE001 - a set plan() refuses, as for steps()
            if not ran:
                raise PlanningCommandFormatError(f"{command.path}: plan() fails on placeholder inputs "
                                       f"({type(exc).__name__}: {exc})") from None
            continue
        ran += 1
        if is_flat_plan(result):
            if not own_steps:
                raise PlanningCommandFormatError(
                    f"{command.path}: plan() returns no phases for some inputs (run flat), so "
                    "the planning command file needs its own steps(input)")
            try:
                flat_lines = _call_steps(command.steps_fn, input_instance(command, env), command.path)
            except Exception:  # noqa: BLE001 - steps that fail here are the template check's to report
                flat_lines = []
            _refuse_flat_say_that(command.path, command.name, flat_lines)
            continue
        try:
            phases = active_phases(result, f"{command.path}: plan()")
            for key, bound in check_bindings(phases, f"{command.path}: plan()").items():
                binds[key] = binds.get(key, False) or bound
        except PhaseError as exc:
            raise PlanningCommandFormatError(str(exc)) from None
        if not phases:
            raise PlanningCommandFormatError(f"{command.path}: plan() returns no phase")
        for phase in phases:
            if phase.output is not None and phase.output not in models:
                from fastworkflow.typed_phase_facts import (
                    check_typed_facts_model,
                    handles_typed_output,
                    union_members,
                )

                # Facts-model ``from_`` is checked at field-source resolution (3.3), not here.
                for message in check_typed_facts_model(
                        command.path, phase.output, {}, {}):
                    raise PlanningCommandFormatError(f"{command.path}: {message}")
                if not handles_typed_output(phase.output):
                    try:
                        fact_sources(phase.output)
                    except FactSourceError as exc:
                        raise PlanningCommandFormatError(f"{command.path}: phase {phase.name!r}: {exc}") from None
                models.append(phase.output)
    for (phase, what, parameter), bound in binds.items():
        if not bound:
            raise PlanningCommandFormatError(
                f"{command.path}: phase {phase!r}: {what} takes {parameter!r}, which names no "
                "earlier phase in any plan; check its spelling")
    return tuple(models)


def check_phase_sources(command: PlanningCommandCommand, signatures: dict) -> list[str]:
    """Load warnings for a phased planning command whose output models name unknown command fields.

    Each output model is checked with ``facts.check_fact_sources`` against the
    workflow's ``signatures`` (``facts.workflow_signatures``). A plain command name
    that several contexts define is resolved against the record's context only when
    the phase runs, so here it passes when some context resolves every field. A
    missing command or field is a warning, not a load failure: a workflow may add a
    command's ``Output`` after the planning command that reads it. Malformed ``from_`` on Input,
    Output, or facts models is reported by ``resolve_field_sources`` without skipping
    the planning command.
    """
    from fastworkflow.observation_offloading.facts import (
        AmbiguousFactSourceError, FactSourceError, check_fact_sources)

    contexts = sorted({name.split("/", 1)[0] for name in signatures if "/" in name})
    warnings = []
    for model in command.phase_outputs:
        try:
            check_fact_sources(model, signatures)
            continue
        except AmbiguousFactSourceError as error:
            problem = str(error)
        except FactSourceError as error:
            warnings.append(f"{command.path}: {error}")
            continue
        for context in contexts:
            try:
                check_fact_sources(model, signatures, context_name=context)
                break
            except FactSourceError:
                continue
        else:
            warnings.append(f"{command.path}: {problem}")
    return warnings


# ---------------------------------------------------------------- field sources (from_)

_PRODUCER_REF_RE = re.compile(
    r"^(?:([A-Za-z_][A-Za-z0-9_]*)/)?([A-Za-z_][A-Za-z0-9_]*)(?:\.([A-Za-z_][A-Za-z0-9_]*))?$"
)


class _ResolvedAvailableFrom(list):
    """List written into ``json_schema_extra["available_from"]`` by :func:`resolve_signature_field_sources`,
    distinct from a hand-written ``available_from`` list so re-running resolution stays idempotent
    while a hand-written ``available_from`` (even alongside ``from_``) is still flagged as an error."""


def _hand_written_available_from_value(extra: Mapping) -> Any | None:
    """The hand-written ``available_from`` value for *extra*, if any (not resolver-written)."""
    if _HAND_WRITTEN_AVAILABLE_FROM_KEY in extra:
        return extra[_HAND_WRITTEN_AVAILABLE_FROM_KEY]
    af = extra.get("available_from")
    if af is not None and not isinstance(af, _ResolvedAvailableFrom):
        return af
    return None


def _spelled_from_text(spelled: Any) -> str:
    """Format a ``from_`` source value for error messages."""
    if isinstance(spelled, list) and len(spelled) == 1:
        return f"{str(spelled[0])!r}"
    if isinstance(spelled, list):
        return f"{[str(s) for s in spelled]!r}"
    return f"{str(spelled)!r}"


@dataclass(frozen=True)
class FieldSourceProblem:
    """One problem or info line reported by :func:`resolve_field_sources`."""

    file: str
    message: str
    severity: Literal["error", "info"] = "error"
    command: str = ""

    def format(self) -> str:
        return f"{self.file}: {self.message}" if self.file else self.message

    def __str__(self) -> str:
        return self.format()


@dataclass(frozen=True)
class FieldSourceResolution:
    """Result of :func:`resolve_field_sources` for one workflow."""

    workflow_folderpath: str
    problems: tuple[FieldSourceProblem, ...]
    #: Command names (execution or planning) that have a ``from_`` / ``available_from``
    #: resolution error. They still load and stay routable or matchable; preview
    #: ``--check`` fails on the reported problems. Not "skipped" at load time.
    skipped_commands: frozenset[str]
    model_to_command: Mapping[type, str]
    signatures: Mapping[str, Any]
    command_files: Mapping[str, str]
    import_errors: Mapping[str, str]
    base_contexts: Mapping[str, tuple[str, ...]]
    #: The planning commands (``_commands/`` with ``PlanGenerator``) that loaded, by
    #: qualified name: what run_planning_command runs and a restored call rebuilds from.
    planning_commands: Mapping[str, PlanningCommandCommand] = field(default_factory=dict)
    #: ``callees[name]``: the qualified names the steps of each planning command call (3.7).
    callees: Mapping[str, frozenset[str]] = field(default_factory=dict)

    @property
    def errors(self) -> tuple[FieldSourceProblem, ...]:
        return tuple(p for p in self.problems if p.severity == "error")

    @property
    def infos(self) -> tuple[FieldSourceProblem, ...]:
        return tuple(p for p in self.problems if p.severity == "info")

    def errors_for(self, command_name: str) -> list[str]:
        return [p.format() for p in self.problems
                if p.severity == "error" and p.command == command_name]


def _annotation_text(annotation: Any) -> str:
    if inspect.isclass(annotation) and not get_args(annotation):
        return annotation.__name__
    text = re.sub(r"\btyping\.", "", repr(annotation))
    return text


def _strip_optional(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
        if len(non_none) < len(get_args(annotation)):
            return Union[tuple(non_none)]
    return annotation


def _producer_types_match(declaring_annotation: Any, target_annotation: Any) -> bool:
    """True when *target_annotation* equals *declaring_annotation*, equals it modulo
    ``Optional``, or is ``list[T]`` / ``Optional[list[T]]`` of it."""
    if declaring_annotation == target_annotation:
        return True
    decl_inner = _strip_optional(declaring_annotation)
    targ_inner = _strip_optional(target_annotation)
    if decl_inner == targ_inner:
        return True
    if get_origin(targ_inner) is list:
        args = get_args(targ_inner)
        if len(args) == 1:
            elem = args[0]
            if elem == declaring_annotation or _strip_optional(elem) == decl_inner:
                return True
    return False


def _discover_dir_command_files(root: str) -> dict[str, str]:
    """``{qualified_name: filepath}`` for ``.py`` command files under *root* (root and
    one context subfolder deep, skipping ``_``- and ``.``-prefixed entries)."""
    if not os.path.isdir(root):
        return {}
    out: dict[str, str] = {}
    for entry in sorted(os.listdir(root)):
        if entry.startswith(("_", ".")):
            continue
        path = os.path.join(root, entry)
        if os.path.isdir(path):
            for sub in sorted(os.listdir(path)):
                if sub.startswith(("_", ".")):
                    continue
                sub_path = os.path.join(path, sub)
                stem, ext = os.path.splitext(sub)
                if ext == ".py" and os.path.isfile(sub_path):
                    out[f"{entry}/{stem}"] = sub_path
            continue
        stem, ext = os.path.splitext(entry)
        if ext == ".py" and os.path.isfile(path):
            out[stem] = path
    return out


def _read_base_contexts(
    workflow_folderpath: str,
) -> tuple[dict[str, tuple[str, ...]], list[FieldSourceProblem]]:
    """``({context: (base, ...)}, problems)`` in depth-first declared order for each context
    of the workflow, using :meth:`CommandContextModel.load_base_contexts`.

    Reads and validates ``_commands/context_inheritance_model.json`` directly without
    importing any command or creating ``___command_info``. If validation fails, the error is
    recorded as a :class:`FieldSourceProblem` on the ``context_inheritance_model.json`` path.
    """
    commands_dir = os.path.join(workflow_folderpath, "_commands")
    model_path = os.path.join(commands_dir, "context_inheritance_model.json")
    if not os.path.isdir(commands_dir) and not os.path.isfile(model_path):
        return {}, []
    try:
        bases = CommandContextModel.load_base_contexts(workflow_folderpath)
    except WorkflowInheritanceModelError as exc:
        inh_path = os.path.join(workflow_folderpath, "workflow_inheritance_model.json")
        return {}, [
            FieldSourceProblem(
                file=inh_path,
                message=f"{type(exc).__name__}: {exc}",
                severity="error",
            )
        ]
    except Exception as exc:  # noqa: BLE001
        return {}, [
            FieldSourceProblem(
                file=model_path,
                message=f"{type(exc).__name__}: {exc}",
                severity="error",
            )
        ]
    return bases, []


def _static_lookup_scopes(context: str, base_contexts: Mapping[str, tuple[str, ...]]) -> tuple[str, ...]:
    """The static lookup path of *context* (``""`` for root): own context, its bases in order,
    then ``""`` (global)."""
    if not context or context == "*":
        return ("",)
    scopes: list[str] = [context]
    for base in base_contexts.get(context, ()):
        if base and base != "*" and base not in scopes:
            scopes.append(base)
    scopes.append("")
    return tuple(scopes)


def _resolve_producer_target(
    ref_context: str | None,
    ref_cmd: str,
    declaring_context: str,
    known_names: set[str],
    base_contexts: Mapping[str, tuple[str, ...]],
) -> tuple[str | None, list[str]]:
    """Resolve ``[ref_context/]ref_cmd`` from *declaring_context*.

    Returns ``(resolved_qualified_name, all_matching_qualified_names)``. When unambiguous,
    *resolved_qualified_name* is the target's qualified name; when 0 or 2+ match on the
    static lookup path, *resolved_qualified_name* is None and *all_matching_qualified_names*
    lists the candidates (on the static path if 2+, else anywhere in the workflow).
    """
    if ref_context:
        qualified = f"{ref_context}/{ref_cmd}"
        if qualified in known_names:
            return qualified, [qualified]
        return None, []

    scopes = _static_lookup_scopes(declaring_context, base_contexts)
    on_path: list[str] = []
    for scope in scopes:
        candidate = f"{scope}/{ref_cmd}" if scope else ref_cmd
        if candidate in known_names and candidate not in on_path:
            on_path.append(candidate)
    if len(on_path) == 1:
        return on_path[0], on_path
    if len(on_path) > 1:
        return None, on_path
    anywhere = sorted(
        q for q in known_names
        if q == ref_cmd or q.rsplit("/", 1)[-1] == ref_cmd
    )
    return None, anywhere


def resolve_signature_field_sources(
    command_name: str,
    signature_cls: type,
    file_path: str,
    resolution: FieldSourceResolution,
    *,
    phase_outputs: Sequence[type] = (),
) -> list[FieldSourceProblem]:
    """Resolve ``from_`` declarations on *signature_cls* (its ``Input``, ``Output``, and
    optional *phase_outputs* facts models) against *resolution*, mutating ``Input`` fields'
    ``json_schema_extra["available_from"]`` in place when valid.

    Can be called per-command (e.g. from ``CommandDirectory.ensure_command_hydrated``) on a
    freshly imported ``Signature`` class. Idempotent if ``available_from`` was already written
    by :func:`resolve_field_sources`.
    """
    problems: list[FieldSourceProblem] = []
    declaring_context = command_name.rpartition("/")[0]
    known_names = set(resolution.signatures) | set(resolution.import_errors)

    # 1. Check Signature.Output fields for illegal from_
    output_cls = getattr(signature_cls, "Output", None)
    if inspect.isclass(output_cls) and issubclass(output_cls, BaseModel):
        for field_name, info in output_cls.model_fields.items():
            extra = info.json_schema_extra if isinstance(info.json_schema_extra, Mapping) else {}
            if "from" in extra:
                problems.append(FieldSourceProblem(
                    file=file_path,
                    message=(f"Signature.Output.{field_name}: an Output field cannot declare "
                             "from_; build the value in __result__"),
                    severity="error",
                    command=command_name,
                ))

    # 2. Check facts models in phase_outputs for from_ (distinct from fact_from)
    for model in phase_outputs:
        if not (inspect.isclass(model) and issubclass(model, BaseModel)):
            continue
        for field_name, info in model.model_fields.items():
            extra = info.json_schema_extra if isinstance(info.json_schema_extra, Mapping) else {}
            if "from" in extra and is_from_source(extra["from"]):
                problems.append(FieldSourceProblem(
                    file=file_path,
                    message=(f"{model.__name__}.{field_name} "
                             f"(from {_spelled_from_text(extra['from'])}): from_ is "
                             "only for a command's Input field; type the field with the "
                             "command's Output instead"),
                    severity="error",
                    command=command_name,
                ))

    # 3. Check Signature.Input fields for hand-written available_from and resolve from_
    input_cls = getattr(signature_cls, "Input", None)
    if not (inspect.isclass(input_cls) and issubclass(input_cls, BaseModel)):
        return problems

    for field_name, info in input_cls.model_fields.items():
        extra = info.json_schema_extra
        if not isinstance(extra, Mapping):
            continue
        has_from = "from" in extra
        hw_af = _hand_written_available_from_value(extra)
        if hw_af is not None:
            af_val = hw_af
            first_cmd = (
                af_val[0] if isinstance(af_val, (list, tuple)) and af_val and isinstance(af_val[0], str)
                else af_val if isinstance(af_val, str) and af_val
                else "<command>"
            )
            problems.append(FieldSourceProblem(
                file=file_path,
                message=(f"Input.{field_name}: available_from is written by from_; "
                         f'write from_("{first_cmd}") or from_("{first_cmd}.<field>")'),
                severity="error",
                command=command_name,
            ))
            # Drop a hand-written edge so consumers do not keep reading it after the error;
            # preserve the value under ``_HAND_WRITTEN_AVAILABLE_FROM_KEY`` for later passes.
            if "available_from" in extra and not isinstance(extra["available_from"], _ResolvedAvailableFrom):
                new_extra = dict(extra)
                new_extra[_HAND_WRITTEN_AVAILABLE_FROM_KEY] = extra["available_from"]
                new_extra.pop("available_from", None)
                info.json_schema_extra = new_extra
            continue

        if not has_from:
            continue

        raw_source = extra["from"]
        if isinstance(raw_source, str):
            sources = [str(raw_source)]
        elif isinstance(raw_source, Sequence) and not isinstance(raw_source, (bytes, bytearray)):
            sources = [s for s in raw_source]
        else:
            sources = []

        if not sources or not all(isinstance(s, str) and s.strip() for s in sources):
            problems.append(FieldSourceProblem(
                file=file_path,
                message=(f"Input.{field_name}: from_ must be '[<Context>/]<command>[.<field>]' "
                         f"or a non-empty list of them, got {raw_source!r}"),
                severity="error",
                command=command_name,
            ))
            continue

        resolved_entries: list[str] = []
        field_ok = True
        for raw_entry in sources:
            entry = str(raw_entry).strip()
            match = _PRODUCER_REF_RE.match(entry)
            if match is None:
                problems.append(FieldSourceProblem(
                    file=file_path,
                    message=(f"Input.{field_name} (from {entry!r}): malformed reference; "
                             "expected '[<Context>/]<command>[.<field>]'"),
                    severity="error",
                    command=command_name,
                ))
                field_ok = False
                continue

            ref_ctx, ref_cmd, ref_field = match.group(1), match.group(2), match.group(3)
            target_qname, candidates = _resolve_producer_target(
                ref_ctx, ref_cmd, declaring_context, known_names, resolution.base_contexts
            )
            if target_qname is None:
                field_ok = False
                if ref_ctx:
                    problems.append(FieldSourceProblem(
                        file=file_path,
                        message=(f"Input.{field_name} (from {entry!r}): no command "
                                 f"{f'{ref_ctx}/{ref_cmd}'!r}"),
                        severity="error",
                        command=command_name,
                    ))
                elif len(candidates) > 1 and all(
                    c.rpartition("/")[0] in _static_lookup_scopes(declaring_context, resolution.base_contexts)
                    for c in candidates
                ):
                    problems.append(FieldSourceProblem(
                        file=file_path,
                        message=(f"Input.{field_name} (from {entry!r}): ambiguous command "
                                 f"{ref_cmd!r} matches {', '.join(candidates)}; qualify it "
                                 "as <Context>/<command>"),
                        severity="error",
                        command=command_name,
                    ))
                elif candidates:
                    scope_label = (
                        f"in {declaring_context}, its bases or the global commands"
                        if declaring_context else "in the global commands"
                    )
                    problems.append(FieldSourceProblem(
                        file=file_path,
                        message=(f"Input.{field_name} (from {entry!r}): no command {ref_cmd} "
                                 f"{scope_label}; qualify it ({', '.join(candidates)})"),
                        severity="error",
                        command=command_name,
                    ))
                else:
                    problems.append(FieldSourceProblem(
                        file=file_path,
                        message=f"Input.{field_name} (from {entry!r}): no command named {ref_cmd!r}",
                        severity="error",
                        command=command_name,
                    ))
                continue

            if target_qname in resolution.import_errors:
                field_ok = False
                problems.append(FieldSourceProblem(
                    file=file_path,
                    message=(f"Input.{field_name} (from {entry!r}): target command "
                             f"{target_qname!r} could not be imported: "
                             f"{resolution.import_errors[target_qname]}"),
                    severity="error",
                    command=command_name,
                ))
                continue

            # A producer edge on a planning command's Input must not name one of its own
            # callees: the edge sends the agent to run the target first, and the steps run it anyway (3.3).
            if target_qname in resolution.callees.get(command_name, frozenset()):
                field_ok = False
                problems.append(FieldSourceProblem(
                    file=file_path,
                    message=(f"Input.{field_name} (from {entry!r}) tells the agent to run "
                             f"{target_qname} first, but this command's own steps call "
                             f"{target_qname}. To describe the value, write "
                             "Field(description=...) instead"),
                    severity="error",
                    command=command_name,
                ))
                continue

            target_sig = resolution.signatures.get(target_qname)
            target_out = getattr(target_sig, "Output", None)
            target_in = getattr(target_sig, "Input", None)
            out_fields = (
                target_out.model_fields
                if inspect.isclass(target_out) and issubclass(target_out, BaseModel)
                else {}
            )
            in_fields = (
                target_in.model_fields
                if inspect.isclass(target_in) and issubclass(target_in, BaseModel)
                else {}
            )

            if ref_field is None:
                resolved_entries.append(target_qname)
                suggestion = (
                    f"{entry}.<field> (e.g. {', '.join(f'{entry}.{f}' for f in list(out_fields)[:3])})"
                    if out_fields else f"{entry}.<field>"
                )
                problems.append(FieldSourceProblem(
                    file=file_path,
                    message=(f"Input.{field_name} (from {entry!r}): command-only producer "
                             f"reference; consider naming an Output field ({suggestion})"),
                    severity="info",
                    command=command_name,
                ))
                continue

            spelled_target = f"{ref_ctx}/{ref_cmd}" if ref_ctx else ref_cmd
            if ref_field in out_fields:
                target_finfo = out_fields[ref_field]
                if not _producer_types_match(info.annotation, target_finfo.annotation):
                    field_ok = False
                    problems.append(FieldSourceProblem(
                        file=file_path,
                        message=(f"Input.{field_name} (from {entry!r}): "
                                 f"{_annotation_text(info.annotation)} does not match "
                                 f"{spelled_target}'s Output.{ref_field}: "
                                 f"{_annotation_text(target_finfo.annotation)}"),
                        severity="error",
                        command=command_name,
                    ))
                    continue
                resolved_entries.append(f"{target_qname}.{ref_field}")
            elif ref_field in in_fields:
                field_ok = False
                problems.append(FieldSourceProblem(
                    file=file_path,
                    message=(f"Input.{field_name} (from {entry!r}): {spelled_target}'s "
                             f"{ref_field} is an Input field; from_ names the Output field a "
                             "value comes from. To describe the value, write "
                             f"Field(description=...); the preview shows "
                             f"{spelled_target}.{ref_field}'s description beside it"),
                    severity="error",
                    command=command_name,
                ))
            else:
                field_ok = False
                problems.append(FieldSourceProblem(
                    file=file_path,
                    message=(f"Input.{field_name} (from {entry!r}): {target_qname} has no "
                             f"Output field {ref_field!r}"),
                    severity="error",
                    command=command_name,
                ))

        if field_ok and resolved_entries:
            new_extra = dict(extra)
            new_extra["available_from"] = _ResolvedAvailableFrom(resolved_entries)
            info.json_schema_extra = new_extra

    return problems


def _typed_phase_output_problems(
    folder: str,
    planning_commands: Mapping[str, PlanningCommandCommand],
    resolution: FieldSourceResolution,
) -> list[FieldSourceProblem]:
    """Load errors for typed phase outputs and step coverage (3.12)."""
    from fastworkflow import typed_phase_facts as tpf

    problems: list[FieldSourceProblem] = []
    signatures = resolution.signatures
    known = set(signatures) | {name.split("/", 1)[-1] for name in signatures}
    for qname, command in sorted(planning_commands.items()):
        fpath = command.path
        for model in command.phase_outputs:
            for message in tpf.check_typed_facts_model(
                fpath, model, resolution.model_to_command, signatures,
            ):
                problems.append(FieldSourceProblem(
                    file=fpath, message=message, severity="error", command=qname,
                ))
        if command.plan_fn is None:
            continue
        for env in _template_grid(command):
            try:
                result = command.plan_fn(input_instance(command, env))
            except Exception:  # noqa: BLE001
                continue
            if is_flat_plan(result):
                continue
            try:
                phases = active_phases(result, fpath)
            except PhaseError:
                continue
            for phase in phases:
                if phase.output is None:
                    continue
                inst = input_instance(command, env)
                try:
                    lines = phase_steps(phase, {"input": inst})
                except PhaseError:
                    # An earlier phase's facts are not there at load: the phase's steps on
                    # placeholder facts are the ones it can run (a callee's step, read as the template has it).
                    try:
                        lines = placeholder_steps(phase)
                    except PhaseError:
                        lines = []
                step_cmds = [name for name, _ in step_commands(lines, known)]
                for message in tpf.check_phase_steps_name_commands(
                    fpath, phase.name, step_cmds, phase.output,
                    resolution.model_to_command, signatures,
                ):
                    problems.append(FieldSourceProblem(
                        file=fpath, message=message, severity="error", command=qname,
                    ))
    return problems


def resolve_field_sources(workflow_folderpath: str) -> FieldSourceResolution:
    """Resolve every ``from_`` declaration across the commands and planning commands of *workflow_folderpath*.

    Runs once per workflow (cached by normalized folder path and command-source fingerprint):
    - reads command files from :class:`CommandDirectory` (``_commands/``, planning commands included),
      recording any import failure rather than dropping the command;
    - builds the ``{Signature.Output | Signature.Input -> qualified_name}`` map (3.12);
    - checks every ``from_`` placement and reference grammar, runs the type checks, flags
      hand-written ``available_from`` declarations, and writes ``available_from`` into
      valid ``Input`` fields' ``json_schema_extra``.
    """
    folder = os.path.abspath(workflow_folderpath)
    fingerprint = compute_commands_source_fingerprint(workflow_folderpath)
    return _resolve_field_sources_cached(folder, fingerprint)


@functools.lru_cache(maxsize=None)
def _resolve_field_sources_cached(folder: str, _source_fingerprint: str) -> FieldSourceResolution:
    problems: list[FieldSourceProblem] = []
    skipped: set[str] = set()
    inh_path = os.path.join(folder, "workflow_inheritance_model.json")
    try:
        inh = WorkflowInheritanceModel.load(folder)
        inh.resolve_base_paths(folder)
    except Exception as exc:  # noqa: BLE001
        problems.append(FieldSourceProblem(
            file=inh_path,
            message=f"{type(exc).__name__}: {exc}",
            severity="error",
        ))
    cmd_dir = CommandDirectory.scan(folder)
    core_names = set(cmd_dir.core_command_names)
    all_files: dict[str, tuple[str, str, bool]] = {}
    for qname, metadata in cmd_dir.map_command_2_metadata.items():
        if qname in core_names:
            continue
        owner = metadata.workflow_folderpath or folder
        all_files[qname] = (
            metadata.response_generation_module_path,
            owner,
            metadata.is_planning,
        )

    signatures: dict[str, Any] = {}
    import_errors: dict[str, str] = {}
    model_to_command: dict[type, str] = {}
    planning_command_phase_outputs: dict[str, tuple[type, ...]] = {}
    planning_commands: dict[str, PlanningCommandCommand] = {}

    for qname in sorted(all_files):
        fpath, owner, is_planning = all_files[qname]
        try:
            module = python_utils.get_module(fpath, owner)
        except Exception as exc:  # noqa: BLE001
            import_errors[qname] = f"{type(exc).__name__}: {exc}"
            continue
        if module is None:
            import_errors[qname] = f"could not import {fpath}"
            continue
        sig = getattr(module, "Signature", None)
        if not inspect.isclass(sig):
            continue
        signatures[qname] = sig
        for attr in ("Input", "Output"):
            model_cls = getattr(sig, attr, None)
            if inspect.isclass(model_cls) and issubclass(model_cls, BaseModel):
                model_to_command[model_cls] = qname
        if is_planning and getattr(module, "PlanGenerator", None) is not None:
            try:
                parsed = command_from_module(qname, module, fpath)
                planning_commands[qname] = parsed.command
                if parsed.command.phase_outputs:
                    planning_command_phase_outputs[qname] = parsed.command.phase_outputs
            except PlanningCommandFormatError as exc:
                msg = str(exc)
                prefix = f"{fpath}: "
                if msg.startswith(prefix):
                    msg = msg[len(prefix):]
                problems.append(FieldSourceProblem(
                    file=fpath, message=msg, severity="error", command=qname
                ))

    base_contexts, ctx_problems = _read_base_contexts(folder)
    for cp in ctx_problems:
        if cp not in problems:
            problems.append(cp)
    resolution = FieldSourceResolution(
        workflow_folderpath=folder,
        problems=(),
        skipped_commands=frozenset(),
        model_to_command=model_to_command,
        signatures=signatures,
        command_files={q: f for q, (f, _, _) in all_files.items()},
        import_errors=import_errors,
        base_contexts=base_contexts,
        planning_commands=planning_commands,
    )

    # The callee graph and its cycles (3.7) are built before the from_ checks, which read it.
    resolution = replace(resolution, callees=_callee_graph(planning_commands, resolution))
    cycle_problems = _cycle_problems(planning_commands, resolution.callees)
    problems.extend(cycle_problems)
    skipped.update(p.command for p in cycle_problems)

    for qname in sorted(signatures):
        fpath = all_files[qname][0]
        cmd_problems = resolve_signature_field_sources(
            qname,
            signatures[qname],
            fpath,
            resolution,
            phase_outputs=planning_command_phase_outputs.get(qname, ()),
        )
        for p in cmd_problems:
            if p not in problems:
                problems.append(p)
            if p.severity == "error":
                skipped.add(qname)

    problems.extend(_typed_phase_output_problems(folder, planning_commands, resolution))

    return replace(
        resolution,
        problems=tuple(problems),
        skipped_commands=frozenset(skipped),
    )


def is_framework_planning_command(command: "PlanningCommandCommand") -> bool:
    """True for a framework-provided planning command (fastworkflow/_framework_commands): system machinery, not
    trained, not inventoried, not listed. It stays loadable and callable by name (``planning_command``)."""
    import fastworkflow.command_directory as command_directory

    source = os.path.abspath(command.path) if command.path else ""
    root = os.path.abspath(str(command_directory.FRAMEWORK_PLANNING_COMMANDS_DIR))
    return source == root or source.startswith(root + os.sep)


def visible_planning_commands(workflow_folderpath: str) -> dict[str, "PlanningCommandCommand"]:
    """The planning commands an inventory counts (the test-set builder, the scoring listings): every loaded planning
    command except the framework's own."""
    return {name: command for name, command in
            resolve_field_sources(workflow_folderpath).planning_commands.items()
            if not is_framework_planning_command(command)}


def planning_command(workflow_folderpath: str, qualified_name: str) -> PlanningCommandCommand | None:
    """The loaded planning command *qualified_name* of a workflow, or None when it has none
    (not a planning command, or its module does not load)."""
    return resolve_field_sources(workflow_folderpath).planning_commands.get(qualified_name)


def _clear_field_sources_cache() -> None:
    _resolve_field_sources_cached.cache_clear()
    from fastworkflow import command_directory  # circular import: planning_commands -> command_context_model -> command_directory

    command_directory.clear_resolved_command_signatures_cache()


resolve_field_sources.cache_clear = _clear_field_sources_cache  # type: ignore[attr-defined]
resolve_field_sources.cache_info = _resolve_field_sources_cached.cache_info  # type: ignore[attr-defined]
resolve_field_sources.cache_parameters = _resolve_field_sources_cached.cache_parameters  # type: ignore[attr-defined]



#: A step that names a command: its first word, snake_case (``find_item``, ``go_up``) or a
#: qualified name (``Resource/show_metadata``); then its chain, cut at ``(``, ``;`` or ``,``
#: outside a quoted value (``go_up, then ...`` is prose after the command).
_STEP_COMMAND = re.compile(r"^([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+|[A-Z]\w*/\w+)\b")
_STEP_ARG = re.compile(r"(?<![\w<])([A-Za-z_]\w*)=")
#: The note a template line ends with when it is written only under a condition
#: (``find_item query=<q> (only when find.uids)``).
_CONDITION_NOTE = re.compile(r"\s*\(only when .*\)$")


def without_condition_note(line) -> str:
    """*line* without its closing ``(only when ...)`` note."""
    return _CONDITION_NOTE.sub("", str(line).rstrip())


def step_commands(lines, known: set[str]) -> list[tuple[str, tuple[str, ...]]]:
    """``(command, parameters)`` of each command step of *lines*: a step whose first word is
    a snake_case name or one of *known* (``find_item query=Dune (note)`` gives
    ``("find_item", ("query",))``); the parameters are the ``key=`` of its chain, before
    the first ``(``, ``;`` or ``,`` outside a quoted value, as code reads them
    (``planning_phases.code_step``)."""
    out = []
    for line in lines:
        text = without_condition_note(str(line).strip())
        first = re.match(r"[\w/]+", text)
        match = _STEP_COMMAND.match(text)
        if match is None and not (first and first.group(0) in known):
            continue
        name = match.group(1) if match else first.group(0)
        chain = _outside_quoted_values(step_chain(text[len(name):])).split(",", 1)[0]
        out.append((name, tuple(_STEP_ARG.findall(chain))))
    return out


def _note_condition_problem(path: str, phase: str, step_text,
                            known: set[str]) -> str | None:
    """The preview --check warning for a non-conditional command step whose note holds a
    condition word (``_NOTE_CONDITION_RE``: runtime words except bare ``only``), or None."""
    text = str(step_text).strip()
    parsed = parse_plan(f"1. {text}", known)
    if not parsed or parsed[0].kind != KIND_COMMAND or not parsed[0].command:
        return None
    clean = text.replace("`", " ").replace("*", " ")
    m_cmd = re.search(rf"\b{re.escape(parsed[0].command)}\b", clean)
    if m_cmd is None:
        return None
    rest = text[m_cmd.end():]
    chain = step_chain(rest)
    note = rest[len(chain):].strip()
    if not note or not _NOTE_CONDITION_RE.search(note):
        return None
    body = note[1:-1].strip() if note.startswith("(") and note.endswith(")") else note.lstrip(";").strip()
    cond_match = _NOTE_CONDITION_RE.search(body)
    if cond_match is None:
        return None
    tail = body[cond_match.start():]
    cond_text = re.sub(
        r"^(?:only\s+when|only\s+if|whenever|when|if|unless|in\s+case)\s+",
        "",
        tail,
        flags=re.IGNORECASE,
    ).strip() or tail.strip()
    without_note = text[:m_cmd.end() + len(chain)].rstrip()
    return (f"{path}: phase {phase!r}, step {text!r}: the note holds a condition, "
            f"but code never reads a note, so {parsed[0].command} would run every time. "
            f'Decide it in code with if, or write when("{cond_text}", "{without_note}") '
            "for the agent to decide")


def _command_steps(command: PlanningCommandCommand, signatures: dict) -> tuple[dict, list[tuple[str, tuple[str, ...]]]]:
    """The workflow's command fields (*signatures* by name, and by the name's last part) and
    each command step of *command*'s template as ``(command, parameters)``: every placeholder
    combination's ``step_commands``, and each ``run``/``full_list`` and ``pick`` call. The parse
    that ``check_step_commands`` checks and the callee graph (``_callee_graph``) both read."""
    with planning_api.command_inputs_for_steps(signatures, collect=True) as (fields_map, steps):
        fields = fields_map or {}
        known = set(fields)
        seen: dict[tuple[str, tuple[str, ...]], None] = {}
        for env in _template_grid(command):
            steps.clear()
            with planning_api.picks_noted() as picks, planning_api.steps_by_phase_noted() as by_phase:
                try:
                    lines = _call_steps(command.steps_fn, input_instance(command, env), command.path)
                except Exception:  # noqa: BLE001 - the template build reports it
                    continue
            for found in (step_commands(lines, known) + list(steps)
                          + [(c, (p,) if p else ()) for c, p in picks]):
                seen.setdefault(found)
    return fields, list(seen)


def check_step_commands(command: PlanningCommandCommand, signatures: dict) -> list[str]:
    """Problems with the commands a planning command's steps name: each command step of its template
    (every placeholder combination; ``step_commands``) and each ``pick`` must name a
    command of the workflow (*signatures*: ``facts.workflow_signatures``), and each
    parameter it passes must be a field of that command's ``Signature.Input``, the schema a
    command call is checked against. Returned as messages: the planning command fails its load
    (``workflow_agent``) and ``fastworkflow command preview --check`` fails."""
    fields, seen = _command_steps(command, signatures)
    problems = []
    for name, args in seen:
        if name not in fields:
            problem = (f"{command.path}: a step runs {name!r}, which is not a command of the "
                       "workflow")
        else:
            unknown = [a for a in args if not any(a in f for f in fields[name])]
            if not unknown:
                continue
            problem = (f"{command.path}: a step passes {', '.join(unknown)} to {name!r}, whose "
                       "Signature.Input has no such field")
        if problem not in problems:
            problems.append(problem)
    return problems


def _callee_graph(planning_commands: Mapping[str, PlanningCommandCommand],
                  resolution: FieldSourceResolution) -> dict[str, frozenset[str]]:
    """``callees[name]`` for each planning command (3.7): the qualified names its steps call.

    The steps are read by ``_command_steps``, the parse ``check_step_commands`` checks, fixed
    steps included. A step's name resolves from the command's own context as a producer's does
    (``_resolve_producer_target``). A name off that path that exactly one command of the
    workflow has resolves to it: a step runs in the context an earlier callee left open
    (``book_loans`` runs ``list_loans`` on the book ``book_by_title`` opened). An ambiguous or
    unknown name is no edge; the cycle guard at run time refuses a cycle the graph cannot see.
    """
    known = set(resolution.signatures)
    graph: dict[str, frozenset[str]] = {}
    for qname, command in sorted(planning_commands.items()):
        _, seen = _command_steps(command, resolution.signatures)
        targets: set[str] = set()
        for name, _ in seen:
            ref_context, _, ref_cmd = name.rpartition("/")
            target, candidates = _resolve_producer_target(
                ref_context or None, ref_cmd, qname.rpartition("/")[0], known, resolution.base_contexts)
            if target is None and len(candidates) == 1:
                target = candidates[0]
            if target is not None:
                targets.add(target)
        graph[qname] = frozenset(targets)
    return graph


def _cycle_through(start: str, callees: Mapping[str, frozenset[str]]) -> list[str] | None:
    """The shortest call path from *start* back to *start* (``[start, ..., start]``), or None
    when *start* is on no cycle."""
    came_from: dict[str, str] = {}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for callee in sorted(callees.get(node, ())):
            if callee == start:
                chain = [node]
                while chain[-1] != start:
                    chain.append(came_from[chain[-1]])
                return list(reversed(chain)) + [start]
            if callee not in came_from:
                came_from[callee] = node
                queue.append(callee)
    return None


def _cycle_problems(planning_commands: Mapping[str, PlanningCommandCommand],
                    callees: Mapping[str, frozenset[str]]) -> list[FieldSourceProblem]:
    """A load error for every planning command on a call cycle, a self-call included (3.7).
    Each names the cycle, so the log says which commands form it."""
    problems = []
    for qname in sorted(planning_commands):
        cycle = _cycle_through(qname, callees)
        if cycle is None:
            continue
        path = " -> ".join(cycle)
        message = (f"a planning command cannot call itself: {path}" if len(cycle) == 2
                   else f"a planning command cannot call itself, even through others: {path}")
        problems.append(FieldSourceProblem(
            file=planning_commands[qname].path, message=message, severity="error", command=qname,
        ))
    return problems


def call_depth(name: str, resolution: FieldSourceResolution, _path: frozenset[str] = frozenset()) -> int:
    """The ``Depth:`` of planning command *name* (3.7): the longest chain of planning commands
    from it, itself counted, an execution command counting 0. A cycle is a load error; the
    guard keeps the walk finite."""
    if name in _path:
        return 0
    below = [call_depth(callee, resolution, _path | {name})
             for callee in sorted(resolution.callees.get(name, ()))
             if callee in resolution.planning_commands]
    return 1 + max(below, default=0)


def _grid_phase_steps(planning_command: PlanningCommandCommand | PlanningCommandFile,
                      signatures: dict) -> tuple[str, list[tuple[str, object]]]:
    """*planning_command*'s path and each ``(phase, step line)`` of its placeholder grid (3.13 checks)."""
    from fastworkflow import planning as planning_api

    with planning_api.command_inputs_for_steps(signatures, collect=True) as (_, steps):
        seen_steps: list[tuple[str, object]] = []
        command = planning_command.command if isinstance(planning_command, PlanningCommandFile) else planning_command
        path = command.path
        for env in _template_grid(command):
            steps.clear()
            with planning_api.steps_by_phase_noted() as by_phase:
                try:
                    lines = _call_steps(command.steps_fn, input_instance(command, env), command.path)
                except Exception:  # noqa: BLE001 - the template build reports it
                    continue
            items = by_phase or [
                (command.name,
                 planning_api.Step(without_condition_note(line), line.helper)
                 if isinstance(line, planning_api.Step) else without_condition_note(line))
                for line in lines
            ]
            for phase_name, step_line in items:
                key = (phase_name, str(step_line), getattr(step_line, "helper", None))
                if key not in {(p, str(s), getattr(s, "helper", None)) for p, s in seen_steps}:
                    seen_steps.append((phase_name, step_line))
        return path, seen_steps


def _hand_typed_warning(path: str, phase: str, step_line: str,
                        known: set[str]) -> str | None:
    """The warning for a step that a helper would write, typed by hand (3.13), or None."""
    from fastworkflow.planning import Step

    if isinstance(step_line, Step):
        return None
    text = str(step_line).strip()
    if not text:
        return None
    parsed = parse_plan(f"1. {text}", known)
    if not parsed or not parsed[0].command:
        return None
    step = parsed[0]
    clean = text.replace("`", " ").replace("*", " ")
    m_cmd = re.search(rf"\b{re.escape(step.command)}\b", clean)
    if m_cmd is None:
        return None
    before = clean[:m_cmd.start()]
    rest = text[m_cmd.end():]
    chain = step_chain(rest)
    note = rest[len(chain):]
    helper: str | None = None
    if step.kind == KIND_SKIPPED and _NEGATION_RE.search(before):
        helper = "do_not_run"
    elif step.kind == KIND_CONDITIONAL and _CONDITION_RE.search(before):
        helper = "when"
    elif step.kind == KIND_COMMAND and _PICK_NAMED_RE.search(text):
        helper = "pick"
    elif step.kind == KIND_COMMAND and any(p in note.lower() for p in FULL_LIST_PHRASES):
        helper = "full_list"
    elif step.kind == KIND_COMMAND:
        helper = "run"
    if helper is None:
        return None
    return (f"{path}: phase {phase!r}, step {text!r}: written by hand; "
            f"write {helper}(...) so that code reads it as you mean")


def check_hand_typed_steps(planning_command: PlanningCommandCommand | PlanningCommandFile, signatures: dict) -> list[str]:
    """Warnings for steps of *planning_command*'s placeholder grid that a helper would write,
    typed by hand (3.13)."""
    from fastworkflow import planning as planning_api

    with planning_api.command_inputs_for_steps(signatures, collect=True) as (fields_map, _):
        known = set(fields_map or {})
    path, seen_steps = _grid_phase_steps(planning_command, signatures)
    warnings: list[str] = []
    for phase_name, step_line in seen_steps:
        warning = _hand_typed_warning(path, phase_name, step_line, known)
        if warning is not None and warning not in warnings:
            warnings.append(warning)
    return warnings


def check_note_condition_steps(planning_command: PlanningCommandCommand | PlanningCommandFile, signatures: dict) -> list[str]:
    """Warnings for non-conditional command steps whose note holds a condition word (3.13)."""
    from fastworkflow import planning as planning_api

    with planning_api.command_inputs_for_steps(signatures, collect=True) as (fields_map, _):
        known = set(fields_map or {})
    path, seen_steps = _grid_phase_steps(planning_command, signatures)
    warnings: list[str] = []
    for phase_name, step_line in seen_steps:
        warning = _note_condition_problem(path, phase_name, step_line, known)
        if warning is not None and warning not in warnings:
            warnings.append(warning)
    return warnings


def phases_for(command: PlanningCommandCommand, args: dict) -> list | None:
    """The phases ``plan()`` gives for ``args`` (complete, checked arguments): those that run.

    ``None`` when ``plan()`` returns ``None`` or ``[]`` (``planning_phases.is_flat_plan``):
    this call runs flat, from ``steps()``. Raises PlanningCommandCallError when ``plan()`` fails,
    returns something else than a list of ``Phase``, or leaves every phase out.
    """
    from fastworkflow.planning_phases import PhaseError, active_phases, is_flat_plan

    instance = input_instance(command, args)
    try:
        raw = command.plan_fn(instance)
        if is_flat_plan(raw):
            return None
        phases = active_phases(raw, f"plan() of {command.name}")
    except PhaseError as exc:
        raise PlanningCommandCallError(str(exc)) from None
    except Exception as exc:  # noqa: BLE001 - workflow code; the planner falls back to the template
        raise PlanningCommandCallError(f"plan() of {command.name} failed: {type(exc).__name__}: {exc}") from None
    if not phases:
        raise PlanningCommandCallError(f"plan() of {command.name} returns no phase")
    return phases


def call_phases(command: PlanningCommandCommand, args: dict) -> list:
    """The phases a call of ``command`` with ``args`` runs: ``phases_for``'s, or, for a flat
    call (no ``plan()``, or ``plan()`` gives no phases), one phase named after the planning command whose
    steps are ``expand(command, args)``. A planning command without phases runs as one phase.
    Raises PlanningCommandCallError as ``phases_for`` and ``expand`` do."""
    from fastworkflow.planning_phases import Phase

    phases = phases_for(command, args) if command.plan_fn is not None else None
    if phases is not None:
        return phases
    steps = expand(command, args)
    return [Phase(command.name, lambda: list(steps))]


def expand(command: PlanningCommandCommand, args: dict) -> list[str]:
    """The planning command's command steps for ``args`` (complete, checked arguments), unnumbered:
    ``steps(input)``. Raises PlanningCommandCallError when ``steps()`` fails or returns something else
    than a list of text."""
    instance = input_instance(command, args)
    try:
        lines = _call_steps(command.steps_fn, instance, command.path)
    except PlanningCommandCallError:
        raise
    except Exception as exc:  # noqa: BLE001 - workflow code; the planner falls back to the template
        raise PlanningCommandCallError(f"steps() of {command.name} failed: {type(exc).__name__}: {exc}") from None
    return lines


def number(steps: list[str]) -> str:
    """Steps as the planner writes them: a numbered list, one per line."""
    return "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))


def _run_validate(command: PlanningCommandCommand, args: dict, request: str, workflow) -> dict:
    """``args`` as the planning command's ``validate_extracted_parameters`` leaves them; raises
    PlanningCommandCallError when it refuses them or raises."""
    import inspect

    instance = input_instance(command, args)
    hook = command.validate
    kwargs = {"command_text": request} if "command_text" in inspect.signature(hook).parameters else {}
    import fastworkflow

    from fastworkflow.utils.signatures import check_hook_accepts

    try:
        result = hook(workflow, command.name, instance, **kwargs)
    except fastworkflow.Refused as refusal:
        raise PlanningCommandInputError(f"validate_extracted_parameters: {refusal}", fields=refusal.fields) from None
    except Exception as exc:  # noqa: BLE001 - workflow code; the planner falls back to the template
        raise PlanningCommandCallError(f"validate_extracted_parameters raised {type(exc).__name__}: {exc}") from None
    check_hook_accepts(result, command.name)
    fields = command.input_model.model_fields
    out = {p.name: getattr(instance, p.name, args.get(p.name)) for p in command.params}
    # an Optional[str] field set to None is the absent argument, "" (``_absent_as_none``)
    return {n: "" if v is None and fields[n].default is None else v for n, v in out.items()}


def _validate_arguments(command: PlanningCommandCommand, args: dict, request: str, workflow=None,
                        notes: list[str] | None = None) -> dict:
    """The validation steps after the options' checks, in their order (3.4, The argument path), for a
    typed planning call, which has no ``Signature`` to run ``validate_parameters`` on: the planning
    command file's ``validate_extracted_parameters`` hook (it may correct fields), then the ``Input``'s
    own rules with the request and workflow as context (``check_input``), then the roles of two
    inputs given one value (``_check_roles``). Raises PlanningCommandInputError when the hook or a rule
    refuses the call, PlanningCommandCallError when the hook raises."""
    checked = args if command.validate is None else _run_validate(command, args, request, workflow)
    for p in command.params:
        if checked[p.name] != args.get(p.name) and notes is not None:
            notes.append(f"{p.name}={args.get(p.name)!r} -> {checked[p.name]!r} (validate_extracted_parameters)")
    written = check_input(command, checked, request, workflow)
    if written:
        if notes is not None:
            notes += [f"{n}={checked[n]!r} -> {v!r} (Input validator)" for n, v in written.items()]
        checked = {**checked, **written}
    _check_roles(command, checked, request)
    return checked


def _validation_text(error) -> str:
    """A pydantic ValidationError as one line: ``field: message; ...`` (a validator's own
    ValueError text without pydantic's "Value error, " prefix)."""
    parts = []
    for e in error.errors():
        message = str(e.get("msg", "")).removeprefix("Value error, ")
        where = ".".join(str(x) for x in e.get("loc", ()))
        parts.append(f"{where}: {message}" if where else message)
    return "; ".join(parts)


def check_input(command: PlanningCommandCommand, args: dict, request: str, workflow=None) -> dict:
    """Validate ``args`` against the planning command's ``Input`` (its field types and its validators), with
    the context ``{"request": request, "workflow": workflow}``. Raises PlanningCommandInputError, whose
    message says which field breaks which rule.

    Returns the fields the validators changed, ``{name: value}``, which the caller writes back onto the
    arguments (an Input validator's returned value is kept, design 3.4). Empty when none changed."""
    from pydantic import ValidationError

    passed = _absent_as_none(command, args)
    try:
        validated = command.input_model.model_validate(passed, context={"request": request, "workflow": workflow})
    except ValidationError as exc:
        raise PlanningCommandInputError(f"the arguments break a rule of {command.name}: {_validation_text(exc)}") from None
    return {name: getattr(validated, name) for name in command.input_model.model_fields
            if name in passed and getattr(validated, name) != passed[name]}


def _check_roles(command: PlanningCommandCommand, args: dict, request: str) -> None:
    """The same value for two required free-text inputs: the one whose role the request does not
    write just before the value is refused (``_role_fits``); when both fit or neither does, the
    first is refused. Raises PlanningCommandInputError."""
    required = [p for p in command.params if p.type == "str" and p.required and not p.values]
    for i, p in enumerate(required):
        for q in required[i + 1:]:
            value = str(args[p.name])
            if not value or value.lower() != str(args[q.name]).lower():
                continue
            fits_p, fits_q = _role_fits(request, value, p), _role_fits(request, value, q)
            loser, other = (q, p) if fits_p and not fits_q else (p, q)
            raise PlanningCommandInputError(f"{loser.name}={args[loser.name]!r} is also the value of {other.name}")


# ---------------------------------------------------------------- arguments and guards

def _clean_str(value) -> str:
    text = str(value).strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'`":
        text = text[1:-1].strip()
    return text.rstrip(" .;,")


def strip_words(text: str, words) -> str:
    """``text`` without the *words* at either end (case-insensitive; the words are whole words of ``text``).

    The value cleanup for a field with no check, as ordinary code: call it from an Input's field
    validator (whose returned value is written back) or from ``Signature.validate_extracted_parameters``.
    A field with a check takes ``in_request: {"ignore": [words]}`` instead, which runs before the match.
    A possessive ``'s`` left on a name stripped from its end goes with it ("Ann's group" -> "Ann").
    Text of one word is kept whole.
    """
    words = tuple(words)
    if not words:
        return text
    lowered = {w.lower() for w in words}
    tokens = text.split()
    changed = True
    while changed and len(tokens) > 1:
        changed = False
        if tokens[0].lower() in lowered:
            tokens, changed = tokens[1:], True
        elif tokens[-1].lower() in lowered:
            tokens, changed = tokens[:-1], True
            if tokens and re.search(r"['’]s$", tokens[-1]):
                tokens[-1] = tokens[-1][:-2]
    return " ".join(tokens).strip()


def _single(value):
    """``value``, or the element of a one-element list given for it (a list, or the list
    written as JSON text: ``'["Payroll"]'``)."""
    if isinstance(value, str) and value.strip().startswith("[") and value.strip().endswith("]"):
        try:
            value = json.loads(value)
        except ValueError:
            return value
    if isinstance(value, (list, tuple)) and len(value) == 1 and not isinstance(value[0], (list, tuple, dict)):
        return value[0]
    return value


def _list_items(value) -> list[str]:
    """The elements a list[str] argument holds: a list or tuple, the list written as JSON text, or
    comma-separated text. None elements are left out here; empty ones are left to the caller."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v is not None]
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = None
        if isinstance(parsed, list):
            return [str(v) for v in parsed if v is not None]
    return text.split(",") if text else []


def _empty(value) -> bool:
    return value is None or str(value).strip().strip("\"'`").lower() in ("", "none", "null")


def _confident(verdicts: dict | None, name: str):
    """The ``(value, probability)`` check of ``name`` when it is sure enough, else None."""
    verdict = (verdicts or {}).get(name)
    return verdict if verdict is not None and verdict[1] >= VERIFY_THRESHOLD else None


def _request_numbers(request: str, args: dict, command: PlanningCommandCommand) -> set[int]:
    """The numbers ``request`` writes (digits or number words) outside the str arguments' text."""
    lowered = request.lower()
    taken: list[tuple[int, int]] = []
    for p in command.params:
        text = str(args.get(p.name) or "").lower() if p.type == "str" else ""
        at = lowered.find(text) if text else -1
        while at >= 0:
            taken.append((at, at + len(text)))
            at = lowered.find(text, at + 1)
    found = set()
    for m in _NUMBER.finditer(request):
        if any(a <= m.start() < b for a, b in taken):
            continue
        word = m.group(1).lower()
        found.add(int(word) if word.isdigit() else NUMBER_WORDS[word])
    return found


def _spans(request: str, text: str) -> list[tuple[int, int]]:
    """Where ``text`` occurs in ``request`` (case-insensitive; any run of spaces, either apostrophe)."""
    if not text:
        return []
    pattern = re.escape(_fold(text)).replace(r"\ ", r"\s+").replace("'", "['’]")
    return [(m.start(), m.end()) for m in re.finditer(pattern, _fold(request), re.I)]


def _whole_identifier(request: str, text: str, identifiers: tuple) -> tuple[str | None, bool]:
    """``(match, cut)`` of a str argument against the workflow's identifier matches in ``request``.

    ``cut`` is true when every occurrence of ``text`` lies inside an identifier match without
    being all of it ("4421" of "acct_4421"); ``match`` is then the whole match when exactly one
    match holds it, else None.
    """
    matches = [(m.start(), m.end()) for p in identifiers for m in p.finditer(request) if m.group(0)]
    occurrences = _spans(request, text)
    if not matches or not occurrences:
        return None, False
    holders = set()
    for a, b in occurrences:
        inside = [(s, e) for s, e in matches if s <= a and b <= e and (s, e) != (a, b)]
        if not inside:
            return None, False
        holders |= {request[s:e] for s, e in inside}
    return (holders.pop() if len(holders) == 1 else None), True


def _name_like_word(word: str) -> bool:
    """A capitalised word that is not a common one ("I", "Before"), or one holding a digit or ``_``."""
    from fastworkflow.split_merge import _COMMON
    w = word.strip(".,;:!?()'\"`")
    return bool(w) and (any(c.isdigit() for c in w) or "_" in w
                        or (w[:1].isupper() and w != "I" and w.lower() not in _COMMON))


def _plain_name(request: str, a: int, b: int, kinds: frozenset[str] = frozenset()) -> bool:
    """``request[a:b]`` is a name written in lower case ("ann lee" of "is ann lee in Payroll"):
    two or more words, none a common one, with no such word right before or after it in its
    clause, so that it cuts no longer name ("ann lee" of "ann lee smith"), and not one name of
    a chain joined by "and" that ends on a company word ("henderson ramirez" of "henderson
    ramirez and lewis and sons", as ``split_merge`` reads a capitalised chain). A single
    lower-case word ("roster", "user") reads as a common noun, not a name.

    A neighbour that is a clause word, a verb of ``split_merge._FINITE`` or a preposition of
    ``split_merge._PLACING`` ("does barbara sanchez hold ...", "is alan cooper among ..."), is no
    part of the name; nor is one of the command's ``kinds`` (its enum values, or their plural)
    right before it when a clause word follows it ("every right brent randolph holds")."""
    from fastworkflow.split_merge import _COMPANY_SUFFIX, _FINITE, _NOT_NAMES, _PLACING, _STOP

    def bare(word: str) -> str:
        return word.strip(".,;:!?()'\"`").lower()

    def plain(word: str) -> bool:
        return bool(bare(word)) and bare(word) not in _NOT_NAMES | _STOP

    def company(ws: list[str]) -> bool:
        return any(bare(w) in _COMPANY_SUFFIX for w in ws)

    def clause_word(word: str) -> bool:
        return _FINITE.fullmatch(bare(word)) is not None or re.fullmatch(_PLACING, bare(word)) is not None

    def kind(word: str) -> bool:
        return bare(word) in kinds or (bare(word).endswith("s") and bare(word)[:-1] in kinds)

    words = request[a:b].split()
    clause = r"[.,;:!?()\"]"
    before = re.split(clause, request[:a])[-1].split()
    # a possessive ends the name ("ann lee's login")
    after = [] if re.match(r"['’]s\b", request[b:]) else re.split(clause, request[b:])[0].split()
    verb_after = bool(after) and clause_word(after[0])
    return (len(words) > 1 and all(plain(w) for w in words)
            and not (before and plain(before[-1]) and not clause_word(before[-1])
                     and not (verb_after and kind(before[-1])))
            and not (after and plain(after[0]) and not verb_after)
            and not (bare(after[0] if after else "") in ("and", "&") and company(after))
            and not (bare(before[-1] if before else "") in ("and", "&") and company(words)))


def _opening_tails(request: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The tails of the subject ``spans`` of ``request`` that open a sentence: such a subject
    may hold the word that opens the sentence ("Treat Ledger" of "Treat Ledger as ..."), and its
    tail ("Ledger") stands for it."""
    tails = []
    for s, e in spans:
        first = re.match(r"\S+\s+", request[s:e])
        if first and (s == 0 or re.search(r"[.?!:;]\s*$", request[:s])):
            tails.append((s + first.end(), e))
    return tails


def _check_subjects(command: PlanningCommandCommand, request: str) -> list[str]:
    """The request's subjects as a check offers them: a sentence-opening one with its tail
    after it (``_opening_tails``), so that a check never has to choose between "Treat Ledger"
    and no record at all."""
    out: list[str] = []
    for subject in _request_subjects(command, request):
        tails = [request[s:e] for s, e in _opening_tails(request, _spans(request, subject))]
        out += [n for n in [subject, *tails] if n.lower() not in {o.lower() for o in out}]
    return out


def _as_subject(request: str, text: str, subjects: list[str],
                kinds: frozenset[str] = frozenset()) -> tuple[str | None, str]:
    """``(argument, why)`` of a ``subject`` argument: ``text`` when it names a record of the
    request, the whole subject when it is one cut short, else None with ``why``.

    An occurrence of the argument (on word boundaries) names a record when it is a subject, or
    the tail of a sentence-opening one ("Ledger" of "Treat Ledger as ..."), or when it cuts no subject (each subject it overlaps it holds whole) and
    holds a subject or a name-like word (``_name_like_word``), or is a name written in lower case
    (``_plain_name``: "ann lee"): "I" of "Before I approve" names nothing. Otherwise an occurrence inside exactly one subject, at its start or end and cut
    there mid-word or at a space ("Payroll_Admin" of "Payroll_Administrator", "Ann" of "Ann
    Lee"), is that subject cut short; one cut at ``_`` or ``-`` ("Payroll" of "Payroll_Approver")
    may be a record of its own and is refused.
    """
    spans = [(s, e) for subject in subjects for s, e in _spans(request, subject)]
    parts = set(_opening_tails(request, spans))

    def joined(i: int) -> bool:   # request[i - 1] and request[i] are letters or digits of one word
        return 0 < i < len(request) and request[i - 1].isalnum() and request[i].isalnum()

    occurrences = _spans(request, text)
    whole_words = [(a, b) for a, b in occurrences if not joined(a) and not joined(b)]
    for a, b in whole_words:
        if (a, b) in parts:
            return text, ""
        cut = [(s, e) for s, e in spans if s < b and a < e and not (a <= s and e <= b)]
        if not cut and (any(a <= s and e <= b for s, e in spans) or any(_name_like_word(w) for w in text.split())
                        or _plain_name(request, a, b, kinds)):
            return text, ""
    snaps = set()
    for a, b in occurrences:
        for s, e in spans:
            if s <= a and b <= e and (s, e) != (a, b) and (s == a or e == b):
                # the subject is cut between request[edge - 1] and request[edge]
                edge, beyond = (b, request[b]) if s == a else (a, request[a - 1])
                if joined(edge) or beyond.isspace():
                    snaps.add(request[s:e])
    if len(snaps) == 1:
        return snaps.pop(), ""
    return None, "names no record of the request (none of its subjects, whole)"


def _request_subjects(command: PlanningCommandCommand, request: str) -> list[str]:
    from fastworkflow.split_merge import request_subjects
    return request_subjects(request, command.identifiers)


#: Dashes folded to "-" when an argument is looked up in the request (non-breaking hyphen etc.).
_DASHES = str.maketrans({c: "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d"})


def _fold(text: str) -> str:
    """``text`` with each Unicode dash as "-" (one character for one: positions are kept)."""
    return text.translate(_DASHES)


def _written(text: str, folded: str, request: str) -> str | None:
    """*text* as the request writes it (its own spelling, dashes and case ignored), or None."""
    at = folded.find(_fold(text).lower()) if text else -1
    return request[at:at + len(text)] if at >= 0 else None


def _screen(command: PlanningCommandCommand, raw: dict, request: str) -> tuple[dict, dict[str, str]]:
    """``(values, refused)``: the ``str`` and ``int`` arguments through the command parameter
    screen (``utils.signatures.screen_parameter_values``): quotes and angle brackets around a
    value removed, a made-up number replaced by the default; ``refused`` says why a placeholder
    or an out-of-range number is refused."""
    from fastworkflow.utils.signatures import screen_parameter_values

    names = [p.name for p in command.params if p.type in ("str", "int") and raw.get(p.name) is not None]
    if not names or command.input_model is None:
        return {}, {}
    values = command.input_model.model_construct(**{n: raw[n] for n in names})
    screen = screen_parameter_values(values, request, request, names)
    refused = {n: "is a placeholder, not a value" for n in screen.placeholder_fields}
    refused.update({n: f"is out of range ({why})" for n, why in screen.out_of_range_problems.items()})
    return {n: getattr(values, n) for n in names}, refused


# The rule sentences of design 3.4 (Descriptions), one per option the check below enforces. The
# CME parameter extractor reads each after the description of a field that carries the option, and
# nothing else reads them: not the preview, nor the listings (``fastworkflow.utils.signatures``
# appends them through ``extraction_sentences_for``).
IN_REQUEST_STR_SENTENCE = "Copy it exactly as the request writes it."   # in_request on a str
IN_REQUEST_INT_SENTENCE = "A number the request writes."                # in_request on an int
IN_REQUEST_LIST_SENTENCE = "Copy each item exactly as the request writes it."  # in_request on a list[str]
SUBJECT_SENTENCE = "Copy it exactly as the request writes it. It must name a record the request names."  # subject
VERIFY_BOOL_SENTENCE = "True only when the request asks for it."       # verify on a bool
VERIFY_BOOL_TRUE_SENTENCE = "False only when the request asks against it."  # verify on a bool that defaults true


def extraction_sentences(p: Param) -> tuple[str, ...]:
    """The rule sentences of the options *p* carries, as ``check_arguments`` enforces them
    (3.4, Descriptions). A subject field's sentence already says what ``in_request`` says, so it
    stands alone: the extractor reads each rule once."""
    sentences: list[str] = []
    if p.subject:
        sentences.append(SUBJECT_SENTENCE)
    elif p.in_request:
        sentences.append({"str": IN_REQUEST_STR_SENTENCE, "int": IN_REQUEST_INT_SENTENCE}.get(
            p.type, IN_REQUEST_LIST_SENTENCE))
    if p.verify and p.type == "bool":
        sentences.append(VERIFY_BOOL_SENTENCE if p.default is not True else VERIFY_BOOL_TRUE_SENTENCE)
    return tuple(sentences)


#: The check options (2.46), read from a field's ``json_schema_extra`` for a command of either kind.
CHECK_OPTIONS = ("in_request", "verify")


def option_params(input_model, where: str) -> tuple[Param, ...]:
    """A ``Param`` for each field of *input_model* (an ``Input``, of either kind of command) that carries
    a check option (2.46). Other fields are not checked. A load error (PlanningCommandFormatError) when
    an option is on a field of a type it does not apply to, or a field's type is not one the
    extractor supports."""
    from types import SimpleNamespace

    out = []
    for name, info in input_model.model_fields.items():
        extra = info.json_schema_extra
        if not isinstance(extra, dict):
            continue
        _retired_option_error(where, name, extra)
        options = {k: v for k, v in extra.items() if k in CHECK_OPTIONS}
        if not options:
            continue
        shadow = SimpleNamespace(annotation=info.annotation, description=info.description, default=info.default,
                                 default_factory=info.default_factory, json_schema_extra=options)
        out.append(_parse_param(name, _spec_from_field(name, shadow, where), where))
    return tuple(out)


@functools.lru_cache(maxsize=None)
def option_command(input_model, name: str, identifiers: tuple = ()) -> PlanningCommandCommand:
    """The checks of an execution command's ``Input`` as a ``PlanningCommandCommand`` holding only the fields
    that carry an option, so ``check_arguments`` and ``check_verdicts`` run on them unchanged (2.46)."""
    return PlanningCommandCommand(name=name, description="", params=option_params(input_model, name),
                                  identifiers=identifiers, input_model=input_model)


def classifier_alone_fields(input_model) -> frozenset[str]:
    """The fields of an ``Input`` (of either kind of command) marked ``verify: "alone"``: the parameter
    extractor does not ask for them, as the typed call does not; the classifier decides them."""
    if input_model is None:
        return frozenset()
    return frozenset(p.name for p in option_params(input_model, getattr(input_model, "__name__", "Input"))
                     if p.alone)


def option_sentences(input_model) -> dict[str, tuple[str, ...]]:
    """``{field: sentences}`` the CME parameter extractor appends to the descriptions of an ``Input``'s fields
    (3.4, Descriptions), for each field whose options have a sentence. Empty for a field without options,
    whichever kind of command the ``Input`` belongs to."""
    if input_model is None:
        return {}
    return {p.name: sentences for p in option_params(input_model, getattr(input_model, "__name__", "Input"))
            if (sentences := extraction_sentences(p))}


def extraction_sentences_for(workflow_folderpath: str | None, qualified_name: str | None) -> dict[str, tuple[str, ...]]:
    """``{field: sentences}`` the CME parameter extractor appends to the descriptions of planning
    command *qualified_name*'s fields, for the fields whose options have a sentence. Empty without a
    workflow (3.4, Descriptions)."""
    if not workflow_folderpath or not qualified_name:
        return {}
    command = planning_command(workflow_folderpath, qualified_name)
    if command is None:
        return {}
    return {p.name: sentences for p in command.params if (sentences := extraction_sentences(p))}


def verify_field_names(workflow_folderpath: str) -> tuple[str, ...]:
    """``<command>.<field>`` for each planning command's field that carries ``verify``: the checks
    that need ``JEV_API_KEY`` (the INFO line at load, design 3.9)."""
    commands = resolve_field_sources(workflow_folderpath).planning_commands
    planning = tuple(f"{qname}.{p.name}" for qname, command in sorted(commands.items())
                     for p in command.params if p.verify)
    return planning + tuple(f"{qname}.{p.name}" for qname, command in sorted(_execution_option_commands(
        workflow_folderpath).items()) for p in command.params if p.verify)


def _execution_option_commands(workflow_folderpath: str) -> dict[str, PlanningCommandCommand]:
    """``{command: option_command}`` for each execution command of the workflow whose ``Input`` carries an option."""
    from fastworkflow.observation_offloading.facts import workflow_signatures

    planning = resolve_field_sources(workflow_folderpath).planning_commands
    out = {}
    for name, signature in workflow_signatures(workflow_folderpath).items():
        model = getattr(signature, "Input", None)
        if not isinstance(model, type) or name in planning:
            continue
        command = option_command(model, name)
        if command.params:
            out[name] = command
    return out


def check_arguments(command: PlanningCommandCommand, raw: dict, request: str,
                    verdicts: dict | None = None,
                    vocabulary: WorkflowVocabulary | None = None) -> tuple[dict, list[str]]:
    """The guarded, complete arguments of a planning command call; ``(args, notes)``.

    - a one-element list given for an argument (``["Payroll"]``, or that list written as JSON
      text) stands for its element;
    - a ``str`` or ``int`` argument first goes through the command parameter screen
      (``utils.signatures.screen_parameter_values``, as a command's parameters do): quotes
      or angle brackets around the whole value are removed, a placeholder (``<name>``,
      ``desired-name`` the request does not write) or an out-of-range number (a negative
      ``count``, a declared ``ge``/``le`` bound) is refused (an optional argument takes its
      default, a required one fails the call), and the extractor's "not given" number or a
      made-up extreme counts as missing;
    - a missing or empty argument takes its default; a required one fails the call;
    - an enum must be one of its values, an int an integer, a float a number (the extractor's
      "not given" marker counts as missing), a bool true/false; a list[str] is its elements
      (cleaned; no check that the request writes them: design 2.22), a list written as JSON text
      or comma-separated text;
    - a str argument loses surrounding quotes, and must then be a
      verbatim (case-insensitive; any Unicode dash matches "-") substring of ``request``;
      it takes the request's own spelling. One that is not: an optional argument takes its default, a required one
      fails the call (the planner fills it once more, then falls back to the template path);
    - a str argument inside an identifier match of the request (``command.identifiers``)
      becomes the whole match; a ``subject`` one must name a record of the request
      (``_as_subject``); one that does not is reset or fails the call as above;
    - an ``in_request`` ``str`` must be a verbatim substring of ``request`` (after the
      usual ``str`` normalisation); an ``in_request`` ``int`` must be a number the request
      writes, when it writes any (see the module doc);
    - ``verdicts`` (``{param: (value, probability)}``, from ``check_verdicts``): a sure
      verdict on a ``verify`` parameter replaces its argument, and stands in for a missing
      or invalid enum (for a subject, ``NO_SUBJECT`` resets it, or fails a required one);
      a ``verify: "alone"`` bool ignores any argument given for it: a sure verdict decides it,
      and it keeps its default otherwise;
    - an optional str argument equal to another str argument takes its default.

    An argument that only applies when others take some value is reset afterwards, by the
    planning command's ``validate_extracted_parameters`` (``_validate_arguments``).

    ``notes`` says what a guard changed. A required argument the guards refuse raises
    PlanningCommandInputError (the planner fills the call once more, ``typed_planning_call``); a sure check
    that the request names no such argument raises PlanningCommandCallError.
    """
    from fastworkflow.utils.signatures import is_sentinel_number

    notes: list[str] = []
    args: dict = {}
    # verify "alone": no model fills the field; its default stands until a sure verdict replaces it
    raw = {**raw, **{p.name: p.default for p in command.params if p.alone}}
    folded = _fold(request).lower()
    subjects = _request_subjects(command, request)
    named = {subject.lower() for subject in subjects}
    if vocabulary is None:
        vocabulary = _vocabulary_for(command)
    kinds = frozenset(v.lower() for p in command.params if p.type == "enum" for v in p.values)
    single = {**raw, **{p.name: raw.get(p.name) if p.type == "list[str]" else _single(raw.get(p.name))
                        for p in command.params}}
    screened, refused = _screen(command, single, request)

    def refuse(p: Param, value, why: str) -> None:
        if p.required:
            raise PlanningCommandInputError(f"{p.name}={value!r} {why}")
        notes.append(f"{p.name}={value!r} {why}: default")
        args[p.name] = p.default

    for p in command.params:
        shown = raw.get(p.name)   # the model's argument, as the notes show it
        value = screened.get(p.name, single[p.name])
        compared = shown          # what a changed spelling is noted against
        sure = _confident(verdicts, p.name) if p.verify else None
        if sure is not None and p.type == "enum" and (_empty(value) or _clean_str(value).lower() not in p.values):
            notes.append(f"{p.name}={shown!r} -> {sure[0]!r} (check {sure[1]:.2f})")
            args[p.name] = sure[0]
            continue
        if sure is not None and p.type == "str":
            chosen = None if sure[0] == NO_SUBJECT else sure[0]
            given = "" if _empty(value) or p.name in refused else strip_words(_clean_str(value), p.ignore)
            if (chosen or "").lower() != given.lower():
                notes.append(f"{p.name}={shown!r} -> {chosen or p.default!r} (check {sure[1]:.2f})")
            if chosen is None:
                if p.required:
                    raise PlanningCommandCallError(f"{p.name}={shown!r}: the check finds no {p.name} in the request")
                args[p.name] = p.default
                continue
            value = compared = chosen
        elif p.name in refused:
            refuse(p, shown, refused[p.name])
            continue
        if p.type == "list[str]":
            # no check by type (design 2.46): the elements are cleaned; with in_request, each one the
            # request does not write is dropped, and the request's own spelling of the rest is kept
            items = [text for text in (_clean_str(item) for item in _list_items(value))
                     if text and text != "None"]
            if p.in_request:
                kept = []
                for item in items:
                    exact = _written(item, folded, request)
                    if exact is None:
                        notes.append(f"{p.name}={item!r} is not written in the request: dropped")
                    else:
                        kept.append(exact)
                if items and not kept and p.required:
                    raise PlanningCommandInputError(f"{p.name}={shown!r} is not written in the request")
                items = kept
            if not items and p.required:
                raise PlanningCommandInputError(f"missing required argument {p.name}")
            args[p.name] = items if items else list(p.default)
            continue
        if _empty(value):
            if p.required:
                raise PlanningCommandInputError(f"missing required argument {p.name}")
            args[p.name] = p.default
            continue
        if p.type == "bool":
            if isinstance(value, str):
                if value.strip().lower() not in ("true", "false"):
                    raise PlanningCommandInputError(f"{p.name}={shown!r} is not true or false")
                value = value.strip().lower() == "true"
            args[p.name] = bool(value)
        elif p.type == "int":
            try:
                number = int(str(value).strip())
            except ValueError:
                raise PlanningCommandInputError(f"{p.name}={shown!r} is not an integer") from None
            if is_sentinel_number(number):   # the extractor's "not given" marker or a made-up extreme
                if p.required:
                    raise PlanningCommandInputError(f"missing required argument {p.name}")
                number = p.default
            args[p.name] = number
        elif p.type == "float":
            try:
                number = float(str(value).strip())
            except ValueError:
                raise PlanningCommandInputError(f"{p.name}={shown!r} is not a number") from None
            if not math.isfinite(number) or is_sentinel_number(number):   # the extractor's "not given" marker
                if p.required:
                    raise PlanningCommandInputError(f"missing required argument {p.name}")
                number = p.default
            args[p.name] = number
        elif p.type == "enum":
            text = _clean_str(value).lower()
            if text not in p.values:
                raise PlanningCommandInputError(f"{p.name}={shown!r} is not one of {', '.join(p.values)}")
            args[p.name] = text
        else:
            text = strip_words(_clean_str(value), p.ignore)
            if text.strip() == "None":
                # the missing-value marker, not a value: the argument is absent (as the parameter
                # screen treats it), so a required one is missing and an optional one takes its default
                if p.required:
                    raise PlanningCommandInputError(f"missing required argument {p.name}")
                args[p.name] = p.default
                continue
            if text and text.lower() not in named:
                # the free-text screens: a template token, a pronoun, a word of the workflow's own
                # vocabulary. A value the request names as a record (a subject) always passes.
                if _placeholder_like(text, folded):
                    refuse(p, shown, PLACEHOLDER_WHY)
                    continue
                if is_pronoun_phrase(text):
                    if len(subjects) != 1:
                        refuse(p, shown, PRONOUN_WHY)
                        continue
                    notes.append(f"{p.name}={shown!r} -> {subjects[0]!r} (the request's one subject)")
                    text = subjects[0]
                elif is_kind_word(text, p.name, vocabulary, _kind_selector_options(p, vocabulary)):
                    refuse(p, shown, KIND_WORD_WHY)
                    continue
            if not (p.in_request or p.subject):
                # no request check without the option (design 2.46): the value as the model wrote it,
                # after the free-text screens above, which refuse a placeholder, a pronoun or a word
                # of the workflow's vocabulary whatever the field's options
                args[p.name] = text
                continue
            at = folded.find(_fold(text).lower()) if text else -1
            if at < 0:
                if p.required:
                    raise PlanningCommandInputError(f"{p.name}={shown!r} is not written in the request")
                notes.append(f"{p.name}={shown!r} is not in the request: default")
                args[p.name] = p.default
                continue
            exact = request[at:at + len(text)]
            whole, cut = _whole_identifier(request, exact, command.identifiers)
            if cut and whole is None:
                refuse(p, shown, "is part of more than one identifier the request writes")
                continue
            exact = whole or exact
            if p.subject:
                named, why = _as_subject(request, exact, subjects, kinds)
                if named is None:
                    refuse(p, shown, why)
                    continue
                exact = named
            if exact != str(compared):
                notes.append(f"{p.name}={shown!r} -> {exact!r}")
            args[p.name] = exact
    for p in command.params:
        if not p.in_request or p.type != "int":   # a str's in_request is the text check below, not a number
            continue
        written = _request_numbers(request, args, command)
        if not written or args[p.name] in written:
            continue
        if len(written) == 1:
            (only,) = written
        elif p.default in written:
            only = p.default
        else:
            raise PlanningCommandInputError(f"{p.name}={args[p.name]} is not a number the request writes "
                                 f"({', '.join(map(str, sorted(written)))})")
        notes.append(f"{p.name}={args[p.name]} -> {only} (the number the request writes)")
        args[p.name] = only
    for p in command.params:
        if not p.in_request or p.type != "str":
            continue
        value = args[p.name]
        if _empty(value) or (not p.required and value == p.default):
            continue
        if value not in request:
            shown = raw.get(p.name, value)
            if p.required:
                raise PlanningCommandInputError(f"{p.name}={shown!r} is not written in the request")
            notes.append(f"{p.name}={shown!r} is not written in the request: default")
            args[p.name] = p.default
    for p in command.params:
        sure = _confident(verdicts, p.name) if p.verify and p.type != "str" else None
        if sure is not None and args[p.name] != sure[0]:
            notes.append(f"{p.name}={args[p.name]!r} -> {sure[0]!r} (check {sure[1]:.2f})")
            args[p.name] = sure[0]
    strs = [p for p in command.params if p.type == "str"]
    for p in strs:
        if p.required or args[p.name] == p.default:
            continue
        if any(o is not p and str(args[o.name]).lower() == str(args[p.name]).lower() for o in strs):
            notes.append(f"{p.name}={args[p.name]!r} repeats another argument: default")
            args[p.name] = p.default
    return args, notes


def command_names(available_commands: str) -> frozenset[str]:
    """Command names of the planner's catalog text (its ``- <name>`` lines)."""
    return frozenset(_COMMAND_LINE.findall(available_commands or ""))


def _without_placeholders(text: str) -> str:
    """``text`` without ``<...>`` placeholders and the ``key=`` holding one."""
    cleaned = _BARE_PLACEHOLDER.sub("", _ARG_PLACEHOLDER.sub("", text))
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip(" ,;:")


#: Words of a command name that say what it does, not what it is about ("open", "list", "uid").
GENERIC_COMMAND_WORDS = frozenset("""open list show find get go up by uid id set add run view read all of to for from
with in on at the a an new next first back current""".split())
_WORD = re.compile(r"[a-z]+")
#: Words that say nothing of what a request asks for ("every", "please", "want").
_STOPWORDS = frozenset("""about also and any are been but can did does each else every has have here how into its
just like many more most much need not now only other over please same some such than that the their them then there
these they this those too very want was were what when where which while who whom whose why will with would you
your""".split())


_SUFFIXES = ("ities", "ity", "ions", "ion", "ings", "ing", "ers", "er", "ies", "ive", "ed", "es", "e", "s", "y")


def _stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[:-len(suffix)]
    return word


def _same_stem(a: str, b: str) -> bool:
    """Two words of one stem once a common ending is cut off ("holders"/"hold", "nesting"/"nested",
    "sensitivity"/"sensitive"): one stem starts the other and is at least four letters long."""
    x, y = sorted((_stem(a), _stem(b)), key=len)
    return len(x) >= 4 and y.startswith(x)


def _open_request_words(request: str, expanded: list[str]) -> set[str]:
    """The request's content words that no word of the expansion's text shares a stem with: the
    parts of the request the expansion does not cover."""
    covered = {w for s in expanded for w in _WORD.findall(s.lower())}
    return {w for w in _WORD.findall(request.lower())
            if len(w) >= 3 and w not in _STOPWORDS and not any(_same_stem(w, c) for c in covered)}


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower())
            if len(w) >= 3 and w not in _STOPWORDS and w not in GENERIC_COMMAND_WORDS}


def _supported(command: str, open_words: set[str], text: str = "") -> bool:
    """A word of the command's name, or of ``text`` (what the command says it does: its
    description, its sample utterances), shares a stem with an open request word
    (``GENERIC_COMMAND_WORDS`` and stop words aside)."""
    words = _content_words(command.replace("_", " ")) | _content_words(text)
    return any(_same_stem(w, r) for w in words for r in open_words)


_CATALOG_ENTRY = re.compile(r"^- ([A-Za-z_][A-Za-z0-9_]*)\s*\n((?:[ \t]+\S.*(?:\n|$))*)", re.M)


def command_descriptions(available_commands: str) -> dict[str, str]:
    """``{name: description}`` of the planner's catalog text: each ``- <name>`` line's first
    indented line, cut at its first sentence (what the command does, not the cautions and
    navigation lines the catalog adds after it). A name listed more than once (one command
    in several contexts) gets each description."""
    out: dict[str, list[str]] = {}
    for name, body in _CATALOG_ENTRY.findall(available_commands or ""):
        first = body.strip().splitlines()[0].strip() if body.strip() else ""
        # A ``parameters:`` line is the typed call's listing (typed_planner_commands_text), not a description.
        if first and not first.endswith(":") and not first.startswith("parameters:"):
            out.setdefault(name, []).append(re.split(r"(?<=[.!?])\s", first, maxsplit=1)[0])
    return {name: " ".join(texts) for name, texts in out.items()}


def command_utterances(workflow_folderpath: str) -> dict[str, list[str]]:
    """``{name: plain_utterances}`` of the workflow's commands, from its routing definition
    (a name in several contexts gets the utterances of each). Planning commands count too:
    their utterances come from their ``Signature``, as an execution command's do. Empty when
    the definition cannot be read; never raises."""
    from fastworkflow.command_routing import RoutingRegistry

    out: dict[str, list[str]] = {}
    try:
        crd = RoutingRegistry.get_definition(workflow_folderpath)
        qualified = {q for context, names in crd.contexts.items()
                     for q in (*names, *crd.get_planning_command_names(context))}
        for q in sorted(qualified):
            try:
                meta = crd.command_directory.get_utterance_metadata(q)
            except Exception:  # noqa: BLE001 - a command without utterances adds nothing
                meta = None
            if meta and meta.plain_utterances:
                out.setdefault(q.split("/")[-1], []).extend(meta.plain_utterances)
    except Exception:  # noqa: BLE001 - the command names alone then decide
        from fastworkflow.utils.logging import logger
        logger.warning(f"could not read command utterances of {workflow_folderpath}", exc_info=True)
    return out


def command_texts(available_commands: str, workflow_folderpath: str | None = None) -> dict[str, str]:
    """``{name: words}`` for ``filter_extra_steps``: the words a command says of itself again
    and again: those whose stem recurs in at least two of its texts (its catalog description,
    ``command_descriptions``, and, with ``workflow_folderpath``, each of its plain_utterances).
    A word one utterance happens to use ("check", "whether", "full picture") says nothing
    of what the command is about; a word several do ("invoices" in "show every invoice
    this customer has" and "which invoices are still open") does. A command with one text only
    gives that text's words. A planning command that no execution command shares keeps every
    word of its description too, so its utterances can add words to its hint but never take one away."""
    descriptions = command_descriptions(available_commands)
    sentences = {name: [desc] for name, desc in descriptions.items()}
    planning_only: set[str] = set()
    if workflow_folderpath:
        for name, utterances in command_utterances(workflow_folderpath).items():
            sentences.setdefault(name, []).extend(utterances)
        planning_only = _planning_only_names(workflow_folderpath)
    out = {}
    for name, texts in sentences.items():
        words = [_content_words(t) for t in texts]
        counts: dict[str, int] = {}
        for ws in words:
            for stem in {_stem(w) for w in ws}:
                counts[stem] = counts.get(stem, 0) + 1
        least = 2 if len(texts) > 1 else 1
        found = {w for ws in words for w in ws if counts[_stem(w)] >= least}
        if name in planning_only:
            # A planning command's utterances only add to the words its description gave it:
            # the hint it had before its utterances were read is kept whole.
            found |= _content_words(descriptions.get(name, ""))
        out[name] = " ".join(sorted(found))
    return out


def _planning_only_names(workflow_folderpath: str) -> set[str]:
    """The simple names of the workflow's planning commands that no execution command shares.
    Empty when the routing definition cannot be read."""
    from fastworkflow.command_routing import RoutingRegistry

    try:
        crd = RoutingRegistry.get_definition(workflow_folderpath)
    except Exception:  # noqa: BLE001 - without the definition no name is known to be planning
        return set()
    execution = {q.split("/")[-1] for names in crd.contexts.values() for q in names}
    planning = {q.split("/")[-1] for context in crd.contexts
                for q in crd.get_planning_command_names(context)}
    return planning - execution


def filter_extra_steps(extra, expanded: list[str], names: frozenset[str], planning_command_name: str,
                       request: str = "", notes: list[str] | None = None,
                       texts: dict[str, str] | None = None) -> tuple[list[str], list[str]]:
    """``(kept, dropped)`` of the model's steps_before/steps_after entries.

    An entry first loses its ``<...>`` placeholders (with the ``key=`` holding one): the
    model cannot know a value the run finds, and the agent passes a placeholder on as one
    (``list_logins login_id=<the person's uid>`` becomes ``list_logins``). Dropped: empty
    entries, lines copied from the planning command's template (``_is_template_line``: the expansion
    already writes them for these arguments, or they do not apply to them), entries
    holding a uid-shaped token ``request`` does not write, entries naming
    the planning command itself (the expansion already runs it), entries naming no available command,
    and entries whose every command the expansion already runs. With ``request`` given, an
    entry is also dropped unless one of the commands it names outside the expansion is asked
    for by a part of the request the expansion leaves open: a word of the command's name
    (``GENERIC_COMMAND_WORDS`` aside) shares a stem with a request word that no word of the
    expansion's text shares one with ("show_tags" for "... and its tags" when the expansion
    says nothing of tags; not a navigation or lookup step the model adds around the
    expansion). ``texts`` (``{name: words}``, see ``command_texts``) widens a command's name
    with what the command says of itself, so a request that names it by a synonym its
    description or sample utterances use still asks for it ("every invoice" for
    ``list_bills`` whose utterances say "invoices"). With no command names known only the
    first four rules apply. ``notes`` (when given) gets one line per placeholder removed.
    """
    expanded_cmds = {w for s in expanded for w in _IDENT.findall(s) if w in names}
    planning_command_stem = planning_command_name.rpartition("/")[2]
    lowered = (request or "").lower()
    open_words = _open_request_words(request, expanded) if request else None
    kept, dropped = [], []
    texts = texts or {}
    for entry in extra or []:
        text = str(entry).strip()
        if not text:
            continue
        if _is_template_line(text) or any(uid.lower() not in lowered for uid in _UID.findall(text)):
            dropped.append(text)
            continue
        cleaned = _without_placeholders(text)
        if not cleaned:
            dropped.append(text)
            continue
        if cleaned != text and notes is not None:
            notes.append(f"extra step {text!r} -> {cleaned!r} (placeholder removed)")
        idents = set(_IDENT.findall(cleaned))
        cmds = idents & names
        if (planning_command_name in cleaned
                or (planning_command_stem in idents and planning_command_stem not in names)
                or (names and (not cmds or cmds <= expanded_cmds))):
            dropped.append(text)
        elif names and open_words is not None and not any(_supported(c, open_words, texts.get(c, ""))
                                                             for c in cmds - expanded_cmds):
            dropped.append(text)
        else:
            kept.append(cleaned)
    return kept, dropped


# ---------------------------------------------------------------- the planner call

def _field_desc(p: Param) -> str:
    desc = p.desc or p.name
    if p.type == "bool":
        if p.default is True:
            return f"{desc}. false only when the request explicitly asks against this; otherwise true"
        return f"{desc}. true only when the request explicitly asks for this; otherwise false"
    if p.type == "str":
        desc += ". Copied verbatim from the request"
    return desc + (" (required)" if p.required else f" (default {json.dumps(_json_value(p))})")


def _json_value(p: Param):
    """``p.default`` as JSON reads it (a list[str] default is a tuple)."""
    return list(p.default) if p.type == "list[str]" else p.default


def typed_signature(command: PlanningCommandCommand):
    """The planning command's dspy signature: user_query in; one typed field per parameter, plus
    steps_before / steps_after, out."""
    import dspy

    fields: dict = {"user_query": (str, dspy.InputField())}
    for p in command.params:
        if p.alone:   # verify "alone": the classifier decides it, the planner is not asked
            continue
        kind = {"str": str, "bool": bool, "int": int, "float": float, "list[str]": list[str]}.get(p.type)
        if p.type == "enum":
            kind = Literal[p.values]
        fields[p.name] = (kind, dspy.OutputField(desc=_field_desc(p)))
    fields["steps_before"] = (list[str], dspy.OutputField(
        desc="plain command steps to run before the skill, only for parts of the request it does not cover; "
             "only values the request writes, never a <placeholder> or an id; usually []"))
    fields["steps_after"] = (list[str], dspy.OutputField(
        desc="plain command steps to run after the skill, only for parts of the request it does not cover; "
             "only values the request writes, never a <placeholder> or an id; usually []"))
    instructions = (
        "Plan the user_query by calling one skill command and, only if needed, plain command steps.\n"
        "Fill the skill's parameters from the request: names and values exactly as the request writes them "
        "(never add a word the request does not attach to the name); a boolean is true only when the request "
        "explicitly asks for that part, otherwise false; an enum is what the request is about; leave a "
        "parameter empty to take its default.\n"
        "Code expands the skill into its command steps. steps_before / steps_after are only for parts of the "
        "request the expansion does not cover, each a short sentence naming one available command; never "
        "repeat a command the expansion already runs; usually both are empty.\n\n"
        f"Skill command:\n{command.signature_text()}")
    return dspy.make_signature(fields, instructions)


def _check_options(p: Param, subjects: list[str] = ()) -> tuple[tuple[str, str], ...]:
    if p.type == "str":
        named = [s for s in subjects if s.lower() != NO_SUBJECT][:MAX_SUBJECT_OPTIONS]
        return tuple((s, f"the request names {s} as the {p.name}") for s in named) + (
            (NO_SUBJECT, f"the request names no {p.name}; each name it writes is something else"),)
    if p.type == "bool":
        if p.default is True:
            # fix-jd9o.9: the value form, not "asks against" (a double negative Jev read near coin-flip:
            # 0.47 on mark_as_complete.is_complete; 0.70 -> 0.97 across default-true bools).
            return (("true", f"true: the request does not mention {p.name}, or says or implies it"),
                    ("false", f"false: the request says not to, or says the opposite of {p.name}"))
        return (("true", f"yes: {p.desc or p.name}"),
                ("false", "no: the request does not ask for this, or says not to"))
    means = dict(p.means)
    return tuple((v, means.get(v, v)) for v in p.values)


def _check_one(lm, command: PlanningCommandCommand, p: Param, request: str) -> tuple[object, float]:
    import dspy
    from dspy.experimental import Choice

    if p.type == "bool":
        question = (f"Which value should the argument '{p.name}' take? It means: {p.desc or p.name}."
                    if p.default is True
                    else f"Does the request ask for this: {p.desc or p.name}?")
    elif p.type == "str":
        question = f"Which record named in the request is the {p.name}? {p.desc}".strip()
    else:
        question = f"Which {p.name} does the request mean? {p.desc}".strip()
    options = _check_options(p, _check_subjects(command, request) if p.type == "str" else ())
    signature = dspy.make_signature({
        "request": (str, dspy.InputField(desc="the user's request")),
        "job": (str, dspy.InputField(desc="the job the request is planned as")),
        "value": (Choice[options], dspy.OutputField(desc=question)),
    }, "Decide one argument of the job from what the request says.")
    with dspy.context(lm=lm):
        choice = dspy.Predict(signature)(request=request, job=command.description).value
    value = str(choice.value)
    probability = float((choice.probabilities or {}).get(value, choice.confidence))
    return (value == "true" if p.type == "bool" else value), probability


def check_verdicts(lm, command: PlanningCommandCommand, request: str) -> dict[str, tuple[object, float]]:
    """``{param: (value, probability)}``: ``lm`` (a classifier) decides each ``verify``
    parameter from the request alone, the parameters in parallel. A failed check is left out
    (the planner's argument then stands); never raises."""
    from fastworkflow.utils.logging import logger

    params = [p for p in command.params if p.verify]
    if any(p.type == "str" for p in params) and not _request_subjects(command, request):
        params = [p for p in params if p.type != "str"]   # nothing to choose from
    if lm is None or not params:
        return {}

    def one(p):
        try:
            return p.name, _check_one(lm, command, p, request)
        except Exception:  # noqa: BLE001 - a check is an extra; the planner's argument stands
            logger.warning(f"could not check planning command argument {command.name}.{p.name}", exc_info=True)
            return None
    with ThreadPoolExecutor(len(params)) as pool:
        return dict(r for r in pool.map(one, params) if r is not None)


#: Appended to the request when the planner fills a planning command call again after its rules refused
#: the first one (``typed_planning_call``).
REFUSED_CALL_NOTE = ("(A first call of this skill was refused: {refusal}. Fill the call again from the "
                     "request above, correcting that.)")


@dataclass
class PlanningCommandCallPlan:
    plan: str
    args: dict
    notes: list[str]
    prediction: object
    #: The phases that run, in order (``planning_phases.Phase``, see ``call_phases``): a phased
    #: planning command's (``plan()``), or, for a flat call, one phase named after the planning command with its
    #: steps. And the extra steps the request asks for before and after the planning command's own.
    phases: list | None = None
    steps_before: list[str] = field(default_factory=list)
    steps_after: list[str] = field(default_factory=list)


def no_argument_call(command: PlanningCommandCommand) -> PlanningCommandCallPlan:
    """The call of a planning command with no parameters: nothing for a typed planning call to fill, so no
    model call is made. Its phases are ``call_phases``'s (always at least one); it has no
    steps before or after, as those are steps the request needs around a call with arguments."""
    return PlanningCommandCallPlan(plan="", args={}, notes=[], prediction=None,
                         phases=call_phases(command, {}))


def typed_planning_call(command: PlanningCommandCommand, request: str, available_commands: str,
                    check_lm=None, workflow=None) -> PlanningCommandCallPlan:
    """Plan ``request`` as a call of ``command``: the typed call, the guards, the expansion.

    A planning command with no parameters is not planned with a model call (``no_argument_call``).

    Runs under the caller's dspy context (the planner LM and adapter); ``available_commands``
    reaches the adapter as extra input, as for the text planner. With ``check_lm`` (a
    classifier LM) the planning command's ``verify`` parameters are decided again from the request,
    alongside the typed call (``check_verdicts``). A planning command file's
    ``validate_extracted_parameters`` then sees the checked arguments (``_validate_arguments``;
    ``workflow`` is passed to it), and the ``Input``'s rules (``check_input``). When the planning command's
    rules or the argument guards refuse the arguments (PlanningCommandInputError: a made-up name, a
    missing or invalid value), the planner fills the call once more, with the
    refusal after the request (``REFUSED_CALL_NOTE``); the guards still read the request alone.
    Raises on any other failure, or a second refusal (the caller falls back to the template
    path).
    """
    if not command.params:
        return no_argument_call(command)
    import dspy

    def typed_call(query: str):
        return dspy.ChainOfThought(typed_signature(command))(
            user_query=query, available_commands=available_commands)

    def checked(prediction) -> tuple[dict, list[str]]:
        raw = {p.name: getattr(prediction, p.name, None) for p in command.params}
        args, notes = check_arguments(command, raw, request, verdicts)
        return _validate_arguments(command, args, request, workflow, notes), notes

    with ThreadPoolExecutor(1) as pool:
        checks = pool.submit(check_verdicts, check_lm, command, request)
        prediction = typed_call(request)
        verdicts = checks.result()
    try:
        args, notes = checked(prediction)
    except PlanningCommandInputError as refused:
        prediction = typed_call(f"{request}\n\n{REFUSED_CALL_NOTE.format(refusal=refused)}")
        args, notes = checked(prediction)
        notes.insert(0, f"first call refused, planned again: {refused}")
    steps = expand(command, args)
    names = command_names(available_commands)
    texts = command_texts(available_commands, getattr(workflow, "folderpath", None))
    before, dropped_before = filter_extra_steps(getattr(prediction, "steps_before", None), steps, names,
                                                command.name, request, notes, texts)
    after, dropped_after = filter_extra_steps(getattr(prediction, "steps_after", None), steps, names,
                                              command.name, request, notes, texts)
    notes += [f"dropped extra step {s!r}" for s in dropped_before + dropped_after]
    phases = call_phases(command, args)
    return PlanningCommandCallPlan(plan=number(before + steps + after), args=args, notes=notes, prediction=prediction,
                         phases=phases, steps_before=list(before), steps_after=list(after))
