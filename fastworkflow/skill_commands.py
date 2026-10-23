r'''Skills as parameterized commands.

A skill is one Python file, ``_skills/<name>.py``, written with the abstractions of a
command file: a ``Signature`` whose docstring describes the job (what the skill matcher and
the text planner read) and whose pydantic ``Input`` declares typed parameters, and a
``class PlanGenerator`` (as a command has a ``ResponseGenerator``) whose methods write the
skill's steps for one set of arguments (``fastworkflow.skill``). The planner fills the
parameters from the request (one typed dspy call), the arguments are checked, and the
``PlanGenerator`` expands them into the numbered plan. A skill without parameters and
without phases gives fixed steps the text planner adapts (see ``workflow_agent.load_skills``);
one without parameters that has phases is a skill call with no arguments (its one plan).

The ``_skills`` folder is laid out as the root of ``_commands`` is: a skill is the file
``_skills/<name>.py`` and the file stem is the skill's name, so it is a Python identifier
(snake_case: ``show_record.py``). A ``.py`` file whose name starts with ``_`` is a helper the
skill files import (``from ._kinds import FIND``), not a skill. A subfolder is reserved for a
skill context (as ``_commands/<Context>/`` is); its files are not loaded yet (it is logged and
skipped). ``_skills/_split.py`` is not a skill either: it configures the split stage (the
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

A skill file::

    from typing import Literal

    from pydantic import BaseModel, Field

    FIND = {"group": "find_group", "unit": "find_unit"}


    class Signature:
        """Show one named group or unit and its members. One job per record."""

        class Input(BaseModel):
            name: str = Field(description="the record's name exactly as the request writes it",
                              json_schema_extra={"strip": ["group", "unit"], "subject": True})
            kind: Literal["group", "unit"] = Field(
                description="what the request is about",
                json_schema_extra={"verify": True,
                                   "means": {"group": "a group of people", "unit": "an organisation unit"}})
            person: str = Field(default="", description="one named person to check for; empty for everyone",
                                json_schema_extra={"subject": True})
            include_risk: bool = Field(default=False, description="the request asks how risky it is",
                                       json_schema_extra={"verify": True})
            count: int = Field(default=2, ge=0, le=20, description="how many members to open",
                               json_schema_extra={"in_request": True})

        plain_utterances = ["who is in the Payroll group", "is Ann Lee in Billing"]

        @staticmethod
        def validate_extracted_parameters(workflow, command, cmd_parameters, command_text=None):
            if cmd_parameters.include_risk and cmd_parameters.person:
                return False, "the risk is assessed for the whole record, not for one person"
            return True, ""


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
  line). ``Input`` (a pydantic ``BaseModel``) declares the parameters in order; a skill
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
  argument screen (see ``check_arguments``). The skill options go in ``json_schema_extra``
  (each optional; unknown keys fail the load):

  - ``strip`` (a ``str``; a list of words): removed from either end of the argument, with a
    possessive ``'s`` before a trailing one: a kind word the model appends to a name ("HR
    feed" -> "HR").
  - ``subject: True`` (a ``str``): the argument names a record of the request (a person, a
    group, a product, a code). Besides being written in the request it must hold one of the
    request's subjects whole (``split_merge.request_subjects``, with the workflow's identifier
    patterns) or a name-like word (capitalised and not a common word, or holding a digit or
    ``_``) or be a name written in lower case (two or more uncommon words cutting no longer
    run of them: "ann lee"), and cut no subject: "I" of "Before I approve" and "Payroll" of "Payroll_Approver"
    are refused (an optional argument takes its default; a required one fails the call). One
    cut short, mid-word or at a space, at the start or end of exactly one subject becomes
    that subject ("Payroll_Admin" of "Payroll_Administrator", "Ann" of "Ann Lee").
  - ``verify: True`` (a ``bool``, an enum or a ``subject`` str): a classifier model decides
    the argument again from the request alone, when the planner path has one
    (``plan_skill_call``'s ``check_lm``); its answer replaces the planner's when it is at least
    ``VERIFY_THRESHOLD`` sure. It reads the field's ``description``, and an enum's ``means``:
    a dict giving each value's meaning (``{"group": "a group, people are its members"}``).
    For a subject it chooses among the request's subjects (at most ``MAX_SUBJECT_OPTIONS``)
    or ``NO_SUBJECT`` ("none": the default, or a failed call for a required parameter); a
    request naming no subject is not checked.
  - ``in_request: True`` (an ``int``): when the request writes numbers (digits or the words
    zero to twenty) outside the text of the other arguments, the argument must be one of
    them; otherwise it takes the only one written, or its default when that is written, and
    with several others the call fails ("the first two" never runs as 3).

- ``PlanGenerator`` (required): its public methods are the skill's phases, in order, each
  returning its steps as a list of str, one per step (empty ones are dropped); see
  ``fastworkflow.skill`` for the rules. The methods build text only: they must not call a
  backend or read state. A repeated part is a ``for`` loop over ``range()``, bounded by the
  int field's ``le``; steps shared by several skills go in a helper module
  (``_skills/_<name>.py``). Lookup tables are plain module-level dicts. A skill whose one
  phase reads and fills no facts runs flat (one plan); one without parameters whose one
  method takes no ``input`` gives fixed steps.
- ``plain_utterances`` (optional, with or without params): example requests for the
  skill, written the way people ask for that job. The skill matcher shows the first
  ``workflow_agent.MATCHER_EXAMPLES`` of them after the description ("e.g. ..."); a skill
  call also keeps them on its command (``SkillCommand.plain_utterances``). Write
  them generically: they are in every matcher prompt.
- ``template`` (optional): the template text of the skill, shown to the planner in the
  skill call's "expands to:" and used as the text planner's template when a skill call
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
  (``fastworkflow.skill_phases``; the loader builds its ``Phase`` list,
  ``skill.planned``). At load the phases are built on the placeholder inputs: their
  parameters must bind (``_checked_plan``), their ``"from"`` must be well formed, and each
  method's template call must succeed (``SkillFormatError`` otherwise); every command step
  must name a workflow command and only parameters of its ``Signature.Input``
  (``check_step_commands``: the skill is skipped at load, logged as an error, and
  ``fastworkflow skill preview --check`` fails); a ``"from"`` naming a command or field the
  workflow lacks is a logged warning (``check_phase_sources``), not a load failure.
- ``validate_extracted_parameters`` (optional, the signature of a command's): called with
  the checked arguments as an ``Input`` (``validate_call``); ``(False, message)`` refuses the
  call (the planner gets the message and one more try, see the rules below), an exception
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
  message)``, is a ``SkillInputError``: the planner is given the message and fills the call
  once more (``plan_skill_call``); a second refusal falls back. So is a required argument
  the guards of ``check_arguments`` refuse (missing, invalid, not written in the request).
  They also run on the placeholder combinations (request ``""``): one they refuse is left out
  of the template.

The file is imported as a command file is (``python_utils.get_module`` under the workflow's
package, so package-relative imports work). A file that fails to import, or whose declaration
is malformed (a ``SkillFormatError`` naming the file), fails its load: the skill is logged as
an error and skipped, the workflow's other skills still load, and ``fastworkflow skill
preview`` fails naming it.

Any ``str`` argument that lies inside a match of one of the workflow's identifier patterns
(``identifier_patterns`` in ``_skills/_split.py``, e.g. ``\bacct_[0-9]+``) without being all of it becomes the
whole match ("4421" of "acct_4421"); inside several, it is refused.

The model's ``steps_before`` / ``steps_after`` lose any ``<...>`` placeholder (and the
``key=`` that holds one); a step holding a uid-shaped token the request does not write is
dropped, and so is a step whose commands outside the expansion no open part of the request
asks for, by a word of the command's name, catalog description or plain_utterances
(``filter_extra_steps``, ``command_texts``).

A skill call's ``str`` and ``int`` arguments first go through the screen a command's
parameters go through (``utils.signatures.screen_parameter_values``: quotes and brackets
removed, placeholders and out-of-range numbers refused), and a str argument is looked up in
the request with Unicode dashes folded to "-" (see ``check_arguments``).
'''
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from types import UnionType
from typing import Callable, Literal, Optional, Union, get_args, get_origin

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


class SkillFormatError(ValueError):
    """A malformed skill file: its ``Signature`` or an ``Input`` field."""


class SkillCallError(ValueError):
    """A skill call that cannot be used (bad argument, failed guard); the planner falls back."""


class SkillInputError(SkillCallError):
    """The arguments are refused: a pydantic validator of the skill's ``Input`` raised,
    ``validate_extracted_parameters`` returned ``(False, message)``, or a guard of
    ``check_arguments`` refused one of the planner's arguments (missing, not a valid value, not
    written in the request). The message says what to fix; ``plan_skill_call`` gives it to the
    planner for one more typed call."""


# ---------------------------------------------------------------- the parsed skill

@dataclass(frozen=True)
class Param:
    name: str
    type: str  # "str" | "bool" | "int" | "enum"
    values: tuple[str, ...] = ()
    default: object = None
    required: bool = False
    desc: str = ""
    strip: tuple[str, ...] = ()
    verify: bool = False
    means: tuple[tuple[str, str], ...] = ()
    in_request: bool = False
    subject: bool = False

    def type_text(self) -> str:
        return "|".join(self.values) if self.type == "enum" else self.type


@dataclass(frozen=True)
class SkillCommand:
    """A skill called with typed arguments (a skill with parameters, or with phases: its
    ``params`` may then be empty): its signature and its ``steps()``."""

    name: str
    description: str
    params: tuple[Param, ...]
    path: str = field(default="", compare=False)
    #: The workflow's identifier patterns (``workflow_agent.identifier_patterns``), set at load.
    identifiers: tuple = field(default=(), compare=False)
    #: The skill's pydantic ``Input``. Skill-call arguments are screened through it like a
    #: command's parameters.
    input_model: type | None = field(default=None, compare=False)
    #: The skill file's ``steps(input)``.
    steps_fn: Callable | None = field(default=None, compare=False)
    #: The template text (``Signature.template``, or ``steps()`` rendered on placeholder
    #: inputs).
    template: tuple[str, ...] = field(default=(), compare=False)
    #: ``Signature.validate_extracted_parameters``, or None.
    validate: Callable | None = field(default=None, compare=False)
    #: ``Signature.plain_utterances``: example requests for the skill.
    plain_utterances: tuple[str, ...] = field(default=(), compare=False)
    #: The skill file's ``plan(input) -> list[Phase]`` (``fastworkflow.skill_phases``), or None.
    plan_fn: Callable | None = field(default=None, compare=False)
    #: The phases' ``output`` models seen when ``plan()`` ran on the placeholder inputs at load.
    phase_outputs: tuple[type, ...] = field(default=(), compare=False)

    def param(self, name: str) -> Param | None:
        return next((p for p in self.params if p.name == name), None)

    def signature_text(self) -> str:
        """``name(p: type = default, ...)`` with the description, parameter docs and steps."""
        parts = []
        for p in self.params:
            s = f"{p.name}: {p.type_text()}"
            if not p.required:
                s += f" = {json.dumps(p.default)}"
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
        raise SkillFormatError(f"{where}: {what} has unknown key(s) {', '.join(unknown)}")


def _parse_param(name: str, spec, where: str) -> Param:
    if not re.fullmatch(_NAME, name) or name in RESERVED_NAMES:
        raise SkillFormatError(f"{where}: bad parameter name {name!r}")
    if not isinstance(spec, dict):
        raise SkillFormatError(f"{where}: parameter {name!r} must be a block with a type")
    _check_keys(spec, {"type", "default", "desc", "strip", "verify", "means", "in_request",
                       "subject"},
                f"parameter {name!r}", where)
    type_text = spec.get("type")
    if not isinstance(type_text, str):
        raise SkillFormatError(f"{where}: parameter {name!r} needs a type")
    values: tuple[str, ...] = ()
    m = re.fullmatch(r"enum\[(.*)\]", type_text.strip())
    if m:
        values = tuple(v.strip() for v in m.group(1).split(","))
        if not values or any(not re.fullmatch(r"[\w-]+", v) for v in values) or len(set(values)) != len(values):
            raise SkillFormatError(f"{where}: parameter {name!r} has a bad enum {type_text!r}")
        kind = "enum"
    elif type_text.strip() in ("str", "bool", "int"):
        kind = type_text.strip()
    else:
        raise SkillFormatError(f"{where}: parameter {name!r} has unknown type {type_text!r} "
                               "(str, bool, int or enum[...])")
    desc = spec.get("desc", "")
    if not isinstance(desc, str):
        raise SkillFormatError(f"{where}: parameter {name!r} desc must be text")
    raw_default = spec.get("default")
    if isinstance(raw_default, dict):
        raise SkillFormatError(f"{where}: parameter {name!r} default must be a value")
    required = raw_default is None and kind != "bool"
    default: object = None
    if kind == "bool":
        if raw_default not in (None, "false"):
            raise SkillFormatError(f"{where}: bool parameter {name!r} must default to false "
                                   "(the model sets it true only when the request asks)")
        default = False
    elif raw_default is not None:
        if kind == "int":
            if not re.fullmatch(r"-?\d+", raw_default):
                raise SkillFormatError(f"{where}: parameter {name!r} default {raw_default!r} is not an int")
            default = int(raw_default)
        elif kind == "enum":
            if raw_default not in values:
                raise SkillFormatError(f"{where}: parameter {name!r} default {raw_default!r} is not one of its values")
            default = raw_default
        else:
            default = raw_default
    strip: tuple[str, ...] = ()
    if "strip" in spec:
        if kind != "str" or not isinstance(spec["strip"], str):
            raise SkillFormatError(f"{where}: only a str parameter takes strip words ({name!r})")
        strip = tuple(w.strip() for w in spec["strip"].split(",") if w.strip())
    subject = _flag(spec, "subject", name, where)
    if subject and kind != "str":
        raise SkillFormatError(f"{where}: only a str parameter takes subject ({name!r})")
    verify = _flag(spec, "verify", name, where)
    if verify and kind not in ("bool", "enum") and not subject:
        raise SkillFormatError(f"{where}: only a bool, enum or subject parameter takes verify ({name!r})")
    means: tuple[tuple[str, str], ...] = ()
    if "means" in spec:
        block = spec["means"]
        if kind != "enum" or not isinstance(block, dict) or not block:
            raise SkillFormatError(f"{where}: means of {name!r} must be a block of its enum values")
        for key, text in block.items():
            if key not in values or not isinstance(text, str):
                raise SkillFormatError(f"{where}: means of {name!r}: {key!r} is not a value of it, or not text")
        means = tuple((v, block[v]) for v in values if v in block)
    in_request = _flag(spec, "in_request", name, where)
    if in_request and kind != "int":
        raise SkillFormatError(f"{where}: only an int parameter takes in_request ({name!r})")
    return Param(name=name, type=kind, values=values, default=default, required=required,
                 desc=desc, strip=strip, verify=verify, means=means, in_request=in_request,
                 subject=subject)


def _flag(spec: dict, key: str, name: str, where: str) -> bool:
    value = spec.get(key, "false")
    if value not in ("true", "false"):
        raise SkillFormatError(f"{where}: parameter {name!r} {key} must be true or false")
    return value == "true"


# ---------------------------------------------------------------- skill files (_skills/<name>.py)

#: Keys a skill file's ``Input`` field may set in ``json_schema_extra``.
SKILL_FIELD_KEYS = frozenset({"strip", "subject", "verify", "means", "in_request"})


def _spec_from_field(name: str, info, where: str) -> dict:
    """The spec of one ``Input`` field (its type, default, description and skill options, as
    text), for ``_parse_param``."""
    from pydantic_core import PydanticUndefined

    annotation = info.annotation
    optional = get_origin(annotation) in (Union, UnionType) and type(None) in get_args(annotation)
    if optional:
        if annotation != Optional[str] or info.default is not None:
            raise SkillFormatError(f"{where}: field {name!r}: Optional is for an argument the request may "
                                   "leave out, Optional[str] = None; give another type a default")
        annotation = str
    if annotation is str or annotation is bool or annotation is int:
        type_text = annotation.__name__
    elif get_origin(annotation) is Literal:
        values = get_args(annotation)
        if not values or not all(isinstance(v, str) for v in values):
            raise SkillFormatError(f"{where}: field {name!r}: a Literal of text values only")
        type_text = f"enum[{', '.join(values)}]"
    else:
        raise SkillFormatError(f"{where}: field {name!r} has type {annotation!r}; a skill field is "
                               "str, bool, int or Literal[...] of text values")
    spec: dict = {"type": type_text}
    if info.description:
        spec["desc"] = info.description
    if info.default_factory is not None:
        raise SkillFormatError(f"{where}: field {name!r}: give a plain default, not a default_factory")
    if optional:
        spec["default"] = ""
    elif info.default is not PydanticUndefined:
        default = info.default
        if default is None or isinstance(default, (dict, list, tuple)):
            raise SkillFormatError(f"{where}: field {name!r} default must be a value of its type "
                                   "(leave the default out for a required parameter)")
        spec["default"] = ("true" if default else "false") if isinstance(default, bool) else str(default)
    extra = info.json_schema_extra
    if extra is None:
        return spec
    if not isinstance(extra, dict):
        raise SkillFormatError(f"{where}: field {name!r} json_schema_extra must be a dict")
    if "requires" in extra:
        raise SkillFormatError(f"{where}: field {name!r}: requires is not a skill option; reset the "
                               "argument (or refuse the call) in Signature.validate_extracted_parameters")
    _check_keys(extra, set(SKILL_FIELD_KEYS), f"field {name!r} json_schema_extra", where)
    for key in ("subject", "verify", "in_request"):
        if key in extra:
            if not isinstance(extra[key], bool):
                raise SkillFormatError(f"{where}: parameter {name!r} {key} must be true or false")
            spec[key] = "true" if extra[key] else "false"
    if "strip" in extra:
        words = extra["strip"]
        if isinstance(words, str) or not isinstance(words, (list, tuple)) or \
                not all(isinstance(w, str) for w in words):
            raise SkillFormatError(f"{where}: parameter {name!r} strip must be a list of words")
        spec["strip"] = ", ".join(words)
    if "means" in extra:
        spec["means"] = extra["means"]
    return spec


@dataclass(frozen=True)
class SkillFile:
    """What a skill file declares: its description and either a parsed command (a skill with
    params or phases) or its fixed steps (a skill without either, ``command`` None)."""

    description: str
    steps: tuple[str, ...]
    command: SkillCommand | None
    #: ``Signature.plain_utterances``: example requests for the skill (with or without params).
    plain_utterances: tuple[str, ...] = ()


def _absent_as_none(command: SkillCommand, args: dict) -> dict:
    """*args* as the skill file's code reads them: an ``Optional[str] = None`` field's
    absent argument (``""``) is None."""
    fields = command.input_model.model_fields
    return {n: None if v == "" and n in fields and fields[n].default is None else v
            for n, v in args.items()}


def input_instance(command: SkillCommand, args: dict):
    """The skill's ``Input`` for *args*, unvalidated, as ``steps()``, ``plan()`` and the hook
    get it (``_absent_as_none``)."""
    return command.input_model.model_construct(**_absent_as_none(command, args))


def _call_steps(fn: Callable, instance, where: str) -> list[str]:
    import inspect

    takes_input = bool(inspect.signature(fn).parameters)
    lines = fn(instance) if takes_input else fn()
    if isinstance(lines, str) or not isinstance(lines, (list, tuple)) or \
            not all(isinstance(line, str) for line in lines):
        raise SkillCallError(f"{where}: steps() must return a list of str")
    return [line.strip() for line in lines if line.strip()]


#: A skill's input combinations are each tried up to this many; past it, a bounded set of
#: them (``input_grid``). Shared by ``placeholder_template`` and the skill preview.
MAX_GRID = 4096
#: Skill files warned about once in this process (past ``MAX_GRID``).
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
    int takes its default (or 1 and 0 when that is 0), an enum each value (its default first).
    """
    if p.type == "str":
        return (f"<{p.name}>",) if p.required or p.default != "" else (f"<{p.name}>", "")
    if p.type == "bool":
        return (True, False)
    if p.type == "int":
        return (p.default,) if p.default else (1, 0)
    if p.default is None:
        return tuple(p.values)
    return (p.default,) + tuple(v for v in p.values if v != p.default)


def _template_grid(command: SkillCommand) -> list[dict]:
    """The argument sets ``placeholder_template`` renders: the base first, then the other
    combinations ``input_grid`` tries (past ``MAX_GRID``, a bounded set, logged once per
    skill file), leaving out a set the skill's ``validate_extracted_parameters`` changes or
    refuses (``validate_call`` never passes one on to ``steps()``; the set it changes to is
    another of the grid, or none)."""
    values = [_template_values(p) for p in command.params]
    sizes = [len(v) for v in values]
    total = 1
    for size in sizes:
        total *= size
    if total > MAX_GRID and command.path not in _warned_grids:
        from fastworkflow.utils.logging import logger

        _warned_grids.add(command.path)
        logger.warning(f"skill {command.name!r}: {total} combinations of its inputs, past {MAX_GRID}: "
                       "its template is rendered from its defaults and each one or two changes of them")
    # past MAX_GRID the grid is centred on the defaults (a bool false, a str empty when that is
    # its default), the inputs most calls have
    first = None if total <= MAX_GRID else tuple(
        v.index(p.default) if not p.required and p.default in v and p.type != "int" else 0
        for p, v in zip(command.params, values))
    grid = [{p.name: v[i] for p, v, i in zip(command.params, values, combo)} for combo in input_grid(sizes, first)]
    out, seen = [], set()
    for env in grid:
        key = tuple(env.items())
        if key in seen:
            continue
        try:
            if command.validate is not None and _run_validate(command, env, "", None) != env:
                continue
            check_input(command, env, "", None)
        except SkillCallError:
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
    if p.type == "int":
        return f"not {p.name}" if 0 in allowed else p.name
    shown = [v for v in p.values if v in allowed]
    if len(p.values) > 2 and len(shown) == len(p.values) - 1:
        return f"{p.name} is not {next(v for v in p.values if v not in allowed)}"
    return f"{p.name} is " + " or ".join(shown)


def _condition_text(command: SkillCommand, present: set, grid: list[dict]) -> str | None:
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

    def __init__(self, command: SkillCommand, grid: list[dict]):
        self.size = len(grid)
        self.full = (1 << self.size) - 1
        self.cache: dict[int, str | None] = {}
        index = {tuple(env.items()): i for i, env in enumerate(grid)}
        # per parameter: (i, j) for each grid point i and each point j that is i with only
        # that parameter's value changed
        self.pairs = [(p, [(i, j) for i, env in enumerate(grid) for v in _template_values(p)
                           if v != env[p.name] for j in [index.get(tuple({**env, p.name: v}.items()))]
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


def placeholder_template(command: SkillCommand) -> list[str]:
    """A skill file's template text when it declares none: ``steps()`` rendered on placeholder
    inputs.

    ``steps()`` runs on every combination of the inputs ``_template_values`` gives (a str
    reads ``<name>``, and "" when that is its default; a bool true and false; an int its
    default, or 1 and 0; an enum each value; past ``MAX_GRID`` combinations, those
    ``input_grid`` covers), and the runs are merged in order. A line not
    every run writes ends with ``(only when ...)``, the condition on the arguments under which
    it is written (``kind is group``, ``person is given``, ``not flag``); lines that differ only
    by an enum's value become one (``<find_group or find_unit, by kind> query=<name>``). A
    repeated part shows its steps for the int's default; declare ``Signature.template`` when
    the planner needs more than that.

    Raises SkillFormatError when ``steps()`` fails on the base inputs (the first of each, or
    the first combination ``validate_extracted_parameters`` leaves in).
    """
    from fastworkflow.skill import TemplateError

    grid, runs = [], []
    for env in _template_grid(command):
        try:
            lines = _call_steps(command.steps_fn, input_instance(command, env), command.path)
        except TemplateError as exc:
            raise SkillFormatError(f"{command.path}: {exc}") from None
        except Exception as exc:  # noqa: BLE001 - a set steps() refuses is left out of the template
            if not grid and not runs:
                raise SkillFormatError(f"{command.path}: steps() fails on placeholder inputs "
                                       f"({type(exc).__name__}: {exc}); declare Signature.template") from None
            continue
        grid.append(env)
        runs.append(lines)
    if not grid:
        raise SkillFormatError(f"{command.path}: validate_extracted_parameters changes or refuses every "
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
            # the facts' condition (skill.PlanGenerator) and the inputs': one "only when"
            both = f"({condition})" if " or " in condition else condition
            out.append(f"{line[:-1]} and {both})")
        else:
            out.append(f"{line} (only when {condition})" if condition else line)
    return out


#: How a phase's template lines end when its method tested facts (``fastworkflow.skill``).
_FACT_CONDITION = " (only when "
#: How a template line ends when no condition on the arguments says when it is written.
_SOME_ARGUMENTS = " (only for some of the arguments)"


def _is_template_line(text: str) -> bool:
    """Whether *text* carries a condition only the skill's TEMPLATE writes (``(only when
    find.uids)``, ``(only for some of the arguments)``): the planner copied a template line."""
    return _FACT_CONDITION in text or _SOME_ARGUMENTS in text


def command_from_module(name: str, module, path: str) -> SkillFile:
    """The skill a skill file (``_skills/<name>.py``, already imported) defines.

    Raises SkillFormatError for a malformed declaration (see the module doc).
    """
    import inspect

    from pydantic import BaseModel

    signature = getattr(module, "Signature", None)
    if not inspect.isclass(signature):
        raise SkillFormatError(f"{path}: a skill file needs a class Signature")
    description = " ".join(inspect.cleandoc(signature.__doc__ or "").split())
    if not description:
        raise SkillFormatError(f"{path}: Signature needs a docstring (the skill's description)")
    from fastworkflow import skill as skill_api
    from fastworkflow.skill_phases import PhaseError

    plan_cls = getattr(module, "PlanGenerator", None)
    if not inspect.isclass(plan_cls) or not skill_api.phase_methods(plan_cls):
        raise SkillFormatError(f"{path}: a skill file needs a class PlanGenerator with at "
                               "least one phase method (fastworkflow.skill)")
    try:
        steps_fn, plan_fn = skill_api.skill_parts(plan_cls)
    except PhaseError as exc:
        raise SkillFormatError(f"{path}: {exc}") from None
    return skill_from_parts(name, signature, path, steps_fn=steps_fn, plan_fn=plan_fn)


def skill_from_parts(name: str, signature, path: str, *, steps_fn: Callable | None = None,
                     plan_fn: Callable | None = None) -> SkillFile:
    """The skill a ``Signature`` and its behaviour define: a flat ``steps_fn(input)`` or a
    ``plan_fn(input) -> list[Phase]`` (or both: a plan that may return no phases runs flat,
    from ``steps_fn``). ``command_from_module`` builds them from the file's ``PlanGenerator``
    class; the runtime reads only these. Raises SkillFormatError."""
    import inspect

    from pydantic import BaseModel

    description = " ".join(inspect.cleandoc(signature.__doc__ or "").split())
    if not description:
        raise SkillFormatError(f"{path}: Signature needs a docstring (the skill's description)")
    own_steps = steps_fn is not None
    if steps_fn is None:
        if plan_fn is None:
            raise SkillFormatError(f"{path}: a skill needs its steps or its phases")
        steps_fn = _concatenated_steps_fn(plan_fn)
    input_model = getattr(signature, "Input", None)
    if input_model is None:
        input_model = type(f"{signature.__name__}Input", (BaseModel,), {})
    if not (inspect.isclass(input_model) and issubclass(input_model, BaseModel)):
        raise SkillFormatError(f"{path}: Signature.Input must be a pydantic BaseModel")
    utterances = getattr(signature, "plain_utterances", ())
    if isinstance(utterances, str) or not all(isinstance(u, str) for u in utterances):
        raise SkillFormatError(f"{path}: plain_utterances must be a list of str")
    validate = getattr(signature, "validate_extracted_parameters", None)
    if validate is not None and not callable(validate):
        raise SkillFormatError(f"{path}: validate_extracted_parameters must be a function")

    params = {n: _parse_param(n, _spec_from_field(n, info, path), path)
              for n, info in input_model.model_fields.items()}
    if not params and plan_fn is None:
        try:
            lines = _call_steps(steps_fn, input_model(), path)
        except SkillCallError as exc:
            raise SkillFormatError(str(exc)) from None
        if not lines:
            raise SkillFormatError(f"{path}: steps() returns no steps")
        return SkillFile(description=description, steps=tuple(lines), command=None,
                         plain_utterances=tuple(utterances))

    command = SkillCommand(name=name, description=description, params=tuple(params.values()), path=path, input_model=input_model, steps_fn=steps_fn, validate=validate,
                           plain_utterances=tuple(utterances), plan_fn=plan_fn)
    if plan_fn is not None:
        command = replace(command, phase_outputs=_checked_plan(command, own_steps=own_steps))
    declared = getattr(signature, "template", None)
    if declared is not None:
        if isinstance(declared, str) or not all(isinstance(t, str) for t in declared):
            raise SkillFormatError(f"{path}: Signature.template must be a list of str")
        template = list(declared)   # as written: it is text for the planner
    else:
        template = placeholder_template(command)
    if not template:
        raise SkillFormatError(f"{path}: a skill needs steps")
    return SkillFile(description=description, steps=tuple(template), command=replace(command, template=tuple(template)),
                     plain_utterances=tuple(utterances))


def _concatenated_steps_fn(plan_fn: Callable) -> Callable:
    """``steps(input)`` of a skill file that defines only ``plan()``: the phases joined."""
    from fastworkflow.skill_phases import concatenated_steps

    def steps(input):
        return concatenated_steps(plan_fn(input))

    return steps


def _checked_plan(command: SkillCommand, own_steps: bool = True) -> tuple[type, ...]:
    """Run ``plan()`` on the placeholder inputs (``_template_grid``) and check what it returns.

    A flat result (``None`` or ``[]``, ``skill_phases.is_flat_plan``) is fine when the
    file defines its own ``steps()`` (*own_steps*), which that call then runs.
    Each other result must be a list of ``Phase`` whose parameters can be bound
    (``skill_phases.check_bindings``) and whose ``output`` models' ``"from"`` are
    well formed (``facts.fact_sources``); a parameter defaulting to None must bind (name an
    earlier phase with an output) in at least one of the results. Returns the output models seen. Whether the
    sources name real command fields is checked against the workflow later
    (``check_phase_sources``), as a warning. Raises SkillFormatError.
    """
    from fastworkflow.observation_offloading.facts import FactSourceError, fact_sources
    from fastworkflow.skill_phases import PhaseError, active_phases, check_bindings, is_flat_plan

    models: list[type] = []
    binds: dict[tuple[str, str, str], bool] = {}
    ran = 0
    for env in _template_grid(command):
        try:
            result = command.plan_fn(input_instance(command, env))
        except Exception as exc:  # noqa: BLE001 - a set plan() refuses, as for steps()
            if not ran:
                raise SkillFormatError(f"{command.path}: plan() fails on placeholder inputs "
                                       f"({type(exc).__name__}: {exc})") from None
            continue
        ran += 1
        if is_flat_plan(result):
            if not own_steps:
                raise SkillFormatError(
                    f"{command.path}: plan() returns no phases for some inputs (run flat), so "
                    "the skill file needs its own steps(input)")
            continue
        try:
            phases = active_phases(result, f"{command.path}: plan()")
            for key, bound in check_bindings(phases, f"{command.path}: plan()").items():
                binds[key] = binds.get(key, False) or bound
        except PhaseError as exc:
            raise SkillFormatError(str(exc)) from None
        if not phases:
            raise SkillFormatError(f"{command.path}: plan() returns no phase")
        for phase in phases:
            if phase.output is not None and phase.output not in models:
                try:
                    fact_sources(phase.output)
                except FactSourceError as exc:
                    raise SkillFormatError(f"{command.path}: phase {phase.name!r}: {exc}") from None
                models.append(phase.output)
    for (phase, what, parameter), bound in binds.items():
        if not bound:
            raise SkillFormatError(
                f"{command.path}: phase {phase!r}: {what} takes {parameter!r}, which names no "
                "earlier phase with an output in any plan; check its spelling")
    return tuple(models)


def check_phase_sources(command: SkillCommand, signatures: dict) -> list[str]:
    """Load warnings for a phased skill whose output models name unknown command fields.

    Each output model is checked with ``facts.check_fact_sources`` against the
    workflow's ``signatures`` (``facts.workflow_signatures``). A plain command name
    that several contexts define is resolved against the record's context only when
    the phase runs, so here it passes when some context resolves every field. A
    missing command or field is a warning, not a load failure: a workflow may add a
    command's ``Output`` after the skill that reads it. (A malformed ``"from"`` already
    failed the load, see ``command_from_module``.)
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


#: A step that names a command: its first word, snake_case (``find_item``, ``go_up``) or a
#: qualified name (``Resource/show_metadata``); then its chain, cut at ``(``, ``;`` or ``,``
#: (``go_up, then ...`` is prose after the command).
_STEP_COMMAND = re.compile(r"^([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+|[A-Z]\w*/\w+)\b([^(;,]*)")
_STEP_ARG = re.compile(r"(?<![\w<])([A-Za-z_]\w*)=")


def step_commands(lines, known: set[str]) -> list[tuple[str, tuple[str, ...]]]:
    """``(command, parameters)`` of each command step of *lines*: a step whose first word is
    a snake_case name or one of *known* (``find_item query=Dune (note)`` gives
    ``("find_item", ("query",))``); the parameters are the ``key=`` of its chain, before
    the first ``(``, ``;`` or ``,``, as code reads them (``skill_phases.code_step``)."""
    out = []
    for line in lines:
        text = re.sub(r"\s*\(only when .*\)$", "", str(line).strip())
        first = re.match(r"[\w/]+", text)
        match = _STEP_COMMAND.match(text)
        if match is None and not (first and first.group(0) in known):
            continue
        name = match.group(1) if match else first.group(0)
        chain = match.group(2) if match else re.split(r"[(;,]", text[len(name):], maxsplit=1)[0]
        out.append((name, tuple(_STEP_ARG.findall(chain))))
    return out


def check_step_commands(command: SkillCommand, signatures: dict) -> list[str]:
    """Problems with the commands a skill's steps name: each command step of its template
    (every placeholder combination; ``step_commands``) and each ``pick`` must name a
    command of the workflow (*signatures*: ``facts.workflow_signatures``), and each
    parameter it passes must be a field of that command's ``Signature.Input``, the schema a
    command call is checked against. Returned as messages: the skill fails its load
    (``workflow_agent``) and ``fastworkflow skill preview --check`` fails."""
    from fastworkflow import skill as skill_api

    fields: dict[str, list[set]] = {}
    for key, sig in signatures.items():
        for name in {key, key.rsplit("/", 1)[-1]}:
            fields.setdefault(name, []).append(
                set(getattr(getattr(sig, "Input", None), "model_fields", {}) or {}))
    known = set(fields)
    seen: dict[tuple[str, tuple[str, ...]], None] = {}
    for env in _template_grid(command):
        with skill_api.picks_noted() as picks:
            try:
                lines = _call_steps(command.steps_fn, input_instance(command, env), command.path)
            except Exception:  # noqa: BLE001 - the template build reports it
                continue
        for found in step_commands(lines, known) + [(c, (p,)) for c, p in picks]:
            seen.setdefault(found)
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


def phases_for(command: SkillCommand, args: dict) -> list | None:
    """The phases ``plan()`` gives for ``args`` (complete, checked arguments): those that run.

    ``None`` when ``plan()`` returns ``None`` or ``[]`` (``skill_phases.is_flat_plan``):
    this call runs flat, from ``steps()``. Raises SkillCallError when ``plan()`` fails,
    returns something else than a list of ``Phase``, or leaves every phase out.
    """
    from fastworkflow.skill_phases import PhaseError, active_phases, is_flat_plan

    instance = input_instance(command, args)
    try:
        raw = command.plan_fn(instance)
        if is_flat_plan(raw):
            return None
        phases = active_phases(raw, f"plan() of {command.name}")
    except PhaseError as exc:
        raise SkillCallError(str(exc)) from None
    except Exception as exc:  # noqa: BLE001 - workflow code; the planner falls back to the template
        raise SkillCallError(f"plan() of {command.name} failed: {type(exc).__name__}: {exc}") from None
    if not phases:
        raise SkillCallError(f"plan() of {command.name} returns no phase")
    return phases


def call_phases(command: SkillCommand, args: dict) -> list:
    """The phases a call of ``command`` with ``args`` runs: ``phases_for``'s, or, for a flat
    call (no ``plan()``, or ``plan()`` gives no phases), one phase named after the skill whose
    steps are ``expand(command, args)``. A skill without phases runs as one phase.
    Raises SkillCallError as ``phases_for`` and ``expand`` do."""
    from fastworkflow.skill_phases import Phase

    phases = phases_for(command, args) if command.plan_fn is not None else None
    if phases is not None:
        return phases
    steps = expand(command, args)
    return [Phase(command.name, lambda: list(steps))]


def expand(command: SkillCommand, args: dict) -> list[str]:
    """The skill's command steps for ``args`` (complete, checked arguments), unnumbered:
    ``steps(input)``. Raises SkillCallError when ``steps()`` fails or returns something else
    than a list of text."""
    instance = input_instance(command, args)
    try:
        lines = _call_steps(command.steps_fn, instance, command.path)
    except SkillCallError:
        raise
    except Exception as exc:  # noqa: BLE001 - workflow code; the planner falls back to the template
        raise SkillCallError(f"steps() of {command.name} failed: {type(exc).__name__}: {exc}") from None
    return lines


def number(steps: list[str]) -> str:
    """Steps as the planner writes them: a numbered list, one per line."""
    return "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))


def _run_validate(command: SkillCommand, args: dict, request: str, workflow) -> dict:
    """``args`` as the skill's ``validate_extracted_parameters`` leaves them; raises
    SkillCallError when it refuses them or raises."""
    import inspect

    instance = input_instance(command, args)
    hook = command.validate
    kwargs = {"command_text": request} if "command_text" in inspect.signature(hook).parameters else {}
    import fastworkflow

    try:
        result = hook(workflow, command.name, instance, **kwargs)
    except fastworkflow.Refused as refusal:
        result = (False, str(refusal))
    except Exception as exc:  # noqa: BLE001 - workflow code; the planner falls back to the template
        raise SkillCallError(f"validate_extracted_parameters raised {type(exc).__name__}: {exc}") from None
    ok, message = result if isinstance(result, tuple) else (bool(result), "")
    if not ok:
        raise SkillInputError(f"validate_extracted_parameters: {message}")
    fields = command.input_model.model_fields
    out = {p.name: getattr(instance, p.name, args.get(p.name)) for p in command.params}
    # an Optional[str] field set to None is the absent argument, "" (``_absent_as_none``)
    return {n: "" if v is None and fields[n].default is None else v for n, v in out.items()}


def _validation_text(error) -> str:
    """A pydantic ValidationError as one line: ``field: message; ...`` (a validator's own
    ValueError text without pydantic's "Value error, " prefix)."""
    parts = []
    for e in error.errors():
        message = str(e.get("msg", "")).removeprefix("Value error, ")
        where = ".".join(str(x) for x in e.get("loc", ()))
        parts.append(f"{where}: {message}" if where else message)
    return "; ".join(parts)


def check_input(command: SkillCommand, args: dict, request: str, workflow=None) -> None:
    """Validate ``args`` against the skill's ``Input`` (its field types and its validators), with
    the context ``{"request": request, "workflow": workflow}``. Raises SkillInputError, whose
    message says which field breaks which rule."""
    from pydantic import ValidationError

    try:
        command.input_model.model_validate(_absent_as_none(command, args),
                                           context={"request": request, "workflow": workflow})
    except ValidationError as exc:
        raise SkillInputError(f"the arguments break a rule of {command.name}: {_validation_text(exc)}") from None


def validate_call(command: SkillCommand, args: dict, request: str, workflow=None,
                  notes: list[str] | None = None) -> dict:
    """``args`` after the skill's ``validate_extracted_parameters`` hook (a skill file's), as for
    a command: ``hook(workflow, skill_name, input, command_text=request)`` returns ``(ok,
    message)`` and may correct fields of ``input`` in place; then the ``Input``'s own rules
    (``check_input``). Raises SkillInputError when the hook or a rule refuses the call,
    SkillCallError when the hook raises; without either ``args`` come back unchanged."""
    checked = args if command.validate is None else _run_validate(command, args, request, workflow)
    for p in command.params:
        if checked[p.name] != args.get(p.name) and notes is not None:
            notes.append(f"{p.name}={args.get(p.name)!r} -> {checked[p.name]!r} (validate_extracted_parameters)")
    check_input(command, checked, request, workflow)
    return checked


# ---------------------------------------------------------------- arguments and guards

def _clean_str(value) -> str:
    text = str(value).strip()
    while len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'`":
        text = text[1:-1].strip()
    return text.rstrip(" .;,")


def _strip_words(text: str, words: tuple[str, ...]) -> str:
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


def _empty(value) -> bool:
    return value is None or str(value).strip().strip("\"'`").lower() in ("", "none", "null")


def _confident(verdicts: dict | None, name: str):
    """The ``(value, probability)`` check of ``name`` when it is sure enough, else None."""
    verdict = (verdicts or {}).get(name)
    return verdict if verdict is not None and verdict[1] >= VERIFY_THRESHOLD else None


def _request_numbers(request: str, args: dict, command: SkillCommand) -> set[int]:
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


def _check_subjects(command: SkillCommand, request: str) -> list[str]:
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


def _request_subjects(command: SkillCommand, request: str) -> list[str]:
    from fastworkflow.split_merge import request_subjects
    return request_subjects(request, command.identifiers)


#: Dashes folded to "-" when an argument is looked up in the request (non-breaking hyphen etc.).
_DASHES = str.maketrans({c: "-" for c in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d"})


def _fold(text: str) -> str:
    """``text`` with each Unicode dash as "-" (one character for one: positions are kept)."""
    return text.translate(_DASHES)


def _screen(command: SkillCommand, raw: dict, request: str) -> tuple[dict, dict[str, str]]:
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


def check_arguments(command: SkillCommand, raw: dict, request: str,
                    verdicts: dict | None = None) -> tuple[dict, list[str]]:
    """The guarded, complete arguments of a skill call; ``(args, notes)``.

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
    - an enum must be one of its values, an int an integer, a bool true/false;
    - a str argument loses surrounding quotes and its strip words, and must then be a
      verbatim (case-insensitive; any Unicode dash matches "-") substring of ``request``;
      it takes the request's own spelling. One that is not: an optional argument takes its default, a required one
      fails the call (the planner fills it once more, then falls back to the template path);
    - a str argument inside an identifier match of the request (``command.identifiers``)
      becomes the whole match; a ``subject`` one must name a record of the request
      (``_as_subject``); one that does not is reset or fails the call as above;
    - an ``in_request`` int must be a number the request writes, when it writes any
      (see the module doc);
    - ``verdicts`` (``{param: (value, probability)}``, from ``check_verdicts``): a sure
      verdict on a ``verify`` parameter replaces its argument, and stands in for a missing
      or invalid enum (for a subject, ``NO_SUBJECT`` resets it, or fails a required one);
    - an optional str argument equal to another str argument takes its default.

    An argument that only applies when others take some value is reset afterwards, by the
    skill's ``validate_extracted_parameters`` (``validate_call``).

    ``notes`` says what a guard changed. A required argument the guards refuse raises
    SkillInputError (the planner fills the call once more, ``plan_skill_call``); a sure check
    that the request names no such argument raises SkillCallError.
    """
    from fastworkflow.utils.signatures import is_sentinel_number

    notes: list[str] = []
    args: dict = {}
    folded = _fold(request).lower()
    subjects = _request_subjects(command, request) if any(p.subject for p in command.params) else []
    kinds = frozenset(v.lower() for p in command.params if p.type == "enum" for v in p.values)
    single = {**raw, **{p.name: _single(raw.get(p.name)) for p in command.params}}
    screened, refused = _screen(command, single, request)

    def refuse(p: Param, value, why: str) -> None:
        if p.required:
            raise SkillInputError(f"{p.name}={value!r} {why}")
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
            given = "" if _empty(value) or p.name in refused else _strip_words(_clean_str(value), p.strip)
            if (chosen or "").lower() != given.lower():
                notes.append(f"{p.name}={shown!r} -> {chosen or p.default!r} (check {sure[1]:.2f})")
            if chosen is None:
                if p.required:
                    raise SkillCallError(f"{p.name}={shown!r}: the check finds no {p.name} in the request")
                args[p.name] = p.default
                continue
            value = compared = chosen
        elif p.name in refused:
            refuse(p, shown, refused[p.name])
            continue
        if _empty(value):
            if p.required:
                raise SkillInputError(f"missing required argument {p.name}")
            args[p.name] = p.default
            continue
        if p.type == "bool":
            if isinstance(value, str):
                if value.strip().lower() not in ("true", "false"):
                    raise SkillInputError(f"{p.name}={shown!r} is not true or false")
                value = value.strip().lower() == "true"
            args[p.name] = bool(value)
        elif p.type == "int":
            try:
                number = int(str(value).strip())
            except ValueError:
                raise SkillInputError(f"{p.name}={shown!r} is not an integer") from None
            if is_sentinel_number(number):   # the extractor's "not given" marker or a made-up extreme
                if p.required:
                    raise SkillInputError(f"missing required argument {p.name}")
                number = p.default
            args[p.name] = number
        elif p.type == "enum":
            text = _clean_str(value).lower()
            if text not in p.values:
                raise SkillInputError(f"{p.name}={shown!r} is not one of {', '.join(p.values)}")
            args[p.name] = text
        else:
            text = _strip_words(_clean_str(value), p.strip)
            at = folded.find(_fold(text).lower()) if text else -1
            if at < 0:
                if p.required:
                    raise SkillInputError(f"{p.name}={shown!r} is not written in the request")
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
        if not p.in_request:
            continue
        written = _request_numbers(request, args, command)
        if not written or args[p.name] in written:
            continue
        if len(written) == 1:
            (only,) = written
        elif p.default in written:
            only = p.default
        else:
            raise SkillInputError(f"{p.name}={args[p.name]} is not a number the request writes "
                                 f"({', '.join(map(str, sorted(written)))})")
        notes.append(f"{p.name}={args[p.name]} -> {only} (the number the request writes)")
        args[p.name] = only
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
        if first and not first.endswith(":"):
            out.setdefault(name, []).append(re.split(r"(?<=[.!?])\s", first, maxsplit=1)[0])
    return {name: " ".join(texts) for name, texts in out.items()}


def command_utterances(workflow_folderpath: str) -> dict[str, list[str]]:
    """``{name: plain_utterances}`` of the workflow's commands, from its routing definition
    (a name in several contexts gets the utterances of each). Empty when the definition
    cannot be read; never raises."""
    from fastworkflow.command_routing import RoutingRegistry

    out: dict[str, list[str]] = {}
    try:
        crd = RoutingRegistry.get_definition(workflow_folderpath)
        qualified = {q for names in crd.contexts.values() for q in names}
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
    gives that text's words."""
    sentences = {name: [desc] for name, desc in command_descriptions(available_commands).items()}
    if workflow_folderpath:
        for name, utterances in command_utterances(workflow_folderpath).items():
            sentences.setdefault(name, []).extend(utterances)
    out = {}
    for name, texts in sentences.items():
        words = [_content_words(t) for t in texts]
        counts: dict[str, int] = {}
        for ws in words:
            for stem in {_stem(w) for w in ws}:
                counts[stem] = counts.get(stem, 0) + 1
        least = 2 if len(texts) > 1 else 1
        out[name] = " ".join(sorted({w for ws in words for w in ws if counts[_stem(w)] >= least}))
    return out


def filter_extra_steps(extra, expanded: list[str], names: frozenset[str], skill_name: str,
                       request: str = "", notes: list[str] | None = None,
                       texts: dict[str, str] | None = None) -> tuple[list[str], list[str]]:
    """``(kept, dropped)`` of the model's steps_before/steps_after entries.

    An entry first loses its ``<...>`` placeholders (with the ``key=`` holding one): the
    model cannot know a value the run finds, and the agent passes a placeholder on as one
    (``list_logins login_id=<the person's uid>`` becomes ``list_logins``). Dropped: empty
    entries, lines copied from the skill's template (``_is_template_line``: the expansion
    already writes them for these arguments, or they do not apply to them), entries
    holding a uid-shaped token ``request`` does not write, entries naming
    the skill itself (the expansion already runs it), entries naming no available command,
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
        cmds = {w for w in _IDENT.findall(cleaned) if w in names}
        if skill_name in cleaned or (names and (not cmds or cmds <= expanded_cmds)):
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
        return f"{desc}. true only when the request explicitly asks for this; otherwise false"
    if p.type == "str":
        desc += ". Copied verbatim from the request"
    return desc + (" (required)" if p.required else f" (default {json.dumps(p.default)})")


def typed_signature(command: SkillCommand):
    """The skill's dspy signature: user_query in; one typed field per parameter, plus
    steps_before / steps_after, out."""
    import dspy

    fields: dict = {"user_query": (str, dspy.InputField())}
    for p in command.params:
        kind = {"str": str, "bool": bool, "int": int}.get(p.type)
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
        return (("true", f"yes: {p.desc or p.name}"),
                ("false", "no: the request does not ask for this, or says not to"))
    means = dict(p.means)
    return tuple((v, means.get(v, v)) for v in p.values)


def _check_one(lm, command: SkillCommand, p: Param, request: str) -> tuple[object, float]:
    import dspy
    from dspy.experimental import Choice

    if p.type == "bool":
        question = f"Does the request ask for this: {p.desc or p.name}?"
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


def check_verdicts(lm, command: SkillCommand, request: str) -> dict[str, tuple[object, float]]:
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
            logger.warning(f"could not check skill argument {command.name}.{p.name}", exc_info=True)
            return None
    with ThreadPoolExecutor(len(params)) as pool:
        return dict(r for r in pool.map(one, params) if r is not None)


#: Appended to the request when the planner fills a skill call again after its rules refused
#: the first one (``plan_skill_call``).
REFUSED_CALL_NOTE = ("(A first call of this skill was refused: {refusal}. Fill the call again from the "
                     "request above, correcting that.)")


@dataclass
class SkillCallPlan:
    plan: str
    args: dict
    notes: list[str]
    prediction: object
    #: The phases that run, in order (``skill_phases.Phase``, see ``call_phases``): a phased
    #: skill's (``plan()``), or, for a flat call, one phase named after the skill with its
    #: steps. And the extra steps the request asks for before and after the skill's own.
    phases: list | None = None
    steps_before: list[str] = field(default_factory=list)
    steps_after: list[str] = field(default_factory=list)


def plan_skill_call(command: SkillCommand, request: str, available_commands: str,
                    check_lm=None, workflow=None) -> SkillCallPlan:
    """Plan ``request`` as a call of ``command``: the typed call, the guards, the expansion.

    Runs under the caller's dspy context (the planner LM and adapter); ``available_commands``
    reaches the adapter as extra input, as for the text planner. With ``check_lm`` (a
    classifier LM) the skill's ``verify`` parameters are decided again from the request,
    alongside the typed call (``check_verdicts``). A skill file's
    ``validate_extracted_parameters`` then sees the checked arguments (``validate_call``;
    ``workflow`` is passed to it), and the ``Input``'s rules (``check_input``). When the skill's
    rules or the argument guards refuse the arguments (SkillInputError: a made-up name, a
    missing or invalid value), the planner fills the call once more, with the
    refusal after the request (``REFUSED_CALL_NOTE``); the guards still read the request alone.
    Raises on any other failure, or a second refusal (the caller falls back to the template
    path).
    """
    import dspy

    def typed_call(query: str):
        return dspy.ChainOfThought(typed_signature(command))(
            user_query=query, available_commands=available_commands)

    def checked(prediction) -> tuple[dict, list[str]]:
        raw = {p.name: getattr(prediction, p.name, None) for p in command.params}
        args, notes = check_arguments(command, raw, request, verdicts)
        return validate_call(command, args, request, workflow, notes), notes

    with ThreadPoolExecutor(1) as pool:
        checks = pool.submit(check_verdicts, check_lm, command, request)
        prediction = typed_call(request)
        verdicts = checks.result()
    try:
        args, notes = checked(prediction)
    except SkillInputError as refused:
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
    return SkillCallPlan(plan=number(before + steps + after), args=args, notes=notes, prediction=prediction,
                         phases=phases, steps_before=list(before), steps_after=list(after))
