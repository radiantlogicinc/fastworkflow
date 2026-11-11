"""What a planning command file writes: ``class PlanGenerator``, its phases, and the step helpers.

A planning command file (``_commands/<name>.py``) reads like a command file. ``class Signature``
declares the planning command exactly as a command's does (docstring, ``Input``,
``plain_utterances``). Where a command has a ``ResponseGenerator``, a planning command has a
``PlanGenerator``, because it generates a plan. The book example is ``tests/planning_tutorial_workflow``'s
``book_details`` command::

    from pydantic import BaseModel, Field

    from fastworkflow.planning import run
    from .find_book import Signature as FindBook        # the commands whose Output the facts are typed by
    from .show_book import Signature as ShowBook


    class Signature:                                  # the book the request names
        plain_utterances = ["who wrote Persuasion", "how many copies of Emma are there"]

        class Input(BaseModel):
            title: str = Field(description="the book's title")


    class PlanGenerator:
        def find(self, input) -> FindBook.Output:     # a phase; it fills find_book's Output
            return [run("find_book", query=input.title)]

        class Book(BaseModel):                        # facts: a field typed with a command's Output
            show: ShowBook.Output

            @property                                 # other values: computed, not copied
            def name(self) -> str:
                return self.show.title

        def book(self, find) -> Book:                 # reads find's facts
            if not find.book_uids:
                return []                             # no steps: skipped when reached
            return [run("show_book", book_uid=find.book_uids[0])]

        def answer(self, input, book=None):           # None: the book phase was skipped
            if book is None:
                return [f"say that no book matches {input.title}"]
            return [f"say that {book.name} is by {book.show.author}; the library owns "
                    f"{book.show.copies} copies"]

The rules (the one-page guide: ``docs/planning_command_file_layout.md``)
-------------------------------------------------------------

1. Each public method of ``PlanGenerator`` is a phase (one executor run), in the order
   written; ``_name`` methods are helpers.
2. A phase's parameters name what it reads: ``input`` and earlier phases by name (None, its
   default, when that phase was skipped).
3. Its return annotation is what it fills: a command's ``Output`` (``-> FindBook.Output``), or a
   facts model whose fields are typed with a command's ``Output`` or ``Input``. Any other value is a
   property or method of the facts model (copies, ``fact_from`` and ``"from"``, are not read); no
   annotation: it fills nothing.
4. It returns its steps as text, exactly the lines the agent reads. ``return []`` skips it:
   always decided when the phase is reached, on the real facts (the method is called once).

Sharing phases: a base class in a ``_`` module (``_commands/_book_phases.py``, ``class BookPhases``)
holds the phases two commands share; ``class PlanGenerator(BookPhases)`` keeps them and adds its
own. A ``PlanGenerator`` may not subclass another command's (a load error): to reuse that command's
whole job, call it in a ``run()`` step.

Optional: ``PlanGenerator.__phases__(self, input)`` returns the phases a call runs, in order
(default: every one); ``return [self.check]`` runs one. ``@phase(over=..., max_items=..., until=...,
noun=..., optional=...)`` makes a phase a fan-out (once per item) or optional.

Steps: ``run("command", k=v)`` runs a command when every value is given; ``run("command",
named=x)`` opens the row a find listed whose label matches ``x``; ``run("command", all_pages=True)``
reads every page; ``when(...)`` and ``do_not_run(...)`` guard a step. ``count(n, noun, plural=None)`` and
``count_and_names_read(names, total, noun)`` write the counted lines. The retired helpers (``pick``,
``full_list``, ``listed``, ``num``) fail the import, each naming its replacement. ``say that ...`` is an answer line
code copies word for word; any other line is the agent's to write.

The template (what the planner sees) calls each method once per input with every fact a
``planning_phases.Placeholder``: it reads ``<find.total>``, ``<find.book_uids[0]>``; a helper
method called with one reads as its name in words; every test on a fact reads as true and is
noted, so the phase's lines end ``(only when find.book_uids)``. A method that fails there, or
writes ``None`` into a step, is a load error.
"""
from __future__ import annotations

import contextlib
import contextvars
import functools
import inspect
import re
from typing import Any, Callable, Optional

from pydantic import BaseModel

from fastworkflow.observation_offloading.finish_check import quote_step_value
from fastworkflow.utils.python_utils import type_text
from fastworkflow.field_options import (ConfirmsOptions, CopiedOptions, FieldOptions,  # noqa: F401
                                        RecordNamedOptions)
from fastworkflow.command_declarations import (UserApproval, descends_to,  # noqa: F401
                                               user_approval_for_execution)
from fastworkflow.planning_phases import (Phase, PhaseError, Placeholder, _template_kwargs,
                                       concatenated_steps, is_building_template,
                                       is_placeholder, tests_noted)

__all__ = [
    "ConfirmsOptions",
    "CopiedOptions",
    "FieldOptions",
    "RecordNamedOptions",
    "UserApproval",
    "count",
    "count_and_names_read",
    "descends_to",
    "do_not_run",
    "phase",
    "run",
    "run_code",
    "user_approval_for_execution",
    "when",
]


class StepFormError(ValueError):
    """A step written in a form the runtime would misread."""


class TemplateError(PhaseError):
    """A phase method that fails, or writes None into a step, while the template is built:
    a load error (``planning_commands.placeholder_template``)."""


# --------------------------------------------------------------------------- steps

class Step(str):
    """A step written by a helper: a ``str`` carrying which helper wrote it (``.helper``)."""

    __slots__ = ("helper",)

    def __new__(cls, text: str, helper: str) -> Step:
        obj = super().__new__(cls, text)
        obj.helper = helper
        return obj

    def note(self, text: str) -> Step:
        """This step with `` (text)`` appended, still marked as the helper's."""
        return Step(f"{self} ({text})", self.helper)


#: The Input field names the step helpers take as their own keywords (7.4). A command with a field of
#: either name fails the load: ``run`` could not pass it to the command.
RESERVED_INPUT_NAMES = ("named", "all_pages")


def reserved_input_name_message(input_model: Any, where: str) -> Optional[str]:
    """The load error for a command ``Input`` with a field named like a step helper's keyword (7.4),
    or None."""
    clash = next((name for name in RESERVED_INPUT_NAMES
                  if name in (getattr(input_model, "model_fields", None) or {})), None)
    if clash is None:
        return None
    return (f"{where}: Input field {clash!r} is reserved: run() takes {clash}= as its own keyword, so "
            f"no command call can pass a field of that name; rename the field")

#: Key recorded for a ``run(..., named=...)`` step whose command declares no row id input: a load problem.
_NAMED_UNRESOLVED = "named"

#: ``(command name -> [set of Input field names, ...], collected (command, keys) list or None,
#: command name -> set of row id input names (None where none is declared))``.
_COMMAND_INPUTS: contextvars.ContextVar[Optional[tuple[
    dict[str, list[set[str]]], Optional[list[tuple[str, tuple[str, ...]]]], dict[str, set[Optional[str]]]]
]] = contextvars.ContextVar("fw_planning_command_command_inputs", default=None)


def row_id_field(input_model: Any) -> Optional[str]:
    """The Input field a row is opened by (``run(command, named=...)``): the one field declared with
    ``from_command_output``, else the one required ``str`` field, else None."""
    fields = getattr(input_model, "model_fields", None) or {}
    declared = [name for name, info in fields.items()
                if isinstance(getattr(info, "json_schema_extra", None), dict)
                and info.json_schema_extra.get("from_command_output")]
    if len(declared) == 1:
        return declared[0]
    required = [name for name, info in fields.items() if info.is_required() and info.annotation is str]
    return required[0] if len(required) == 1 else None


@contextlib.contextmanager
def command_inputs_for_steps(signatures: Optional[dict[str, Any]], collect: bool = False):
    """Set the workflow's command ``Signature.Input`` field sets and row ids for ``run``.

    With *collect* (``planning_commands.check_step_commands``), each call's ``(command, keys)`` is
    appended to the yielded list instead of raising ``StepFormError`` on the first bad keyword.
    """
    fields: Optional[dict[str, list[set[str]]]] = None
    row_ids: dict[str, set[Optional[str]]] = {}
    steps: Optional[list[tuple[str, tuple[str, ...]]]] = [] if collect else None
    if signatures is not None:
        fields = {}
        for key, sig in signatures.items():
            input_model = getattr(sig, "Input", None)
            names = set(getattr(input_model, "model_fields", {}) or {})
            row_id = row_id_field(input_model)
            for name in {key, key.rsplit("/", 1)[-1]}:
                fields.setdefault(name, []).append(names)
                row_ids.setdefault(name, set()).add(row_id)
    token = _COMMAND_INPUTS.set((fields, steps, row_ids) if fields is not None else None)
    try:
        yield (fields, steps) if collect else fields
    finally:
        _COMMAND_INPUTS.reset(token)


def _check_step_keywords(command: str, keys: tuple[str, ...]) -> None:
    state = _COMMAND_INPUTS.get()
    if state is None:
        return
    fields, steps, _ = state
    if steps is not None:
        steps.append((command, keys))
        return
    if not keys or command not in fields:
        return
    unknown = [k for k in keys if not any(k in f for f in fields[command])]
    if unknown:
        raise StepFormError(
            f"a step passes {', '.join(unknown)} to {command!r}, whose "
            "Signature.Input has no such field"
        )


def _unknown_keys(command: str, keys: tuple[str, ...]) -> list[str]:
    """The keys of a step that the command's ``Signature.Input`` has no field for (none without signatures)."""
    state = _COMMAND_INPUTS.get()
    if state is None or not keys or command not in state[0]:
        return []
    return [k for k in keys if not any(k in f for f in state[0][command])]


def _row_id_of(command: str) -> tuple[bool, Optional[str]]:
    """``(command is known with its signature, its row id input or None)``. Unknown without a
    signature (a run, outside the load check)."""
    state = _COMMAND_INPUTS.get()
    if state is None or command not in state[2]:
        return False, None
    ids = state[2][command]
    return True, (next(iter(ids)) if len(ids) == 1 else None)


def _named_problem(command: str, message: str, reported: bool) -> None:
    """A ``run(..., named=...)`` step that cannot open its row: a load error (raised), or, while the
    load check collects, a problem the check reports. *reported*: an unknown keyword already is the
    problem, so the row problem is not added."""
    state = _COMMAND_INPUTS.get()
    if state is not None and state[1] is not None:
        if not reported:
            state[1].append((command, (_NAMED_UNRESOLVED,)))
        return
    raise StepFormError(message)


def _is_step_placeholder(value: Any) -> bool:
    if is_placeholder(value):
        return True
    return isinstance(value, str) and len(value) >= 3 and value.startswith("<") and value.endswith(">") \
        and "<" not in value[1:-1] and ">" not in value[1:-1]


def _step_value(value: Any) -> str:
    if _is_step_placeholder(value):
        return str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return quote_step_value(value)


def _command_text(command: str, values: dict[str, Any]) -> str:
    parts = [f"{k}={_step_value(v)}" for k, v in values.items() if v is not None]
    return f"{command} {' '.join(parts)}" if parts else str(command)


def run(command: str, *, named: Any = None, all_pages: bool = False, **inputs: Any) -> Step:
    """A command step code runs when every required value is given: ``<command> <k=v ...>``.

    ``inputs`` are the command's own ``Input`` field names; a ``None`` value is left out.

    - ``all_pages=True``: a paged command read through every page (up to ``CODE_MAX_PAGES``, then
      the agent reads on from the next page): ``<command> <k=v ...> (full list)``.
    - ``named=<name>``: open the row of a listing this call read whose label holds that name (or the
      only row). Its row id is the input named by ``from_command_output``, else the command's one
      required ``str`` field. Given a concrete id, the step is ``<command> <id field>=<value>``;
      without one it is ``<command> with the uid of the <name> match``, which code runs on the
      matching row.
    """
    if named is not None and all_pages:
        raise StepFormError(f"run({command!r}): named= opens one row and all_pages=True reads every "
                            "page; give one of them")
    _check_step_keywords(command, tuple(inputs))
    typo = bool(_unknown_keys(command, tuple(inputs)))
    given = {k: v for k, v in inputs.items() if v is not None}
    if named is None:
        text = _command_text(command, given)
        return Step(f"{text} (full list)" if all_pages else text, "run")
    if not str(named).strip():
        raise StepFormError(f"run({command!r}): named= is the name the row is picked by")
    known, row_id = _row_id_of(command)
    if not known:
        # Outside the load check (no signatures): a concrete value is written; none is the row's words.
        if any(not is_placeholder(v) for v in given.values()):
            return Step(_command_text(command, given), "run")
        return Step(f"{command} with the uid of the {named} match", "run")
    if row_id is not None and row_id in given and not is_placeholder(given[row_id]):
        return Step(_command_text(command, given), "run")
    if row_id is None:
        _named_problem(command, f"run({command!r}, named=...): {command} declares no row id input "
                                "(a field from_command_output, or its one required str field): pass "
                                "the id input explicitly", typo)
    elif set(given) - {row_id}:
        _named_problem(command, f"run({command!r}, named=...): with no {row_id} given, the step opens "
                                "the row by name and takes no other input: pass the row's id, or drop "
                                "the other input", typo)
    return Step(f"{command} with the uid of the {named} match", "run")


def when(condition: str, step: str) -> Step:
    """A conditional step the agent decides whether to run: ``only when <condition>, <step>``."""
    return Step(f"only when {condition}, {step}", "when")


def do_not_run(command: str) -> Step:
    """A negated step code passes over, and a guard: ``do not run <command>``."""
    return Step(f"do not run {command}", "do_not_run")

#: ``(phase_name, step_text)`` of each step written while ``steps_by_phase_noted`` is open.
_PHASE_STEPS: contextvars.ContextVar[Optional[list[tuple[str, str]]]] = contextvars.ContextVar(
    "fw_planning_command_phase_steps", default=None)


@contextlib.contextmanager
def steps_by_phase_noted():
    """A list that gets ``(phase_name, step_text)`` of each step a phase returns
    (before the template's ``(only when ...)`` suffix is appended)."""
    seen: list[tuple[str, str]] = []
    token = _PHASE_STEPS.set(seen)
    try:
        yield seen
    finally:
        _PHASE_STEPS.reset(token)


def note_phase_steps(phase_name: str, lines: Any) -> None:
    """Record *lines* under *phase_name* when ``steps_by_phase_noted`` is open."""
    seen = _PHASE_STEPS.get()
    if seen is not None and isinstance(lines, (list, tuple)):
        for line in lines:
            text = str(line).strip()
            if text:
                kept = Step(text, line.helper) if isinstance(line, Step) else text
                seen.append((phase_name, kept))

# --------------------------------------------------------------------------- filters

def count(number: Any, noun: str, plural: Optional[str] = None) -> str:
    """"1 login", "2 logins" ("2 people" with ``plural="people"``). A placeholder keeps its text:
    a Placeholder's == is always true, so the plural test is not made on it."""
    if is_placeholder(number):
        return f"{number} {plural or noun + 's'}"
    return f"{number} " + (noun if number == 1 else plural or noun + "s")


def count_and_names_read(names: Any, total: Any, noun: str, same: Optional[tuple] = None,
                         plural: Optional[str] = None) -> str:
    """"N <noun>s (a, b)", "none (0 <noun>s)", with "K of N shown" when fewer were read.
    *same* is ``("<what>", items)`` of a list already stated in full: when *names* holds
    exactly those items, all N shown, it is "N <noun>s (the same <what>)"."""
    if is_placeholder(names) or is_placeholder(total):
        return f"{count(total, noun, plural)} ({names})"
    if same and len(names) == total > 1 and sorted(names) == sorted(same[1]):
        return f"{count(total, noun, plural)} (the same {same[0]})"
    if not names:
        return f"none ({count(total, noun, plural)})" if not total else (
            f"{count(total, noun, plural)}, none of them read")
    shown = f"; {len(names)} of {total} shown" if len(names) < total else ""
    return f"{count(total, noun, plural)} ({', '.join(names)}{shown})"


#: The helpers removed from this module, each with the spelling that replaces it. Importing one
#: (``from fastworkflow.planning import pick``) fails the load with that message.
_REMOVED = {
    "pick": "pick is removed: write run(command, named=<name>, <id input>=<value>), "
            "e.g. run(\"open_book\", named=input.title)",
    "full_list": "full_list is removed: write run(command, ..., all_pages=True)",
    "from_": "from_ is removed: write json_schema_extra={\"from_command_output\": \"<command>.<field>\"} "
             "on the Input field",
    "listed": "listed is removed: write count_and_names_read(names, total, noun)",
    "num": "num is removed: write the number inline, e.g. f\"{value}\" (see the planning command docs)",
}


def __getattr__(name: str) -> Any:
    if name in _REMOVED:
        raise ImportError(_REMOVED[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# --------------------------------------------------------------------------- @phase

_OPTIONS = ("over", "max_items", "until", "noun", "optional", "output", "bind")


def phase(**options: Any) -> Callable:
    """What a phase needs beyond the rules (``planning_phases.Phase``'s options):

    - ``over``: run the phase once per item: ``"find.uids"`` (a list an earlier phase's output
      names: a field, or a property of it or of a typed field's model) or a function of ``input``
      giving the items. The method's one parameter that
      is neither ``input`` nor a phase gets the item (``def book(self, uid)``);
    - ``max_items``, ``noun``, ``until`` (a function whose parameter named after this phase
      gets the items' facts so far; true stops the fan-out), ``bind``;
    - ``optional``: the job goes on when this phase fails;
    - ``output``: the facts model, when it depends on the input.

    Any option may be a function of ``input`` alone: it is decided when the call is planned.
    """
    unknown = set(options) - set(_OPTIONS)
    if unknown:
        raise TypeError(f"phase(): unknown option(s) {sorted(unknown)}; known: {_OPTIONS}")

    def mark(fn):
        fn.__phase_options__ = {**getattr(fn, "__phase_options__", {}), **options}
        return fn

    return mark


# --------------------------------------------------------------------------- loading

def phase_methods(cls: type) -> list[tuple[str, Callable]]:
    """The phase methods of a ``PlanGenerator`` class (``_name`` are helpers): every public
    function ``self.<name>`` reaches, as Python resolves it (an override replaces the base's
    method), listed in definition order, the bases' first. That order is the plan only when
    one class decides it (``_order_problem``)."""
    order: dict[str, None] = {}
    for klass in reversed(cls.__mro__):
        order.update(dict.fromkeys(vars(klass)))
    return [(name, value) for name in order if not name.startswith("_")
            and inspect.isfunction(value := inspect.getattr_static(cls, name))]


def _order_problem(cls: type) -> Optional[str]:
    """Why *cls* must write its own ``__phases__``, or None. A subclass without one keeps the
    plan of its base (its ``__phases__``, or its phases in order), its overrides in their
    place; when it adds phases, or has phases from several bases, only the author knows
    where they go, so the order is written, never guessed."""
    if "__phases__" in vars(cls):
        return None
    bases = [b for b in cls.__bases__ if b is not object and phase_methods(b)]
    inherited = {name for b in bases for name, _ in phase_methods(b)}
    new = [name for name, _ in phase_methods(cls) if name not in inherited]
    if not bases or (len(bases) == 1 and not new):
        return None
    every = ", ".join(f"self.{name}" for name, _ in phase_methods(cls))
    named = [f"{b.__module__.rsplit('.', 1)[-1]}.{b.__qualname__}" for b in bases]
    what = (f"adds the phase(s) {', '.join(new)} to those of {named[0]}" if new
            and len(bases) == 1 else f"has phases from {', '.join(named)}")
    return (f"{cls.__qualname__} {what}: write __phases__(self, input) returning the phases a "
            f"call runs, in order (from: {every})")


def _custom_phases(cls: type) -> Optional[Callable]:
    """``cls.__phases__`` when the class or a base (not ``object``) defines one."""
    return next((vars(klass)["__phases__"] for klass in cls.__mro__
                 if klass is not object and "__phases__" in vars(klass)), None)


def _defines_call(cls: type) -> bool:
    """Whether the class or a base (not ``object``) defines ``__call__``: no longer read."""
    return any(klass is not object and "__call__" in vars(klass) for klass in cls.__mro__)


def _namespace(cls: type) -> dict[str, Any]:
    """The names of *cls* and its bases, as its attributes resolve them."""
    names: dict[str, Any] = {}
    for klass in reversed(cls.__mro__):
        names.update(vars(klass))
    return names


def _parameters(fn: Callable) -> list[str]:
    return [p for p in inspect.signature(fn).parameters if p != "self"]


def _output(cls: type, fn: Callable, options: dict, input: Any) -> Any:
    from fastworkflow.typed_phase_facts import union_members

    if "output" in options:
        return _decided(options["output"], input)
    annotation = fn.__annotations__.get("return")
    if isinstance(annotation, str):
        try:
            annotation = eval(annotation, fn.__globals__, _namespace(cls))  # noqa: S307
        except Exception as error:  # noqa: BLE001
            raise PhaseError(f"phase {fn.__name__!r}: cannot read its return annotation "
                             f"{annotation}: {error}") from None
    if annotation is None or annotation is inspect.Signature.empty:
        return None
    if union_members(annotation) is not None:
        return annotation
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return annotation
    raise PhaseError(f"phase {fn.__name__!r}: its return annotation must be the facts model "
                     f"it fills (a pydantic model), got {type_text(annotation)}; a model that depends "
                     "on the input goes in @phase(output=lambda input: ...)")


def _decided(value: Any, input: Any) -> Any:
    """An option given as a function of ``input`` only, decided now; other values as given."""
    if callable(value) and not inspect.isclass(value) and _parameters(value) == ["input"]:
        return value(input)
    return value


def _with_input(fn: Callable, input: Any) -> Callable:
    """*fn* with ``input`` given; its signature without it, so the runtime binds the rest
    by name (``planning_phases.bound_arguments``)."""
    sig = inspect.signature(fn)
    if "input" not in sig.parameters:
        return fn

    @functools.wraps(fn)
    def bound(**kwargs):
        return fn(input=input, **kwargs)

    bound.__signature__ = sig.replace(
        parameters=[p for name, p in sig.parameters.items() if name != "input"])
    return bound


#: A None written into a step (``f"query={input.group}"`` with no group): a load error.
_NONE = re.compile(r"(?<![\w<])None(?![\w>])")


def _template_self(plan: Any) -> Any:
    """``self`` while the template is built: *plan* as an instance of a subclass of its own
    class (so ``super()`` works in a method) in which a helper method (``self._name(...)``)
    called with an unknown fact is not run; it reads ``<name in words>``."""
    template = object.__new__(_template_class(type(plan)))
    template.__dict__.update(getattr(plan, "__dict__", {}))
    return template


@functools.lru_cache(maxsize=None)
def _template_class(cls: type) -> type:
    def __getattribute__(self, name):
        value = super(template, self).__getattribute__(name)
        if name == "__result__" or not name.startswith("_") or name.startswith("__") or not callable(value):
            return value

        @functools.wraps(value)
        def helper(*args, **kwargs):
            if any(_unknown(a) for a in (*args, *kwargs.values())):
                return Placeholder(name.strip("_").replace("_", " "))
            return value(*args, **kwargs)

        return helper

    template = type(cls.__name__, (cls,), {"__getattribute__": __getattribute__,
                                           "__module__": cls.__module__,
                                           "__qualname__": cls.__qualname__})
    return template


def _unknown(value: Any) -> bool:
    if is_placeholder(value):
        return True
    if isinstance(value, (list, tuple, set)):
        return any(is_placeholder(v) for v in value)
    if isinstance(value, dict):
        return any(is_placeholder(v) for v in value.values())
    return False


def _phase_functions(plan: Any, fn: Callable, input: Any) -> tuple[Callable, Callable]:
    """``(steps, writes_steps)`` of phase method *fn* for one call: the runtime's
    ``Phase.steps_fn`` and ``Phase.when``, both taking the phase's facts by name.

    At run time ``writes_steps`` calls the method (once) and keeps its result, which
    ``steps`` then returns: ``return []`` skips the phase when it is reached. While the
    template is built, ``steps`` calls it with ``Placeholder`` facts and ends each line with
    the tests the method made on them, ``(only when find.uids)``."""
    sig = inspect.signature(fn)
    name = fn.__name__
    kept: list = []   # [(facts, steps or the error)] of the last run-time call

    def call(owner: Any, facts: dict) -> Any:
        if "input" in sig.parameters:
            facts = {**facts, "input": input}
        return fn(owner, **facts)

    def template(facts: dict) -> list[str]:
        with tests_noted() as tests:
            try:
                lines = call(_template_self(plan), facts)
            except Exception as error:  # noqa: BLE001 - reported as a load error
                raise TemplateError(f"phase {name!r} fails while its template is built (every "
                                    f"fact unknown): {type(error).__name__}: {error}") from None
        if not isinstance(lines, (list, tuple)):
            return lines   # the runtime's check reports it
        for line in lines:
            if _NONE.search(str(line)):
                raise TemplateError(f"phase {name!r} writes None into a step ({str(line)!r}): "
                                    "a value it formats is None for some input")
        note_phase_steps(name, lines)
        when = f" (only when {' and '.join(tests)})" if tests else ""
        return [f"{line}{when}" for line in lines if str(line).strip()]

    def steps(**facts):
        if is_building_template():
            return template(facts)
        if kept and kept[0][0] == facts:
            result = kept.pop()[1]
            if isinstance(result, Exception):
                raise result
            note_phase_steps(name, result)
            return result
        lines = call(plan, facts)
        note_phase_steps(name, lines)
        return lines

    def writes_steps(**facts):
        try:
            lines = call(plan, facts)
        except Exception as error:  # noqa: BLE001 - raised again by steps: the phase fails
            kept[:] = [(facts, error)]
            return True
        kept[:] = [(facts, lines)]
        return not isinstance(lines, (list, tuple)) or any(str(s).strip() for s in lines)

    def only_when() -> list[str]:
        """The tests the template's call made on the facts (the preview's condition)."""
        with tests_noted() as tests:
            with contextlib.suppress(Exception):
                call(_template_self(plan), {p: Placeholder(p) for p in public.parameters})
        return tests

    public = sig.replace(parameters=[p for n, p in sig.parameters.items()
                                     if n not in ("self", "input")])
    for f in (steps, writes_steps):
        f.__signature__ = public
        f.__qualname__ = f"{type(plan).__qualname__}.{name}"
    writes_steps.only_when = only_when
    return steps, writes_steps


def planned(cls: type) -> Callable[[Any], list[Phase]]:
    """``planned(input)``: the phases a call of the ``PlanGenerator`` class *cls* runs, in
    order (``__phases__``'s, or every phase method)."""
    if not phase_methods(cls):
        raise PhaseError(f"{cls.__qualname__}: a PlanGenerator needs at least one phase method")
    methods = dict(phase_methods(cls))
    for name, fn in methods.items():
        if next(iter(inspect.signature(fn).parameters), None) != "self":
            raise PhaseError(f"PlanGenerator.{name}: a phase method takes self first, as a "
                             "ResponseGenerator's methods do")
    for name, fn in methods.items():
        if "output" in getattr(fn, "__phase_options__", {}) and "return" in fn.__annotations__:
            raise PhaseError(f"phase {name!r} of {cls.__qualname__} has both @phase(output=...) and a "
                             "return annotation: use the return annotation when the facts type is "
                             "fixed (or a union for whichever ran), and output=... only when the type "
                             "depends on the input")
    order = list(methods)
    if _defines_call(cls):
        raise PhaseError("PlanGenerator defines __call__; rename it to __phases__")
    problem = _order_problem(cls)
    if problem:
        raise PhaseError(problem)
    custom = _custom_phases(cls)
    instance = cls()

    def chosen(input) -> list[str]:
        if custom is None:
            return order
        names = []
        for method in custom(instance, input) or ():
            name = getattr(method, "__name__", None)
            if getattr(method, "__self__", None) is not instance or name not in methods:
                raise PhaseError(f"{cls.__qualname__}.__phases__ must return the plan's own "
                                 f"phase methods (self.find, ...), got {method!r}")
            if getattr(method, "__func__", None) is not methods[name]:
                raise PhaseError(f"{cls.__qualname__}.__phases__ returns a base's {name} that "
                                 f"the class overrides: return self.{name}, and call the "
                                 f"base's from it with super().{name}(...)")
            names.append(name)
        return names

    def plan(input):
        phases = []
        for name in chosen(input):
            fn = methods[name]
            options = dict(getattr(fn, "__phase_options__", {}))
            settings: dict[str, Any] = {"output": _output(cls, fn, options, input)}
            for key in ("max_items", "bind", "noun", "optional"):
                if key in options:
                    settings[key] = _decided(options[key], input)
            steps, writes_steps = _phase_functions(instance, fn, input)
            if "over" in options:
                settings["over"] = _decided(options["over"], input)
                items = [p for p in _parameters(fn) if p not in {"input", *order}]
                if len(items) != 1:
                    raise PhaseError(f"phase {name!r} runs once per item (over=...): it needs "
                                     "one parameter for the item besides input and earlier "
                                     f"phases, got {items}")
                settings["item"] = items[0]
                until = _decided(options.get("until"), input)
                if until is not None:
                    settings["until"] = _with_input(until, input)
            else:
                settings["when"] = writes_steps
            phases.append(Phase(name, steps, **settings))
        return phases

    plan.__qualname__ = f"{cls.__qualname__}.plan"
    return plan


def _runs_flat(phases: list[Phase]) -> bool:
    """A call runs flat when it has one phase that fills no facts, reads none and is not a
    fan-out or optional: the runtime then runs it as one plan named after the planning command."""
    if len(phases) != 1:
        return False
    ph = phases[0]
    return (ph.output is None and not ph.fans_out and not ph.optional
            and not inspect.signature(ph.steps_fn).parameters)


def planning_command_parts(cls: type) -> tuple[Callable, Optional[Callable]]:
    """``(steps_fn, plan_fn)`` of a ``PlanGenerator`` class, as
    ``planning_commands.planning_command_from_parts`` takes them: ``plan_fn(input)`` is the phases a call
    runs, or None when it runs flat; ``steps_fn(input)`` is a flat call's steps, or the
    phases' template (``concatenated_steps``). ``plan_fn`` is None when every call runs flat
    (one phase method, reading only ``input``, no ``__phases__``, no options, no facts)."""
    phases_of = planned(cls)

    def plan_fn(input):
        phases = phases_of(input)
        return None if _runs_flat(phases) else phases

    def steps_fn(input):
        phases = phases_of(input)
        if _runs_flat(phases):
            return phases[0].steps_fn()
        return concatenated_steps(phases)

    methods = phase_methods(cls)
    has_result = any("__result__" in klass.__dict__ for klass in cls.__mro__)
    always_flat = (len(methods) == 1 and _custom_phases(cls) is None
                   and not getattr(methods[0][1], "__phase_options__", None)
                   and _parameters(methods[0][1]) in ([], ["input"])
                   and _output(cls, methods[0][1], {}, None) is None
                   and not has_result)
    return steps_fn, (None if always_flat else plan_fn)


def run_code(workflow_folder: str, command: str, input: Optional[dict] = None,
             start_with: tuple[str, ...] = ()):
    """One call of *command* on the workflow at *workflow_folder*, run with no model, as ``fastworkflow command
    run-code`` runs it (``fastworkflow.planning_code_run``): a planning command phase by phase, stopping at its
    first open step, or an execution command's CommandOutput. Returns a ``CodeRun`` (``text``, ``answer``,
    ``output``, ``facts``, ``open_step``); raises ``RunCodeError`` when the command does not load, a step fails
    or a hook raises. *start_with* are steps run by code first. Its steps are real commands on a real session:
    use it on a development instance, and in tests on a copy of the workflow.
    """
    from fastworkflow.planning_code_run import run_code as _run_code

    return _run_code(workflow_folder, command, input, start_with)
