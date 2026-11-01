"""Planning command phases: a planning command run as bounded sub-plans, with facts passed between them.

A planning command file writes its phases as the methods of ``class PlanGenerator``
(``fastworkflow.planning``); the loader builds this module's runtime form from it: a list of
``Phase`` per call (``planning.planned``), each method's steps as its ``steps_fn`` and its
``return []`` as its ``when`` (decided when the phase is reached; the method runs once). The
runtime form, as the loader builds it or a test writes it by hand::

    Phase("find", find_steps, output=Found)
    Phase("item", item_steps, output=ItemFacts, when=item_writes_steps)
    Phase("book", book_steps, output=Book, over="find.uids", item="uid", max_items=3)

``Phase(name, steps_fn, output=None, when=True, over=None, item="item", max_items=None,
bind=True, until=None, noun=None, optional=False)``:

- ``name`` is a Python identifier, unique in the plan.
- ``steps_fn`` returns the phase's steps (a list of str). Its parameters are bound BY NAME
  to the outputs of earlier phases: ``lambda find: ...`` gets the ``find`` phase's filled
  ``output`` model, as a typed object. A zero-argument ``steps_fn`` gets nothing. A
  parameter is bound only when a phase of that name ran earlier and declares an
  ``output``; otherwise it keeps its default, and one with no default fails the phase
  (``PhaseError``) -- so a plan that leaves a phase out on its input builds its later
  phases without that parameter, while a phase skipped at run time (a callable ``when``,
  a failed ``optional`` phase) is read through a parameter defaulting to ``None``.
- ``output`` is a facts model (``fastworkflow.observation_offloading.facts``): each field
  names the command field it is copied from. It is filled in code after the phase ran.
  ``None``: the phase passes nothing on (typically the last phase, which writes the answer).
- ``when=False`` leaves the phase out, the same as not listing it. ``when`` may also be a
  callable whose parameters bind by name to earlier phases' outputs, like ``steps_fn``
  (``when=lambda find: bool(find.item_uids)``). It is called just before the phase would
  run (``runs_now``); a false result skips the phase, which then counts as not run: a
  later parameter named after it keeps its default. See 8 below.
- ``over`` makes the phase a fan-out, run once per item (see 7 below); ``steps_fn``
  gets the item as its parameter named ``item``. ``until`` (a fan-out only) stops it
  early once the items' outputs so far meet a condition (see 9 below). ``noun`` (a
  fan-out only) is the plural noun its items are called in the answer's notes
  (``"people"``, ``"holders"``); without it the notes say "items".
- ``optional=True``: a phase the request does not need; its failure does not stop the
  job (see 10 below).

A call whose ``PlanGenerator`` gives one phase that reads and fills no facts is not phased:
it runs flat (``is_flat_plan``: the loader's plan gives ``None``), as one phase named after
the planning command (``planning_commands.call_phases``). A planning command can so be phased for some inputs only.

The planning command's template (what planning command matching, the planning command call's plan text and the template
fallback read) is the phases' steps joined with every earlier fact a ``Placeholder``
(``concatenated_steps``).

How the runner (``WorkflowExecutionContext``) runs a phased planning command call, phase by phase:

1. ``steps_fn`` is called with the earlier outputs bound; its steps are the phase's plan.
   The planner does not rewrite them. The planning command call's extra steps the request asked for before the planning command's own (``steps_before``) run in a
   phase of their own before the first phase (``steps_before_phase``); those after go
   after the last phase's steps (``steps_after``); extra COMMAND steps after an answer-only
   last phase run in a phase of their own just before it (``extra_steps_phase``), so the
   last phase stays answer-only.
2. The phase runs as its own sub-plan: a fresh executor run with a step budget of
   ``phase_step_budget()`` (``PHASE_MAX_STEPS``, default 15), its own finish
   check against its own steps, its step numbers continuing the turn's. There is no
   step-cap hand-off: a phase that reaches its budget simply ends.
   The executor of a phase other than the last sees only that phase's steps, framed
   as one part of a larger job, and not the request (``phase_query``): it does those
   steps and finishes. The last phase also sees the request, which it answers.
3. Its ``output`` is filled from the command runs archived during the phase
   (``runs_for_facts``). The phase's RECORD is the one the workflow is on when the
   phase's run ends. A command that ran on a record (a ``"<Context> <label>"``
   clause), or entered one, counts only on the phase's record: the runs that ran on
   it plus the run that entered it (``runs_on_record``). A command that ran only on
   bare contexts (the root, a context with no label) counts wherever the phase
   ended, so a phase that looks something up from a hub keeps its lookup when it
   then opened a record. Runs of earlier phases never count. Of these, only runs that
   do one of the phase's own steps count (``runs_on_plan``, ``finish_check.satisfies``):
   a step giving a concrete value (``find_item query=Ann``) takes only runs with that
   value (any page of it), so an off-plan lookup never fills the fact; a step whose
   argument is a placeholder or prose takes any run of its command with arguments.
   A planned lookup run in another form (``find_item query=An`` for ``query=Ann``, or
   with no query) that did not move the workflow still counts when no run does the
   step itself and its OUTPUT names every value the step gives (``Ann`` among the
   matches; ``runs_finding_planned_values``). With only off-plan runs, the fill fails
   saying the planned run was not found.
   A step that leaves an argument open (``entity_type=<the family it names>``) may be
   run with several values: when some of those reads listed rows and others listed
   nothing, the ones that listed nothing were probes of the wrong value and are
   dropped (``runs_without_empty_probes``), so the facts are the read that found the
   records, not the last read. A re-read of the same query is never a probe.
   One exception: a planned step that opens the record the workflow is already on
   cannot run (the agent is told it is already there, and nothing runs). When no run
   of the phase does such a step, the earlier run of the turn that does it, that
   entered the phase's record, and whose parameters (the uid it opened) are words of
   the record's clause stands in for it (``runs_that_entered_record``), so the opening
   command's facts come from the run that really opened the record.
4. A fill that fails (``MissingFactError``, or values that do not fit the model), a
   ``steps_fn`` that fails, or a parameter with nothing to bind stops the planning command call:
   no later phase runs, nothing is guessed, and the sub-task's answer is the facts the
   phases that ran filled, rendered for a person (``found_facts``: labels in words, short
   lists), followed by a line naming the phase that failed and why (``failure_text``). No text the phases' runs wrote is shown, so
   nothing the failed phase made up reaches the answer. An ``optional`` phase is the
   exception (10 below).
5. The sub-task's answer is the LAST phase's answer: the last phase writes the answer,
   from the facts its steps carry (built in code by its ``steps_fn``; ``facts_text``
   renders a facts model as ``field: value`` text for them).
6. ANSWER-ONLY phases: a phase whose steps name no command (``is_answer_only``: no
   command, conditional or navigation step as ``finish_check.parse_plan`` reads them) is
   not an executor run. One model call with no tools (``write_answer``) writes its answer
   from the request and its steps, told "Do not run any command; write the answer from
   these facts only." (``ANSWER_ONLY_RULE``) and to name every item of every list. It
   cannot call a workflow command. When every step is verbatim (``say that ...``) or a
   guard (``apply no change``), code writes the answer
   from the steps with no model call (``answer_by_code``, ``answer_step_kind``).
   Its answer is checked in code (``answer_only_answer``):
   field markers and echoed framing are stripped (``clean_answer``), an empty reply (or
   only a code fence) or one that cannot be parsed is retried once and then written in
   code from the steps, and the
   rows of the lists the steps give in full that it does not name are appended. A long
   list the steps give only as background is cut short in its prompt (``capped_steps``).
7. FAN-OUT phases (``Phase(..., over=...)``): the phase runs once per item, each item its
   own sub-plan (its own executor run, finish check and fill). ``over`` is a list of items
   or ``"<earlier phase>.<list field>"``; ``fan_out_items`` drops empty values and repeats
   and keeps at most ``fan_out_cap`` (``max_items``, else ``FANOUT_MAX_ITEMS``, 10);
   a capped fan-out adds ``cap_note`` to the answer. With ``bind`` (default)
   each item is bound to the record that names it (``item_record``): its output is filled
   on that record, its finish check flags a step run on another item's record and an item
   never opened (``item_record_issues``), and an item with an output that was never
   opened fails the phase. A later phase's parameter of the fan-out phase's name gets the
   list of the per-item outputs, in item order.
8. A callable ``when`` is evaluated when the runner reaches the phase (never again on a
   resume after ask_user). A skipped phase runs nothing, fills nothing and writes no
   answer; its ``fw.skill.phase`` span has ``skipped``. A ``when`` that raises, or a
   parameter with nothing to bind, fails the phase. When the last phase is skipped, the
   answer is that of the last phase that ran.
9. ``until`` (``Phase(..., over=..., until=callable)``) is called after each item's
   output is filled (``stops_after``). Its parameter named after the phase gets the list
   of the outputs so far (typed, in item order); its other parameters bind to earlier
   phases' outputs. True stops the fan-out: the remaining items do not run.
   ``max_items`` still caps the items run. The answer then ends with ``fan_out_note``:
   it says the fan-out stopped on its condition, or (the condition never met) that it
   was capped.
10. An ``optional`` phase that fails (its fill, its ``steps_fn``, its ``when``, its
   fan-out items) does not stop the call: it counts as not run (no output, so a later
   parameter named after it keeps its default; its runs' answers are dropped), its span
   records the error with ``optional: true``, and the next phase runs.
11. STEPS RUN BY CODE (``CodeSteps``): code runs a
   phase's leading mechanical steps (``code_step``) through the agent's own command tool
   before the executor's model takes a step, each a normal trajectory step marked "run
   by code". An open step that opens the one row of the previous step's Output that
   names its lookup value exactly (or its only row) runs too (``pick_step``).
   It stops at the first open step or a failed step, and the agent does the rest. A
   step whose note asks for the full list (``FULL_LIST_PHRASES``) is paged by code: it
   runs each later page the output names, up to ``CODE_MAX_PAGES`` pages. A phase run
   wholly by code makes no model call, except the one answer call of a phase whose
   answer is shown (the last phase, the extra-steps phase). Never for planner-written
   steps or on a resume.

Each phase run is a ``fw.skill.phase`` span (see ``tracing.SPAN_CONTRACTS``).
"""
from __future__ import annotations

import contextlib
import contextvars
import inspect
import json
import re
import types
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Union, get_args, get_origin

from pydantic import BaseModel

from fastworkflow.observation_offloading import finish_check
from fastworkflow.observation_offloading.facts import facts_text
from fastworkflow.observation_offloading.labels import context_clause

#: The executor step budget of one phase. A module constant: the tests monkeypatch it.
PHASE_MAX_STEPS = 15


class PhaseError(ValueError):
    """A phase that cannot be built or run: a bad ``Phase``, ``steps_fn`` or binding."""


#: The most items one fan-out phase runs, unless its ``max_items`` says otherwise.
FANOUT_MAX_ITEMS = 10

_REFERENCE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)$")


class _FromStr(str):
    """A ``from_`` source string: behaves identically to ``str``, while allowing
    load-time resolution to distinguish ``from_(...)`` from ``fact_from(...)`` or
    a hand-written ``json_schema_extra={"from": ...}`` on a facts model."""


class _FromList(list):
    """A ``from_`` source list: behaves identically to ``list[str]``, while allowing
    load-time resolution to distinguish ``from_(...)`` from ``fact_from(...)``."""


def is_from_source(value: Any) -> bool:
    """Whether *value* was constructed by :func:`from_`."""
    return isinstance(value, (_FromStr, _FromList))


def output_members(annotation: Any) -> tuple[type[BaseModel], ...]:
    """The pydantic models a phase output annotation stands for, in order.

    A model is its own one member. A union (``X | Y``, ``Union[X, Y]``, and ``Optional[X]``
    with None skipped) is its members (design 3.12: a plain name several contexts define).
    Empty for anything else, or for a union with a member that is not a model.
    """
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return (annotation,)
    if get_origin(annotation) not in (Union, types.UnionType):
        return ()
    members: list[type[BaseModel]] = []
    for arg in get_args(annotation):
        if arg is type(None):
            continue
        if not (inspect.isclass(arg) and issubclass(arg, BaseModel)):
            return ()
        members.append(arg)
    return tuple(members)


def placeholder_member(annotation: Any, field: Optional[str] = None) -> Any:
    """The model a phase's placeholder output is built from. A union's: the first member that
    has *field* (a field a later phase reads), else the first member; which member a run gives
    is decided when it runs. Any other annotation is returned as it is."""
    members = output_members(annotation)
    if not members:
        return annotation
    if field is not None:
        return next((m for m in members if field in m.model_fields), members[0])
    return members[0]


def from_(source: str | Sequence[str], **field_kwargs: Any) -> Any:
    """Declare a producer edge on a command's ``Signature.Input`` field: ``Field(**field_kwargs)``
    with ``{"from": source}`` merged into the caller's ``json_schema_extra`` (other keys kept).

    ``book_uid: str = from_("find_book.book_uids", description="the uid of the book to open")``
    """
    from pydantic import Field

    marked: Any
    if isinstance(source, str):
        marked = _FromStr(source)
    elif isinstance(source, Sequence):
        marked = _FromList(_FromStr(s) if isinstance(s, str) else s for s in source)
    else:
        marked = source
    extra = {**(field_kwargs.pop("json_schema_extra", None) or {}), "from": marked}
    return Field(json_schema_extra=extra, **field_kwargs)


def fact_from(source: str | Sequence[str], **kwargs: Any) -> Any:
    """A facts-model field copied from ``"<command>.<field>"``: ``Field(**kwargs)`` with
    ``json_schema_extra={"from": source}`` (merged into any ``json_schema_extra`` given,
    e.g. ``{"page_by": "cursor"}``). *source* may also be a list of such names.

    ``uid: str = fact_from("open_book_by_uid.book_uid")``
    """
    from pydantic import Field

    extra = {**(kwargs.pop("json_schema_extra", None) or {}), "from": source}
    return Field(json_schema_extra=extra, **kwargs)


@dataclass(frozen=True)
class Phase:
    """One phase of a planning command's ``plan()``: a bounded sub-plan with a typed output.

    With ``over`` the phase is a FAN-OUT: it runs once per item, each item its own
    sub-plan. ``over`` is either the items themselves (a list, e.g. names taken from
    the input) or ``"<earlier phase>.<list field>"``, a list field of an earlier
    phase's output (e.g. the uids a search found). ``steps_fn`` gets the item as its
    parameter named ``item`` (default ``"item"``). ``max_items`` caps the items run
    (default ``FANOUT_MAX_ITEMS``, 10). ``bind``: each item is bound to the
    record whose ``"<Context> <label>"`` clause names it (see ``item_record``).
    A later phase's parameter named after a fan-out phase gets the LIST of the
    per-item outputs, in item order. ``until`` stops a fan-out early: called after each
    item with the outputs so far (its parameter named after the phase), true stops it.
    ``noun``: the plural noun for the items in the answer's notes (``cap_note``,
    ``fan_out_note``), e.g. ``"people"``; None says "items". The notes never name the phase.

    ``when`` is a bool, or a callable of earlier phases' outputs (bound by parameter
    name, like ``steps_fn``) evaluated just before the phase runs (``runs_now``).

    ``optional``: the request does not need this phase (it reads something the answer
    can do without). When it fails, the job does not stop: the phase counts as not run
    (no output, no answer text) and the later phases run (see 10 below). A phase is
    required by default; a planning command computes ``optional`` from its input in ``plan()``.
    """

    name: str
    steps_fn: Callable[..., list[str]]
    output: Optional[type[BaseModel]] = None
    when: Any = True
    over: Any = None
    item: str = "item"
    max_items: Optional[int] = None
    bind: bool = True
    until: Optional[Callable[..., Any]] = None
    noun: Optional[str] = None
    optional: bool = False

    def __post_init__(self):
        if isinstance(self.over, list):
            object.__setattr__(self, "over", tuple(self.over))

    @property
    def has_condition(self) -> bool:
        """Whether ``when`` is a callable, decided on earlier facts when the phase is reached."""
        return callable(self.when)

    @property
    def fans_out(self) -> bool:
        """Whether the phase runs once per item (``over`` is set)."""
        return self.over is not None

    @property
    def over_reference(self) -> Optional[tuple[str, str]]:
        """``(phase, field)`` when ``over`` names an earlier phase's list field."""
        if isinstance(self.over, str):
            match = _REFERENCE_RE.match(self.over.strip())
            if match:
                return match.group(1), match.group(2)
        return None


class Placeholder(str):
    """A fact that is not known yet: what a phase method gets for each earlier phase's
    output (and a fan-out's item) while the TEMPLATE is built (``building_template``).

    Lenient for reading: any attribute, index or call gives another placeholder named by
    its path (``<find.total>``, ``<find.uids[0]>``, ``<find.only_uid()>``); iterating gives
    one item, ``<find.labels[*]>`` (each of them). A number the code converts reads 999999.

    EVERY TEST ON IT READS AS TRUE: ``bool()``, ``len()`` (one item), ``==``, ``!=``, ``<``,
    ``in`` ... so the template shows the steps written when each condition holds. Each test
    is noted while ``tests_noted`` is open (``find.uids``, ``find.total > 3``): the template
    says the phase's steps are written "only when" they hold. ``x is None`` is not a test
    it can see (a placeholder is never None).

    A fact named like a public ``str`` method (``title``, ``count``, ``format``, ``split``,
    ...) still reads as the fact, not the method (``__getattribute__``): only a name this
    class means itself (``index``, below) or a dunder keeps ``str``'s own meaning.
    """

    def __new__(cls, path: str):
        self = super().__new__(cls, f"<{path}>")
        self._path = path
        return self

    __hash__ = str.__hash__

    def __getattribute__(self, name):
        # A public name only ``str`` defines, not this class (``title``, ``count``,
        # ``format``, ``split``, ...): raise so ``__getattr__`` reads it as the fact's own
        # attribute, instead of the str method a fact named like one would otherwise hit.
        if (not name.startswith("_") and name not in type(self).__dict__
                and hasattr(str, name)):
            raise AttributeError(name)
        return object.__getattribute__(self, name)

    def __getattr__(self, name):
        if name.startswith("__") or name == "_path":
            raise AttributeError(name)
        return Placeholder(f"{self._path}.{name}")

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self
        return Placeholder(f"{self._path}[{key!r}]" if isinstance(key, str)
                           else f"{self._path}[{key}]")

    def __call__(self, *args, **kwargs):
        return Placeholder(f"{self._path}()")

    def __iter__(self):
        _noted(self._path)
        yield Placeholder(f"{self._path}[*]")

    def index(self, *args, **kwargs):
        """As a list's: the one item's place, 0 (not ``str.index``)."""
        return 0

    def __len__(self):
        _noted(self._path)
        return 1

    def __bool__(self):
        _noted(self._path)
        return True

    def __contains__(self, item):
        _noted(f"{_expr(item)} in {self._path}")
        return True

    def _compared(self, op: str, other: Any) -> bool:
        _noted(f"{self._path} {op} {_expr(other)}")
        return True

    def __eq__(self, other):
        return self._compared("==", other)

    def __ne__(self, other):
        return self._compared("!=", other)

    def __lt__(self, other):
        return self._compared("<", other)

    def __le__(self, other):
        return self._compared("<=", other)

    def __gt__(self, other):
        return self._compared(">", other)

    def __ge__(self, other):
        return self._compared(">=", other)

    def __int__(self):
        return 999999

    def __float__(self):
        return 999999.0

    def __str__(self):
        return f"<{self._path}>"

    __repr__ = __str__

    def __format__(self, spec):
        return str(self)


def _expr(value: Any) -> str:
    return value._path if isinstance(value, Placeholder) else repr(value)


def is_placeholder(value: Any) -> bool:
    """Whether *value* is a fact not known yet (``Placeholder``)."""
    return isinstance(value, Placeholder)


_BUILDING_TEMPLATE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "fw_planning_command_building_template", default=False)
_TESTS: contextvars.ContextVar[Optional[list]] = contextvars.ContextVar(
    "fw_planning_command_tests", default=None)


@contextlib.contextmanager
def building_template():
    """Set while the framework builds a phased planning command's TEMPLATE (``concatenated_steps``,
    ``placeholder_steps``): phase methods then get ``Placeholder`` facts. Never set while
    a phase runs."""
    token = _BUILDING_TEMPLATE.set(True)
    try:
        yield
    finally:
        _BUILDING_TEMPLATE.reset(token)


def is_building_template() -> bool:
    """Whether the template is being built (``building_template``)."""
    return _BUILDING_TEMPLATE.get()


@contextlib.contextmanager
def tests_noted():
    """A list of the tests made on placeholders inside the block (``find.uids``,
    ``find.total > 3``), each once, in order."""
    tests: list = []
    token = _TESTS.set(tests)
    try:
        yield tests
    finally:
        _TESTS.reset(token)


def _noted(test: str) -> None:
    tests = _TESTS.get()
    if tests is not None and test not in tests:
        tests.append(test)


def _template_kwargs(phase: Phase, earlier: Optional[set] = None) -> dict[str, Any]:
    """Every parameter of *phase*'s ``steps_fn`` (a fact, or the fan-out's item) as a
    ``Placeholder`` named after it. With *earlier* (the names of the earlier phases of the
    call that fill facts), a parameter with a default that names none of them keeps its
    default: that phase is not in the call, so its facts are never there."""
    return {
        parameter.name: Placeholder(parameter.name) for parameter in _parameters(phase)
        if parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
        and (earlier is None or parameter.default is parameter.empty
             or parameter.name in earlier or (phase.fans_out and parameter.name == phase.item))
    }


def phase_step_budget() -> int:
    """The executor step budget of one phase: the module constant ``PHASE_MAX_STEPS`` (15)."""
    return PHASE_MAX_STEPS


def fan_out_cap(phase: Optional[Phase] = None) -> int:
    """The most items a fan-out phase runs: its ``max_items``, else ``FANOUT_MAX_ITEMS``."""
    if phase is not None and phase.max_items:
        return int(phase.max_items)
    return FANOUT_MAX_ITEMS


def is_flat_plan(result: Any) -> bool:
    """Whether a ``plan()`` result means "run this call flat": ``None`` or an empty list.

    The planning command's ``steps(input)`` is then used exactly as for a planning command without ``plan()``.
    """
    return result is None or (isinstance(result, (list, tuple)) and not result)


def active_phases(phases: Any, where: str = "plan()") -> list[Phase]:
    """The phases of a ``plan()`` result that may run, checked: ``when`` true, or a
    callable ``when`` (decided when the phase is reached, ``runs_now``).

    Raises PhaseError when the result is not a list of ``Phase``, a name is not an
    identifier or repeats, ``steps_fn`` is not callable, ``output`` is not a pydantic
    model, or ``until`` is set on a phase that does not fan out or is not callable.
    """
    if isinstance(phases, (str, bytes)) or not isinstance(phases, (list, tuple)):
        raise PhaseError(f"{where} must return a list of Phase")
    seen: set[str] = set()
    out = []
    for phase in phases:
        if not isinstance(phase, Phase):
            raise PhaseError(f"{where} must return a list of Phase, got {type(phase).__name__}")
        if not isinstance(phase.name, str) or not phase.name.isidentifier():
            raise PhaseError(f"{where}: a phase name must be a Python identifier, got {phase.name!r}")
        if phase.name in seen:
            raise PhaseError(f"{where}: two phases are named {phase.name!r}")
        seen.add(phase.name)
        if not callable(phase.steps_fn):
            raise PhaseError(f"{where}: phase {phase.name!r}: steps_fn must be callable")
        if phase.output is not None:
            from fastworkflow.typed_phase_facts import union_members

            if union_members(phase.output) is None and not (
                    inspect.isclass(phase.output) and issubclass(phase.output, BaseModel)):
                raise PhaseError(f"{where}: phase {phase.name!r}: output must be a pydantic model")
        if phase.fans_out:
            if isinstance(phase.over, str) and phase.over_reference is None:
                raise PhaseError(
                    f"{where}: phase {phase.name!r}: over must be a list of items or "
                    f"'<earlier phase>.<field>', got {phase.over!r}")
            if not isinstance(phase.over, (str, tuple)):
                raise PhaseError(
                    f"{where}: phase {phase.name!r}: over must be a list of items or "
                    f"'<earlier phase>.<field>', got {type(phase.over).__name__}")
            if not isinstance(phase.item, str) or not phase.item.isidentifier():
                raise PhaseError(f"{where}: phase {phase.name!r}: item must be a Python "
                                 f"identifier, got {phase.item!r}")
            if phase.max_items is not None and (
                    isinstance(phase.max_items, bool) or not isinstance(phase.max_items, int)
                    or phase.max_items < 1):
                raise PhaseError(f"{where}: phase {phase.name!r}: max_items must be a "
                                 f"positive int, got {phase.max_items!r}")
        if phase.noun is not None:
            if not phase.fans_out:
                raise PhaseError(f"{where}: phase {phase.name!r}: noun needs a fan-out "
                                 "phase (over=...)")
            if not isinstance(phase.noun, str) or not phase.noun.strip():
                raise PhaseError(f"{where}: phase {phase.name!r}: noun must be a non-empty "
                                 f"str, got {phase.noun!r}")
        if not isinstance(phase.optional, bool):
            raise PhaseError(f"{where}: phase {phase.name!r}: optional must be a bool, got "
                             f"{phase.optional!r}")
        if phase.until is not None:
            if not phase.fans_out:
                raise PhaseError(f"{where}: phase {phase.name!r}: until needs a fan-out "
                                 "phase (over=...)")
            if not callable(phase.until):
                raise PhaseError(f"{where}: phase {phase.name!r}: until must be callable")
        if phase.has_condition or phase.when:
            out.append(phase)
    return out


def _parameters(phase: Phase, fn: Optional[Callable] = None,
                what: str = "steps_fn") -> list[inspect.Parameter]:
    try:
        return list(inspect.signature(fn or phase.steps_fn).parameters.values())
    except (TypeError, ValueError) as error:
        raise PhaseError(f"phase {phase.name!r}: cannot read {what}'s parameters: {error}") from None


def _bound(phase: Phase, fn: Callable, what: str, outputs: Mapping[str, Any], *,
           phase_names: frozenset[str] | None = None) -> dict[str, Any]:
    """*fn*'s keyword arguments: each parameter named like an entry of *outputs*.

    With *phase_names*, a parameter named after a phase of the call that did not run or
    filled nothing receives ``None`` instead of its default. ``*args`` / ``**kwargs``
    receive nothing.
    """
    kwargs: dict[str, Any] = {}
    for parameter in _parameters(phase, fn, what):
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if parameter.name in outputs:
            kwargs[parameter.name] = outputs[parameter.name]
        elif phase_names is not None and parameter.name in phase_names:
            kwargs[parameter.name] = None
        elif parameter.default is parameter.empty:
            raise PhaseError(
                f"phase {phase.name!r}: {what} needs {parameter.name!r}, but no earlier phase "
                f"of that name ran with an output")
    return kwargs


def _called(phase: Phase, fn: Callable, what: str, kwargs: Mapping[str, Any]) -> bool:
    try:
        return bool(fn(**kwargs))
    except Exception as error:  # noqa: BLE001 - workflow code; the phase fails cleanly
        raise PhaseError(
            f"phase {phase.name!r}: {what} failed: {type(error).__name__}: {error}") from None


def runs_now(phase: Phase, outputs: Mapping[str, Any], *,
             phase_names: frozenset[str] | None = None) -> bool:
    """Whether *phase* runs, now that the runner reached it.

    A bool ``when`` was already applied by ``plan()`` (``active_phases``). A callable
    ``when`` is called with its parameters bound by name to *outputs* (the earlier phases
    that ran with an output, as for ``steps_fn``); its result is read as a bool. Raises
    PhaseError when a parameter cannot be bound or ``when`` raises.
    """
    if not phase.has_condition:
        return bool(phase.when)
    return _called(phase, phase.when, "when",
                    _bound(phase, phase.when, "when", outputs, phase_names=phase_names))


def stops_after(phase: Phase, so_far: Sequence[Any], outputs: Mapping[str, Any], *,
                phase_names: frozenset[str] | None = None) -> bool:
    """Whether the fan-out *phase* stops after the items that ran (``until``).

    ``until``'s parameter named after the phase gets *so_far*, the list of the items'
    outputs so far (typed, in item order); its other parameters bind by name to
    *outputs*, the earlier phases' outputs. A phase without ``until`` never stops early.
    Raises PhaseError when a parameter cannot be bound or ``until`` raises.
    """
    if phase.until is None:
        return False
    merged = {**outputs, phase.name: list(so_far)}
    return _called(phase, phase.until, "until",
                   _bound(phase, phase.until, "until", merged, phase_names=phase_names))


#: ``item`` of ``bound_arguments`` / ``phase_steps`` when no item is given.
NO_ITEM: Any = object()


def bound_arguments(phase: Phase, outputs: Mapping[str, Any],
                    item: Any = NO_ITEM, *,
                    phase_names: frozenset[str] | None = None) -> dict[str, Any]:
    """``steps_fn``'s keyword arguments: each parameter named like an earlier phase's output.

    *outputs* maps the name of each earlier phase that ran and filled an output to that
    output (a fan-out phase's: the list of its per-item outputs). With *phase_names*, a
    parameter named after a phase of the call that did not run or filled nothing receives
    ``None`` instead of its default. Without *phase_names*, a parameter with no output of
    its name keeps its default; one with no default raises PhaseError. ``*args`` /
    ``**kwargs`` receive nothing. For a fan-out phase, the parameter named ``phase.item``
    receives *item* (when given).
    """
    kwargs: dict[str, Any] = {}
    for parameter in _parameters(phase):
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if phase.fans_out and parameter.name == phase.item:
            if item is not NO_ITEM:
                kwargs[parameter.name] = item
            elif parameter.default is parameter.empty:
                kwargs[parameter.name] = None
            continue
        if parameter.name in outputs:
            kwargs[parameter.name] = outputs[parameter.name]
        elif phase_names is not None and parameter.name in phase_names:
            kwargs[parameter.name] = None
        elif parameter.default is parameter.empty:
            raise PhaseError(
                f"phase {phase.name!r}: steps_fn needs {parameter.name!r}, but no earlier phase "
                f"of that name ran with an output"
            )
    return kwargs


def _checked_steps(lines: Any, phase: Phase) -> list[str]:
    from fastworkflow.planning import Step

    if isinstance(lines, str) or not isinstance(lines, (list, tuple)) or \
            not all(isinstance(line, str) for line in lines):
        raise PhaseError(f"phase {phase.name!r}: steps_fn must return a list of str")
    return [
        (Step(line.strip(), line.helper) if isinstance(line, Step) else line.strip())
        for line in lines if line.strip()
    ]


def phase_steps(phase: Phase, outputs: Mapping[str, Any], item: Any = NO_ITEM, *,
                phase_names: frozenset[str] | None = None) -> list[str]:
    """The phase's steps, ``steps_fn`` called with the earlier outputs (and *item*) bound.

    Raises PhaseError when a parameter cannot be bound, ``steps_fn`` raises, or it
    returns something other than a list of str.
    """
    kwargs = bound_arguments(phase, outputs, item, phase_names=phase_names)
    try:
        lines = phase.steps_fn(**kwargs)
    except Exception as error:  # noqa: BLE001 - workflow code; the phase fails cleanly
        raise PhaseError(
            f"phase {phase.name!r}: steps_fn failed: {type(error).__name__}: {error}") from None
    return _checked_steps(lines, phase)


def _distinct(values: Iterable[Any]) -> list[Any]:
    out: list[Any] = []
    for value in values:
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        if value not in out:
            out.append(value)
    return out


def fan_out_items(phase: Phase, outputs: Mapping[str, Any]) -> tuple[list[Any], int]:
    """``(items, total)``: the items a fan-out phase runs, capped, and how many there were.

    The items are ``phase.over`` itself (a list) or the list field it names of an earlier
    phase's output (a fan-out phase's outputs: that field of each, joined). Empty values
    and repeats are dropped, order kept. At most ``fan_out_cap(phase)`` are kept.
    Raises PhaseError when the named phase did not run with an output or its field is
    not there.
    """
    reference = phase.over_reference
    if reference is None:
        values: list[Any] = list(phase.over or ())
    else:
        source, field_name = reference
        if source not in outputs:
            raise PhaseError(f"phase {phase.name!r}: over={phase.over!r}, but phase {source!r} "
                             "did not run with an output")
        found = outputs[source]
        values = []
        for model in (found if isinstance(found, list) else [found]):
            if field_name not in type(model).model_fields:
                raise PhaseError(f"phase {phase.name!r}: over={phase.over!r}: "
                                 f"{type(model).__name__} has no field {field_name!r}")
            value = getattr(model, field_name)
            values.extend(value if isinstance(value, (list, tuple)) else [value])
    items = _distinct(values)
    return items[:fan_out_cap(phase)], len(items)


def concatenated_steps(phases: Iterable[Phase]) -> list[str]:
    """The phases' steps joined in order, each built with no earlier outputs: the TEMPLATE.

    The flat ``steps(input)`` of a phased planning command. Built under ``building_template``: every
    parameter of ``steps_fn`` naming an earlier phase of the call (its facts) or a fan-out's
    item is a ``Placeholder``, so the steps read as if every phase ran; one naming a phase
    the call does not have keeps its default. Phases with ``when`` false
    are left out; a phase with a callable ``when`` is kept (it may run). A fan-out phase
    over a list of items gives its steps once per item (capped); one over an earlier
    phase's field gives them once, with its item a placeholder.
    """
    if is_flat_plan(phases):
        raise PhaseError("plan() returned no phases (run flat): the planning command's flat steps are "
                         "its own, not concatenated_steps(...)")
    out: list[str] = []
    earlier: set[str] = set()
    from fastworkflow.planning import _PHASE_STEPS, note_phase_steps

    with building_template():
        for phase in active_phases(list(phases)):
            kwargs = _template_kwargs(phase, earlier)
            if phase.output is not None:
                earlier.add(phase.name)
            seen = _PHASE_STEPS.get()
            before = len(seen) if seen is not None else 0
            if phase.fans_out and phase.over_reference is None:
                for item in fan_out_items(phase, {})[0]:
                    checked = _checked_steps(phase.steps_fn(**{**kwargs, phase.item: item}),
                                             phase)
                    if seen is not None and len(seen) == before:
                        note_phase_steps(phase.name, checked)
                    before = len(seen) if seen is not None else 0
                    out.extend(checked)
                continue
            checked = _checked_steps(phase.steps_fn(**kwargs), phase)
            if seen is not None and len(seen) == before:
                note_phase_steps(phase.name, checked)
            out.extend(checked)
    return out


def placeholder_steps(phase: Phase) -> list[str]:
    """*phase*'s steps as the template has them (``concatenated_steps``; one item's, the
    item a placeholder, for a fan-out). Raises PhaseError."""
    try:
        with building_template():
            return _checked_steps(phase.steps_fn(**_template_kwargs(phase)), phase)
    except PhaseError:
        raise
    except Exception as error:  # noqa: BLE001 - workflow code
        raise PhaseError(
            f"phase {phase.name!r}: steps_fn failed: {type(error).__name__}: {error}") from None


#: The name of the phase that runs a planning command call's extra command steps (``steps_after``)
#: when the last phase is answer-only (``extra_steps_phase``).
EXTRA_STEPS_PHASE = "extra_steps"


def extra_steps_phase(phases: Sequence[Phase], steps_after: Sequence[str], *,
                      commands: Iterable[str] = (),
                      core_commands: Iterable[str] = ()) -> tuple[Optional[Phase], list[str]]:
    """``(phase, kept)``: where a planning command call's extra steps after its own go.

    The extra steps the planner kept from the request (``steps_after``) go after the last
    phase's steps. When the last phase is meant to be answer-only (its steps, built with
    no earlier outputs, name no command: ``is_answer_only``), a COMMAND step among them
    would make it an executor run and take away the answer-only handling (the list rules
    and the append). Those steps then run in a phase of their own, ``EXTRA_STEPS_PHASE``,
    placed just before the last phase (``with_phase_before``); the other extra steps
    (``say ...``) stay with the last phase (*kept*). Otherwise *phase* is None and *kept*
    is every step. The last phase stays answer-only.
    """
    from fastworkflow.planning_commands import number

    steps_after = [str(step) for step in steps_after if str(step).strip()]
    commands, core_commands = list(commands), list(core_commands)
    if not steps_after or not phases:
        return None, steps_after
    try:
        last = placeholder_steps(phases[-1])
    except PhaseError:
        return None, steps_after
    if not is_answer_only(number(last), commands, core_commands):
        return None, steps_after
    moved = [step for step in steps_after
             if not is_answer_only(number([step]), commands, core_commands)]
    if not moved:
        return None, steps_after
    return (Phase(_free_phase_name(EXTRA_STEPS_PHASE, phases), lambda: list(moved)),
            [step for step in steps_after if step not in moved])


#: The name of the phase that runs a planning command call's extra steps before its own
#: (``steps_before``): ``steps_before_phase``.
STEPS_BEFORE_PHASE = "extra_steps_before"


def steps_before_phase(phases: Sequence[Phase], steps_before: Sequence[str]) -> Optional[Phase]:
    """The phase that runs the extra steps the planner kept before the planning command's own
    (``steps_before``), or None when there are none.

    It runs first (``with_phase_before(phases, phase, 0)``), as a phase of its own: the
    agent runs its steps (code never runs planner-written steps), and the planning command's first
    phase then runs as it does with no extra steps before it, its code-run steps included.
    The workflow's current context and the facts carry over as between any two phases.
    """
    steps_before = [str(step) for step in steps_before if str(step).strip()]
    if not steps_before:
        return None
    return Phase(_free_phase_name(STEPS_BEFORE_PHASE, phases), lambda: list(steps_before))


def _free_phase_name(name: str, phases: Sequence[Phase]) -> str:
    """*name*, or *name* with a number (``name_2``, ...) when a phase is already so named."""
    names, base, k = {phase.name for phase in phases}, name, 2
    while name in names:
        name, k = f"{base}_{k}", k + 1
    return name


def with_phase_before(phases: Sequence[Phase], phase: Optional[Phase],
                      before: int = -1) -> list[Phase]:
    """*phases* with *phase* inserted before the phase at *before* (the last by default);
    *phases* unchanged when *phase* is None."""
    phases = list(phases)
    if phase is None:
        return phases
    index = before if before >= 0 else len(phases) + before
    return phases[:index] + [phase] + phases[index:]


def check_bindings(phases: Sequence[Phase], where: str = "plan()") -> dict[tuple[str, str, str], bool]:
    """Check, without running anything, that every phase's parameters can be bound.

    A parameter without a default must name an earlier phase that declares an output
    (a fan-out phase's item parameter is bound to its item). A fan-out phase's
    ``"<phase>.<field>"`` must name an earlier phase with an output that has that field.
    The same holds for a callable ``when``'s parameters, and for ``until``'s, which may
    also name the phase itself (its outputs so far) when it declares an output.
    Raises PhaseError.

    Returns, for each parameter whose default is None (a fact parameter that may be left
    unbound), ``(phase, "steps_fn" | "when" | "until", parameter)`` -> whether it binds in
    this plan. The planning command loader fails on one that binds in no plan the inputs give (a
    misspelled phase name). A parameter with another default (``family=family``, a value
    the function keeps) is not a fact parameter and is not returned.
    """
    defaults: dict[tuple[str, str, str], bool] = {}
    with_output: dict[str, tuple[type[BaseModel], ...]] = {}
    earlier_names: set[str] = set()
    for phase in phases:
        reference = phase.over_reference
        if reference is not None:
            source, field_name = reference
            if source not in with_output:
                raise PhaseError(f"{where}: phase {phase.name!r}: over={phase.over!r} names "
                                 f"{source!r}, which is not an earlier phase with an output")
            members = with_output[source]
            if not any(field_name in member.model_fields for member in members):
                # a member's module is its command's file: find_book.Output, not Output
                names = " | ".join(f"{member.__module__.rsplit('.', 1)[-1]}.{member.__name__}"
                                   for member in members) or "no model"
                raise PhaseError(f"{where}: phase {phase.name!r}: over={phase.over!r}: "
                                 f"{names} has no field {field_name!r}")
        if phase.fans_out and phase.item in with_output:
            raise PhaseError(f"{where}: phase {phase.name!r}: its item parameter {phase.item!r} "
                             "is also an earlier phase's name; give item= another name")
        for parameter in _parameters(phase):
            if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                continue
            if phase.fans_out and parameter.name == phase.item:
                continue
            if parameter.default is None:
                defaults[(phase.name, "steps_fn", parameter.name)] = parameter.name in earlier_names
            if parameter.default is parameter.empty and parameter.name not in earlier_names:
                raise PhaseError(
                    f"{where}: phase {phase.name!r} takes {parameter.name!r}, which is not an "
                    "earlier phase of this call; give the parameter a default or build the "
                    "phase without it"
                )
        conditions = [("when", phase.when, set())] if phase.has_condition else []
        if phase.until is not None:
            own = {phase.name} if phase.output is not None else set()
            conditions.append(("until", phase.until, own))
        for what, fn, also in conditions:
            for parameter in _parameters(phase, fn, what):
                if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                    continue
                if parameter.default is None:
                    defaults[(phase.name, what, parameter.name)] = \
                        parameter.name in earlier_names or parameter.name in also
                if parameter.default is parameter.empty and \
                        parameter.name not in earlier_names and parameter.name not in also:
                    raise PhaseError(
                        f"{where}: phase {phase.name!r}: {what} takes {parameter.name!r}, "
                        "which is not an earlier phase of this call"
                        + (" (or this phase, which has an output)" if what == "until" else "")
                        + "; give the parameter a default")
        if phase.output is not None:
            with_output[phase.name] = output_members(phase.output)
        earlier_names.add(phase.name)
    return defaults


def _entered(runs: Sequence[Mapping[str, Any]], index: int, record: Optional[str]) -> Any:
    """The clause the workflow was on after run *index*: the next run's, or *record* for the last."""
    return runs[index + 1].get("context_clause") if index + 1 < len(runs) else record


def runs_on_record(runs: Sequence[Mapping[str, Any]], record: Optional[str]) -> list[dict[str, Any]]:
    """The runs of *runs* (one phase's, in run order) that belong to *record*.

    *record* is the ``"<Context> <label>"`` clause the phase ended on. A run belongs
    when it ran on the record, or when it is the run that entered the record: the
    next run ran on it, or it is the last run and the phase ended on it. The
    entering run is returned with its ``context_clause`` set to *record*, so a
    ``facts_from_runs(..., context=record)`` keeps it.
    """
    out: list[dict[str, Any]] = []
    for index, run in enumerate(runs):
        if run.get("context_clause") == record:
            out.append(dict(run))
        elif _entered(runs, index, record) == record:
            out.append({**run, "context_clause": record})
    return out


def is_record_clause(clause: Any) -> bool:
    """Whether *clause* names a record: ``"<Context> <label>"``, not a bare context or the root.

    A context name is one identifier, so a clause with a label has a space in it.
    """
    return " " in str(clause or "").strip()


def record_bound_commands(runs: Sequence[Mapping[str, Any]], record: Optional[str]) -> set[str]:
    """The commands of *runs* (one phase's) that ran on a record, or entered one.

    A run counts as run on the clause it ran on and, when it moved the workflow, on
    the clause it moved it to (see ``runs_on_record``). A command none of whose runs
    touched a record ran only on bare contexts (the root, a context with no label).
    """
    bound: set[str] = set()
    for index, run in enumerate(runs):
        if is_record_clause(run.get("context_clause")) or is_record_clause(_entered(runs, index, record)):
            bound.add(str(run.get("command") or ""))
    return bound


def runs_for_facts(runs: Sequence[Mapping[str, Any]], record: Optional[str]) -> list[dict[str, Any]]:
    """The runs of one phase its output is filled from, all labelled with *record*.

    - A command that ran on a record, or entered one, during the phase counts only
      on the phase's *record* (``runs_on_record``): a phase never reads another
      record's runs.
    - A command that ran only on bare contexts (the root, a context with no label)
      during the phase counts wherever the phase ended: what a phase looked up from
      a context with no record stays its own when it then opened a record.

    *runs* are the phase's own runs, in run order; runs of earlier phases are never
    passed in.
    """
    bound = record_bound_commands(runs, record)
    out: list[dict[str, Any]] = []
    for index, run in enumerate(runs):
        on_record = (run.get("context_clause") == record
                     or _entered(runs, index, record) == record)
        if on_record or str(run.get("command") or "") not in bound:
            out.append({**run, "context_clause": record})
    return out


def _plain(command: Any) -> str:
    return str(command or "").strip().split("/")[-1].lower()


def planned_steps(plan_text: str, commands: Iterable[str] = (),
                  core_commands: Iterable[str] = ()) -> dict[str, list[Any]]:
    """The phase's command steps (``finish_check.parse_plan``), by plain command name.

    Command and conditional steps only: a negated, answer or navigation-only step
    plans no run.
    """
    from fastworkflow.observation_offloading import finish_check

    by_command: dict[str, list[Any]] = {}
    for step in finish_check.parse_plan(plan_text, list(commands), list(core_commands)):
        if step.kind in (finish_check.KIND_COMMAND, finish_check.KIND_CONDITIONAL) and step.command:
            by_command.setdefault(step.command.lower(), []).append(step)
    return by_command


def runs_on_plan(runs: Sequence[Mapping[str, Any]], steps: Mapping[str, Sequence[Any]],
                 names: Iterable[str] = ()) -> list[dict[str, Any]]:
    """The runs of *runs* that do one of the phase's planned *steps* (``planned_steps``).

    A run of a command some step names counts only when it satisfies one of that
    command's steps (``finish_check.satisfies``): a step giving a concrete value
    (``find_x query=Ann``) takes only runs with that value, so an off-plan lookup of
    another value never counts; each page of the planned query does. A step whose
    argument is a placeholder or prose takes any run of its command with arguments.
    A run of a command no step names is kept: the earlier rules decide it.
    """
    from fastworkflow.observation_offloading import finish_check

    names = {str(n).lower() for n in names}
    out = []
    for run in runs:
        group = steps.get(_plain(run.get("command")))
        if group is None or any(finish_check.satisfies(step, run, names) for step in group):
            out.append(dict(run))
    return out


def _text_leaves(value: Any) -> Iterable[str]:
    """Every scalar of a command's output (nested lists and mappings walked), as text."""
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _text_leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _text_leaves(item)
    elif value is not None and not isinstance(value, bool):
        yield str(value)


def output_names_values(run: Mapping[str, Any], values: Iterable[Any]) -> bool:
    """Whether *run*'s output names every one of *values*: each is one scalar of the
    output, whole (a label, a uid; compared after ``finish_check.plain_value``: spacing,
    outer quotes and case ignored), so ``Dune`` is not named by ``Dune Messiah``.
    An output that names nothing (no output, no values) never does."""
    from fastworkflow.observation_offloading import finish_check

    leaves = {finish_check.plain_value(leaf) for leaf in _text_leaves(run.get("output"))}
    values = [finish_check.plain_value(value) for value in values]
    return bool(leaves) and bool(values) and all(value and value in leaves for value in values)


def narrowed_output(output: Any, values: Iterable[Any]) -> Optional[dict[str, Any]]:
    """*output* cut down to the one row that holds *values*; None when no single row does.

    Rows are read as ``missing_list_items`` reads them: the output's list fields of one
    length are the columns of one table (uids and their labels), row *i* holding item *i*
    of each. A value found in some row is looked for in the rows; one found in no list
    must be a scalar of the output outside them (a scope the listing echoes). Exactly one
    row (of one table) must hold every value found in rows; that table's columns are then
    cut to that row, and every other field is kept as it is. With no value in any row the
    output is kept whole. Values compare after ``finish_check.plain_value``, whole.
    """
    from fastworkflow.observation_offloading import finish_check

    if not isinstance(output, Mapping):
        return None
    plain = finish_check.plain_value
    values = [plain(value) for value in values if plain(value)]
    lists = {key: value for key, value in output.items() if isinstance(value, (list, tuple))}
    cells = {plain(leaf) for value in lists.values() for leaf in _text_leaves(value)}
    outside = {plain(leaf) for key, value in output.items() if key not in lists
               for leaf in _text_leaves(value)}
    in_rows = [value for value in values if value in cells]
    if any(value not in cells and value not in outside for value in values):
        return None
    if not in_rows:
        return dict(output)
    hits: list[tuple[int, int]] = []
    for length in sorted({len(value) for value in lists.values()}):
        table = [value for value in lists.values() if len(value) == length]
        for i in range(length):
            row = {plain(leaf) for column in table for leaf in _text_leaves(column[i])}
            if all(value in row for value in in_rows):
                hits.append((length, i))
    if len(hits) != 1:
        return None
    length, i = hits[0]
    return {key: ([value[i]] if key in lists and len(value) == length else value)
            for key, value in output.items()}


def _finding_planned_values(runs: Sequence[Mapping[str, Any]],
                            steps: Mapping[str, Sequence[Any]],
                            done: Sequence[Mapping[str, Any]],
                            names: Iterable[str]) -> dict[int, dict[str, Any]]:
    """``runs_finding_planned_values`` by position in *runs*: index -> the narrowed run."""
    from fastworkflow.observation_offloading import finish_check

    names = {str(n).lower() for n in names}
    unmet = [(command, step) for command, group in steps.items() for step in group
             if step.arguments and not any(
                 _plain(run.get("command")) == command and run.get("success")
                 and finish_check.satisfies(step, run, names) for run in done)]
    out: dict[int, dict[str, Any]] = {}
    for index, run in enumerate(runs):
        if not run.get("success") or run.get("moved"):
            continue
        for command, step in unmet:
            if _plain(run.get("command")) != command or \
                    not output_names_values(run, step.arguments.values()):
                continue
            narrowed = narrowed_output(run.get("output"), step.arguments.values())
            if narrowed is not None:
                out[index] = {**run, "output": narrowed}
                break
    return out


def runs_finding_planned_values(runs: Sequence[Mapping[str, Any]],
                                steps: Mapping[str, Sequence[Any]],
                                done: Sequence[Mapping[str, Any]] = (),
                                names: Iterable[str] = ()) -> list[dict[str, Any]]:
    """Runs of a planned command that ran with other values, but whose OUTPUT holds the planned ones.

    The executor often runs a planned lookup in another form than planned: the query
    spelled differently (``query=Ann Le`` for ``query=Ann Lee``, fuzzy-corrected by the
    command), cut short by parameter extraction (``query=Ann``), or not given at all
    (the whole catalogue listed). Such a run does not satisfy the step's values
    (``finish_check.satisfies``), yet its output holds the record the step was after.
    For each of the phase's planned *steps* that gives concrete values (``step.arguments``)
    and that no run of *done* (the runs that do a step) does, this returns the runs of
    *runs* (in run order) that:

    - are runs of the step's command, succeeded and have an output;
    - did not move the workflow (``moved`` false): a lookup or listing, never an opener,
      so a run that opened another record never stands in for one that opens this one;
    - name every concrete value of the step in their output (``output_names_values``):
      the planned name among the matches, the planned uid among the rows; and
    - hold them in ONE row of the output (``narrowed_output``).

    Each run is returned with its output cut down to that row, so a fact read from it
    (the first match's uid) is the planned record's, never another row's. A run whose
    output does not name the planned values (a lookup of another name, another scope),
    or names them in no single row, is never returned.
    """
    return list(_finding_planned_values(runs, steps, done, names).values())


def _names_record(run: Mapping[str, Any], record: Optional[str]) -> bool:
    """Whether every parameter *run* was given is a word of the *record* clause.

    The clause is ``"<Context> <instance label>"``; an opening run's parameters (the
    uid it opened) name the record it opened when the label carries them. A run given
    no parameter names nothing, so it never matches.
    """
    from fastworkflow.observation_offloading import finish_check

    given = finish_check._given_inputs(run.get("input"))
    clause = finish_check.plain_value(record)
    return bool(given) and all(
        re.search(r"(?<![A-Za-z0-9_])" + re.escape(finish_check.plain_value(value))
                  + r"(?![A-Za-z0-9_])", clause) is not None
        for value in given.values())


def runs_that_entered_record(earlier_runs: Sequence[Mapping[str, Any]],
                             runs: Sequence[Mapping[str, Any]], record: Optional[str],
                             steps: Mapping[str, Sequence[Any]],
                             names: Iterable[str] = ()) -> list[dict[str, Any]]:
    """Earlier runs that stand in for a planned step the phase found already done.

    A phase whose step opens a record the workflow is already on cannot run it: the
    agent is told it is already there, and no command runs. The step is then done by
    the run that put the workflow on that record before the phase began. For each of
    the phase's planned *steps* that none of its own *runs* (as archived, in run order)
    does, this returns the latest of *earlier_runs* (the turn's runs before the phase,
    in run order) that:

    - succeeded and does that step (``finish_check.satisfies``: the step's concrete
      values, e.g. the uid it opens, are the ones the run ran with);
    - ENTERED *record*: it ran on another clause and the next run of the turn ran on
      *record* (or it is the turn's last run and the phase ended on *record*); and
    - names *record*: each parameter it was given (the uid it opened) is a word of the
      *record* clause (``"<Context> <instance label>"``).

    Each is returned with its ``context_clause`` set to *record*. Nothing is returned
    when *record* is not a record clause. A run that only ran on the record (a listing)
    never stands in: only the run that moved the workflow onto it does, so a phase
    never takes an earlier phase's readings, and never claims a record it is not on.
    """
    from fastworkflow.observation_offloading import finish_check

    if not is_record_clause(record) or not steps:
        return []
    names = {str(n).lower() for n in names}
    turn = list(earlier_runs) + list(runs)
    entering = [
        run for index, run in enumerate(turn[:len(earlier_runs)])
        if run.get("success") and run.get("context_clause") != record
        and _entered(turn, index, record) == record and _names_record(run, record)
    ]
    out: list[dict[str, Any]] = []
    for command, group in steps.items():
        for step in group:
            if any(_plain(run.get("command")) == command and run.get("success")
                   and finish_check.satisfies(step, run, names) for run in runs):
                continue
            doing = [run for run in entering if _plain(run.get("command")) == command
                     and finish_check.satisfies(step, run, names)]
            if doing:
                carried = {**doing[-1], "context_clause": record}
                if carried not in out:
                    out.append(carried)
    return out


def runs_standing_in_for_openers(earlier_runs: Sequence[Mapping[str, Any]],
                                 runs: Sequence[Mapping[str, Any]], record: Optional[str],
                                 steps: Mapping[str, Sequence[Any]], sources: Iterable[Any],
                                 done: Sequence[Mapping[str, Any]] = (),
                                 names: Iterable[str] = ()) -> list[dict[str, Any]]:
    """Runs of the planned opening commands, stood in for by another command that entered *record*.

    A phase's opening step (``list_x x_uid=U``, which opens record U) is done just as
    well by a successful run of ANOTHER command that entered the same record with the
    same value under the same parameter name (``open_x_by_uid x_uid=U``). For each of
    the phase's planned *steps* that gives concrete values (``step.arguments``) and
    that no run of *done* does (``finish_check.satisfies``), this looks through the
    turn's runs (*earlier_runs* then the phase's *runs*, in run order) for the latest
    run that:

    - succeeded, and is not a run of the step's command;
    - was given each of the step's arguments, under the same name, with the same
      value (after ``plain_value``);
    - ENTERED *record*: it ran on another clause and the next run ran on *record*
      (or it is the last run and the phase ended on *record*); and
    - names *record* (each parameter it was given is a word of the clause).

    For such a run, one run of the step's command is returned for each command name
    the phase output's *sources* (``facts.fact_sources``) spell for it: its ``input``
    holds the stand-in's values for the step's arguments, it has no output, and its
    ``context_clause`` is *record*. So facts sourced from the planned command's Input
    fields take the value; its Output fields stay unfilled. Nothing is returned when
    *record* is not a record clause.
    """
    from fastworkflow.observation_offloading import finish_check

    if not is_record_clause(record) or not steps:
        return []
    names = {str(n).lower() for n in names}
    turn = list(earlier_runs) + list(runs)
    entering = [
        run for index, run in enumerate(turn)
        if run.get("success") and run.get("context_clause") != record
        and _entered(turn, index, record) == record and _names_record(run, record)
    ]
    spelled: dict[str, list[str]] = {}
    for source in sources:
        command = str(getattr(source, "command", "") or "")
        if command and command not in spelled.setdefault(_plain(command), []):
            spelled[_plain(command)].append(command)
    out: list[dict[str, Any]] = []
    for command, group in steps.items():
        if command not in spelled:
            continue
        for step in group:
            if not step.arguments or any(
                    _plain(run.get("command")) == command and run.get("success")
                    and finish_check.satisfies(step, run, names) for run in done):
                continue
            standing = []
            for run in entering:
                if _plain(run.get("command")) == command:
                    continue
                given = finish_check._given_inputs(run.get("input"))
                if all(key in given and finish_check.plain_value(given[key]) == value
                       for key, value in step.arguments.items()):
                    standing.append(run)
            if not standing:
                continue
            given = finish_check._given_inputs(standing[-1].get("input"))
            for name in spelled[command]:
                carried = {**standing[-1], "command": name, "output": None,
                           "input": {key: given[key] for key in step.arguments},
                           "context_clause": record, "stands_in": True}
                if carried not in out:
                    out.append(carried)
    return out


def planned_record(runs: Sequence[Mapping[str, Any]], plan_text: Optional[str], *,
                   ended_on: Optional[str] = None, commands: Iterable[str] = (),
                   core_commands: Iterable[str] = ()) -> Optional[str]:
    """The one record the phase's planned steps ran on, when the run ended elsewhere; else None.

    The successful runs of *runs* (one phase's, in run order) of a command one of
    the phase's planned command steps names -- with any values; runs of commands no
    step names are left out -- are read for the record clauses each ran on or moved
    the workflow to. When those runs touched records and all of them the same one,
    and *ended_on* (the clause the run ended on) is not it, that record is returned:
    the phase did its steps there, then wandered off with other commands. A run that
    moved from one record to another touches both, so a phase whose steps go from one
    record to the next, or that ran a step's command on another record, is never
    re-bound. None when no such run touched a record, or they touched several.
    """
    if not plan_text or not str(plan_text).strip():
        return None
    names = set(commands) | set(core_commands)
    names |= {str(run.get("command") or "") for run in runs if run.get("command")}
    steps = planned_steps(plan_text, names, core_commands)
    if not steps:
        return None
    # Every successful run of a command a step names, whatever its values: a run of
    # the step's command on another record (opening b2 for a step that opens b1)
    # is the step done on the wrong record, not a wander, so it blocks the re-bind.
    on_plan = {id(run) for run in runs
               if _plain(run.get("command")) in steps and run.get("success")}
    records: set[str] = set()
    for index, run in enumerate(runs):
        if id(run) not in on_plan:
            continue
        for clause in (run.get("context_clause"), _entered(runs, index, ended_on)):
            if is_record_clause(clause):
                records.add(str(clause))
    if len(records) != 1:
        return None
    (record,) = records
    return None if record == ended_on else record


def runs_without_empty_probes(runs: Sequence[Mapping[str, Any]],
                              steps: Mapping[str, Sequence[Any]],
                              sources: Iterable[Any] = (),
                              names: Iterable[str] = ()) -> list[dict[str, Any]]:
    """*runs* without the reads that probed a step's open argument and listed nothing.

    A step that leaves an argument open (a placeholder or prose: ``entity_type=<the
    family it names>``) lets the agent try more than one value. When the successful
    runs of one command that satisfy one such step (``finish_check.satisfies``) read
    two or more QUERIES that give the same parameters and differ only in the value of
    a parameter the step does not fix, and some of those queries listed rows while
    the others listed nothing (``facts._listed_nothing`` on every page), the queries
    that listed nothing were probes of the wrong value: their runs are dropped, so
    the facts come from the read that found the records, not from the last read.

    Nothing else changes: a re-read of the same query (the same values: a listing
    read again after a change) is never a probe, a query that gives other parameters
    is left to the "broadest query" rule, and when every query listed nothing, or
    every one listed rows, the runs are kept as they are (the latest still wins).
    Queries and pages are told apart as ``facts_from_runs`` tells them (``_page_key``),
    with the paging parameters of *sources* (the model's ``fact_sources``).
    """
    from fastworkflow.observation_offloading import finish_check
    from fastworkflow.observation_offloading.facts import (
        PAGING_PARAMETERS, _listed_nothing, _page_key)

    names = {str(n).lower() for n in names}
    paging_of: dict[str, set[str]] = {}
    for source in sources:
        paging_of.setdefault(_plain(source.command), set()).update(source.page_by)
    dropped: set[int] = set()
    for command, group in steps.items():
        paging = set(PAGING_PARAMETERS) | paging_of.get(command, set())
        for step in group:
            mine = [index for index, run in enumerate(runs)
                    if _plain(run.get("command")) == command and run.get("success")
                    and finish_check.satisfies(step, run, names)]
            queries: dict[str, list[int]] = {}
            for index in mine:
                query, _page, _size = _page_key(runs[index].get("input"), paging, None)
                queries.setdefault(query, []).append(index)
            if len(queries) < 2:
                continue
            given = [json.loads(query) for query in queries]
            if not all(isinstance(q, dict) for q in given) or \
                    len({frozenset(q) for q in given}) != 1:
                continue  # other parameters: not values of one open argument
            empty = {query for query, indexes in queries.items()
                     if all(_listed_nothing(runs[i].get("output")) for i in indexes)}
            if empty and len(empty) < len(queries):
                dropped.update(i for query in empty for i in queries[query])
    return [dict(run) for index, run in enumerate(runs) if index not in dropped]


def _step_text(step: Any) -> str:
    return f"step {step.number} ({' '.join(str(step.text).split())})"


def fill_output(model: type[BaseModel], runs: Sequence[Mapping[str, Any]], *,
                record: Optional[str], record_context: Optional[str] = None,
                plan_text: Optional[str] = None, commands: Iterable[str] = (),
                core_commands: Iterable[str] = (),
                earlier_runs: Sequence[Mapping[str, Any]] = (),
                ended_on: Optional[str] = None) -> BaseModel:
    """*model* filled from one phase's *runs* (``runs_for_facts``) on its *record*.

    *record_context* is the record's context name, which plain command names resolve
    against. With *plan_text* (the phase's numbered steps), only runs that do one of
    its steps count (``runs_on_plan``); *commands* / *core_commands* are the
    workflow's command names the steps are read with (the runs' and the model's
    source commands are always added). A required field whose source command a step
    names, and that no run doing that step holds, fails saying the planned run was
    not found; it never falls back to an off-plan run, except a planned lookup run in
    another form whose output holds every value the step gives
    (``runs_finding_planned_values``). Raises ``FactsError``
    (``MissingFactError`` for a required field no run holds). Reads of a step's open
    argument with values that listed nothing, beside one that listed rows, do not count
    (``runs_without_empty_probes``).

    *earlier_runs* are the turn's runs before the phase, in run order. A planned step
    none of the phase's runs does, because the workflow was already on the record it
    opens, is done by the earlier run that entered that record
    (``runs_that_entered_record``); that run counts as the phase's, before its own runs.

    A planned opening step with concrete values that no run does is done by a
    successful run of another command that entered *record* with the same values
    under the same parameter names (``runs_standing_in_for_openers``); facts sourced
    from the planned command's Input fields take those values.

    *ended_on*: the clause the run ended on, when it is not *record* (a fan-out item
    bound to its record, ``item_record``, that moved on afterwards). The last run then
    entered *ended_on*, not *record*.
    """
    from fastworkflow.observation_offloading.facts import (
        MissingFactError, fact_sources, facts_from_runs)

    if ended_on is not None and ended_on != record:
        # A run that never succeeded, on the clause the run ended on: the real last run
        # is read as moving the workflow there, not onto *record*. It fills nothing.
        runs = list(runs) + [{"command": "", "context_clause": ended_on, "success": False,
                              "input": {}, "output": None}]
    # Whether each run moved the workflow (an opener), read before runs_for_facts
    # relabels every kept run with *record*.
    runs = [{**run, "moved": _entered(runs, index, record) != run.get("context_clause")}
            for index, run in enumerate(runs)]
    kept = runs_for_facts(runs, record)
    steps: dict[str, list[Any]] = {}
    if plan_text and str(plan_text).strip():
        names = set(commands) | set(core_commands)
        names |= {str(run.get("command") or "") for run in runs if run.get("command")}
        names |= {source.command for source in fact_sources(model)}
        steps = planned_steps(plan_text, names, core_commands)
        plain_names = {_plain(n) for n in names}
        on_plan = runs_on_plan(kept, steps, plain_names)
        # A planned lookup run in another form, whose output holds the planned values.
        found = _finding_planned_values(kept, steps, on_plan, plain_names)
        kept = ([found.get(index) or run for index, run in enumerate(kept)
                 if index in found or run in on_plan] if found else on_plan)
        # An open argument probed with a value that listed nothing (fix-xhy3).
        kept = runs_without_empty_probes(kept, steps, fact_sources(model), plain_names)
        if earlier_runs:
            kept = runs_that_entered_record(earlier_runs, runs, record, steps, plain_names) + kept
        kept = runs_standing_in_for_openers(
            earlier_runs, runs, record, steps, fact_sources(model), done=kept,
            names=plain_names) + kept
    from fastworkflow.typed_phase_facts import (
        fill_typed_phase_output, handles_typed_output, validate_phase_output_value,
    )

    try:
        if handles_typed_output(model):
            filled = fill_typed_phase_output(
                model, kept, context=record, plan_text=plan_text,
            )
            return validate_phase_output_value(model, filled)
        return facts_from_runs(model, kept, context=record, context_name=record_context or None)
    except MissingFactError as error:
        source = getattr(error, "source", None)
        group = steps.get(_plain(source.command)) if source is not None else None
        if not group or any(_plain(run.get("command")) == _plain(source.command)
                            and run.get("success") for run in kept):
            raise
        planned = "; ".join(_step_text(step) for step in group)
        raise MissingFactError(
            f"{error}: the planned run was not found: no run of this phase does {planned}",
            source=source) from None


def is_answer_only(plan_text: str, commands: Iterable[str] = (),
                   core_commands: Iterable[str] = ()) -> bool:
    """Whether a phase's steps name no command: it only writes the answer.

    The steps are read as the finish check reads them (``finish_check.parse_plan``):
    answer-only when no step is a command or a conditional step, and no step is a
    navigation step (a core command the step does not negate or merely mention in an
    answer step). *commands* are the workflow's command names (and the phase output's
    source commands); with none known nothing is answer-only, so a phase is never
    kept from its commands because their names could not be read. An empty plan is
    not answer-only. A step's closing ``(only when ...)`` note (a template line written
    only under a condition) is not read: it does not make the step conditional.
    """
    from fastworkflow.observation_offloading import finish_check
    from fastworkflow.planning_commands import without_condition_note

    commands, core_commands = list(commands), list(core_commands)
    if not commands or not str(plan_text or "").strip():
        return False
    plan_text = "\n".join(without_condition_note(line) for line in str(plan_text).splitlines())
    steps = finish_check.parse_plan(plan_text, commands, core_commands)
    if not steps:
        return False
    core = {finish_check.plain_command_name(c).lower() for c in core_commands if c}
    for step in steps:
        if step.kind in (finish_check.KIND_COMMAND, finish_check.KIND_CONDITIONAL):
            return False
        # A navigation step reads as a command step when only the core commands count.
        if step.command and step.command.lower() in core and finish_check._classify(
                step.number, step.text, core, set()).kind in (
                    finish_check.KIND_COMMAND, finish_check.KIND_CONDITIONAL):
            return False
    return True


#: What an answer-only phase is told, after its steps' framing.
ANSWER_ONLY_RULE = "Do not run any command; write the answer from these facts only."
#: The rest of a phase's framing (``phase_query``). A reply that echoes any of it has it
#: stripped (``clean_answer``): it is an instruction, never part of the answer.
_DO_ONLY_THESE = "Do only these steps, then finish."
_LATER_PARTS = "Later parts will do the rest."
_EARLIER_PARTS = "Earlier parts already ran; their findings are in the steps."
_COPY_RULE = ("Copy every name, id and number exactly as the steps give it, and name every "
              "item of every list the steps give, in full: never shorten a list, count it "
              "instead of naming it, or write a placeholder for it.")
#: Said when code ran the phase's leading steps (``CodeSteps``) before the executor.
_CODE_STEPS_DONE = ("Steps marked 'run by code' in the trajectory are done; do the "
                    "remaining steps.")
#: Said to the last phase when code ran every one of its steps (``CodeSteps.done_inputs``).
_CODE_STEPS_ALL_DONE = ("Every step below has run (by code); their outputs are in the "
                        "trajectory. Answer from those outputs, and run a command only when "
                        "the request needs one that no step ran.")
_FRAMING_SENTENCES = (ANSWER_ONLY_RULE, _DO_ONLY_THESE, _LATER_PARTS, _EARLIER_PARTS, _COPY_RULE,
                      _CODE_STEPS_DONE, _CODE_STEPS_ALL_DONE)
#: "This is part 2 of 3 (items) for item 1 of 2 (u1), the last part of this job." and the
#: like: the opening of a phase's framing.
_PART_RE = re.compile(
    r"This is part \d+ of \d+ \([^()\n]*\)(?: for item \d+ of \d+ \([^\n]*?\))?"
    r"(?:,? (?:the last part of this job|of a larger job))?\s*[.:]?", re.IGNORECASE)
_ANSWER_REQUEST_RE = re.compile(r"(?:Do these steps, then )?[Aa]nswer the request"
                                r"(?: for this item only)?:\s*$", re.MULTILINE)


def phase_query(request: str, plan_text: str, *, index: int, count: int, name: str,
                answer_only: bool = False, item: Optional[tuple[int, int, Any]] = None,
                code_steps: bool = False, code_ran_all: bool = False,
                result_driven: bool = False) -> str:
    """The executor's input for phase *index* (0-based) of *count*: its steps, framed.

    A phase other than the last gets only its own steps: the request is left out, so
    the run cannot take on the rest of the job (later phases do it). The last phase
    gets the request too, since it writes the answer to it, unless *result_driven* is
    true (``__result__`` writes the call's answer). An *answer_only* phase is told to
    run no command and to write from the facts its steps carry (``ANSWER_ONLY_RULE``).
    *item* ``(k, n, value)`` (0-based *k*) frames one item's sub-plan of a fan-out phase.
    *code_steps*: code runs the phase's leading steps first (``CodeSteps``), and the run
    is told those are done (``_CODE_STEPS_DONE``). *code_ran_all*: code ran every step of
    the last phase; the run is told to answer from their outputs (``_CODE_STEPS_ALL_DONE``),
    with the copy rule answer-only phases get.
    """
    part = f"part {index + 1} of {count} ({name})"
    by_code = f" {_CODE_STEPS_DONE}" if code_steps else ""
    if item is not None:
        part += f" for item {item[0] + 1} of {item[1]} ({item[2]})"
    if result_driven or index < count - 1:
        if answer_only:
            return f"This is {part} of a larger job. {ANSWER_ONLY_RULE}\n{plan_text}"
        return (f"This is {part} of a larger job. {_DO_ONLY_THESE} "
                f"{_LATER_PARTS}{by_code}\n{plan_text}")
    earlier = f" {_EARLIER_PARTS}" if count > 1 else ""
    scope = " for this item only" if item is not None else ""
    if answer_only:
        return (f"{request}\n\nThis is {part}, the last part of this job.{earlier} "
                f"{ANSWER_ONLY_RULE} {_COPY_RULE} Answer the request{scope}:\n{plan_text}")
    if code_ran_all:
        return (f"{request}\n\nThis is {part}, the last part of this job.{earlier} "
                f"{_CODE_STEPS_ALL_DONE} {_COPY_RULE} Answer the request{scope}:\n{plan_text}")
    return (f"{request}\n\nThis is {part}, the last part of this job.{earlier}{by_code} "
            f"Do these steps, then answer the request{scope}:\n{plan_text}")


# ----------------------------------------------------------------------
# Steps run by code (fix-6llp)
# ----------------------------------------------------------------------

#: Words in a step's note (after the chain) that ask for every page of a paged command.
#: Only then does code read the later pages the output names (up to ``CODE_MAX_PAGES``);
#: any other note (or none) means page 1 is what the step wants.
FULL_LIST_PHRASES = ("full list", "every page", "all pages", "until complete", "page through")
#: Core commands code runs as a step of their own, written with nothing after them.
_CODE_NAVIGATION = ("go_up", "reset_context")
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _code_step_arguments(chain: str) -> Optional[dict[str, tuple[str, bool]]]:
    """``key -> (value, was_quoted)`` of *chain* as ``_code_step`` compares them against the
    bound model.

    A quoted value (``key="..."`` or ``key='...'``) is taken whole (its outer quote pair
    removed, internal quotes and placeholders kept); an unquoted value is read as
    ``finish_check.step_arguments`` reads it, and any placeholder or empty value returns
    None.
    """
    from fastworkflow.observation_offloading import finish_check

    probe = finish_check._QUOTED_VALUE_RE.sub(
        lambda m: "=" + "x" * (len(m.group(0)) - 1), chain
    )
    keys = list(finish_check._ARGUMENT_KEY_RE.finditer(probe))
    if not keys:
        return None
    out: dict[str, tuple[str, bool]] = {}
    for i, match in enumerate(keys):
        end = keys[i + 1].start() if i + 1 < len(keys) else len(chain)
        span = chain[match.end():end].strip()
        q = re.match(r"""^("[^"\n]*"|'[^'\n]*')\s*$""", span)
        if q:
            inner = q.group(1)[1:-1]
            if not inner:
                return None
            out[match.group(1)] = (" ".join(inner.split()).strip(" .,:").casefold(), True)
            continue
        if span.startswith(('"', "'")):
            return None
        raw = finish_check._value_text(span)
        value = finish_check.plain_value(raw)
        if not value or finish_check._PLACEHOLDER_RE.search(raw):
            return None
        out[match.group(1)] = (value, False)
    return out


def _code_step(step: Any, parameters_class: Any) -> tuple[Optional[str], str]:
    """``(command, reason)``: see ``code_step``; *reason* says why, in a few words."""
    from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import (
        ParameterExtraction)
    from fastworkflow.observation_offloading import finish_check

    if step.kind != finish_check.KIND_COMMAND or not step.command:
        return None, f"{step.kind} step"
    clean = step.text.replace("`", " ").replace("*", " ")
    first = _WORD_RE.search(clean)
    if first is None or first.group(0) != step.command:
        return None, "command not first"
    # The chain: the text after the command up to the first "(" or ";" outside a quoted
    # value (finish_check.step_chain). The rest is a note.
    chain = finish_check.step_chain(step.text[first.end():]).strip()
    fields = getattr(parameters_class, "model_fields", None) or {}
    if not chain:
        if any(info.is_required() for info in fields.values()):
            return None, "missing required field"
        return step.command, "bare" if fields else "no parameters"
    if not fields:
        return None, "prose after command"
    command = f"{step.command} {chain}"
    bound = ParameterExtraction._extract_parameters_from_key_values(
        command, parameters_class, step.command, code_written=True)
    if bound is None:
        return None, "not key=value"
    expected = _code_step_arguments(chain)
    if expected is None:
        raw_keys = list(finish_check._ARGUMENT_KEY_RE.finditer(
            finish_check._QUOTED_VALUE_RE.sub(
                lambda m: "=" + "x" * (len(m.group(0)) - 1), chain
            )
        ))
        for i, match in enumerate(raw_keys):
            end = raw_keys[i + 1].start() if i + 1 < len(raw_keys) else len(chain)
            span = chain[match.end():end].strip()
            if span.startswith(('"', "'")):
                return None, "prose in value"
        return None, "placeholder"
    given = set(expected)
    for key, (value, was_quoted) in expected.items():
        actual = getattr(bound, key, None)
        norm_actual = (" ".join(str(actual).split()).strip(" .,:").casefold()
                       if was_quoted and actual is not None
                       else finish_check.plain_value(actual))
        if norm_actual != value:
            return None, "prose in value"
    if any(info.is_required() and name not in given for name, info in fields.items()):
        return None, "missing required field"
    return command, "key=value"


def code_step(step: Any, parameters_class: Any) -> Optional[str]:
    """The command code runs for plan step *step*, or None when the step is open.

    *step* is a ``finish_check.PlanStep``; *parameters_class* is its command's parameters
    class (None for a command with none). Mechanical, read only with the existing parsers:
    a command step (``KIND_COMMAND``) whose first word is its command, and whose CHAIN (the
    text after the command, cut at the first ``(`` or ``;`` outside a quoted value,
    ``finish_check.step_chain``) is empty with every field defaulted, or binds with the
    key=value parser (which declines placeholders, examples and values that do not
    validate), every bound value the step's own value
    (``finish_check.step_arguments``, so prose swallowed into a value is caught) and every
    required field given. Code runs ``command chain``: the note after it is never sent.
    A paging value the step writes (``start=20``) is a value like any other.
    """
    return _code_step(step, parameters_class)[0]


_PICK_NAMED_RE = re.compile(r"\bwith the (?:uid|id) of the (.+?) match\b", re.IGNORECASE)


def _uid_parameter(step: Any,
                   parameters_classes: Sequence[Any]) -> tuple[Optional[str], str, str]:
    """``(uid_param, rest, reason)`` for a candidate pick *step*, or ``(None, "", reason)``."""
    from fastworkflow.observation_offloading import finish_check

    clean = step.text.replace("`", " ").replace("*", " ")
    first = _WORD_RE.search(clean)
    if step.kind != finish_check.KIND_COMMAND or not parameters_classes \
            or first is None or first.group(0) != step.command:
        return None, "", "not a pick"
    required = {tuple(name for name, info in (getattr(cls, "model_fields", None) or {}).items()
                      if info.is_required()) for cls in parameters_classes}
    # The one required field the step leaves to fill: a uid (its other required fields,
    # if any, it writes concretely).
    fields = [name for name in required.pop() if name not in step.arguments] \
        if len(required) == 1 else []
    if len(fields) != 1 or not _is_uid_field(fields[0]):
        return None, "", "not one required uid"
    # Any placeholder but the uid's leaves the step to the agent. The step's other values
    # stay as written; only the uid's placeholder is filled.
    chain = finish_check.step_chain(clean[first.end():]).strip()
    keys = list(finish_check._ARGUMENT_KEY_RE.finditer(chain))
    named = [match.group(1) for match in keys]
    if set(named) - {fields[0]} != set(step.arguments) or \
            (step.arguments and named.count(fields[0]) != 1):
        return None, "", "not a pick"
    rest = ""
    if step.arguments:
        at = named.index(fields[0])
        rest = " ".join(part.strip() for part in (
            chain[:keys[at].start()],
            chain[keys[at + 1].start():] if at + 1 < len(keys) else "") if part.strip())
    return fields[0], rest, ""


def _producer_sources(parameters_classes: Sequence[Any],
                      uid_param: str) -> list[tuple[str, str]]:
    """``(command, list_field)`` producer declarations on *uid_param* across *parameters_classes*."""
    sources: list[tuple[str, str]] = []
    for cls in parameters_classes:
        info = (getattr(cls, "model_fields", None) or {}).get(uid_param)
        extra = getattr(info, "json_schema_extra", None)
        if not isinstance(extra, Mapping):
            continue
        for key in ("available_from", "from"):
            raw = extra.get(key) or ()
            for entry in ([raw] if isinstance(raw, str) else raw):
                if isinstance(entry, str) and "." in entry:
                    cmd, _, field = entry.strip().rpartition(".")
                    if cmd and field and (cmd, field) not in sources:
                        sources.append((cmd, field))
    return sources


def _fitting_uid_column(uid_param: str, parameters_classes: Sequence[Any],
                        run_command: str, output: Any) -> Optional[str]:
    """The one uid list column of *output* that fits *uid_param*, or None."""
    from fastworkflow.observation_offloading.facts import _command_matches

    if not isinstance(output, Mapping):
        return None
    uids = [str(key) for key, value in output.items()
            if _is_uid_field(str(key)) and isinstance(value, (list, tuple))]
    if len(uids) != 1:
        return None
    col = uids[0]
    if col == f"{uid_param}s":
        return col
    sources = _producer_sources(parameters_classes, uid_param)
    if any(field == col and (not run_command or _command_matches(cmd, run_command))
           for cmd, field in sources):
        return col
    if "_" not in uid_param and not sources:
        return col
    return None


def _lookup_values_for_step(step: Any, lookup_or_run: Any) -> list[str]:
    """The non-paging argument values of *lookup_or_run* that *step* names."""
    from fastworkflow.observation_offloading import finish_check
    from fastworkflow.observation_offloading.facts import PAGING_PARAMETERS

    if isinstance(lookup_or_run, Mapping):
        args = finish_check._given_inputs(lookup_or_run.get("input"))
    else:
        args = getattr(lookup_or_run, "arguments", None) or {}
    match = _PICK_NAMED_RE.search(step.text)
    if match is not None:
        wanted = _folded(match.group(1))
        return [finish_check.plain_value(value) for key, value in args.items()
                if key not in PAGING_PARAMETERS and wanted and _folded(value) == wanted]
    text = _folded(step.text)
    return [finish_check.plain_value(value) for key, value in args.items()
            if key not in PAGING_PARAMETERS and _folded(value)
            and re.search(rf"(?<![a-z0-9_]){re.escape(_folded(value))}(?![a-z0-9_])", text)]


def pick_listing(
    step: Any,
    runs: Sequence[Mapping[str, Any]],
    parameters_classes: Sequence[Any],
) -> Optional[tuple[Mapping[str, Any], dict[str, Any]]]:
    """The latest run in *runs* (the call's own runs, in run order) whose arguments hold
    the value *step* names and whose ``Output`` has one uid list that fits *step*'s uid
    parameter, with the pages of that listing joined (``facts._join_list``); None when
    none fits. Never after a page that shows more pages."""
    from fastworkflow.observation_offloading.facts import (
        PAGING_PARAMETERS, _join_list, _page_key, _scalar_of_pages)

    uid_param, _, _ = _uid_parameter(step, parameters_classes)
    if uid_param is None:
        return None
    for run in reversed(runs):
        if not run.get("success") or not _lookup_values_for_step(step, run):
            continue
        if _fitting_uid_column(uid_param, parameters_classes,
                               str(run.get("command") or ""), run.get("output")) is None:
            continue
        if run.get("more_pages"):
            return None
        cmd = _plain(run.get("command"))
        query, _, _ = _page_key(run.get("input"), PAGING_PARAMETERS, None)
        pages = [r for r in runs if r.get("success") and _plain(r.get("command")) == cmd
                 and isinstance(r.get("output"), Mapping)
                 and _page_key(r.get("input"), PAGING_PARAMETERS, None)[0] == query]
        if len(pages) <= 1:
            return run, dict(run["output"])
        joined: dict[str, Any] = {}
        for key, val in run["output"].items():
            keyed = [(*_page_key(r.get("input"), PAGING_PARAMETERS, None),
                      r["output"][key], r["output"])
                     for r in pages if key in r["output"]]
            joined[key] = (_join_list(keyed) if isinstance(val, (list, tuple))
                           else _scalar_of_pages(keyed))
        return run, joined
    return None


def pick_step(step: Any, lookup: Any, output: Any,
              parameters_classes: Sequence[Any]) -> tuple[Optional[str], str]:
    """``(command, reason)`` for an open *step* that opens one row of a listing.

    *lookup* is the listing step or run (a ``finish_check.PlanStep`` or a command-run
    mapping) and *output* its Output (pages joined when read with ``full_list``,
    ``pick_listing``). Code runs *step* as ``command P=<uid>`` only when:

    - *step* is a command step whose first word is its command, and its command has the
      same required fields under every parameters class, of which the step leaves exactly
      one, P, unwritten: a uid (``_is_uid_field``). The step writes no value of its own,
      or its only placeholder is P's (``open_book book_uid=<the Dune uid> format=pdf``):
      its other values, all concrete, are sent as written;
    - *step* names a concrete value of *lookup* (``open_book_by_uid with the uid of the
      Dune match`` after ``find_book query=Dune``), paging values aside;
    - *output* has exactly one uid list that fits P (``<x>_uids`` for ``<x>_uid``, or the
      list a producer ``from_`` on P names), and
      exactly one row holds the named values whole (``Dune``, not ``Dune 1``; spacing,
      quotes and case ignored), or the output has exactly one row at all.

    Anything else (no such row, two exact matches, no rows) leaves the step to the agent:
    *command* is None and *reason* says why.
    """
    from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import (
        ParameterExtraction)
    from fastworkflow.observation_offloading import finish_check

    uid_param, rest, reason = _uid_parameter(step, parameters_classes)
    if uid_param is None:
        return None, reason
    values = _lookup_values_for_step(step, lookup)
    if not values:
        return None, "names no lookup value"
    run_command = str(lookup.get("command") if isinstance(lookup, Mapping)
                      else getattr(lookup, "command", "") or "")
    uid_col = _fitting_uid_column(uid_param, parameters_classes, run_command, output)
    if uid_col is None or not output[uid_col]:
        return None, "no uid column"
    rows = (narrowed_output(output, values) or output)[uid_col]
    if len(rows) != 1 or rows[0] is None or isinstance(rows[0], (Mapping, list, tuple, bool)):
        return None, "no single match"
    command = (f"{step.command} {uid_param}={finish_check.quote_step_value(rows[0])}"
               + (f" {rest}" if rest else ""))
    for cls in parameters_classes:
        bound = ParameterExtraction._extract_parameters_from_key_values(
            command, cls, step.command, code_written=True)
        if bound is None or finish_check.plain_value(getattr(bound, uid_param, None)) != \
                finish_check.plain_value(rows[0]):
            return None, "uid does not bind"
        if any(finish_check.plain_value(getattr(bound, key, None)) != value
               for key, value in step.arguments.items()):
            return None, "prose in value"
    return command, "only row" if len(output[uid_col]) == 1 else "exact match"


def parameter_classes(routing: Any, planning: Optional[Mapping[str, Any]] = None
                      ) -> Callable[[str], list]:
    """``command name -> its parameters classes`` (one per context that defines it, None
    for one with no parameters) in the routing definition *routing*.

    *planning* is the workflow's planning commands (qualified name -> ``PlanningCommandCommand``, as
    ``planning_commands.resolve_field_sources(folder).planning_commands``): routing does not hold
    them, and a step that names one is read against its ``Input`` model (its one parameters class).
    """
    from fastworkflow import ModuleType
    from fastworkflow.observation_offloading.finish_check import plain_command_name

    planning = planning or {}

    def classes(command: str) -> list:
        found = [routing.get_command_class(name, ModuleType.COMMAND_PARAMETERS_CLASS)
                 for name in routing.routing_definition_map
                 if name != "wildcard" and plain_command_name(name).lower() == command.lower()]
        found += [planning_command.input_model for name, planning_command in planning.items()
                  if plain_command_name(name).lower() == command.lower()]
        return found

    return classes


def plan_code_steps(plan_text: str, commands: Iterable[str], core_commands: Iterable[str],
                    parameters: Callable[[str], list]) -> list[tuple[Any, Optional[str], str]]:
    """``(step, command, reason)`` for each step of *plan_text*, as code reads it.

    *command* is what code runs (``code_step``); ``""`` for a step code passes over, since
    it names nothing to run (an answer step, a negated step); None for an open step. A
    core ``go_up`` / ``reset_context`` written alone runs. *parameters* gives a command's
    parameters classes (``parameter_classes``); a step is mechanical only under all of them.
    """
    from fastworkflow.observation_offloading import finish_check

    core = {finish_check.plain_command_name(c).lower() for c in core_commands if c}
    read = []
    for step in finish_check.parse_plan(plan_text, list(commands), list(core)):
        if step.kind == finish_check.KIND_SKIPPED and step.command.lower() in core:
            # A step about a core command: navigation, unless it is negated or an answer.
            nav = finish_check._classify(step.number, step.text, core, set())
            if nav.kind == finish_check.KIND_SKIPPED:
                read.append((step, "", "passed over"))
            elif nav.kind == finish_check.KIND_COMMAND and nav.command.lower() in _CODE_NAVIGATION:
                read.append((step, *_code_step(nav, None)))
            else:
                read.append((step, None, f"{nav.kind} navigation step"))
        elif step.kind == finish_check.KIND_SKIPPED:
            read.append((step, "", "passed over"))
        elif step.kind != finish_check.KIND_COMMAND:
            read.append((step, None, f"{step.kind} step"))
        else:
            results = [_code_step(step, cls) for cls in parameters(step.command)] or [
                (None, "unknown command")]
            texts = {text for text, _ in results}
            read.append((step, texts.pop(), results[0][1]) if len(texts) == 1
                        else (step, None, "differs by context"))
    return read


def step_paging(step: Any, parameters: Callable[[str], list]) -> tuple[list[str], bool]:
    """How code pages command *step* (a ``finish_check`` plan step) when it runs it:
    ``(paging, every_page)``.

    *paging* is the paging parameters (``facts.PAGING_PARAMETERS``) of the step's command
    (*parameters*, as for ``plan_code_steps``), or [] when it has none or the step writes
    one (then the step reads the page it names). With *paging*, code reads page 1;
    *every_page* says the step's note (from the first "(" or ";" after its command) asks
    for the full list (``FULL_LIST_PHRASES``), so code goes on to each later page the
    output names, up to ``CODE_MAX_PAGES`` pages (fix-ajir.36). Otherwise page 1 is all
    the step reads.
    ``CodeSteps`` and the planning command preview (``code_step_pages``) both read paging here.
    """
    from fastworkflow.observation_offloading.facts import PAGING_PARAMETERS
    from fastworkflow.observation_offloading.finish_check import step_chain

    fields = {name for cls in parameters(step.command) if cls is not None
              for name in cls.model_fields}
    paging = [name for name in PAGING_PARAMETERS if name in fields]
    if not paging or any(name in step.arguments for name in paging):
        return [], False
    text = step.text.replace("`", " ").replace("*", " ")
    rest = text[text.find(step.command) + len(step.command):]
    note = rest[len(step_chain(rest)):]
    return paging, any(phrase in note.lower() for phrase in FULL_LIST_PHRASES)


def code_step_pages(plan_text: str, commands: Iterable[str], core_commands: Iterable[str],
                    parameters: Callable[[str], list]) -> tuple[list[int], list[int]]:
    """``(every_page, page_one)``: the numbers of the steps of *plan_text* that run a paged
    command without writing a paging value, split by ``step_paging``: those whose note
    asks for every page (code reads up to ``CODE_MAX_PAGES`` pages) and those that read
    page 1 only. A step counts whether ``plan_code_steps`` reads it as run by code or as open
    (code may still run it: a pick, or a value that is a placeholder in a preview); one
    code passes over (an answer or negated step) does not."""
    every_page: list[int] = []
    page_one: list[int] = []
    for step, command, _ in plan_code_steps(plan_text, commands, core_commands, parameters):
        if command == "" or not step.command:
            continue
        paging, every = step_paging(step, parameters)
        if paging:
            (every_page if every else page_one).append(step.number)
    return every_page, page_one


#: The most pages code reads of one full-list step, page 1 included (fix-ajir.36). Past
#: it code goes on with the pages read: the facts give the total and how many were read.
CODE_MAX_PAGES = 20


def _next_page(observation: Any, paging: Iterable[str], start: int) -> Optional[tuple[str, int]]:
    """``(name, value)``: the later page an observation names for the page at *start*: the
    largest value of one of the command's paging parameters it writes (``Next page:
    start=60``) that is above *start*, or None (the last page, or no paging named)."""
    names = "|".join(re.escape(name) for name in paging)
    if not names:
        return None
    found = [(int(value), name) for name, value in re.findall(
        rf"(?<![A-Za-z0-9_])({names})\s*=\s*(\d+)", str(observation or ""))
        if int(value) > start]
    if not found:
        return None
    value, name = max(found)
    return name, value


def _more_pages(observation: Any, paging: Iterable[str]) -> bool:
    """Whether a page-1 observation names a later page: one of the command's paging
    parameters with a positive value (``start=20``), as a paged command tells its reader."""
    names = "|".join(re.escape(name) for name in paging)
    return bool(names) and re.search(rf"(?<![A-Za-z0-9_])(?:{names})\s*=\s*0*[1-9]",
                                     str(observation or "")) is not None


class CodeSteps:
    """The leading steps of one phase run that code runs, before (or instead of) the agent.

    ``fastWorkflowReAct._run_loop`` (``code=``) asks ``next()`` for each ``(thought, command)``
    and runs it through the agent's own command tool as a normal trajectory step, then
    reports it with ``ran(step_index, observation)``. Code passes over answer and negated
    steps, and stops at the first open step (``plan_code_steps``; an open step that opens
    the one matching row of the step code ran before it, ``pick_step``, runs), after a step that
    failed. A step on a paged command whose note asks for the full list
    (``FULL_LIST_PHRASES``) runs again for each later page its output names, up to
    ``CODE_MAX_PAGES`` pages, each a step of its own (fix-ajir.36; code then goes on with
    the pages read). A step whose command is not available where the workflow is goes
    there first when ``context_navigation.code_path`` finds a no-value path; each move is
    a step of its own. Code never fails the phase: it hands over to the agent.

    ``stop`` says why code stopped (``done`` when it ran every step; ``open step N``,
    ``step N failed``, ``no path for step N``; ``stopped`` when
    the executor's ``on_step_complete`` hook ended the run) and ``count``
    how many steps it ran, moves included. *own_steps* steps of the plan are the phase's
    own; planner-written steps after them are never run by code: one that runs anything
    (not an answer or negated step) is an open step, so the agent runs it. *answers*: the run's
    answer is shown, so a phase run wholly by code still writes it: the agent then takes a
    step from code's last one, with ``done_inputs`` (set by the caller) over its inputs.
    """

    def __init__(self, plan_text: str, *, workflow: Callable[[], Any],
                 runs: Callable[[int], list], commands: Iterable[str],
                 core_commands: Iterable[str], own_steps: int, answers: bool,
                 call_runs: Optional[Callable[[], list]] = None,
                 planning: Optional[Mapping[str, Any]] = None):
        import fastworkflow

        self.workflow, self.runs, self.answers = workflow, runs, answers
        self.call_runs = call_runs
        routing = fastworkflow.RoutingRegistry.get_definition(workflow().folderpath)
        # A step naming a planning command is a command step too: code runs it through the
        # agent's command tool, with the planning command's Input as its parameters (3.5).
        self._parameters = parameter_classes(routing, planning)
        read = plan_code_steps(plan_text, commands, core_commands, self._parameters)
        self.items = read[:own_steps]
        # The first planner-written step after the phase's own that runs anything.
        self._after = next((step for step, command, _ in read[own_steps:] if command != ""),
                           None)
        self.count = 0
        self.stop: Optional[str] = None
        #: Inputs the agent's loop gets when code ran every step (``finished``).
        self.done_inputs: dict[str, Any] = {}
        self._position = 0
        self._moves: list[str] = []
        self._reached: set[int] = set()
        self._current: Optional[tuple[Any, list, bool]] = None
        # The last step code ran (step, step index, observation), for a pick (pick_step).
        self._lookup: Optional[tuple[Any, int, Any]] = None
        self._step_info: dict[int, tuple[int, bool]] = {}
        self._picks: dict[int, str] = {}
        # A full-list step's paging (fix-ajir.36): its command, the page read last, the
        # pages read, and the next page to read (name, value) or None.
        self._paged: Optional[tuple[str, int, int]] = None
        self._next: Optional[tuple[str, int]] = None

    @property
    def runs_any(self) -> bool:
        """Whether code runs the plan's first step that runs anything."""
        return next((bool(command) for _, command, _ in self.items if command != ""), False)

    @property
    def finished(self) -> bool:
        """Whether code ran every step: no agent loop is needed."""
        return self.stop == "done"

    def next(self) -> Optional[tuple[str, str]]:
        """The next ``(thought, command)`` code runs, or None: code is done or stops."""
        if self.stop is not None:
            return None
        if self._next is not None:
            step, paging, _ = self._current
            name, value = self._next
            self._next = None
            command, _, pages = self._paged
            self._paged = (command, value, pages + 1)
            self._current = (step, paging, False)
            return (f"Step {step.number} of the plan, run by code (its page at "
                    f"{name}={value}).", f"{command} {name}={value}")
        if self._moves:
            step = self.items[self._position][0]
            self._current = (step, [], True)
            return (f"Run by code to reach the context of step {step.number}.",
                    self._moves.pop(0))
        while self._position < len(self.items) and self.items[self._position][1] == "":
            self._position += 1
        if self._position == len(self.items):
            self.stop = f"open step {self._after.number}" if self._after else "done"
            return None
        step, command, _ = self.items[self._position]
        if command is None:
            command = self._pick(step)
            if command is None:
                self.stop = f"open step {step.number}"
                return None
            self.items[self._position] = (step, command, "pick")
        if step.number not in self._reached:
            self._reached.add(step.number)
            self._moves = self._path_to(command) or []
            if self._moves:
                return self.next()
        self._position += 1
        paging = self._paging(step, command)
        self._current = (step, paging, False)
        self._paged = (command, 0, 1) if paging else None
        return (f"Step {step.number} of the plan, run by code"
                + (" (its first page)" if paging else "")
                + self._picks.get(step.number, "") + ".", command)

    def ran(self, step_index: int, observation: Any) -> None:
        """How the command ``next()`` gave went: a failure, or a page-1 run with more
        pages, stops code."""
        step, paging, move = self._current
        self.count += 1
        if not self._succeeded(step_index, observation):
            refused = "is not available in the current context" in str(observation or "")
            self.stop = (f"no path for step {step.number}" if refused and not move
                         else f"step {step.number} failed")
        elif paging and self._paged is not None:
            # Code reads the page the observation names next, up to CODE_MAX_PAGES.
            _, start, pages = self._paged
            later = _next_page(observation, paging, start)
            if later is not None and pages < CODE_MAX_PAGES:
                self._next = later
        if not move:
            from fastworkflow.observation_offloading.facts import PAGING_PARAMETERS

            more = not paging and _more_pages(observation, PAGING_PARAMETERS)
            self._step_info[step_index] = (step.number, more)
        self._lookup = None if move else (step, step_index, observation)

    def _call_runs(self) -> list[dict[str, Any]]:
        """The call's own runs so far, in run order, for ``pick_listing``."""
        try:
            raw = (list(self.call_runs()) if self.call_runs is not None
                   else [r for idx in sorted(self._step_info) for r in self.runs(idx)])
        except Exception:  # noqa: BLE001 - runs that cannot be read give no row to pick
            return []
        enriched: list[dict[str, Any]] = []
        for run in raw:
            entry = dict(run)
            idx = entry.get("step_index")
            if idx in self._step_info:
                num, more = self._step_info[idx]
                entry["plan_step_number"] = num
                if more:
                    entry["more_pages"] = True
            enriched.append(entry)
        return enriched

    def _pick(self, step: Any) -> Optional[str]:
        """The command for open *step* when it opens one row of a listing this call read
        (``pick_listing``, ``pick_step``), or None. Never after a page that shows more pages."""
        classes = self._parameters(step.command)
        found = pick_listing(step, self._call_runs(), classes)
        if found is None:
            return None
        chosen, output = found
        command, reason = pick_step(step, chosen, output, classes)
        if command is not None:
            num = chosen.get("plan_step_number")
            where = f" of step {num}" if num is not None else ""
            self._picks[step.number] = (f" (the only exact match{where})"
                                        if reason == "exact match"
                                        else f" (the only row{where})")
        return command

    def _succeeded(self, step_index: int, observation: Any) -> bool:
        """No failure marker and every command run of the step succeeded; the reply for
        entering the context the workflow is already in ran nothing and counts as done."""
        from fastworkflow.context_navigation import ALREADY_IN_CONTEXT
        from fastworkflow.observation_offloading.finish_check import run_failed

        text = str(observation or "")
        if run_failed(text):
            return False
        try:
            runs = self.runs(step_index)
        except Exception:  # noqa: BLE001 - runs that cannot be read are not known to succeed
            return False
        return all(run.get("success") for run in runs) if runs else ALREADY_IN_CONTEXT in text

    def _path_to(self, command: str) -> Optional[list[str]]:
        """The no-value moves to a context where *command* runs; [] when it runs here."""
        from fastworkflow.command_executor import AlreadyInContextError, CommandNotFoundError
        from fastworkflow.context_navigation import code_path
        from fastworkflow.workflow_agent import _explicit_agent_command

        workflow = self.workflow()
        try:
            _explicit_agent_command(command, workflow)
        except AlreadyInContextError:
            return []
        except CommandNotFoundError:
            return code_path(workflow, command.split()[0])
        except Exception:  # noqa: BLE001 - the step runs, and the tool says what is wrong
            return []
        return []

    def _paging(self, step: Any, command: str) -> list[str]:
        """The paging parameters (``facts.PAGING_PARAMETERS``) of the step's command when
        the step writes none of them and its note asks for the full list
        (``FULL_LIST_PHRASES``): code reads page 1 and each later page the output names,
        up to ``CODE_MAX_PAGES`` (``step_paging``)."""
        paging, every_page = step_paging(step, self._parameters)
        return paging if every_page else []


_ANSWER_SIGNATURE = None


def _answer_signature():
    global _ANSWER_SIGNATURE
    if _ANSWER_SIGNATURE is None:
        import dspy

        class WriteAnswerFromFacts(dspy.Signature):
            """Write the answer the input asks for, from the facts it gives and nothing else.
            No command or tool can be run. Do not invent, recompute or reword values: copy
            every name, id and number exactly as given. When a fact the answer needs is not
            given, say that it is not known."""

            user_query: str = dspy.InputField(desc="the request, the job's framing and the facts")
            final_answer: str = dspy.OutputField(
                desc="the answer, from the facts only, naming every item of every list given")

        _ANSWER_SIGNATURE = WriteAnswerFromFacts
    return _ANSWER_SIGNATURE


#: A DSPy field marker (``[[ ## completed ## ]]``) a reply may carry into its text.
_FIELD_MARKER_RE = re.compile(r"\[\[\s*##\s*[A-Za-z0-9_ ]*?\s*##\s*\]\]")
#: A Markdown code fence line (```` ``` ```` or ```` ```text ````).
_FENCE_RE = re.compile(r"^\s*(```|~~~)[A-Za-z0-9_+-]*\s*$", re.MULTILINE)
#: An answer with fewer word characters than this (markers and fences left out) is empty.
MIN_ANSWER_CHARS = 4


#: A line that is only a DSPy field name left from a marker (``completed``, ``## completed``).
_BARE_MARKER_RE = re.compile(r"^[\[\]#\s:]*(?:completed|final_answer|reasoning)[\[\]#\s:]*$",
                             re.IGNORECASE | re.MULTILINE)
#: A short line that echoes a prompt template's placeholder (``Input was: {final_answer}``).
_PLACEHOLDER_LINE_RE = re.compile(r"^[^\n{}]{0,40}\{[a-z][a-z0-9_]*\}[ \t.]*$", re.MULTILINE)


def clean_answer(text: Any) -> str:
    """*text* without what is never part of an answer, and trimmed.

    Removed: DSPy field markers (``[[ ## completed ## ]]``) and a line that is only the
    name a marker left (``completed``); a short line that echoes a template placeholder
    (``Input was: {final_answer}``); and the phase framing a reply copied
    (``This is part 6 of 6 (answer).``, ``ANSWER_ONLY_RULE`` and the other framing
    sentences of ``phase_query``)."""
    cleaned = _FIELD_MARKER_RE.sub("", str(text or ""))
    cleaned = _BARE_MARKER_RE.sub("", cleaned)
    cleaned = _PLACEHOLDER_LINE_RE.sub("", cleaned)
    cleaned = _PART_RE.sub("", cleaned)
    for sentence in _FRAMING_SENTENCES:
        cleaned = cleaned.replace(sentence, "")
    cleaned = _ANSWER_REQUEST_RE.sub("", cleaned)
    lines = [line.rstrip() for line in cleaned.strip().splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def is_empty_answer(text: Any) -> bool:
    """Whether *text* says nothing: empty, only a code fence or markers, or fewer than
    ``MIN_ANSWER_CHARS`` letters and digits."""
    visible = _FENCE_RE.sub("", clean_answer(text))
    return len(re.findall(r"[A-Za-z0-9]", visible)) < MIN_ANSWER_CHARS


_STEP_NUMBER_RE = re.compile(r"^\s*\d+[.)]\s*")
_SAY_RE = re.compile(r"^say(?:\s+that)?\s+", re.IGNORECASE)

#: The words a guard step starts with (``answer_step_kind``): with a negation word
#: (``finish_check._NEGATION_RE``) and no facts, it only says what not to do
#: (``apply no change``) and adds nothing to the answer.
ANSWER_STEP_WORDS = {
    "guard": ("apply", "do not", "don't", "never", "make no",
              "present them as options only", "present it as options only"),
}


def _starts_with(text: str, words: Sequence[str]) -> bool:
    return any(re.match(re.escape(word) + r"\b", text, re.IGNORECASE) for word in words)


def answer_step_kind(step: str) -> str:
    """How code reads one answer-only step (its number dropped):

    - ``"verbatim"``: it starts with ``say that`` (``say that b1 has loans l1, l2``): code
      writes it as it stands. This is the only form code copies;
    - ``"guard"``: no facts, only what not to do: it starts with a guard word
      (``ANSWER_STEP_WORDS["guard"]``) AND has a negation word (``finish_check._NEGATION_RE``:
      ``apply no change``): it adds nothing to the answer;
    - ``"judgement"``: anything else (``say which is riskier: A 3, B 5``,
      ``apply a 10% discount``): the model must write it.
    """
    from fastworkflow.observation_offloading.finish_check import _NEGATION_RE

    step = _STEP_NUMBER_RE.sub("", str(step or "")).strip()
    if re.match(r"say\s+that\s", step, re.IGNORECASE):
        return "verbatim"
    head, sep, body = step.partition(": ")
    if (not (sep and body.strip()) and _starts_with(step, ANSWER_STEP_WORDS["guard"])
            and _NEGATION_RE.search(step)):
        return "guard"
    return "judgement"


def _answer_line(step: str) -> str:
    """One step written as an answer line: its leading ``say`` / ``say that`` dropped, its
    first letter in capitals, ending with a full stop (``say that P`` is ``P.``)."""
    line = _SAY_RE.sub("", step).strip()
    line = line[:1].upper() + line[1:]
    return line if line.endswith((".", "!", "?")) else line + "."


def answer_from_steps(plan_text: str) -> str:
    """An answer written in code from an answer-only phase's steps.

    The steps carry the facts as finished text (``say that b1 has loans l1, l2``). Each
    step but a guard (``answer_step_kind``) becomes one line (``_answer_line``), its number
    and its leading ``say`` / ``say that`` dropped, in step order, with no numbering.
    """
    lines = [_STEP_NUMBER_RE.sub("", line).strip() for line in str(plan_text or "").splitlines()]
    return "\n".join(_answer_line(line) for line in lines
                     if line and answer_step_kind(line) != "guard")


#: The fewest items of a comma run that ``bulleted_answer`` writes as sub-bullets.
MIN_BULLETED_ITEMS = 4
#: An item with a second value after it in brackets (``Emma [b2]``, ``Emma (b2)``).
_TAGGED_ITEM_RE = re.compile(r"(.+) (?:\[([^\[\]]+)\]|\(([^()]+)\))")


def _output_texts(value: Any, texts: set[str]) -> set[str]:
    """Every text value in *value* (a model, mapping, list or text), stripped, added to *texts*."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, Mapping):
        value = list(value.values())
    if isinstance(value, (list, tuple)):
        for item in value:
            _output_texts(item, texts)
    elif isinstance(value, str) and value.strip():
        texts.add(value.strip())
    return texts


def _is_output_value(piece: str, texts: set[str]) -> bool:
    """Whether *piece* is exactly one output text, or one followed by another in brackets."""
    tagged = _TAGGED_ITEM_RE.fullmatch(piece)
    return piece in texts or bool(
        tagged and tagged[1] in texts and (tagged[2] or tagged[3]) in texts)


def _first_value(candidates: Iterable[str], texts: set[str]) -> Optional[str]:
    """The first of *candidates* that is an output value (``_is_output_value``), if any."""
    return next((text for text in candidates if _is_output_value(text, texts)), None)


def _list_runs(line: str, texts: set[str]) -> tuple[list[tuple[str, list[str]]], str]:
    """*line*'s comma runs of ``MIN_BULLETED_ITEMS`` output values or more
    (``_is_output_value``), as (text before the run, items), and the text after the last
    run. The first item may follow ``(``, ``": "`` or a comma (the shortest value there);
    the last may come before ``)``, ``.``, ``;`` or a comma (the longest)."""
    pieces = line.split(", ")
    for i, piece in enumerate(pieces):
        first = _first_value([piece[m.end():] for m in re.finditer(r"\(|: ", piece)][::-1]
                             + [piece], texts)
        if first is None:
            continue
        n = i + 1                       # the first piece after the whole values
        while n < len(pieces) and _is_output_value(pieces[n], texts):
            n += 1
        items, after = [first, *pieces[i + 1:n]], (pieces[n] if n < len(pieces) else "")
        last = _first_value([after[:m.start()] for m in re.finditer(r"[).;]", after)][::-1], texts)
        items += [last] if last else []
        if len(items) >= MIN_BULLETED_ITEMS:
            head = ", ".join(pieces[:i] + [piece[:len(piece) - len(first)]])
            runs, tail = _list_runs(line[len(head) + len(", ".join(items)):], texts)
            return [(head, items), *runs], tail
    return [], line


def bulleted_answer(answer: str, outputs: Mapping[str, Any]) -> str:
    """A code-written answer (``answer_from_steps``) in Markdown, for a person to read.

    Each line becomes a bullet when there are several. Inside a line, a comma run of
    ``MIN_BULLETED_ITEMS`` items or more becomes sub-bullets, in order, only when EVERY
    item is exactly a text value of *outputs* (or one with another in brackets after it,
    ``Emma [b2]``); the text before the run stays on the line, ending in ``:``, and text
    after it goes on below. Only the brackets, colon, commas or full stop around a run are
    dropped; any other line is left as it is. The checks read the answer before this.
    """
    texts = _output_texts(outputs, set())
    lines = [line for line in answer.splitlines() if line.strip()]
    pad, bullet = ("  ", "- ") if len(lines) > 1 else ("", "")
    out: list[str] = []
    for line in lines:
        runs, tail = _list_runs(line, texts)
        if not runs or not runs[0][0].strip(" ,;"):    # no run, or no text before the first
            out.append(bullet + line)
            continue
        for n, (head, items) in enumerate(runs):
            head = head.lstrip(" ),.;").rstrip(" (:,;")
            if n:
                out.append("")
            if head:
                out.append((bullet if n == 0 else pad) + head + ":")
            out.extend(f"{pad}- {item}" for item in items)
        tail = tail.lstrip(" ),.;")
        if re.search(r"\w", tail):
            out += ["", pad + tail]
    return "\n".join(out)


def answer_by_code(plan_text: str) -> bool:
    """Whether code writes the answer of an answer-only phase with these steps, with no
    model call: every step is verbatim or a guard (``answer_step_kind``) and at
    least one is verbatim."""
    kinds = [answer_step_kind(line) for line in str(plan_text or "").splitlines() if line.strip()]
    return "verbatim" in kinds and "judgement" not in kinds


#: A step that tells the answer to name or list what it gives says one of these words.
_NAMING_RE = re.compile(r"\b(name|names|naming|list|lists|listing|every|each)\b", re.IGNORECASE)


def _naming_steps(plan_text: str) -> list[str]:
    """The steps (lines) of *plan_text* that tell the answer to name or list something."""
    return [line for line in str(plan_text or "").splitlines() if _NAMING_RE.search(line)]


def _names_list_in_full(value: Sequence[str], plan_text: str) -> bool:
    """Whether ONE step of *plan_text* gives every item of the list *value* (as whole words)
    and tells the answer to name or list it: a naming word (``_NAMING_RE``) outside the
    items themselves, so an item's own text ("Accounts listed twice") never makes the
    step a naming step."""
    for line in _naming_steps(plan_text):
        if not all(names_value(line, item) for item in value):
            continue
        rest = line
        for item in sorted(value, key=len, reverse=True):
            rest = rest.replace(item, " ")
        if _NAMING_RE.search(rest):
            return True
    return False


def _list_rows(model: Any, plan_text: str) -> list[tuple[str, ...]]:
    """The rows of *model*'s list fields of text that *plan_text* gives in full.

    A list field counts when it has two items or more and every item is whole words of
    ONE step that tells the answer to name or list (``name``, ``list``, ``every``,
    ``each``; ``_naming_steps``): the planning command passed the whole list to be named. A one-row
    list is a match or a scope the answer names in its own words; a list given only as
    background (in a step that names nothing) is not checked. List fields of one length are read
    as columns of one table (uids and their labels): row *i* holds item *i* of each.
    """
    if isinstance(model, BaseModel):
        fields = {name: getattr(model, name) for name in type(model).model_fields}
    elif isinstance(model, Mapping):
        fields = dict(model)
    else:
        return []
    columns: dict[int, list[list[str]]] = {}
    for value in fields.values():
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            continue
        if not all(isinstance(item, str) and item.strip() for item in value):
            continue
        if _names_list_in_full(value, plan_text):
            columns.setdefault(len(value), []).append(list(value))
    rows: list[tuple[str, ...]] = []
    for length, fields in columns.items():
        rows.extend(tuple(field[i] for field in fields) for i in range(length))
    return rows


def _row_text(row: Sequence[str], plan_text: str) -> str:
    """One row as the steps write it (the shortest stretch of the steps holding all its
    values, when that is close to their length), else its values joined."""
    if len(row) == 1:
        return row[0]
    plain = _folded(plan_text)
    spans = []
    for value in row:
        wanted = _folded(value)
        spans.append([m.span() for m in re.finditer(
            r"(?<![A-Za-z0-9_])" + re.escape(wanted) + r"(?![A-Za-z0-9_])", plain)])
    best = None
    for start, end in spans[0]:
        lo, hi = start, end
        for other in spans[1:]:
            near = [s for s in other if abs(s[0] - start) <= 200]
            if not near:
                lo = None
                break
            s = min(near, key=lambda s: abs(s[0] - start))
            lo, hi = min(lo, s[0]), max(hi, s[1])
        if lo is not None and (best is None or hi - lo < best[1] - best[0]):
            best = (lo, hi)
    total = sum(len(v) for v in row)
    if best is not None and best[1] - best[0] <= total + 6 * len(row):
        text = " ".join(str(plan_text).split())
        if len(text) == len(plain):
            lo, hi = best
            # Close a bracket the stretch opened: "Emma (b2" -> "Emma (b2)".
            for opener, closer in ("()", "[]"):
                while (text[lo:hi].count(opener) > text[lo:hi].count(closer)
                       and hi < len(text) and text[hi] == closer):
                    hi += 1
            return text[lo:hi].strip(" ,;")
    return " ".join(row)


def _folded(value: Any) -> str:
    """Text compared case-folded, whitespace collapsed, same length as its collapsed form."""
    return " ".join(str(value or "").split()).casefold()


def missing_list_items(answer: str, plan_text: str, outputs: Mapping[str, Any]) -> list[str]:
    """The rows of the lists the steps give in full that *answer* does not name, as text.

    *outputs* are the earlier phases' outputs (``bound_arguments``' source: a model, or a
    fan-out phase's list of models). A row is missing when one of its values is not whole
    words of the answer. Rows are written as the steps write them (``_row_text``), in the
    order the outputs give them, each once.
    """
    out: list[str] = []
    for value in outputs.values():
        for model in (value if isinstance(value, list) else [value]):
            for row in _list_rows(model, plan_text):
                if all(names_value(answer, item) for item in row):
                    continue
                text = _row_text(row, plan_text)
                if text not in out:
                    out.append(text)
    return out


#: The line before the rows an answer-only phase's answer left out, appended in code.
MISSING_ITEMS_LEAD = "Also in the facts, not named above:"


def write_answer(query: str) -> str:
    """An answer-only phase's answer: one model call on *query* (``phase_query`` with
    ``answer_only``), with no tools. Uses the LM of the caller's ``dspy.context``. A reply
    that cannot be parsed, or that is empty (``is_empty_answer``: nothing, only a code
    fence or markers), is retried once. Field markers are stripped (``clean_answer``)."""
    import dspy
    from dspy.utils.exceptions import AdapterParseError

    predictor = dspy.Predict(_answer_signature())
    answer = ""
    for attempt in range(2):
        try:
            answer = clean_answer(getattr(predictor(user_query=query), "final_answer", ""))
        except AdapterParseError:
            if attempt:
                raise
            continue
        if not is_empty_answer(answer):
            return answer
    return answer


def answer_only_answer(query: str, plan_text: str, outputs: Mapping[str, Any]) -> str:
    """The answer of an answer-only phase, checked in code.

    When every step is verbatim or a guard (``answer_by_code``), code writes it from the
    steps (``answer_from_steps``), with no model call. Otherwise ``write_answer`` writes it
    (retrying an empty or unparsable reply once); still empty or unparsable, code writes it
    from the steps. Then every
    row of the lists the steps give in full that the answer does not name
    (``missing_list_items``, against *outputs*, the earlier phases' outputs) is appended
    after ``MISSING_ITEMS_LEAD``, so a list is never dropped, cut short or garbled out of
    the answer. Last, an answer code wrote is laid out for reading (``bulleted_answer``).
    """
    from dspy.utils.exceptions import AdapterParseError

    by_code = answer_by_code(plan_text)
    try:
        answer = answer_from_steps(plan_text) if by_code else write_answer(query)
    except AdapterParseError as error:
        from fastworkflow.utils.logging import logger

        logger.warning(f"answer-only reply could not be parsed twice; written from the steps: {error}")
        answer = ""
    if is_empty_answer(answer):
        answer, by_code = answer_from_steps(plan_text), True
    missing = missing_list_items(answer, plan_text, outputs)
    if by_code:
        answer = bulleted_answer(answer, outputs)
    if missing:
        answer = f"{answer}\n\n{MISSING_ITEMS_LEAD} {', '.join(missing)}."
    return answer


# ----------------------------------------------------------------------
# Long lists: capped where the answer is not told to name them
# ----------------------------------------------------------------------

#: The most items of one list an answer-only phase's prompt shows when its steps do not
#: tell the answer to name that list (``capped_steps``). High enough that an ordinary listing
#: (a few dozen rights or members) is given whole; it stops a flood (a record's 857 findings).
FACTS_MAX_LIST_ITEMS = 100
#: The most items of one list a stopped job's answer shows (``found_facts``): short lists
#: for a person to read; the count says the rest.
STOPPED_FACTS_MAX_ITEMS = 10
#: The separators a step's steps_fn joins a list's items with.
_LIST_SEPARATORS = (", ", "; ", " | ", ",", ";")


def facts_list_cap() -> int:
    """The most items of one list an answer-only prompt shows when the answer is not told to
    name it: the module constant ``FACTS_MAX_LIST_ITEMS`` (100)."""
    return FACTS_MAX_LIST_ITEMS


def capped_steps(plan_text: str, outputs: Mapping[str, Any],
                 max_items: Optional[int] = None) -> str:
    """*plan_text* (an answer-only phase's steps) with each long background list cut short.

    A list is a list field of text of an earlier phase's output (*outputs*, as for
    ``missing_list_items``) with more than *max_items* items (default ``facts_list_cap``).
    Where a step gives its items in order, joined by one separator (``", "``, ``"; "``,
    ...), that run becomes the first *max_items* items and ``"(and K more; M in all)"``.
    A list the steps tell the answer to name (the lists ``missing_list_items`` checks:
    every item whole words of one naming step) is never cut, and neither is a list the
    steps write in another form. The count and the first items stay, so the answer can
    still say how many there are.
    """
    from fastworkflow.observation_offloading.facts import capped_items

    cap = max_items or facts_list_cap()
    text = str(plan_text or "")
    for value in outputs.values():
        for model in (value if isinstance(value, list) else [value]):
            if isinstance(model, BaseModel):
                fields = [getattr(model, name) for name in type(model).model_fields]
            elif isinstance(model, Mapping):
                fields = list(model.values())
            else:
                continue
            for items in fields:
                if (not isinstance(items, (list, tuple)) or len(items) <= cap
                        or not all(isinstance(item, str) and item.strip() for item in items)
                        or _names_list_in_full(items, text)):
                    continue
                for sep in _LIST_SEPARATORS:
                    joined = sep.join(items)
                    if joined in text:
                        text = text.replace(joined, capped_items(items, cap, sep))
                        break
    return text


# ----------------------------------------------------------------------
# Fan-out: one item's record binding
# ----------------------------------------------------------------------


def names_value(clause: Any, value: Any) -> bool:
    """Whether the ``"<Context> <label>"`` *clause* names *value* (as whole words)."""
    from fastworkflow.observation_offloading import finish_check

    wanted = finish_check.plain_value(value)
    return bool(wanted) and re.search(
        r"(?<![A-Za-z0-9_])" + re.escape(wanted) + r"(?![A-Za-z0-9_])",
        finish_check.plain_value(clause)) is not None


def _run_names_item(run: Mapping[str, Any], item: Any) -> bool:
    """Whether *run*'s archived input or output names *item* (spacing and case ignored)."""
    wanted = finish_check.plain_value(item)
    if not wanted:
        return False
    for blob in (run.get("input"), run.get("output")):
        if not isinstance(blob, Mapping):
            continue
        for value in blob.values():
            if finish_check.plain_value(value) == wanted:
                return True
    return False


def _record_from_opening_run(runs: Sequence[Mapping[str, Any]], item: Any, ended_on: Optional[str],
                             planned: Optional[Iterable[str]]) -> Optional[str]:
    """Record clause when a planned step opened *item* but the context has no instance label."""
    wanted_plain = {_plain(c) for c in planned} if planned is not None else None
    for index, run in enumerate(runs):
        if not run.get("success"):
            continue
        if wanted_plain is not None and _plain(run.get("command")) not in wanted_plain:
            continue
        if not _run_names_item(run, item):
            continue
        for clause in (run.get("context_clause"), _entered(runs, index, ended_on), ended_on):
            if is_record_clause(clause) and names_value(clause, item):
                return str(clause)
        name = str(run.get("context_name") or "").strip()
        if not name:
            for clause in (ended_on, _entered(runs, index, ended_on), run.get("context_clause")):
                text = str(clause or "").strip()
                if text and " " not in text:
                    name = text.split()[0]
                    break
        if name:
            return context_clause(name, finish_check.plain_value(item))
    return None


def item_record(runs: Sequence[Mapping[str, Any]], item: Any,
                ended_on: Optional[str] = None,
                planned: Optional[Iterable[str]] = None) -> Optional[str]:
    """The record an item of a fan-out phase is bound to, or None when it was never opened.

    It is a record clause that names *item* (``names_value``) among the clauses the
    item's *runs* (in run order) ran on or moved the workflow to, and *ended_on*, the
    clause the item's run ended on: *ended_on* when it is one of them, else the last.
    When the workflow declares no instance label on the descended context (``open_book``
    with ``book_uid=<item>``), a clause is synthesized from the context name and *item*.
    """
    candidates: list[str] = []
    clauses = []
    for index, run in enumerate(runs):
        clauses += [run.get("context_clause"), _entered(runs, index, ended_on)]
    clauses.append(ended_on)
    for clause in clauses:
        if is_record_clause(clause) and names_value(clause, item) and clause not in candidates:
            candidates.append(clause)
    if ended_on in candidates:
        return ended_on
    if candidates:
        return candidates[-1]
    return _record_from_opening_run(runs, item, ended_on, planned)


def item_record_issues(runs: Sequence[Mapping[str, Any]], item: Any, others: Iterable[Any],
                       ended_on: Optional[str] = None,
                       commands: Optional[Iterable[str]] = None) -> list[str]:
    """What the finish check flags for one item of a fan-out phase; ``[]`` on a correct run.

    - The item was never opened: no run of it ran on, or moved the workflow to, a record
      that names it (``item_record``).
    - A step ran on ANOTHER item's record: a run whose clause names one of *others* (the
      phase's other items) and not *item*. With *commands* (the plain names of the
      phase's command steps, ``planned_steps``) only runs of those count, so leaving
      the previous item's record (``go_up``) is not a step on it. A run that moved the
      workflow onto the item's own record (it opened it from there) is not flagged.
    """
    issues: list[str] = []
    if item_record(runs, item, ended_on, planned=commands) is None:
        issues.append(f"item {item} was never opened: no command ran on a record that names it")
    others = [other for other in others if other != item]
    wanted = None if commands is None else {_plain(c) for c in commands}
    for index, run in enumerate(runs):
        clause = run.get("context_clause")
        if not is_record_clause(clause) or names_value(clause, item):
            continue
        if wanted is not None and _plain(run.get("command")) not in wanted:
            continue
        if names_value(_entered(runs, index, ended_on), item):
            continue
        other = next((o for o in others if names_value(clause, o)), None)
        if other is not None:
            step = run.get("step_index")
            where = f"step {step}: " if step is not None else ""
            issues.append(f"{where}{_plain(run.get('command')) or 'a command'} ran on {clause}, "
                          f"the record of item {other}, not of item {item}")
    return issues


def item_record_note(item: Any, issues: Sequence[str], max_bytes: int = 1024) -> Optional[str]:
    """The finish-check note for an item's *issues*, or None when there are none."""
    if not issues:
        return None
    lines = [f"Record check: this part is about item {item} only."]
    lines += [f"- {issue}." for issue in issues]
    lines.append(f"Open the record of item {item} and do the steps there, then finish.")
    note = "\n".join(lines)
    return note.encode("utf-8")[:max_bytes].decode("utf-8", "ignore")


def cap_note(name: str, kept: int, total: int, noun: Optional[str] = None) -> str:
    """The line the answer gets when fan-out phase *name* ran only *kept* of *total* items.

    The line never names the phase (an internal name). It counts in *noun* (the phase's
    ``noun``, a plural like ``"people"``), else in "items": "Note: only the first 3 of 5
    people were covered."
    """
    return f"Note: only the first {kept} of {total} {(noun or 'items').strip()} were covered."


def fan_out_note(name: str, ran: int, total: int, stopped: bool,
                 noun: Optional[str] = None) -> Optional[str]:
    """The line the answer gets at the end of fan-out phase *name*, or None.

    *ran* items of *total* ran. *stopped*: its ``until`` condition was met, so the rest
    did not run (a note only when some were left). Otherwise a fan-out that ran fewer
    than *total* was capped (``cap_note``). The line counts in *noun*, else in "items",
    and never names the phase.
    """
    if stopped:
        if ran >= total:
            return None
        return (f"Note: the work stopped after {ran} of {total} {(noun or 'items').strip()} "
                "because its stop condition was met.")
    return cap_note(name, ran, total, noun) if total > ran else None


def _is_uid_field(name: str) -> bool:
    return bool(re.search(r"(?:^|_)(?:uid|uids|id|ids)$", name))


def _field_label(name: str) -> str:
    """A facts field's name as words: ``permission_labels`` -> ``permissions``,
    ``labels`` -> ``names``, ``account_total`` -> ``account total``."""
    words = name.split("_")
    if words[-1] in ("labels", "label") and len(words) > 1:
        head = " ".join(words[:-1])
        if words[-1] == "label" or head.endswith("s"):
            return head
        return head[:-1] + "ies" if re.search(r"[^aeiou]y$", head) else head + "s"
    if words == ["labels"]:
        return "names"
    return " ".join(words)


def _readable_value(value: Any, max_items: int) -> str:
    from fastworkflow.observation_offloading.facts import capped_items

    if value is None:
        return "none"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, (BaseModel, Mapping)):
        inner = readable_facts(value, max_items)
        return f"({'; '.join(line[2:] for line in inner)})" if inner else "none"
    if isinstance(value, (list, tuple)):
        return capped_items([_readable_value(v, max_items) for v in value], max_items) or "none"
    return " ".join(str(value).split())


def readable_facts(facts: Any, max_items: Optional[int] = None) -> list[str]:
    """One filled output as ``- <label>: <value>`` lines, for a person to read.

    Fields in the model's order, each named in words (``_field_label``); a list shows its
    first *max_items* items (default ``STOPPED_FACTS_MAX_ITEMS``) and
    ``"(and K more; M in all)"``;
    a bool is yes/no. A list of uids (a field named ``uids``/``*_uids``) is left out when a
    list of text of the same length stands beside it (its labels: the rows are named by
    them); a single uid, which identifies a record, is kept (with ``label``, on one line:
    ``- Dune (uid b1)``). Only the values given: no text is added that the facts do not hold.
    """
    cap = max_items or STOPPED_FACTS_MAX_ITEMS
    if isinstance(facts, BaseModel):
        values = {name: getattr(facts, name) for name in type(facts).model_fields}
    elif isinstance(facts, Mapping):
        values = dict(facts)
    else:
        return [f"- {_readable_value(facts, cap)}"]
    text_lengths = {len(v) for name, v in values.items()
                    if isinstance(v, (list, tuple)) and v and not _is_uid_field(name)
                    and all(isinstance(item, str) for item in v)}
    out = []
    # The record a model is about, by name and uid on one line.
    named = (isinstance(values.get("uid"), str) and isinstance(values.get("label"), str)
             and values["label"].strip())
    for name, value in values.items():
        if (_is_uid_field(name) and isinstance(value, (list, tuple)) and value
                and len(value) in text_lengths):
            continue  # its rows are named by the labels beside it
        if named and name in ("uid", "label"):
            if name == "label":
                out.append(f"- {' '.join(value.split())} (uid {values['uid']})")
            continue
        out.append(f"- {_field_label(name)}: {_readable_value(value, cap)}")
    return out


def found_facts(outputs: Mapping[str, Any], max_items: Optional[int] = None) -> list[str]:
    """What the phases that ran found, for a failed call's answer: one block per phase with
    an output, ``"<phase>:"`` then its facts as ``- <label>: <value>`` lines
    (``readable_facts``; a fan-out phase: one block per item filled, ``"<phase>, item k:"``).

    *outputs* maps each phase name, in run order, to its filled output (a model, a dumped
    model, or a fan-out phase's list of them). These are facts filled in code, never a
    model's text. Long lists are capped at *max_items* (default ``STOPPED_FACTS_MAX_ITEMS``).
    """
    out = []
    for name, value in outputs.items():
        label = name.replace("_", " ")
        if isinstance(value, list):
            for k, item in enumerate(value):
                out.append("\n".join([f"{label}, item {k + 1}:",
                                      *readable_facts(item, max_items)]))
        elif value is not None:
            out.append("\n".join([f"{label}:", *readable_facts(value, max_items)]))
    return out


def failure_text(answers: Sequence[str], phase: str, error: str, not_run: Sequence[str],
                 notes: Sequence[str] = ()) -> str:
    """The sub-task's answer when phase *phase* failed: what was found, then what failed.

    *answers* are the blocks of what was found (``found_facts``: the facts the phases
    that ran filled), never text the failed phase's run wrote."""
    found = "\n\n".join(answer for answer in answers if answer)
    rest = f" Not run: {', '.join(not_run)}." if not_run else ""
    note = f"This job stopped at its part {phase!r}, which failed: {error}.{rest}"
    note = " ".join([note, *notes])
    return f"{FOUND_FACTS_LEAD}\n\n{found}\n\n{note}" if found else note


#: The line before the facts a stopped job's answer shows (``failure_text``).
FOUND_FACTS_LEAD = "Found before the job stopped (facts read from the commands that ran):"


def bound_result_arguments(result_fn: Callable, input: Any, outputs: Mapping[str, Any], *,
                           phase_names: frozenset[str]) -> dict[str, Any]:
    """Keyword arguments for ``__result__``: ``input`` and each phase name of the call."""
    kwargs: dict[str, Any] = {}
    parameters = inspect.signature(result_fn).parameters
    if "input" in parameters:
        kwargs["input"] = input
    for parameter in parameters.values():
        if parameter.name in ("self", "input"):
            continue
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if parameter.name in outputs:
            kwargs[parameter.name] = outputs[parameter.name]
        elif parameter.name in phase_names:
            kwargs[parameter.name] = None
        elif parameter.default is not parameter.empty:
            continue
        else:
            raise PhaseError(
                f"__result__ needs {parameter.name!r}, which is not a phase of this call")
    return kwargs


def typed_phase_outputs(raw: Mapping[str, Any], phases: Sequence[Phase]) -> dict[str, Any]:
    """Phase outputs as pydantic models (or lists of them for a fan-out), for ``__result__``."""
    from fastworkflow.typed_phase_facts import validate_phase_output_value

    by_name = {phase.name: phase for phase in phases}
    typed: dict[str, Any] = {}
    for name, value in raw.items():
        phase = by_name.get(name)
        if phase is None or phase.output is None:
            typed[name] = value
            continue
        if phase.fans_out:
            typed[name] = [validate_phase_output_value(phase.output, item) for item in value]
        else:
            typed[name] = validate_phase_output_value(phase.output, value)
    return typed


def command_response_from_call_result(value: Any, output_model: type[BaseModel] | None):
    """Build a ``CommandResponse`` from what ``__result__`` or ``__call__`` returned (3.14)."""
    import fastworkflow

    if isinstance(value, fastworkflow.CommandResponse):
        return value
    if isinstance(value, str):
        return fastworkflow.CommandResponse(response=value, output=None)
    if output_model is not None and isinstance(value, output_model):
        return fastworkflow.CommandResponse(response=value.model_dump_json(), output=value)
    raise TypeError(type(value).__name__)


def _result_fn_on_class(plan_cls: type) -> Callable | None:
    """The ``__result__`` function as defined on *plan_cls* or a base (not inherited descriptor)."""
    for klass in plan_cls.__mro__:
        if "__result__" in klass.__dict__:
            return klass.__dict__["__result__"]
    return None


def run_call_result(plan_cls: type, input: Any, outputs: Mapping[str, Any], *,
                    phase_names: frozenset[str], output_model: type[BaseModel] | None,
                    command_name: str, request: str) -> tuple[Any, bool]:
    """Run ``__result__`` once; on failure or an unexpected return, one model call writes the answer.

    Returns ``(command_response, fallback_used)``.
    """
    import fastworkflow
    from fastworkflow.utils.logging import logger

    result_fn = _result_fn_on_class(plan_cls)
    if result_fn is None:
        raise PhaseError(f"{plan_cls.__qualname__} defines no __result__")
    instance = plan_cls()
    kwargs = bound_result_arguments(result_fn, input, outputs, phase_names=phase_names)
    try:
        raw = result_fn(instance, **kwargs)
        if isinstance(raw, str) and not raw.strip():
            raise TypeError("empty str")
        try:
            return command_response_from_call_result(raw, output_model), False
        except TypeError as error:
            raise TypeError(str(error)) from None
    except Exception as error:  # noqa: BLE001 - runtime rule 3.16
        logger.warning("planning command %r: __result__ failed (%s: %s); writing the answer "
                       "with one model call", command_name, type(error).__name__, error)
        return fastworkflow.CommandResponse(response=write_answer(request), output=None), True


#: The author-facing API: what a planning command file writes (``Phase``, ``concatenated_steps``,
#: ``fact_from``, ``facts_text``) and what a workflow's tests read to check its planning commands
#: without a model. The other names serve the framework (runner, loader, preview).
__all__ = [
    "Phase",
    "Placeholder",
    "building_template",
    "is_building_template",
    "tests_noted",
    "is_placeholder",
    "PhaseError",
    "active_phases",
    "answer_by_code",
    "answer_from_steps",
    "answer_step_kind",
    "cap_note",
    "check_bindings",
    "code_step_pages",
    "concatenated_steps",
    "fact_from",
    "facts_text",
    "fan_out_items",
    "fan_out_note",
    "fill_output",
    "from_",
    "is_answer_only",
    "is_from_source",
    "parameter_classes",
    "phase_steps",
    "plan_code_steps",
    "runs_now",
    "step_paging",
    "stops_after",
]
