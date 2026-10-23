"""What a skill file writes: ``class PlanGenerator``, ``@phase``, ``pick`` and three filters.

A skill file (``_skills/<name>.py``) reads like a command file. ``class Signature`` declares
the skill exactly as a command's does (docstring, ``Input``, ``plain_utterances``,
``validate_extracted_parameters``, ``fastworkflow.Refused``). Where a command has a
``ResponseGenerator`` because it generates a response, a skill has a ``PlanGenerator``
because it generates a plan::

    from fastworkflow.skill import fact_from, pick


    class PlanGenerator:
        def find(self, input) -> Match:                  # a phase; it fills a Match
            return ["open_catalog",
                    f"find_item query={input.name}"]

        class Item(BaseModel):                           # facts: like a command's Output
            uid: str = fact_from("open_item_by_uid.item_uid")
            parts: list[str] = fact_from("list_parts.labels")

        def item(self, input, find) -> Item:             # reads the find phase's facts
            if not find.uids:
                return []                                # no steps: skipped when reached
            return [pick("open_item_by_uid", item_uid=find.only_uid(input.name),
                         named=input.name),
                    "list_parts (full list)"]

        def answer(self, input, item=None):              # None: the item phase was skipped
            if item is None:
                return [f"say that no item named {input.name} was found", "apply no change"]
            return [f"say that {item.uid} has parts {self._parts(item)}", "apply no change"]

        def _parts(self, item):                          # a helper, not a phase
            return ", ".join(item.parts) or "none"

The rules (the one-page guide: ``docs/skill_file_layout.md``)
-------------------------------------------------------------

1. Each public method of ``PlanGenerator`` is a phase (one executor run), in the order
   written; ``_name`` methods are helpers.
2. A phase's parameters name what it reads: ``input`` and earlier phases by name (None, its
   default, when that phase was skipped).
3. Its return annotation is the facts model it fills (``fact_from("command.field")`` per
   field); none: it passes nothing on.
4. It returns its steps as text, exactly the lines the agent reads. ``return []`` skips it:
   always decided when the phase is reached, on the real facts (the method is called once).

Optional: ``PlanGenerator.__call__(self, input)`` returns the phases a call runs, in order
(default: every one); a call of one phase that reads and fills no facts runs flat, as one
plan. ``@phase(over=..., max_items=..., until=..., noun=..., optional=...)`` makes a phase a
fan-out (once per item) or optional.

The template (what the planner sees) calls each method once per input with every fact a
``skill_phases.Placeholder``: it reads ``<find.total>``, ``<find.uids[0]>``,
``<find.labels[*]>``; a helper method called with one reads as its name in words
(``<their reach>``); every test on one reads as true and is noted, so the phase's lines end
``(only when find.uids)``. A method that fails there, or writes ``None`` into a step, is a
load error.
"""
from __future__ import annotations

import contextlib
import contextvars
import functools
import inspect
import re
from typing import Any, Callable, Optional

from pydantic import BaseModel

from fastworkflow.skill_phases import (Phase, PhaseError, Placeholder, _template_kwargs,
                                       concatenated_steps, fact_from, is_building_template,
                                       is_placeholder, tests_noted)

__all__ = ["count", "fact_from", "listed", "num", "phase", "pick"]


class StepFormError(ValueError):
    """A step written in a form the runtime would misread."""


class TemplateError(PhaseError):
    """A phase method that fails, or writes None into a step, while the template is built:
    a load error (``skill_commands.placeholder_template``)."""


# --------------------------------------------------------------------------- pick

#: ``(command, uid parameter)`` of each ``pick`` while ``picks_noted`` is open.
_PICKS: contextvars.ContextVar[Optional[list]] = contextvars.ContextVar("fw_skill_picks",
                                                                        default=None)


@contextlib.contextmanager
def picks_noted():
    """A list that gets ``(command, uid parameter)`` of each ``pick`` made inside the block
    (the load-time step check reads the parameter the text no longer shows)."""
    picks: list = []
    token = _PICKS.set(picks)
    try:
        yield picks
    finally:
        _PICKS.reset(token)


def pick(command: str, *, named: str, **uid: Any) -> str:
    """Open one row the previous step listed: ``command <uid_param>=<uid>`` when the facts
    give the uid; else (None at run time, unknown in the template) ``command with the uid of
    the NAMED match``, which code runs on the one row named exactly."""
    if len(uid) != 1 or not next(iter(uid)).endswith("_uid"):
        raise StepFormError(f"pick({command!r}): give exactly one uid parameter "
                            f"(..._uid=...), got {sorted(uid)}")
    (param, value), = uid.items()
    if not str(named).strip():
        raise StepFormError(f"pick({command!r}): named= is the name the row is picked by")
    picks = _PICKS.get()
    if picks is not None:
        picks.append((command, param))
    if value is None or is_placeholder(value):
        return f"{command} with the uid of the {named} match"
    return f"{command} {param}={value}"


# --------------------------------------------------------------------------- filters

def num(value: Any, none: str = "none recorded", grouped: bool = False) -> str:
    """A number exactly as read: 4.0 is "4", 2.5 stays "2.5"; with *grouped*, 5000000.0 is
    "5,000,000". None is *none*."""
    if is_placeholder(value):
        return str(value)
    if value is None:
        return none
    if float(value).is_integer():
        return f"{int(value):,}" if grouped else str(int(value))
    return f"{float(value):,}" if grouped else repr(float(value))


def count(number: Any, noun: str, plural: Optional[str] = None) -> str:
    """"1 login", "2 logins" ("2 people" with ``plural="people"``)."""
    if is_placeholder(number):
        return f"{number} {plural or noun + 's'}"
    return f"{number} " + (noun if number == 1 else plural or noun + "s")


def listed(names: Any, total: Any, noun: str, same: Optional[tuple] = None,
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


# --------------------------------------------------------------------------- @phase

_OPTIONS = ("over", "max_items", "until", "noun", "optional", "output", "bind")


def phase(**options: Any) -> Callable:
    """What a phase needs beyond the rules (``skill_phases.Phase``'s options):

    - ``over``: run the phase once per item: ``"find.uids"`` (an earlier phase's list
      field) or a function of ``input`` giving the items. The method's one parameter that
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
    """The phase methods of a ``PlanGenerator`` class, in definition order (``_name`` are
    helpers)."""
    return [(name, value) for name, value in vars(cls).items()
            if not name.startswith("_") and inspect.isfunction(value)]


def _parameters(fn: Callable) -> list[str]:
    return [p for p in inspect.signature(fn).parameters if p != "self"]


def _output(cls: type, fn: Callable, options: dict, input: Any) -> Optional[type[BaseModel]]:
    if "output" in options:
        return _decided(options["output"], input)
    annotation = fn.__annotations__.get("return")
    if isinstance(annotation, str):
        try:
            annotation = eval(annotation, fn.__globals__, dict(vars(cls)))  # noqa: S307
        except Exception as error:  # noqa: BLE001
            raise PhaseError(f"phase {fn.__name__!r}: cannot read its return annotation "
                             f"{annotation!r}: {error}") from None
    if annotation is None or annotation is inspect.Signature.empty:
        return None
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return annotation
    raise PhaseError(f"phase {fn.__name__!r}: its return annotation must be the facts model "
                     f"it fills (a pydantic model), got {annotation!r}; a model that depends "
                     "on the input goes in @phase(output=lambda input: ...)")


def _decided(value: Any, input: Any) -> Any:
    """An option given as a function of ``input`` only, decided now; other values as given."""
    if callable(value) and not inspect.isclass(value) and _parameters(value) == ["input"]:
        return value(input)
    return value


def _with_input(fn: Callable, input: Any) -> Callable:
    """*fn* with ``input`` given; its signature without it, so the runtime binds the rest
    by name (``skill_phases.bound_arguments``)."""
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


class _TemplateSelf:
    """``self`` while the template is built: a helper method (``self._name(...)``) called
    with an unknown fact is not run; it reads ``<name in words>``."""

    def __init__(self, plan: Any):
        self._plan = plan

    def __getattr__(self, name):
        value = getattr(self._plan, name)
        if not name.startswith("_") or name.startswith("__") or not callable(value):
            return value

        @functools.wraps(value)
        def helper(*args, **kwargs):
            if any(_unknown(a) for a in (*args, *kwargs.values())):
                return Placeholder(name.strip("_").replace("_", " "))
            return value(*args, **kwargs)

        return helper


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
                lines = call(_TemplateSelf(plan), facts)
            except Exception as error:  # noqa: BLE001 - reported as a load error
                raise TemplateError(f"phase {name!r} fails while its template is built (every "
                                    f"fact unknown): {type(error).__name__}: {error}") from None
        if not isinstance(lines, (list, tuple)):
            return lines   # the runtime's check reports it
        for line in lines:
            if _NONE.search(str(line)):
                raise TemplateError(f"phase {name!r} writes None into a step ({str(line)!r}): "
                                    "a value it formats is None for some input")
        when = f" (only when {' and '.join(tests)})" if tests else ""
        return [f"{line}{when}" for line in lines if str(line).strip()]

    def steps(**facts):
        if is_building_template():
            return template(facts)
        if kept and kept[0][0] == facts:
            result = kept.pop()[1]
            if isinstance(result, Exception):
                raise result
            return result
        return call(plan, facts)

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
                call(_TemplateSelf(plan), {p: Placeholder(p) for p in public.parameters})
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
    order (``__call__``'s, or every phase method)."""
    if not phase_methods(cls):
        raise PhaseError(f"{cls.__qualname__}: a PlanGenerator needs at least one phase method")
    methods = dict(phase_methods(cls))
    for name, fn in methods.items():
        if next(iter(inspect.signature(fn).parameters), None) != "self":
            raise PhaseError(f"PlanGenerator.{name}: a phase method takes self first, as a "
                             "ResponseGenerator's methods do")
    order = list(methods)
    custom = vars(cls).get("__call__")
    instance = cls()

    def chosen(input) -> list[str]:
        if custom is None:
            return order
        names = []
        for method in custom(instance, input) or ():
            name = getattr(method, "__name__", None)
            if getattr(method, "__self__", None) is not instance or name not in methods:
                raise PhaseError(f"{cls.__qualname__}.__call__ must return the plan's own "
                                 f"phase methods (self.find, ...), got {method!r}")
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
    fan-out or optional: the runtime then runs it as one plan named after the skill."""
    if len(phases) != 1:
        return False
    ph = phases[0]
    return (ph.output is None and not ph.fans_out and not ph.optional
            and not inspect.signature(ph.steps_fn).parameters)


def skill_parts(cls: type) -> tuple[Callable, Optional[Callable]]:
    """``(steps_fn, plan_fn)`` of a ``PlanGenerator`` class, as
    ``skill_commands.skill_from_parts`` takes them: ``plan_fn(input)`` is the phases a call
    runs, or None when it runs flat; ``steps_fn(input)`` is a flat call's steps, or the
    phases' template (``concatenated_steps``). ``plan_fn`` is None when every call runs flat
    (one phase method, reading only ``input``, no ``__call__``, no options, no facts)."""
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
    always_flat = (len(methods) == 1 and vars(cls).get("__call__") is None
                   and not getattr(methods[0][1], "__phase_options__", None)
                   and _parameters(methods[0][1]) in ([], ["input"])
                   and _output(cls, methods[0][1], {}, None) is None)
    return steps_fn, (None if always_flat else plan_fn)
