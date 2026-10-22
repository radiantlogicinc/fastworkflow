"""Skill previews: every skill of a workflow rendered to markdown, generated from its code.

``fastworkflow skill preview <workflow_folder> [--out <dir>] [--check]`` loads the
workflow's skills with the skill loader (``workflow_agent.load_skills`` /
``load_skill_commands``) and writes one ``<out>/<skill>.md`` per skill. The default
``<out>`` is ``docs/skill_previews`` next to the workflow folder
(``<workflow_folder>/../docs/skill_previews``, see ``default_out_dir``). With
``--check`` nothing is written: the command exits non-zero when a preview in ``<out>``
differs from a fresh render, is missing, or is an extra ``.md`` file no skill renders.

A preview, in the order of the skill-file layout (``docs/skill_file_layout.md``):

1. the skill's name, its description (the ``Signature`` docstring) and its kind (flat
   steps, or phases);
2. its inputs: name, type, default, the sample value the preview renders with, the
   ``verify`` and ``subject`` flags and the description, then the other options
   (``strip``, ``means``, ``in_request``);
3. its ``plain_utterances``;
4. for a phased skill (``plan()``): when ``plan()`` returns phases at all (when some inputs
   make the call run flat), then the phases, each with its output model, when it runs and
   how it fans out (``over``, ``item``, ``max_items``, ``until``, ``bind``, ``noun``), and the
   facts each output model holds with their ``"from"`` sources. A cell that differs by
   input (the output model, a callable ``when``, the fan-out) gives each value with the
   inputs that give it ("when kind = crate: ..."); an ``optional`` phase's "runs" cell ends
   "; optional (its failure does not stop the job)", with the inputs it is optional for
   when not all;
5. its steps for the sample inputs: for a phased skill phase by phase, for a flat skill
   the ``steps()`` list. Each different render the inputs give is shown once (see
   ``variants``).

Sample inputs come from the ``Input`` model (``sample_value``): a str takes its default,
else its first ``examples`` value, else a placeholder ``<name>``; a bool is false; an int
its default (else its first example, else 1), raised or lowered into its ``ge``/``le``
bounds; an enum its default, else its first value. The inputs that may change the plan are
the bools, the enums, the optional strs whose default is empty (given as the sample, or
left empty) and the ints (the sample, and the first of its examples, 0 and 3 that differs
and is in bounds). Every combination of them is rendered (past ``MAX_GRID``, the sample
inputs and each one or two changes of them, as for the skill template:
``skill_commands.input_grid``); an input whose values never change the rendered steps is
left at its sample value.
The combinations of the rest are grouped by the render they give, and each render is
titled by the fewest inputs that tell its combinations from the others
(``describe_inputs``: "when compare = true", "unless kind = kit"). The "runs" cell is
written the same way, so it names the input that decides a phase, not every combination.
Past ``MAX_VARIANTS`` renders, only those a pairwise cover of the inputs reaches, and one
of each other phase layout, are rendered; the phases table still reads them all. A
combination the skill's ``validate_extracted_parameters`` refuses or changes is listed,
not rendered (no skill call ever reaches ``steps()`` with it).

In a phased skill, a phase's ``steps_fn`` is called with placeholder outputs of the
earlier phases (``placeholder_output``): each fact reads as where it comes from, so a
step built from the first uid ``find`` found reads ``<find.item_uids[0]>``. A list fact
holds one placeholder item; an int fact is 1 and a float ``PLACEHOLDER_NUMBER`` (999999) in
arithmetic, and each prints as its placeholder (a number the code converts with ``int()``
or ``float()`` reads 999999); a bool is false. A str placeholder is never cut by
``split``/``partition``, so ``label.split("_")[0]`` keeps the whole placeholder. Facts
that are one value at run time get one placeholder (``_linked_values``): a uid fact
(source field ``*_uid``) is the first entry of an earlier list of those uids
(``*_uids``), so code that looks it up there finds it. A fan-out phase is rendered per
item: over a list of items, each item; over an earlier phase's list, its first and its
second entry (that list's model holds two placeholder items), and the item's output takes
the item as its uid and the list's parallel entries (its label). The phase's output
reaches later phases as a list of one placeholder per item rendered (``<items[0].uid>``).
Code that branches on facts is therefore rendered along one path, so a phase whose
``steps_fn`` reads earlier outputs is marked (``PLACEHOLDER_NOTE``). When ``steps_fn``
fails on the placeholders, the phase is rendered with no earlier outputs (as
``concatenated_steps`` does) and the preview says so. For inputs ``plan()`` returns no
phases for (``None`` or ``[]``), the flat ``steps()`` are rendered and the preview says the
call runs flat.

A callable ``when`` or ``until`` is shown by its source (``condition_text``): a lambda's
own text, read from the skill file with ``inspect``, on one line, with the closure
variables, ``input.<field>`` reads and parameter defaults it uses written as their values
when short and plain; else the function's name.

The output is byte-stable: no time, no absolute path, inputs and phases in declaration
order, combinations in a fixed order.
"""
from __future__ import annotations

import ast
import dataclasses
import inspect
import itertools
import json
import os
import re
import types
import typing
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union, get_args, get_origin

from pydantic import BaseModel

from fastworkflow.skill_commands import MAX_GRID, input_grid  # noqa: F401 - MAX_GRID bounds input_grid

#: The first line of every preview; ``write_previews`` only removes ``.md`` files that carry it.
MARKER = "<!-- Generated by `fastworkflow skill preview` from {source}. Do not edit: regenerate. -->"
#: Every different render of the plan-changing inputs is shown up to this many; past it,
#: those a pairwise cover of the inputs reaches and one of each other phase layout.
MAX_VARIANTS = 16


class PreviewError(ValueError):
    """A workflow whose skills cannot be previewed (no ``_skills`` folder, a skill not loaded)."""


# ---------------------------------------------------------------- placeholders

#: The value a float fact of an earlier phase has in arithmetic: obviously not a real
#: value, so a number the code converts (``int(score)``, ``float(level)``, which drop the
#: placeholder text) reads 999999 and is told apart from a real one (``PLACEHOLDER_NOTE``).
PLACEHOLDER_NUMBER = 999999.0


class _PlaceholderStr(str):
    """A str fact of an earlier phase (or a sample input): its placeholder text, split as one
    word. ``split``, ``rsplit``, ``partition`` and ``rpartition`` never cut it, so a skill
    that takes the part of a name before ``"_"`` keeps the whole placeholder."""

    def split(self, sep=None, maxsplit=-1):
        return [self]

    def rsplit(self, sep=None, maxsplit=-1):
        return [self]

    def partition(self, sep):
        return (self, "", "")

    def rpartition(self, sep):
        return ("", "", self)


class _PlaceholderInt(int):
    """An int fact of an earlier phase: 1 in arithmetic (a list fact holds one item, or the
    few a fan-out shows), its placeholder when printed."""

    def __new__(cls, text: str):
        obj = super().__new__(cls, 1)
        obj._text = text
        return obj

    def __str__(self) -> str:
        return self._text

    __repr__ = __str__

    def __format__(self, spec: str) -> str:
        return self._text


class _PlaceholderFloat(float):
    """A float fact of an earlier phase: ``PLACEHOLDER_NUMBER`` in arithmetic, its
    placeholder when printed."""

    def __new__(cls, text: str):
        obj = super().__new__(cls, PLACEHOLDER_NUMBER)
        obj._text = text
        return obj

    def __str__(self) -> str:
        return self._text

    __repr__ = __str__

    def __format__(self, spec: str) -> str:
        return self._text


def _is_list_type(annotation: Any) -> bool:
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        inner = [a for a in get_args(annotation) if a is not type(None)]
        return bool(inner) and _is_list_type(inner[0])
    if origin is typing.Annotated:
        return _is_list_type(get_args(annotation)[0])
    return origin in (list, tuple, set, frozenset) or (
        inspect.isclass(origin) and origin.__module__ == "collections.abc"
        and origin.__name__ in ("Sequence", "Iterable", "Collection"))


def _placeholder(annotation: Any, path: str, width: int = 1) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (Union, types.UnionType):
        inner = [a for a in args if a is not type(None)]
        return _placeholder(inner[0], path, width) if inner else None
    if origin is typing.Annotated:
        return _placeholder(args[0], path, width)
    if origin is Literal:
        return args[0] if args else None
    if _is_list_type(annotation):
        items = [_placeholder(args[0] if args else str, f"{path}[{i}]") for i in range(width)]
        return tuple(items) if origin is tuple else items
    if origin is dict:
        return {}
    if annotation is bool:
        return False
    if annotation is int:
        return _PlaceholderInt(f"<{path}>")
    if annotation is float:
        return _PlaceholderFloat(f"<{path}>")
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return placeholder_output(annotation, path)
    return _PlaceholderStr(f"<{path}>")


def placeholder_output(model: type[BaseModel], prefix: str, width: int = 1,
                       values: Optional[dict] = None) -> BaseModel:
    """*model* with every field a placeholder naming it: ``<prefix.field>``.

    A list field holds *width* placeholder items (``<prefix.field[0]>``, ...). *values*
    gives some fields a value instead (a fact that is the same as an earlier one, see
    ``_linked_values``). Built with ``model_construct`` (no validation), so the model's own
    methods work on it.
    """
    values = values or {}
    return model.model_construct(**{
        name: values[name] if name in values else _placeholder(info.annotation, f"{prefix}.{name}", width)
        for name, info in model.model_fields.items()})


def _sources(model: type[BaseModel]) -> dict[str, str]:
    """``{fact: source field}`` of *model*'s facts (``"from": "<command>.<field>"``)."""
    from fastworkflow.observation_offloading.facts import FactSourceError, fact_sources

    try:
        return {s.field: s.source_field for s in fact_sources(model)}
    except FactSourceError:
        return {}


def _is_uid(source_field: str) -> bool:
    return source_field == "uid" or source_field.endswith("_uid")


def _linked_values(model: type[BaseModel], earlier: list[tuple[str, Any]], phase=None,
                   item: Any = None, index: int = 0) -> dict:
    """The facts of *model* that are the same value as an earlier phase's, so placeholder
    facts stay consistent: a uid matched against an earlier list is found in it.

    - A fan-out item's output (*phase* fans out over ``<source>.<list>``, this is item
      *index*, *item*): a fact named like the item, or one whose source field is the
      singular of the iterated list's (``identity_uid`` for ``identity_uids``), is the item;
      any other str fact whose source field is the singular of a list of the same source
      model (``label`` for ``labels``) is that list's entry *index*.
    - Any other uid fact (source field ``uid`` or ``*_uid``) is the first entry of the
      latest earlier list fact whose source field is its plural (``group_uid`` and
      ``group_uids``).
    """
    sources = _sources(model)
    out: dict[str, Any] = {}
    iterated = None
    if phase is not None and getattr(phase, "over_reference", None) and isinstance(item, str):
        name, list_field = phase.over_reference
        source = dict(earlier).get(name)
        source = source[0] if isinstance(source, list) and source else source
        if isinstance(source, BaseModel):
            iterated = (source, list_field)
    for field_name, info in model.model_fields.items():
        if _is_list_type(info.annotation):
            continue
        source_field = sources.get(field_name, "")
        if phase is not None and isinstance(item, str) and field_name == phase.item:
            out[field_name] = item
            continue
        if iterated is not None:
            source, list_field = iterated
            listed = _sources(type(source))
            if source_field and listed.get(list_field) == source_field + "s":
                out[field_name] = item
                continue
            sibling = next((f for f, s in listed.items() if source_field and s == source_field + "s"
                            and isinstance(getattr(source, f, None), list)
                            and len(getattr(source, f)) > index), None)
            if sibling is not None:
                out[field_name] = getattr(source, sibling)[index]
                continue
        if not _is_uid(source_field):
            continue
        # the entry of this item for a fan-out's item, else the first
        at = index if phase is not None and phase.fans_out else 0
        for _, value in reversed(earlier):
            found = None
            for one in (value if isinstance(value, list) else [value]):
                if not isinstance(one, BaseModel):
                    continue
                hit = next((f for f, s in _sources(type(one)).items() if s == source_field + "s"
                            and isinstance(getattr(one, f, None), list)), None)
                if hit is not None:
                    listed = getattr(one, hit)
                    found = listed[at] if len(listed) > at else _MISSING
                    break
            if found is _MISSING:
                break
            if found is not None:
                out[field_name] = found
                break
    return out


# ---------------------------------------------------------------- markdown helpers

def _code(text: str) -> str:
    """*text* as one markdown code span."""
    text = " ".join(str(text).split())
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * (longest + 1)
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{fence}{pad}{text}{pad}{fence}"


def _text(text: Any, cell: bool = False) -> str:
    """*text* as markdown prose, on one line: ``<`` escaped, and ``|`` in a table cell."""
    out = " ".join(str(text).split()).replace("<", "\\<")
    return out.replace("|", "\\|") if cell else out


def _type_text(annotation: Any) -> str:
    if inspect.isclass(annotation) and not get_args(annotation):
        return annotation.__name__
    return re.sub(r"\btyping\.", "", repr(annotation))


def _describe(value: Any) -> str:
    if inspect.isclass(value):
        return value.__name__
    if callable(value):
        return getattr(value, "__qualname__", type(value).__name__)
    return repr(value)


def _lambda_node(fn: Any, hint: str = "") -> Optional[tuple[str, ast.Lambda]]:
    """``(file text, node)`` of lambda *fn* in its file, or None when it cannot be told apart.

    Lambdas on *fn*'s first line with *fn*'s parameter names are the candidates; with
    several, those given as the keyword *hint* (``when=``, ``until=``) are preferred.
    """
    code = getattr(fn, "__code__", None)
    if code is None:
        return None
    lines, _ = inspect.findsource(fn)
    text = "".join(lines)
    tree = ast.parse(text)
    params = list(code.co_varnames[:code.co_argcount + code.co_kwonlyargcount])
    keyed = {id(node.value) for node in ast.walk(tree)
             if isinstance(node, ast.keyword) and node.arg == hint}
    found = [node for node in ast.walk(tree)
             if isinstance(node, ast.Lambda) and node.lineno == code.co_firstlineno
             and [a.arg for a in node.args.args + node.args.kwonlyargs] == params]
    if len(found) > 1 and hint:
        found = [node for node in found if id(node) in keyed] or found
    texts = {ast.get_source_segment(text, node) for node in found}
    texts.discard(None)
    return (text, found[0]) if len(texts) == 1 else None


#: The longest value ``condition_text`` writes in place of a name.
_MAX_VALUE_TEXT = 60


def _simple(value: Any) -> bool:
    """A value short and plain enough to write into a condition's source."""
    if value is None or isinstance(value, (bool, int, float)):
        return True
    if isinstance(value, str):
        return len(value) <= _MAX_VALUE_TEXT
    if isinstance(value, (tuple, list, frozenset, set)):
        return len(value) <= 6 and all(_simple(v) and not isinstance(v, (tuple, list, set, frozenset))
                                       for v in value)
    return False


def _value_text(value: Any) -> str:
    if isinstance(value, str):
        return repr(str(value))
    if isinstance(value, (set, frozenset)):
        return repr(sorted(value, key=repr))
    if isinstance(value, tuple):
        return repr(tuple(_plain(v) for v in value))
    if isinstance(value, list):
        return repr([_plain(v) for v in value])
    return repr(value)


def _plain(value: Any) -> Any:
    return str(value) if isinstance(value, str) else value


_MISSING = object()


def _free_value(fn: Any, name: str) -> Any:
    """The value the free name *name* of *fn* has: a closure cell, else a global."""
    code = fn.__code__
    if name in code.co_freevars and fn.__closure__:
        try:
            return fn.__closure__[code.co_freevars.index(name)].cell_contents
        except ValueError:   # an empty cell
            return _MISSING
    return getattr(fn, "__globals__", {}).get(name, _MISSING)


def _byte_offsets(text: str) -> list[int]:
    """The UTF-8 byte offset of each line's start in *text* (ast columns are in bytes)."""
    out, at = [], 0
    for line in text.splitlines(keepends=True):
        out.append(at)
        at += len(line.encode("utf-8"))
    return out


def _bound_source(fn: Any, text: str, node: ast.Lambda) -> str:
    """The lambda's source with its free names (and the defaults of its parameters)
    written as the values they are bound to, where the value is short and plain: a
    closure ``name`` reads ``'Ann'``, ``input.kind`` reads ``'kit'``."""
    starts = _byte_offsets(text)
    data = text.encode("utf-8")

    def offset(line: int, col: int) -> int:
        return starts[line - 1] + col

    begin, end = offset(node.lineno, node.col_offset), offset(node.end_lineno, node.end_col_offset)
    params = {a.arg for a in node.args.args + node.args.kwonlyargs + node.args.posonlyargs}
    if node.args.vararg:
        params.add(node.args.vararg.arg)
    if node.args.kwarg:
        params.add(node.args.kwarg.arg)
    local = set(params)
    for inner in ast.walk(node.body):
        if isinstance(inner, ast.comprehension):
            local |= {n.id for n in ast.walk(inner.target) if isinstance(n, ast.Name)}
        elif isinstance(inner, ast.Lambda):
            local |= {a.arg for a in inner.args.args + inner.args.kwonlyargs}
        elif isinstance(inner, ast.NamedExpr):
            local.add(inner.target.id)
    edits: list[tuple[int, int, str]] = []

    def chain(expr: ast.AST) -> Optional[tuple[str, list[str]]]:
        attrs: list[str] = []
        while isinstance(expr, ast.Attribute):
            attrs.insert(0, expr.attr)
            expr = expr.value
        return (expr.id, attrs) if isinstance(expr, ast.Name) else None

    def visit(expr: ast.AST) -> None:
        if isinstance(expr, (ast.Attribute, ast.Name)) and isinstance(getattr(expr, "ctx", None), ast.Load):
            found = chain(expr)
            if found is not None and found[0] not in local:
                value = _free_value(fn, found[0])
                for attr in found[1]:
                    if value is _MISSING:
                        break
                    value = getattr(value, attr, _MISSING)
                if value is not _MISSING and not callable(value) and _simple(value):
                    edits.append((offset(expr.lineno, expr.col_offset),
                                  offset(expr.end_lineno, expr.end_col_offset), _value_text(value)))
                    return
        for child in ast.iter_child_nodes(expr):
            visit(child)

    visit(node.body)
    positional = node.args.posonlyargs + node.args.args
    defaults = list(zip(positional[len(positional) - len(node.args.defaults):], node.args.defaults,
                        fn.__defaults__ or ()))
    kw_values = fn.__kwdefaults__ or {}
    defaults += [(a, d, kw_values.get(a.arg)) for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults)
                 if d is not None]
    for _, expr, value in defaults:
        if not isinstance(expr, ast.Constant) and _simple(value):
            edits.append((offset(expr.lineno, expr.col_offset),
                          offset(expr.end_lineno, expr.end_col_offset), _value_text(value)))
    out = data[begin:end]
    for start, stop, new in sorted(edits, reverse=True):
        out = out[:start - begin] + new.encode("utf-8") + out[stop - begin:]
    return out.decode("utf-8")


def _one_line(source: str) -> str:
    """*source* on one line: each line's spaces collapsed, lines joined by one space, but
    none after an opening bracket or before a closing one."""
    out = ""
    for line in source.splitlines():
        part = " ".join(line.split())
        if not part:
            continue
        if out and not out.endswith(("(", "[", "{")) and not part.startswith((")", "]", "}")):
            out += " "
        out += part
    return out


def condition_text(fn: Any, hint: str = "") -> str:
    """A phase condition (a callable ``when`` or ``until``) as the preview shows it.

    A lambda: its source (``lambda find: bool(find.item_uids)``), on one line, read from
    the skill file, with each free name it reads (a closure variable, ``input.<field>``, a
    global) and each parameter default written as the value it is bound to when that value
    is short and plain (``find.pick('Ann')`` for ``find.pick(name)``); a function: its
    name. A lambda whose source cannot be read (or told apart from another on its line)
    falls back to its name, ``<lambda>``.
    """
    name = getattr(fn, "__name__", None) or type(fn).__name__
    if name == "<lambda>":
        try:
            found = _lambda_node(fn, hint)
            source = _bound_source(fn, *found) if found else None
        except (OSError, TypeError, SyntaxError, ValueError, IndexError):
            source = None
        if source:
            return _one_line(source)
    return name


# ---------------------------------------------------------------- inputs

@dataclass(frozen=True)
class _Axis:
    """One input that may change the plan, and the values it is rendered with (sample first)."""

    name: str
    values: tuple
    labels: tuple[str, ...]


def _field_info(command, name: str):
    model = getattr(command, "input_model", None)
    return model.model_fields.get(name) if model is not None else None


def _examples(command, name: str) -> list:
    info = _field_info(command, name)
    return list(getattr(info, "examples", None) or []) if info is not None else []


def _int_bounds(command, name: str) -> tuple[Optional[int], Optional[int]]:
    """The smallest and largest value the int field allows (``ge``/``gt``/``le``/``lt``)."""
    low = high = None
    info = _field_info(command, name)
    for item in getattr(info, "metadata", None) or []:
        for attr, shift, is_low in (("ge", 0, True), ("gt", 1, True), ("le", 0, False), ("lt", -1, False)):
            bound = getattr(item, attr, None)
            if isinstance(bound, (int, float)) and not isinstance(bound, bool):
                value = int(bound) + shift
                if is_low:
                    low = value if low is None else max(low, value)
                else:
                    high = value if high is None else min(high, value)
    return low, high


def _allowed(value: int, bounds: tuple[Optional[int], Optional[int]]) -> bool:
    low, high = bounds
    return (low is None or value >= low) and (high is None or value <= high)


def sample_value(command, p) -> Any:
    """The value parameter *p* of skill *command* is previewed with (see the module doc)."""
    examples = _examples(command, p.name)
    if p.type == "str":
        if not p.required and p.default:
            return p.default
        return str(examples[0]) if examples else _PlaceholderStr(f"<{p.name}>")
    if p.type == "bool":
        return False
    if p.type == "int":
        bounds = _int_bounds(command, p.name)
        if not p.required and p.default is not None:
            value = p.default
        else:
            value = int(examples[0]) if examples else 1
        if not _allowed(value, bounds):
            low, high = bounds
            value = low if low is not None and value < low else high
        return value
    if p.default is not None:
        return p.default
    return str(examples[0]) if examples and examples[0] in p.values else p.values[0]


#: An int input is tried with its sample and the first of these it allows that differs:
#: its examples, then 0, then 3 (a typical small number).
_INT_TRIES = (0, 3)


def _axes(command) -> list[_Axis]:
    out = []
    for p in command.params:
        sample = sample_value(command, p)
        if p.type == "bool":
            out.append(_Axis(p.name, (False, True), ("false", "true")))
        elif p.type == "enum":
            values = (sample,) + tuple(v for v in p.values if v != sample)
            out.append(_Axis(p.name, values, values))
        elif p.type == "str" and not p.required and p.default == "":
            given = sample or _PlaceholderStr(f"<{p.name}>")
            out.append(_Axis(p.name, (given, ""), ("given", "empty")))
        elif p.type == "int" and sample is not None:
            bounds = _int_bounds(command, p.name)
            tries = [int(e) for e in _examples(command, p.name)
                     if isinstance(e, (int, float, str)) and str(e).lstrip("-").isdigit()]
            other = next((v for v in [*tries, *_INT_TRIES] if v != sample and _allowed(v, bounds)), None)
            if other is not None:
                out.append(_Axis(p.name, (sample, other), (str(sample), str(other))))
    return out


def _sample_args(command) -> dict:
    return {p.name: sample_value(command, p) for p in command.params}


# ---------------------------------------------------------------- one render

@dataclass
class _PhaseRender:
    name: str
    reads: list[str]
    output: Optional[type]
    steps: list[str]
    settings: list[str] = field(default_factory=list)
    note: str = ""
    #: A callable ``when``, as ``condition_text`` shows it ("" for a bool ``when``).
    condition: str = ""
    #: How the phase fans out, for the phases table ("" when it does not).
    fan_out: str = ""
    #: Its failure does not stop the job (``Phase(optional=True)``).
    optional: bool = False
    #: Its steps were rendered with placeholder facts of earlier phases.
    placeholders: bool = False
    #: A fan-out phase: ``(heading, steps)`` per item rendered (None: not a fan-out).
    items: Optional[list[tuple[str, list[str]]]] = None

    def key(self) -> tuple:
        return (self.name, tuple(self.reads), self.output.__name__ if self.output else "",
                tuple(self.steps), tuple(self.settings), self.note, self.condition, self.fan_out,
                self.optional, self.placeholders,
                None if self.items is None else tuple((h, tuple(s)) for h, s in self.items))


@dataclass
class _Render:
    """The steps of one combination of inputs: flat steps, or phases (with those left out)."""

    steps: list[str] = field(default_factory=list)
    phases: list[_PhaseRender] = field(default_factory=list)
    left_out: list[str] = field(default_factory=list)
    error: str = ""
    #: ``plan()`` returned no phases for these inputs: the call runs flat, from ``steps()``.
    flat: bool = False

    def key(self) -> tuple:
        return (tuple(self.steps), self.error, tuple(self.left_out), self.flat,
                tuple(p.key() for p in self.phases))


_PHASE_FIELDS = ("name", "steps_fn", "output", "when", "optional")
#: Phase settings that hold a condition, shown by its source.
_CONDITION_FIELDS = ("until",)


def _setting_text(name: str, value: Any) -> str:
    if name in _CONDITION_FIELDS and callable(value):
        return condition_text(value, name)
    if isinstance(value, tuple):
        return repr([_plain(v) for v in value])
    return _describe(_plain(value))


def _settings(phase) -> list[str]:
    """The phase's other settings (fields a later ``Phase`` may add), when not their default.

    A callable ``when`` is listed too, by its source (``condition_text``).
    """
    out = []
    if not dataclasses.is_dataclass(phase):
        return out
    if callable(getattr(phase, "when", None)):
        out.append(f"when={condition_text(phase.when, 'when')}")
    for f in dataclasses.fields(phase):
        if f.name in _PHASE_FIELDS:
            continue
        value = getattr(phase, f.name)
        default = f.default if f.default is not dataclasses.MISSING else (
            f.default_factory() if f.default_factory is not dataclasses.MISSING else dataclasses.MISSING)
        if value != default:
            out.append(f"{f.name}={_setting_text(f.name, value)}")
    return out


def _fan_out_text(phase) -> str:
    """How *phase* fans out, for the phases table: "" when it does not."""
    over = getattr(phase, "over", None)
    if over is None:
        return ""
    source = over if isinstance(over, str) else repr([_plain(v) for v in over])
    parts = [f"per item of {_code(source)} as {_code(getattr(phase, 'item', 'item'))}"]
    if getattr(phase, "noun", None):
        parts[0] += f" ({phase.noun})"
    if getattr(phase, "max_items", None):
        parts.append(f"at most {phase.max_items}")
    if getattr(phase, "until", None) is not None:
        parts.append(f"until {_code(condition_text(phase.until, 'until'))}")
    if getattr(phase, "bind", True) is False:
        parts.append("not bound to a record")
    return "; ".join(parts)


#: How many items of a fan-out over an earlier phase's list the preview renders: the first,
#: and one later item (its steps may differ, e.g. by going back up first).
FAN_OUT_SHOWN = 2


def _fan_out_items(phase, outputs: dict) -> tuple[list[Any], list[str]]:
    """The items a fan-out phase is rendered for, and the heading of each."""
    from fastworkflow.skill_phases import fan_out_items

    items, total = fan_out_items(phase, outputs)
    shown = f"`{phase.item}` = "
    if phase.over_reference is None:
        return items, [f"For item {i} of {len(items)}, {shown}{_code(v)}:" for i, v in enumerate(items, 1)]
    items = items[:FAN_OUT_SHOWN]
    heads = [f"For the first item, {shown}{_code(items[0])}:"] if items else []
    heads += [f"For each later item (shown for the second), {shown}{_code(v)}:" for v in items[1:]]
    return items, heads


def _render_phases(command, args: dict) -> _Render:
    from fastworkflow import skill_commands
    from fastworkflow.skill_phases import Phase, PhaseError, active_phases, is_flat_plan, phase_steps

    try:
        raw = command.plan_fn(command.input_model.model_construct(**args))
        if is_flat_plan(raw):
            # No phases for these inputs: the call runs flat, from steps().
            try:
                return _Render(steps=skill_commands.expand(command, args), flat=True)
            except skill_commands.SkillCallError as error:
                return _Render(error=str(error), flat=True)
        phases = active_phases(raw, f"plan() of {command.name}")
    except Exception as error:  # noqa: BLE001 - workflow code; shown in the preview
        return _Render(error=f"plan() failed: {type(error).__name__}: {error}")
    render = _Render(left_out=[p.name for p in raw if isinstance(p, Phase) and not p.when])
    # A list a later phase fans out over holds FAN_OUT_SHOWN placeholder items (so does every
    # list of its model, which the item's steps may read by index).
    fanned = {p.over_reference[0] for p in phases if p.fans_out and p.over_reference}
    outputs: dict[str, Any] = {}
    earlier: list[tuple[str, Any]] = []
    for phase in phases:
        try:
            names = list(inspect.signature(phase.steps_fn).parameters)
        except (TypeError, ValueError):
            names = []
        note = ""
        items = heads = None
        if phase.fans_out:
            try:
                items, heads = _fan_out_items(phase, outputs)
            except PhaseError as error:
                items, heads, note = None, None, f"Items not rendered: {error}."
        try:
            if items is not None:
                per_item = [phase_steps(phase, outputs, item) for item in items]
                lines = [line for steps in per_item for line in steps]
            else:
                per_item, lines = None, phase_steps(phase, outputs)
        except PhaseError as error:
            items = heads = per_item = None
            note = f"Rendered without earlier facts: with placeholder facts, {error}."
            try:
                lines = phase_steps(phase, {})
            except PhaseError:
                kwargs = {n: None for n in names}
                try:
                    lines = [str(s) for s in phase.steps_fn(**kwargs)]
                except Exception as inner:  # noqa: BLE001
                    lines, note = [], f"steps_fn failed: {type(inner).__name__}: {inner}"
        reads = [n for n in names if n in outputs]
        render.phases.append(_PhaseRender(
            name=phase.name, reads=reads, output=phase.output,
            steps=lines, settings=_settings(phase), note=note,
            condition=condition_text(phase.when, "when") if callable(phase.when) else "",
            fan_out=_fan_out_text(phase), optional=bool(getattr(phase, "optional", False)),
            placeholders=bool(reads) and not note,
            items=None if per_item is None else list(zip(heads, per_item))))
        if phase.output is not None:
            width = FAN_OUT_SHOWN if phase.name in fanned else 1
            if phase.fans_out:
                # A fan-out phase passes on the list of its items' outputs, one per item rendered.
                shown = items if items is not None else [None]
                value = [placeholder_output(phase.output, f"{phase.name}[{i}]", width,
                                            _linked_values(phase.output, earlier, phase, item, i))
                         for i, item in enumerate(shown)]
            else:
                value = placeholder_output(phase.output, phase.name, width,
                                           _linked_values(phase.output, earlier))
            outputs[phase.name] = value
            earlier.append((phase.name, value))
    return render


def _render(command, args: dict) -> _Render:
    from fastworkflow import skill_commands

    if command.plan_fn is not None:
        return _render_phases(command, args)
    try:
        return _Render(steps=skill_commands.expand(command, args))
    except skill_commands.SkillCallError as error:
        return _Render(error=str(error))


def _validated(command, args: dict) -> tuple[bool, str]:
    """Whether a skill call can reach ``steps()`` with *args*, and why not."""
    from fastworkflow import skill_commands

    if command.validate is None:
        return True, ""
    try:
        if skill_commands._run_validate(command, args, "", None) != args:
            return False, "validate_extracted_parameters changes it"
    except skill_commands.SkillCallError as error:
        return False, str(error)
    return True, ""


# ---------------------------------------------------------------- which inputs decide

def _pairwise(sizes: list[int], first: tuple) -> list[tuple]:
    """Combinations (index tuples) covering every pair of values of every two inputs,
    *first* first: a greedy cover, deterministic."""
    n = len(sizes)
    if n < 2:
        rest = [(i,) for i in range(sizes[0])] if n else []
        return [first] + [c for c in rest if c != first]
    uncovered = {(i, a, j, b) for i in range(n) for j in range(i + 1, n)
                 for a in range(sizes[i]) for b in range(sizes[j])}

    def cover(combo):
        for i in range(n):
            for j in range(i + 1, n):
                uncovered.discard((i, combo[i], j, combo[j]))

    out = [first]
    cover(first)
    while uncovered:
        i, a, j, b = min(uncovered)
        combo = list(first)
        combo[i], combo[j] = a, b
        for k in range(n):
            if k in (i, j):
                continue

            def gain(v):
                return sum((min(k, m), (v if k < m else combo[m]), max(k, m),
                            (combo[m] if k < m else v)) in uncovered for m in range(n) if m != k)
            combo[k] = max(range(sizes[k]), key=lambda v: (gain(v), -v))
        out.append(tuple(combo))
        cover(out[-1])
    return out


def _deciding(members: list[tuple], universe: list[tuple]) -> tuple[int, ...]:
    """The fewest inputs (indexes, in order) whose values tell *members* from the rest of
    *universe*."""
    inside = set(members)
    width = len(universe[0]) if universe else 0
    for size in range(width + 1):
        for chosen in itertools.combinations(range(width), size):
            seen: dict[tuple, bool] = {}
            if all(seen.setdefault(tuple(c[i] for i in chosen), c in inside) == (c in inside)
                   for c in universe):
                return chosen
    return tuple(range(width))


def _terms(members: list[tuple], universe: list[tuple], axes: list[_Axis]) -> list[str]:
    """*members* as alternatives over the inputs that decide them (``_deciding``)."""
    chosen = _deciding(members, universe)
    if not chosen:
        return []
    tuples = list(dict.fromkeys(tuple(c[i] for i in chosen) for c in members))
    terms: list[tuple[tuple, list]] = []   # (the values of all but the last input, last values)
    for values in tuples:
        head, last = values[:-1], values[-1]
        if terms and terms[-1][0] == head:
            terms[-1][1].append(last)
        else:
            terms.append((head, [last]))
    out = []
    for head, last in terms:
        parts = [f"{axes[i].name} = {axes[i].labels[v]}" for i, v in zip(chosen, head)]
        parts.append(f"{axes[chosen[-1]].name} = "
                     + " or ".join(axes[chosen[-1]].labels[v] for v in last))
        out.append(", ".join(parts))
    return out


def _weight(terms: list[str]) -> int:
    """How much a description says: its input values, each ``=`` and each ``or``."""
    return sum(term.count(" = ") + term.count(" or ") for term in terms)


def describe_inputs(members: list[tuple], universe: list[tuple], axes: list[_Axis]) -> str:
    """When the combinations *members* (of *universe*) happen, by the inputs that decide it.

    ``"always"`` when they are all of *universe*; else ``"when <terms>"`` or, when that
    names fewer values (``_weight``), ``"unless <terms>"``; terms are joined by ``"; or "``.
    """
    rest = [c for c in universe if c not in set(members)]
    if not rest:
        return "always"
    yes, no = _terms(members, universe, axes), _terms(rest, universe, axes)
    if _weight(no) < _weight(yes):
        return "unless " + "; or ".join(no)
    return "when " + "; or ".join(yes)


#: The title of the combinations ``validate_extracted_parameters`` refuses, when they all
#: have one reason: every combination not rendered.
OTHER_COMBINATIONS = "every other combination"


@dataclass
class _Variant:
    #: When this render happens, by the inputs that decide it (``describe_inputs``).
    when: str
    render: Optional[_Render]
    refused: str = ""
    #: The combinations of the deciding inputs that give this render (index tuples).
    members: list = field(default_factory=list)
    axes: list = field(default_factory=list)
    #: Rendered in the plan section (past ``MAX_VARIANTS`` renders, not every one is).
    shown: bool = True

    def title(self) -> str:
        return "for every combination" if self.when == "always" else self.when


def _layout(render: _Render) -> tuple:
    """The shape of a render: flat, or its phases with their outputs and fan-outs."""
    return (render.flat, bool(render.error),
            tuple((p.name, p.output.__name__ if p.output else "", p.fan_out) for p in render.phases))


def variants(command) -> tuple[list[_Variant], list[str]]:
    """The renders of skill *command*: one per different render the inputs give.

    Returns ``(variants, changing)``: *changing* names the inputs whose value changes the
    render, in declaration order. The combinations of *changing* (the other inputs at their
    sample value) are each rendered, past ``MAX_GRID`` those ``input_grid`` tries, and
    grouped by render: each variant is one render, titled by the inputs that decide it
    (``describe_inputs``), the first the one of the sample inputs. Past ``MAX_VARIANTS``
    renders, only those a pairwise cover reaches and one of each other layout (``_layout``)
    are ``shown`` in the plan section; the phases table reads them all. A combination
    ``validate_extracted_parameters`` changes or refuses is a variant without a render,
    one per reason.
    """
    base = _sample_args(command)
    axes = _axes(command)
    cache: dict[tuple, tuple[Optional[_Render], str]] = {}

    def render_of(combo: tuple) -> tuple[Optional[_Render], str]:
        if combo not in cache:
            args = {**base, **{a.name: a.values[i] for a, i in zip(axes, combo)}}
            ok, why = _validated(command, args)
            cache[combo] = (_render(command, args), "") if ok else (None, why)
        return cache[combo]

    def key_of(combo: tuple):
        render = render_of(combo)[0]
        return render.key() if render is not None else None

    sizes = [len(a.values) for a in axes]
    sample = tuple(0 for _ in axes)
    grid = input_grid(sizes, sample)
    on_grid = set(grid)
    # the base of the combinations: the sample inputs, or the first valid combination
    start = next((c for c in grid if key_of(c) is not None), sample)
    changing = []
    for k in range(len(axes)):
        if any(key_of(c) is not None and other in on_grid and key_of(other) is not None
               and key_of(other) != key_of(c)
               for c in grid for i in range(sizes[k])
               for other in [c[:k] + (i,) + c[k + 1:]] if i != c[k]):
            changing.append(k)
    shown_axes = [axes[k] for k in changing]
    shown_sizes = [sizes[k] for k in changing]

    def full(sub: tuple) -> tuple:
        combo = list(start)
        for k, i in zip(changing, sub):
            combo[k] = i
        return tuple(combo)

    first_sub = tuple(start[k] for k in changing)
    subs = input_grid(shown_sizes, first_sub)
    groups: dict[Any, list[tuple]] = {}
    refused: dict[str, list[tuple]] = {}
    for sub in subs:
        render, why = render_of(full(sub))
        if render is None:
            refused.setdefault(why, []).append(sub)
        else:
            groups.setdefault(render.key(), []).append(sub)
    shown = list(groups)
    if len(groups) > MAX_VARIANTS:
        # the renders a pairwise cover reaches, and one of each other phase layout
        reached = set(_pairwise(shown_sizes, first_sub))
        shown = [k for k, members in groups.items() if reached & set(members)]
        layouts = {_layout(render_of(full(groups[k][0]))[0]) for k in shown}
        for k, members in groups.items():
            layout = _layout(render_of(full(members[0]))[0])
            if layout not in layouts:
                layouts.add(layout)
                shown.append(k)
        shown = [k for k in groups if k in set(shown)]
    universe = [s for members in groups.values() for s in members]
    out = [_Variant(describe_inputs(members, universe, shown_axes), render_of(full(members[0]))[0],
                    members=members, axes=shown_axes, shown=k in shown)
           for k, members in groups.items()]
    for why, members in refused.items():
        when = (OTHER_COMBINATIONS if len(refused) == 1
                else describe_inputs(members, subs, shown_axes))
        out.append(_Variant(when, None, why, members=members, axes=shown_axes))
    return out, [a.name for a in shown_axes]


# ---------------------------------------------------------------- the preview

def _inputs_section(command) -> list[str]:
    lines = ["## Inputs", ""]
    if command is None or not command.params:
        return lines + ["None: the steps are fixed.", ""]
    lines += ["| input | type | default | sample | verify | subject | description |",
              "|---|---|---|---|---|---|---|"]
    notes = []
    for p in command.params:
        type_text = f"one of {', '.join(p.values)}" if p.type == "enum" else p.type
        if p.required:
            default = "required"
        elif p.type == "bool":
            default = "false"
        else:
            default = _code(json.dumps(p.default) if p.type == "str" else p.default)
        sample = sample_value(command, p)
        sample_text = ("false" if sample is False else
                       _code(sample) if sample != "" else '`""`')
        lines.append(f"| `{p.name}` | {_text(type_text, True)} | {default} | {sample_text.replace('|', chr(92) + '|')} "
                     f"| {'yes' if p.verify else ''} | {'yes' if p.subject else ''} | {_text(p.desc, True)} |")
        if p.strip:
            notes.append(f"- `{p.name}` strips the words {', '.join(_code(w) for w in p.strip)} from its ends.")
        if p.means:
            notes.append(f"- `{p.name}` means: " + "; ".join(f"{_code(v)}: {_text(t)}" for v, t in p.means) + ".")
        if p.in_request:
            notes.append(f"- `{p.name}` must be a number the request writes (`in_request`).")
    lines.append("")
    if notes:
        lines += notes + [""]
    return lines


def _utterances_section(utterances) -> list[str]:
    lines = ["## Example requests", ""]
    if not utterances:
        return lines + ["None declared.", ""]
    return lines + [f"- {_text(u)}" for u in utterances] + [""]


def _numbered(steps: list[str]) -> list[str]:
    if not steps:
        return ["No steps."]
    return [f"{i}. {_code(s)}" for i, s in enumerate(steps, 1)]


def _phase_order(found: list[_Variant]) -> list[str]:
    order: list[str] = []
    for variant in found:
        if variant.render is None:
            continue
        at = 0
        for phase in variant.render.phases:
            if phase.name in order:
                at = order.index(phase.name) + 1
            else:
                order.insert(at, phase.name)
                at += 1
    return order


def _with_phase(name: str, found: list[_Variant]) -> list[tuple[_Variant, _PhaseRender]]:
    """``(variant, phase)`` for each render that has phase *name*, in render order."""
    return [(v, p) for v in found if v.render is not None for p in v.render.phases if p.name == name]


def _universe(found: list[_Variant]) -> list[tuple]:
    """The combinations whose render has phases (the phases table describes within them)."""
    return [m for v in found if v.render is not None and v.render.phases for m in v.members]


def _axes_of(found: list[_Variant]) -> list[_Axis]:
    return next((v.axes for v in found), [])


def _by_value(name: str, found: list[_Variant], value) -> list[tuple[str, str]]:
    """``(text, when)`` for each different ``value(phase)`` of phase *name* across the
    renders that have it, *when* telling (among those renders) which inputs give it."""
    pairs = _with_phase(name, found)
    within = [m for v, _ in pairs for m in v.members]
    texts: dict[str, list[tuple]] = {}
    for variant, phase in pairs:
        texts.setdefault(value(phase), []).extend(variant.members)
    if len(texts) == 1:
        return [(text, "always") for text in texts]
    return [(text, describe_inputs(members, within, _axes_of(found))) for text, members in texts.items()]


def _runs_when(name: str, found: list[_Variant]) -> str:
    """The phases table's "runs" cell: the inputs the phase runs for, and its callable ``when``."""
    present = [m for v, _ in _with_phase(name, found) for m in v.members]
    when = describe_inputs(present, _universe(found), _axes_of(found))
    text = ("always" if when == "always"
            else _text(f"only {when}" if when.startswith("when ") else when, True))
    optional = _optional_text(name, found)
    conditions = _by_value(name, found, lambda p: p.condition)
    shown = [(c, w) for c, w in conditions if c]
    if not shown:
        return text + optional
    if len(conditions) == 1:
        cond = _code(shown[0][0]).replace("|", "\\|")
    else:
        cond = " or ".join(_code(c).replace("|", "\\|") + f" ({_text(w, True)})" for c, w in shown)
        cond += "".join(f"; no condition {_text(w, True)}" for c, w in conditions if not c)
    return (f"only when {cond}" if text == "always"
            else f"{text}, and then only when {cond}") + optional


def _optional_text(name: str, found: list[_Variant]) -> str:
    """"; optional" (with the inputs it is optional for, when not all) for a phase whose
    failure does not stop the job; "" for a required phase."""
    pairs = _by_value(name, found, lambda p: p.optional)
    optional = [w for value, w in pairs if value]
    if not optional:
        return ""
    if len(pairs) == 1:
        return "; optional (its failure does not stop the job)"
    return f"; optional (its failure does not stop the job) {_text(optional[0], True)}"


def _cell_alternatives(pairs: list[tuple[str, str]]) -> str:
    """Different texts of one cell, each with the inputs that give it, one per line."""
    if len(pairs) == 1:
        return pairs[0][0]
    return "<br>".join(f"{_text(when, True)}: {text}" for text, when in pairs)


def _output_cell(name: str, found: list[_Variant]) -> str:
    pairs = _by_value(name, found, lambda p: f"`{p.output.__name__}`" if p.output else "none")
    return _cell_alternatives(pairs)


def _fan_out_cell(name: str, found: list[_Variant]) -> str:
    pairs = [(t.replace("|", "\\|") if t else "no", w)
             for t, w in _by_value(name, found, lambda p: p.fan_out)]
    return _cell_alternatives(pairs)


def _phases_section(found: list[_Variant]) -> list[str]:
    from fastworkflow.observation_offloading.facts import FactSourceError, fact_sources

    order = _phase_order(found)
    lines = ["## Phases", ""]
    every = [m for v in found if v.render is not None for m in v.members]
    phased = _universe(found)
    if len(phased) < len(every):
        when = describe_inputs(phased, every, _axes_of(found))
        lines += [_text(f"`plan()` returns phases only {when}" if when.startswith("when ")
                        else f"`plan()` returns phases {when}")
                  + "; for other inputs the call runs flat, from `steps()`. The table is "
                  "about the phased calls.", ""]
    lines += ["| phase | output | runs | fans out |", "|---|---|---|---|"]
    for name in order:
        lines.append(f"| `{name}` | {_output_cell(name, found)} "
                     f"| {_runs_when(name, found)} | {_fan_out_cell(name, found)} |")
    lines += ["", "## Facts", "",
              "What each phase's output holds, filled in code from the commands the phase ran.",
              ""]
    shown = False
    described: dict[type, str] = {}
    for name in order:
        outputs = list(dict.fromkeys(p.output for _, p in _with_phase(name, found) if p.output))
        for output in outputs:
            shown = True
            lines += [f"### `{name}` -> `{output.__name__}`", ""]
            if output in described:
                lines += [f"The same model as `{described[output]}`.", ""]
                continue
            described[output] = name
            doc = inspect.getdoc(output)
            if doc and output.__doc__:
                lines += [_text(doc), ""]
            try:
                sources = {s.field: s for s in fact_sources(output)}
            except FactSourceError as error:
                sources, lines = {}, lines + [f"Malformed \"from\": {_text(error)}", ""]
            lines += ["| fact | type | from |", "|---|---|---|"]
            for field_name, info in output.model_fields.items():
                source = sources.get(field_name)
                origin = _code(f"{source.command}.{source.source_field}") if source else "not filled from a command"
                lines.append(f"| `{field_name}` | {_code(_type_text(info.annotation)).replace('|', chr(92) + '|')} | {origin} |")
            lines.append("")
    if not shown:
        lines += ["No phase passes facts on.", ""]
    return lines


#: The line a phase whose steps were built from placeholder facts gets.
PLACEHOLDER_NOTE = ("Steps below are rendered with placeholder facts: `<phase.fact>` stands "
                    "for that fact as read at run time, and 999999 (or 999,999) for a number "
                    "fact the code converted; values computed from facts (counts, comparisons, "
                    "matches) are not real.")
#: The line a render for inputs ``plan()`` returns no phases for gets.
FLAT_NOTE = "`plan()` returns no phases for these inputs: the call runs flat, from `steps()`."
#: The line a fan-out phase over an empty list of items gets.
NO_ITEMS_NOTE = "No items: `over` is empty for these inputs, so the phase runs nothing."


def _render_lines(render: _Render, level: str, first: Optional[_Render] = None,
                  first_title: str = "") -> list[str]:
    """*render* as markdown. A phase the same as in *first* (the first plan shown) is one line."""
    flat = [FLAT_NOTE, ""] if render.flat else []
    if render.error:
        return flat + [f"The steps cannot be rendered: {_text(render.error)}", ""]
    if not render.phases:
        return flat + _numbered(render.steps) + [""]
    same = {p.key() for p in first.phases} if first is not None else set()
    lines = []
    for index, phase in enumerate(render.phases, 1):
        lines += [f"{level} Phase {index}: `{phase.name}`", ""]
        if phase.key() in same:
            lines += [f"As in the plan {first_title}.", ""]
            continue
        about = [f"Reads: {', '.join(f'`{r}`' for r in phase.reads)}." if phase.reads else "Reads no earlier facts.",
                 f"Output: `{phase.output.__name__}`." if phase.output else "Output: none."]
        if phase.settings:
            about.append(f"Settings: {', '.join(_code(s) for s in phase.settings)}.")
        lines += [" ".join(about), ""]
        if phase.note:
            lines += [f"_{_text(phase.note)}_", ""]
        if phase.placeholders:
            lines += [f"_{PLACEHOLDER_NOTE}_", ""]
        if phase.items is None:
            lines += _numbered(phase.steps) + [""]
        elif not phase.items:
            lines += [NO_ITEMS_NOTE, ""]
        else:
            for head, steps in phase.items:
                lines += [head, ""]
                lines += _numbered(steps) + [""]
    if render.left_out:
        lines += [f"Left out (`when` is false): {', '.join(f'`{n}`' for n in render.left_out)}.", ""]
    return lines


def _steps_section(found: list[_Variant], changing: list[str], phased: bool) -> list[str]:
    title = "Plan" if phased else "Steps"
    lines = [f"## {title}", ""]
    rendered = [v for v in found if v.render is not None]
    shown = [v for v in rendered if v.shown]
    refused = [v for v in found if v.render is None]
    if not changing:
        lines += ["For the sample inputs; no input changes them.", ""]
        if rendered:
            lines += _render_lines(rendered[0].render, "###")
    else:
        names = ", ".join(f"`{n}`" for n in changing)
        one, noun = ("plan", "plans") if phased else ("list of steps", "lists of steps")
        if len(shown) < len(rendered):
            lines += [f"The inputs that change them are {names}. They give {len(rendered)} "
                      f"different {noun}, too many to show each: shown are the {len(shown)} "
                      "that a pairwise cover of the inputs reaches (every two inputs' values "
                      "together at least once) or that have a phase layout no other shown one "
                      "has. Each is titled by the inputs that decide it." + (
                          " The phases table covers all of them." if phased else ""), ""]
        else:
            lines += [f"For the sample inputs, once for each different {one} the inputs "
                      f"that change them give: {names}. Each is titled by the inputs that "
                      "decide it.", ""]
        first = shown[0] if shown else None
        for variant in shown:
            lines += [f"### {title} {_text(variant.title())}", ""]
            lines += _render_lines(variant.render, "####",
                                   first.render if phased and variant is not first else None,
                                   _text(first.title()) if first else "")
    if refused:
        lines += ["Not rendered, as no skill call reaches `steps()` with them "
                  "(`validate_extracted_parameters` changes or refuses them):", ""]
        lines += [f"- {_text(variant.title()) or 'the sample inputs'}: {_text(variant.refused)}"
                  for variant in refused] + [""]
    return lines


def render_preview(name: str, description: str, steps: tuple, command,
                   utterances: tuple, source: str) -> str:
    """The markdown preview of one skill (see the module doc). *source* is the skill file's
    path relative to the workflow folder (``_skills/<name>.py``)."""
    phased = command is not None and command.plan_fn is not None
    kind = ("phased: `plan()` runs it as bounded sub-plans, each passing facts on" if phased
            else "flat: one list of steps" if command is not None
            else "fixed steps, no inputs")
    lines = [MARKER.format(source=source), "", f"# {name}", "", _text(description), "",
             f"Source: `{source}`. Kind: {kind}.", ""]
    lines += _inputs_section(command)
    lines += _utterances_section(utterances)
    if command is None:
        lines += ["## Steps", ""] + _numbered(list(steps)) + [""]
    else:
        found, changing = variants(command)
        if phased:
            lines += _phases_section(found)
        lines += _steps_section(found, changing, phased)
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- a workflow's previews

def default_out_dir(workflow_folder: str) -> str:
    """``docs/skill_previews`` next to the workflow folder: ``<workflow_folder>/../docs/skill_previews``."""
    return os.path.join(os.path.dirname(os.path.abspath(workflow_folder)), "docs", "skill_previews")


def render_previews(workflow_folder: str) -> dict[str, str]:
    """``{"<skill>.md": preview}`` for every skill of the workflow, by skill name.

    Raises PreviewError when the workflow has no ``_skills`` folder or a skill file
    could not be loaded (the loader logs and skips one that does not import), and the
    loader's SkillFormatError for a malformed one.
    """
    from fastworkflow import workflow_agent

    folder = os.path.abspath(workflow_folder)
    root = os.path.join(folder, "_skills")
    if not os.path.isdir(root):
        raise PreviewError(f"{workflow_folder} has no _skills folder")
    files = [name for name, _ in workflow_agent._skill_files(root)]
    loaded = workflow_agent.load_skills(folder)
    commands = workflow_agent.load_skill_commands(folder)
    utterances = workflow_agent.skill_utterances(folder)
    missing = sorted(set(files) - {name for name, _, _ in loaded})
    if missing:
        raise PreviewError(f"skill file(s) not loaded (see the log): {', '.join(missing)}")
    return {f"{name}.md": render_preview(name, description, steps, commands.get(name),
                                         utterances.get(name, ()), f"_skills/{name}.py")
            for name, description, steps in sorted(loaded)}


def check_previews(workflow_folder: str, out_dir: Optional[str] = None) -> list[str]:
    """The problems of the previews in *out_dir* against a fresh render; empty when current.

    A problem is a preview that differs, one that is missing, or an extra ``.md`` file.
    """
    out_dir = out_dir or default_out_dir(workflow_folder)
    fresh = render_previews(workflow_folder)
    present = sorted(f for f in os.listdir(out_dir) if f.endswith(".md")) if os.path.isdir(out_dir) else []
    problems = []
    for name, text in fresh.items():
        path = os.path.join(out_dir, name)
        if name not in present:
            problems.append(f"missing: {name}")
            continue
        with open(path, encoding="utf-8", newline="") as handle:
            if handle.read() != text:
                problems.append(f"stale: {name}")
    problems += [f"extra: {name}" for name in present if name not in fresh]
    return problems


def write_previews(workflow_folder: str, out_dir: Optional[str] = None) -> dict[str, list[str]]:
    """Write every skill's preview to *out_dir*; remove generated previews no skill renders.

    Returns ``{"written": [...], "unchanged": [...], "removed": [...], "kept": [...]}``
    (file names). An extra ``.md`` file without the generated marker is kept, not removed.
    """
    out_dir = out_dir or default_out_dir(workflow_folder)
    fresh = render_previews(workflow_folder)
    os.makedirs(out_dir, exist_ok=True)
    result: dict[str, list[str]] = {"written": [], "unchanged": [], "removed": [], "kept": []}
    for name, text in fresh.items():
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8", newline="") as handle:
                if handle.read() == text:
                    result["unchanged"].append(name)
                    continue
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        result["written"].append(name)
    for name in sorted(os.listdir(out_dir)):
        if not name.endswith(".md") or name in fresh:
            continue
        path = os.path.join(out_dir, name)
        with open(path, encoding="utf-8", errors="replace") as handle:
            generated = handle.readline().startswith("<!-- Generated by `fastworkflow skill preview`")
        if generated:
            os.remove(path)
            result["removed"].append(name)
        else:
            result["kept"].append(name)
    return result


def preview_main(args) -> int:
    """``fastworkflow skill preview``: write the previews, or with ``--check`` compare them."""
    out_dir = args.out or default_out_dir(args.workflow_folder)
    if args.check:
        problems = check_previews(args.workflow_folder, out_dir)
        if problems:
            print(f"skill previews in {out_dir} are not current "
                  f"(run: fastworkflow skill preview {args.workflow_folder}"
                  f"{' --out ' + args.out if args.out else ''}):")
            for problem in problems:
                print(f"  {problem}")
            return 1
        print(f"skill previews in {out_dir} are current")
        return 0
    result = write_previews(args.workflow_folder, out_dir)
    print(f"{out_dir}: {len(result['written'])} written, {len(result['unchanged'])} unchanged, "
          f"{len(result['removed'])} removed")
    for name in result["kept"]:
        print(f"  kept {name}: not a generated preview (--check reports it as extra)")
    return 0


__all__ = [
    "MARKER",
    "MAX_VARIANTS",
    "PLACEHOLDER_NUMBER",
    "PreviewError",
    "check_previews",
    "condition_text",
    "default_out_dir",
    "describe_inputs",
    "placeholder_output",
    "preview_main",
    "render_preview",
    "render_previews",
    "sample_value",
    "variants",
    "write_previews",
]
