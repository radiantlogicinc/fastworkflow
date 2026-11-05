"""Facts filled in code from the command runs a run archived.

A *facts model* is a pydantic model whose fields are typed with the commands that
produce them (design 3.12)::

    class Found(BaseModel):
        listing: ListItems.Output      # every field of that command's Output
        query: Optional[FindItem.Input] = None

A field typed with a command's ``Signature.Output`` gets that command's Output from its
run in the phase; a field typed with its ``Signature.Input`` gets the arguments that run
was called with. Properties, methods and validators compute any other value. A field
typed with anything else is a load error. The copy edges of the earlier design
(``fact_from``, a ``"from"`` on a facts field) are no longer read (design 2.33,
Phase 15): such a field is a load error that says so.

The typed fill is ``typed_phase_facts.fill_typed_phase_output``. It uses the paging and
query rules below (``_page_key``, ``_join_list``, ``_scalar_of_pages``), so the
rules apply to typed facts as they did to copies.

Facts carry VALUES (ids, labels, counts), never observation aliases: ``O<n>``
numbering restarts with each executor run.

Only successful runs count, and, when a context is given, only runs that ran in that
context (the ``"<Context> <label>"`` clause the archive stores).

A required field with no value fails with ``MissingFactError`` naming the field. It is
never guessed. A field with a default keeps it.

Pages and queries
-----------------

A run's Input is normalised before runs are compared. A parameter counts as NOT
GIVEN, and is left out of the key, when its value is:

* ``None``, or text that is empty or spells ``none``/``null`` (what an extractor
  writes for a missing value);
* the "not given" sentinel parameter extraction fills in (the ``NOT_FOUND`` env
  value), or the extractor's invalid-number marker (``INVALID_INT_VALUE`` /
  ``INVALID_FLOAT_VALUE``);
* equal to the field's default in the command's ``Signature.Input`` (only when
  ``signatures`` is given, since that is where the defaults are known);
* for a paging parameter, ``0``: the first row is where an omitted position starts.

The paging parameters are the Input parameters named as a row position: ``start`` (``PAGING_PARAMETERS``).
parameters form the run's QUERY; the paging ones say which PAGE of it the run read.

A list value is a page of its query's rows. The pages joined are those of the
BROADEST query that ran -- the one with the fewest given parameters; between
equally broad queries, the one run latest -- one page per distinct paging
position, a re-read replacing the earlier read of that page. A run of any other
query is a narrowed (filtered) or different read and is left out, so a filtered
lookup never adds its rows to the full list. A non-list value (one per run, e.g.
a uid each run opened) is still appended for every distinct normalised Input.

A scalar field follows the same rule: its value comes from the runs of the
broadest query (between equally broad queries, the one run latest), the value
each returned (or ran with, for a field of the Input). The page does not narrow
the query, so a total read on a later page of the broadest query counts as much
as one read on its first page. A whole-list read followed by a filtered read
therefore gives the whole list's total, not the filtered one's.

When that query was read over several pages, the scalar is taken from the page
that carries it (``_scalar_of_pages``), each distinct page counted once:

* a later page that listed no row (every list field of its Output empty: a page
  read past the end) carries nothing;
* ``None`` is not carried (a page that does not report the value, e.g. a
  whole-scope count only the first page computes);
* a bool is the union of the pages: true when any page says so;
* any other value is that of the latest page that carries one.

So a later page, or an empty page past the end, never overwrites a count, total
or flag an earlier page carried. When every run is one page of one query, the
latest run wins as before.
"""
from __future__ import annotations

import json
import os
import types
import typing
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Optional, Sequence

from pydantic import BaseModel, ValidationError

from fastworkflow.plan_frame import runs_outside_ranges


class FactsError(ValueError):
    """A facts model could not be filled."""


class MissingFactError(FactsError):
    """A required field has no value in the runs.

    ``source`` is the field's ``FactSource`` (``None`` when not known).
    """

    def __init__(self, message: str = "", source: Optional["FactSource"] = None):
        super().__init__(message)
        self.source = source


#: Input parameters taken as a row position (a page of the same query) when a
#: facts field does not name its own with ``"page_by"``.
PAGING_PARAMETERS = ("start",)


def _is_int_annotation(annotation: Any) -> bool:
    """Whether a field annotation is ``int`` or ``Optional[int]`` (not ``bool``, not ``datetime``)."""
    if annotation is int:
        return True
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        args = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        return len(args) == 1 and args[0] is int
    return False


def paging_parameters(input_model: Any) -> tuple[str, ...]:
    """The paging parameters of a command's ``Input`` (design 2.49, 3.14): the fields named ``start``,
    whose type is ``int`` or ``Optional[int]``, in ``PAGING_PARAMETERS`` order.

    ``()`` for no ``Input`` or no such field: a ``start`` that is a ``datetime`` or a ``skip`` that is
    a ``bool`` is an ordinary field. Every reader of ``PAGING_PARAMETERS`` for a command asks this.
    """
    fields = getattr(input_model, "model_fields", None)
    if not isinstance(fields, Mapping):
        return ()
    return tuple(name for name in PAGING_PARAMETERS
                 if name in fields and _is_int_annotation(fields[name].annotation))


def output_page(output: Any) -> Optional[tuple[int, int]]:
    """``(rows, total)`` of an ``Output`` (a model or its dict) that is a page (design 3.14): its
    ``total`` is an int and its list fields share one length, the rows of one page. None otherwise."""
    values = output.model_dump() if isinstance(output, BaseModel) else output
    if not isinstance(values, Mapping):
        return None
    total = values.get("total")
    if not isinstance(total, int) or isinstance(total, bool):
        return None
    lengths = {len(value) for name, value in values.items()
               if name != "total" and isinstance(value, (list, tuple))}
    if len(lengths) != 1:
        return None
    return lengths.pop(), total


@dataclass(frozen=True)
class FactSource:
    """One field's declared source."""

    field: str
    command: str
    source_field: str
    is_list: bool
    required: bool
    page_by: tuple[str, ...] = PAGING_PARAMETERS

    @property
    def spelled(self) -> str:
        return f"{self.command}.{self.source_field}"


def fact_sources(model: type[BaseModel]) -> list[FactSource]:
    """The copy sources of *model*: always none (design 2.33, Phase 15).

    A facts model's field is typed with the command it comes from (3.12), so no field copies
    another command's value and nothing is parsed from a ``"from"``. A field that still
    carries a ``"from"`` is a load error (``typed_phase_facts.check_typed_facts_model``).
    Kept, returning ``[]``, for callers that take a source list.
    """
    return []


def _command_matches(declared: str, ran: str) -> bool:
    if declared == ran:
        return True
    return "/" not in declared and ran.split("/")[-1] == declared


def workflow_signatures(workflow_folderpath: str) -> dict[str, Any]:
    """Each command of the workflow at *workflow_folderpath*, mapped to its ``Signature`` class.

    Commands without a ``Signature`` are left out.
    """
    import fastworkflow
    from fastworkflow.command_routing import RoutingRegistry

    routing = RoutingRegistry.get_definition(workflow_folderpath)
    names = {name for commands in routing.contexts.values() for name in commands}
    signatures: dict[str, Any] = {}
    for name in sorted(names):
        try:
            signature = routing.get_command_class(
                name, fastworkflow.ModuleType.INPUT_FOR_PARAM_EXTRACTION_CLASS)
        except Exception:  # noqa: BLE001 - a command with no Signature module
            signature = None
        if signature is not None:
            signatures[name] = signature
    return signatures


def _not_given_sentinels() -> tuple[Any, ...]:
    from fastworkflow.utils.signatures import INVALID_FLOAT_VALUE, INVALID_INT_VALUE, NOT_FOUND

    return (NOT_FOUND, INVALID_INT_VALUE, INVALID_FLOAT_VALUE)


def _given(value: Any, sentinels: tuple[Any, ...]) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and value.strip().strip("\"'`").lower() in ("", "none", "null"):
        return False
    return not any(_same(value, sentinel) for sentinel in sentinels)


def _page_key(
    run_input: Any,
    page_by: Iterable[str],
    defaults: Optional[Mapping[str, Any]],
) -> tuple[str, str, int]:
    """``(query, page, size)`` of a run's Input, normalised as the module docstring says.

    *query* is the given non-paging parameters and *page* the given paging ones, each
    as canonical JSON; *size* is how many parameters the query gives. *defaults* are
    the command's Input defaults, when known.
    """
    if not isinstance(run_input, Mapping):
        return json.dumps(run_input, sort_keys=True, default=str), "{}", 0
    sentinels = _not_given_sentinels()
    paging = set(page_by)
    query: dict[str, Any] = {}
    page: dict[str, Any] = {}
    for name, value in run_input.items():
        if not _given(value, sentinels):
            continue
        if defaults and defaults.get(name) is not None and _same(value, defaults[name]):
            continue
        if name in paging:
            if _same(value, 0) and not isinstance(value, bool):
                continue  # the first row: where an omitted position starts
            page[name] = value
        else:
            query[name] = value
    return (json.dumps(query, sort_keys=True, default=str),
            json.dumps(page, sort_keys=True, default=str), len(query))


def _same(value: Any, other: Any) -> bool:
    try:
        return bool(value == other)
    except Exception:  # noqa: BLE001 - values that cannot be compared differ
        return False


def _join_list(found: list[tuple[str, str, int, Any, Any]]) -> list[Any]:
    """Join the values of *found* ``(query, page, query_size, value, output)``, in run order."""
    pages = [item for item in found if isinstance(item[3], (list, tuple))]
    query = None
    if pages:
        broadest = min(size for _q, _p, size, _v, _o in pages)
        query = [q for q, _p, size, _v, _o in pages if size == broadest][-1]
    by_key: dict[tuple[str, str], Any] = {}
    for run_query, run_page, _size, value, _output in found:
        if isinstance(value, (list, tuple)) and run_query != query:
            continue  # a narrowed or different query: not a page of the list
        by_key[(run_query, run_page)] = value
    joined: list[Any] = []
    for value in by_key.values():
        if isinstance(value, (list, tuple)):
            joined.extend(value)
        else:
            joined.append(value)
    return joined


#: The page key of a run that gave no paging parameter: the first page.
_FIRST_PAGE = "{}"


def _listed_nothing(output: Any) -> bool:
    """Whether a run's Output has list fields and every one of them is empty (no row)."""
    if not isinstance(output, Mapping):
        return False
    lists = [value for value in output.values() if isinstance(value, (list, tuple))]
    return bool(lists) and not any(lists)


def _scalar_of_pages(found: list[tuple[str, str, int, Any, Any]]) -> Any:
    """A scalar's value from *found* ``(query, page, query_size, value, output)``, in run order.

    The query is the broadest one (the fewest given parameters; between equally broad
    queries, the one run latest); the page a run read does not narrow it. Of its
    runs, each distinct page counts once (a re-read replaces the earlier read), and:

    * a later page that listed no row (an empty page past the end) carries nothing;
    * a value of ``None`` is not carried (the page did not report it);
    * a bool is the union of the pages that carry it: true when any page says so;
    * any other value is that of the latest page that carries one.

    So a whole-scope count the first page carries is never overwritten by a later
    page that reports none, or by an empty page read past the end.
    """
    broadest = min(size for _q, _p, size, _v, _o in found)
    query = [q for q, _p, size, _v, _o in found if size == broadest][-1]
    by_page: dict[str, tuple[Any, Any]] = {}
    for run_query, run_page, _size, value, output in found:
        if run_query == query:
            by_page.pop(run_page, None)  # a re-read moves the page to its latest place
            by_page[run_page] = (value, output)
    pages = [(page, value, output) for page, (value, output) in by_page.items()]
    carrying = [item for item in pages
                if item[0] == _FIRST_PAGE or not _listed_nothing(item[2])] or pages
    values = [value for _page, value, _output in carrying if value is not None]
    if not values:
        return carrying[-1][1]
    if all(isinstance(value, bool) for value in values):
        return any(values)
    return values[-1]


def capped_items(items: Iterable[str], max_items: Optional[int], sep: str = ", ") -> str:
    """*items* joined with *sep*; past *max_items*, the first ones and
    ``"(and K more; M in all)"``. ``None`` (or 0) caps nothing."""
    items = list(items)
    if not max_items or len(items) <= max_items:
        return sep.join(items)
    return (f"{sep.join(items[:max_items])} (and {len(items) - max_items} more; "
            f"{len(items)} in all)")


def _fact_value_text(value: Any, max_items: Optional[int] = None) -> str:
    if value is None:
        return "none"
    if isinstance(value, BaseModel):
        return f"({facts_text(value, max_items=max_items)})"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Mapping):
        return f"({facts_text(value, max_items=max_items)})" if value else "none"
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=str) if isinstance(value, (set, frozenset)) else list(value)
        return capped_items((_fact_value_text(item, max_items) for item in items),
                            max_items) or "none"
    return " ".join(str(value).split())


def facts_text(facts: Any, *, fields: Optional[Iterable[str]] = None, sep: str = "; ",
               max_items: Optional[int] = None) -> str:
    """*facts* (a filled facts model) as compact ``field: value`` text, for a planning command's steps.

    ``uid: b1; loan_uids: l1, l2; total: 3``. A list is joined with ``", "`` (``none`` when
    empty), ``None`` is ``none``, a bool ``true``/``false``, a nested model or mapping is
    shown the same way in parentheses, and whitespace in a value is collapsed to one space.
    Fields follow the model's order; *fields* picks some of them, in that order. A list of
    models (a fan-out phase's outputs) is each model in parentheses, joined with ``", "``.
    *max_items* caps each list: its first *max_items* items, then ``"(and K more; M in
    all)"`` (``capped_items``); by default nothing is capped, since a step that tells the
    answer to name a list must give it whole.

    The text is plain data: a phase that writes its answer from earlier facts (an
    answer-only phase, see ``fastworkflow.planning_phases``) reads the values as given.
    """
    if isinstance(facts, (list, tuple)):
        return ", ".join(f"({facts_text(item, fields=fields, sep=sep, max_items=max_items)})"
                         for item in facts) or "none"
    if isinstance(facts, BaseModel):
        values = {name: getattr(facts, name) for name in type(facts).model_fields}
    elif isinstance(facts, Mapping):
        values = dict(facts)
    else:
        raise TypeError(f"facts_text needs a facts model, got {type(facts).__name__}")
    names = list(values) if fields is None else [name for name in fields if name in values]
    return sep.join(f"{name}: {_fact_value_text(values[name], max_items)}" for name in names)


__all__ = [
    "FactSource",
    "FactsError",
    "MissingFactError",
    "PAGING_PARAMETERS",
    "capped_items",
    "fact_sources",
    "facts_text",
    "output_page",
    "paging_parameters",
    "workflow_signatures",
]
