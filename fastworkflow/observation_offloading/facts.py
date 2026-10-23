"""Facts filled in code from the command runs a run archived.

A *facts model* is a pydantic model whose fields each name where their value
comes from::

    class Found(BaseModel):
        item_uids: list[str] = Field(json_schema_extra={"from": "list_items.item_uids"})
        total: int = Field(json_schema_extra={"from": "list_items.total"})
        query: Optional[str] = Field(default=None, json_schema_extra={"from": "find_item.query"})

``"<command>.<field>"`` names a field of that command's ``Signature.Output``
(what it returned in ``CommandResponse.output``) or, when the Output has no
such field, of its ``Signature.Input`` (the parameters it ran with). The
command is spelled as the workflow names it: qualified (``Context/list_items``)
or plain (``list_items``). A plain name that more than one context defines is
resolved against the context of the record the runs were on (``context_name``,
else the stored context of the runs ``context`` selects): ``list_items`` on a
``Context`` record means ``Context/list_items``. One that still names more
than one command is rejected as an ambiguous source, never guessed.

Facts carry VALUES (ids, labels, counts), never observation aliases: ``O<n>``
numbering restarts with each executor run.

``facts_from_runs`` fills the model from runs as ``RuntimeHandleArchive.command_runs``
returns them, by these rules and nothing else -- no model call, no parsing of
response text:

* Only successful runs count, and, when ``context`` is given, only runs that
  ran in that context (the ``"<Context> <label>"`` clause the archive stores).
* A scalar field takes its value from the LATEST such run of the BROADEST query
  (see "Pages and queries" below) of the command whose stored Output (or Input)
  holds the field.
* A list field collects across such runs, in run order: a list value is
  joined, any other value appended. Which runs count is decided on each run's
  *normalised* Input (see "Pages and queries" below): a run with the same
  normalised Input as an earlier one (the same page read again) replaces that
  run's values in its place, so a re-read never duplicates rows; and list
  values (pages) are joined only across the runs of ONE query.
* A required field with no value fails with ``MissingFactError`` naming the
  field and its source. It is never guessed. A field with a default keeps it.
* A ``"from"`` that is malformed, or -- when ``signatures`` is given -- names a
  command or field that does not exist, fails with ``FactSourceError``.

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

The paging parameters are the ones a facts field names in ``"page_by"`` (a name
or a list of names: ``json_schema_extra={"from": "list_items.item_uids",
"page_by": "cursor"}``), else the Input parameters named as a row position:
``start``, ``offset`` or ``skip`` (``PAGING_PARAMETERS``). The other given
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
from typing import Any, Iterable, Mapping, Optional

from pydantic import BaseModel, ValidationError


class FactsError(ValueError):
    """A facts model could not be filled."""


class FactSourceError(FactsError):
    """A field's ``"from"`` is malformed or names no known command field."""


class AmbiguousFactSourceError(FactSourceError):
    """A plain command name in a ``"from"`` names more than one command."""


class MissingFactError(FactsError):
    """A required field has no value in the runs.

    ``source`` is the field's ``FactSource`` (``None`` when not known).
    """

    def __init__(self, message: str = "", source: Optional["FactSource"] = None):
        super().__init__(message)
        self.source = source


#: Input parameters taken as a row position (a page of the same query) when a
#: facts field does not name its own with ``"page_by"``.
PAGING_PARAMETERS = ("start", "offset", "skip")


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


def _is_list_annotation(annotation: Any) -> bool:
    origin = typing.get_origin(annotation)
    if origin in (list, tuple, set, frozenset):
        return True
    if origin in (typing.Union, types.UnionType):
        return any(
            _is_list_annotation(arg)
            for arg in typing.get_args(annotation)
            if arg is not type(None)
        )
    return annotation in (list, tuple, set, frozenset)


def fact_sources(model: type[BaseModel]) -> list[FactSource]:
    """Every field of *model* that declares a ``"from"``, parsed.

    Raises ``FactSourceError`` for a ``"from"`` that is not
    ``"<command>.<field>"``. Fields without one are not listed.
    """
    sources = []
    for name, info in model.model_fields.items():
        extra = info.json_schema_extra
        spelled = extra.get("from") if isinstance(extra, Mapping) else None
        if spelled is None:
            continue
        command, dot, source_field = str(spelled).strip().partition(".")
        if (not isinstance(spelled, str) or not dot or not command or not source_field
                or "." in source_field):
            raise FactSourceError(
                f"{model.__name__}.{name}: \"from\" must be \"<command>.<field>\", "
                f"got {spelled!r}"
            )
        page_by = extra.get("page_by", PAGING_PARAMETERS)
        if isinstance(page_by, str):
            page_by = (page_by,)
        if not all(isinstance(p, str) and p for p in page_by):
            raise FactSourceError(
                f"{model.__name__}.{name}: \"page_by\" must be a parameter name or a "
                f"list of them, got {extra.get('page_by')!r}"
            )
        sources.append(FactSource(
            field=name,
            command=command,
            source_field=source_field,
            is_list=_is_list_annotation(info.annotation),
            required=info.is_required(),
            page_by=tuple(page_by),
        ))
    return sources


def _command_matches(declared: str, ran: str) -> bool:
    if declared == ran:
        return True
    return "/" not in declared and ran.split("/")[-1] == declared


def _pick(model: type[BaseModel], source: FactSource, names: set[str],
          context_name: Optional[str]) -> str:
    """The one command *names* leaves for *source*, resolved by *context_name*."""
    if len(names) == 1:
        return next(iter(names))
    if context_name and f"{context_name}/{source.command}" in names:
        return f"{context_name}/{source.command}"
    raise AmbiguousFactSourceError(
        f"{model.__name__}.{source.field} (from {source.spelled!r}): ambiguous source: "
        f"{source.command!r} names {len(names)} commands ({', '.join(sorted(names))})"
        + (f" and none is in context {context_name!r}" if context_name else "")
        + "; qualify it as <Context>/<command>"
    )


def _model_fields(cls: Any) -> set[str]:
    fields = getattr(cls, "model_fields", None)
    return set(fields) if isinstance(fields, Mapping) else set()


def check_fact_sources(
    model: type[BaseModel],
    signatures: Mapping[str, Any],
    *,
    context_name: Optional[str] = None,
) -> list[FactSource]:
    """Check that every ``"from"`` of *model* names a known command and field.

    *signatures* maps each command's name, as the workflow spells it
    (``Context/command``, or plain for a root command), to its ``Signature``
    class (anything with optional ``Input``/``Output`` pydantic models), as
    ``workflow_signatures`` builds it. A plain name that several contexts
    define resolves to ``<context_name>/<command>``; without one it is an
    ``AmbiguousFactSourceError``. Returns the sources with ``command``
    replaced by the resolved name; raises ``FactSourceError``.
    """
    resolved = []
    for source in fact_sources(model):
        where = f"{model.__name__}.{source.field} (from {source.spelled!r})"
        matches = {name for name in signatures if _command_matches(source.command, name)}
        if not matches:
            raise FactSourceError(f"{where}: no command named {source.command!r}")
        name = _pick(model, source, matches, context_name)
        signature = signatures[name]
        known = (_model_fields(getattr(signature, "Output", None))
                 | _model_fields(getattr(signature, "Input", None)))
        if source.source_field not in known:
            raise FactSourceError(
                f"{where}: {name} has no Output or Input field "
                f"{source.source_field!r}"
            )
        resolved.append(replace(source, command=name))
    return resolved


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


def _value_in(run: Mapping[str, Any], field: str) -> tuple[bool, Any]:
    for part in ("output", "input"):
        data = run.get(part)
        if isinstance(data, Mapping) and field in data:
            return True, data[field]
    return False, None


def _not_given_sentinels() -> tuple[Any, ...]:
    import fastworkflow
    from fastworkflow.utils.signatures import INVALID_FLOAT_VALUE, INVALID_INT_VALUE

    not_found = fastworkflow.get_env_var(
        "NOT_FOUND", default=os.getenv("NOT_FOUND") or "NOT_FOUND")
    return (not_found, INVALID_INT_VALUE, INVALID_FLOAT_VALUE)


def _input_defaults(signature: Any) -> dict[str, Any]:
    """The defaults of *signature*'s ``Input`` fields that have one."""
    fields = getattr(getattr(signature, "Input", None), "model_fields", None)
    if not isinstance(fields, Mapping):
        return {}
    return {name: info.default for name, info in fields.items()
            if not info.is_required() and info.default_factory is None}


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


def facts_from_runs(
    model: type[BaseModel],
    runs: Iterable[Mapping[str, Any]],
    *,
    context: Optional[str] = None,
    context_name: Optional[str] = None,
    signatures: Optional[Mapping[str, Any]] = None,
) -> BaseModel:
    """Fill *model* from *runs*, by the rules in this module's docstring.

    *runs* are command runs in the order they ran, as
    ``RuntimeHandleArchive.command_runs`` returns them (several scopes' runs may
    be concatenated in order). *context*, when given, keeps only the runs on
    that record (their ``context_clause``). *context_name* is the record's
    context, which plain command names resolve against; by default the stored
    context of the runs *context* selects. *signatures*, when given, is
    checked first with ``check_fact_sources``.
    """
    eligible = [
        run for run in runs
        if run.get("success") and (context is None or run.get("context_clause") == context)
    ]
    if context_name is None and context is not None:
        names = {str(run["context_name"]) for run in eligible if run.get("context_name")}
        if len(names) == 1:
            context_name = next(iter(names))
        elif not names:
            # Runs stored without a context name: the clause starts with it.
            context_name = context.split(" ", 1)[0] or None
    sources = (check_fact_sources(model, signatures, context_name=context_name)
               if signatures is not None else fact_sources(model))
    values: dict[str, Any] = {}
    for source in sources:
        matching = [run for run in eligible
                    if _command_matches(source.command, str(run.get("command") or ""))]
        ran = {str(run.get("command") or "") for run in matching}
        if len(ran) > 1:
            chosen = _pick(model, source, ran, context_name)
            matching = [run for run in matching if run.get("command") == chosen]
        found = []
        for run in matching:
            present, value = _value_in(run, source.source_field)
            if present:
                found.append((run.get("command"), run.get("input"), value, run.get("output")))
        if not found:
            if source.required:
                raise MissingFactError(
                    f"{model.__name__}.{source.field}: no successful run of "
                    f"{source.command!r} holds {source.source_field!r} "
                    f"(from {source.spelled!r})"
                    + (f" in context {context!r}" if context is not None else ""),
                    source=source,
                )
            continue
        keyed = []
        for command, run_input, value, output in found:
            defaults = (_input_defaults(signatures.get(str(command or "")))
                        if signatures is not None else None)
            keyed.append((*_page_key(run_input, source.page_by, defaults), value, output))
        values[source.field] = (_join_list(keyed) if source.is_list
                                else _scalar_of_pages(keyed))
    try:
        return model.model_validate(values)
    except ValidationError as error:
        raise FactsError(f"{model.__name__}: the values found do not fit the model: {error}") from error


def facts_from_archive(
    model: type[BaseModel],
    archive: Any,
    scope: Any,
    *,
    context: Optional[str] = None,
    context_name: Optional[str] = None,
    signatures: Optional[Mapping[str, Any]] = None,
) -> BaseModel:
    """``facts_from_runs`` over the command runs *archive* holds for *scope*."""
    return facts_from_runs(
        model, archive.command_runs(scope), context=context,
        context_name=context_name, signatures=signatures,
    )


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
    """*facts* (a filled facts model) as compact ``field: value`` text, for a skill's steps.

    ``uid: b1; loan_uids: l1, l2; total: 3``. A list is joined with ``", "`` (``none`` when
    empty), ``None`` is ``none``, a bool ``true``/``false``, a nested model or mapping is
    shown the same way in parentheses, and whitespace in a value is collapsed to one space.
    Fields follow the model's order; *fields* picks some of them, in that order. A list of
    models (a fan-out phase's outputs) is each model in parentheses, joined with ``", "``.
    *max_items* caps each list: its first *max_items* items, then ``"(and K more; M in
    all)"`` (``capped_items``); by default nothing is capped, since a step that tells the
    answer to name a list must give it whole.

    The text is plain data: a phase that writes its answer from earlier facts (an
    answer-only phase, see ``fastworkflow.skill_phases``) reads the values as given.
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
    "AmbiguousFactSourceError",
    "FactSource",
    "FactSourceError",
    "FactsError",
    "MissingFactError",
    "PAGING_PARAMETERS",
    "capped_items",
    "check_fact_sources",
    "fact_sources",
    "facts_from_archive",
    "facts_from_runs",
    "facts_text",
    "workflow_signatures",
]
