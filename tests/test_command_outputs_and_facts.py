"""Command outputs carried and archived as JSON; facts filled from them (fix-hia1.1).

Offline, no model call. The commands and models are a neutral library example.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import dspy
import pytest
from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow import CommandOutput, CommandResponse, tracing
from fastworkflow.observability import store as observability_store
from fastworkflow.observation_offloading import (
    AmbiguousFactSourceError,
    FactSourceError,
    MissingFactError,
    RuntimeHandleArchive,
    RuntimeHandleScope,
    archive_step,
    check_fact_sources,
    command_run_record,
    facts_from_archive,
    facts_from_runs,
)
from fastworkflow.observation_offloading.agent import build_compacting_step
from fastworkflow.observation_offloading.labels import alias_line
from fastworkflow.observation_offloading.state import (
    note_command_run,
    reset_observation_state,
)
from fastworkflow.utils.react import fastWorkflowReAct
from fastworkflow.workflow_agent import _hold_command_run

pytestmark = pytest.mark.usefixtures("reference_agent_window")

SCOPE = RuntimeHandleScope(channel_id="fixture-channel", turn_key="fixture-turn")
SHELF = "Shelf C4"


class ListBooks:
    class Input(BaseModel):
        start: Optional[int] = None

    class Output(BaseModel):
        book_uids: list[str]
        titles: list[str]
        total: int


class FindBook:
    class Input(BaseModel):
        query: str

    class Output(BaseModel):
        uid: str


SIGNATURES = {"Shelf/list_books": ListBooks, "Library/find_book": FindBook}


@pytest.fixture(autouse=True)
def _clean_state():
    reset_observation_state()
    yield
    reset_observation_state()


@pytest.fixture
def store(tmp_path):
    return RuntimeHandleArchive(str(tmp_path / "observability.sqlite3"))


def run(command, *, output=None, parameters=None, success=True, context=SHELF):
    return command_run_record(command_name=command, success=success, context_clause=context,
                              context_name=context.split(" ")[0] if context else "",
                              parameters=parameters, output=output)


def execute_step(index, command, text):
    return {
        f"tool_name_{index}": "execute_workflow_query",
        f"tool_args_{index}": {"command": command},
        f"observation_{index}": alias_line(f"O{index}") + text,
    }


# ----------------------------------------------------------------------
# CommandResponse.output
# ----------------------------------------------------------------------


def test_a_command_response_carries_its_output_and_defaults_to_none():
    out = ListBooks.Output(book_uids=["b1"], titles=["Dune"], total=1)
    assert CommandResponse(response="1 book").output is None
    assert CommandResponse(response="1 book", output=out).output is out
    dumped = CommandResponse(response="1 book", output=out).model_dump(mode="json")
    assert dumped["output"] == {"book_uids": ["b1"], "titles": ["Dune"], "total": 1}
    restored = CommandResponse.model_validate(dumped)
    assert restored.output == dumped["output"]


# ----------------------------------------------------------------------
# The archive
# ----------------------------------------------------------------------


def test_a_command_with_an_output_is_archived_as_json_beside_unchanged_text(store):
    trajectory = execute_step(0, "list_books", "1 book.\nb1  Dune")
    out = ListBooks.Output(book_uids=["b1"], titles=["Dune"], total=1)
    note_command_run(SCOPE, "O0", run("Shelf/list_books", output=out,
                                      parameters=ListBooks.Input(start=None)))
    assert archive_step(trajectory, 0, scope=SCOPE, selected_archive=store)

    assert store.get(SCOPE, "O0")["text"] == "1 book.\nb1  Dune"
    assert trajectory["observation_0"] == alias_line("O0") + "1 book.\nb1  Dune"
    assert store.command_runs(SCOPE) == [{
        "alias": "O0", "step_index": 0, "ordinal": 0, "command": "Shelf/list_books",
        "success": True, "context_clause": SHELF, "context_name": "Shelf",
        "input": {"start": None},
        "output": {"book_uids": ["b1"], "titles": ["Dune"], "total": 1},
    }]


def test_a_command_without_an_output_is_archived_as_before(store, tmp_path):
    plain = RuntimeHandleArchive(str(tmp_path / "plain.sqlite3"))
    archive_step(execute_step(0, "show_details", "title: Dune"), 0, scope=SCOPE, selected_archive=plain)
    note_command_run(SCOPE, "O0", run("Book/show_details"))
    archive_step(execute_step(0, "show_details", "title: Dune"), 0, scope=SCOPE, selected_archive=store)

    assert store.get(SCOPE, "O0") == plain.get(SCOPE, "O0")
    assert store.command_runs(SCOPE)[0]["output"] is None
    assert plain.command_runs(SCOPE) == []


def test_an_archive_with_the_older_persist_signature_still_works():
    seen = {}

    class OlderArchive:
        def persist(self, scope, *, alias, command_name, step_index, text, text_sha256,
                    context_clause=None, context_changed=False):
            seen["text"] = text

        def persist_event(self, scope, event):
            pass

    assert archive_step(execute_step(0, "show_details", "title: Dune"), 0,
                        scope=SCOPE, selected_archive=OlderArchive())
    assert seen["text"] == "title: Dune"


def test_command_run_json_is_redacted_like_the_text(store, monkeypatch):
    secret = "sk-" + "a" * 24
    note_command_run(SCOPE, "O0", run("Library/find_book", output={"uid": secret},
                                      parameters={"query": "Dune"}))
    archive_step(execute_step(0, "find_book", "b1"), 0, scope=SCOPE, selected_archive=store)
    assert store.command_runs(SCOPE)[0]["output"] == {"uid": "[REDACTED]"}


def test_old_rows_and_a_store_from_before_the_runs_table_still_read(tmp_path):
    path = tmp_path / "observability.sqlite3"
    first = RuntimeHandleArchive(str(path))
    archive_step(execute_step(0, "show_details", "title: Dune"), 0, scope=SCOPE, selected_archive=first)
    # The DB as an earlier build of this schema version left it: no runs table.
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE offload_command_runs")
    assert first.command_runs(SCOPE) == []
    assert first.get(SCOPE, "O0")["text"] == "title: Dune"

    reopened = RuntimeHandleArchive(str(path))
    assert reopened.get(SCOPE, "O0")["text"] == "title: Dune"
    assert reopened.command_runs(SCOPE) == []
    assert observability_store.ObservabilityStore(str(path)).has_feature(
        observability_store.FEATURE_OFFLOAD_COMMAND_RUNS_V1)
    note_command_run(SCOPE, "O1", run("Book/show_details"))
    archive_step(execute_step(1, "show_details", "title: Dune"), 1, scope=SCOPE,
                 selected_archive=reopened)
    assert [r["alias"] for r in reopened.command_runs(SCOPE)] == ["O1"]


def test_forgetting_a_channel_erases_its_command_runs(store):
    note_command_run(SCOPE, "O0", run("Library/find_book", output={"uid": "b1"}))
    archive_step(execute_step(0, "find_book", "b1"), 0, scope=SCOPE, selected_archive=store)
    observability_store.ObservabilityStore(store.db_path).forget_channel(SCOPE.channel_id)
    assert store.command_runs(SCOPE) == []


class _Signature(dspy.Signature):
    user_query: str = dspy.InputField()
    answer: str = dspy.OutputField()


def test_the_execute_step_holds_its_command_output_for_the_archive(store):
    """The real ReAct loop and compaction hook; only the reasoning call is scripted."""
    host = SimpleNamespace(
        current_turn_key=SCOPE.turn_key,
        observability_channel_id=SCOPE.channel_id,
        observability_experiment_claim={},
        trace_span_stack=[],
    )
    out = FindBook.Output(uid="b1")

    def execute_workflow_query(command: str) -> str:
        """Run one command against the workflow."""
        command_output = CommandOutput(
            command_response=CommandResponse(response="b1  Dune", output=out),
            command_name="Library/find_book",
            command_parameters=FindBook.Input(query="Dune"),
        )
        _hold_command_run(host, command_output, "Library")
        return alias_line("O0") + "b1  Dune"

    agent = fastWorkflowReAct(
        _Signature, tools=[execute_workflow_query], max_iters=3,
        on_step_complete=build_compacting_step(
            host, selected_archive=store, on_step_complete=lambda idx, trajectory: False),
    )
    host.workflow_tool_agent = agent
    agent.react = lambda **kwargs: SimpleNamespace(
        next_thought="find it", next_tool_name="execute_workflow_query",
        next_tool_args={"command": "find_book query=Dune"})
    trajectory = agent.trajectory  # the loop's own, as forward() runs it
    with tracing.host_scope(host):
        agent._run_loop(trajectory, 0, {"user_query": "find Dune"}, 3, 0)

    assert trajectory["observation_0"] == alias_line("O0") + "b1  Dune"
    [stored] = store.command_runs(SCOPE)
    assert (stored["command"], stored["input"], stored["output"]) == (
        "Library/find_book", {"query": "Dune"}, {"uid": "b1"})


# ----------------------------------------------------------------------
# Facts
# ----------------------------------------------------------------------


class Found(BaseModel):
    book_uids: list[str] = Field(json_schema_extra={"from": "list_books.book_uids"})
    total: int = Field(json_schema_extra={"from": "list_books.total"})
    query: str = Field(json_schema_extra={"from": "Library/find_book.query"})
    uid: Optional[str] = Field(default=None, json_schema_extra={"from": "find_book.uid"})


def page(uids, start=None, total=3, context=SHELF):
    return run("Shelf/list_books", parameters={"start": start}, context=context,
               output={"book_uids": uids, "titles": uids, "total": total})


RUNS = [
    run("Library/find_book", parameters={"query": "Dune"}, output={"uid": "x0"}, context="Library"),
    run("Library/find_book", parameters={"query": "Emma"}, output={"uid": "x1"}, context="Library"),
    page(["b1", "b2"]),
    run("Shelf/list_books", parameters={"start": 2}, success=False, output=None),
    page(["b3"], start=2),
    page(["b1", "b2"]),  # the first page read again
]


def test_facts_fill_from_outputs_and_inputs_with_exact_values():
    facts = facts_from_runs(Found, RUNS, signatures=SIGNATURES)
    assert facts == Found(book_uids=["b1", "b2", "b3"], total=3, query="Emma", uid="x1")


def test_list_fields_join_pages_in_order_and_skip_failed_runs():
    facts = facts_from_runs(Found, RUNS[:3] + RUNS[3:5])
    assert facts.book_uids == ["b1", "b2", "b3"]


def test_a_scalar_takes_the_latest_successful_run():
    runs = [page(["b1"], total=1), page(["b2"], start=1, total=7),
            run("Shelf/list_books", output={"book_uids": [], "titles": [], "total": 9},
                parameters={"start": 5}, success=False)]
    assert facts_from_runs(Found, runs + RUNS[:1]).total == 7


def test_a_run_in_another_context_is_not_used():
    runs = [page(["b1"], total=1), page(["z9"], total=5, context="Shelf D1"),
            run("Library/find_book", parameters={"query": "Dune"}, output={"uid": "x0"})]
    facts = facts_from_runs(Found, runs, context=SHELF)
    assert (facts.book_uids, facts.total) == (["b1"], 1)


def test_a_missing_required_field_names_the_field_and_its_source():
    with pytest.raises(MissingFactError, match=r"Found\.query.*'Library/find_book\.query'"):
        facts_from_runs(Found, [page(["b1"])])


def test_an_optional_field_with_no_value_keeps_its_default():
    runs = [page(["b1"]), run("Library/find_book", parameters={"query": "Dune"})]
    assert facts_from_runs(Found, runs).uid is None


def test_an_unknown_command_or_field_is_rejected():
    class UnknownCommand(BaseModel):
        x: str = Field(json_schema_extra={"from": "burn_book.uid"})

    class UnknownField(BaseModel):
        x: str = Field(json_schema_extra={"from": "find_book.isbn"})

    class Malformed(BaseModel):
        x: str = Field(json_schema_extra={"from": "find_book"})

    with pytest.raises(FactSourceError, match="no command named 'burn_book'"):
        check_fact_sources(UnknownCommand, SIGNATURES)
    with pytest.raises(FactSourceError, match="no Output or Input field 'isbn'"):
        facts_from_runs(UnknownField, RUNS, signatures=SIGNATURES)
    with pytest.raises(FactSourceError, match="must be"):
        facts_from_runs(Malformed, RUNS)


def test_an_ambiguous_short_name_must_be_qualified():
    class Ambiguous(BaseModel):
        x: str = Field(json_schema_extra={"from": "find_book.uid"})

    with pytest.raises(AmbiguousFactSourceError, match="ambiguous source"):
        check_fact_sources(Ambiguous, {**SIGNATURES, "Shelf/find_book": FindBook})


# Two contexts, each with its own same-named command.
class CountItems:
    class Input(BaseModel):
        start: Optional[int] = None

    class Output(BaseModel):
        item_uids: list[str]
        total: int


TWO_CONTEXTS = {"Shelf/count_items": CountItems, "Cart/count_items": CountItems}
CART = "Cart k7"
TWO_CONTEXT_RUNS = [
    run("Shelf/count_items", output={"item_uids": ["s1"], "total": 1}),
    run("Cart/count_items", output={"item_uids": ["c1", "c2"], "total": 2}, context=CART),
]


def test_a_qualified_source_picks_its_context_command():
    class CartTotal(BaseModel):
        total: int = Field(json_schema_extra={"from": "Cart/count_items.total"})
        item_uids: list[str] = Field(json_schema_extra={"from": "Cart/count_items.item_uids"})

    facts = facts_from_runs(CartTotal, TWO_CONTEXT_RUNS, signatures=TWO_CONTEXTS)
    assert (facts.total, facts.item_uids) == (2, ["c1", "c2"])


def test_a_plain_source_resolves_against_the_record_context():
    class Total(BaseModel):
        total: int = Field(json_schema_extra={"from": "count_items.total"})

    # The record the runs were on decides, with or without signatures.
    assert facts_from_runs(Total, TWO_CONTEXT_RUNS, context=CART).total == 2
    assert facts_from_runs(Total, TWO_CONTEXT_RUNS, context=SHELF,
                           signatures=TWO_CONTEXTS).total == 1
    assert check_fact_sources(Total, TWO_CONTEXTS, context_name="Cart")[0].command == "Cart/count_items"


def test_a_plain_source_that_cannot_be_resolved_is_ambiguous():
    class Total(BaseModel):
        total: int = Field(json_schema_extra={"from": "count_items.total"})

    with pytest.raises(AmbiguousFactSourceError, match=r"ambiguous source.*Cart/count_items, Shelf/count_items"):
        check_fact_sources(Total, TWO_CONTEXTS)
    with pytest.raises(AmbiguousFactSourceError, match="ambiguous source"):
        facts_from_runs(Total, TWO_CONTEXT_RUNS)
    with pytest.raises(AmbiguousFactSourceError, match="none is in context 'Desk'"):
        facts_from_runs(Total, TWO_CONTEXT_RUNS, context_name="Desk")


# Pages and queries: a list field joins the pages of one query, keyed on the normalised Input.
class BrowseShelf:
    class Input(BaseModel):
        filter: Optional[str] = None
        start: Optional[int] = None
        order: str = "title"

    class Output(BaseModel):
        book_uids: list[str]
        total: int


BROWSE = {"Shelf/browse_shelf": BrowseShelf}


class Browsed(BaseModel):
    book_uids: list[str] = Field(json_schema_extra={"from": "browse_shelf.book_uids"})
    total: int = Field(json_schema_extra={"from": "browse_shelf.total"})


def browse(uids, total=3, **parameters):
    return run("Shelf/browse_shelf", parameters=parameters,
               output={"book_uids": uids, "total": total})


def test_none_default_and_not_given_sentinel_read_the_same_page():
    runs = [
        browse(["b1", "b2"], filter=None, start=None),
        browse(["b1", "b2"], filter="NOT_FOUND", start=0),       # the extractor's sentinel; row 0
        browse(["b1", "b2"], filter="None", order="title"),       # "None" text; the field default
        browse(["b1", "b2"], filter="", start=-sys.maxsize),      # empty text; the int marker
    ]
    facts = facts_from_runs(Browsed, runs, signatures=BROWSE)
    assert facts.book_uids == ["b1", "b2"]
    # Without signatures the defaults are unknown; the other forms still match.
    assert facts_from_runs(Browsed, runs[:2] + runs[3:]).book_uids == ["b1", "b2"]


def test_a_different_page_offset_is_joined_and_a_reread_page_replaces_its_read():
    runs = [browse(["b1", "b2"], start=0), browse(["b3"], start=2),
            browse(["b1", "b2"]), browse(["b3"], start=2, filter=None)]
    assert facts_from_runs(Browsed, runs, signatures=BROWSE).book_uids == ["b1", "b2", "b3"]


def test_a_filtered_reread_is_not_joined_into_the_full_list():
    full = [browse(["b1", "b2"]), browse(["b3"], start=2)]
    narrowed = browse(["b2"], total=1, filter="Dune")
    for runs in (full[:1] + [narrowed] + full[1:], full + [narrowed]):
        facts = facts_from_runs(Browsed, runs, signatures=BROWSE)
        assert facts.book_uids == ["b1", "b2", "b3"]
        assert facts.total == 3  # a scalar: the latest run of the broadest query
    # Only narrowed reads: the latest of the equally narrow queries is the list.
    other = browse(["b9"], total=1, filter="Emma")
    assert facts_from_runs(Browsed, [narrowed, other]).book_uids == ["b9"]


def test_a_scalar_comes_from_the_broadest_query_not_a_later_filtered_read():
    runs = [browse(["b1", "b2"], total=3), browse(["b3"], total=3, start=2),
            browse(["b2"], total=1, filter="Dune")]
    facts = facts_from_runs(Browsed, runs, signatures=BROWSE)
    assert facts == Browsed(book_uids=["b1", "b2", "b3"], total=3)
    # A later page of the broadest query is as good as its first page.
    runs = [browse(["b1", "b2"], total=3), browse(["b3"], total=4, start=2),
            browse(["b2"], total=1, filter="Dune")]
    assert facts_from_runs(Browsed, runs, signatures=BROWSE).total == 4


def test_a_scalar_takes_the_latest_run_when_every_run_is_the_same_query():
    runs = [browse(["b1"], total=1, filter="Dune"), browse(["b1"], total=2, filter="Dune"),
            browse(["b1"], total=5, filter="Dune", start=1)]
    assert facts_from_runs(Browsed, runs, signatures=BROWSE).total == 5


def test_a_field_can_name_its_own_paging_parameter():
    class ByCursor(BaseModel):
        book_uids: list[str] = Field(
            json_schema_extra={"from": "browse_shelf.book_uids", "page_by": "order"})

    runs = [browse(["b1"], order="a"), browse(["b2"], order="b")]
    assert facts_from_runs(ByCursor, runs).book_uids == ["b1", "b2"]
    assert facts_from_runs(Browsed, runs).book_uids == ["b2"]  # two queries; the latest wins


# A listing whose first page alone carries a whole-scope tally (later pages report none).
class OverdueShelf:
    class Input(BaseModel):
        start: Optional[int] = None

    class Output(BaseModel):
        book_uids: list[str]
        total: int
        overdue_uids: list[str] = []
        overdue_count: Optional[int] = None
        overdue_complete: bool = False


class Overdue(BaseModel):
    uids: list[str] = Field(json_schema_extra={"from": "overdue_shelf.overdue_uids"})
    count: Optional[int] = Field(json_schema_extra={"from": "overdue_shelf.overdue_count"})
    complete: bool = Field(json_schema_extra={"from": "overdue_shelf.overdue_complete"})
    total: int = Field(json_schema_extra={"from": "overdue_shelf.total"})


def overdue_page(uids, start=None, total=5, **tally):
    return run("Shelf/overdue_shelf", parameters={"start": start},
               output={"book_uids": uids, "total": total, **tally})


def test_paged_scalars_come_from_the_page_that_carries_them_not_the_last_page():
    first = overdue_page(["b1", "b2"], overdue_uids=["b2", "b4"], overdue_count=2,
                         overdue_complete=True)
    runs = [first, overdue_page(["b3", "b4"], start=2), overdue_page(["b5"], start=4)]
    facts = facts_from_runs(Overdue, runs, signatures={"Shelf/overdue_shelf": OverdueShelf})
    assert facts == Overdue(uids=["b2", "b4"], count=2, complete=True, total=5)
    # The order the pages were read in does not matter.
    assert facts_from_runs(Overdue, runs[1:] + runs[:1]) == facts


def test_an_empty_page_past_the_end_never_overwrites_an_earlier_page():
    runs = [browse(["b1"], total=1), browse([], total=0, start=1)]
    assert facts_from_runs(Browsed, runs, signatures=BROWSE) == Browsed(book_uids=["b1"], total=1)
    # An empty FIRST page is a real answer: nothing listed, total 0.
    assert facts_from_runs(Browsed, [browse([], total=0)]).total == 0
    # Only empty pages: the latest one still gives the value.
    assert facts_from_runs(Browsed, [browse([], total=0, start=3)]).total == 0


def test_a_paged_flag_is_true_when_any_page_says_so():
    runs = [browse(["b1"], total=2), browse(["b2"], total=2, start=1)]

    class Done(BaseModel):
        done: bool = Field(json_schema_extra={"from": "browse_shelf.done"})

    pages = [{**r, "output": {**r["output"], "done": done}} for r, done in zip(runs, (False, True))]
    assert facts_from_runs(Done, pages).done is True
    assert facts_from_runs(Done, pages[::-1]).done is True
    # One page read twice: the latest read stands, as for any re-read.
    reread = [{**pages[0], "output": {**pages[0]["output"], "done": True}}, pages[0]]
    assert facts_from_runs(Done, reread).done is False


def test_facts_fill_from_the_archive(store):
    note_command_run(SCOPE, "O0", run("Library/find_book", parameters=FindBook.Input(query="Dune"),
                                      output=FindBook.Output(uid="b1")))
    archive_step(execute_step(0, "find_book", "b1  Dune"), 0, scope=SCOPE, selected_archive=store)
    note_command_run(SCOPE, "O1", run("Shelf/list_books", parameters=ListBooks.Input(),
                                      output=ListBooks.Output(book_uids=["b1"], titles=["Dune"], total=1)))
    archive_step(execute_step(1, "list_books", "1 book"), 1, scope=SCOPE, selected_archive=store)

    facts = facts_from_archive(Found, store, SCOPE, signatures=SIGNATURES)
    assert facts == Found(book_uids=["b1"], total=1, query="Dune", uid="b1")
    assert fastworkflow.observation_offloading.facts_from_runs is facts_from_runs


def test_facts_text_renders_a_facts_model_as_compact_field_value_text():
    from fastworkflow.observation_offloading.facts import facts_text

    class Inner(BaseModel):
        code: str
        hits: int

    class ItemFacts(BaseModel):
        uid: str
        part_labels: list[str]
        total: int
        note: Optional[str] = None
        open: bool = True
        empty: list[str] = []
        worst: Optional[Inner] = None

    facts = ItemFacts(uid="i1", part_labels=["a  b", "c"], total=3,
                      worst=Inner(code="X1", hits=2))
    assert facts_text(facts) == (
        "uid: i1; part_labels: a b, c; total: 3; note: none; open: true; empty: none; "
        "worst: (code: X1; hits: 2)")
    assert facts_text(facts, fields=["total", "uid", "missing"]) == "total: 3; uid: i1"
    assert facts_text(facts, fields=["uid", "total"], sep="\n") == "uid: i1\ntotal: 3"
    # A fan-out phase's outputs: each in parentheses.
    assert facts_text([Inner(code="A", hits=1), Inner(code="B", hits=0)]) == \
        "(code: A; hits: 1), (code: B; hits: 0)"
    assert facts_text([]) == "none"
