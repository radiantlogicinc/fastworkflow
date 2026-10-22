"""Skill phases (fix-hia1.2): plan() and Phase in skill files, and the sub-plan runner.

Offline, no model: the executor is a scripted fake, the command runs it "archived"
are handed to the runner directly. The skill is a neutral library example.
"""
from __future__ import annotations

import re
import types
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock

import dspy
import pytest
from pydantic import BaseModel, Field

from fastworkflow import tracing
from fastworkflow.observation_offloading.archive import command_run_record
from fastworkflow.skill_commands import (
    SkillCallPlan,
    SkillFormatError,
    check_phase_sources,
    command_from_module,
    call_phases,
    phases_for,
)
from fastworkflow.skill_phases import (
    PHASE_MAX_STEPS_ENV,
    Phase,
    PhaseError,
    bound_arguments,
    concatenated_steps,
    fact_from,
    phase_query,
    phase_step_budget,
    fill_output,
    planned_record,
    planned_steps,
    runs_for_facts,
    runs_on_record,
    runs_that_entered_record,
)
from fastworkflow.observation_offloading import finish_check
from fastworkflow.observation_offloading.facts import MissingFactError
from fastworkflow.turn import TurnStatus
from fastworkflow.utils.react import fastWorkflowReAct
from fastworkflow.workflow_execution_context import WorkflowExecutionContext
from tests.test_turn_result_capture import (  # noqa: F401 - shared fixtures
    _make_agent_ctx,
    _set_agents,
    initialized_fastworkflow,
    todo_workflow_path,
)

LIBRARY = "Library"
BOOK = "Book b1 Dune"


class Found(BaseModel):
    book_uids: list[str] = fact_from("find_book.book_uids", min_length=1)


class BookFacts(BaseModel):
    uid: str = fact_from("open_book_by_uid.book_uid")
    loan_uids: list[str] = fact_from("list_loans.loan_uids")


class NoteFacts(BaseModel):
    total: int = fact_from("count_notes.total")


#: What each steps_fn was called with, by phase name.
SEEN: dict[str, dict] = {}


def find_steps(name):
    return ["open_library", f"find_book query={name}"]


def book_steps(find: Optional[Found] = None):
    SEEN["book"] = {"find": find}
    uid = find.book_uids[0] if find else "<the uid of the match>"
    return [f"open_book_by_uid book_uid={uid}", "list_loans (full list)"]


def note_steps(book: Optional[BookFacts] = None):
    SEEN["notes"] = {"book": book}
    return [f"count_notes on {book.uid if book else 'the book'}"]


def report_steps(book: Optional[BookFacts] = None, notes: Optional[NoteFacts] = None):
    SEEN["report"] = {"book": book, "notes": notes}
    if book is None:
        return ["say who has the book on loan"]
    line = f"say that {book.uid} has loans {', '.join(book.loan_uids) or 'none'}"
    return [line + (f" and {notes.total} notes" if notes else "")]


class LibrarySignature:
    """Show one named book and who has it on loan."""

    class Input(BaseModel):
        name: str = Field(description="the book's title")
        notes: bool = Field(default=False, description="the request asks about its notes")


def plan(input):
    return [
        Phase("find", lambda: find_steps(input.name), output=Found),
        Phase("book", lambda find: book_steps(find), output=BookFacts),
        Phase("notes", lambda book: note_steps(book), output=NoteFacts, when=input.notes),
        Phase("report", report_steps),
    ]


def _module(**attrs):
    module = types.ModuleType("library_skill")
    module.Signature = LibrarySignature
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _phased_command(**attrs):
    attrs.setdefault("plan", plan)
    return command_from_module("show_book", _module(**attrs), "_skills/show_book.py").command


def _flat_command():
    return command_from_module(
        "show_book", _module(steps=lambda input: find_steps(input.name)),
        "_skills/show_book.py").command


# ----------------------------------------------------------------------
# The skill file form and the loader
# ----------------------------------------------------------------------


def test_a_skill_with_plan_and_no_steps_gets_the_concatenation_as_its_steps():
    command = _phased_command()
    instance = LibrarySignature.Input(name="Dune", notes=True)
    assert command.steps_fn(instance) == [
        "open_library", "find_book query=Dune",
        "open_book_by_uid book_uid=<the uid of the match>", "list_loans (full list)",
        "count_notes on the book", "say who has the book on loan",
    ]
    assert command.steps_fn(instance) == concatenated_steps(plan(instance))
    assert command.phase_outputs == (Found, BookFacts, NoteFacts)


def test_a_skill_without_plan_loads_as_before():
    command = _flat_command()
    assert command.plan_fn is None and command.phase_outputs == ()


def test_when_false_leaves_the_phase_out():
    command = _phased_command()
    names = [p.name for p in phases_for(command, {"name": "Dune", "notes": False})]
    assert names == ["find", "book", "report"]
    names = [p.name for p in phases_for(command, {"name": "Dune", "notes": True})]
    assert names == ["find", "book", "notes", "report"]


def test_a_malformed_from_fails_the_load():
    class Bad(BaseModel):
        uid: str = fact_from("open_book_by_uid")

    with pytest.raises(SkillFormatError, match="must be"):
        _phased_command(plan=lambda input: [Phase("find", lambda: ["x"], output=Bad)])


def test_a_parameter_no_earlier_phase_provides_fails_the_load():
    with pytest.raises(SkillFormatError, match="'book'"):
        _phased_command(plan=lambda input: [Phase("report", lambda book: ["say it"])])
    # a fact parameter defaulting to None must name a phase with an output in some plan
    with pytest.raises(SkillFormatError, match="phase 'book': steps_fn takes 'fnd', which names no "
                                               "earlier phase with an output in any plan"):
        _phased_command(plan=lambda input: [Phase("find", lambda: ["find it"], output=Found),
                                            Phase("book", lambda fnd=None: book_steps(fnd))])
    with pytest.raises(SkillFormatError, match="phase 'report': when takes 'bok'"):
        _phased_command(plan=lambda input: [Phase("find", lambda: ["find it"], output=Found),
                                            Phase("report", lambda: ["say it"], when=lambda bok=None: True)])
    # one plan of the input grid binding it is enough (notes runs only when input.notes);
    # a parameter with another default keeps a value, it is not a fact
    assert _phased_command(plan=lambda input: plan(input) + [
        Phase("tail", lambda notes=None, kind="book": [f"say the {kind} notes"])]).plan_fn


def test_a_plan_that_does_not_return_phases_fails_the_load():
    with pytest.raises(SkillFormatError, match="list of Phase"):
        _phased_command(plan=lambda input: ["find_book"])
    with pytest.raises(SkillFormatError, match="two phases"):
        _phased_command(plan=lambda input: [Phase("a", lambda: ["x"]), Phase("a", lambda: ["y"])])


def test_a_plan_needs_a_skill_with_params():
    class NoParams:
        """Show the shelves."""

    module = types.ModuleType("shelves")
    module.Signature = NoParams
    module.plan = lambda input: [Phase("a", lambda: ["list_shelves"])]
    with pytest.raises(SkillFormatError, match="needs a skill with params"):
        command_from_module("shelves", module, "_skills/shelves.py")


class FindBook:
    class Input(BaseModel):
        query: str

    class Output(BaseModel):
        book_uids: list[str]


class OpenBook:
    class Input(BaseModel):
        book_uid: str


class ListLoans:
    class Output(BaseModel):
        loan_uids: list[str]


def test_a_missing_source_field_is_a_load_warning_not_a_failure():
    command = _phased_command()
    signatures = {"Library/find_book": FindBook, "Library/open_book_by_uid": OpenBook,
                  "Book/list_loans": ListLoans}
    warnings = check_phase_sources(command, signatures)
    # count_notes does not exist yet: one warning, for NoteFacts only.
    assert len(warnings) == 1 and "count_notes" in warnings[0]


def test_a_plain_name_several_contexts_define_passes_when_one_context_resolves_it():
    command = _phased_command()
    signatures = {"Library/find_book": FindBook, "Library/open_book_by_uid": OpenBook,
                  "Book/list_loans": ListLoans, "Shelf/list_loans": FindBook,
                  "Book/count_notes": SimpleNamespace(Output=NoteFacts)}
    assert check_phase_sources(command, signatures) == []


def test_parameters_bind_by_name_and_keep_their_defaults():
    found = Found(book_uids=["b1"])
    phase = Phase("report", report_steps)
    assert bound_arguments(phase, {"find": found}) == {}
    assert bound_arguments(Phase("book", lambda find: []), {"find": found}) == {"find": found}
    with pytest.raises(PhaseError, match="needs 'find'"):
        bound_arguments(Phase("book", lambda find: []), {})


def test_the_step_budget_is_configurable_and_defaults_to_15(monkeypatch):
    monkeypatch.delenv(PHASE_MAX_STEPS_ENV, raising=False)
    assert phase_step_budget() == 15
    monkeypatch.setenv(PHASE_MAX_STEPS_ENV, "4")
    assert phase_step_budget() == 4
    monkeypatch.setenv(PHASE_MAX_STEPS_ENV, "none")
    assert phase_step_budget() == 15


def test_runs_on_the_record_include_the_run_that_entered_it():
    runs = [
        {"command": "find_book", "context_clause": LIBRARY},
        {"command": "open_book_by_uid", "context_clause": LIBRARY},
        {"command": "list_loans", "context_clause": BOOK},
    ]
    assert [r["command"] for r in runs_on_record(runs, BOOK)] == ["open_book_by_uid", "list_loans"]
    assert all(r["context_clause"] == BOOK for r in runs_on_record(runs, BOOK))
    # A phase that ends right after opening the record keeps the opening run.
    assert [r["command"] for r in runs_on_record(runs[:2], BOOK)] == ["open_book_by_uid"]


OTHER_BOOK = "Book b2 Emma"


def test_a_command_that_ran_only_on_bare_contexts_counts_wherever_the_phase_ended():
    # A search from the hub, then the run wandered onto a record and ended there.
    runs = [
        {"command": "open_library", "context_clause": ""},
        {"command": "find_book", "context_clause": LIBRARY},
        {"command": "open_book_by_uid", "context_clause": LIBRARY},
        {"command": "list_loans", "context_clause": BOOK},
    ]
    assert [r["command"] for r in runs_on_record(runs, BOOK)] == ["open_book_by_uid", "list_loans"]
    kept = runs_for_facts(runs, BOOK)
    assert [r["command"] for r in kept] == ["open_library", "find_book", "open_book_by_uid", "list_loans"]
    assert {r["context_clause"] for r in kept} == {BOOK}


def test_a_command_that_touched_a_record_counts_only_on_the_phase_record():
    # Book b1 was opened and read, then book b2 was opened: b1's runs are not b2's.
    runs = [
        {"command": "open_book_by_uid", "context_clause": LIBRARY},
        {"command": "list_loans", "context_clause": BOOK},
        {"command": "go_up", "context_clause": BOOK},
        {"command": "open_book_by_uid", "context_clause": LIBRARY},
    ]
    kept = runs_for_facts(runs, OTHER_BOOK)
    assert [r["command"] for r in kept] == ["open_book_by_uid"]
    assert kept[-1] is not runs[-1] and kept[-1]["context_clause"] == OTHER_BOOK


def _cmd_run(command, input, output=None, context=LIBRARY, success=True):
    return {"command": command, "success": success, "context_clause": context,
            "context_name": context.split(" ")[0] or None, "input": input, "output": output}


FIND_PLAN = "1. open_library\n2. find_book query=Dune"


class Opened(BaseModel):
    uid: str = fact_from("open_book_by_uid.book_uid")


def test_an_off_plan_lookup_does_not_replace_the_planned_fact():
    runs = [_cmd_run("open_library", None, context=""),
            _cmd_run("Library/find_book", {"query": "Dune"}, {"book_uids": ["b1"]}),
            _cmd_run("Library/find_book", {"query": "Emma"}, {"book_uids": ["b2"]})]
    # Without the plan the later, equally broad lookup wins; with it, the planned one does.
    assert fill_output(Found, runs, record=LIBRARY).book_uids == ["b2"]
    assert fill_output(Found, runs, record=LIBRARY, plan_text=FIND_PLAN).book_uids == ["b1"]
    # The value compares as the facts read it: case and spacing do not matter.
    runs[1]["input"] = {"query": "  dune ", "start": None}
    assert fill_output(Found, runs, record=LIBRARY, plan_text=FIND_PLAN).book_uids == ["b1"]


def test_only_an_off_plan_lookup_fails_naming_the_planned_step():
    runs = [_cmd_run("open_library", None, context=""),
            _cmd_run("Library/find_book", {"query": "Emma"}, {"book_uids": ["b2"]})]
    with pytest.raises(MissingFactError, match=r"Found\.book_uids.*the planned run was not found.*"
                                               r"step 2 \(find_book query=Dune\)"):
        fill_output(Found, runs, record=LIBRARY, plan_text=FIND_PLAN)


def test_pages_of_the_planned_query_still_join():
    runs = [_cmd_run("find_book", {"query": "Dune", "start": None}, {"book_uids": ["b1", "b3"]}),
            _cmd_run("find_book", {"query": "Emma"}, {"book_uids": ["b2"]}),
            _cmd_run("find_book", {"query": "Dune", "start": 2}, {"book_uids": ["b4"]})]
    filled = fill_output(Found, runs, record=LIBRARY, plan_text=FIND_PLAN)
    assert filled.book_uids == ["b1", "b3", "b4"]


def test_a_step_with_a_prose_or_placeholder_argument_keeps_the_looser_match():
    runs = [_cmd_run("open_book_by_uid", {"book_uid": "b1"}),
            _cmd_run("open_book_by_uid", {"book_uid": "b2"})]
    for plan in ("1. open_book_by_uid with the uid of the Dune match",
                 "1. open_book_by_uid book_uid=<the uid of the match>"):
        assert fill_output(Opened, runs, record=LIBRARY, plan_text=plan).uid == "b2"
    # A concrete uid takes only its own run.
    assert fill_output(Opened, runs, record=LIBRARY,
                       plan_text="1. open_book_by_uid book_uid=b1").uid == "b1"


def test_a_command_no_step_names_keeps_the_earlier_rules():
    runs = [_cmd_run("find_book", {"query": "Emma"}, {"book_uids": ["b2"]})]
    assert fill_output(Found, runs, record=LIBRARY, plan_text="1. say the uid").book_uids == ["b2"]


class FoundNamed(BaseModel):
    book_uids: list[str] = fact_from("find_book.book_uids", min_length=1)
    labels: list[str] = fact_from("find_book.labels")


FIND_TITLE_PLAN = "1. open_library\n2. find_book query=Dune Messiah"


@pytest.mark.parametrize("query", ["Dune Mesiah", "Dune", None])
def test_a_lookup_in_another_form_whose_output_names_the_planned_value_counts(query):
    # Misspelt (fuzzy-corrected by the command), cut short, or not given at all: the
    # run's output still lists the planned title, so it fills the fact.
    runs = [_cmd_run("open_library", None, context=""),
            _cmd_run("Library/find_book", {"query": query},
                     {"book_uids": ["b1", "b5"], "labels": ["Dune", "Dune Messiah"]})]
    filled = fill_output(FoundNamed, runs, record=LIBRARY, plan_text=FIND_TITLE_PLAN)
    # Narrowed to the row that holds the planned title: never another row's uid.
    assert filled.book_uids == ["b5"] and filled.labels == ["Dune Messiah"]


def test_the_planned_row_is_picked_out_of_many():
    # A bare lookup listed the whole catalogue: the fact is the planned row, not the first.
    labels = [f"Title {k}" for k in range(14)] + ["Dune Messiah"]
    runs = [_cmd_run("find_book", {"query": None},
                     {"book_uids": [f"b{k}" for k in range(15)], "labels": labels, "total": 15})]
    filled = fill_output(FoundNamed, runs, record=LIBRARY, plan_text=FIND_TITLE_PLAN)
    assert (filled.book_uids, filled.labels) == (["b14"], ["Dune Messiah"])
    # Two rows hold the value: no single row, so the run is refused as before.
    runs[0]["output"]["labels"][0] = "Dune Messiah"
    with pytest.raises(MissingFactError, match="the planned run was not found"):
        fill_output(FoundNamed, runs, record=LIBRARY, plan_text=FIND_TITLE_PLAN)


def test_narrowed_output_reads_equal_length_lists_as_rows():
    from fastworkflow.skill_phases import narrowed_output

    output = {"uids": ["u1", "u2", "u3"], "labels": ["Ann", "Bo", "Cy"], "codes": ["x", "y"],
              "scope": "branch", "total": 3}
    assert narrowed_output(output, ["Bo"]) == {"uids": ["u2"], "labels": ["Bo"],
                                               "codes": ["x", "y"], "scope": "branch", "total": 3}
    # Two values must sit in the same row; a value outside the lists may be a scalar.
    assert narrowed_output(output, ["Bo", "u2"])["uids"] == ["u2"]
    assert narrowed_output(output, ["Bo", "u3"]) is None
    assert narrowed_output(output, ["branch", "Cy"])["uids"] == ["u3"]
    assert narrowed_output(output, ["branch"]) == output
    assert narrowed_output(output, ["Dee"]) is None


def test_a_lookup_in_another_form_whose_output_lacks_the_planned_value_still_fails():
    # "Dune Messiah" is only part of a label ("Dune Messiah 2"): only a whole value counts.
    for labels in (["Emma"], ["Dune Messiah 2"], []):
        runs = [_cmd_run("find_book", {"query": "Dun"},
                         {"book_uids": ["b2"] * len(labels), "labels": labels})]
        with pytest.raises(MissingFactError, match="the planned run was not found"):
            fill_output(FoundNamed, runs, record=LIBRARY, plan_text=FIND_TITLE_PLAN)


def test_the_planned_run_still_wins_over_a_lookup_in_another_form():
    runs = [_cmd_run("find_book", {"query": "Dune Messiah"},
                     {"book_uids": ["b5"], "labels": ["Dune Messiah"]}),
            _cmd_run("find_book", {"query": None},
                     {"book_uids": ["b1", "b5", "b9"], "labels": ["Dune", "Dune Messiah", "Emma"]})]
    filled = fill_output(FoundNamed, runs, record=LIBRARY, plan_text=FIND_TITLE_PLAN)
    assert filled.book_uids == ["b5"]


def test_a_listing_with_another_scope_never_counts_for_the_planned_scope():
    # The wrong-record case: the planned scope word is not in the output, so an empty
    # listing of another scope never fills the planned listing's facts.
    class Loans(BaseModel):
        loan_uids: list[str] = fact_from("list_loans.loan_uids")

    plan = "1. list_loans scope=branch scope_uid=x1 (no filter)"
    runs = [_cmd_run("list_loans", {"scope": "shelf", "scope_uid": "x1"},
                     {"loan_uids": [], "scope": "shelf", "scope_uid": "x1"})]
    with pytest.raises(MissingFactError, match="the planned run was not found"):
        fill_output(Loans, runs, record=LIBRARY, plan_text=plan)
    runs[0]["output"]["scope"] = "branch"   # its output now names both values
    assert fill_output(Loans, runs, record=LIBRARY, plan_text=plan).loan_uids == []


def test_an_opener_with_another_value_never_counts_even_when_its_output_names_the_planned_one():
    # open b2 moved the workflow onto b2; its output mentions b1, but an opener of another
    # record never stands in for the planned opener.
    class OpenedWithLink(BaseModel):
        uid: str = fact_from("open_book_by_uid.book_uid")

    runs = [_cmd_run("open_book_by_uid", {"book_uid": "b2"}, {"book_uid": "b2", "sequel_of": "b1"}),
            _cmd_run("list_loans", {}, {"loan_uids": []}, context=OTHER_BOOK)]
    with pytest.raises(MissingFactError, match="the planned run was not found"):
        fill_output(OpenedWithLink, runs, record=OTHER_BOOK,
                    plan_text="1. open_book_by_uid book_uid=b1")


class OpenedBook(BaseModel):
    uid: str = fact_from("open_book_by_uid.book_uid")
    title: str = fact_from("open_book_by_uid.title")
    loan_uids: list[str] = fact_from("list_loans.loan_uids")


BOOK_PLAN = "1. open_book_by_uid book_uid=b1\n2. list_loans (full list)"


def _opened(uid, title, context=LIBRARY):
    return _cmd_run("Library/open_book_by_uid", {"book_uid": uid},
                    {"book_uid": uid, "title": title}, context=context)


def test_a_phase_already_on_its_planned_record_fills_from_the_run_that_opened_it():
    # An earlier phase opened b1; this phase's step 1 (open b1) was refused unrun
    # ("already in context"), so the phase's own runs are only on the record.
    earlier = [_cmd_run("Library/find_book", {"query": "Dune"}, {"book_uids": ["b1"]}),
               _opened("b1", "Dune")]
    runs = [_cmd_run("Book/list_loans", {"start": None}, {"loan_uids": ["l1"]}, context=BOOK)]
    with pytest.raises(MissingFactError, match="the planned run was not found"):
        fill_output(OpenedBook, runs, record=BOOK, plan_text=BOOK_PLAN)
    filled = fill_output(OpenedBook, runs, record=BOOK, plan_text=BOOK_PLAN, earlier_runs=earlier)
    # Input and Output facts both come from the run that really opened the record.
    assert (filled.uid, filled.title, filled.loan_uids) == ("b1", "Dune", ["l1"])
    # A placeholder step takes the run that entered the record the same way.
    placeholder = "1. open_book_by_uid book_uid=<the uid>\n2. list_loans (full list)"
    assert fill_output(OpenedBook, runs, record=BOOK, plan_text=placeholder,
                       earlier_runs=earlier).uid == "b1"


def test_a_phase_on_a_different_record_than_its_step_opens_still_fails():
    # b1 was opened earlier, then b2; the phase is on b2 but its step opens b1.
    earlier = [_opened("b1", "Dune"),
               _cmd_run("Book/list_loans", {"start": None}, {"loan_uids": ["l1"]}, context=BOOK),
               _cmd_run("go_up", None, context=BOOK),
               _opened("b2", "Emma")]
    runs = [_cmd_run("Book/list_loans", {"start": None}, {"loan_uids": ["l9"]}, context=OTHER_BOOK)]
    with pytest.raises(MissingFactError, match=r"OpenedBook\.uid.*the planned run was not found"):
        fill_output(OpenedBook, runs, record=OTHER_BOOK, plan_text=BOOK_PLAN, earlier_runs=earlier)
    # Even when the archive misses the run that opened b2, so that opening b1 looks
    # like the run that entered b2: b1 is not a word of b2's clause.
    with pytest.raises(MissingFactError, match="the planned run was not found"):
        fill_output(OpenedBook, runs, record=OTHER_BOOK, plan_text=BOOK_PLAN,
                    earlier_runs=earlier[:1])
    # At the root or a bare context there is no record to be already on.
    with pytest.raises(MissingFactError):
        fill_output(OpenedBook, runs, record=LIBRARY, plan_text=BOOK_PLAN, earlier_runs=earlier)


def test_only_the_run_that_entered_the_record_stands_in_never_an_earlier_reading():
    # list_loans ran on b1 in an earlier phase; this phase skipped it: not carried over.
    earlier = [_opened("b1", "Dune"),
               _cmd_run("Book/list_loans", {"start": None}, {"loan_uids": ["l1"]}, context=BOOK)]
    with pytest.raises(MissingFactError, match=r"OpenedBook\.loan_uids"):
        fill_output(OpenedBook, [], record=BOOK, plan_text=BOOK_PLAN, earlier_runs=earlier)
    assert runs_that_entered_record(earlier, [], BOOK, planned_steps(
        BOOK_PLAN, ["open_book_by_uid", "list_loans"]), ["open_book_by_uid", "list_loans"]) == [
        {**earlier[0], "context_clause": BOOK}]


def test_a_normal_opening_in_the_phase_is_unchanged():
    earlier = [_opened("b1", "Dune (old title)")]
    runs = [_cmd_run("go_up", None, context=BOOK),
            _opened("b1", "Dune"),
            _cmd_run("Book/list_loans", {"start": None}, {"loan_uids": ["l1"]}, context=BOOK)]
    without = fill_output(OpenedBook, runs, record=BOOK, plan_text=BOOK_PLAN)
    with_earlier = fill_output(OpenedBook, runs, record=BOOK, plan_text=BOOK_PLAN,
                               earlier_runs=earlier)
    assert with_earlier == without
    assert (without.uid, without.title) == ("b1", "Dune")


def test_the_finish_check_counts_an_already_in_context_reply_as_its_opening_step():
    rows = [{"command": "open_book_by_uid book_uid=b1", "step_index": 5,
             "text": "You are already in context 'Book' (b1 Dune): 'open_book_by_uid' is how "
                     "you enter it."},
            {"command": "list_loans", "step_index": 6, "text": "1 loan(s)"}]
    result = finish_check.check(BOOK_PLAN, rows, ["open_book_by_uid", "list_loans"], values=True)
    assert result.unmet == []


def test_a_phase_other_than_the_last_sees_only_its_steps():
    query = phase_query("show the book Dune and who has it", "1. open_library\n2. find_book query=Dune",
                        index=0, count=3, name="find")
    assert query.startswith("This is part 1 of 3 (find) of a larger job. Do only these steps, then finish.")
    assert "Later parts will do the rest." in query
    assert query.endswith("1. open_library\n2. find_book query=Dune")
    assert "who has it" not in query


def test_the_last_phase_sees_the_request():
    query = phase_query("show the book Dune and who has it", "1. say that b1 has loans l1",
                        index=2, count=3, name="report")
    assert query.startswith("show the book Dune and who has it\n\n")
    assert "part 3 of 3 (report)" in query and "answer the request" in query
    assert query.endswith("1. say that b1 has loans l1")


def test_the_last_phase_code_ran_wholly_is_told_to_answer_from_the_outputs():
    # fix-hia1.34: nothing remains, so never "do these steps"; the copy rule applies.
    from fastworkflow import skill_phases
    from fastworkflow.skill_phases import clean_answer

    query = phase_query("show the book Dune", "1. find_book query=Dune\n2. list_loans",
                        index=0, count=1, name="show", code_steps=True, code_ran_all=True)
    assert query.startswith("show the book Dune\n\nThis is part 1 of 1 (show), the last part")
    assert skill_phases._CODE_STEPS_ALL_DONE in query and skill_phases._COPY_RULE in query
    assert "Do these steps" not in query and skill_phases._CODE_STEPS_DONE not in query
    assert query.endswith("Answer the request:\n1. find_book query=Dune\n2. list_loans")
    assert clean_answer("Dune is on loan. " + query.split("\n\n", 1)[1].split("\n1.")[0]) == (
        "Dune is on loan.")
    # A phase other than the last keeps its own framing.
    middle = phase_query("q", "1. find_book query=Dune", index=0, count=2, name="find",
                         code_steps=True, code_ran_all=True)
    assert middle == phase_query("q", "1. find_book query=Dune", index=0, count=2,
                                 name="find", code_steps=True)


# ----------------------------------------------------------------------
# The runner
# ----------------------------------------------------------------------


class RecordingSink:
    def __init__(self):
        self.spans: list[tracing.Span] = []

    def emit_span(self, span):
        self.spans.append(span)

    def emit_turn_record(self, record):
        return True

    def record_conversation_label(self, *args):
        pass

    def named(self, name):
        return [span for span in self.spans if span.name == name]


def _run(command, output=None, parameters=None, success=True, context=LIBRARY):
    return dict(command=command, output=output, parameters=parameters, success=success,
                context=context)


#: What the fake executor does in each phase: the runs it makes, the record it ends
#: on, its answer, how many steps it takes and whether it reaches its budget.
SCRIPT = {
    "find": dict(runs=[_run("open_library", context=""),
                       _run("find_book", {"book_uids": ["b1"]}, {"query": "Dune"})],
                 record=LIBRARY, answer="found b1"),
    "book": dict(runs=[_run("open_book_by_uid", None, {"book_uid": "b1"}),
                       _run("list_loans", {"loan_uids": ["l1", "l2"]}, context=BOOK)],
                 record=BOOK, answer="b1 has two loans"),
    "notes": dict(runs=[_run("count_notes", {"total": 3}, context=BOOK)],
                  record=BOOK, answer="3 notes"),
    "report": dict(runs=[], record=BOOK, answer="Dune (b1) is on loan: l1, l2."),
}


class FakeExecutor:
    """Scripted executor: each call takes its phase's steps from first_step and answers."""

    def __init__(self, ctx, script, suspend_in=None):
        self.ctx, self.script, self.suspend_in = ctx, script, suspend_in
        self.calls: list[dict] = []
        self.plans: list[str] = []
        self.runs: list[dict] = []
        self.record = ""
        self.trajectory: dict = {}
        self.finish_reviewer = None
        self.notes: list = []
        self.disabled_tools = frozenset()
        self.iteration_counter = 0
        self.max_iters = 25
        self._pending = None

    def _phase(self, query):
        # One item of a fan-out phase: "<phase>[<item>]", framed "(<phase>) for item k of n (<item>)".
        for key in self.script:
            if key.endswith("]") and "[" in key:
                name, item = key[:-1].split("[", 1)
                if re.search(rf"\({re.escape(name)}\) for item \d+ of \d+ \({re.escape(item)}\)",
                             query):
                    return key
        return next((name for name in self.script if f"({name})" in query), "find")

    def _steps(self, name, start, max_iters):
        entry = self.script[name]
        count = entry.get("steps", max(1, len(entry["runs"])))
        taken = min(count, max_iters)
        self.trajectory = {}
        for i in range(taken):
            self.trajectory[f"thought_{start + i}"] = "t"
            self.trajectory[f"tool_name_{start + i}"] = "execute_workflow_query"
            self.trajectory[f"observation_{start + i}"] = "o"
            if i < len(entry["runs"]):
                r = entry["runs"][i]
                self.runs.append({**command_run_record(
                    command_name=r["command"], success=r["success"], context_clause=r["context"],
                    context_name=r["context"].split(" ")[0], parameters=r["parameters"],
                    output=r["output"]), "step_index": start + i})
        self.record = entry["record"]
        if self.finish_reviewer is not None:
            self.notes.append(self.finish_reviewer(self.trajectory, max_iters))
        return SimpleNamespace(final_answer=entry["answer"], suspended=False,
                               exhausted=entry.get("exhausted", taken < count))

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        self.plans.append(self.ctx._initial_plan_text)
        name = self._phase(kwargs["user_query"])
        if name == self.suspend_in:
            self.suspend_in = None
            self._pending = (name, kwargs.get("first_step", 0), kwargs.get("max_iters", 25))
            return SimpleNamespace(suspended=True, clarification="Which edition?")
        return self._steps(name, kwargs.get("first_step", 0), kwargs.get("max_iters", 25))

    def resume(self, observation):
        name, start, max_iters = self._pending
        return self._steps(name, start, max_iters)

    def export_suspended(self):
        return {} if self._pending else None

    def clear_suspension(self):
        pass


@pytest.fixture
def phased(initialized_fastworkflow, todo_workflow_path, monkeypatch):
    """An agent context whose sub-task is a call of the phased skill ``show_book``."""
    SEEN.clear()
    monkeypatch.delenv(PHASE_MAX_STEPS_ENV, raising=False)
    ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch)
    sink = RecordingSink()
    ctx.set_trace_sink(sink)
    command = _phased_command()
    state = SimpleNamespace(args={"name": "Dune", "notes": False}, command=command,
                            planned=[], script=dict(SCRIPT), suspend_in=None, steps_after=[],
                            steps_before=[])

    def fake_plan(user_query, session, with_agent_inputs_and_trajectory=False,
                  planner_lm=None, skill_command=None, **kwargs):
        state.planned.append(skill_command)
        if skill_command is not None:
            # As workflow_agent: every skill call runs phased; a flat one as one phase.
            call = SkillCallPlan(plan="", args=dict(state.args), notes=[], prediction=None,
                                 phases=call_phases(skill_command, state.args),
                                 steps_after=list(state.steps_after),
                                 steps_before=list(state.steps_before))
            session._phased_skill_call = (skill_command, call)
        session._initial_plan_text = "1. find_book query=Dune"
        return user_query

    monkeypatch.setattr("fastworkflow.workflow_agent.build_query_with_next_steps", fake_plan)
    monkeypatch.setattr("fastworkflow.workflow_agent.planner_skill_for",
                        lambda text, folder: ("", state.command))
    monkeypatch.setattr("fastworkflow.workflow_agent.split_into_subtasks",
                        lambda *args, **kwargs: ["show the book Dune"])

    def no_hand_off(k):
        raise AssertionError("a phased skill must not be handed off")

    monkeypatch.setattr(ctx, "_hand_off", no_hand_off)
    reviews = []
    monkeypatch.setattr(
        "fastworkflow.observation_offloading.finish_check.review_finish",
        lambda **kwargs: reviews.append(kwargs) or None)

    def start(suspend_in=None):
        SEEN.clear()  # loading the skill ran every steps_fn on placeholders
        executor = FakeExecutor(ctx, state.script, suspend_in=suspend_in)
        _set_agents(ctx, executor)
        monkeypatch.setattr(
            ctx, "_archived_command_runs",
            lambda from_step=None, to_step=None: [
                run for run in executor.runs
                if (from_step is None or run["step_index"] >= from_step)
                and (to_step is None or run["step_index"] <= to_step)])
        monkeypatch.setattr(ctx, "_current_record",
                            lambda: (executor.record, executor.record.split(" ")[0]))
        return executor

    # An answer-only phase (steps that name no command) is one model call, not an
    # executor run: scripted here like the executor, by the phase named in its input.
    answered: list[str] = []

    def fake_answer(query):
        answered.append(query)
        name = next((n for n in state.script if f"({n})" in query), "report")
        return state.script[name]["answer"]

    monkeypatch.setattr("fastworkflow.skill_phases.write_answer", fake_answer)
    # These tests script the model's answer; the answer written by code is tested on its own
    # (test_code_writes_an_answer_only_phase_and_its_span_says_so and around it).
    monkeypatch.setattr("fastworkflow.skill_phases.answer_by_code", lambda plan_text: False)
    state.ctx, state.sink, state.reviews, state.start = ctx, sink, reviews, start
    state.answered = answered
    return state


def test_three_phases_run_as_three_sub_plans_in_order(phased):
    executor = phased.start()
    result = phased.ctx.process_turn("show the book Dune")

    assert result.status == TurnStatus.COMPLETED
    queries = [call["user_query"] for call in executor.calls]
    # Two executor runs; the report phase names no command, so it is answer-only.
    assert len(queries) == 2
    assert "part 1 of 3 (find)" in queries[0] and "find_book query=Dune" in queries[0]
    assert "part 2 of 3 (book)" in queries[1] and "open_book_by_uid book_uid=b1" in queries[1]
    assert len(phased.answered) == 1
    assert "part 3 of 3 (report)" in phased.answered[0]
    assert "say that b1 has loans l1, l2" in phased.answered[0]
    # Each phase's steps come from its steps_fn, not the planner: one plan call for the sub-task.
    assert len(phased.planned) == 1
    # Step numbers continue across the phases: find took 2 steps.
    assert [call["first_step"] for call in executor.calls] == [0, 2]
    # The last phase writes the answer.
    assert result.answer == "Dune (b1) is on loan: l1, l2."


def test_each_phase_gets_the_earlier_phases_outputs_as_typed_objects(phased):
    phased.args["notes"] = True
    phased.start()
    phased.ctx.process_turn("show the book Dune and its notes")

    assert isinstance(SEEN["book"]["find"], Found)
    assert SEEN["book"]["find"].book_uids == ["b1"]
    assert isinstance(SEEN["notes"]["book"], BookFacts)
    assert SEEN["notes"]["book"].uid == "b1"
    report = SEEN["report"]
    assert report["book"].loan_uids == ["l1", "l2"] and report["notes"].total == 3


def test_a_skipped_phase_does_not_run(phased):
    executor = phased.start()
    phased.ctx.process_turn("show the book Dune")

    assert not any("(notes)" in call["user_query"] for call in executor.calls)
    assert "notes" not in SEEN
    assert SEEN["report"]["notes"] is None


def test_a_missing_fact_fails_the_phase_cleanly_and_stops_the_call(phased):
    phased.script["book"] = dict(SCRIPT["book"], runs=SCRIPT["book"]["runs"][:1])  # no list_loans
    executor = phased.start()
    result = phased.ctx.process_turn("show the book Dune")

    assert result.status == TurnStatus.COMPLETED
    assert len(executor.calls) == 2  # report never ran
    # Only facts filled in code are shown: the find phase's, never a run's own text,
    # so the failed book phase's answer ("b1 has two loans") does not reach the user.
    assert result.answer.startswith(f"{FOUND_FACTS_LEAD}\n\nfind:\n- book uids: b1\n\n")
    assert "found b1" not in result.answer and "b1 has two loans" not in result.answer
    assert "stopped at its part 'book'" in result.answer
    assert "BookFacts.loan_uids" in result.answer and "list_loans" in result.answer
    assert "Not run: report." in result.answer
    span = phased.sink.named(tracing.SPAN_SKILL_PHASE)[-1]
    assert span.status == tracing.STATUS_ERROR
    assert "BookFacts.loan_uids" in span.attributes["facts_error"]
    assert phased.ctx._phase_run is None


def test_the_facts_come_only_from_the_phase_own_runs(phased):
    # count_notes ran on the same record, but in the book phase: the notes phase
    # itself ran nothing, so its fact is missing.
    phased.args["notes"] = True
    phased.script["book"] = dict(SCRIPT["book"], runs=SCRIPT["book"]["runs"] + SCRIPT["notes"]["runs"])
    phased.script["notes"] = dict(SCRIPT["notes"], runs=[])
    phased.start()
    result = phased.ctx.process_turn("show the book Dune and its notes")
    assert "stopped at its part 'notes'" in result.answer
    assert "NoteFacts.total" in result.answer


def _knows_the_toy_commands(phased, monkeypatch):
    """The toy commands are known by name (as a workflow's routing names its commands), so a
    flat skill's steps are command steps, not answer-only ones."""
    monkeypatch.setattr(phased.ctx, "_phase_command_names", lambda: {
        "commands": ["open_library", "find_book", "open_book_by_uid", "list_loans",
                     "count_notes"], "core_commands": []})


def test_a_skill_without_plan_runs_as_one_phase_with_the_executor_step_cap(phased, monkeypatch):
    monkeypatch.setenv(PHASE_MAX_STEPS_ENV, "3")
    _knows_the_toy_commands(phased, monkeypatch)
    phased.command = _flat_command()
    executor = phased.start()
    result = phased.ctx.process_turn("show the book Dune")

    assert len(executor.calls) == 1
    # Its budget is the executor's own max_iters, not FW_SKILL_PHASE_MAX_STEPS.
    assert executor.calls[0]["max_iters"] == 25
    (span,) = phased.sink.named(tracing.SPAN_SKILL_PHASE)
    assert span.attributes["phase"] == "show_book" and span.attributes["phase_count"] == 1
    assert span.attributes["step_budget"] == 25
    assert result.answer == "found b1"


def test_a_phase_that_reaches_its_budget_is_not_handed_off(phased, monkeypatch):
    monkeypatch.setenv(PHASE_MAX_STEPS_ENV, "3")
    phased.script["book"] = dict(SCRIPT["book"], steps=9)
    executor = phased.start()
    result = phased.ctx.process_turn("show the book Dune")

    # _hand_off raises in this fixture; the turn completes.
    assert result.status == TurnStatus.COMPLETED
    assert [call["max_iters"] for call in executor.calls] == [3, 3]
    book = phased.sink.named(tracing.SPAN_SKILL_PHASE)[1]
    assert book.attributes["exhausted"] is True and book.attributes["steps_used"] == 3
    assert book.attributes["step_budget"] == 3


def test_the_per_phase_budget_defaults_to_15(phased):
    executor = phased.start()
    phased.ctx.process_turn("show the book Dune")
    assert {call["max_iters"] for call in executor.calls} == {15}


def test_the_react_loop_stops_at_the_budget_it_is_given(monkeypatch):
    def a_tool(command: str) -> str:
        """Runs a command."""
        return f"ran {command}"

    agent = fastWorkflowReAct("user_query -> answer", tools=[a_tool], max_iters=25)
    monkeypatch.setattr(
        agent, "_call_with_potential_trajectory_truncation",
        lambda *args, **kwargs: dspy.Prediction(
            next_thought="more", next_tool_name="a_tool", next_tool_args={"command": "x"}),
    )
    monkeypatch.setattr(agent, "_extract_prediction", lambda trajectory, **kwargs: {"answer": "ok"})

    prediction = agent.forward(user_query="hi", available_commands="c", first_step=2, max_iters=4)

    assert prediction.exhausted is True
    assert sorted(int(k.rsplit("_", 1)[1]) for k in prediction.trajectory if k.startswith("thought_")) \
        == [2, 3, 4, 5]


def test_the_finish_check_runs_per_phase_against_that_phase_steps(phased):
    executor = phased.start()
    phased.ctx.process_turn("show the book Dune")

    assert executor.plans[0] == "1. open_library\n2. find_book query=Dune"
    assert executor.plans[1] == "1. open_book_by_uid book_uid=b1\n2. list_loans (full list)"
    assert [r["plan_text"] for r in phased.reviews] == executor.plans
    assert [r["floor"] for r in phased.reviews] == [0, 2]
    # A phase's check compares step values; unphased runs' checks do not.
    assert all(r["values"] is True for r in phased.reviews)


def test_the_trace_shows_each_phase(phased):
    phased.start()
    phased.ctx.process_turn("show the book Dune")

    spans = phased.sink.named(tracing.SPAN_SKILL_PHASE)
    assert [s.attributes["phase"] for s in spans] == ["find", "book", "report"]
    assert [s.attributes["phase_index"] for s in spans] == [0, 1, 2]
    assert {s.attributes["phase_count"] for s in spans} == {3}
    assert {s.attributes["step_budget"] for s in spans} == {15}
    assert [s.attributes["steps_used"] for s in spans] == [2, 2, 0]
    assert [s.attributes["answer_only"] for s in spans] == [False, False, True]
    assert spans[0].attributes["facts"] == {"book_uids": ["b1"]}
    assert spans[1].attributes["facts"] == {"uid": "b1", "loan_uids": ["l1", "l2"]}
    assert spans[1].attributes["record"] == BOOK
    assert spans[2].attributes["facts"] is None
    assert all(s.status == tracing.STATUS_OK for s in spans)
    contract = tracing.SPAN_CONTRACTS[tracing.SPAN_SKILL_PHASE].attributes
    for span in spans:
        assert set(span.attributes) - {tracing.ATTR_SPAN_CONTRACT_VERSION} <= contract
    # Each phase's executor run is its child.
    executes = phased.sink.named(tracing.SPAN_AGENT_EXECUTE)
    assert [e.parent_span_id for e in executes] == [s.span_id for s in spans]


def test_a_phase_suspended_on_ask_user_resumes_and_the_rest_run(phased):
    executor = phased.start(suspend_in="book")
    first = phased.ctx.process_turn("show the book Dune")
    assert first.status == TurnStatus.AWAITING_USER
    assert phased.ctx._phase_run["index"] == 1

    blob = phased.ctx.serialize_state(channel_id="phases-roundtrip")
    restored = WorkflowExecutionContext(run_as_agent=True)
    restored.apply_serialized_state(blob)
    assert restored._phase_run == phased.ctx._phase_run

    second = phased.ctx.process_turn("the first edition")
    assert second.status == TurnStatus.COMPLETED
    assert second.answer == "Dune (b1) is on loan: l1, l2."
    assert "(report)" in phased.answered[-1]
    spans = phased.sink.named(tracing.SPAN_SKILL_PHASE)
    assert [(s.attributes["phase"], s.attributes["resumed"]) for s in spans] == [
        ("find", False), ("book", False), ("book", True), ("report", False)]
    assert spans[1].attributes["suspended"] is True


def test_a_phase_executor_is_held_to_its_phase(phased):
    executor = phased.start()
    phased.ctx.process_turn("show the book Dune")

    queries = [call["user_query"] for call in executor.calls]
    for query in queries:
        # Only the phase's steps, framed as a part; the request is not in the input.
        assert "show the book Dune" not in query
        assert "Do only these steps, then finish. Later parts will do the rest." in query
    assert queries[0].endswith("1. open_library\n2. find_book query=Dune")
    assert queries[1].endswith("1. open_book_by_uid book_uid=b1\n2. list_loans (full list)")
    # The last phase answers the request, so it sees it.
    last = phased.answered[-1]
    assert last.startswith("show the book Dune\n\n")
    assert "Answer the request" in last
    # The executor span records the same input.
    executes = phased.sink.named(tracing.SPAN_AGENT_EXECUTE)
    assert [e.attributes["agent_input"] for e in executes] == queries + [last]


def test_a_search_phase_that_ended_on_a_record_fills_from_its_own_search(phased):
    # The find phase searched from the hub, then opened the book and ended on it.
    phased.script["find"] = dict(SCRIPT["find"], runs=SCRIPT["find"]["runs"] + [
        _run("open_book_by_uid", None, {"book_uid": "b1"}),
        _run("list_loans", {"loan_uids": ["l1"]}, context=BOOK)], record=BOOK)
    phased.start()
    result = phased.ctx.process_turn("show the book Dune")

    assert result.answer == "Dune (b1) is on loan: l1, l2."
    find = phased.sink.named(tracing.SPAN_SKILL_PHASE)[0]
    assert find.status == tracing.STATUS_OK
    assert find.attributes["facts"] == {"book_uids": ["b1"]}
    assert find.attributes["record"] == BOOK


def test_an_off_plan_search_in_a_phase_does_not_replace_its_fact(phased):
    # The find phase also looked up another title: the planned search fills the fact.
    phased.script["find"] = dict(SCRIPT["find"], runs=SCRIPT["find"]["runs"] + [
        _run("find_book", {"book_uids": ["b2"]}, {"query": "Emma"})])
    phased.start()
    result = phased.ctx.process_turn("show the book Dune")

    assert result.answer == "Dune (b1) is on loan: l1, l2."
    find = phased.sink.named(tracing.SPAN_SKILL_PHASE)[0]
    assert find.attributes["facts"] == {"book_uids": ["b1"]}


def test_a_record_phase_never_reads_another_record_runs(phased):
    # The book phase read b1's loans, then opened b2 and ended there: b2 has no loans read,
    # and opening b2 is not the planned step (open b1), so b2's uid is not taken either.
    phased.script["book"] = dict(SCRIPT["book"], runs=SCRIPT["book"]["runs"] + [
        _run("go_up", context=BOOK),
        _run("open_book_by_uid", None, {"book_uid": "b2"})], record=OTHER_BOOK)
    executor = phased.start()
    result = phased.ctx.process_turn("show the book Dune")

    assert len(executor.calls) == 2
    assert "stopped at its part 'book'" in result.answer and "BookFacts.uid" in result.answer
    assert "the planned run was not found" in result.answer
    book = phased.sink.named(tracing.SPAN_SKILL_PHASE)[1]
    assert book.status == tracing.STATUS_ERROR and book.attributes["record"] == OTHER_BOOK


def test_a_search_of_an_earlier_phase_never_counts(phased):
    # find_book ran only in the find phase; a later phase whose output reads it, and
    # which ran no search itself, has nothing to fill it from.
    def plan_reading_the_search_again(input):
        return [
            Phase("find", lambda: find_steps(input.name), output=Found),
            Phase("again", lambda find: ["say the uid again"], output=Found),
            Phase("report", lambda: report_steps()),
        ]

    phased.command = _phased_command(plan=plan_reading_the_search_again)
    phased.script["again"] = dict(runs=[], record=LIBRARY, answer="b1")
    executor = phased.start()
    result = phased.ctx.process_turn("show the book Dune")

    # "again" names no command: answer-only, so it ran nothing and has no run to fill from.
    assert len(executor.calls) == 1 and len(phased.answered) == 1
    assert "stopped at its part 'again'" in result.answer and "Found.book_uids" in result.answer


def test_a_phase_whose_opening_step_finds_the_record_already_open_fills_its_facts(phased):
    # The find phase also opened b1 and ended on it. The book phase's step 1 (open b1)
    # was answered "already in context" and ran nothing; its other step ran on b1.
    phased.script["find"] = dict(SCRIPT["find"], runs=SCRIPT["find"]["runs"] + [
        _run("open_book_by_uid", None, {"book_uid": "b1"})], record=BOOK)
    phased.script["book"] = dict(SCRIPT["book"], runs=SCRIPT["book"]["runs"][1:])
    phased.start()
    result = phased.ctx.process_turn("show the book Dune")

    assert result.answer == "Dune (b1) is on loan: l1, l2."
    book = phased.sink.named(tracing.SPAN_SKILL_PHASE)[1]
    assert book.status == tracing.STATUS_OK
    assert book.attributes["facts"] == {"uid": "b1", "loan_uids": ["l1", "l2"]}


# ----------------------------------------------------------------------
# Answer-only phases (fix-hia1.7)
# ----------------------------------------------------------------------

from fastworkflow.skill_phases import (  # noqa: E402
    ANSWER_ONLY_RULE,
    FANOUT_MAX_ITEMS_ENV,
    check_bindings,
    fan_out_items,
    is_answer_only,
    item_record,
    item_record_issues,
    write_answer,
)

COMMANDS = ["find_book", "open_book_by_uid", "list_loans", "count_notes"]
CORE = ["go_up", "reset_context"]


def test_a_phase_whose_steps_name_no_command_is_answer_only():
    assert is_answer_only("1. say that b1 has loans l1, l2\n2. compare the two", COMMANDS, CORE)
    assert not is_answer_only("1. find_book query=Dune", COMMANDS, CORE)
    # A conditional step, with or without a command, may need a run.
    assert not is_answer_only("1. list_loans only when the book has loans", COMMANDS, CORE)
    assert not is_answer_only("1. if a list is too long, page it", COMMANDS, CORE)
    # A navigation step needs the tools too; an answer step that only mentions a command does not.
    assert not is_answer_only("1. go_up\n2. say it", COMMANDS, CORE)
    assert is_answer_only("1. say which loans list_loans returned: l1, l2", COMMANDS, CORE)
    # With no command names known, or no steps, nothing is answer-only.
    assert not is_answer_only("1. say it", [], CORE)
    assert not is_answer_only("", COMMANDS, CORE)


def test_an_answer_only_query_says_to_run_no_command():
    last = phase_query("show the book Dune", "1. say that b1 has loans l1", index=2, count=3,
                       name="report", answer_only=True)
    assert last.startswith("show the book Dune\n\n")
    assert ANSWER_ONLY_RULE in last and ANSWER_ONLY_RULE == (
        "Do not run any command; write the answer from these facts only.")
    assert last.endswith("1. say that b1 has loans l1")
    middle = phase_query("show the book Dune", "1. say it", index=1, count=3, name="sum",
                         answer_only=True)
    assert ANSWER_ONLY_RULE in middle and "show the book Dune" not in middle


def test_write_answer_is_one_model_call_with_no_tools():
    from dspy.utils.dummies import DummyLM

    lm = DummyLM([{"final_answer": "Dune (b1) is on loan: l1, l2."}])
    with dspy.context(lm=lm):
        assert write_answer("show the book Dune ... 1. say that b1 has loans l1, l2") == \
            "Dune (b1) is on loan: l1, l2."
    assert len(lm.history) == 1
    assert not lm.history[0]["kwargs"].get("tools")


def test_an_answer_only_phase_never_runs_a_workflow_command(phased, monkeypatch):
    executor = phased.start()
    tool_calls = []
    monkeypatch.setattr(phased.ctx, "_execute_workflow_query",
                        lambda *a, **k: tool_calls.append(a) or "ran", raising=False)
    result = phased.ctx.process_turn("show the book Dune")

    # The report phase never reached the executor, so nothing could call execute_workflow_query.
    assert not any("(report)" in call["user_query"] for call in executor.calls)
    assert tool_calls == []
    assert len(phased.answered) == 1
    assert ANSWER_ONLY_RULE in phased.answered[0]
    assert "say that b1 has loans l1, l2" in phased.answered[0]
    # Its answer is the turn's answer.
    assert result.answer == "Dune (b1) is on loan: l1, l2."


def test_the_trace_shows_an_answer_only_phase_and_its_model_call(phased):
    phased.start()
    phased.ctx.process_turn("show the book Dune")

    report = phased.sink.named(tracing.SPAN_SKILL_PHASE)[-1]
    assert report.attributes["phase"] == "report"
    assert report.attributes["answer_only"] is True and report.attributes["steps_used"] == 0
    assert report.status == tracing.STATUS_OK
    execute = phased.sink.named(tracing.SPAN_AGENT_EXECUTE)[-1]
    assert execute.parent_span_id == report.span_id
    assert ANSWER_ONLY_RULE in execute.attributes["agent_input"]
    assert execute.attributes["final_answer"] == "Dune (b1) is on loan: l1, l2."
    # No tool call under it.
    assert not [s for s in phased.sink.named(tracing.SPAN_AGENT_TOOL_CALL)
                if s.parent_span_id == execute.span_id]


# ----------------------------------------------------------------------
# Fan-out: one sub-plan per item, bound to its record (fix-hia1.3)
# ----------------------------------------------------------------------

FIRST, SECOND, THIRD = "Book b1 Dune", "Book b2 Emma", "Book b3 Ivanhoe"


def each_book_steps(uid=None):
    return [f"open_book_by_uid book_uid={uid or '<the uid>'}", "list_loans (full list)"]


def report_books_steps(books: Optional[list] = None):
    SEEN["report_books"] = {"books": books}
    if books is None:
        return ["say who has each book on loan"]
    return ["say who has each book on loan: "
            + "; ".join(f"{b.uid} has {', '.join(b.loan_uids) or 'none'}" for b in books)]


def fan_plan(input):
    return [
        Phase("find", lambda: find_steps(input.name), output=Found),
        Phase("books", each_book_steps, output=BookFacts, over="find.book_uids", item="uid"),
        Phase("report", lambda books: report_books_steps(books)),
    ]


def _book_item(uid, clause, loans):
    return dict(runs=[_run("go_up", context="Book bx"),
                      _run("open_book_by_uid", None, {"book_uid": uid}),
                      _run("list_loans", {"loan_uids": loans}, context=clause)],
                record=clause, answer=f"{uid} has {len(loans)} loans")


FAN_SCRIPT = {
    "find": dict(runs=[_run("open_library", context=""),
                       _run("find_book", {"book_uids": ["b1", "b2", "b3"]}, {"query": "Dune"})],
                 record=LIBRARY, answer="found b1, b2, b3"),
    "books[b1]": _book_item("b1", FIRST, ["l1"]),
    "books[b2]": _book_item("b2", SECOND, ["l2", "l3"]),
    "books[b3]": _book_item("b3", THIRD, []),
    "report": dict(runs=[], record=THIRD, answer="b1: l1; b2: l2, l3; b3: none."),
}


@pytest.fixture
def fanned(phased, monkeypatch):
    monkeypatch.delenv(FANOUT_MAX_ITEMS_ENV, raising=False)
    phased.command = _phased_command(plan=fan_plan)
    phased.script = dict(FAN_SCRIPT)
    return phased


def test_n_items_run_as_n_sub_plans_and_one_final_phase(fanned):
    executor = fanned.start()
    result = fanned.ctx.process_turn("show the book Dune")

    queries = [call["user_query"] for call in executor.calls]
    assert len(queries) == 4  # find, then one sub-plan per book
    for k, uid in enumerate(["b1", "b2", "b3"], start=1):
        assert f"part 2 of 3 (books) for item {k} of 3 ({uid})" in queries[k]
        assert f"open_book_by_uid book_uid={uid}" in queries[k]
        assert "Do only these steps" in queries[k]
    # The final phase gets the list of per-item outputs, in item order.
    books = SEEN["report_books"]["books"]
    assert [type(b) for b in books] == [BookFacts] * 3
    assert [(b.uid, b.loan_uids) for b in books] == [("b1", ["l1"]), ("b2", ["l2", "l3"]),
                                                     ("b3", [])]
    assert "b2 has l2, l3" in fanned.answered[-1]
    assert result.answer == "b1: l1; b2: l2, l3; b3: none."
    spans = fanned.sink.named(tracing.SPAN_SKILL_PHASE)
    assert [(s.attributes["phase"], s.attributes.get("item")) for s in spans] == [
        ("find", None), ("books", "b1"), ("books", "b2"), ("books", "b3"), ("report", None)]
    items = spans[1:4]
    assert [s.attributes["record"] for s in items] == [FIRST, SECOND, THIRD]
    assert [s.attributes["record_issues"] for s in items] == [[], [], []]
    assert {(s.attributes["item_count"], s.attributes["items_total"]) for s in items} == {(3, 3)}
    # Leaving the previous item's record (go_up) is not a step on it: no note on a correct run.
    assert executor.notes == [None] * 4


def test_a_step_on_another_item_record_is_flagged(fanned):
    # Item b2's list_loans ran on b1's record: the check flags it, and b2's fact is not
    # taken from b1's reading.
    fanned.script["books[b2]"] = dict(
        runs=[_run("list_loans", {"loan_uids": ["l1"]}, context=FIRST)],
        record=FIRST, answer="b2 has 1 loan")
    executor = fanned.start()
    result = fanned.ctx.process_turn("show the book Dune")

    note = executor.notes[2]
    assert note and "list_loans ran on Book b1 Dune, the record of item b1, not of item b2" in note
    assert "item b2 was never opened" in note
    span = fanned.sink.named(tracing.SPAN_SKILL_PHASE)[2]
    assert span.status == tracing.STATUS_ERROR and span.attributes["item"] == "b2"
    assert any("the record of item b1" in issue for issue in span.attributes["record_issues"])
    assert "stopped at its part 'books'" in result.answer
    assert "item b2 was never opened" in result.answer


def test_an_item_never_opened_is_flagged():
    runs = [{"command": "list_loans", "context_clause": LIBRARY, "step_index": 4}]
    assert item_record(runs, "b2", LIBRARY) is None
    assert item_record_issues(runs, "b2", ["b1", "b2"], LIBRARY) == [
        "item b2 was never opened: no command ran on a record that names it"]


def test_the_item_record_check_passes_a_correct_run_and_flags_a_wrong_one():
    good = [{"command": "go_up", "context_clause": FIRST, "step_index": 6},
            {"command": "open_book_by_uid", "context_clause": LIBRARY, "step_index": 7},
            {"command": "list_loans", "context_clause": SECOND, "step_index": 8}]
    planned = ["open_book_by_uid", "list_loans"]
    assert item_record(good, "b2", SECOND) == SECOND
    assert item_record_issues(good, "b2", ["b1", "b2"], SECOND, planned) == []
    # Without the planned commands, go_up on b1's record is a run on another item's record.
    assert item_record_issues(good, "b2", ["b1", "b2"], SECOND) == [
        "step 6: go_up ran on Book b1 Dune, the record of item b1, not of item b2"]
    # Bound to the item's record even when the run moved on afterwards.
    assert item_record(good + [{"command": "go_up", "context_clause": SECOND}], "b2",
                       LIBRARY) == SECOND


def test_a_bound_item_fills_from_its_record_even_when_it_ended_elsewhere(fanned):
    fanned.script["books[b2]"] = dict(_book_item("b2", SECOND, ["l2"]), record=LIBRARY)
    fanned.script["books[b2]"]["runs"] = fanned.script["books[b2]"]["runs"] + [
        _run("go_up", context=SECOND)]
    fanned.start()
    fanned.ctx.process_turn("show the book Dune")

    span = fanned.sink.named(tracing.SPAN_SKILL_PHASE)[2]
    assert span.status == tracing.STATUS_OK
    assert span.attributes["record"] == SECOND
    assert span.attributes["facts"] == {"uid": "b2", "loan_uids": ["l2"]}


def test_the_cap_is_applied_and_said_in_the_answer(fanned, monkeypatch):
    monkeypatch.setenv(FANOUT_MAX_ITEMS_ENV, "2")
    executor = fanned.start()
    result = fanned.ctx.process_turn("show the book Dune")

    assert len(executor.calls) == 3  # find + two items
    assert [b.uid for b in SEEN["report_books"]["books"]] == ["b1", "b2"]
    assert result.answer.endswith(
        "Note: only the first 2 of 3 items were covered (part 'books' runs at most 2 items).")
    spans = fanned.sink.named(tracing.SPAN_SKILL_PHASE)
    assert {(s.attributes["item_count"], s.attributes["items_total"])
            for s in spans if "item" in s.attributes} == {(2, 3)}


def test_the_cap_defaults_to_10_and_a_phase_may_set_its_own(monkeypatch):
    monkeypatch.delenv(FANOUT_MAX_ITEMS_ENV, raising=False)
    many = Found(book_uids=[f"b{i}" for i in range(14)])
    phase = Phase("books", each_book_steps, output=BookFacts, over="find.book_uids", item="uid")
    items, total = fan_out_items(phase, {"find": many})
    assert len(items) == 10 and total == 14
    small = Phase("books", each_book_steps, over="find.book_uids", item="uid", max_items=3)
    assert fan_out_items(small, {"find": many}) == (["b0", "b1", "b2"], 14)
    # Repeats and empty values are dropped, order kept.
    listed = Phase("books", each_book_steps, over=["b2", "b1", "b2", ""], item="uid")
    assert fan_out_items(listed, {}) == (["b2", "b1"], 2)


def test_fan_out_over_a_list_from_the_input(phased):
    class Titles(LibrarySignature.Input):
        pass

    def titles_plan(input):
        names = [n.strip() for n in input.name.split(",")]
        return [
            Phase("each", lambda title: [f"find_book query={title}"], output=Found,
                  over=names, item="title", bind=False),
            Phase("report", lambda each: [
                "say the matches: " + "; ".join(", ".join(f.book_uids) for f in each or [])]),
        ]

    phased.command = _phased_command(plan=titles_plan)
    phased.args["name"] = "Dune, Emma"
    phased.script = {
        "each[Dune]": dict(runs=[_run("find_book", {"book_uids": ["b1"]}, {"query": "Dune"})],
                           record=LIBRARY, answer="b1"),
        "each[Emma]": dict(runs=[_run("find_book", {"book_uids": ["b2"]}, {"query": "Emma"})],
                           record=LIBRARY, answer="b2"),
        "report": dict(runs=[], record=LIBRARY, answer="Dune is b1; Emma is b2."),
    }
    executor = phased.start()
    result = phased.ctx.process_turn("find Dune and Emma")

    assert [c["user_query"].split("\n")[0] for c in executor.calls] == [
        "This is part 1 of 2 (each) for item 1 of 2 (Dune) of a larger job. Do only these "
        "steps, then finish. Later parts will do the rest.",
        "This is part 1 of 2 (each) for item 2 of 2 (Emma) of a larger job. Do only these "
        "steps, then finish. Later parts will do the rest."]
    assert "say the matches: b1; b2" in phased.answered[-1]
    assert result.answer == "Dune is b1; Emma is b2."
    # The flat steps give the steps once per item.
    assert concatenated_steps(titles_plan(LibrarySignature.Input(name="Dune, Emma")))[:2] == [
        "find_book query=Dune", "find_book query=Emma"]


def test_a_fan_out_item_suspended_on_ask_user_resumes_and_the_rest_run(fanned):
    executor = fanned.start(suspend_in="books[b2]")
    first = fanned.ctx.process_turn("show the book Dune")
    assert first.status == TurnStatus.AWAITING_USER
    assert fanned.ctx._phase_run["item_index"] == 1
    assert fanned.ctx._phase_run["items"] == ["b1", "b2", "b3"]

    second = fanned.ctx.process_turn("the first edition")
    assert second.status == TurnStatus.COMPLETED
    assert [b.uid for b in SEEN["report_books"]["books"]] == ["b1", "b2", "b3"]
    assert len(executor.calls) == 4


def test_a_bad_over_fails_the_load():
    with pytest.raises(SkillFormatError, match="over must be"):
        _phased_command(plan=lambda input: [Phase("a", lambda item: ["x"], over="not a reference")])
    with pytest.raises(SkillFormatError, match="not an earlier phase with an output"):
        _phased_command(plan=lambda input: [
            Phase("books", each_book_steps, over="find.book_uids", item="uid")])
    with pytest.raises(SkillFormatError, match="has no field 'uids'"):
        _phased_command(plan=lambda input: [
            Phase("find", lambda: ["find_book query=x"], output=Found),
            Phase("books", each_book_steps, over="find.uids", item="uid")])
    with pytest.raises(PhaseError, match="also an earlier phase's name"):
        check_bindings([Phase("find", lambda: ["x"], output=Found),
                        Phase("books", lambda find: [], over="find.book_uids", item="find")])


# ----------------------------------------------------------------------
# plan() may run a call flat; a condition on earlier facts; a fan-out that
# stops on a condition (fix-hia1.9)
# ----------------------------------------------------------------------

from fastworkflow.skill_phases import (  # noqa: E402
    cap_note,
    fan_out_note,
    is_flat_plan,
    runs_now,
    stops_after,
)


def flat_steps(input):
    return find_steps(input.name) + ["say what was found"]


def notes_only_plan(input):
    """Phased only when the notes are asked for; otherwise the call runs flat."""
    if not input.notes:
        return None
    return plan(input)


def test_a_plan_that_returns_none_or_nothing_means_run_flat():
    assert is_flat_plan(None) and is_flat_plan([]) and is_flat_plan(())
    assert not is_flat_plan([Phase("a", lambda: ["x"])])
    command = _phased_command(plan=notes_only_plan, steps=flat_steps)
    assert phases_for(command, {"name": "Dune", "notes": False}) is None
    assert [p.name for p in phases_for(command, {"name": "Dune", "notes": True})] == [
        "find", "book", "notes", "report"]
    empty = _phased_command(plan=lambda input: [] if not input.notes else plan(input),
                            steps=flat_steps)
    assert phases_for(empty, {"name": "Dune", "notes": False}) is None
    # The steps() the planner and skill matching read are the file's own.
    assert command.steps_fn(LibrarySignature.Input(name="Dune")) == [
        "open_library", "find_book query=Dune", "say what was found"]


def test_a_plan_that_may_run_flat_needs_its_own_steps():
    with pytest.raises(SkillFormatError, match="needs its own steps"):
        _phased_command(plan=notes_only_plan)
    with pytest.raises(PhaseError, match="run flat"):
        concatenated_steps(None)
    # Every phase left out by when=False is still an error, not a flat run.
    with pytest.raises(SkillFormatError, match="no phase"):
        _phased_command(plan=lambda input: [Phase("a", lambda: ["x"], when=False)],
                        steps=flat_steps)


def test_a_flat_plan_result_runs_the_call_as_a_flat_skill_call(phased, monkeypatch):
    _knows_the_toy_commands(phased, monkeypatch)
    phased.command = _phased_command(plan=notes_only_plan, steps=flat_steps)
    executor = phased.start()
    result = phased.ctx.process_turn("show the book Dune")

    assert len(executor.calls) == 1
    assert executor.calls[0]["max_iters"] == 25       # one phase, with the executor's step cap
    assert [s.attributes["phase"] for s in phased.sink.named(tracing.SPAN_SKILL_PHASE)] == [
        "show_book"]
    assert phased.ctx._phase_run is None
    assert result.answer == "found b1"


def test_the_same_skill_runs_phased_for_other_inputs(phased):
    phased.command = _phased_command(plan=notes_only_plan, steps=flat_steps)
    phased.args["notes"] = True
    executor = phased.start()
    result = phased.ctx.process_turn("show the book Dune and its notes")

    assert [s.attributes["phase"] for s in phased.sink.named(tracing.SPAN_SKILL_PHASE)] == [
        "find", "book", "notes", "report"]
    assert {call["max_iters"] for call in executor.calls} == {15}
    assert result.answer == "Dune (b1) is on loan: l1, l2."


def has_loans_plan(input):
    return [
        Phase("find", lambda: find_steps(input.name), output=Found),
        Phase("book", lambda find: book_steps(find), output=BookFacts),
        Phase("notes", lambda book: note_steps(book), output=NoteFacts,
              when=lambda book: bool(book.loan_uids)),
        Phase("report", report_steps),
    ]


def test_a_callable_when_binds_earlier_outputs_by_name():
    phase = Phase("notes", lambda: [], when=lambda book: bool(book.loan_uids))
    assert runs_now(phase, {"book": BookFacts(uid="b1", loan_uids=["l1"])}) is True
    assert runs_now(phase, {"book": BookFacts(uid="b1", loan_uids=[])}) is False
    with pytest.raises(PhaseError, match="when needs 'book'"):
        runs_now(phase, {})
    with pytest.raises(PhaseError, match="when failed: ZeroDivisionError"):
        runs_now(Phase("x", lambda: [], when=lambda: 1 / 0), {})
    assert runs_now(Phase("x", lambda: [], when=True), {}) is True
    # A callable when is kept by plan() and in the flat steps; it is decided at run time.
    command = _phased_command(plan=has_loans_plan)
    assert [p.name for p in phases_for(command, {"name": "Dune"})] == [
        "find", "book", "notes", "report"]
    assert "count_notes on the book" in command.steps_fn(LibrarySignature.Input(name="Dune"))


def test_a_callable_when_whose_parameter_nothing_provides_fails_the_load():
    with pytest.raises(SkillFormatError, match="when takes 'book'"):
        _phased_command(plan=lambda input: [
            Phase("find", lambda: ["find_book query=x"], output=Found),
            Phase("notes", lambda: ["count_notes"], when=lambda book: True)])


def test_a_phase_whose_condition_holds_runs(phased):
    phased.command = _phased_command(plan=has_loans_plan)
    phased.start()
    phased.ctx.process_turn("show the book Dune")

    assert SEEN["notes"]["book"].uid == "b1"
    assert SEEN["report"]["notes"].total == 3
    notes = phased.sink.named(tracing.SPAN_SKILL_PHASE)[2]
    assert notes.attributes["phase"] == "notes" and "skipped" not in notes.attributes


def test_a_phase_whose_condition_fails_is_skipped_and_counts_as_not_run(phased):
    phased.command = _phased_command(plan=has_loans_plan)
    phased.script["book"] = dict(SCRIPT["book"], runs=[
        SCRIPT["book"]["runs"][0], _run("list_loans", {"loan_uids": []}, context=BOOK)])
    executor = phased.start()
    phased.ctx.process_turn("show the book Dune")

    assert not any("(notes)" in call["user_query"] for call in executor.calls)
    assert "notes" not in SEEN
    assert SEEN["report"]["notes"] is None          # bound to it: its default
    spans = phased.sink.named(tracing.SPAN_SKILL_PHASE)
    assert [s.attributes["phase"] for s in spans] == ["find", "book", "notes", "report"]
    assert spans[2].attributes["skipped"] is True and spans[2].status == tracing.STATUS_OK
    assert not [s for s in phased.sink.named(tracing.SPAN_AGENT_EXECUTE)
                if s.parent_span_id == spans[2].span_id]
    # The report keeps its place in the job: part 4 of 4.
    assert "part 4 of 4 (report)" in phased.answered[-1]


def test_a_condition_that_raises_fails_the_phase(phased):
    phased.command = _phased_command(plan=lambda input: [
        Phase("find", lambda: find_steps(input.name), output=Found),
        Phase("report", lambda find: ["say it"], when=lambda find: find.missing_field)])
    phased.start()
    result = phased.ctx.process_turn("show the book Dune")

    assert "stopped at its part 'report'" in result.answer
    assert "when failed: AttributeError" in result.answer
    span = phased.sink.named(tracing.SPAN_SKILL_PHASE)[-1]
    assert span.status == tracing.STATUS_ERROR and span.attributes["skipped"] is False


def test_a_skipped_last_phase_leaves_the_answer_of_the_phase_before(phased):
    phased.command = _phased_command(plan=lambda input: [
        Phase("find", lambda: find_steps(input.name), output=Found),
        Phase("book", lambda find: book_steps(find), output=BookFacts),
        Phase("report", lambda book=None: report_steps(book), when=lambda book: not book.loan_uids)])
    phased.start()
    result = phased.ctx.process_turn("show the book Dune")
    assert result.answer == "b1 has two loans"


FIVE = ["b1", "b2", "b3", "b4", "b5"]
FIVE_LOANS = {"b1": ["l1"], "b2": [], "b3": ["l3"], "b4": ["l4"], "b5": []}


def two_with_loans_plan(max_items=None, noun=None):
    def plan_(input):
        return [
            Phase("find", lambda: find_steps(input.name), output=Found),
            Phase("books", each_book_steps, output=BookFacts, over="find.book_uids", item="uid",
                  max_items=max_items, noun=noun,
                  until=lambda books: sum(1 for b in books if b.loan_uids) >= 2),
            Phase("report", lambda books: report_books_steps(books)),
        ]
    return plan_


@pytest.fixture
def five(phased, monkeypatch):
    monkeypatch.delenv(FANOUT_MAX_ITEMS_ENV, raising=False)
    phased.script = {
        "find": dict(runs=[_run("open_library", context=""),
                           _run("find_book", {"book_uids": FIVE}, {"query": "Dune"})],
                     record=LIBRARY, answer="found five"),
        **{f"books[{uid}]": _book_item(uid, f"Book {uid} T{uid}", FIVE_LOANS[uid])
           for uid in FIVE},
        "report": dict(runs=[], record=LIBRARY, answer="the books with loans"),
    }
    return phased


def test_until_binds_the_outputs_so_far_by_the_phase_name():
    phase = Phase("books", lambda uid: [], output=BookFacts, over="find.book_uids", item="uid",
                  until=lambda books, find: len(books) >= len(find.book_uids) - 1)
    found = Found(book_uids=["b1", "b2", "b3"])
    one = BookFacts(uid="b1", loan_uids=[])
    assert stops_after(phase, [one], {"find": found}) is False
    assert stops_after(phase, [one, one], {"find": found}) is True
    assert stops_after(Phase("x", lambda: []), [one], {}) is False   # no until


def test_a_fan_out_stops_once_its_condition_holds(five):
    five.command = _phased_command(plan=two_with_loans_plan())
    executor = five.start()
    result = five.ctx.process_turn("show the book Dune")

    assert len(executor.calls) == 4    # find + b1, b2, b3: b1 and b3 have loans
    assert [b.uid for b in SEEN["report_books"]["books"]] == ["b1", "b2", "b3"]
    assert result.answer.endswith(
        "Note: part 'books' stopped after 3 of 5 items because its stop condition was met.")
    items = [s for s in five.sink.named(tracing.SPAN_SKILL_PHASE) if "item" in s.attributes]
    assert [s.attributes["stop_condition_met"] for s in items] == [False, False, True]


def test_the_cap_still_applies_to_a_fan_out_with_a_condition(five):
    five.command = _phased_command(plan=two_with_loans_plan(max_items=2))
    executor = five.start()
    result = five.ctx.process_turn("show the book Dune")

    assert len(executor.calls) == 3    # find + b1, b2: one match, then the cap
    assert [b.uid for b in SEEN["report_books"]["books"]] == ["b1", "b2"]
    assert result.answer.endswith(
        "Note: only the first 2 of 5 items were covered (part 'books' runs at most 2 items).")
    assert "stop condition" not in result.answer


def test_the_fan_out_notes_tell_stopped_from_capped():
    assert fan_out_note("books", 3, 5, stopped=True) == (
        "Note: part 'books' stopped after 3 of 5 items because its stop condition was met.")
    assert fan_out_note("books", 5, 5, stopped=True) is None      # nothing was left out
    assert fan_out_note("books", 2, 5, stopped=False).startswith("Note: only the first 2 of 5")
    assert fan_out_note("books", 5, 5, stopped=False) is None


def test_a_noun_names_the_items_in_the_notes_without_the_phase_name():
    assert fan_out_note("books", 3, 5, stopped=True, noun="books on loan") == (
        "Note: the work stopped after 3 of 5 books on loan because its stop condition was met.")
    assert fan_out_note("books", 2, 5, stopped=False, noun="people") == (
        "Note: only the first 2 of 5 people were covered.")
    assert cap_note("books", 2, 5, noun="people") == "Note: only the first 2 of 5 people were covered."
    assert fan_out_note("books", 5, 5, stopped=False, noun="people") is None


def test_a_noun_is_used_in_the_answer_note(five):
    five.command = _phased_command(plan=two_with_loans_plan(noun="titles"))
    five.start()
    result = five.ctx.process_turn("show the book Dune")
    assert result.answer.endswith(
        "Note: the work stopped after 3 of 5 titles because its stop condition was met.")
    assert "'books'" not in result.answer


def test_a_capped_fan_out_with_a_noun_says_the_noun(five):
    five.command = _phased_command(plan=two_with_loans_plan(max_items=2, noun="titles"))
    five.start()
    result = five.ctx.process_turn("show the book Dune")
    assert result.answer.endswith("Note: only the first 2 of 5 titles were covered.")


def test_two_capped_fan_outs_with_one_noun_say_the_note_once(fanned, monkeypatch):
    # Two fan-outs over the same items (a job split in two to fit the step budget).
    monkeypatch.setenv(FANOUT_MAX_ITEMS_ENV, "2")
    fanned.command = _phased_command(plan=lambda input: [
        Phase("find", lambda: find_steps(input.name), output=Found),
        Phase("books", each_book_steps, output=BookFacts, over="find.book_uids", item="uid",
              noun="titles"),
        Phase("again", each_book_steps, output=BookFacts, over="find.book_uids", item="uid",
              noun="titles"),
        Phase("report", lambda books: report_books_steps(books)),
    ])
    for uid, clause, loans in (("b1", FIRST, ["l1"]), ("b2", SECOND, ["l2", "l3"])):
        fanned.script[f"again[{uid}]"] = _book_item(uid, clause, loans)
    fanned.start()
    result = fanned.ctx.process_turn("show the book Dune")
    assert result.answer.count("Note:") == 1
    assert result.answer.endswith("Note: only the first 2 of 3 titles were covered.")


def test_a_bad_noun_fails_the_load():
    with pytest.raises(SkillFormatError, match="noun needs a fan-out"):
        _phased_command(plan=lambda input: [Phase("a", lambda: ["x"], noun="people")])
    for bad in ("", "  ", 3):
        with pytest.raises(SkillFormatError, match="noun must be a non-empty str"):
            _phased_command(plan=_noun_plan(bad))


def _noun_plan(noun):
    return lambda input: [
        Phase("find", lambda: ["find_book query=x"], output=Found),
        Phase("books", each_book_steps, over="find.book_uids", item="uid", noun=noun)]


def test_a_bad_until_fails_the_load():
    with pytest.raises(SkillFormatError, match="until needs a fan-out"):
        _phased_command(plan=lambda input: [Phase("a", lambda: ["x"], until=lambda a: True)])
    with pytest.raises(SkillFormatError, match="until takes 'books'"):
        _phased_command(plan=lambda input: [
            Phase("find", lambda: ["find_book query=x"], output=Found),
            Phase("books", each_book_steps, over="find.book_uids", item="uid",
                  until=lambda books: True)])   # no output: nothing to give `books`


# -- a stand-in opener, and a phase re-bound to its planned record (fix-hia1.10) --

SHELF = "Shelf s1 Fiction"


class ListedBook(BaseModel):
    uid: str = fact_from("Shelf/list_books.book_uid")
    loan_uids: list[str] = fact_from("list_loans.loan_uids")


LISTED_PLAN = "1. go_up\n2. list_books book_uid=b1 (this opens it)\n3. list_loans (full list)"


def _listed_runs(opener_uid="b1", opened_record=BOOK):
    return [_cmd_run("go_up", None, context=SHELF),
            _cmd_run("go_up", None, context="Shelf s1 Fiction"),
            _opened(opener_uid, "Dune"),
            _cmd_run("Book/list_loans", {"start": None}, {"loan_uids": ["l1"]},
                     context=opened_record)]


def test_another_command_that_entered_the_record_with_the_value_stands_in_for_the_opener():
    filled = fill_output(ListedBook, _listed_runs(), record=BOOK, plan_text=LISTED_PLAN)
    assert (filled.uid, filled.loan_uids) == ("b1", ["l1"])
    # The planned opener's own run is used when it ran.
    runs = [_cmd_run("Shelf/list_books", {"book_uid": "b1"}, None, context=SHELF),
            _cmd_run("Book/list_loans", {"start": None}, {"loan_uids": ["l1"]}, context=BOOK)]
    assert fill_output(ListedBook, runs, record=BOOK, plan_text=LISTED_PLAN).uid == "b1"
    # It also stands in from before the phase (an earlier phase opened the record).
    runs = _listed_runs()
    assert fill_output(ListedBook, runs[3:], record=BOOK, plan_text=LISTED_PLAN,
                       earlier_runs=runs[:3]).uid == "b1"


def test_an_opener_with_another_value_does_not_stand_in():
    # b2 was opened, and the phase is on b2: the step's b1 is not that value.
    runs = _listed_runs("b2", OTHER_BOOK)
    runs[2]["context_clause"] = LIBRARY
    with pytest.raises(MissingFactError, match=r"ListedBook\.uid.*the planned run was not found"):
        fill_output(ListedBook, runs, record=OTHER_BOOK, plan_text=LISTED_PLAN)


def test_an_opener_onto_another_record_does_not_stand_in():
    # The run opened b1, but the phase ended on b2 (the record it is filled on).
    runs = _listed_runs() + [_cmd_run("go_up", None, context=BOOK),
                             _opened("b2", "Emma")]
    with pytest.raises(MissingFactError, match=r"ListedBook\.uid"):
        fill_output(ListedBook, runs, record=OTHER_BOOK, plan_text=LISTED_PLAN)
    # A failed opener never stands in.
    runs = _listed_runs()
    runs[2]["success"] = False
    with pytest.raises(MissingFactError, match=r"ListedBook\.uid"):
        fill_output(ListedBook, runs, record=BOOK, plan_text=LISTED_PLAN)
    # A step without a concrete value takes no stand-in.
    with pytest.raises(MissingFactError, match=r"ListedBook\.uid"):
        fill_output(ListedBook, _listed_runs(), record=BOOK,
                    plan_text="1. list_books book_uid=<the uid>\n2. list_loans")


REBIND_PLAN = "1. open_book_by_uid book_uid=b1\n2. list_loans (full list)"


def test_a_phase_that_wandered_off_is_rebound_to_the_record_its_steps_ran_on():
    runs = [_opened("b1", "Dune"),
            _cmd_run("Book/list_loans", {"start": None}, {"loan_uids": ["l1"]}, context=BOOK),
            _cmd_run("reset_context", None, context=BOOK),
            _cmd_run("open_archive", None, context=""),
            _cmd_run("Archive/list_boxes", {}, {"box_uids": ["x"]}, context="Archive")]
    assert planned_record(runs, REBIND_PLAN, ended_on="Archive") == BOOK
    filled = fill_output(OpenedBook, runs, record=BOOK, plan_text=REBIND_PLAN, ended_on="Archive")
    assert (filled.uid, filled.loan_uids) == ("b1", ["l1"])
    # Ended on the record: nothing to re-bind.
    assert planned_record(runs[:2], REBIND_PLAN, ended_on=BOOK) is None
    assert planned_record(runs, "", ended_on="Archive") is None


def test_a_phase_whose_steps_ran_on_two_records_is_not_rebound():
    runs = [_opened("b1", "Dune"),
            _cmd_run("Book/list_loans", {"start": None}, {"loan_uids": ["l1"]}, context=BOOK),
            _cmd_run("go_up", None, context=BOOK),
            _opened("b2", "Emma"),
            _cmd_run("Book/list_loans", {"start": None}, {"loan_uids": ["l2"]}, context=OTHER_BOOK),
            _cmd_run("reset_context", None, context=OTHER_BOOK)]
    assert planned_record(runs, REBIND_PLAN + "\n3. open_book_by_uid book_uid=b2",
                          ended_on="") is None
    # The step's command run on another record (b2 opened for a step that opens b1)
    # is the step done on the wrong record: no re-bind.
    assert planned_record(runs[:4], REBIND_PLAN, ended_on=OTHER_BOOK) is None
    # A step that moves from one record to the next touches both.
    drill = [_cmd_run("Book/list_loans", {"start": None}, {"loan_uids": ["l1"]}, context=BOOK),
             _cmd_run("Book/open_loan", {"loan_uid": "l1"}, {}, context=BOOK),
             _cmd_run("Loan/show_due", {}, {}, context="Loan l1 Due")]
    assert planned_record(drill, "1. list_loans\n2. open_loan loan_uid=l1",
                          ended_on="") is None
    # Only planned steps count: off-plan runs on records do not.
    assert planned_record(runs[:3] + [_cmd_run("Archive/list_boxes", {}, {}, context="Archive a1 X")],
                          REBIND_PLAN, ended_on="Archive a1 X") == BOOK


# ----------------------------------------------------------------------
# Optional phases; a failed call shows only facts; answer-only checks (stage 10)
# ----------------------------------------------------------------------

from fastworkflow.skill_phases import (  # noqa: E402
    FOUND_FACTS_LEAD,
    MISSING_ITEMS_LEAD,
    answer_from_steps,
    answer_only_answer,
    clean_answer,
    failure_text,
    found_facts,
    is_empty_answer,
    missing_list_items,
)


def optional_notes_plan(optional=True, when=True):
    def plan_(input):
        return [
            Phase("find", lambda: find_steps(input.name), output=Found),
            Phase("book", lambda find: book_steps(find), output=BookFacts),
            Phase("notes", lambda book: note_steps(book), output=NoteFacts, when=when,
                  optional=optional),
            Phase("report", report_steps),
        ]
    return plan_


def test_an_optional_phase_that_fails_does_not_stop_the_call(phased):
    phased.command = _phased_command(plan=optional_notes_plan())
    phased.script["notes"] = dict(SCRIPT["notes"], runs=[], answer="it has 7 notes")  # no count_notes
    phased.start()
    result = phased.ctx.process_turn("show the book Dune")

    assert "stopped at its part" not in result.answer
    assert result.answer == "Dune (b1) is on loan: l1, l2."
    assert "7 notes" not in result.answer            # the failed phase's own text is dropped
    assert SEEN["report"]["notes"] is None          # counts as not run: the default
    assert "part 4 of 4 (report)" in phased.answered[-1]
    notes = phased.sink.named(tracing.SPAN_SKILL_PHASE)[2]
    assert notes.status == tracing.STATUS_ERROR and "NoteFacts.total" in notes.attributes["facts_error"]
    assert phased.ctx._phase_run is None


def test_a_required_phase_that_fails_still_stops_the_call(phased):
    phased.command = _phased_command(plan=optional_notes_plan(optional=False))
    phased.script["notes"] = dict(SCRIPT["notes"], runs=[], answer="it has 7 notes")
    phased.start()
    result = phased.ctx.process_turn("show the book Dune")

    assert "stopped at its part 'notes'" in result.answer and "Not run: report." in result.answer
    assert "7 notes" not in result.answer
    # The facts already filled, as code rendered them.
    assert "find:\n- book uids: b1" in result.answer
    assert "book:\n- uid: b1\n- loan uids: l1, l2" in result.answer


def test_an_optional_phase_whose_condition_raises_is_left_out(phased):
    phased.command = _phased_command(plan=optional_notes_plan(
        when=lambda book: book.missing_field))
    phased.start()
    result = phased.ctx.process_turn("show the book Dune")
    assert result.answer == "Dune (b1) is on loan: l1, l2."


def test_optional_must_be_a_bool():
    with pytest.raises(SkillFormatError, match="optional must be a bool"):
        _phased_command(plan=lambda input: [
            Phase("find", lambda: ["find_book query=x"], output=Found, optional="yes")])


def test_found_facts_renders_each_phase_output_in_code():
    assert found_facts({"find": {"book_uids": ["b1"]},
                        "books": [{"uid": "b1"}, {"uid": "b2"}], "none": []}) == [
        "find:\n- book uids: b1", "books, item 1:\n- uid: b1", "books, item 2:\n- uid: b2"]


def test_a_stopped_job_shows_its_facts_readably():
    # Stage 10 (7b): a raw key: value dump with every uid column. Now: labels in words, a
    # uid column left out where its labels name the rows, a record's own uid kept, and
    # long lists cut short with their count.
    facts = {"uid": "s1", "label": "Shelf C4", "book_uids": ["b1", "b2"],
             "book_loan_uids": ["l1", "l2"], "book_labels": ["Dune", "Emma"],
             "book_total": 2, "complete": True, "score": 4.0, "owner_uids": ["p9"],
             "note_labels": [f"n{k}" for k in range(30)]}
    block = found_facts({"shelf_read": facts}, max_items=3)[0]
    assert block == ("shelf read:\n- Shelf C4 (uid s1)\n- books: Dune, Emma\n"
                     "- book total: 2\n- complete: yes\n- score: 4\n- owner uids: p9\n"
                     "- notes: n0, n1, n2 (and 27 more; 30 in all)")
    assert found_facts({"unit": {"identity_labels": ["Ann"], "label": None, "uid": "u1"}})[0] \
        == "unit:\n- identities: Ann\n- label: none\n- uid: u1"
    text = failure_text([block], "loans", "no loans read", ["report"])
    assert text.startswith(f"{FOUND_FACTS_LEAD}\n\nshelf read:\n")
    assert text.endswith("This job stopped at its part 'loans', which failed: no loans read. "
                         "Not run: report.")


def test_an_answer_only_answer_that_drops_a_list_item_gets_it_appended(phased):
    phased.command = _phased_command(plan=lambda input: [
        Phase("find", lambda: find_steps(input.name), output=Found),
        Phase("book", lambda find: book_steps(find), output=BookFacts),
        Phase("report", lambda book=None: [f"name every loan of {book.uid}: "
                                           + ", ".join(book.loan_uids)] if book else
              ["name every loan"])])
    phased.script["report"] = dict(SCRIPT["report"], answer="Dune (b1) is on loan: l1.")
    phased.start()
    result = phased.ctx.process_turn("show the book Dune")
    assert result.answer == f"Dune (b1) is on loan: l1.\n\n{MISSING_ITEMS_LEAD} l2."


def test_an_answer_only_reply_that_is_empty_is_written_from_the_steps(phased, monkeypatch):
    monkeypatch.setattr("fastworkflow.skill_phases.write_answer", lambda query: "```\n```")
    phased.start()
    result = phased.ctx.process_turn("show the book Dune")
    assert result.answer == "B1 has loans l1, l2."


def test_an_answer_only_reply_that_cannot_be_parsed_is_written_from_the_steps(monkeypatch):
    from dspy.utils.exceptions import AdapterParseError

    def unparsable(query):
        raise AdapterParseError(adapter_name="ChatAdapter", lm_response="?",
                                signature=dspy.Signature("user_query -> final_answer"))

    monkeypatch.setattr("fastworkflow.skill_phases.write_answer", unparsable)
    book = BookFacts(uid="b1", loan_uids=["l1", "l2", "l3", "l4"])
    # written from the steps, so laid out as a code-written answer, and still checked
    steps = "1. say which loan is due first\n2. say what b1 has on loan, naming each: l1, l2, l3, l4"
    assert answer_only_answer("q", steps, {"book": book}) == (
        "- Which loan is due first.\n- What b1 has on loan, naming each:\n  - l1\n  - l2\n  - l3\n  - l4")


def test_write_answer_retries_an_empty_reply_once_and_strips_field_markers():
    from dspy.utils.dummies import DummyLM

    lm = DummyLM([{"final_answer": "```"},
                  {"final_answer": "Dune (b1) is on loan: l1, l2. [[ ## completed ## ]]"}])
    with dspy.context(lm=lm):
        assert write_answer("1. say that b1 has loans l1, l2") == "Dune (b1) is on loan: l1, l2."
    assert len(lm.history) == 2


def test_fact_from_is_the_field_of_a_fact_copied_from_a_command_field():
    class Page(BaseModel):
        uids: list[str] = fact_from("list_items.item_uids", min_length=1,
                                    json_schema_extra={"page_by": "cursor"})
        total: int = fact_from(["count_items.total", "list_items.total"], default=0)

    uids, total = Page.model_fields["uids"], Page.model_fields["total"]
    assert uids.json_schema_extra == {"page_by": "cursor", "from": "list_items.item_uids"}
    assert uids.is_required() and Page.model_json_schema()["properties"]["uids"]["minItems"] == 1
    assert total.json_schema_extra == {"from": ["count_items.total", "list_items.total"]}
    assert total.default == 0


def test_the_answer_checks_in_code():
    assert clean_answer("a [[ ## completed ## ]]\n[[## final_answer ##]]b  ") == "a\nb"
    for empty in ("", "```", "```text\n\n```", "[[ ## completed ## ]]", "ok."):
        assert is_empty_answer(empty)
    assert not is_empty_answer("Dune is on loan.")
    assert answer_from_steps("1. say that b1 has loans l1\n2. apply no change") == "B1 has loans l1."
    assert answer_from_steps("1. b1 has loans l1") == "B1 has loans l1."


def test_missing_list_items_reads_lists_the_steps_give_in_full_as_rows():
    class Shelf(BaseModel):
        uids: list[str] = []
        labels: list[str] = []
        notes: list[str] = []

    plan = "1. say that the shelf holds books, naming every one: Dune (b1), Emma (b2), Ivanhoe (b3)"
    shelf = Shelf(uids=["b1", "b2", "b3"], labels=["Dune", "Emma", "Ivanhoe"],
                  notes=["not in the steps"])
    # Each row the answer does not name in full, as the steps write it.
    assert missing_list_items("Dune (b1) and Emma.", plan, {"shelf": shelf}) == [
        "Emma (b2)", "Ivanhoe (b3)"]
    assert missing_list_items("Dune (b1), Emma (b2), Ivanhoe (b3).", plan, {"shelf": shelf}) == []
    # A fan-out phase's list of outputs; a list the steps do not give in full is not checked.
    assert missing_list_items("nothing", plan, {"shelves": [shelf, Shelf()]}) == [
        "Dune (b1)", "Emma (b2)", "Ivanhoe (b3)"]
    assert missing_list_items("nothing", "1. say how many books", {"shelf": shelf}) == []
    # A list given only as background (no step tells the answer to name it) is not checked.
    assert missing_list_items("nothing", "1. say that the shelf holds Dune (b1), Emma (b2), "
                              "Ivanhoe (b3)", {"shelf": shelf}) == []


def test_answer_only_answer_appends_rows_left_out(monkeypatch):
    monkeypatch.setattr("fastworkflow.skill_phases.write_answer", lambda query: "Loans: l1.")
    book = BookFacts(uid="b1", loan_uids=["l1", "l2"])
    # "list ...: facts" is no verbatim step (only say / compare / present are), so the model writes it.
    assert answer_only_answer("q", "1. list the loans of b1: l1, l2",
                              {"book": book}) == \
        f"Loans: l1.\n\n{MISSING_ITEMS_LEAD} l2."


# ----------------------------------------------------------------------
# Code writes an answer-only phase whose every step is verbatim or a guard (fix-hia1.30)
# ----------------------------------------------------------------------

from fastworkflow.skill_phases import answer_by_code, answer_step_kind  # noqa: E402


def test_answer_steps_are_verbatim_guard_or_judgement():
    assert answer_step_kind("1. say that b1 has loans l1, l2") == "verbatim"
    assert answer_step_kind("say what Dune holds: l1, l2") == "verbatim"
    assert answer_step_kind("compare the two books: Dune has 2 loans") == "verbatim"
    assert answer_step_kind("present the renewals as options only, and say that none "
                            "was applied: renew l1") == "verbatim"
    assert answer_step_kind("2. apply no change") == "guard"
    assert answer_step_kind("present them as options only; apply none of them") == "guard"
    assert answer_step_kind("say which book is riskier") == "judgement"
    assert answer_step_kind("say what Dune holds:") == "judgement"      # no facts given
    assert answer_step_kind("list the loans of b1: l1, l2") == "judgement"
    assert answer_step_kind("b1 has loans l1") == "judgement"


def test_answer_from_steps_writes_each_verbatim_step_as_a_line():
    steps = ("1. say that b1 has loans l1, l2\n"
             "2. say what Dune holds: l1, l2\n"
             "3. present the renewals on offer as options only, and say that none was "
             "applied: renew l1, renew l2\n"
             "4. compare the two books: Dune has 2 loans; Emma has 1\n"
             "5. apply no change\n"
             "6. present them as options only; apply none of them")
    assert answer_by_code(steps)
    assert answer_from_steps(steps) == (
        "B1 has loans l1, l2.\n"
        "What Dune holds: l1, l2.\n"
        "None was applied. Renew l1, renew l2.\n"
        "Dune has 2 loans; Emma has 1.")
    # A guard alone gives nothing to write; one judgement step leaves it to the model.
    assert not answer_by_code("1. apply no change")
    assert not answer_by_code("1. say that b1 has loans l1\n2. say which loan is due first")


def test_answer_only_answer_makes_no_model_call_when_every_step_is_verbatim(monkeypatch):
    calls = []
    monkeypatch.setattr("fastworkflow.skill_phases.write_answer",
                        lambda query: calls.append(query) or "Model text.")
    book = BookFacts(uid="b1", loan_uids=["l1", "l2"])
    steps = "1. say what b1 has on loan: l1, l2\n2. apply no change"
    assert answer_only_answer("q", steps, {"book": book}) == "What b1 has on loan: l1, l2."
    assert calls == []
    # The appended rows are still checked: a guard that names a list the answer leaves out.
    steps = "1. say that b1 has loans l1\n2. never drop a loan (l1, l2), name each"
    assert answer_only_answer("q", steps, {"book": book}) == \
        f"B1 has loans l1.\n\n{MISSING_ITEMS_LEAD} l2."
    assert calls == []
    # A judgement step keeps the model call (and the append).
    steps = "1. say that b1 has loans and name each: l1, l2\n2. say which loan is due first"
    assert answer_only_answer("q", steps, {"book": book}) == \
        f"Model text.\n\n{MISSING_ITEMS_LEAD} l1, l2."
    assert len(calls) == 1


def test_bulleted_answer_splits_only_runs_of_output_values():
    from fastworkflow.skill_phases import bulleted_answer

    class Shelf(BaseModel):
        uids: list[str] = []
        labels: list[str] = []

    shelf = Shelf(uids=["b1", "b2", "b3", "b4"], labels=["Dune", "Emma", "Ivanhoe", "Kim"])
    answer = ("B1 holds 4 books (Dune, Emma, Ivanhoe, Kim); 2 loans (l1, l2).\n"
              "Tagged: Dune [b1], Emma [b2], Ivanhoe [b3], Kim [b4].\n"
              "Readers: Dune, Emma, Ivanhoe, Bob.\n"
              "Three: Dune, Emma, Ivanhoe.")
    assert bulleted_answer(answer, {"shelf": shelf}) == (
        "- B1 holds 4 books:\n  - Dune\n  - Emma\n  - Ivanhoe\n  - Kim\n\n  2 loans (l1, l2).\n"
        "- Tagged:\n  - Dune [b1]\n  - Emma [b2]\n  - Ivanhoe [b3]\n  - Kim [b4]\n"
        "- Readers: Dune, Emma, Ivanhoe, Bob.\n"     # Bob is no output value: as it was
        "- Three: Dune, Emma, Ivanhoe.")             # fewer than four items: as it was
    # One line is no bullet; two runs in it, then the text after them.
    assert bulleted_answer("Books (Dune, Emma, Ivanhoe, Kim); copies (b1, b2, b3, b4), "
                           "all on the shelf.", {"shelves": [shelf]}) == (
        "Books:\n- Dune\n- Emma\n- Ivanhoe\n- Kim\n\ncopies:\n- b1\n- b2\n- b3\n- b4\n\n"
        "all on the shelf.")
    assert bulleted_answer("B1 has loans l1, l2.", {"shelf": shelf}) == "B1 has loans l1, l2."


def test_answer_only_answer_bullets_a_code_written_answer_after_its_checks(monkeypatch):
    monkeypatch.setattr("fastworkflow.skill_phases.write_answer", lambda query: "a, b, c, d")
    book = BookFacts(uid="b1", loan_uids=["l1", "l2", "l3", "l4"])
    steps = ("1. say what b1 has on loan, naming each: l1, l2, l3, l4\n"
             "2. say that it is due back\n3. apply no change")
    assert answer_only_answer("q", steps, {"book": book}) == (
        "- What b1 has on loan, naming each:\n  - l1\n  - l2\n  - l3\n  - l4\n"
        "- It is due back.")
    # A model-written answer is left as the model wrote it.
    assert answer_only_answer("q", "1. say which loan is due first", {"book": book}) == \
        "a, b, c, d"


def test_code_writes_an_answer_only_phase_and_its_span_says_so(phased, monkeypatch):
    # The fixture scripts the model's answers; here code writes this one.
    monkeypatch.setattr("fastworkflow.skill_phases.answer_by_code", answer_by_code)
    phased.start()
    result = phased.ctx.process_turn("show the book Dune")
    assert result.answer == "B1 has loans l1, l2."
    assert phased.answered == []                       # no model call wrote it
    spans = phased.sink.named(tracing.SPAN_SKILL_PHASE)
    report = spans[-1]
    assert report.attributes["answer_only"] is True
    assert report.attributes["answer_by_code"] is True
    assert all("answer_by_code" not in span.attributes for span in spans[:-1])


# ----------------------------------------------------------------------
# Stage-10 follow-ups: leaked framing, long background lists, extra command steps
# ----------------------------------------------------------------------

from fastworkflow.skill_phases import (  # noqa: E402
    EXTRA_STEPS_PHASE,
    STEPS_BEFORE_PHASE,
    capped_steps,
    extra_steps_phase,
    facts_text,
    phase_query,
    steps_before_phase,
)


def test_clean_answer_strips_echoed_framing_and_template_leftovers():
    # 4.01 R3: the reply copied its framing; 4.03 R2: a template placeholder line;
    # 3.07 R1: the completion marker's name alone on the last line.
    assert clean_answer("None was applied. This is part 6 of 6 (answer).") == "None was applied."
    assert clean_answer("- identity/enable\nInput was: {final_answer}") == "- identity/enable"
    assert clean_answer("people:\n  1: Ann\ncompleted") == "people:\n  1: Ann"
    assert clean_answer("Done.\n## completed ##\n") == "Done."
    # Every sentence of the framing phase_query writes is stripped when echoed.
    query = phase_query("show the book", "1. say it", index=2, count=3, name="report",
                        answer_only=True, item=(0, 2, "b1"))
    echoed = clean_answer("Dune is on loan.\n" + query.split("\n\n", 1)[1].split("\n1.")[0])
    assert echoed == "Dune is on loan."
    middle = phase_query("q", "1. find_book query=Dune", index=0, count=3, name="find")
    assert clean_answer("found b1. " + middle.split("\n")[0]) == "found b1."
    # Ordinary text, a bullet that says completed, and braces in prose stay.
    kept = "Status: completed\n- completed\nThe set {a, b} is complete."
    assert clean_answer(kept) == kept


def test_capped_steps_cuts_a_long_background_list_and_never_a_list_to_name():
    class Risk(BaseModel):
        score: float
        finding_labels: list[str]
        loan_uids: list[str] = []

    labels = [f"Finding {k}" for k in range(40)]
    risk = Risk(score=1.0, finding_labels=labels)
    plan = "1. say that the risk score is 1; 40 findings: " + "; ".join(labels)
    capped = capped_steps(plan, {"risk": risk}, max_items=3)
    assert capped == ("1. say that the risk score is 1; 40 findings: Finding 0; Finding 1; "
                      "Finding 2 (and 37 more; 40 in all)")
    # A list the steps tell the answer to name is given whole (the append rule's lists).
    named = "1. name every finding: " + ", ".join(labels)
    assert capped_steps(named, {"risk": risk}, max_items=3) == named
    # A naming word inside the items themselves does not make the step a naming step.
    listed = [f"Accounts listed twice {k}" for k in range(5)]
    plan = "1. say that it has 5 findings: " + ", ".join(listed)
    assert "(and 3 more; 5 in all)" in capped_steps(
        plan, {"risk": Risk(score=1.0, finding_labels=listed)}, max_items=2)
    # Short lists and lists written in another form are left as they are.
    assert capped_steps("1. say l1, l2", {"risk": Risk(score=1, finding_labels=["l1", "l2"])},
                        max_items=3) == "1. say l1, l2"


def test_facts_text_caps_lists_only_when_asked():
    book = BookFacts(uid="b1", loan_uids=[f"l{k}" for k in range(6)])
    assert facts_text(book) == "uid: b1; loan_uids: l0, l1, l2, l3, l4, l5"
    assert facts_text(book, max_items=2) == "uid: b1; loan_uids: l0, l1 (and 4 more; 6 in all)"


def test_extra_command_steps_move_out_of_an_answer_only_last_phase():
    phases = [Phase("find", lambda: find_steps("Dune"), output=Found),
              Phase("report", lambda: ["say that b1 has loans l1, l2"])]
    extra, kept = extra_steps_phase(phases, ["count_notes on b1", "say how many notes"],
                                    commands=COMMANDS, core_commands=CORE)
    assert extra.name == EXTRA_STEPS_PHASE and extra.steps_fn() == ["count_notes on b1"]
    assert kept == ["say how many notes"]
    # No command among them, or a last phase that runs commands itself: nothing moves.
    assert extra_steps_phase(phases, ["say it"], commands=COMMANDS, core_commands=CORE) == \
        (None, ["say it"])
    runs = phases[:1] + [Phase("report", lambda: ["list_loans", "say them"])]
    assert extra_steps_phase(runs, ["count_notes on b1"], commands=COMMANDS,
                             core_commands=CORE) == (None, ["count_notes on b1"])
    # A name already taken gets a suffix.
    taken = [Phase(EXTRA_STEPS_PHASE, lambda: ["find_book query=x"])] + phases
    assert extra_steps_phase(taken, ["count_notes"], commands=COMMANDS,
                             core_commands=CORE)[0].name == f"{EXTRA_STEPS_PHASE}_2"


def test_a_planner_command_step_runs_before_the_answer_only_last_phase(phased, monkeypatch):
    # 3.06 R2/R3: the planner's extra open_portrait step landed in the answer phase, which
    # then ran as an executor with no list rules. It now runs in its own phase before it.
    phased.steps_after = ["count_notes on b1", "say how many notes it has"]
    phased.script["extra_steps"] = dict(
        runs=[_run("count_notes", {"total": 3}, context=BOOK)], record=BOOK,
        answer="b1 has 3 notes.")
    # The workflow's commands, as the routing definition would list them.
    monkeypatch.setattr(phased.ctx, "_phase_command_names",
                        lambda: {"commands": list(COMMANDS), "core_commands": list(CORE)})
    executor = phased.start()
    result = phased.ctx.process_turn("show the book Dune")

    queries = [call["user_query"] for call in executor.calls]
    assert len(queries) == 3
    assert "part 3 of 4 (extra_steps)" in queries[2] and "count_notes on b1" in queries[2]
    assert "show the book Dune" not in queries[2]
    # The last phase is still answer-only and keeps the step that names no command.
    assert len(phased.answered) == 1 and "part 4 of 4 (report)" in phased.answered[0]
    assert "say how many notes it has" in phased.answered[0]
    assert "count_notes" not in phased.answered[0]
    spans = phased.sink.named(tracing.SPAN_SKILL_PHASE)
    assert [s.attributes["phase"] for s in spans] == ["find", "book", "extra_steps", "report"]
    assert spans[-1].attributes["answer_only"] is True
    assert result.answer == "Dune (b1) is on loan: l1, l2.\n\nb1 has 3 notes."


def test_the_extra_steps_phase_is_rebuilt_after_a_restore(phased, monkeypatch):
    phased.steps_after = ["count_notes on b1"]
    phased.script["extra_steps"] = dict(
        runs=[_run("count_notes", {"total": 3}, context=BOOK)], record=BOOK,
        answer="b1 has 3 notes.")
    monkeypatch.setattr(phased.ctx, "_phase_command_names",
                        lambda: {"commands": list(COMMANDS), "core_commands": list(CORE)})
    monkeypatch.setattr("fastworkflow.workflow_agent.load_skill_commands",
                        lambda folder: {"show_book": phased.command})
    phased.start(suspend_in="extra_steps")
    first = phased.ctx.process_turn("show the book Dune")
    assert first.status == TurnStatus.AWAITING_USER
    state = phased.ctx._phase_run
    assert state["phases"] == ["find", "book", "extra_steps", "report"]
    assert state["extra_steps"] == ["count_notes on b1"] and state["steps_after"] == []
    phased.ctx._phase_live = None  # as after a restore: rebuilt from the skill and the state
    second = phased.ctx.process_turn("go on")
    assert second.status == TurnStatus.COMPLETED
    assert second.answer == "Dune (b1) is on loan: l1, l2.\n\nb1 has 3 notes."


# ----------------------------------------------------------------------
# The planner's steps_before run in a phase of their own, before the first phase
# ----------------------------------------------------------------------

def test_steps_before_get_a_phase_of_their_own():
    phases = [Phase("find", lambda: find_steps("Dune"), output=Found)]
    assert steps_before_phase(phases, []) is None
    assert steps_before_phase(phases, ["  "]) is None
    before = steps_before_phase(phases, ["count_notes on b1"])
    assert before.name == STEPS_BEFORE_PHASE and before.steps_fn() == ["count_notes on b1"]
    assert before.output is None and not before.fans_out
    # A name already taken gets a suffix.
    taken = [Phase(STEPS_BEFORE_PHASE, lambda: ["find_book query=x"])] + phases
    assert steps_before_phase(taken, ["count_notes"]).name == f"{STEPS_BEFORE_PHASE}_2"


def _with_steps_before(phased, monkeypatch):
    phased.steps_before = ["count_notes on b1"]
    phased.script[STEPS_BEFORE_PHASE] = dict(
        runs=[_run("count_notes", {"total": 3}, context=BOOK)], record=BOOK,
        answer="b1 has 3 notes.")
    monkeypatch.setattr(phased.ctx, "_phase_command_names",
                        lambda: {"commands": list(COMMANDS), "core_commands": list(CORE)})


def test_steps_before_run_first_then_the_first_phase_runs_as_without_them(phased, monkeypatch):
    _with_steps_before(phased, monkeypatch)
    executor = phased.start()
    result = phased.ctx.process_turn("show the book Dune")

    assert result.status == TurnStatus.COMPLETED
    queries = [call["user_query"] for call in executor.calls]
    assert len(queries) == 3
    assert f"part 1 of 4 ({STEPS_BEFORE_PHASE})" in queries[0]
    assert "count_notes on b1" in queries[0] and "find_book" not in queries[0]
    # The first phase gets only its own steps, numbered from 1, as with no steps_before.
    assert "part 2 of 4 (find)" in queries[1] and "count_notes" not in queries[1]
    assert executor.plans[1].startswith("1. ")
    spans = phased.sink.named(tracing.SPAN_SKILL_PHASE)
    assert [s.attributes["phase"] for s in spans] == [STEPS_BEFORE_PHASE, "find", "book",
                                                      "report"]
    # The steps_before phase fills nothing; the first phase still fills its facts, from
    # its own runs, and later phases bind them.
    assert spans[0].attributes["facts"] is None
    assert spans[1].attributes["facts"] == {"book_uids": ["b1"]}
    assert "open_book_by_uid book_uid=b1" in queries[2]
    # Its run's steps continue the turn's numbering: the first phase starts after it.
    assert spans[1].attributes["first_step"] > spans[0].attributes["first_step"]
    # Their report comes first in the answer, as the request asked for them first.
    assert result.answer == "b1 has 3 notes.\n\nDune (b1) is on loan: l1, l2."


def test_the_steps_before_phase_is_rebuilt_after_a_restore(phased, monkeypatch):
    _with_steps_before(phased, monkeypatch)
    monkeypatch.setattr("fastworkflow.workflow_agent.load_skill_commands",
                        lambda folder: {"show_book": phased.command})
    phased.start(suspend_in=STEPS_BEFORE_PHASE)
    first = phased.ctx.process_turn("show the book Dune")
    assert first.status == TurnStatus.AWAITING_USER
    state = phased.ctx._phase_run
    assert state["phases"] == [STEPS_BEFORE_PHASE, "find", "book", "report"]
    assert state["before_phase"] == STEPS_BEFORE_PHASE
    phased.ctx._phase_live = None  # as after a restore: rebuilt from the skill and the state
    second = phased.ctx.process_turn("go on")
    assert second.status == TurnStatus.COMPLETED
    assert second.answer == "b1 has 3 notes.\n\nDune (b1) is on loan: l1, l2."
