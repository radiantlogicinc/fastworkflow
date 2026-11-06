"""fix-fixv: a command that declares ``writes = True`` runs inside a planning command's phase only when a
step of that phase names it.

Each case runs the library's real planning commands on a copy of the library, through the real phase
runner and the executor's own dispatch (``real_dispatch``: only the extraction hop is scripted, as the
other library tests script it); the executor's model is scripted (``Model``), the accepted LLM boundary.
"""
from __future__ import annotations

import os

import pytest

import fastworkflow
from fastworkflow import command_writes, tracing
from fastworkflow.command_routing import RoutingRegistry
from fastworkflow.turn import TurnStatus
from fastworkflow.workflow_agent import _what_can_i_do
from tests.test_turn_result_capture import no_review_gate  # noqa: F401 - the review gate is off for these turns
from tests.test_step_helpers import (
    LIBRARY,
    _library_workflow,
    _make_library_runner_env,
    _real_planning_command,
)
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401 - shared fixture

AUTHOR_BOOKS = "Library/author_books"
NOTE_MEMBER = "Library/note_member"
WRITES_FILES = (
    "_commands/Library/author_books.py",
    "_commands/Library/note_member.py",
    "_commands/Library/_wording.py",
)


@pytest.fixture
def writes_env(tmp_path, initialized_fastworkflow, monkeypatch):
    state = _make_library_runner_env(
        tmp_path, initialized_fastworkflow, monkeypatch,
        extra_commands=WRITES_FILES, real_dispatch=True)
    yield state
    state.ctx.pop_active_workflow()


def _library(state):
    return state.wf.root_command_context


def _notes(state) -> dict[str, list[str]]:
    library = _library(state)
    return {**{uid: list(book.notes) for uid, book in library.books.items()},
            **{uid: list(member.notes) for uid, member in library.members.items()}}


def _outputs(state, name: str):
    return [o for o in state.ctx._turn_outputs if (o.command_name or "").split("/")[-1] == name]


def test_the_library_declares_add_note_as_its_one_write_and_note_member_writes_through_its_steps(
        initialized_fastworkflow):
    # add_note declares writes = True; note_member's steps run add_note, so it writes too; author_books
    # (find_book, open_book) reaches no write: it is read-only.
    RoutingRegistry.clear_registry()
    writes = command_writes.write_commands(LIBRARY)
    assert writes == {"Record/add_note", NOTE_MEMBER}
    assert AUTHOR_BOOKS not in writes


def test_a_read_only_planning_command_refuses_a_note_its_phase_does_not_name(writes_env):
    # The recorded failure (fix-48np.1, L2): inside author_books, the phase agent added "the cover is
    # torn" to a book. Here the agent finishes the find phase, then, on a book its book phase opened,
    # adds a note no step names. Without the declaration the note lands on the book (Persuasion, b4).
    before = _notes(writes_env)
    writes_env.model.finish()
    writes_env.model.run('add_note text="the spine is cracked"')
    writes_env.model.finish()
    writes_env.planning_command = _real_planning_command(AUTHOR_BOOKS, "_commands/Library/author_books.py").command
    writes_env.args = {"author": "Austen"}

    result = writes_env.ctx.process_turn("show every Austen book in the library")

    assert result.status == TurnStatus.COMPLETED
    assert _notes(writes_env) == before            # no record changed
    [refused] = _outputs(writes_env, "add_note")
    assert refused.success is False
    assert "changes the application's data" in refused.command_response.response
    assert writes_env.model.asked, "the agent was asked for a step, and tried the write"
    # the books were still opened, by the phase's own steps
    assert [o.command_response.output.book_uid for o in _outputs(writes_env, "open_book")] == ["b3", "b4"]


def test_a_planning_command_whose_step_names_the_write_runs_it(writes_env):
    # note_member's phase names add_note: the guard lets it run, once, on the member's record.
    writes_env.planning_command = _real_planning_command(NOTE_MEMBER, "_commands/Library/note_member.py").command
    writes_env.args = {"member": "Raj Patel", "text": "he paid his fine"}

    result = writes_env.ctx.process_turn("note on Raj Patel's record that he paid his fine")

    assert result.status == TurnStatus.COMPLETED
    assert _library(writes_env).members["m2"].notes == ["prefers email reminders", "he paid his fine"]
    [added] = _outputs(writes_env, "add_note")
    assert added.success is True


def test_the_listing_text_of_a_read_only_phase_has_no_add_note(writes_env):
    # The listing itself, read with a phase's steps in place: add_note is gone from the Book listing.
    library = _library(writes_env)
    writes_env.wf.current_command_context = library.books["b3"]
    assert "add_note" in _what_can_i_do(writes_env.ctx)
    writes_env.ctx._phase_run = {"plan": "1. open_book book_uid=b3"}
    try:
        assert "add_note" not in _what_can_i_do(writes_env.ctx)
        assert "show_notes" in _what_can_i_do(writes_env.ctx)
        writes_env.ctx._phase_run = {"plan": "1. add_note text=hi"}
        assert "add_note" in _what_can_i_do(writes_env.ctx)
    finally:
        writes_env.ctx._phase_run = None


def test_writes_must_be_a_bool_on_an_execution_command(tmp_path):
    RoutingRegistry.clear_registry()
    path = os.path.join(LIBRARY, "_commands", "Record", "add_note.py")
    text = open(path, encoding="utf-8").read().replace("writes = True", 'writes = "yes"')
    folder = _library_workflow(tmp_path, command_texts={"_commands/Record/add_note.py": text})
    with pytest.raises(command_writes.WritesDeclarationError, match=r"Record/add_note's writes must be True or False"):
        RoutingRegistry.get_definition(folder)
    RoutingRegistry.clear_registry()


def test_a_planning_command_may_not_declare_writes(tmp_path):
    RoutingRegistry.clear_registry()
    path = os.path.join(LIBRARY, "_commands", "Library", "note_member.py")
    text = open(path, encoding="utf-8").read().replace(
        "class PlanGenerator:\n", "class PlanGenerator:\n    writes = True\n\n")
    folder = _library_workflow(tmp_path, command_texts={"_commands/Library/note_member.py": text},
                               extra_commands=WRITES_FILES)
    with pytest.raises(command_writes.WritesDeclarationError, match="note_member is a planning command"):
        RoutingRegistry.get_definition(folder)
    RoutingRegistry.clear_registry()
