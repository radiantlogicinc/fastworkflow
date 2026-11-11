"""fix-1744 and fix-46mm: the library's planning commands declare their inputs so that the shared extractor's
checks keep a drafted (or typed-call) value right.

- ``note_member.text`` (fix-46mm) is ``in_request``: the note is the request's own words, so an invented
  note is refused; and its ``Input`` refuses a note that holds the member's name, which is the request
  around the note ("Note on Mia Chen's record that ..."), not the note.
- ``overdue_summary.branch`` (fix-1744) is an optional input: a question about one branch binds it, words
  that name every branch leave it empty, and a name that is no branch is refused.

Each answer below is one recorded in the fix-48np.1 drafting runs (the corpus requests L3, L8, L9, L10,
L18, T6). Integration: a copy of the library example (``annotated_library``), its real routing, the real
DSPy extractor (``parameter_extraction.extract_arguments``, the path a draft and the typed call share) and
the real checks; the model boundary is a scripted ``DummyLM``. No test calls a model.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

import fastworkflow
from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import extract_arguments
from fastworkflow.planning_commands import call_phases, planning_command
from fastworkflow.planning_phases import phase_steps
from tests.test_approval_binding import annotated_library  # noqa: F401 - the library copy
from tests.test_drafted_subtask_inputs import _framework, scripted  # noqa: F401 - fixtures

NOTE_MEMBER = "Library/note_member"
OVERDUE_SUMMARY = "Library/overdue_summary"
NOTE_START = "(A first extraction from this command was refused:"


@pytest.fixture
def library(annotated_library):  # noqa: F811
    folder = str(annotated_library)
    fastworkflow.RoutingRegistry.get_definition(folder, load_cached=False)
    return fastworkflow.Workflow.create(folder, workflow_id_str=f"decl_{uuid.uuid4().hex[:8]}")


# ---------------------------------------------------------------- fix-46mm: the note is the user's words


def test_a_note_in_the_request_s_words_stands(library, scripted):
    lm = scripted({"reasoning": "", "member": "Raj Patel", "text": "he paid his fine"})

    result = extract_arguments(library, NOTE_MEMBER, "Add a note to Raj Patel's record saying he paid his fine.")

    assert result.valid and not result.retried
    assert (result.params.member, result.params.text) == ("Raj Patel", "he paid his fine")
    assert len(lm.prompts) == 1


def test_an_invented_note_is_refused_and_the_retry_takes_the_request_s_words(library, scripted):
    # r2 L10: the draft wrote "Requested extension for upcoming project deadline", which the request does
    # not write. in_request refuses it; the extractor is asked once more and copies the request.
    lm = scripted({"reasoning": "", "member": "Mia Chen", "text": "Requested extension for upcoming project deadline"},
                  {"reasoning": "", "member": "Mia Chen", "text": "she asked for an extension"})

    result = extract_arguments(library, NOTE_MEMBER, "Note on Mia Chen's record that she asked for an extension.")

    assert result.valid and result.retried
    assert result.params.text == "she asked for an extension"
    assert NOTE_START in lm.prompts[1] and "not written in the request" in lm.prompts[1]


@pytest.mark.parametrize("sub_task, text", [
    # r1/r2 L10: the note text carried the instruction
    ("Note on Mia Chen's record that she asked for an extension.",
     "Note on Mia Chen's record that she asked for an extension"),
    # r1 L18: the whole bundled request became the note
    ("Add a note to Ann Lee's record saying she moved to Riverside and report her loans.",
     "Add a note to Ann Lee's record saying she moved to Riverside and report her loans"),
])
def test_a_note_that_holds_the_request_around_it_is_refused(library, scripted, sub_task, text):
    member = "Mia Chen" if "Mia" in sub_task else "Ann Lee"
    lm = scripted({"reasoning": "", "member": member, "text": text},
                  {"reasoning": "", "member": member, "text": text})

    result = extract_arguments(library, NOTE_MEMBER, sub_task)

    assert not result.valid and result.retried
    assert "holds the member's name" in result.refusal
    assert "holds the member's name" in lm.prompts[1]   # the retry was told why


def test_the_note_field_s_rule_reaches_the_extractor(library, scripted):
    lm = scripted({"reasoning": "", "member": "Raj Patel", "text": "he paid his fine"})

    extract_arguments(library, NOTE_MEMBER, "Add a note to Raj Patel's record saying he paid his fine.")

    assert "(to note that someone called: 'someone called'). Copy it exactly as the request writes it." \
        in lm.prompts[0]


# ---------------------------------------------------------------- fix-1744: overdue_summary takes a branch


@pytest.mark.parametrize("sub_task, extracted, branch", [
    ("How many overdue loans does the Riverside branch have?", "Riverside branch", "Riverside"),   # L9
    ("how many overdue loans does Central have?", "Central", "Central"),                           # T6
    ("How many loans are overdue across all branches?", "all branches", ""),                        # L8
    ("Give me an overdue summary.", "", ""),                                                        # L12
])
def test_the_branch_a_question_names_binds_and_words_for_every_branch_leave_it_empty(
        library, scripted, sub_task, extracted, branch):
    scripted({"reasoning": "", "branch": extracted})

    result = extract_arguments(library, OVERDUE_SUMMARY, sub_task)

    assert result.valid, result.refusal
    assert result.params.branch == branch


def test_a_name_that_is_no_branch_is_refused_with_the_branches(library, scripted):
    scripted({"reasoning": "", "branch": "Northside"}, {"reasoning": "", "branch": "Northside"})

    result = extract_arguments(library, OVERDUE_SUMMARY, "how many loans are overdue at Northside?")

    assert not result.valid
    assert "not a branch of the library" in result.refusal and "Central, Riverside" in result.refusal


@pytest.mark.parametrize("branch, where", [("", "across the library"), ("Riverside", "at Riverside")])
def test_the_answer_step_names_the_branches_and_the_scope_it_counts(library, branch, where):
    # The last phase's agent writes the answer; its step carries the branches the first phase listed
    # (the recorded answers left them out, or wrote "{{list_branches}}").
    command = planning_command(library.folderpath, OVERDUE_SUMMARY)
    branches_phase, overdue_phase = call_phases(command, {"branch": branch})
    outputs = {"branches": SimpleNamespace(names=["Central", "Riverside"], total=2)}

    steps = phase_steps(overdue_phase, outputs, phase_names=frozenset({"branches", "overdue"}))

    assert steps[0] == ("count_overdue branch=Riverside" if branch else "count_overdue")
    assert steps[1] == f"name the branches (Central, Riverside) and the number of overdue loans {where}"
