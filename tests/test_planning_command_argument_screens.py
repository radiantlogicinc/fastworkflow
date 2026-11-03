"""The free-text screens of a skill call's arguments, and the wording of the question they produce.

Real components throughout: the tutorial workflow's skills (copied to a temporary folder, so the
repo's own ``___command_info`` is never read for writing or touched), one skill added to that copy
with two free-text inputs and a declared choice, the real ``check_arguments`` and ``_validate_arguments``,
and the real ``refusal_question``. No planner model is called.
"""
import shutil
from pathlib import Path

import pytest

from fastworkflow import planning_commands as sc
from fastworkflow import workflow_agent

TUTORIAL = Path(__file__).resolve().parent / "planning_tutorial_workflow"

#: A skill with two required free-text inputs (``member``, ``branch``), an optional one (``note``)
#: and a declared choice (``entity_type``). Written into the temporary copy only.
TRANSFER_LOAN = '''
from typing import Literal
from pydantic import BaseModel, Field


class Signature:
    """Move one member's loan of a book to a branch."""

    class Input(BaseModel):
        member: str = Field(description="the member whose loan moves")
        branch: str = Field(description="the branch the loan moves to")
        entity_type: Literal["identity", "account"] = Field(
            default="identity", description="the kind of record the request names")
        note: str = Field(default="", description="a note on the move")


class PlanGenerator:
    def steps(self, input):
        return [f"find the member {input.member}", f"move the loan to the {input.branch} branch"]
'''

AUTHOR_QUESTION = ("I can't run author books yet: your request does not say which author to use. "
                   "Which author should I use?")


@pytest.fixture
def workflow(tmp_path):
    """A temporary copy of the tutorial workflow with the ``transfer_loan`` skill added."""
    copy = tmp_path / "tutorial"
    shutil.copytree(TUTORIAL, copy, ignore=shutil.ignore_patterns("___command_info", "__pycache__", "versions"))
    (copy / "_commands" / "transfer_loan.py").write_text(TRANSFER_LOAN, encoding="utf-8")
    return str(copy)


def _planning_command(folder: str, name: str) -> sc.PlanningCommandCommand:
    return workflow_agent.load_planning_command_map(folder)[name]


def _raw(command: sc.PlanningCommandCommand, **values) -> dict:
    """The planner's raw values for every param of the skill: the given ones, None for the rest."""
    return {p.name: values.get(p.name) for p in command.params}


def _refused(command: sc.PlanningCommandCommand, values: dict, request: str) -> sc.PlanningCommandInputError:
    with pytest.raises(sc.PlanningCommandInputError) as refused:
        sc.check_arguments(command, _raw(command, **values), request)
    return refused.value


# ---------------------------------------------------------------- a placeholder is never shown back

@pytest.mark.parametrize("value", ["{AUTHOR_PLACEHOLDER}", "<entity_name>", "NAME_1", "unspecified_right",
                                   "John Doe"], ids=["braces", "angle", "name-number", "unspecified", "invented"])
def test_a_placeholder_is_refused_and_the_question_asks_plainly(workflow, value):
    command = _planning_command(workflow, "author_books")
    refusal = _refused(command, {"author": value}, "Which books did the author write?")

    question = sc.refusal_question(command, refusal)
    assert question == AUTHOR_QUESTION
    assert value not in question


def test_an_optional_input_given_a_placeholder_takes_its_default(workflow):
    command = _planning_command(workflow, "transfer_loan")
    args, notes = sc.check_arguments(command, _raw(command, member="Ann Lee", branch="Main",
                                                   note="<entity_name>"), "Move Ann Lee's loan to the Main branch")

    assert args["note"] == ""
    assert any(note.startswith("note=") and note.endswith(": default") for note in notes)


# ---------------------------------------------------------------- a pronoun

@pytest.mark.parametrize("value", ["that one", "it", "her"])
def test_a_pronoun_with_no_single_subject_is_refused(workflow, value):
    command = _planning_command(workflow, "author_books")
    refusal = _refused(command, {"author": value}, "Which books did the author of that one write?")

    assert sc.refusal_question(command, refusal) == AUTHOR_QUESTION


def test_a_pronoun_with_one_subject_takes_that_subject(workflow):
    command = _planning_command(workflow, "author_books")
    args, notes = sc.check_arguments(command, _raw(command, author="it"),
                                     "What do you have by Frank Herbert, and how many copies does it have?")

    assert args["author"] == "Frank Herbert"
    assert any("the request's one subject" in note for note in notes)


# ---------------------------------------------------------------- a word of the workflow's vocabulary

@pytest.mark.parametrize("planning_command, values, asked_in, required", [
    # the input's own name
    ("author_books", {"author": "author"}, "Which books did the author write?", "author"),
    # a field name of another skill's input
    ("find_books", {"query": "title"}, "Which titles by Asimov?", "query"),
    # a word of a command name ("transfer_loan" gives "loan")
    ("transfer_loan", {"member": "loan", "branch": "Main"}, "Move the loan to the Main branch", "member"),
    # a declared choice of another input ("kind" is Literal book/member on find_record)
    ("transfer_loan", {"member": "Ann Lee", "branch": "account"}, "Move Ann Lee's loan to the account branch",
     "branch"),
    # a kind of record written in the request itself
    ("find_books", {"query": "book"}, "Which book by Asimov?", "query"),
], ids=["own-name", "field-name", "command-word", "declared-choice", "kind-word"])
def test_a_vocabulary_word_is_refused_as_an_unwritten_value(workflow, planning_command, values, asked_in, required):
    command = _planning_command(workflow, planning_command)
    refusal = _refused(command, values, asked_in)

    assert sc.refusal_question(command, refusal) == (
        f"I can't run {planning_command.replace('_', ' ')} yet: your request does not say which {required} to use. "
        f"Which {required} should I use?")


def test_a_vocabulary_word_given_to_an_optional_input_takes_its_default(workflow):
    command = _planning_command(workflow, "transfer_loan")
    args, notes = sc.check_arguments(command, _raw(command, member="Ann Lee", branch="Main", note="note"),
                                     "Move Ann Lee's loan to the Main branch")

    assert args["note"] == ""
    assert any(note.startswith("note=") and note.endswith(": default") for note in notes)


# ---------------------------------------------------------------- declared choices and values the request names

def test_a_declared_choice_is_never_refused(workflow):
    command = _planning_command(workflow, "transfer_loan")
    for choice in ("identity", "account"):
        args, _ = sc.check_arguments(command, _raw(command, member="Ann Lee", branch="Main", entity_type=choice),
                                     "Move Ann Lee's loan to the Main branch")
        assert args["entity_type"] == choice


def test_a_declared_choice_that_is_also_a_word_is_never_refused(workflow):
    command = _planning_command(workflow, "find_record")
    args, _ = sc.check_arguments(command, _raw(command, name="Dune", kind="book"), "Is there a book called Dune?")

    assert args == {"name": "Dune", "kind": "book"}


@pytest.mark.parametrize("planning_command, values, asked_in", [
    ("find_books", {"query": "Dune"}, "Which books called Dune?"),
    ("author_books", {"author": "Frank Herbert"}, "What do you have by Frank Herbert?"),
    ("transfer_loan", {"member": "Ann Lee", "branch": "Main"}, "Move Ann Lee's loan to the Main branch"),
], ids=["title", "person", "member-and-branch"])
def test_a_value_the_request_names_in_its_role_passes(workflow, planning_command, values, asked_in):
    command = _planning_command(workflow, planning_command)
    args, _ = sc.check_arguments(command, _raw(command, **values), asked_in)

    assert {name: args[name] for name in values} == values


# ---------------------------------------------------------------- the same value for two inputs

def test_the_same_value_for_two_inputs_is_refused_for_the_role_the_request_does_not_write(workflow):
    command = _planning_command(workflow, "transfer_loan")
    args, _ = sc.check_arguments(command, _raw(command, member="Ann Lee", branch="Ann Lee"),
                                 "Move the loan of member Ann Lee to branch Ann Lee")

    with pytest.raises(sc.PlanningCommandInputError) as refused:
        sc._validate_arguments(command, args, "Move the loan of member Ann Lee to branch Ann Lee", None)

    assert str(refused.value) == "branch='Ann Lee' is also the value of member"
    assert sc.refusal_question(command, refused.value) == (
        "I can't run transfer loan yet: 'Ann Lee' is the member in your request, not the branch. "
        "Which branch should I use?")


def test_the_same_value_for_two_inputs_refuses_the_first_when_neither_role_is_written(workflow):
    command = _planning_command(workflow, "transfer_loan")
    request = "Ann Lee has a loan at Ann Lee"
    args, _ = sc.check_arguments(command, _raw(command, member="Ann Lee", branch="Ann Lee"), request)

    with pytest.raises(sc.PlanningCommandInputError) as refused:
        sc._validate_arguments(command, args, request, None)

    assert str(refused.value) == "member='Ann Lee' is also the value of branch"
