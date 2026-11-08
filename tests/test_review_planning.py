"""The review planning commands (fix-3rsq.plan-review.1, stage 2): review_subtasks and review_plan as framework
planning commands in the global scope, the core review_proposal command, the app override, and the stage 3
round-1 builder.

Integration tests on a copy of the trained library (the approval fixtures). The agent is the real agent and
the planning steps run by code; the only scripted parts are the agent's first step and the user's replies,
which arrive as ordinary turns. No test calls a model.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow.command_directory import CommandDirectory
from fastworkflow import planning_commands as pc
from fastworkflow.review_engine import AWAITING, CONFIRMED, EXPIRED, REVISED, open_review
from fastworkflow.review_proposals import (Candidate, PlanProposal, SubTaskProposal, build_round1_proposal)
from fastworkflow.turn import TurnStatus
from tests.test_approval_binding import (  # noqa: F401 - the real agent on the library copy, and the guards
    _added, _ask, _finish, annotated_library, no_live_model, no_model_call_for_parameters, todo)
from tests.test_review_engine import _tasks  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

LIBRARY = Path(fastworkflow.__file__).resolve().parent / "examples" / "library"


# ----------------------------------------------------------------------
# Loading: framework planning commands, global scope, lowest precedence
# ----------------------------------------------------------------------


def test_the_framework_planning_commands_are_global_and_loaded_into_any_workflow():
    routing = fastworkflow.RoutingRegistry.get_definition(str(LIBRARY))
    directory = routing.command_directory.map_command_2_metadata
    assert "review_subtasks" in directory and "review_plan" in directory        # no context folder: global
    assert directory["review_subtasks"].is_planning and directory["review_plan"].is_planning
    assert directory["review_proposal"].framework and not directory["review_proposal"].is_planning
    assert "review_proposal" not in routing.command_directory.core_command_names
    assert "review_proposal" not in routing.command_directory.get_execution_commands()
    loaded = pc.resolve_field_sources(str(LIBRARY)).planning_commands
    assert "review_subtasks" in loaded and "review_plan" in loaded
    assert not [p for p in pc.resolve_field_sources(str(LIBRARY)).problems if "review" in (p.command or "")]


def test_an_app_command_of_the_same_name_overrides_the_framework_one(tmp_path):
    folder = tmp_path / "library_override"
    shutil.copytree(LIBRARY, folder, ignore=shutil.ignore_patterns("___workflow_contexts", "___convo_info",
                                                                  "__pycache__", "versions"), symlinks=True)
    (folder / "_commands" / "review_plan.py").write_text(
        '"""App review_plan: overrides the framework one."""\n'
        "from pydantic import BaseModel\n"
        "from fastworkflow.planning import run\n\n\n"
        "class Signature:\n"
        '    """App plan review."""\n'
        "    plain_utterances = ['app plan review']\n\n"
        "    class Input(BaseModel):\n"
        "        goals: str\n\n"
        "    class Output(BaseModel):\n"
        "        items: list[str] = []\n\n\n"
        "class PlanGenerator:\n"
        "    def app_only(self, input):\n"
        "        return [run('find_member', query=input.goals)]\n")
    directory = CommandDirectory.load(str(folder), hydrate_utterances=False)
    metadata = directory.map_command_2_metadata["review_plan"]
    assert Path(metadata.response_generation_module_path).resolve().parent == (folder / "_commands").resolve()
    # The framework's review_subtasks is still there: only the overridden name changed.
    assert Path(directory.map_command_2_metadata["review_subtasks"].response_generation_module_path).parent \
        == Path(fastworkflow.__file__).resolve().parent / "_framework_commands"


# ----------------------------------------------------------------------
# End to end: the agent calls the review, the user edits and accepts, the typed output is returned
# ----------------------------------------------------------------------


def test_review_subtasks_runs_end_to_end_with_scripted_replies_and_returns_the_accepted_proposal(todo):
    # The framework calls the planning command directly (not the agent's name resolution, which refuses it).
    from fastworkflow.utils.react import AskUserSuspend

    todo.wf.current_command_context = todo.library
    awaiting = todo.ctx.run_framework_planning_command(
        "review_subtasks", {"subtasks": "book a flight; tell Ann"}, "review the sub-tasks")
    assert awaiting.command_response.artifacts.get("awaiting_user") is True
    assert "1. book a flight" in awaiting.command_response.response
    record = todo.store.active()
    assert record is not None and record.kind == "sub_tasks" and record.state == AWAITING
    assert "1. book a flight" in record.rendering and "2. tell Ann" in record.rendering

    edited = todo.ctx.process_turn("change 2: inputs=Ann Lee")
    assert "2. tell Ann (Ann Lee)" in edited.answer
    assert record.state == REVISED

    todo.script.steps = _finish()                  # the agent's finish after the accepted step (scripted)
    done = todo.ctx.process_turn("accept")
    assert record.state == CONFIRMED
    assert done.status is TurnStatus.COMPLETED
    assert "Ann Lee" in (done.answer or "") or any(
        "Ann Lee" in (o.command_response.response or "") for o in done.command_outputs)


def test_review_plan_runs_end_to_end_with_scripted_replies_and_returns_the_accepted_plan(todo):
    # A plan is goals with bound steps (arch 19.0 invariant 1): the edit changes a goal's step, and the accepted
    # plan is the typed Output.
    from fastworkflow.utils.react import AskUserSuspend  # noqa: F401 - the framework call may raise it

    todo.wf.current_command_context = todo.library
    awaiting = todo.ctx.run_framework_planning_command(
        "review_plan", {"goals": "book a flight = find_member query=Ann | say booked"}, "review the plan")
    assert awaiting.command_response.artifacts.get("awaiting_user") is True
    record = todo.store.active()
    assert record is not None and record.kind == "plan" and record.state == AWAITING
    assert "1. book a flight: [find_member query=Ann, say booked]" in record.rendering

    edited = todo.ctx.process_turn("change 1: steps=say booked, notify Ann")
    assert "notify Ann" in edited.answer and record.state == REVISED

    todo.script.steps = _finish()
    done = todo.ctx.process_turn("accept")
    assert record.state == CONFIRMED and done.status is TurnStatus.COMPLETED
    assert any("notify Ann" in (o.command_response.response or "") for o in done.command_outputs)


# ----------------------------------------------------------------------
# The sub-task model (fix-3rsq.plan-review.2.4): goal, outcome, inputs, means; the JSON round trip
# ----------------------------------------------------------------------


def _sample_proposal():
    from fastworkflow.review_proposals import SubTask, SubTaskInput, Means

    return SubTaskProposal(items=[
        SubTask(goal="book a loan for Ann", outcome="Ann holds a loan on Dune",
                inputs=[SubTaskInput(name="title", value="Dune")],
                means=Means(kind="planning", name="Library/book_loans")),
        SubTask(goal="tell Ann", outcome="Ann has been told",
                inputs=[SubTaskInput(from_subtask=1)],
                means=Means(kind="plain", name="Library/notify")),
        SubTask(goal="summarise the week", outcome="a summary exists"),
    ])


def test_a_sub_task_round_trips_through_json_with_its_goal_outcome_inputs_and_means():
    """The review's accepted proposal is JSON (review_proposal.Output.accepted) and review_subtasks rebuilds each
    item with SubTask(**item): the round trip keeps every field, and the fingerprint does not move."""
    import json

    from fastworkflow.review_proposals import SubTask

    proposal = _sample_proposal()
    rebuilt = SubTaskProposal.model_validate_json(proposal.model_dump_json())
    assert rebuilt == proposal
    assert rebuilt.fingerprint() == proposal.fingerprint()
    assert SubTaskProposal.model_validate({"items": proposal.binding()["items"]}) == proposal
    accepted = json.loads(proposal.model_dump_json())["items"]
    assert [SubTask(**item) for item in accepted] == list(proposal.items)
    first = rebuilt.items[0]
    assert (first.goal, first.outcome) == ("book a loan for Ann", "Ann holds a loan on Dune")
    assert first.inputs[0].value == "Dune" and first.means.kind == "planning"
    assert rebuilt.items[1].inputs[0].from_subtask == 1
    assert rebuilt.items[2].means.kind == "from_scratch"


def test_the_rendering_shows_each_sub_task_its_goal_and_inputs():
    text = _sample_proposal().rendering()
    assert "1. book a loan for Ann (title: Dune)" in text
    assert "2. tell Ann (from sub-task 1)" in text
    assert "3. summarise the week" in text
    assert "means:" not in text and "desired outcome" not in text


def test_the_model_rendering_keeps_means_outcome_and_input_keys():
    text = _sample_proposal().model_rendering()
    assert "means: planning:Library/book_loans" in text
    assert "desired outcome: Ann holds a loan on Dune" in text


def test_the_text_forms_of_means_and_inputs_are_read_and_bad_ones_refused():
    from pydantic import ValidationError

    from fastworkflow.review_proposals import Means, SubTaskInput

    assert Means.from_text("from_scratch") == Means()
    assert Means.from_text("planning:Library/book_loans") == Means(kind="planning", name="Library/book_loans")
    assert Means.from_text(" plain:notify ") == Means(kind="plain", name="notify")
    for bad in ("book_loans", "planning:", "execute:x", ""):
        with pytest.raises(ValueError):
            Means.from_text(bad)
    with pytest.raises(ValidationError):
        Means(kind="plain")                       # a command kind names its command
    with pytest.raises(ValidationError):
        Means(kind="from_scratch", name="x")      # from_scratch names none
    assert SubTaskInput.from_text("title=Dune") == SubTaskInput(name="title", value="Dune")
    assert SubTaskInput.from_text("from sub-task 2") == SubTaskInput(from_subtask=2)
    assert SubTaskInput.from_text("who=from sub-task 3") == SubTaskInput(name="who", from_subtask=3)
    assert SubTaskInput.from_text("Ann Lee") == SubTaskInput(value="Ann Lee")
    with pytest.raises(ValidationError):
        SubTaskInput(value="x", from_subtask=1)   # a value or 'from sub-task N', not both
    with pytest.raises(ValidationError):
        SubTaskInput()                            # and not neither
    with pytest.raises(ValidationError):
        SubTaskInput(from_subtask=0)


def test_a_sub_task_has_no_completion_field_because_completion_comes_only_from_the_finish_check():
    """Arch 19.0 invariant 3: a goal is never completed by its command's success. The model carries no status that a
    command result could set; the finish check is the only completion (documented on SubTask)."""
    from fastworkflow.review_proposals import SubTask

    assert not {"done", "completed", "complete", "success", "status"} & set(SubTask.model_fields)
    assert "finish check" in (SubTask.__doc__ or "")
    assert "CommandOutput.success" in (SubTask.__doc__ or "")


def test_the_round_one_builder_prefers_a_planning_command_then_a_plain_one_then_from_scratch():
    from fastworkflow.review_proposals import Means, SubTaskInput

    candidates = {
        1: [Candidate("Library/send", is_planning=False, score=0.9),
            Candidate("Library/book_loans", is_planning=True, score=0.4)],   # a planning command in top-k wins
        2: [Candidate("Library/notify", is_planning=False, score=0.6)],       # the best plain command
        # sub-task 3 has no candidate: from scratch
    }
    proposal = build_round1_proposal(
        ["lend Dune to Ann", "tell Ann", "summarise"], candidates,
        inputs={2: [SubTaskInput(from_subtask=1)]}, outcomes={1: "Ann holds a loan on Dune"})
    assert proposal.items[0].means == Means(kind="planning", name="Library/book_loans")
    assert proposal.items[0].outcome == "Ann holds a loan on Dune"
    assert proposal.items[1].means == Means(kind="plain", name="Library/notify")
    assert proposal.items[1].inputs[0].from_subtask == 1
    assert proposal.items[2].means == Means() and proposal.items[2].outcome == ""


def test_a_proposal_review_accepts_more_than_two_replies_and_expires_at_its_cap(todo):
    """fix-3rsq.plan-review.2.5 / fix-kk46.4: eight unresolved edits fill the cap; accept on that turn is honoured;
    a further unresolved reply expires without applying."""
    store = todo.store
    store.records.clear()
    record = open_review(todo.ctx, store, _tasks("book a flight", "tell Ann"), surface=store.surface)
    assert record.ttl_turns == 8
    for number in range(8):
        todo.ctx.process_turn(f"change 2: count={number + 2}")
    assert record.turns_seen == 8 and record.state == REVISED
    todo.script.steps = _finish()
    todo.ctx.process_turn("accept")
    assert record.state == CONFIRMED and record.turns_seen == 8

    store.records.clear()
    record = open_review(todo.ctx, store, _tasks("book a flight", "tell Ann"), surface=store.surface)
    for number in range(8):
        todo.ctx.process_turn(f"change 2: count={number + 2}")
    assert record.turns_seen == 8
    todo.ctx.process_turn("change 2: count=99")
    assert record.state == EXPIRED and record.turns_seen == 8
    assert not [r for r in todo.store.records if r.state == CONFIRMED]


def test_a_proposal_review_with_three_replies_is_still_open_and_accepts(todo):
    todo.wf.current_command_context = todo.library
    todo.ctx.run_framework_planning_command("review_subtasks", {"subtasks": "book a flight; tell Ann"},
                                            "review the sub-tasks")
    record = todo.store.active()
    for text in ("change 2: inputs=query=Ann Lee", "change 1: inputs=title=Dune",
                 "change 2: inputs=query=Ann Lee, Dune"):
        todo.ctx.process_turn(text)
    assert record.turns_seen == 3 and record.state == REVISED
    todo.script.steps = _finish()
    todo.ctx.process_turn("accept")
    assert record.state == CONFIRMED


def test_an_approval_still_expires_after_two_replies(todo, annotated_library):
    """The approval kind keeps its cap of two replies (TTL_TURNS): two other replies expire it, and nothing runs."""
    _ask(todo, "add_note <text>Ann</text>", steps=_finish())
    approval = todo.store.active()
    assert approval is not None and approval.ttl_turns == 2
    todo.ctx.process_turn("maybe later")
    todo.ctx.process_turn("sure")
    todo.ctx.process_turn("yes")
    assert approval.state == "expired"
    assert _added(todo.chores, todo.base) == []
