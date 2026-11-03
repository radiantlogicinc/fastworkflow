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
from fastworkflow.review_engine import AWAITING, CONFIRMED, REVISED
from fastworkflow.review_proposals import (Candidate, PlanProposal, SubTaskProposal, build_round1_proposal)
from fastworkflow.turn import TurnStatus
from tests.test_approval_binding import (  # noqa: F401 - the real agent on the library copy, and the guards
    _ask, _finish, annotated_library, no_live_model, no_model_call_for_parameters, todo)
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
    assert "2. tell Ann (inputs: Ann Lee" in edited.answer
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
