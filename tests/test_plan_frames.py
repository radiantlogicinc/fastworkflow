"""Phase 4 of the unified command architecture (design 3.5): the PlanFrame stack.

Phase 4 moves each planning-command call's sub-task state onto its own PlanFrame, carries
nested frames across a suspension, and adds the turn-wide step budget and the step floor.
It must not change behaviour: a turn with no nested frame numbers, runs and restores
exactly as before, and the session state schema is not bumped (absent keys mean no
nested frames).
"""

from __future__ import annotations

import copy
import json
import logging
import uuid
from contextlib import suppress
from types import SimpleNamespace

import pytest

import fastworkflow
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.observation_offloading.state import (
    archive_for_path,
    observability_db_path,
    scope_for_host,
)
from fastworkflow.plan_frame import (
    DEFAULT_TURN_MAX_STEPS,
    TURN_MAX_STEPS_ENV,
    PlanFrame,
    turn_max_steps,
)
from fastworkflow.session_state_store import SCHEMA_VERSION
from fastworkflow.state_serialization import validate_state
from fastworkflow.turn import TurnStatus
from fastworkflow.utils.logging import logger as fastworkflow_logger
from fastworkflow.workflow_execution_context import WorkflowExecutionContext
from tests.test_skill_code_steps import Model
from tests.test_step_helpers import LibraryRunner, _library_workflow
from tests.test_turn_result_capture import (  # noqa: F401 - shared fixtures
    initialized_fastworkflow,
    todo_workflow_path,
)

#: The four session-blob keys Phase 4 adds. Absent from an older blob, they restore as "no nested frames".
NEW_BLOB_KEYS = ("frames", "frame_react", "turn_child_step_ranges", "turn_steps_used")
#: The turn frame's sub-task fields, the ones the turn frame keeps in the existing top-level keys.
TURN_SUBTASK_FIELDS = ("subtasks", "origins", "subtask_index", "subtask_answers", "subtask_exhausted")
#: A real command of the todo workflow, the qualified name a nested frame would carry.
NESTED_COMMAND = "TodoListManager/get_todo_list"


def _turn_subtask_state(ctx: WorkflowExecutionContext) -> dict:
    turn = ctx._turn_frame()
    return {name: copy.deepcopy(getattr(turn, name)) for name in TURN_SUBTASK_FIELDS}


def _nested_frame() -> PlanFrame:
    return PlanFrame(
        kind="planning", command_name=NESTED_COMMAND, depth=1, alias="/get_todo_list#1",
        subtasks=["s"], origins=[0], child_step_ranges=[[3, 5]], invoking_step=2,
    )


def _todo_context(todo_workflow_path: str, session_key: str) -> WorkflowExecutionContext:
    ctx = WorkflowExecutionContext(run_as_agent=True, session_key=session_key)
    ctx.bind_app_workflow(
        fastworkflow.Workflow.create(todo_workflow_path, workflow_id_str=session_key)
    )
    return ctx


@pytest.fixture
def warnings_logged():
    """Collect WARNING records off the fastWorkflow logger, which does not propagate to caplog."""
    collected: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record):
            collected.append(record)

    handler = _Collector(level=logging.WARNING)
    previous_level = fastworkflow_logger.level
    fastworkflow_logger.setLevel(min(previous_level, logging.WARNING))
    fastworkflow_logger.addHandler(handler)
    try:
        yield collected
    finally:
        fastworkflow_logger.removeHandler(handler)
        fastworkflow_logger.setLevel(previous_level)


@pytest.fixture
def no_turn_budget_env(monkeypatch):
    """No FW_TURN_MAX_STEPS from the process environment or the workflow's env file."""
    monkeypatch.delenv(TURN_MAX_STEPS_ENV, raising=False)
    monkeypatch.delitem(fastworkflow._env_vars, TURN_MAX_STEPS_ENV, raising=False)


# ----------------------------------------------------------------------
# PlanFrame alone
# ----------------------------------------------------------------------


def test_a_planning_frame_round_trips_every_serialised_field():
    frame = PlanFrame(
        kind="planning", command_name="Library/find_book", depth=1, alias="/find_book#1",
        subtasks=["find Dune"], origins=[2], subtask_index=1, subtask_answers=["Dune is b1"],
        subtask_exhausted=True, phase_run={"phase": 2, "runs": [{"command": "find_book"}]},
        child_step_ranges=[[3, 5]], invoking_step=2, fast_path=True,
        # Not serialised, so they must not appear in the state and must not come back.
        evidence_floors={0: 4}, initial_plan_text="1. look up", phase_live=("cmd", []),
        phased_skill_call=("cmd", None), agent=object(),
    )

    state = frame.to_state()

    assert state == {
        "kind": "planning", "command_name": "Library/find_book", "depth": 1,
        "alias": "/find_book#1", "subtasks": ["find Dune"], "origins": [2], "subtask_index": 1,
        "subtask_answers": ["Dune is b1"], "subtask_exhausted": True,
        "phase_run": {"phase": 2, "runs": [{"command": "find_book"}]},
        "child_step_ranges": [[3, 5]], "invoking_step": 2, "fast_path": True,
    }
    restored = PlanFrame.from_state(json.loads(json.dumps(state)))
    assert restored.to_state() == state
    assert restored.evidence_floors == {} and restored.initial_plan_text == ""
    assert restored.phase_live is None and restored.phased_skill_call is None and restored.agent is None


def test_to_state_returns_fresh_containers_that_do_not_change_the_frame():
    frame = PlanFrame(
        kind="planning", command_name="Library/find_book", depth=1, alias="/find_book#1",
        subtasks=["find Dune"], origins=[2], subtask_answers=["Dune is b1"],
        phase_run={"runs": [{"command": "find_book"}]}, child_step_ranges=[[3, 5]],
    )

    state = frame.to_state()
    state["subtasks"].append("mutated")
    state["origins"].append(9)
    state["subtask_answers"].append("mutated")
    state["child_step_ranges"][0][0] = 99
    state["phase_run"]["runs"].append({})

    assert frame.subtasks == ["find Dune"]
    assert frame.origins == [2]
    assert frame.subtask_answers == ["Dune is b1"]
    assert frame.child_step_ranges == [[3, 5]]
    assert frame.phase_run == {"runs": [{"command": "find_book"}]}
    assert frame.to_state()["child_step_ranges"] == [[3, 5]]


def test_from_state_of_an_empty_dict_is_the_turn_frame_default():
    frame = PlanFrame.from_state({})

    assert frame == PlanFrame.turn()
    assert frame.kind == "turn" and frame.command_name is None and frame.depth == 0
    assert frame.alias == "" and frame.subtasks == [] and frame.origins == []
    assert frame.subtask_index == 0 and frame.subtask_answers == [] and frame.subtask_exhausted is False
    assert frame.phase_run is None and frame.child_step_ranges == []
    assert frame.invoking_step is None and frame.fast_path is False


def test_absent_origins_are_each_entry_its_own_origin():
    assert PlanFrame.from_state({"subtasks": ["a", "b"]}).origins == [0, 1]


# ----------------------------------------------------------------------
# A real context: nested frames through serialize_state / apply_serialized_state
# ----------------------------------------------------------------------


def test_nested_frames_and_step_counters_survive_serialize_and_restore(
    initialized_fastworkflow, todo_workflow_path
):
    session_key = f"plan-frames-{uuid.uuid4().hex}"
    ctx = _todo_context(todo_workflow_path, session_key)
    nested = _nested_frame()
    ctx._frames.append(nested)
    ctx._turn_frame().child_step_ranges = [[1, 2]]
    ctx._turn_steps_used = 7

    blob = ctx.serialize_state(channel_id=session_key)
    validate_state(blob)
    wire = json.loads(json.dumps(blob))

    for key in NEW_BLOB_KEYS:
        assert key in wire
    assert wire["frames"] == [nested.to_state()]
    assert wire["frame_react"] == [None]
    assert wire["turn_child_step_ranges"] == [[1, 2]]
    assert wire["turn_steps_used"] == 7
    assert wire == blob

    restored = _todo_context(todo_workflow_path, session_key)
    restored.apply_serialized_state(wire)

    assert len(restored._frames) == 2
    assert [frame.to_state() for frame in restored._frames] == [
        frame.to_state() for frame in ctx._frames
    ]
    top = restored._frames[1]
    assert top.alias == "/get_todo_list#1" and top.depth == 1 and top.command_name == NESTED_COMMAND
    assert top.child_step_ranges == [[3, 5]] and top.invoking_step == 2
    assert restored._turn_frame().child_step_ranges == [[1, 2]]
    assert restored._turn_steps_used == 7


def test_a_blob_without_the_new_keys_restores_one_frame_and_the_old_keys_as_before(
    initialized_fastworkflow, todo_workflow_path
):
    session_key = f"plan-frames-old-{uuid.uuid4().hex}"
    ctx = _todo_context(todo_workflow_path, session_key)
    turn = ctx._turn_frame()
    turn.subtasks = ["find X", "find Y"]
    turn.origins = [0, 1]
    turn.subtask_index = 1
    turn.subtask_answers = ["X is here"]
    ctx._next_step = 4
    ctx._frames.append(_nested_frame())
    ctx._turn_frame().child_step_ranges = [[1, 2]]
    ctx._turn_steps_used = 3
    blob = ctx.serialize_state(channel_id=session_key)

    # What a blob written before Phase 4 looks like: the same payload without the four new keys.
    old_blob = json.loads(json.dumps({k: v for k, v in blob.items() if k not in NEW_BLOB_KEYS}))
    validate_state(old_blob)
    assert not set(NEW_BLOB_KEYS) & set(old_blob)

    restored = _todo_context(todo_workflow_path, session_key)
    restored.apply_serialized_state(old_blob)

    assert len(restored._frames) == 1
    assert restored._turn_frame().child_step_ranges == []
    assert restored._turn_steps_used == 0
    assert _turn_subtask_state(restored) == {
        "subtasks": ["find X", "find Y"], "origins": [0, 1], "subtask_index": 1,
        "subtask_answers": ["X is here"], "subtask_exhausted": False,
    }
    assert restored._next_step == 4


def test_the_session_schema_version_is_still_four():
    # Phase 4 deliberately did not bump SCHEMA_VERSION (design 3.5, Serialisation): the new keys
    # are optional and an absent key means "no nested frames". Bump it only with that decision revisited.
    assert SCHEMA_VERSION == 4


def test_the_current_frame_alias_is_the_top_frames_and_pops_back_to_the_turn(
    initialized_fastworkflow, todo_workflow_path
):
    ctx = _todo_context(todo_workflow_path, f"plan-frames-alias-{uuid.uuid4().hex}")
    assert ctx.current_frame_alias == ""

    ctx._frames.append(PlanFrame(kind="planning", command_name="Library/a", depth=1, alias="/a#1"))
    assert ctx.current_frame_alias == "/a#1"
    ctx._frames.append(PlanFrame(kind="planning", command_name="Library/b", depth=2, alias="/a#1/b#1"))
    assert ctx.current_frame_alias == "/a#1/b#1"

    ctx._frames.pop()
    assert ctx.current_frame_alias == "/a#1"
    ctx._frames.pop()
    assert ctx.current_frame_alias == ""


# ----------------------------------------------------------------------
# The turn-wide step budget
# ----------------------------------------------------------------------


def test_the_turn_budget_is_the_default_when_unset(no_turn_budget_env):
    assert DEFAULT_TURN_MAX_STEPS == 400
    assert turn_max_steps() == 400


def test_a_positive_integer_overrides_the_turn_budget(monkeypatch, no_turn_budget_env):
    monkeypatch.setitem(fastworkflow._env_vars, TURN_MAX_STEPS_ENV, "50")
    assert turn_max_steps() == 50


def test_the_workflow_env_file_wins_over_the_process_environment(monkeypatch, no_turn_budget_env):
    monkeypatch.setenv(TURN_MAX_STEPS_ENV, "75")
    assert turn_max_steps() == 75
    monkeypatch.setitem(fastworkflow._env_vars, TURN_MAX_STEPS_ENV, "50")
    assert turn_max_steps() == 50


@pytest.mark.parametrize("raw", ["0", "-3", "abc"])
def test_a_bad_turn_budget_warns_and_takes_the_default(raw, monkeypatch, no_turn_budget_env, warnings_logged):
    monkeypatch.setitem(fastworkflow._env_vars, TURN_MAX_STEPS_ENV, raw)

    assert turn_max_steps() == 400

    [record] = [r for r in warnings_logged if TURN_MAX_STEPS_ENV in r.getMessage()]
    assert record.levelno == logging.WARNING
    assert f"{TURN_MAX_STEPS_ENV}={raw!r}" in record.getMessage()
    assert "using the default 400" in record.getMessage()


# ----------------------------------------------------------------------
# Real turns on the library: scripted model, real ReAct loop, real command runs
# ----------------------------------------------------------------------


@pytest.fixture
def library_sessions(monkeypatch, initialized_fastworkflow):
    """Builds real agent-mode contexts on a library copy and tears them down after the test.

    The LLM boundary is faked as the repo's tests fake it: the ReAct steps are scripted
    (``Model``), commands run through ``LibraryRunner``, and the planner, splitter and
    summary calls return fixed text. The agent is the one the app builds
    (``_ensure_agent_initialized``), so it carries the step floor a real turn's agent has.
    """
    made = []

    def make(folder: str, *, subtasks: list[str] | None = None, startup: bool = True):
        ctx = WorkflowExecutionContext(run_as_agent=True)
        wf = fastworkflow.Workflow.create(folder, workflow_id_str=f"plan-frames-{uuid.uuid4().hex}")
        ctx.bind_app_workflow(wf)
        ctx.push_active_workflow(wf)
        runner = LibraryRunner(wf)
        monkeypatch.setattr(
            CommandExecutor, "invoke_command",
            classmethod(lambda cls, session, command: runner(session, command)),
        )
        if startup:
            runner(ctx, "startup")
            runner.sent.clear()
        monkeypatch.setattr(
            "fastworkflow.workflow_agent.build_query_with_next_steps",
            lambda user_query, session, **kwargs: user_query,
        )
        monkeypatch.setattr(
            "fastworkflow.workflow_agent.planner_skill_for",
            lambda text, folder_path, **_: ("", None),
        )
        monkeypatch.setattr(
            "fastworkflow.workflow_agent.split_into_subtasks",
            lambda query, **_: list(subtasks or [query]),
        )
        monkeypatch.setattr(
            ctx, "_extract_conversation_summary",
            lambda user_query, actions, final: ("summary", "{}"),
        )
        ctx._ensure_agent_initialized()
        agent = ctx.workflow_tool_agent
        model = Model(agent)
        monkeypatch.setattr(agent, "_call_with_potential_trajectory_truncation", model)
        session = SimpleNamespace(ctx=ctx, wf=wf, runner=runner, agent=agent, model=model)
        made.append(session)
        return session

    yield make

    for session in made:
        with suppress(Exception):
            session.ctx.pop_active_workflow()
        with suppress(Exception):
            session.ctx.close()


def _archived_steps(ctx: WorkflowExecutionContext) -> list[tuple[int, str]]:
    rows = archive_for_path(observability_db_path(ctx)).command_runs(scope_for_host(ctx))
    return [(row["step_index"], row["command"]) for row in rows]


def _multi_subtask_turn(folder: str, library_sessions, *, floor_kept: bool):
    session = library_sessions(folder, subtasks=["find Dune", "find Dune again"])
    if not floor_kept:
        session.agent.step_floor = None
    # Two sub-tasks, each a search and a finish, in the Library context both need. A finish
    # step takes a step index of its own, so the second sub-task's search is step 3.
    session.model.run("find_book query=Dune")
    session.model.run("list_branches")
    session.model.finish()
    session.model.run("find_book query=Dune")
    session.model.finish()

    result = session.ctx.process_turn("look up Dune, then look up Dune again")

    return SimpleNamespace(
        status=result.status,
        answer=result.answer,
        steps=_archived_steps(session.ctx),
        trajectory_keys=[sorted(entry["trajectory"]) for entry in session.model.asked],
        step_floor_set=session.agent.step_floor is not None,
    )


def test_a_multi_subtask_turn_numbers_its_steps_the_same_with_and_without_the_step_floor(
    tmp_path, library_sessions
):
    built = _multi_subtask_turn(_library_workflow(tmp_path / "built"), library_sessions,
                                floor_kept=True)
    unset = _multi_subtask_turn(_library_workflow(tmp_path / "unset"), library_sessions,
                                floor_kept=False)

    assert built.step_floor_set is True
    assert unset.step_floor_set is False
    assert built.status == TurnStatus.COMPLETED and unset.status == TurnStatus.COMPLETED
    assert built.steps == unset.steps
    assert built.trajectory_keys == unset.trajectory_keys
    assert built.answer == unset.answer
    # The two sub-tasks number their steps in one run, the second from the step after the first's last.
    assert built.steps == [(0, "Library/find_book"), (1, "Library/list_branches"),
                           (3, "Library/find_book")]


def test_an_ask_user_suspension_resumes_on_a_fresh_context_with_the_step_floor_set(
    tmp_path, library_sessions
):
    folder = _library_workflow(tmp_path / "wf")
    first = library_sessions(folder)
    first.model.run("find_book query=Dune")
    first.model.steps.append(("ask", "ask_user", {"clarification_request": "Which Dune?"}))

    suspended = first.ctx.process_turn("look up Dune")

    assert suspended.status == TurnStatus.AWAITING_USER
    assert first.ctx.awaiting_user
    session_key = "plan-frames-resume"
    blob = json.loads(json.dumps(first.ctx.serialize_state(channel_id=session_key)))
    validate_state(blob)

    resumed = library_sessions(folder, startup=False)
    resumed.ctx.apply_serialized_state(blob)

    assert resumed.ctx.awaiting_user
    assert resumed.agent is resumed.ctx.workflow_tool_agent
    assert resumed.agent.step_floor is not None
    again = resumed.ctx.serialize_state(channel_id=session_key)
    for key in NEW_BLOB_KEYS:
        assert again[key] == blob[key]

    resumed.model.steps.append(("done", "finish", {}))
    result = resumed.ctx.process_turn("the blue one")

    assert result.status == TurnStatus.COMPLETED
    assert not resumed.ctx.awaiting_user
    assert "the answer" in result.answer


def test_a_library_turn_restores_with_no_nested_frames_and_the_next_turn_runs(
    tmp_path, library_sessions
):
    folder = _library_workflow(tmp_path / "wf")
    first = library_sessions(folder)
    first.model.run("find_book query=Dune")
    first.model.finish()
    done = first.ctx.process_turn("look up Dune")
    assert done.status == TurnStatus.COMPLETED
    # A completed turn leaves no sub-task state behind: _process_agent_message clears it.
    assert _turn_subtask_state(first.ctx)["subtasks"] == []
    # So the turn frame is given the sub-task state a split turn holds mid-run, for the restore to carry.
    turn = first.ctx._turn_frame()
    turn.subtasks = ["find Dune", "find Dune again"]
    turn.origins = [0, 1]
    turn.subtask_index = 1
    turn.subtask_answers = ["Dune is b1"]
    first.ctx._next_step = 3
    subtask_state = _turn_subtask_state(first.ctx)
    next_step = first.ctx._next_step

    blob = json.loads(json.dumps(first.ctx.serialize_state(channel_id="plan-frames-library")))
    validate_state(blob)
    assert blob["frames"] == [] and blob["frame_react"] == []
    assert blob["turn_child_step_ranges"] == [] and blob["turn_steps_used"] == 0

    # The restored session runs startup as a fresh app would, which puts the workflow in
    # the Library context. apply_serialized_state does not restore the workflow's navigation
    # position (see the strict xfail below), so this is what lets the next turn's find run.
    restored = library_sessions(folder)
    restored.ctx.apply_serialized_state(blob)

    assert len(restored.ctx._frames) == 1
    assert restored.ctx._turn_steps_used == 0
    assert restored.ctx._turn_frame().child_step_ranges == []
    assert _turn_subtask_state(restored.ctx) == subtask_state
    assert restored.ctx._next_step == next_step

    restored.model.run("find_book query=Dune")
    restored.model.finish()
    again = restored.ctx.process_turn("look up Dune again")

    assert again.status == TurnStatus.COMPLETED
    assert "the answer" in again.answer
    assert restored.runner.sent == ["find_book query=Dune"]


@pytest.mark.xfail(
    strict=True,
    reason="pre-existing, not Phase 4 (tracking issue fix-cgs): apply_serialized_state does not restore the workflow's "
           "command context (workflow_execution_context.py, the 'navigation depth may not match' "
           "debug log). A restored session starts where a new workflow starts, not where its turn left it.",
)
def test_a_restored_session_resumes_in_the_command_context_its_turn_left(tmp_path, library_sessions):
    folder = _library_workflow(tmp_path / "wf")
    first = library_sessions(folder)
    first.model.run("find_book query=Dune")
    first.model.finish()
    assert first.ctx.process_turn("look up Dune").status == TurnStatus.COMPLETED
    left_in = first.wf.current_command_context_name
    blob = json.loads(json.dumps(first.ctx.serialize_state(channel_id="plan-frames-navigation")))

    restored = library_sessions(folder, startup=False)
    restored.ctx.apply_serialized_state(blob)

    assert restored.wf.current_command_context_name == left_in
