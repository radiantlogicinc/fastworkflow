"""Self and pool distillation (distillation_compare.compare_self / compare_pool),
the distill CLI's mode flags, and two fixes to the live double pass.

Pair selection is a pure function (``pool_pairs``), tested here without a model.
The snapshot test runs the real todo-list workflow.
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path
from queue import Queue

import pytest

import fastworkflow
from fastworkflow.distillation_compare import (
    TaskComparison,
    attempt_passed,
    pool_pairs,
    provenance_records,
    run_pairs,
)


def test_outcome_decides_and_reward_is_the_fallback():
    assert attempt_passed({"outcome": "pass", "reward": 0.0}) is True
    assert attempt_passed({"outcome": "FAIL", "reward": 1.0}) is False
    assert attempt_passed({"outcome": None, "reward": 1.0}) is True
    assert attempt_passed({"reward": 0.5}) is False
    assert attempt_passed({"outcome": "error"}) is None
    assert attempt_passed({"reward": "n/a"}) is None


def test_self_pairs_every_failing_trial_with_the_first_passing_one():
    trials = {"t1": [("run", 1, False), ("run", 2, True), ("run", 3, False), ("run", 4, True)]}
    pairs = pool_pairs(trials, {})
    assert [(p.student_attempt, p.reference_attempt, p.reference_source) for p in pairs] == [
        (1, 2, "self"), (3, 2, "self")]


def test_a_task_with_no_pass_or_no_fail_is_skipped_with_the_reason():
    pairs = pool_pairs({"all-pass": [("run", 1, True)], "all-fail": [("run", 1, False)]}, {})
    reasons = {p.task_id: p.skipped_reason for p in pairs}
    assert reasons["all-pass"].startswith("no failing student trial")
    assert reasons["all-fail"].startswith("no passing reference")


def test_a_teacher_is_used_only_when_no_student_passed_and_only_if_it_passed():
    students = {"t1": [("s1", 1, False), ("s2", 1, False)], "t2": [("s1", 1, False)]}
    teachers = {"t1": [("teach", 1, False), ("teach", 2, True)], "t2": [("teach", 1, False)]}
    pairs = pool_pairs(students, teachers)
    t1 = [p for p in pairs if p.task_id == "t1"]
    assert [(p.student_experiment, p.reference_experiment, p.reference_attempt,
             p.reference_source) for p in t1] == [("s1", "teach", 2, "teacher"),
                                                  ("s2", "teach", 2, "teacher")]
    t2 = [p for p in pairs if p.task_id == "t2"]
    assert t2[0].skipped_reason and t2[0].reference_source == ""


def test_pairs_per_task_can_be_capped_and_tasks_chosen():
    students = {"t1": [("s", a, a == 9) for a in range(1, 10)], "t2": [("s", 1, False)]}
    pairs = pool_pairs(students, {}, task_ids=["t1"], max_pairs_per_task=2)
    assert [(p.task_id, p.student_attempt) for p in pairs] == [("t1", 1), ("t1", 2)]


def test_provenance_records_each_pair_and_what_it_produced():
    result = TaskComparison(task_id="t1", teacher_attempt=2, student_attempt=1,
                            execution_insights=["Read before writing."],
                            student_experiment="s", reference_experiment="s",
                            reference_source="self")
    assert provenance_records([result]) == [{
        "task_id": "t1", "student_experiment": "s", "student_attempt": 1,
        "reference_experiment": "s", "reference_attempt": 2, "reference_source": "self",
        "student_passed": None, "reference_passed": None,
        "skipped_reason": None, "execution_insights": ["Read before writing."],
        "planning_insights": [],
    }]


# ----------------------------------------------------------------------
# Teacher vs student: the attempts are chosen by the caller (run_pairs)
# ----------------------------------------------------------------------

TEACHER = {"16": [1, 2], "41": [1, 2], "99": [1]}
STUDENT = {"16": [1, 2, 3, 4, 5], "41": [1, 2, 3, 4, 5], "87": [1, 2]}


def _pairs(**kwargs):
    return [(p.task_id, p.student_attempt, p.reference_attempt, p.skipped_reason)
            for p in run_pairs(TEACHER, STUDENT, **kwargs)]


def test_by_default_the_lowest_attempts_are_compared_on_shared_tasks():
    assert _pairs() == [("16", 1, 1, None), ("41", 1, 1, None)]


def test_one_teacher_attempt_against_chosen_student_attempts_for_every_task():
    assert _pairs(teacher_attempt=2, student_attempts=[3, 5, 3]) == [
        ("16", 3, 2, None), ("16", 5, 2, None), ("41", 3, 2, None), ("41", 5, 2, None)]


def test_all_student_attempts():
    assert [p[1] for p in _pairs(task_ids=["16"], student_attempts="all")] == [1, 2, 3, 4, 5]


def test_a_missing_attempt_or_task_is_skipped_with_the_reason():
    pairs = _pairs(task_ids=["16", "87", "99"], teacher_attempt=2, student_attempts=[5, 9])
    assert pairs[0] == ("16", 5, 2, None)
    assert pairs[1][:3] == ("16", 9, 2) and pairs[1][3].startswith("student attempt 9 not recorded")
    assert pairs[2][3] == "task not recorded in teacher run"
    assert pairs[3][3] == "task not recorded in student run"
    missing_reference = run_pairs({"50": [1]}, {"50": [1, 2]}, teacher_attempt=2)
    assert missing_reference[0].skipped_reason.startswith("teacher attempt 2 not recorded")


@pytest.mark.parametrize("argv, problem", [
    (["--teacher", "a", "--student", "b", "--teacher-attempt", "2",
      "--student-attempt", "1", "--student-attempt", "3"], None),
    (["--teacher", "a", "--student", "b", "--student-attempt", "all"], None),
    (["--self", "a", "--teacher-attempt", "2"], "need --teacher and --student"),
    (["--teacher", "a", "--student", "b", "--student-attempt", "all",
      "--student-attempt", "2"], "cannot be combined"),
    (["--teacher", "a", "--student", "b", "--student-attempt", "x"], "attempt numbers or 'all'"),
    (["--teacher", "a", "--student", "b", "--max-pairs-per-task", "2"], "--max-pairs-per-task"),
])
def test_attempt_flags_belong_to_teacher_student_mode(argv, problem):
    from fastworkflow.cli import _distill_mode_error, _student_attempt_choice

    args = _distill_args(argv)
    error = _distill_mode_error(args)
    assert (error is None) if problem is None else (problem in error)
    if problem is None:
        assert _student_attempt_choice(args.student_attempts) in ([1, 3], "all")


def _distill_args(argv: list[str]):
    from fastworkflow.cli import add_distill_parser

    parser = argparse.ArgumentParser()
    add_distill_parser(parser.add_subparsers(dest="command"))
    return parser.parse_args(["distill", "wf", *argv])


@pytest.mark.parametrize("argv, ok", [
    (["--teacher", "a", "--student", "b"], True),
    (["--self", "a"], True),
    (["--pool-students", "a", "b", "--pool-teachers", "c"], True),
    (["--teacher", "a"], False),
    (["--self", "a", "--teacher", "b", "--student", "c"], False),
    (["--pool-teachers", "c"], False),
    ([], False),
])
def test_distill_takes_exactly_one_mode(argv, ok):
    from fastworkflow.cli import _distill_mode_error

    assert (_distill_mode_error(_distill_args(argv)) is None) is ok


# ----------------------------------------------------------------------
# The live double pass (distillation.distill_message)
# ----------------------------------------------------------------------


@pytest.fixture
def session(tmp_path):
    from fastworkflow.command_routing import RoutingRegistry
    from fastworkflow.distillation import DistillationSession
    from fastworkflow.workflow_execution_context import WorkflowExecutionContext

    fastworkflow.init({})
    RoutingRegistry.clear_registry()
    path = str(Path(__file__).parent.joinpath("todo_list_workflow").resolve())
    workflow = fastworkflow.Workflow.create(path, workflow_id_str=f"distil-{uuid.uuid4().hex}")
    ctx = WorkflowExecutionContext(run_as_agent=False)
    ctx.bind_app_workflow(workflow)
    ctx.push_active_workflow(workflow)
    yield DistillationSession(ctx)
    RoutingRegistry.clear_registry()


def test_restoring_a_snapshot_undoes_what_the_first_pass_changed(session):
    workflow = session.chat_session.get_active_workflow()
    workflow.context["records"] = {"T-1": "open"}
    snapshot = session.snapshot_workflow_state()
    workflow.context["records"]["T-1"] = "closed"   # the teacher pass changes state
    session.restore_workflow_state(snapshot)
    assert workflow.context["records"] == {"T-1": "open"}
    workflow.context["records"]["T-1"] = "closed"   # the student pass changes it again
    session.restore_workflow_state(snapshot)        # the snapshot itself was not touched
    assert workflow.context["records"] == {"T-1": "open"}


def test_the_student_pass_is_kept_off_the_command_trace_queue(session):
    queue = Queue()
    session.chat_session._command_trace_queue = queue
    with session.trace_queue_detached():
        assert session.chat_session.command_trace_queue is None
    assert session.chat_session.command_trace_queue is queue


def test_compare_runs_compares_exactly_the_chosen_attempts_and_labels_outcomes(tmp_path):
    """End to end on a real store: graded attempts of two runs, identical recorded
    actions (so no divergence and no judge call). compare_runs must compare the
    attempts asked for and carry each side's outcome to the result."""
    import json as _json

    from fastworkflow import state_paths, tracing
    from fastworkflow.distillation_compare import compare_runs
    from fastworkflow.observability import store as obs
    from tests.benchmark_pins import pin_for

    workflow = tmp_path / "wf"
    workflow.mkdir()
    store = obs.ObservabilityStore(state_paths.observability_db(str(workflow)))
    graded = {"T": {1: "fail", 2: "pass"}, "S": {1: "pass", 2: "fail", 3: "fail"}}
    turns, spans = [], []
    for exp, attempts in graded.items():
        store.create_experiment(exp, exp, declared_tasks=1, declared_attempts=len(attempts),
                                **pin_for(str(workflow), ["16"]))
        for attempt, outcome in attempts.items():
            channel = f"c-{exp}-{attempt}"
            store.start_attempt(exp, "16", attempt, channel)
            store.finish_attempt(exp, "16", attempt, outcome=outcome, outcome_source="grader")
            key = f"turn-{exp}-{attempt}"
            turns.append({
                "turn_key": key, "channel_id": channel, "conversation_id": 1, "ordinal": 1,
                "user_message": "close T-1", "refined_user_message": None,
                "entry_workflow_name": "wf", "entry_context": "global", "status": "completed",
                "success": 1, "failure_reason": None, "answer": "done",
                "conversation_summary": None, "conversation_traces": None,
                "started_at": "2026-10-01T00:00:00+00:00",
                "completed_at": "2026-10-01T00:01:00+00:00", "suspended_ms": 0,
                "continuation_of": None, "record_version": 1,
                "record_json": _json.dumps({"turn_output": {"turn_key": key}}),
                "experiment_id": exp, "task_id": "16", "attempt": attempt,
            })
            spans.append(tracing.Span(
                span_id=f"{key}-x", trace_id=key, name=tracing.SPAN_COMMAND_EXECUTE,
                channel_id=channel, command_name="close_ticket", start_ns=10, end_ns=20,
                status=tracing.STATUS_OK, experiment_id=exp, task_id="16", attempt=attempt,
                attributes={"raw_command": "close_ticket", "parameters": {"id": "T-1"},
                            "response_text": "closed"},
            ))
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        for row in turns:
            assert store.upsert_turn_row(conn, row, [], obs.Redactor())
        store.upsert_span_rows(conn, spans, obs.Redactor())
        conn.commit()

    results = compare_runs(
        store, teacher_experiment_id="T", student_experiment_id="S",
        workflow_folderpath=str(workflow), teacher_attempt=2, student_attempts=[2, 3, 7],
        write_insights=False, write_feedback=False,
    )
    assert [(r.student_attempt, r.teacher_attempt, r.skipped_reason is None) for r in results] == [
        (2, 2, True), (3, 2, True), (7, 2, False)]
    assert [(r.student_passed, r.reference_passed) for r in results[:2]] == [(False, True)] * 2
    assert all(not r.diverged for r in results[:2])
    assert {(r.student_experiment, r.reference_experiment) for r in results} == {("S", "T")}

    default = compare_runs(
        store, teacher_experiment_id="T", student_experiment_id="S",
        workflow_folderpath=str(workflow), write_insights=False, write_feedback=False,
    )
    assert [(r.student_attempt, r.teacher_attempt, r.student_passed, r.reference_passed)
            for r in default] == [(1, 1, True, False)]
