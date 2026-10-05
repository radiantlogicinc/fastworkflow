"""Focused tests for the driver-neutral experiment lifecycle controller."""

from __future__ import annotations

import json
import sqlite3
import threading

import pytest

from fastworkflow import state_paths
from fastworkflow.observability import store as obs
from fastworkflow.experiment.runner import (
    ExperimentController,
    MissingExperimentLifecycleFeature,
    experiment_store_readiness,
)


@pytest.fixture
def workflow(tmp_path, monkeypatch) -> str:
    monkeypatch.setenv(obs.CAPTURE_PROFILE_VAR, "evidence")
    folder = str(tmp_path / "workflow")
    obs.ObservabilityStore(state_paths.observability_db(folder))
    return folder


@pytest.fixture
def db_path(workflow) -> str:
    return state_paths.observability_db(workflow)


def _controller(workflow: str, *, external: bool = False) -> ExperimentController:
    identity = experiment_store_readiness(state_paths.observability_db(workflow))["store_id"]
    return ExperimentController(
        workflow, identity, migrate=False, external=external
    )


def _create(
    controller: ExperimentController,
    experiment_id: str = "exp-1",
    *,
    tasks: int = 1,
    attempts: int = 1,
    task_ids=None,
) -> None:
    task_ids = task_ids or [f"t{index}" for index in range(tasks)]
    controller.create_experiment(
        experiment_id,
        "label",
        declared_tasks=tasks,
        declared_attempts=attempts,
        declarations=[
            (task_id, attempt, f"channel:{task_id}:{attempt}")
            for task_id in task_ids
            for attempt in range(1, attempts + 1)
        ],
    )


def _start_and_terminalize(
    controller: ExperimentController,
    experiment_id: str = "exp-1",
    *,
    task_id: str = "t0",
    attempt: int = 1,
) -> None:
    controller.start_attempt(
        experiment_id,
        task_id,
        attempt,
        f"channel:{task_id}:{attempt}",
    )
    controller.terminalize_attempt(
        experiment_id,
        task_id,
        attempt,
        execution_status="completed",
    )


def test_external_controller_refuses_capture_regime_mismatch(workflow, db_path):
    with pytest.raises(obs.CaptureRegimeChanged):
        ExperimentController(
            workflow,
            experiment_store_readiness(db_path)["store_id"],
            migrate=False,
            external=True,
            capture_profile="debug",
            capture_policy_version=obs.CAPTURE_POLICY_VERSION,
        )


def test_external_controller_has_no_sink_writer_or_pruning_side_effect(
    workflow, db_path, monkeypatch
):
    before = {thread.ident for thread in threading.enumerate()}

    def forbidden(*args, **kwargs):
        raise AssertionError("external metadata controller invoked capture machinery")

    monkeypatch.setattr(obs, "get_observability_sink", forbidden)
    monkeypatch.setattr(obs.SQLiteTraceSink, "__init__", forbidden)
    monkeypatch.setattr(obs.ObservabilityStore, "prune", forbidden)
    controller = ExperimentController(
        workflow,
        experiment_store_readiness(db_path)["store_id"],
        migrate=False,
        external=True,
    )

    assert controller.store.db_path == db_path
    assert {thread.ident for thread in threading.enumerate()} == before


def test_external_controller_never_migrates_and_refuses_missing_feature(
    tmp_path, monkeypatch
):
    folder = str(tmp_path / "workflow")
    path = state_paths.observability_db(folder)
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE diagnostics "
        "(key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT INTO diagnostics VALUES ('schema_features', ?, 'now')",
        (json.dumps([obs.FEATURE_EXPERIMENTS_V1]),),
    )
    conn.commit()
    conn.close()

    def forbidden(self):
        raise AssertionError("external controller ran DDL")

    monkeypatch.setattr(obs.ObservabilityStore, "_ensure_schema", forbidden)
    with pytest.raises(MissingExperimentLifecycleFeature):
        ExperimentController(
            folder, "missing-store-identity", migrate=False, external=True
        )


def test_compatible_finish_attempt_runs_running_to_complete(workflow):
    controller = _controller(workflow)
    _create(controller)
    controller.start_attempt("exp-1", "t0", 1, "channel:t0:1")
    controller.finish_attempt(
        "exp-1",
        "t0",
        1,
        outcome="pass",
        outcome_source="grader",
        reward=1.0,
    )

    assert controller.complete_experiment("exp-1") == "complete"
    assert controller.store.experiment_scores("exp-1")["reportable"] is True


def test_external_capture_complete_is_natively_non_reportable(workflow):
    controller = _controller(workflow, external=True)
    _create(controller)
    _start_and_terminalize(controller)
    controller.record_outcome(
        "exp-1", "t0", 1, outcome="pass", outcome_source="external-grader"
    )

    assert controller.complete_experiment("exp-1") == "capture_complete"
    score = controller.store.experiment_scores("exp-1")
    assert score["status"] == "capture_complete"
    assert score["reportable"] is False
    assert score["pass_at_1"] is None
    assert "capture_complete" in score["reason_not_reportable"]


def test_deferred_evaluation_runs_awaiting_evaluation_to_complete(workflow):
    controller = _controller(workflow)
    _create(controller)
    _start_and_terminalize(controller)

    assert controller.complete_experiment("exp-1") == "awaiting_evaluation"
    pending = controller.store.experiment_scores("exp-1")
    assert pending["reportable"] is False
    assert pending["pass_at_1"] is None

    controller.record_outcome(
        "exp-1", "t0", 1, outcome="pass", outcome_source="later-grader"
    )
    assert controller.complete_experiment("exp-1") == "complete"


def test_invalid_is_terminal(workflow):
    controller = _controller(workflow)
    _create(controller)
    assert (
        controller.invalidate_experiment("exp-1", "operator", "bad capture")
        == "invalid"
    )
    assert controller.complete_experiment("exp-1") == "invalid"
    with pytest.raises(obs.ExperimentIsClosed):
        controller.start_attempt("exp-1", "t0", 1, "channel:t0:1")


def test_restart_only_accepts_execution_open_attempts(workflow):
    controller = _controller(workflow)
    _create(controller, tasks=2, task_ids=["open", "closed"])
    controller.start_attempt("exp-1", "open", 1, "channel:open:1")
    controller.start_attempt("exp-1", "closed", 1, "channel:closed:1")
    controller.terminalize_attempt(
        "exp-1", "closed", 1, execution_status="completed"
    )

    assert controller.restart_attempt("exp-1", "open", 1) == 0
    with pytest.raises(obs.AttemptValueConflict):
        controller.restart_attempt("exp-1", "closed", 1)


def test_terminal_values_are_idempotent_and_conflicts_are_refused(workflow):
    controller = _controller(workflow)
    _create(controller)
    controller.start_attempt(
        "exp-1",
        "t0",
        1,
        "channel:t0:1",
        source_attempt_key={"driver": "job-7"},
    )
    controller.start_attempt(
        "exp-1",
        "t0",
        1,
        "channel:t0:1",
        source_attempt_key={"driver": "job-7"},
    )
    controller.terminalize_attempt(
        "exp-1", "t0", 1, execution_status="completed"
    )
    controller.terminalize_attempt(
        "exp-1", "t0", 1, execution_status="completed"
    )
    with pytest.raises(obs.AttemptValueConflict):
        controller.terminalize_attempt(
            "exp-1", "t0", 1, execution_status="failed"
        )

    kwargs = {
        "outcome": "pass",
        "outcome_source": "grader",
        "reward": 1.0,
        "detail": {"rubric": "v1"},
    }
    controller.record_outcome("exp-1", "t0", 1, **kwargs)
    controller.record_outcome("exp-1", "t0", 1, **kwargs)
    with pytest.raises(obs.AttemptValueConflict):
        controller.record_outcome(
            "exp-1", "t0", 1, outcome="fail", outcome_source="grader"
        )


def test_source_attempt_key_is_versioned_and_scrubbed(workflow):
    controller = _controller(workflow)
    _create(controller)
    secret = "sk-0123456789abcdefghijklmnopqrstuv"
    controller.start_attempt(
        "exp-1",
        "t0",
        1,
        "channel:t0:1",
        source_attempt_key={"driver_key": secret},
    )
    row = controller.store.experiment_attempt_rows("exp-1")[0]
    value = json.loads(row["source_attempt_json"])
    assert value["version"] == 1
    assert secret not in row["source_attempt_json"]


def test_declared_denominator_is_still_enforced(workflow):
    controller = _controller(workflow)
    _create(controller, tasks=2)
    _start_and_terminalize(controller)
    controller.record_outcome(
        "exp-1", "t0", 1, outcome="pass", outcome_source="grader"
    )

    assert controller.complete_experiment("exp-1") == "invalid"
    assert controller.store.get_experiment("exp-1")["invalid_reason"] == (
        "attempt_shortfall"
    )


def test_markerless_ready_schema_is_refused_for_external_open(workflow, db_path):
    identity = obs.ObservabilityStore(
        db_path, migrate=False
    ).store_identity()
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM diagnostics WHERE key='schema_features'")
    conn.commit()
    conn.close()

    with pytest.raises(MissingExperimentLifecycleFeature):
        ExperimentController(
            workflow,
            identity,
            migrate=False,
            external=True,
        )
