"""Exact experiment declarations and explicit store-identity handshakes."""

from __future__ import annotations

import json
import os
import sqlite3

import pytest

from fastworkflow import state_paths
from fastworkflow.observability import store as obs
from fastworkflow.experiment.runner import (
    ExperimentController,
    MissingExperimentLifecycleFeature,
    experiment_store_readiness,
)


@pytest.fixture
def installed_db(tmp_path, monkeypatch):
    monkeypatch.setenv(obs.CAPTURE_PROFILE_VAR, "evidence")
    folder = str(tmp_path / "workflow")
    store = obs.ObservabilityStore(state_paths.observability_db(folder))
    identity = store.store_identity()
    assert identity is not None
    return folder, identity


def _controller(installed_db, *, external=True):
    folder, identity = installed_db
    return ExperimentController(
        folder,
        identity,
        migrate=False,
        external=external,
    )


def _create(controller, declarations, *, evidence_segments=0):
    tasks = {task_id for task_id, _, _ in declarations}
    attempts = len(declarations) // len(tasks)
    controller.create_experiment(
        "exp-1",
        "label",
        declared_tasks=len(tasks),
        declared_attempts=attempts,
        declarations=declarations,
        required_evidence_segments=evidence_segments,
    )


def _finish(controller, task_id, native_attempt, source_key):
    channel = f"channel:{task_id}:{native_attempt}"
    controller.start_attempt(
        "exp-1",
        task_id,
        native_attempt,
        channel,
        source_key=source_key,
    )
    controller.finish_attempt(
        "exp-1",
        task_id,
        native_attempt,
        outcome="pass",
        outcome_source="grader",
    )


def test_exact_declaration_is_immutable_and_same_count_replacement_fails(
    installed_db,
):
    controller = _controller(installed_db, external=False)
    original = [("task-a", 1, "source-a"), ("task-b", 1, "source-b")]
    replacement = [("task-a", 1, "source-a"), ("task-c", 1, "source-c")]
    _create(controller, original)

    with pytest.raises(obs.ExperimentDeclarationConflict):
        controller.store.declare_experiment_attempts("exp-1", replacement)

    assert {
        (row["task_id"], row["native_attempt"], row["source_key"])
        for row in controller.store.experiment_attempt_declarations("exp-1")
    } == set(original)


def test_start_attempt_rejects_undeclared_identity(installed_db):
    controller = _controller(installed_db)
    _create(controller, [("task-a", 7, "driver-job-7")])

    with pytest.raises(obs.UndeclaredExperimentAttempt):
        controller.start_attempt(
            "exp-1",
            "task-b",
            7,
            "channel:task-b:7",
            source_key="driver-job-7",
        )
    with pytest.raises(obs.UndeclaredExperimentAttempt):
        controller.start_attempt(
            "exp-1",
            "task-a",
            7,
            "channel:task-a:7",
            source_key="different-source",
        )


def test_completion_rejects_same_count_wrong_task_mutation(installed_db):
    controller = _controller(installed_db, external=False)
    declarations = [("task-a", 1, "source-a"), ("task-b", 1, "source-b")]
    _create(controller, declarations)
    _finish(controller, "task-a", 1, "source-a")
    _finish(controller, "task-b", 1, "source-b")

    with controller.store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """UPDATE experiment_attempts
                  SET task_id='task-c', source_key='source-c'
                WHERE experiment_id='exp-1' AND task_id='task-b'"""
        )
        conn.commit()

    assert controller.complete_experiment("exp-1") == "invalid"
    experiment = controller.store.get_experiment("exp-1")
    assert experiment["invalid_reason"] == "attempt_shortfall"
    assert "exactly match" in experiment["invalid_detail"]


def test_explicit_store_identity_mismatch_is_refused(installed_db):
    folder, _ = installed_db
    with pytest.raises(obs.StoreIdentityMismatch):
        ExperimentController(
            folder,
            "different-store-id",
            migrate=False,
            external=True,
        )


def test_external_controller_cannot_enable_migration(installed_db):
    folder, identity = installed_db
    with pytest.raises(ValueError, match="migrate=False"):
        ExperimentController(
            folder,
            identity,
            migrate=True,
            external=True,
        )


def test_controller_records_into_the_live_db_its_state_root_resolves(
    installed_db, monkeypatch, tmp_path
):
    """No caller can point a controller at another DB (single live DB §6):
    another state root resolves another DB, which an external controller
    refuses to bring into existence."""
    folder, identity = installed_db
    controller = ExperimentController(
        folder,
        identity,
        migrate=False,
        external=True,
    )
    assert controller.db_path == state_paths.observability_db(folder)
    assert controller.store_identity == identity

    monkeypatch.setenv(
        "FASTWORKFLOW_STATE_ROOT", str(tmp_path / "unrelated-state-root")
    )
    with pytest.raises(MissingExperimentLifecycleFeature):
        ExperimentController(folder, identity, migrate=False, external=True)
    assert not os.path.exists(state_paths.observability_db(folder))


def test_external_open_refuses_old_schema_without_declaration_feature(tmp_path):
    folder = str(tmp_path / "workflow")
    path = state_paths.observability_db(folder)
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE diagnostics "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO diagnostics VALUES ('schema_features', ?, 'now')",
            (
                json.dumps(
                    [
                        obs.FEATURE_EXPERIMENTS_V1,
                        obs.FEATURE_EXPERIMENT_LIFECYCLE_V1,
                    ]
                ),
            ),
        )

    with pytest.raises(MissingExperimentLifecycleFeature):
        ExperimentController(
            folder,
            "old-store",
            migrate=False,
            external=True,
        )


def test_external_open_refuses_marker_from_incomplete_mixed_version(tmp_path):
    folder = str(tmp_path / "workflow")
    path = state_paths.observability_db(folder)
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE diagnostics "
            "(key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        conn.executemany(
            "INSERT INTO diagnostics VALUES (?, ?, 'now')",
            [
                (
                    "schema_features",
                    json.dumps(
                        [
                            obs.FEATURE_EXPERIMENTS_V1,
                            obs.FEATURE_EXPERIMENT_LIFECYCLE_V1,
                            obs.FEATURE_EXPERIMENT_DECLARATIONS_V1,
                        ]
                    ),
                ),
                (obs.STORE_IDENTITY_DIAGNOSTIC, "mixed-store"),
                (
                    obs.CAPTURE_REGIME_DIAGNOSTIC,
                    json.dumps(
                        {
                            "capture_profile": "evidence",
                            "capture_policy_version": obs.CAPTURE_POLICY_VERSION,
                        }
                    ),
                ),
            ],
        )

    with pytest.raises(MissingExperimentLifecycleFeature):
        ExperimentController(
            folder,
            "mixed-store",
            migrate=False,
            external=True,
        )


def test_required_evidence_segment_count_blocks_completion(installed_db):
    controller = _controller(installed_db, external=False)
    _create(
        controller,
        [("task-a", 1, "source-a")],
        evidence_segments=1,
    )
    _finish(controller, "task-a", 1, "source-a")

    assert controller.complete_experiment("exp-1") == "invalid"
    experiment = controller.store.get_experiment("exp-1")
    assert experiment["invalid_reason"] == "evidence_run_invalid"
    assert "1 required" in experiment["invalid_detail"]


def test_readiness_reports_durable_store_identity_and_regime(installed_db):
    folder, identity = installed_db
    path = state_paths.observability_db(folder)
    payload = experiment_store_readiness(path)

    assert payload == {
        "store_id": identity,
        "resolved_path": os.path.realpath(path),
        "capture_profile": "evidence",
        "capture_policy_version": obs.CAPTURE_POLICY_VERSION,
    }
    assert obs.ObservabilityStore(path).store_identity() == identity


def test_declaration_schema_has_distinct_feature_marker(installed_db):
    folder, _ = installed_db
    path = state_paths.observability_db(folder)
    store = obs.ObservabilityStore(path, migrate=False)
    with sqlite3.connect(path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        features = set(
            json.loads(
                conn.execute(
                    "SELECT value FROM diagnostics "
                    "WHERE key='schema_features'"
                ).fetchone()[0]
            )
        )

    assert "experiment_attempt_declarations" in tables
    assert obs.FEATURE_EXPERIMENT_DECLARATIONS_V1 in features
    assert obs.FEATURE_EXPERIMENT_DECLARATIONS_V1 not in {
        obs.FEATURE_EXPERIMENTS_V1,
        obs.FEATURE_EXPERIMENT_LIFECYCLE_V1,
    }
    assert store.experiment_declaration_schema_ready()

