"""Focused public-path tests for preregistered external attempts."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient

import fastworkflow
from fastworkflow.observability import store as obs
from fastworkflow.experiment.runner import (
    ExperimentController,
    MissingExperimentLifecycleFeature,
)


@dataclass
class BindingHarness:
    main: Any
    client: TestClient
    controller: ExperimentController
    workflow_path: str

    def register(
        self,
        *,
        channel_id: str = "registered:task-1:1",
        ttl_seconds: float = 300.0,
    ):
        return self.controller.register_attempt(
            "exp-http",
            "task-1",
            1,
            "driver-job-1",
            channel_id,
            ttl_seconds=ttl_seconds,
        )

    def initialize(self, bootstrap, *, channel_id: str | None = None):
        return self.client.post(
            "/initialize",
            json={
                "channel_id": channel_id or bootstrap.channel_id,
                "experiment_bootstrap": {
                    "registration_id": bootstrap.registration_id,
                    "secret": bootstrap.secret,
                    "store_id": self.controller.store_identity,
                },
            },
        )


@pytest.fixture
def binding_harness(tmp_path, monkeypatch):
    package_path = fastworkflow.get_fastworkflow_package_path()
    workflow_path = os.path.join(package_path, "examples", "hello_world")
    if not os.path.isdir(workflow_path):
        pytest.skip(f"hello_world workflow not found at {workflow_path}")

    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    monkeypatch.setenv(obs.CAPTURE_PROFILE_VAR, "evidence")
    sys.argv = ["pytest", "--workflow_path", workflow_path]
    import fastworkflow.run_fastapi_mcp.__main__ as main

    main = importlib.reload(main)
    with TestClient(main.app) as client:
        readiness = client.get("/probes/readyz")
        assert readiness.status_code == 200
        store_readiness = readiness.json()["experiment_store_readiness"]
        controller = ExperimentController(
            store_readiness["resolved_path"],
            store_readiness["store_id"],
            migrate=False,
            external=True,
            capture_profile=store_readiness["capture_profile"],
            capture_policy_version=store_readiness["capture_policy_version"],
        )
        controller.create_experiment(
            "exp-http",
            "HTTP binding",
            declared_tasks=1,
            declared_attempts=1,
            declarations=[("task-1", 1, "driver-job-1")],
        )
        yield BindingHarness(main, client, controller, workflow_path)


def test_readiness_reports_exact_experiment_store(binding_harness):
    payload = binding_harness.client.get("/probes/readyz").json()
    readiness = payload["experiment_store_readiness"]

    assert readiness == {
        "store_id": binding_harness.controller.store_identity,
        "resolved_path": os.path.realpath(binding_harness.controller.db_path),
        "capture_profile": "evidence",
        "capture_policy_version": obs.CAPTURE_POLICY_VERSION,
    }


def test_preregistered_initialize_claims_before_wec_and_writes_labelled_turn(
    binding_harness, monkeypatch
):
    bootstrap = binding_harness.register()
    original_wec = (
        importlib.import_module("fastworkflow.run_fastapi_mcp.utils")
        .WorkflowExecutionContext
    )
    constructed_after_claim = []

    def checked_wec(*args, **kwargs):
        with sqlite3.connect(binding_harness.controller.db_path) as conn:
            row = conn.execute(
                """SELECT state, conversation_id, server_incarnation
                     FROM experiment_attempt_claims
                    WHERE registration_id=?""",
                (bootstrap.registration_id,),
            ).fetchone()
        constructed_after_claim.append(
            bool(row and row[0] == "claimed" and row[1] and row[2])
        )
        return original_wec(*args, **kwargs)

    monkeypatch.setattr(
        "fastworkflow.run_fastapi_mcp.utils.WorkflowExecutionContext",
        checked_wec,
    )
    initialized = binding_harness.initialize(bootstrap)
    assert initialized.status_code == 200
    assert constructed_after_claim == [True]

    token = initialized.json()["access_token"]
    turn = binding_harness.client.post(
        "/perform_action",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "action": {
                "command_name": "add_two_numbers",
                "parameters": {"first_num": 2, "second_num": 3},
            }
        },
    )
    assert turn.status_code == 200

    with sqlite3.connect(binding_harness.controller.db_path) as conn:
        conn.row_factory = sqlite3.Row
        claim = conn.execute(
            """SELECT conversation_id, epoch, server_incarnation
                 FROM experiment_attempt_claims
                WHERE registration_id=?""",
            (bootstrap.registration_id,),
        ).fetchone()
        conversation = conn.execute(
            """SELECT experiment_id, task_id, attempt
                 FROM conversations
                WHERE channel_id=? AND conversation_id=?""",
            (bootstrap.channel_id, claim["conversation_id"]),
        ).fetchone()
        stored_turn = conn.execute(
            """SELECT experiment_id, task_id, attempt, claim_epoch,
                      server_incarnation, conversation_id
                 FROM turns
                WHERE channel_id=?
                ORDER BY rowid DESC LIMIT 1""",
            (bootstrap.channel_id,),
        ).fetchone()

    assert tuple(conversation) == ("exp-http", "task-1", 1)
    assert tuple(stored_turn) == (
        "exp-http",
        "task-1",
        1,
        claim["epoch"],
        claim["server_incarnation"],
        claim["conversation_id"],
    )


@pytest.mark.parametrize(
    "failure", ["unknown", "guess", "expired", "channel", "store"]
)
def test_invalid_bootstraps_fail_before_session_work(binding_harness, failure):
    bootstrap = binding_harness.register()
    payload = {
        "registration_id": bootstrap.registration_id,
        "secret": bootstrap.secret,
        "store_id": binding_harness.controller.store_identity,
    }
    channel_id = bootstrap.channel_id
    if failure == "unknown":
        payload["registration_id"] = "reg-does-not-exist"
    elif failure == "guess":
        payload["secret"] = "0" * len(bootstrap.secret)
    elif failure == "expired":
        with sqlite3.connect(binding_harness.controller.db_path) as conn:
            conn.execute(
                "UPDATE experiment_attempt_claims SET expires_at=? "
                "WHERE registration_id=?",
                (time.time() - 1, bootstrap.registration_id),
            )
            conn.commit()
    elif failure == "channel":
        channel_id = "registered:wrong-channel"
    elif failure == "store":
        payload["store_id"] = "different-store"

    response = binding_harness.client.post(
        "/initialize",
        json={"channel_id": channel_id, "experiment_bootstrap": payload},
    )

    assert response.status_code == 403
    assert channel_id not in binding_harness.main.session_manager._sessions


def test_replayed_bootstrap_and_unauthenticated_reissue_are_refused(
    binding_harness,
):
    bootstrap = binding_harness.register()
    first = binding_harness.initialize(bootstrap)
    assert first.status_code == 200

    reissue = binding_harness.client.post(
        "/initialize", json={"channel_id": bootstrap.channel_id}
    )
    assert reissue.status_code == 401

    asyncio.run(
        binding_harness.main.session_manager.remove_session(bootstrap.channel_id)
    )
    replay = binding_harness.initialize(bootstrap)
    assert replay.status_code == 403
    assert bootstrap.channel_id not in binding_harness.main.session_manager._sessions


def test_authenticated_checkpoint_restore_rebinds_current_claim(binding_harness):
    bootstrap = binding_harness.register()
    initialized = binding_harness.initialize(bootstrap)
    assert initialized.status_code == 200
    assert binding_harness.main.session_manager.checkpoint_for_shutdown([]) == 1
    asyncio.run(
        binding_harness.main.session_manager.remove_session(bootstrap.channel_id)
    )

    response = binding_harness.client.get(
        "/conversations",
        headers={
            "Authorization": f"Bearer {initialized.json()['access_token']}"
        },
    )

    assert response.status_code == 200
    restored = binding_harness.main.session_manager._sessions[bootstrap.channel_id]
    assert restored.execution_context.observability_experiment_claim["epoch"] == 1


def test_checkpoint_epoch_mismatch_fails_before_restored_session(
    binding_harness,
):
    bootstrap = binding_harness.register()
    initialized = binding_harness.initialize(bootstrap)
    assert initialized.status_code == 200
    assert binding_harness.main.session_manager.checkpoint_for_shutdown([]) == 1
    asyncio.run(
        binding_harness.main.session_manager.remove_session(bootstrap.channel_id)
    )
    with sqlite3.connect(binding_harness.controller.db_path) as conn:
        conn.execute(
            """UPDATE experiment_attempt_claims
                  SET epoch=epoch + 1
                WHERE registration_id=?""",
            (bootstrap.registration_id,),
        )
        conn.commit()

    response = binding_harness.client.get(
        "/conversations",
        headers={
            "Authorization": f"Bearer {initialized.json()['access_token']}"
        },
    )

    assert response.status_code == 409
    assert bootstrap.channel_id not in binding_harness.main.session_manager._sessions


def test_ordinary_initialize_never_claims_and_remains_compatible(
    binding_harness, monkeypatch
):
    def unexpected_claim(*args, **kwargs):
        raise AssertionError("ordinary initialize performed an experiment lookup")

    monkeypatch.setattr(obs.ObservabilityStore, "claim_attempt", unexpected_claim)
    response = binding_harness.client.post(
        "/initialize", json={"channel_id": "ordinary-channel"}
    )

    assert response.status_code == 200
    runtime = binding_harness.main.session_manager._sessions["ordinary-channel"]
    assert runtime.execution_context.observability_experiment_claim == {}


def test_bootstrap_refused_when_the_observability_store_cannot_open(
    binding_harness, monkeypatch
):
    bootstrap = binding_harness.register()
    monkeypatch.setattr(
        "fastworkflow.run_fastapi_mcp.utils.get_observability_sink",
        lambda workflow_path: None,
    )

    response = binding_harness.initialize(bootstrap)

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "could not be opened" in detail
    assert "enabled" not in detail
    assert bootstrap.channel_id not in binding_harness.main.session_manager._sessions


def test_bootstrap_refused_when_claim_feature_is_missing(
    binding_harness, monkeypatch
):
    bootstrap = binding_harness.register()

    def missing_feature(db_path):
        raise MissingExperimentLifecycleFeature(db_path)

    monkeypatch.setattr(
        "fastworkflow.run_fastapi_mcp.utils.experiment_store_readiness",
        missing_feature,
    )
    response = binding_harness.initialize(bootstrap)

    assert response.status_code == 503
    assert bootstrap.channel_id not in binding_harness.main.session_manager._sessions


# ----------------------------------------------------------------------
# fix-qe2: the runtime snapshot -- answered by the probe, stamped on bind
# ----------------------------------------------------------------------


def _attempt_row(binding_harness, task_id="task-1", attempt=1):
    rows = binding_harness.controller.store.experiment_attempt_rows(
        "exp-http", task_id=task_id
    )
    return next(r for r in rows if int(r["attempt"]) == attempt)


def _raw_stamp(binding_harness):
    with sqlite3.connect(binding_harness.controller.db_path) as conn:
        return conn.execute(
            """SELECT runtime_snapshot_json FROM experiment_attempts
                WHERE experiment_id='exp-http' AND task_id='task-1' AND attempt=1"""
        ).fetchone()[0]


def test_the_runtime_snapshot_is_absent_unless_asked_for(binding_harness):
    """Probes are frequent; this is opt-in like ?memory and ?observability."""
    assert "runtime" not in binding_harness.client.get("/probes/readyz").json()


def test_the_runtime_probe_reports_a_valid_credential_free_snapshot(
    binding_harness,
):
    response = binding_harness.client.get("/probes/readyz?runtime=true")

    assert response.status_code == 200
    body = response.json()
    runtime = body["runtime"]
    assert body["status"] == "ready"
    assert runtime["configuration_valid"] is True
    assert runtime["runtime_metadata_registered"] is True
    assert runtime["pid"] == os.getpid()
    assert runtime["capture_profile"] == "evidence"
    assert runtime["command_surface_count"] > 0
    assert isinstance(runtime["effective_features"], dict)
    # The snapshot itself carries no path, no store location, no env value.
    # (The surrounding body's `experiment_store_readiness.resolved_path` is
    # the pre-existing fix-rj2 handshake, outside this snapshot.)
    rendered = json.dumps(runtime)
    assert binding_harness.workflow_path not in rendered
    assert binding_harness.controller.db_path not in rendered
    for word in ("KEY", "SECRET", "TOKEN", "PASSWORD"):
        assert word not in rendered.upper()


def test_an_invalid_runtime_configuration_makes_the_pod_not_ready(
    binding_harness, monkeypatch
):
    monkeypatch.setattr(
        "fastworkflow.run_fastapi_mcp.__main__.runtime_readiness_snapshot",
        lambda _path: {"configuration_valid": False, "runtime_metadata_registered": False},
    )

    response = binding_harness.client.get("/probes/readyz?runtime=true")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["runtime_configuration"] == "invalid"
    # Without the flag the probe keeps its ordinary answer.
    assert binding_harness.client.get("/probes/readyz").status_code == 200


def test_binding_stamps_the_servers_runtime_snapshot_on_the_attempt(
    binding_harness,
):
    """Taken in-process at the bind, from the same function the probe answers
    with: what the attempt record says about its server is what the driver
    could have asserted against."""
    bootstrap = binding_harness.register()
    assert binding_harness.initialize(bootstrap).status_code == 200

    probed = binding_harness.client.get("/probes/readyz?runtime=true").json()["runtime"]
    row = _attempt_row(binding_harness)

    assert row["runtime_snapshot"] == probed
    assert row["runtime_snapshot"]["configuration_valid"] is True
    assert row["runtime_snapshot"]["pid"] == os.getpid()
    # Projected once, decoded; the raw column is not duplicated.
    assert "runtime_snapshot_json" not in row
    # And it is a JSON document on disk, readable without this code.
    assert json.loads(_raw_stamp(binding_harness)) == probed


def test_binding_records_null_when_the_snapshot_is_unavailable(
    binding_harness, monkeypatch
):
    """A probe failure never blocks the bind. The attempt runs, and the null
    stamp is visible evidence that the configuration was not certified."""

    def unavailable(_path):
        raise RuntimeError("no runtime description")

    monkeypatch.setattr(
        "fastworkflow.run_fastapi_mcp.utils.runtime_readiness_snapshot",
        unavailable,
    )
    bootstrap = binding_harness.register()

    assert binding_harness.initialize(bootstrap).status_code == 200
    row = _attempt_row(binding_harness)
    assert row["runtime_snapshot"] is None
    assert "runtime_snapshot_json" not in row
    assert _raw_stamp(binding_harness) is None
