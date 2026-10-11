"""Focused cross-process claim, epoch, sink, and checkpoint fencing tests."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import fastworkflow
from fastworkflow.observability import store as obs
from fastworkflow import state_paths, tracing
from fastworkflow.checkpoint_store import (
    PROTOCOL_VERSION,
    ChannelCheckpointStore,
    CheckpointIdentity,
)
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.experiment.runner import ExperimentController
from fastworkflow.run_fastapi_mcp import checkpoint
from fastworkflow.run_fastapi_mcp.utils import refuse_registered_token_reissue
from fastworkflow.session_state_store import SCHEMA_VERSION as PENDING_SCHEMA_VERSION
from fastworkflow.turn import TurnOutput, TurnResult, TurnStatus
from fastworkflow.workflow_execution_context import WorkflowExecutionContext


@pytest.fixture
def controller(tmp_path, monkeypatch):
    folder = str(tmp_path / "workflow")
    store = obs.ObservabilityStore(state_paths.observability_db(folder))
    controller = ExperimentController(
        folder, store.store_identity(), migrate=False, external=True
    )
    controller.create_experiment(
        "exp-1",
        "claim fencing",
        declared_tasks=1,
        declared_attempts=1,
        declarations=[("task-1", 1, "driver-job-1")],
    )
    return controller


def _register(controller, *, ttl=300.0):
    return controller.register_attempt(
        "exp-1",
        "task-1",
        1,
        "driver-job-1",
        "registered:task-1:1",
        ttl_seconds=ttl,
    )


def _claim(controller, bootstrap, incarnation, *, lease=300.0):
    return controller.claim_attempt(
        bootstrap,
        server_incarnation=incarnation,
        lease_seconds=lease,
    )


def _replace(controller, *, incarnation="server-2"):
    bootstrap = controller.register_attempt(
        "exp-1",
        "task-1",
        1,
        "driver-job-1",
        "registered:task-1:1",
        recovery={"owned_process_dead": True},
    )
    return _claim(controller, bootstrap, incarnation)


def _turn_result(claim, *, artifact=None, status=TurnStatus.COMPLETED):
    command_outputs = []
    if artifact is not None:
        command_outputs.append(
            fastworkflow.CommandOutput(
                command_name="artifact",
                command_response=fastworkflow.CommandResponse(
                    response="ok", artifacts={"payload": artifact}
                ),
            )
        )
    return TurnResult(
        turn_output=TurnOutput(
            turn_key="turn-fixed",
            status=status,
            answer="ok",
            command_outputs=command_outputs,
        ),
        channel_id=claim.channel_id,
        conversation_id=claim.conversation_id,
        user_message="hello",
        experiment_id=claim.experiment_id,
        task_id=claim.task_id,
        attempt=claim.attempt,
        claim_epoch=claim.epoch,
        server_incarnation=claim.server_incarnation,
    )


def _race_claim_process(db_path, bootstrap, incarnation, barrier, outcomes):
    store = obs.ObservabilityStore(db_path, migrate=False)
    barrier.wait()
    try:
        claim = store.claim_attempt(
            bootstrap,
            channel_id=bootstrap["channel_id"],
            server_incarnation=incarnation,
        )
        outcomes.put(("claimed", claim["server_incarnation"]))
    except obs.AttemptClaimError as exc:
        outcomes.put(("refused", type(exc).__name__))


def test_secret_has_entropy_is_hashed_and_expires(controller):
    bootstrap = _register(controller)
    assert len(bytes.fromhex(bootstrap.secret)) == 32
    with sqlite3.connect(controller.db_path) as conn:
        stored = conn.execute(
            "SELECT secret_hash FROM experiment_attempt_claims"
        ).fetchone()[0]
    assert bootstrap.secret not in stored
    assert stored == hashlib.sha256(bootstrap.secret.encode()).hexdigest()

    expiring = controller.store.register_attempt
    with sqlite3.connect(controller.db_path) as conn:
        conn.execute(
            "UPDATE experiment_attempt_claims SET expires_at=?",
            (time.time() - 1,),
        )
        conn.commit()
    with pytest.raises(obs.AttemptClaimError, match="expired"):
        controller.store.claim_attempt(
            bootstrap.as_dict(),
            channel_id=bootstrap.channel_id,
            server_incarnation="server-expired",
        )
    assert expiring is not None


def test_secret_replay_guess_and_channel_mismatch_are_refused(controller):
    bootstrap = _register(controller)
    wrong = dict(bootstrap.as_dict(), secret="0" * len(bootstrap.secret))
    with pytest.raises(obs.AttemptClaimError, match="secret mismatch"):
        controller.store.claim_attempt(
            wrong,
            channel_id=bootstrap.channel_id,
            server_incarnation="server-wrong",
        )
    with pytest.raises(obs.AttemptClaimError, match="channel mismatch"):
        controller.store.claim_attempt(
            bootstrap.as_dict(),
            channel_id="registered:other",
            server_incarnation="server-other",
        )
    claim = _claim(controller, bootstrap, "server-good")
    with pytest.raises(obs.AttemptClaimError, match="already consumed"):
        _claim(controller, bootstrap, "server-replay")
    assert claim.epoch == 1


def test_two_process_stores_race_one_claim_exactly_one_wins(controller, monkeypatch):
    # Spawn children inherit sys.path; another repo's `tests` package may precede ours.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    bootstrap = _register(controller)
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    outcomes = context.Queue()
    processes = [
        context.Process(
            target=_race_claim_process,
            args=(
                controller.db_path,
                bootstrap.as_dict(),
                f"server-{index}",
                barrier,
                outcomes,
            ),
        )
        for index in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        # Covers spawn start-up (importing fastworkflow takes >10s on a loaded box);
        # the barrier, not this timeout, is what makes the race simultaneous.
        process.join(timeout=60)
        assert process.exitcode == 0
    results = [outcomes.get(timeout=2) for _ in processes]
    assert [result[0] for result in results].count("claimed") == 1
    assert [result[0] for result in results].count("refused") == 1


def test_claim_atomically_reserves_labelled_conversation_before_binding(controller):
    claim = _claim(controller, _register(controller), "server-1")
    with sqlite3.connect(controller.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM conversations WHERE channel_id=?",
            (claim.channel_id,),
        ).fetchone()
    assert (row["conversation_id"], row["experiment_id"], row["task_id"]) == (
        claim.conversation_id,
        claim.experiment_id,
        claim.task_id,
    )
    ctx = WorkflowExecutionContext(session_key=claim.channel_id)
    controller.bind_claim(ctx, claim)
    assert ctx.observability_conversation_id == claim.conversation_id


def test_lease_expiry_alone_cannot_issue_replacement(controller):
    first = _claim(controller, _register(controller), "server-1", lease=0.001)
    time.sleep(0.01)
    with pytest.raises(obs.AttemptClaimError, match="proof"):
        _register(controller)
    second = _replace(controller)
    assert second.epoch == first.epoch + 1


def test_stale_admission_and_command_dispatch_fail_after_replacement(controller):
    first = _claim(controller, _register(controller), "server-1", lease=0.001)
    ctx = WorkflowExecutionContext(session_key=first.channel_id)
    controller.bind_claim(ctx, first)
    time.sleep(0.01)
    _replace(controller)

    with pytest.raises(obs.StaleExperimentClaim):
        ctx.assert_experiment_claim_current()
    with pytest.raises(obs.StaleExperimentClaim):
        CommandExecutor.invoke_command(ctx, "anything")


def test_stale_turn_artifact_and_derived_records_invalidate(controller, monkeypatch):
    first = _claim(controller, _register(controller), "server-1", lease=0.001)
    monkeypatch.setenv("FW_OBS_INLINE_ARTIFACT_BYTES", "1")
    turn_row, artifacts = obs.serialize_turn_result(
        _turn_result(first, artifact="large artifact")
    )
    time.sleep(0.01)
    _replace(controller)

    with controller.store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        assert controller.store.upsert_turn_row(
            conn, turn_row, artifacts, controller.store._store_redactor()
        ) is False
        conn.commit()
    assert controller.store.get_turn("turn-fixed") is None
    assert not controller.store.get_artifact(artifacts[0]["artifact_id"])
    with pytest.raises(obs.StaleExperimentClaim):
        controller.record_evidence_segment(
            "exp-1",
            1,
            "evr-stale",
            {"valid": True},
            claim=first,
        )
    with sqlite3.connect(controller.db_path) as conn:
        diagnostic = json.loads(
            conn.execute(
                "SELECT value FROM diagnostics WHERE key='stale_claim_records'"
            ).fetchone()[0]
        )
    assert diagnostic["invalidating"] is True
    assert diagnostic["count"] >= 2
    assert controller.store.get_experiment("exp-1")["status"] == "invalid"


def test_root_span_epoch_transition_wins_and_late_old_update_is_rejected(controller):
    first = _claim(controller, _register(controller), "server-1", lease=0.001)
    root_id = tracing.root_span_id("turn-fixed")
    old_open = tracing.Span(
        span_id=root_id,
        trace_id="turn-fixed",
        name=tracing.SPAN_TURN,
        channel_id=first.channel_id,
        experiment_id=first.experiment_id,
        task_id=first.task_id,
        attempt=first.attempt,
        claim_epoch=first.epoch,
        server_incarnation=first.server_incarnation,
    )
    with controller.store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        controller.store.upsert_span_rows(
            conn, [old_open], controller.store._store_redactor()
        )
        conn.commit()

    time.sleep(0.01)
    second = _replace(controller)
    resumed = tracing.Span(
        **{
            **old_open.__dict__,
            "status": tracing.STATUS_OK,
            "end_ns": time.time_ns(),
            "claim_epoch": second.epoch,
            "server_incarnation": second.server_incarnation,
        }
    )
    late_old = tracing.Span(
        **{
            **old_open.__dict__,
            "status": tracing.STATUS_ERROR,
            "end_ns": time.time_ns() + 1,
        }
    )
    with controller.store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        controller.store.upsert_span_rows(
            conn, [resumed, late_old], controller.store._store_redactor()
        )
        conn.commit()
    stored = controller.store.get_spans("turn-fixed")[0]
    assert stored["claim_epoch"] == second.epoch
    assert stored["server_incarnation"] == second.server_incarnation
    assert stored["status"] == tracing.STATUS_OK


def test_normal_and_suspended_checkpoint_round_trip_epoch(
    controller, tmp_path, monkeypatch
):
    claim = _claim(controller, _register(controller), "server-1")
    ctx = WorkflowExecutionContext(session_key=claim.channel_id)
    controller.bind_claim(ctx, claim)
    pending = ctx.serialize_state(channel_id=claim.channel_id)
    restored_ctx = WorkflowExecutionContext(session_key=claim.channel_id)
    restored_ctx.apply_serialized_state(pending)
    assert restored_ctx.observability_experiment_claim["epoch"] == claim.epoch

    identity = CheckpointIdentity("dep", "fp", claim.channel_id, "session-1")
    store = ChannelCheckpointStore(str(tmp_path / "checkpoints"))
    generation = store.publish(
        identity,
        context={"workflow_context": {}, "command_contexts": {}},
        runtime={
            "experiment_claim": {
                key: value
                for key, value in claim.as_dict().items()
                if key
                in {
                    "experiment_id",
                    "task_id",
                    "attempt",
                    "epoch",
                    "server_incarnation",
                }
            }
        },
        startup={},
        launch_context={},
        state_version=1,
    )
    record = store.load(identity)
    workflow = SimpleNamespace(context={})
    monkeypatch.setattr(
        checkpoint.serialization_hooks,
        "restore_command_contexts",
        lambda workflow, contexts: None,
    )
    runtime = checkpoint.restore(
        workflow, record, current_launch={}, channel_id=claim.channel_id
    )
    assert generation == 1
    assert runtime["experiment_claim"]["epoch"] == claim.epoch
    assert PROTOCOL_VERSION == 2
    assert PENDING_SCHEMA_VERSION == 4


def test_registered_reissue_refused_but_ordinary_runtime_is_unchanged(controller):
    claim = _claim(controller, _register(controller), "server-1")
    registered_ctx = WorkflowExecutionContext(session_key=claim.channel_id)
    controller.bind_claim(registered_ctx, claim)
    with pytest.raises(HTTPException) as exc:
        refuse_registered_token_reissue(
            SimpleNamespace(execution_context=registered_ctx)
        )
    assert exc.value.status_code == 401

    ordinary = WorkflowExecutionContext(session_key="ordinary")
    ordinary.bind_observability_identity(channel_id="ordinary")
    refuse_registered_token_reissue(SimpleNamespace(execution_context=ordinary))
    blob = ordinary.serialize_state(channel_id="ordinary")
    restored = WorkflowExecutionContext(session_key="ordinary")
    restored.apply_serialized_state(blob)
    assert restored.observability_experiment_claim == {}
