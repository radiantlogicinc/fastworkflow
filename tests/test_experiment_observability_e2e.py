"""FastWorkflow-only external experiment observability, end to end."""

from __future__ import annotations

import hashlib
import importlib
import json
import multiprocessing
import os
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import fastworkflow
from fastworkflow.observability import store as obs
from fastworkflow import tracing
from fastworkflow.experiment.runner import ExperimentController
from fastworkflow.observability.workspace import WORKSPACE_SCHEMA
from fastworkflow.review.adapters import (
    exp028_answer_rating_to_sidecar_export,
    ido_rating_to_sidecar_export,
    sidecar_export_to_exp028_answer_rating,
    sidecar_export_to_ido_rating,
)
from fastworkflow.run_chatbot.server import ChatbotServer
from fastworkflow.turn import TurnOutput, TurnResult, TurnStatus


def _race_claim_process(
    db_path: str,
    bootstrap: dict[str, Any],
    incarnation: str,
    barrier: Any,
    outcomes: Any,
) -> None:
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_bytes(path: Path) -> dict[str, str]:
    return {
        str(candidate): _sha256(candidate)
        for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm"))
        if candidate.exists()
    }


def _archive_unchanged(source: Path, destination: Path) -> dict[str, Any]:
    before = _source_bytes(source)
    archive = obs.ObservabilityStore(str(source), migrate=False).archive_to(
        str(destination)
    )
    assert _source_bytes(source) == before
    assert archive["source_bytes_verified_unchanged"] is True
    assert archive["sidecar_free"] is True
    assert not Path(f"{destination}-wal").exists()
    assert not Path(f"{destination}-shm").exists()
    return archive


def _http_request(
    server: ChatbotServer,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    *,
    capability: str | None = None,
) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.port}{path}",
        method=method,
        data=None if body is None else json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {server.token}",
            "Content-Type": "application/json",
        },
    )
    if capability is not None:
        request.add_header("X-Review-Capability", capability)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _turn_row(
    turn_key: str,
    experiment_id: str,
    task_id: str,
    channel_id: str,
) -> dict[str, Any]:
    return {
        "turn_key": turn_key,
        "channel_id": channel_id,
        "conversation_id": None,
        "ordinal": None,
        "user_message": f"run {task_id}",
        "refined_user_message": None,
        "entry_workflow_name": "observability-e2e",
        "entry_context": "historical",
        "status": "completed",
        "success": 1,
        "failure_reason": None,
        "answer": "historical answer",
        "conversation_summary": "historical summary",
        "conversation_traces": "historical trace",
        "started_at": "2026-09-04T00:00:00+00:00",
        "completed_at": "2026-09-04T00:00:01+00:00",
        "suspended_ms": 0,
        "continuation_of": None,
        "record_version": 1,
        "experiment_id": experiment_id,
        "task_id": task_id,
        "attempt": 1,
        "claim_epoch": None,
        "server_incarnation": None,
        "record_json": json.dumps(
            {
                "turn_output": {
                    "turn_key": turn_key,
                    "status": "completed",
                    "success": True,
                    "answer": "historical answer",
                }
            }
        ),
    }


def _assignment(
    assignment_id: str,
    turn_ref: dict[str, str],
    question_ids: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "id": assignment_id,
        "rater_slots": ["rater-a", "rater-b"],
        "adjudicator_slots": ["adjudicator"],
        "blinded": True,
        "rows": [{"id": "shared-row", "turn_ref": turn_ref}],
        "questions": [
            {
                "id": question_id,
                "prompt": f"Rate {question_id}.",
                "type": "bounded-note",
                "max_length": 80,
            }
            for question_id in question_ids
        ],
    }


def _exercise_review_contract(
    server: ChatbotServer,
    assignment: dict[str, Any],
    to_contract: Any,
    to_sidecar: Any,
) -> list[str]:
    status, created = _http_request(
        server, "POST", "/api/review/assignments", assignment
    )
    assert status == 201
    answer_path = f"/api/review/assignments/{assignment['id']}/answers"
    question_id = assignment["questions"][0]["id"]
    submitted: list[str] = []
    for rater_slot in ("rater-a", "rater-b"):
        capability = created["rater_capabilities"][rater_slot]
        for revision in (1, 2):
            answer = f"{assignment['id']}-{rater_slot}-revision-{revision}"
            submitted.append(answer)
            answer_status, payload = _http_request(
                server,
                "POST",
                answer_path,
                {
                    "row_id": "shared-row",
                    "question_id": question_id,
                    "answer": answer,
                },
                capability=capability,
            )
            assert answer_status == 200
            assert payload["answer"]["revision"] == revision

    export_status, exported = _http_request(
        server,
        "GET",
        f"/api/review/assignments/{assignment['id']}/export",
        capability=created["adjudicator_capabilities"]["adjudicator"],
    )
    assert export_status == 200
    sidecar_export = exported["export"]
    contract = to_contract(sidecar_export)
    assert contract["schema"] == assignment["id"]
    assert to_sidecar(json.loads(json.dumps(contract))) == sidecar_export
    return submitted


@pytest.fixture
def loopback_observability(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    package_path = fastworkflow.get_fastworkflow_package_path()
    workflow_path = os.path.join(package_path, "examples", "hello_world")
    if not os.path.isdir(workflow_path):
        pytest.skip(f"hello_world workflow not found at {workflow_path}")

    root_a = tmp_path / "state-a"
    root_b = tmp_path / "state-b"
    root_a.mkdir()
    root_b.mkdir()
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(root_a))
    monkeypatch.setenv(obs.CAPTURE_PROFILE_VAR, "evidence")
    monkeypatch.setenv("FW_OBS_INLINE_ARTIFACT_BYTES", "1")
    monkeypatch.setattr(
        sys, "argv", ["pytest", "--workflow_path", workflow_path]
    )

    import fastworkflow.run_fastapi_mcp.__main__ as fastapi_main

    fastapi_main = importlib.reload(fastapi_main)
    with TestClient(fastapi_main.app) as client:
        readiness = client.get("/probes/readyz").json()[
            "experiment_store_readiness"
        ]
        controller = ExperimentController(
            readiness["resolved_path"],
            readiness["store_id"],
            migrate=False,
            external=True,
            capture_profile=readiness["capture_profile"],
            capture_policy_version=readiness["capture_policy_version"],
        )
        for experiment_id, task_id in (
            ("exp-pending", "task-pending"),
            ("exp-capture", "task-capture"),
            ("exp-race", "task-race"),
            ("exp-stale", "task-stale"),
        ):
            controller.create_experiment(
                experiment_id,
                experiment_id,
                declared_tasks=1,
                declared_attempts=1,
                declarations=[(task_id, 1, f"driver-{task_id}")],
                required_evidence_segments=(
                    1 if experiment_id == "exp-capture" else 0
                ),
            )

        http_turns: dict[str, str] = {}
        for experiment_id, task_id in (
            ("exp-pending", "task-pending"),
            ("exp-capture", "task-capture"),
        ):
            bootstrap = controller.register_attempt(
                experiment_id,
                task_id,
                1,
                f"driver-{task_id}",
                f"registered:{task_id}:1",
            )
            initialized = client.post(
                "/initialize",
                json={
                    "channel_id": bootstrap.channel_id,
                    "experiment_bootstrap": {
                        "registration_id": bootstrap.registration_id,
                        "secret": bootstrap.secret,
                        "store_id": controller.store_identity,
                    },
                },
            )
            assert initialized.status_code == 200
            token = initialized.json()["access_token"]
            action = client.post(
                "/perform_action",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "action": {
                        "command_name": "add_two_numbers",
                        "parameters": {"first_num": 2, "second_num": 3},
                    }
                },
            )
            assert action.status_code == 200
            assert action.json()["status"] == "completed"
            http_turns[task_id] = (
                action.json().get("logical_turn_key")
                or action.json()["turn_key"]
            )

            reissue = client.post(
                "/initialize", json={"channel_id": bootstrap.channel_id}
            )
            assert reissue.status_code == 401
            controller.terminalize_attempt(
                experiment_id, task_id, 1, execution_status="completed"
            )

        controller.record_outcome(
            "exp-capture",
            "task-capture",
            1,
            outcome="pass",
            outcome_source="external-grader",
        )
        race_bootstrap = controller.register_attempt(
            "exp-race",
            "task-race",
            1,
            "driver-task-race",
            "registered:task-race:1",
        )
        context = multiprocessing.get_context("spawn")
        barrier = context.Barrier(2)
        outcomes = context.Queue()
        processes = [
            context.Process(
                target=_race_claim_process,
                args=(
                    controller.db_path,
                    race_bootstrap.as_dict(),
                    f"server-incarnation-{index}",
                    barrier,
                    outcomes,
                ),
            )
            for index in range(2)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
        race_results = [outcomes.get(timeout=2) for _ in processes]
        assert [result[0] for result in race_results].count("claimed") == 1
        assert [result[0] for result in race_results].count("refused") == 1

        stale_bootstrap = controller.register_attempt(
            "exp-stale",
            "task-stale",
            1,
            "driver-task-stale",
            "registered:task-stale:1",
        )
        stale_claim = controller.claim_attempt(
            stale_bootstrap,
            server_incarnation="server-stale-old",
            lease_seconds=0.001,
        )
        stale_result = TurnResult(
            turn_output=TurnOutput(
                turn_key="queued-stale-turn",
                status=TurnStatus.COMPLETED,
                answer="must not persist",
                command_outputs=[
                    fastworkflow.CommandOutput(
                        command_name="artifact",
                        command_response=fastworkflow.CommandResponse(
                            response="ok",
                            artifacts={"payload": "queued stale artifact"},
                        ),
                    )
                ],
            ),
            channel_id=stale_claim.channel_id,
            conversation_id=stale_claim.conversation_id,
            user_message="stale",
            experiment_id=stale_claim.experiment_id,
            task_id=stale_claim.task_id,
            attempt=stale_claim.attempt,
            claim_epoch=stale_claim.epoch,
            server_incarnation=stale_claim.server_incarnation,
        )
        stale_span = tracing.Span(
            span_id="queued-stale-span",
            trace_id="queued-stale-turn",
            name=tracing.SPAN_TURN,
            channel_id=stale_claim.channel_id,
            experiment_id=stale_claim.experiment_id,
            task_id=stale_claim.task_id,
            attempt=stale_claim.attempt,
            claim_epoch=stale_claim.epoch,
            server_incarnation=stale_claim.server_incarnation,
        )
        sink = obs.sink_for_db_path(controller.db_path)
        assert sink is not None
        writer_entered = threading.Event()
        release_writer = threading.Event()
        original_apply_batch = sink._apply_batch

        def paused_apply_batch(conn: sqlite3.Connection, items: list[Any]) -> None:
            writer_entered.set()
            assert release_writer.wait(timeout=10)
            original_apply_batch(conn, items)

        monkeypatch.setattr(sink, "_apply_batch", paused_apply_batch)
        sink._sync_breaker_until = time.monotonic() + 300
        assert sink.emit_turn_record(stale_result) is False
        sink.emit_span(stale_span)
        assert writer_entered.wait(timeout=10)
        time.sleep(0.01)
        replacement = controller.register_attempt(
            "exp-stale",
            "task-stale",
            1,
            "driver-task-stale",
            "registered:task-stale:1",
            recovery={"owned_process_dead": True},
        )
        controller.claim_attempt(
            replacement, server_incarnation="server-stale-new"
        )
        release_writer.set()
        assert sink.flush(timeout=10)
        monkeypatch.setattr(sink, "_apply_batch", original_apply_batch)

        assert controller.store.get_turn("queued-stale-turn") is None
        assert controller.store.get_spans("queued-stale-turn") == []
        with sqlite3.connect(controller.db_path) as conn:
            diagnostic = json.loads(
                conn.execute(
                    "SELECT value FROM diagnostics "
                    "WHERE key='stale_claim_records'"
                ).fetchone()[0]
            )
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM artifacts "
                    "WHERE turn_key='queued-stale-turn'"
                ).fetchone()[0]
                == 0
            )
        assert diagnostic["count"] >= 2
        assert diagnostic["invalidating"] is True

        controller.drain_before_certify()
        controller.record_evidence_segment(
            "exp-capture",
            1,
            "capture-segment",
            {
                "valid": True,
                "started_at": "2026-09-04T00:00:00+00:00",
                "completed_at": "2026-09-04T00:01:00+00:00",
                "problems": [],
            },
        )
        assert controller.complete_experiment("exp-pending") == (
            "awaiting_evaluation"
        )
        assert controller.complete_experiment("exp-capture") == "capture_complete"
        assert (
            controller.store.experiment_scores("exp-capture")["reportable"]
            is False
        )

    source_a = Path(controller.db_path)
    source_b = root_b / "observability.sqlite3"
    store_b = obs.ObservabilityStore(str(source_b))
    store_b.create_experiment(
        "exp-native-b",
        "native second store",
        declared_tasks=1,
        declared_attempts=1,
    )
    store_b.start_attempt(
        "exp-native-b", "task-native-b", 1, "channel-native-b"
    )
    store_b.finish_attempt(
        "exp-native-b",
        "task-native-b",
        1,
        outcome="pass",
        outcome_source="historical-grader",
    )
    with store_b._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        assert store_b.upsert_turn_row(
            conn,
            _turn_row(
                "turn-native-b",
                "exp-native-b",
                "task-native-b",
                "channel-native-b",
            ),
            [],
            store_b._store_redactor(),
        )
        conn.commit()

    identity_b = store_b.store_identity()
    assert identity_b is not None
    assert controller.store_identity != identity_b
    archives = tmp_path / "archives"
    archives.mkdir()
    archive_a = _archive_unchanged(source_a, archives / "store-a.sqlite3")
    archive_b = _archive_unchanged(source_b, archives / "store-b.sqlite3")
    turn_ref_a = {
        "store_id": "store-a",
        "logical_turn_key": http_turns["task-pending"],
    }
    manifest = archives / "workspace.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": WORKSPACE_SCHEMA,
                "workspace_id": "observability-e2e",
                "label": "External experiment observability E2E",
                "stores": [
                    {
                        "store_id": "store-a",
                        "label": "HTTP capture",
                        "path": Path(archive_a["path"]).name,
                        "mode": "sealed",
                        "sha256": archive_a["sha256"],
                        "store_identity": archive_a["store_identity"],
                    },
                    {
                        "store_id": "store-b",
                        "label": "Historical capture",
                        "path": Path(archive_b["path"]).name,
                        "mode": "sealed",
                        "sha256": archive_b["sha256"],
                        "store_identity": archive_b["store_identity"],
                    },
                ],
                "experiments": [
                    {
                        "experiment_id": "logical-native",
                        "segments": [
                            {
                                "store_id": "store-a",
                                "local_experiment_id": "exp-pending",
                            },
                            {
                                "store_id": "store-b",
                                "local_experiment_id": "exp-native-b",
                            },
                        ],
                    }
                ],
                "projected_attempts": [
                    {
                        "logical_attempt": {
                            "experiment_id": "logical-projected",
                            "task_id": "task-projected",
                            "attempt": 1,
                        },
                        "attempt_refs": [
                            {
                                "store_id": "store-a",
                                "local_experiment_id": "exp-pending",
                                "task_id": "task-pending",
                                "attempt": 1,
                                "turn_ref": turn_ref_a,
                            },
                            {
                                "store_id": "store-b",
                                "local_experiment_id": "exp-native-b",
                                "task_id": "task-native-b",
                                "attempt": 1,
                                "turn_ref": {
                                    "store_id": "store-b",
                                    "logical_turn_key": "turn-native-b",
                                },
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    yield {
        "controller": controller,
        "manifest": manifest,
        "archive_a": Path(archive_a["path"]),
        "archive_b": Path(archive_b["path"]),
        "turn_ref_a": turn_ref_a,
        "turn_key_a": http_turns["task-pending"],
    }


def test_external_experiment_observability_end_to_end(loopback_observability):
    evidence = loopback_observability
    archive_before = {
        path: path.read_bytes()
        for path in (evidence["archive_a"], evidence["archive_b"])
    }
    server = ChatbotServer(
        port=0, workspace_manifest_path=str(evidence["manifest"])
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    submitted_answers: list[str] = []
    try:
        assert len(_http_request(server, "GET", "/api/workspace/stores")[1]["stores"]) == 2
        native_attempts = _http_request(
            server,
            "GET",
            "/api/workspace/experiment/logical-native/attempts",
        )[1]["attempts"]
        assert {attempt["store_id"] for attempt in native_attempts} == {
            "store-a",
            "store-b",
        }
        projected = _http_request(
            server,
            "GET",
            "/api/workspace/projected_attempts"
            "?experiment=logical-projected&task=task-projected&attempt=1",
        )[1]["projected_attempts"]
        assert {
            source["resolved_turn"]["store_id"]
            for source in projected[0]["resolved_sources"]
        } == {"store-a", "store-b"}
        scoped_turn = _http_request(
            server,
            "GET",
            f"/api/workspace/turn/store-a/{evidence['turn_key_a']}",
        )[1]["turn"]
        assert scoped_turn["task_id"] == "task-pending"

        ido_assignment = _assignment(
            "ido-rating-v1",
            evidence["turn_ref_a"],
            (
                "primary",
                "contributing_causes",
                "coverage",
                "declines",
                "note",
            ),
        )
        submitted_answers.extend(
            _exercise_review_contract(
                server,
                ido_assignment,
                sidecar_export_to_ido_rating,
                ido_rating_to_sidecar_export,
            )
        )
        exp028_assignment = _assignment(
            "exp028-answer-rating-v1",
            evidence["turn_ref_a"],
            ("presented", "not_presented", "overclaimed", "not_decidable"),
        )
        submitted_answers.extend(
            _exercise_review_contract(
                server,
                exp028_assignment,
                sidecar_export_to_exp028_answer_rating,
                exp028_answer_rating_to_sidecar_export,
            )
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)

    assert {
        path: path.read_bytes()
        for path in (evidence["archive_a"], evidence["archive_b"])
    } == archive_before
    controller = evidence["controller"]
    assert controller.store.get_feedback(evidence["turn_key_a"]) is None
    turn = controller.store.get_turn(evidence["turn_key_a"])
    assert turn is not None
    memory = controller.store.get_memory_window(
        turn["channel_id"], turn["conversation_id"], 10
    )
    serialized_memory = json.dumps(memory)
    assert all(answer not in serialized_memory for answer in submitted_answers)
