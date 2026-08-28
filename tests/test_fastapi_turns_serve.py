"""Integration tests for serving turns from the store (Phase 3, fix-kw7.4).

Covers fix-85g.9/.10 as absorbed by the observability design (§3.4, §4, [R9],
[A39]):

- ``GET /turns/{turn_key}`` consults the in-memory TurnRegistry FIRST, then
  falls back to the observability store; it resolves BOTH the execution key a
  deferred 202 handed out and the workflow's logical turn key.
- Deferred 202 bodies carry ``logical_turn_key`` alongside the execution key
  once known, so a deferred caller can recover a completed turn from the store
  (which is keyed by the logical key only).
- Cross-channel reads are 404, indistinguishable from unknown keys ([A39]).
- ``GET /turns/{turn_key}/trace`` replays the turn's spans from the store,
  non-destructively and repeatably; the live streaming drain is untouched.

No mocks (testing_rules.mdc): everything drives real direct actions against
tests/hello_world_workflow through the real FastAPI app, with the real SQLite
observability store under a per-test temp state root (fastapi_hermetic). The
observability writer is asynchronous, so tests flush the sink before asserting
on store reads.
"""

from __future__ import annotations

import importlib
import os
import sys
import time
import uuid

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def workflow_path():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(project_root, "tests", "hello_world_workflow")
    if not os.path.isdir(path):
        pytest.skip(f"hello_world_workflow not found at {path}")
    return path


@pytest.fixture
def env_files():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_file = os.path.join(project_root, "env", ".env")
    passwords_file = os.path.join(project_root, "passwords", ".env")
    if not os.path.isfile(env_file) or not os.path.isfile(passwords_file):
        pytest.skip("env files missing for FastAPI tests")
    return env_file, passwords_file


@pytest.fixture
def app_module(workflow_path, env_files, tmp_path):
    env_file, passwords_file = env_files
    sys.argv = [
        "pytest",
        "--workflow_path",
        workflow_path,
        "--env_file_path",
        env_file,
        "--passwords_file_path",
        passwords_file,
    ]
    import fastworkflow.run_fastapi_mcp.__main__ as main

    importlib.reload(main)
    from tests.fastapi_hermetic import init_fastapi_hermetic_env, restore_fastapi_env

    previous_env = init_fastapi_hermetic_env(
        env_file, passwords_file, tmp_path / "workflow_contexts"
    )
    try:
        yield main
    finally:
        restore_fastapi_env(previous_env)


def _channel(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _add_action() -> dict:
    return {
        "command_name": "add_two_numbers",
        "parameters": {"first_num": 2.0, "second_num": 3.0},
    }


def _count_calls(call_log: str) -> int:
    if not os.path.isfile(call_log):
        return 0
    with open(call_log) as fh:
        return sum(bool(line.strip()) for line in fh)


def _flush_observability(workflow_path: str) -> None:
    """Block until everything the server emitted so far is in the store.

    The observability writer is a background thread; SQLiteTraceSink.flush()
    blocks until enqueued writes land. get_observability_sink returns the SAME
    process-wide sink the server's runtimes attached (one sink per DB path).
    """
    from fastworkflow.observability_store import get_observability_sink

    sink = get_observability_sink(workflow_path)
    if sink is None:
        pytest.skip("observability disabled (FW_OBSERVABILITY=0)")
    assert sink.flush(), "observability writer failed to flush"


def _init_tokens(client: TestClient, channel_id: str) -> dict:
    resp = client.post("/initialize", json={"channel_id": channel_id})
    assert resp.status_code == 200
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _poll_turn_until_done(
    client: TestClient,
    headers: dict,
    logical_key: str,
    workflow_path: str,
    deadline_seconds: float = 25,
) -> dict:
    """Poll GET /turns/{logical} until the completed turn is readable.

    After the registry drops a finished (non-retained) execution there is a
    short window before the async writer lands the record; flushing on a 404
    makes the store deterministic.
    """
    deadline = time.time() + deadline_seconds
    while time.time() < deadline:
        resp = client.get(f"/turns/{logical_key}", headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            if data["exec_state"] == "done":
                return data
        else:
            assert resp.status_code == 404  # registry retired, store not landed yet
            _flush_observability(workflow_path)
        time.sleep(0.3)
    pytest.fail("turn never became readable as done")


# ---------------------------------------------------------------------------
# [R9] 202 carries the logical key; registry-first lookup by either key
# ---------------------------------------------------------------------------

def test_deferred_202_carries_logical_key_and_registry_resolves_both_keys(
    app_module, workflow_path, tmp_path, monkeypatch
):
    call_log = str(tmp_path / "calls.log")
    monkeypatch.setenv("FW_TEST_ADD_CALL_LOG", call_log)
    monkeypatch.setenv("FW_TEST_ADD_SLEEP_SECONDS", "4")

    channel_id = _channel("serve_defer")
    with TestClient(app_module.app) as client:
        headers = _init_tokens(client, channel_id)

        resp = client.post(
            "/perform_action",
            headers=headers,
            json={"action": _add_action(), "timeout_seconds": 2},
        )
        assert resp.status_code == 202
        body = resp.json()
        exec_key = body["turn_key"]
        assert body["exec_state"] == "running"
        # [R9]: the deferred body hands out the logical key once known — the
        # work began well inside the 2s wait window, so it is known here.
        logical_key = body.get("logical_turn_key")
        assert logical_key, "202 body must carry logical_turn_key once known [R9]"
        assert logical_key != exec_key

        # Registry-first: the in-flight execution answers under EITHER key.
        by_exec = client.get(f"/turns/{exec_key}", headers=headers)
        assert by_exec.status_code == 200
        assert by_exec.json()["exec_state"] == "running"
        assert by_exec.json()["logical_turn_key"] == logical_key

        by_logical = client.get(f"/turns/{logical_key}", headers=headers)
        assert by_logical.status_code == 200
        assert by_logical.json()["turn_key"] == exec_key
        assert by_logical.json()["exec_state"] == "running"

        # GET is a read, never a submission: polling must not re-run the work.
        done = _poll_turn_until_done(client, headers, logical_key, workflow_path)

    assert done["status"] == "completed"
    assert done["success"] is True
    assert done["logical_turn_key"] == logical_key
    assert _count_calls(call_log) == 1


# ---------------------------------------------------------------------------
# Store fallback after completion (the DB is keyed by the logical key only)
# ---------------------------------------------------------------------------

def test_completed_turn_is_served_from_the_store(
    app_module, workflow_path, tmp_path, monkeypatch
):
    call_log = str(tmp_path / "calls.log")
    monkeypatch.setenv("FW_TEST_ADD_CALL_LOG", call_log)

    channel_id = _channel("serve_store")
    with TestClient(app_module.app) as client:
        headers = _init_tokens(client, channel_id)

        resp = client.post(
            "/perform_action",
            headers=headers,
            json={"action": _add_action(), "timeout_seconds": 30},
        )
        assert resp.status_code == 200
        body = resp.json()
        exec_key = body["turn_key"]
        logical_key = body["logical_turn_key"]
        assert logical_key and logical_key != exec_key

        _flush_observability(workflow_path)

        # A finished perform_action is not retained by the registry, so this
        # read is the store fallback: the stored record comes back parsed.
        read = client.get(f"/turns/{logical_key}", headers=headers)
        assert read.status_code == 200
        data = read.json()
        assert data["turn_key"] == logical_key
        assert data["logical_turn_key"] == logical_key
        assert data["exec_state"] == "done"
        assert data["status"] == "completed"
        assert data["success"] is True
        assert data["command_outputs"], "stored record lost its command outputs"
        # record_json parsed into the response — the full internal TurnResult.
        assert data["record"]["turn_output"]["turn_key"] == logical_key
        assert "5" in (data["answer"] or "")

        # Repeatable: a second read answers identically (reads are pure).
        again = client.get(f"/turns/{logical_key}", headers=headers)
        assert again.status_code == 200
        assert again.json() == data

        # The execution key was the registry's handle only; the store is keyed
        # by the logical key, so once the record retires it no longer resolves.
        by_exec = client.get(f"/turns/{exec_key}", headers=headers)
        assert by_exec.status_code == 404


# ---------------------------------------------------------------------------
# [A39] channel scoping: foreign and unknown keys are the same 404
# ---------------------------------------------------------------------------

def test_cross_channel_reads_are_indistinguishable_from_missing(
    app_module, workflow_path, tmp_path, monkeypatch
):
    call_log = str(tmp_path / "calls.log")
    monkeypatch.setenv("FW_TEST_ADD_CALL_LOG", call_log)
    monkeypatch.setenv("FW_TEST_ADD_SLEEP_SECONDS", "2")

    channel_a = _channel("serve_owner")
    channel_b = _channel("serve_intruder")
    with TestClient(app_module.app) as client:
        headers_a = _init_tokens(client, channel_a)
        headers_b = _init_tokens(client, channel_b)

        resp = client.post(
            "/perform_action",
            headers=headers_a,
            json={"action": _add_action(), "timeout_seconds": 1},
        )
        assert resp.status_code == 202
        exec_key = resp.json()["turn_key"]
        logical_key = resp.json()["logical_turn_key"]
        assert logical_key

        # While in flight (registry-served): the other channel sees nothing.
        assert client.get(f"/turns/{exec_key}", headers=headers_b).status_code == 404
        assert client.get(f"/turns/{logical_key}", headers=headers_b).status_code == 404
        # ... while the owner sees the running execution.
        assert client.get(f"/turns/{exec_key}", headers=headers_a).status_code == 200

        _poll_turn_until_done(client, headers_a, logical_key, workflow_path)

        # After completion (store-served): still nothing for the other channel,
        # for the turn and for its trace.
        cross_turn = client.get(f"/turns/{logical_key}", headers=headers_b)
        assert cross_turn.status_code == 404
        cross_trace = client.get(f"/turns/{logical_key}/trace", headers=headers_b)
        assert cross_trace.status_code == 404

        # An unknown key answers exactly the same way (no existence leak): the
        # 404 body differs only by the echoed key.
        unknown_key = f"missing{uuid.uuid4().hex}"
        unknown = client.get(f"/turns/{unknown_key}", headers=headers_b)
        assert unknown.status_code == 404
        assert cross_turn.json()["detail"].replace(logical_key, unknown_key) == (
            unknown.json()["detail"]
        )

        # The owner still reads its own turn and trace.
        assert client.get(f"/turns/{logical_key}", headers=headers_a).status_code == 200
        assert (
            client.get(f"/turns/{logical_key}/trace", headers=headers_a).status_code
            == 200
        )


# ---------------------------------------------------------------------------
# fix-85g.10: non-destructive trace replay from the spans table
# ---------------------------------------------------------------------------

def test_trace_replay_returns_spans_and_is_repeatable(
    app_module, workflow_path, tmp_path, monkeypatch
):
    call_log = str(tmp_path / "calls.log")
    monkeypatch.setenv("FW_TEST_ADD_CALL_LOG", call_log)

    channel_id = _channel("serve_trace")
    with TestClient(app_module.app) as client:
        headers = _init_tokens(client, channel_id)

        resp = client.post(
            "/perform_action",
            headers=headers,
            json={"action": _add_action(), "timeout_seconds": 30},
        )
        assert resp.status_code == 200
        logical_key = resp.json()["logical_turn_key"]
        assert logical_key

        _flush_observability(workflow_path)

        first = client.get(f"/turns/{logical_key}/trace", headers=headers)
        assert first.status_code == 200
        payload = first.json()
        assert payload["logical_turn_key"] == logical_key
        spans = payload["spans"]
        assert spans, "the turn produced no spans in the store"
        assert all(span["trace_id"] == logical_key for span in spans)

        names = {span["name"] for span in spans}
        assert "fw.turn" in names, f"missing fw.turn root span (got {sorted(names)})"
        root = next(span for span in spans if span["name"] == "fw.turn")
        assert isinstance(root["attributes"], dict)
        assert root["attributes"]["turn_key"] == logical_key
        assert root["attributes"]["channel_id"] == channel_id
        assert root["status"] != "open", "root span was never closed"

        # Non-destructive replay: reading again returns the same spans (unlike
        # the live streaming queue drain, which is read-once).
        second = client.get(f"/turns/{logical_key}/trace", headers=headers)
        assert second.status_code == 200
        assert second.json()["spans"] == spans


# ---------------------------------------------------------------------------
# Registry-first also covers retained (recently-completed) startup turns
# ---------------------------------------------------------------------------

def test_startup_turn_is_registry_served_before_and_after_completion(
    app_module, workflow_path, tmp_path, monkeypatch
):
    call_log = str(tmp_path / "calls.log")
    monkeypatch.setenv("FW_TEST_ADD_CALL_LOG", call_log)
    monkeypatch.setenv("FW_TEST_ADD_SLEEP_SECONDS", "3")

    channel_id = _channel("serve_startup")
    body = {
        "channel_id": channel_id,
        "user_id": "u_serve",
        "startup_action": _add_action(),
        "timeout_seconds": 1,
    }
    with TestClient(app_module.app) as client:
        resp = client.post("/initialize", json=body)
        assert resp.status_code == 202
        startup_key = resp.json()["startup_turn_key"]
        headers = {"Authorization": f"Bearer {resp.json()['access_token']}"}

        # In flight: registry answers by the execution key.
        running = client.get(f"/turns/{startup_key}", headers=headers)
        assert running.status_code == 200
        assert running.json()["exec_state"] == "running"

        # Poll /initialize (the three-state already-exists branch) to done.
        final = None
        deadline = time.time() + 20
        while time.time() < deadline:
            poll = client.post("/initialize", json=body)
            if poll.json()["startup_exec_state"] == "done":
                final = poll.json()
                break
            time.sleep(0.3)
        assert final is not None, "startup never completed"

        # [R9] the initialize envelope reports the logical key too, and it is
        # the startup TurnOutput's own key.
        logical_key = final["startup_logical_turn_key"]
        assert logical_key == final["startup_output"]["turn_key"]
        assert logical_key != startup_key

        # Recently completed: startup records are retained, so BOTH keys still
        # resolve registry-first — no `record` field, which only the store
        # fallback adds.
        by_exec = client.get(f"/turns/{startup_key}", headers=headers)
        assert by_exec.status_code == 200
        data = by_exec.json()
        assert data["exec_state"] == "done"
        assert data["status"] == "completed"
        assert data["logical_turn_key"] == logical_key
        assert "record" not in data

        by_logical = client.get(f"/turns/{logical_key}", headers=headers)
        assert by_logical.status_code == 200
        assert by_logical.json()["turn_key"] == startup_key

    assert _count_calls(call_log) == 1


# ---------------------------------------------------------------------------
# [R9] queued-window guard (epic review follow-up): while an execution is
# still QUEUED behind runtime.lock, the WEC's current_turn_key is the
# PREVIOUS turn's key (it is never cleared between turns) and pre_turn_key is
# still unset — resolve_logical_turn_key must not stamp the stale key.
# ---------------------------------------------------------------------------


def test_queued_execution_never_adopts_the_previous_turns_logical_key():
    from types import SimpleNamespace

    from fastworkflow.run_fastapi_mcp.turns import (
        ExecState,
        TurnExecution,
        resolve_logical_turn_key,
    )

    execn = TurnExecution(
        turn_key="E1", channel_id="chan", kind="invoke_agent", idempotency_key="i1"
    )
    registry = SimpleNamespace(active_turn_key=lambda channel_id: "E1")
    runtime = SimpleNamespace(
        execution_context=SimpleNamespace(current_turn_key="PREVIOUS-TURN-KEY")
    )

    # Queued: the snapshot has not been taken; the stale key must not stick.
    assert resolve_logical_turn_key(execn, runtime, registry) is None
    assert execn.logical_turn_key is None

    # Running with the snapshot taken: a differing key can only be the key
    # _begin_turn minted for THIS execution (single-flight per channel).
    execn.exec_state = ExecState.RUNNING
    execn.pre_turn_key = "PREVIOUS-TURN-KEY"
    runtime.execution_context.current_turn_key = "THIS-TURN-KEY"
    assert resolve_logical_turn_key(execn, runtime, registry) == "THIS-TURN-KEY"
    assert execn.logical_turn_key == "THIS-TURN-KEY"
