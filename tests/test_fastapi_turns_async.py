"""Integration tests for the turns-based async execution engine (Step 1).

Scope (per fix-85g.8): drive everything through ``POST /initialize`` with a
``startup_action`` so NO workflow training is required — a startup_action
dispatches a command directly by name, bypassing NLU/intent detection. The
``add_two_numbers`` application function in ``tests/hello_world_workflow`` has
env-var test hooks (``FW_TEST_ADD_SLEEP_SECONDS`` to simulate a long-running
command, ``FW_TEST_ADD_CALL_LOG`` to count invocations).

These assert the wait-or-defer behavior, the three-state "already exists"
branch (§3.3), single-flight (a retry rejoins the SAME execution and never
starts a second one), and the pointer-based 409 busy guard (§3.4).
"""

from __future__ import annotations

import importlib
import os
import sys
import time
import uuid
import warnings

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
def app_module(workflow_path, env_files):
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
    import fastworkflow
    from dotenv import dotenv_values

    fastworkflow.init({**dotenv_values(env_file), **dotenv_values(passwords_file)})
    if fastworkflow.RoutingRegistry:
        fastworkflow.RoutingRegistry.clear_registry()
    return main


def _startup_action() -> dict:
    return {
        "command_name": "add_two_numbers",
        "parameters": {"first_num": 2.0, "second_num": 3.0},
    }


def _count_calls(call_log: str) -> int:
    if not os.path.isfile(call_log):
        return 0
    with open(call_log) as fh:
        return sum(bool(line.strip())
               for line in fh)


def _channel(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def test_initialize_startup_fast_path_runs_inline_once(app_module, tmp_path, monkeypatch):
    """A quick startup command finishes within the wait window: 200 + startup_output, one call."""
    call_log = str(tmp_path / "calls.log")
    monkeypatch.setenv("FW_TEST_ADD_CALL_LOG", call_log)
    # No sleep -> command returns immediately, well within the default wait window.

    channel_id = _channel("fast")
    body = {"channel_id": channel_id, "user_id": "u_fast", "startup_action": _startup_action()}

    # Context manager keeps a single event loop alive across requests so a
    # deferred background turn survives between calls (wait-or-defer).
    with TestClient(app_module.app) as client:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            resp = client.post("/initialize", json=body)

    assert resp.status_code == 200
    data = resp.json()
    assert data["startup_exec_state"] == "done"
    assert data["startup_turn_key"]
    assert data["startup_error"] is None
    assert data["startup_output"] is not None
    # startup_output preserves the CommandOutput shape (command_responses).
    assert "command_responses" in data["startup_output"]
    assert _count_calls(call_log) == 1
    # The server no longer goes through the deprecated process_message().
    assert all(
        "process_message" not in str(w.message) for w in caught
    ), "process_message DeprecationWarning should not be emitted"


def test_initialize_startup_defers_and_is_single_flight(app_module, tmp_path, monkeypatch):
    # sourcery skip: extract-duplicate-method
    """A slow startup defers (202); retries rejoin the SAME execution and it runs exactly once."""
    call_log = str(tmp_path / "calls.log")
    monkeypatch.setenv("FW_TEST_ADD_CALL_LOG", call_log)
    monkeypatch.setenv("FW_TEST_ADD_SLEEP_SECONDS", "3")

    channel_id = _channel("defer")
    # Short wait window (< 3s sleep) so the request defers instead of blocking.
    body = {
        "channel_id": channel_id,
        "user_id": "u_defer",
        "startup_action": _startup_action(),
        "timeout_seconds": 1,
    }

    with TestClient(app_module.app) as client:
        # First call defers: still running, no output yet (NOT a silently-empty result).
        resp1 = client.post("/initialize", json=body)
        assert resp1.status_code == 202
        d1 = resp1.json()
        assert d1["startup_exec_state"] == "running"
        assert d1["startup_output"] is None
        turn_key = d1["startup_turn_key"]
        assert turn_key

        # Retry while still running (§3.3): same channel hits the "already exists"
        # branch and rejoins the SAME execution -> same turn_key, still running.
        resp2 = client.post("/initialize", json=body)
        assert resp2.status_code == 202
        d2 = resp2.json()
        assert d2["startup_exec_state"] == "running"
        assert d2["startup_turn_key"] == turn_key

        # Poll (by retrying /initialize) until the deferred execution completes.
        final = None
        deadline = time.time() + 15
        while time.time() < deadline:
            resp = client.post("/initialize", json=body)
            data = resp.json()
            if data["startup_exec_state"] == "done":
                final = (resp, data)
                break
            assert data["startup_turn_key"] == turn_key
            time.sleep(0.3)

        assert final is not None, "deferred startup never completed"
    resp_done, data_done = final
    assert resp_done.status_code == 200
    assert data_done["startup_turn_key"] == turn_key
    assert data_done["startup_output"] is not None
    assert "command_responses" in data_done["startup_output"]

    # The crux: despite 1 defer + 1 in-flight retry + several polling retries,
    # the long-running command executed EXACTLY ONCE (no duplicate LLM/work).
    assert _count_calls(call_log) == 1


def test_initialize_respects_request_timeout_seconds(app_module, tmp_path, monkeypatch):
    """A short per-request timeout_seconds governs the wait window (vs the 60s default)."""
    call_log = str(tmp_path / "calls.log")
    monkeypatch.setenv("FW_TEST_ADD_CALL_LOG", call_log)
    monkeypatch.setenv("FW_TEST_ADD_SLEEP_SECONDS", "3")

    channel_id = _channel("reqtimeout")
    # The default timeout_seconds is 60; if it were used the call would block
    # ~3s and return 200. A short per-request timeout_seconds must instead defer.
    body = {
        "channel_id": channel_id,
        "user_id": "u_reqto",
        "startup_action": _startup_action(),
        "timeout_seconds": 1,
    }

    with TestClient(app_module.app) as client:
        resp = client.post("/initialize", json=body)
        # Honored the request's 1s window (< 3s sleep) -> deferred, despite the
        # 60s default.
        assert resp.status_code == 202
        data = resp.json()
        assert data["startup_exec_state"] == "running"
        turn_key = data["startup_turn_key"]

        # Recoverable to completion via the three-state already-exists branch.
        deadline = time.time() + 15
        done = False
        while time.time() < deadline:
            poll = client.post("/initialize", json=body)
            if poll.json()["startup_exec_state"] == "done":
                assert poll.status_code == 200
                assert poll.json()["startup_turn_key"] == turn_key
                done = True
                break
            time.sleep(0.3)
        assert done, "deferred startup never completed"
    assert _count_calls(call_log) == 1


def test_busy_channel_rejects_concurrent_turn(app_module, tmp_path, monkeypatch):
    # sourcery skip: extract-method
    """While a startup is in flight, another turn on the same channel gets 409 (§3.4)."""
    call_log = str(tmp_path / "calls.log")
    monkeypatch.setenv("FW_TEST_ADD_CALL_LOG", call_log)
    monkeypatch.setenv("FW_TEST_ADD_SLEEP_SECONDS", "3")

    channel_id = _channel("busy")
    body = {
        "channel_id": channel_id,
        "user_id": "u_busy",
        "startup_action": _startup_action(),
        "timeout_seconds": 1,
    }

    with TestClient(app_module.app) as client:
        resp1 = client.post("/initialize", json=body)
        assert resp1.status_code == 202
        access_token = resp1.json()["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}

        # A concurrent direct action on the busy channel must be rejected with 409
        # based on the registry active-execution pointer (NOT lock.locked()), and
        # must NOT execute the command.
        resp_action = client.post(
            "/perform_action",
            headers=headers,
            json={"action": _startup_action(), "timeout_seconds": 1},
        )
        assert resp_action.status_code == 409

        # Only the startup turn ran; the rejected action did not add a call.
        deadline = time.time() + 15
        while time.time() < deadline:
            resp = client.post("/initialize", json=body)
            if resp.json()["startup_exec_state"] == "done":
                break
            time.sleep(0.3)
    assert _count_calls(call_log) == 1
