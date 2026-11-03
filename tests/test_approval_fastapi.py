"""Approval binding on the FastAPI host (ADR-EXP-025 sections 8, 10, 11): the token round trip, the
user-typed gate on /perform_action, /cancel_pending, and eviction clearing pending approvals.

The host runs the real app and session cache on an annotated copy of the todo workflow (the same
copy the approval unit tests use). The routes exercised here are deterministic (perform_action, and
invoke_agent answering a pending approval), so no model is called.
"""
from __future__ import annotations

import asyncio
import json
import importlib
import os
import shutil
import sys
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fastworkflow import approval_binding as ab
from pathlib import Path as _Path

from tests.test_approval_binding import _annotate

# The FastAPI routes exercised here (perform_action, and invoke_agent answering a pending approval) do
# not route by intent, so this file keeps the todo copy: its approval path needs no trained model.
TODO = _Path(__file__).resolve().parent / "todo_list_workflow"
_COPY_IGNORE = shutil.ignore_patterns("___command_info", "___workflow_contexts", "___convo_info", "__pycache__")
from tests.todo_list_workflow.application.todo_list import TodoList
from tests.todo_list_workflow.application.todo_manager import TodoListManager

MANDATORY = "TodoList/set_properties"


@pytest.fixture(autouse=True)
def no_live_model(monkeypatch):
    """No test here calls a model: a call through LiteLLM fails the test instead of reaching a provider."""
    import litellm

    def _refuse(*args, **kwargs):
        raise AssertionError("a live model call was attempted from an approval test")

    monkeypatch.setattr(litellm, "completion", _refuse)
    monkeypatch.setattr(litellm, "acompletion", _refuse)


@pytest.fixture
def env_files():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_file = os.path.join(project_root, "env", ".env")
    passwords_file = os.path.join(project_root, "passwords", ".env")
    if not os.path.isfile(env_file) or not os.path.isfile(passwords_file):
        pytest.skip("env files missing for FastAPI tests")
    return env_file, passwords_file


@pytest.fixture
def app_module(tmp_path, env_files, monkeypatch):
    # The host's identity is verified only with signed tokens (ADR section 8). Pin the default here:
    # other test modules may have left the flag off, and these tests are about the verified case.
    from fastworkflow.run_fastapi_mcp import jwt_manager
    monkeypatch.setattr(jwt_manager, "EXPECT_ENCRYPTED_JWT", True)
    folder = tmp_path / "todo_annotated"
    shutil.copytree(TODO, folder, ignore=_COPY_IGNORE, symlinks=True)
    _annotate(folder, "TodoList/set_properties.py", '"mandatory"')
    env_file, passwords_file = env_files
    sys.argv = ["pytest", "--workflow_path", str(folder), "--env_file_path", env_file,
                "--passwords_file_path", passwords_file]
    import fastworkflow.run_fastapi_mcp.__main__ as main

    importlib.reload(main)
    from tests.fastapi_hermetic import init_fastapi_hermetic_env, restore_fastapi_env

    previous_env = init_fastapi_hermetic_env(env_file, passwords_file, tmp_path / "workflow_contexts")
    try:
        yield main
    finally:
        restore_fastapi_env(previous_env)


def _setup_todo_context(app_module, channel_id: str):
    """Put the channel's workflow into a TodoList context, with two lists to tell apart."""
    async def go():
        runtime = await app_module.session_manager.get_session(channel_id)
        wf = runtime.execution_context.app_workflow
        root = TodoListManager(str(Path(wf.folderpath).parent / f"todo_{uuid.uuid4().hex}.json"))
        wf.root_command_context = root
        chores = TodoList(id=7, description="chores")
        chores.parent = root
        root.lists[7] = chores
        wf.current_command_context = chores
        # A list is identified by its id in the instance label (as a record is), so the context locates.
        from fastworkflow.context_identity import context_class_for
        context_class_for(wf, "TodoList").instance_label_attr = "id"
        return wf, chores
    return asyncio.run(go())


def _auth(init: dict) -> dict:
    return {"Authorization": f"Bearer {init['access_token']}"}


def test_user_typed_gate_token_round_trip_over_http(app_module):
    channel_id = f"approval_{uuid.uuid4().hex[:8]}"
    client = TestClient(app_module.app)
    init = client.post("/initialize", json={"channel_id": channel_id, "user_id": "alice"}).json()
    headers = _auth(init)
    wf, chores = _setup_todo_context(app_module, channel_id)

    action = {"command_name": MANDATORY, "command": "set_properties assign_to=Ann",
              "parameters": {"assign_to": "Ann", "is_complete": None}}
    resp = client.post("/perform_action", headers=headers, json={"action": action})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    token = body["approval"]["token"]
    assert token and "You asked to run TodoList/set_properties" in body["answer"]
    assert token not in body["answer"]
    assert chores.assign_to == ""

    # The token is delivered once: a poll of the same turn does not show it again.
    assert ab.store_for(wf).active() is not None      # answering a record, never the agent
    done = client.post("/invoke_agent", headers=headers,
                       json={"user_query": "yes", "approval_token": token, "timeout_seconds": 10})
    assert done.status_code == 200, done.text
    assert "approval" not in done.json()
    assert chores.assign_to == "Ann"


def test_a_wrong_token_over_http_does_not_run(app_module):
    channel_id = f"approval_{uuid.uuid4().hex[:8]}"
    client = TestClient(app_module.app)
    headers = _auth(client.post("/initialize", json={"channel_id": channel_id, "user_id": "alice"}).json())
    wf, chores = _setup_todo_context(app_module, channel_id)
    action = {"command_name": MANDATORY, "command": "set_properties assign_to=Ann",
              "parameters": {"assign_to": "Ann", "is_complete": None}}
    client.post("/perform_action", headers=headers, json={"action": action})
    assert ab.store_for(wf).active() is not None      # answering a record, never the agent
    resp = client.post("/invoke_agent", headers=headers,
                       json={"user_query": "yes", "approval_token": "forged", "timeout_seconds": 10})
    assert resp.status_code == 200
    assert chores.assign_to == ""
    assert ab.MSG_NO_LONGER_VALID in resp.json()["answer"]


def test_cancel_pending_endpoint_cancels_the_approval(app_module):
    channel_id = f"approval_{uuid.uuid4().hex[:8]}"
    client = TestClient(app_module.app)
    headers = _auth(client.post("/initialize", json={"channel_id": channel_id, "user_id": "alice"}).json())
    wf, chores = _setup_todo_context(app_module, channel_id)
    action = {"command_name": MANDATORY, "command": "set_properties assign_to=Ann",
              "parameters": {"assign_to": "Ann", "is_complete": None}}
    client.post("/perform_action", headers=headers, json={"action": action})
    assert ab.store_for(wf).active() is not None

    resp = client.post("/cancel_pending", headers=headers)
    assert resp.status_code == 200
    assert ab.store_for(wf).active() is None
    assert chores.assign_to == ""


def test_a_pending_approval_survives_a_restart_and_resolves_once_with_its_token(app_module):
    """ADR R6 (owner answer 4): the record lives in the session's durable pending state with its token
    hash only. Dropping the live session without a session end (a crash, or an unload) and rehydrating
    it restores the record; the token answers it, once.

    Eviction itself is not exercised: this todo context is not checkpointable (its locator hooks are
    absent), so the eviction path pins it in memory. That is a property of the test app, reported.
    """
    channel_id = f"approval_{uuid.uuid4().hex[:8]}"
    client = TestClient(app_module.app)
    headers = _auth(client.post("/initialize", json={"channel_id": channel_id, "user_id": "alice"}).json())
    wf, chores = _setup_todo_context(app_module, channel_id)
    action = {"command_name": MANDATORY, "command": "set_properties assign_to=Ann",
              "parameters": {"assign_to": "Ann", "is_complete": None}}
    token = client.post("/perform_action", headers=headers, json={"action": action}).json()["approval"]["token"]
    stored = app_module.session_manager.session_state_store.load(channel_id)
    assert stored is not None and token not in json.dumps(stored, default=str)
    assert [r["state"] for r in stored["approval"]["records"]] == ["awaiting"]

    # Restart: the live session is gone from memory; the durable pending state is all that is left.
    async def _crash():
        app_module.session_manager._sessions.pop(channel_id, None)     # no close, no session end
    asyncio.run(_crash())
    resp = client.post("/invoke_agent", headers=headers,
                       json={"user_query": "yes", "approval_token": token, "timeout_seconds": 10})
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "completed"
    runtime = asyncio.run(app_module.session_manager.get_session(channel_id))
    restored = ab.store_for(runtime.execution_context.app_workflow)
    assert [(r.state, r.outcome) for r in restored.records] == [("executed", "success")]
    # Once only: with nothing awaiting, the same token resolves nothing (checked without a model call).
    assert ab.resolve_reply(runtime.execution_context, runtime.execution_context.app_workflow, "yes") is None


def test_session_end_clears_pending_approvals_and_their_stored_state(app_module):
    channel_id = f"approval_{uuid.uuid4().hex[:8]}"
    client = TestClient(app_module.app)
    headers = _auth(client.post("/initialize", json={"channel_id": channel_id, "user_id": "alice"}).json())
    wf, chores = _setup_todo_context(app_module, channel_id)
    action = {"command_name": MANDATORY, "command": "set_properties assign_to=Ann",
              "parameters": {"assign_to": "Ann", "is_complete": None}}
    client.post("/perform_action", headers=headers, json={"action": action})
    assert ab.store_for(wf).active() is not None

    asyncio.run(app_module.session_manager.remove_session(channel_id))     # session end
    assert ab.store_for(wf).active() is None
    assert not app_module.session_manager.session_state_store.exists(channel_id)
    assert chores.assign_to == ""
