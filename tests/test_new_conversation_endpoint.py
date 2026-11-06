"""POST /new_conversation over HTTP, with no live model call (fix-ayez).

A paused ask_user question in conversation A is parked when the New conversation button is pressed. The first message
in the new conversation starts a new turn and is not taken as A's answer, and A's question comes back when A is bound
again. Deleting A drops the state parked for it, in memory and in the saved channel blob.

The model boundary is scripted: one trajectory step per turn, the sub-task split, the query refinement and the topic
label. Asked for a step nobody scripted, the test fails rather than calling out. The LiteLLM refusal fixture from
test_approval_fastapi.py is imported too, so an unexpected completion call fails the test as well.
"""
from __future__ import annotations

import asyncio
import importlib
import os
import sys
import uuid

import pytest
from fastapi.testclient import TestClient

from fastworkflow.run_fastapi_mcp.utils import drop_parked_pending_for_conversation
from tests.test_approval_binding import annotated_library  # noqa: F401 - the library copy the turns run on
from tests.test_approval_fastapi import no_live_model  # noqa: F401 - LiteLLM refuses any completion call

QUESTION = "Which Aaron do you mean?"


@pytest.fixture
def env_files():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_file = os.path.join(project_root, "env", ".env")
    passwords_file = os.path.join(project_root, "passwords", ".env")
    if not os.path.isfile(env_file) or not os.path.isfile(passwords_file):
        pytest.skip("env files missing for FastAPI tests")
    return env_file, passwords_file


@pytest.fixture
def app_module(annotated_library, env_files, tmp_path):  # noqa: F811 - the library copy, served by the real app
    """The FastAPI app on the library copy, so a turn takes the same path the isolation tests take."""
    env_file, passwords_file = env_files
    sys.argv = ["pytest", "--workflow_path", str(annotated_library), "--env_file_path", env_file,
                "--passwords_file_path", passwords_file]
    import fastworkflow.run_fastapi_mcp.__main__ as main

    importlib.reload(main)
    from tests.fastapi_hermetic import init_fastapi_hermetic_env, restore_fastapi_env

    previous_env = init_fastapi_hermetic_env(env_file, passwords_file, tmp_path / "workflow_contexts")
    try:
        yield main
    finally:
        restore_fastapi_env(previous_env)


@pytest.fixture
def scripted_model(monkeypatch, app_module):  # noqa: F811
    """The model boundary: each step is scripted, and an unscripted step fails the test (no finish by default)."""
    from tests.fastapi_turn_seam import install_turn_seam

    return install_turn_seam(monkeypatch, finish_when_empty=False, app_module=app_module)


def _auth(init: dict) -> dict:
    return {"Authorization": f"Bearer {init['access_token']}"}


def _runtime(app_module, channel_id: str):
    return asyncio.run(app_module.session_manager.get_session(channel_id))


def _park_question_in_first_conversation(app_module, scripted_model):
    """Conversation A asks a question and the turn suspends. Returns (client, headers, channel, A's turn, A's id)."""
    channel = f"newconv_{uuid.uuid4().hex[:8]}"
    client = TestClient(app_module.app)
    init = client.post("/initialize", json={"channel_id": channel, "user_id": "alice"})
    assert init.status_code == 200, init.text
    headers = _auth(init.json())
    conv_a = _runtime(app_module, channel).active_conversation_id

    scripted_model.steps[:] = [("ask_user", {"clarification_request": QUESTION})]
    asked = client.post("/invoke_agent", headers=headers,
                        json={"user_query": "who is Aaron?", "timeout_seconds": 60})
    assert asked.status_code == 200, asked.text
    assert asked.json()["status"] == "awaiting_user"
    return client, headers, channel, asked.json()["turn_key"], conv_a


def test_new_conversation_parks_a_paused_question_and_the_next_message_starts_a_turn(app_module, scripted_model):
    client, headers, channel, asked_turn, conv_a = _park_question_in_first_conversation(app_module, scripted_model)
    runtime = _runtime(app_module, channel)
    assert runtime.execution_context.awaiting_user

    rotated = client.post("/new_conversation", headers=headers, json={})
    assert rotated.status_code == 200, rotated.text
    conv_b = runtime.active_conversation_id
    assert conv_b != conv_a
    assert not runtime.execution_context.awaiting_user          # the question is not live in the new conversation
    assert runtime.execution_context.has_parked_pending()       # ... it is parked under conversation A

    scripted_model.steps[:] = [("finish", {})]
    reply = client.post("/invoke_agent", headers=headers,
                        json={"user_query": "who am I?", "timeout_seconds": 60})
    assert reply.status_code == 200, reply.text
    data = reply.json()
    assert data["turn_key"] != asked_turn                      # a new turn, not the resume of A's
    assert data["status"] == "completed"
    assert scripted_model.steps == []                          # the model was asked for its steps afresh
    assert runtime.execution_context.has_parked_pending()      # A's question still waits, parked, not dropped


def test_deleting_the_first_conversation_drops_its_parked_state_in_memory_and_on_disk(app_module, scripted_model):
    client, headers, channel, _asked_turn, conv_a = _park_question_in_first_conversation(app_module, scripted_model)
    runtime = _runtime(app_module, channel)
    assert client.post("/new_conversation", headers=headers, json={}).status_code == 200
    store = app_module.session_manager.session_state_store
    assert str(conv_a) in (store.load(channel) or {}).get("parked_pending", {})

    dropped = drop_parked_pending_for_conversation(
        app_module.session_manager, channel, conv_a, runtime=runtime)

    assert dropped == 1
    assert not runtime.execution_context.has_parked_pending()
    saved = store.load(channel) or {}
    assert not saved.get("parked_pending")
    assert scripted_model.steps == []


def test_a_restart_prunes_the_parked_state_of_a_conversation_deleted_while_the_process_was_down(app_module, scripted_model):
    """The archive and delete routes live in run_chatbot, so the deletion is only visible to this process as a missing
    conversations row. A restart reads the saved blob, prunes it against the store, and saves the pruned blob back."""
    client, headers, channel, _asked_turn, conv_a = _park_question_in_first_conversation(app_module, scripted_model)
    assert client.post("/new_conversation", headers=headers, json={}).status_code == 200
    store = app_module.session_manager.session_state_store
    assert str(conv_a) in (store.load(channel) or {}).get("parked_pending", {})
    live = _runtime(app_module, channel)
    observability = live.observability_store
    assert observability is not None

    with observability._connect() as conn:
        conn.execute("DELETE FROM conversations WHERE channel_id=? AND conversation_id=?", (channel, conv_a))
        conn.commit()

    async def _crash():
        app_module.session_manager._sessions.pop(channel, None)     # no close, no session end: a restart
    asyncio.run(_crash())

    listed = client.get("/conversations", headers=headers)
    assert listed.status_code == 200, listed.text
    restored = _runtime(app_module, channel)
    assert restored is not None and restored is not live
    assert not restored.execution_context.has_parked_pending()
    assert not (store.load(channel) or {}).get("parked_pending")
    assert scripted_model.steps == []
