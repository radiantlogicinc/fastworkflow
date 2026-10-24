"""Archive one conversation the way an experiment is archived."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path

import pytest

from fastworkflow import state_paths
from fastworkflow.observability import store as obs
from fastworkflow.run_chatbot import server as run_chatbot_server


def _turn_row(turn_key: str, message: str, *, experiment_id=None) -> dict:
    return {
        "turn_key": turn_key,
        "channel_id": "chatbot",
        "conversation_id": 1,
        "ordinal": 1,
        "user_message": message,
        "refined_user_message": None,
        "entry_workflow_name": "talk_to_ido",
        "entry_context": "*",
        "status": "completed",
        "success": 1,
        "failure_reason": None,
        "answer": "recorded answer",
        "conversation_summary": None,
        "conversation_traces": None,
        "started_at": "2026-10-16T23:44:24+00:00",
        "completed_at": "2026-10-16T23:45:00+00:00",
        "suspended_ms": 0,
        "continuation_of": None,
        "record_version": 1,
        "experiment_id": experiment_id,
        "task_id": None,
        "attempt": None,
        "claim_epoch": None,
        "server_incarnation": None,
        "record_json": json.dumps({"turn_key": turn_key, "status": "completed"}),
    }


def _write_turn(store: obs.ObservabilityStore, turn_key: str, message: str, **kwargs) -> None:
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        assert store.upsert_turn_row(
            conn, _turn_row(turn_key, message, **kwargs), [], obs.Redactor()
        )
        conn.commit()


def test_archiving_an_empty_conversation_removes_the_row(tmp_path, monkeypatch):
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    workflow = tmp_path / "talk_to_ido"
    workflow.mkdir()
    store = obs.ObservabilityStore(state_paths.observability_db(str(workflow)))
    store.mint_conversation_id("chatbot")
    store.record_conversation_label("chatbot", 1, "Risk Score Comparison", "a leftover")
    destination = Path(store.db_path).parent / "archives" / "conversation-chatbot-1.sqlite3"
    destination.parent.mkdir()

    archived = store.archive_conversation("chatbot", 1, str(destination))

    assert archived["archived"] is True
    assert archived["turn_count"] == 0
    assert archived["topic"] == "Risk Score Comparison"
    assert destination.is_file()
    assert oct(destination.stat().st_mode & 0o777) == "0o444"
    assert store.list_conversations() == []
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM conversation_archives"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM conversation_counters WHERE channel_id='chatbot'"
        ).fetchone()[0] == 0
    with sqlite3.connect(f"{destination.resolve().as_uri()}?mode=ro", uri=True) as conn:
        topic = conn.execute("SELECT topic FROM conversations").fetchone()[0]
        turns = conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
    assert topic == "Risk Score Comparison"
    assert turns == 0

    removed = store.delete_archived_conversation("chatbot", 1)
    assert removed["file_removed"] is True
    assert not destination.exists()
    with pytest.raises(obs.ConversationNotArchived):
        store.delete_archived_conversation("chatbot", 1)


def test_archiving_a_channel_drops_its_counter_without_reusing_its_id(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    workflow = tmp_path / "talk_to_ido"
    workflow.mkdir()
    store = obs.ObservabilityStore(state_paths.observability_db(str(workflow)))
    store.mint_conversation_id("chatbot")
    destination = Path(store.db_path).parent / "archives" / "conversation-chatbot-1.sqlite3"
    destination.parent.mkdir()
    store.archive_conversation("chatbot", 1, str(destination))

    assert store.mint_conversation_id("chatbot") == 2


def test_archiving_keeps_the_turns_and_drops_them_from_the_live_db(tmp_path, monkeypatch):
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    workflow = tmp_path / "talk_to_ido"
    workflow.mkdir()
    store = obs.ObservabilityStore(state_paths.observability_db(str(workflow)))
    store.mint_conversation_id("chatbot")
    _write_turn(store, "turn-1", "what is the risk score")
    destination = Path(store.db_path).parent / "archives" / "conversation-chatbot-1.sqlite3"
    destination.parent.mkdir()

    archived = store.archive_conversation("chatbot", 1, str(destination))

    assert archived["turn_count"] == 1
    assert store.list_turns() == []
    with sqlite3.connect(f"{destination.resolve().as_uri()}?mode=ro", uri=True) as conn:
        message = conn.execute("SELECT user_message FROM turns").fetchone()[0]
    assert message == "what is the risk score"


def test_a_conversation_that_belongs_to_an_experiment_stays_there(tmp_path, monkeypatch):
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    workflow = tmp_path / "talk_to_ido"
    workflow.mkdir()
    store = obs.ObservabilityStore(state_paths.observability_db(str(workflow)))
    store.mint_conversation_id("chatbot", experiment_id="exp-1")
    destination = Path(store.db_path).parent / "archives" / "conversation-chatbot-1.sqlite3"
    destination.parent.mkdir()

    with pytest.raises(obs.ConversationArchiveRefused, match="archive that experiment"):
        store.archive_conversation("chatbot", 1, str(destination))
    assert not destination.exists()
    assert len(store.list_conversations()) == 1


def test_chatbot_archives_and_deletes_one_conversation(tmp_path, monkeypatch):
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    workflow = tmp_path / "talk_to_ido"
    workflow.mkdir()
    db_path = state_paths.observability_db(str(workflow))
    store = obs.ObservabilityStore(db_path)
    store.mint_conversation_id("chatbot")
    store.record_conversation_label("chatbot", 1, "Risk Score Comparison", None)
    server = run_chatbot_server.ChatbotServer(db_path, workflow_path=str(workflow), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        created = _json(server, "POST", "/api/archived-conversations", {
            "channel_id": "chatbot", "conversation_id": 1,
        })
        assert created["turn_count"] == 0
        listing = _json(server, "GET", "/api/archived-conversations")
        assert listing["archived"][0]["topic"] == "Risk Score Comparison"
        detail = _json(server, "GET", "/api/archived-conversations/chatbot/1")
        assert detail["turns"] == []
        navigation = _json(server, "GET", "/api/navigation")
        assert "Risk Score Comparison" not in json.dumps(navigation)
        deleted = _json(server, "DELETE", "/api/archived-conversations/chatbot/1")
        assert deleted["file_removed"] is True
        listing = _json(server, "GET", "/api/archived-conversations")
        assert listing["archived"] == []
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _json(server, method: str, path: str, body=None) -> dict:
    import urllib.request

    data = None if body is None and method == "GET" else json.dumps(body or {}).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.port}{path}", data=data, method=method
    )
    request.add_header("Authorization", f"Bearer {server.token}")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode())


def test_archive_conversation_dom(tmp_path, monkeypatch):
    dependency = os.environ.get("TEST_JSDOM_ROOT")
    if not dependency:
        pytest.skip("Set TEST_JSDOM_ROOT to run DOM integration with jsdom")
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    workflow = tmp_path / "talk_to_ido"
    workflow.mkdir()
    db_path = state_paths.observability_db(str(workflow))
    store = obs.ObservabilityStore(db_path)
    store.mint_conversation_id("chatbot")
    store.mint_conversation_id("chatbot")
    store.record_conversation_label("chatbot", 1, "Risk Score Comparison", None)
    store.record_conversation_label("chatbot", 2, "Second chat", None)
    server = run_chatbot_server.ChatbotServer(db_path, workflow_path=str(workflow), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        import subprocess

        script = Path(__file__).with_name("chatbot_conversation_archive_dom.cjs")
        result = subprocess.run(
            ["node", str(script), dependency,
             f"http://127.0.0.1:{server.port}/?token={server.token}"],
            capture_output=True, text=True, timeout=40,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    finally:
        server.shutdown()
        thread.join(timeout=5)
