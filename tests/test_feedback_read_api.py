"""The agent-memory feedback table and its read API are gone (fix-9eg.16).

This module used to prove that `upsert_feedback` / `get_feedback` /
`list_feedback` and the `GET /api/feedback` routes worked on a turn with no
conversation summary. The owner retired that whole surface: it was a mutable
one-row-per-turn store whose only reader was `get_memory_window`, which
replayed it into the agent's `dspy.History`. The tests are kept and inverted
rather than deleted, because "this is gone" is the behavior that now has to
hold, and a future reader deserves to find out here that it went on purpose.

What replaced it is a different thing and is covered elsewhere
(`test_human_feedback.py`, `test_task_feedback.py`): append-only review notes
with provenance, an owner-confirmed category, evidence anchors, and no path
back into a prompt.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import urllib.error
import urllib.parse
import urllib.request

import pytest

from fastworkflow.observability import store as obs
from fastworkflow.run_chatbot import server as run_chatbot_server


def _turn(turn_key: str, channel_id: str) -> dict:
    return {
        "turn_key": turn_key,
        "channel_id": channel_id,
        "conversation_id": None,
        "ordinal": None,
        "user_message": "waiting",
        "refined_user_message": None,
        "entry_workflow_name": "workflow",
        "entry_context": "context",
        "status": "awaiting_user",
        "success": 0,
        "failure_reason": None,
        "answer": "",
        "conversation_summary": None,
        "conversation_traces": None,
        "started_at": "2026-09-04T00:00:00+00:00",
        "completed_at": None,
        "suspended_ms": 0,
        "continuation_of": None,
        "record_version": 1,
        "record_json": json.dumps({"turn_output": {"turn_key": turn_key}}),
    }


def _get(server, path: str) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.port}{path}",
        headers={"Authorization": f"Bearer {server.token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _seeded_store(tmp_path):
    store = obs.ObservabilityStore(str(tmp_path / "observability.sqlite3"))
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        for turn_key, channel_id in (
            ("turn/no-summary", "channel-a"),
            ("turn-b", "b"),
        ):
            assert store.upsert_turn_row(
                conn, _turn(turn_key, channel_id), [], obs.Redactor()
            )
        conn.commit()
    return store


@pytest.mark.parametrize(
    "name", ["upsert_feedback", "get_feedback", "list_feedback"]
)
def test_the_agent_memory_feedback_apis_are_gone(tmp_path, name):
    """No dual stack: the methods are absent, not deprecated shims."""
    store = _seeded_store(tmp_path)
    assert not hasattr(store, name)
    assert not hasattr(obs.ReadOnlyObservabilityStore, name)


def test_a_fresh_store_has_no_feedback_table(tmp_path):
    """The table is not created, so nothing can quietly start writing it."""
    store = _seeded_store(tmp_path)
    with store._connect() as conn:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert "feedback" not in names
    assert "human_feedback" in names
    with store._connect() as conn, pytest.raises(sqlite3.OperationalError):
        conn.execute("SELECT * FROM feedback")


def test_the_memory_window_no_longer_carries_a_feedback_key(tmp_path):
    """The injection into `dspy.History` is what the removal was about."""
    store = _seeded_store(tmp_path)
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = _turn("turn-c", "channel-a")
        row.update(
            conversation_id=1,
            ordinal=1,
            status="completed",
            success=1,
            conversation_summary="did the thing",
        )
        assert store.upsert_turn_row(conn, row, [], obs.Redactor())
        conn.commit()
    window = store.get_memory_window("channel-a", 1, max_turns=10)
    assert window == [
        {"conversation summary": "did the thing", "conversation_traces": None}
    ]
    assert all("feedback" not in entry for entry in window)


def test_the_feedback_read_routes_are_gone(tmp_path):
    """A client still calling them gets a 404, not a different payload.

    Deliberately not re-pointed at the review notes: those live on
    `/api/feedback-notes` and `/api/task-feedback`. Answering the old path
    with new-shaped data would hide the removal from exactly the caller who
    needs to know about it.
    """
    store = _seeded_store(tmp_path)
    server = run_chatbot_server.ChatbotServer(store.db_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        encoded = urllib.parse.quote("turn/no-summary", safe="")
        assert _get(server, "/api/feedback")[0] == 404
        assert _get(server, f"/api/feedback/{encoded}")[0] == 404
        # The replacement reads answer on their own paths.
        status, payload = _get(
            server, f"/api/feedback-notes?turn_key={encoded}"
        )
        assert status == 200 and payload["feedback"] == []
        status, payload = _get(server, "/api/feedback-taxonomy")
        assert status == 200
        assert [c["value"] for c in payload["categories"]] == [
            "observations_analysis",
            "conclusions",
            "recommendations",
        ]
    finally:
        server.shutdown()
        thread.join(timeout=5)
