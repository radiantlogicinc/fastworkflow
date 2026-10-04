"""Reusing a recorded message as a new chat turn (fix-9eg.7.4).

The feature is deliberately small: a recorded user message can be copied into
the live composer, edited, and sent as an ordinary turn. What has to be true
is mostly about what does NOT happen — reading evidence runs nothing, the
recorded turn is not modified, nothing is submitted without an explicit
action, and the turn that results is a new one under current settings rather
than a claim to have reproduced the old one.

Costs nothing to run: the DOM half drives the real page against a real
chatbot server over a store seeded here, and the identity half drives the
real streaming route with the session's runtime answering deterministically
(the pattern from test_chat_streaming_and_artifacts.py).
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

import pytest

from fastworkflow import state_paths
from fastworkflow.observability import store as obs
from fastworkflow.run_chatbot import server as run_chatbot_server

# Fixtures and helpers reused from the streaming/artifact module, which owns
# the hermetic FastAPI app and the deterministic-turn helper.
from tests import test_chat_streaming_and_artifacts as chat_fixtures
from tests.test_chat_streaming_and_artifacts import (
    _channel,
    _stream_a_real_deterministic_turn,
)

# Bound by assignment rather than imported by name: pytest registers a fixture
# under the name it is bound to either way, but a test taking `app_module` as a
# parameter would be flagged as redefining an import (F811).
hello_world_path = chat_fixtures.hello_world_path
env_files = chat_fixtures.env_files
app_module = chat_fixtures.app_module

MULTILINE_TURN = "20260825T130000-reuse1"
WITHHELD_TURN = "20260825T130500-reuse2"
MULTILINE_MESSAGE = (
    "first line of the recorded message\n"
    "second line, indented:\n"
    "    - a bullet the person typed\n"
)
WITHHELD_MESSAGE = json.dumps(
    {
        "__fw_capture__": True,
        "reason": "user_message withheld by the strict profile",
        "classification": "pii",
        "original_bytes": 61,
        "digest": "sha256:0f0f",
    }
)


def _turn_row(turn_key: str, user_message: str) -> dict:
    return {
        "turn_key": turn_key,
        "channel_id": "chan1",
        "conversation_id": 1,
        "ordinal": 1 if turn_key == MULTILINE_TURN else 2,
        "user_message": user_message,
        "refined_user_message": None,
        "entry_workflow_name": "hello_world",
        "entry_context": "*",
        "status": "completed",
        "success": 1,
        "failure_reason": None,
        "answer": "answer for " + turn_key,
        "conversation_summary": None,
        "conversation_traces": None,
        "started_at": "2026-08-25T13:00:00+00:00",
        "completed_at": "2026-08-25T13:00:05+00:00",
        "suspended_ms": 0,
        "continuation_of": None,
        "record_version": 1,
        "record_json": json.dumps(
            {"turn_output": {"turn_key": turn_key, "status": "completed",
                             "success": True, "command_outputs": []}}
        ),
    }


@pytest.fixture
def recorded_workflow_path(tmp_path, monkeypatch) -> str:
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    workflow = tmp_path / "recorded_workflow"
    workflow.mkdir()
    return str(workflow)


@pytest.fixture
def recorded_store(recorded_workflow_path) -> str:
    """A store holding one reusable message and one the policy withheld."""
    db_path = state_paths.observability_db(recorded_workflow_path)
    store = obs.ObservabilityStore(db_path)
    redactor = obs.Redactor()
    assert store.mint_conversation_id("chan1") == 1
    store.record_conversation_label("chan1", 1, "Recorded", "Earlier session")

    conn = store._connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        for turn_key, message in (
            (MULTILINE_TURN, MULTILINE_MESSAGE),
            (WITHHELD_TURN, WITHHELD_MESSAGE),
        ):
            assert store.upsert_turn_row(
                conn, _turn_row(turn_key, message), [], redactor
            )
        conn.commit()
    finally:
        conn.close()
    return db_path


@pytest.fixture
def recorded_server(recorded_store, recorded_workflow_path):
    srv = run_chatbot_server.ChatbotServer(
        recorded_store, workflow_path=recorded_workflow_path, port=0
    )
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    thread.join(timeout=5)


def test_a_recorded_message_survives_the_round_trip_through_the_store(recorded_store):
    """The line breaks the composer has to preserve are really in the record."""
    served = obs.ObservabilityStore(recorded_store).get_turn(MULTILINE_TURN)
    assert served["user_message"] == MULTILINE_MESSAGE
    assert "\n" in served["user_message"]


def test_reusing_a_recorded_message_in_a_real_dom(recorded_server):
    """Copy, edit and send, in a browser, against real recorded evidence.

    Asserts the negative space as well as the feature: no request leaves the
    page when evidence is opened or copied, the record is unchanged
    afterwards, a withheld message offers nothing to reuse, and a turn is
    submitted only when Send is pressed.
    """
    jsdom_root = os.environ.get("TEST_JSDOM_ROOT")
    if not jsdom_root:
        pytest.skip("Set TEST_JSDOM_ROOT to run DOM integration with jsdom")
    script = Path(__file__).with_name("chatbot_reuse_message_dom.cjs")
    result = subprocess.run(
        [
            "node", str(script), jsdom_root,
            f"http://127.0.0.1:{recorded_server.port}/?token={recorded_server.token}",
            MULTILINE_TURN, WITHHELD_TURN, MULTILINE_MESSAGE,
        ],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_resending_the_same_text_is_a_new_turn_not_the_old_one(app_module):
    """Identity, on the real route: same words, different turn.

    The reuse action promises a new execution rather than a replay. Sending
    the same text twice through the deterministic route is what that promise
    looks like at the server: two turn keys, two records, and the first one
    still readable exactly as it was.
    """
    channel_id = _channel("reuse")
    query = "what can i do"

    client, headers, first = _stream_a_real_deterministic_turn(
        app_module, channel_id, query=query
    )
    assert first.status_code == 200, first.text
    first_key = first.headers["X-FW-Turn-Key"]
    first_output = [
        json.loads(line) for line in first.text.splitlines() if line.strip()
    ][-1]
    assert first_output["type"] == "output"
    first_logical = first_output["turn_key"]

    stored_before = client.get(f"/turns/{first_key}", headers=headers)
    assert stored_before.status_code == 200, stored_before.text

    # A second, ordinary submission of the same words. The registry dedupes
    # RETRIES of one submission by idempotency key, so a genuinely new send
    # has to wait for the first to retire — which it has, by now.
    for _ in range(50):
        if not app_module.turn_registry.has_active(channel_id):
            break
        time.sleep(0.05)
    second = client.post(
        "/invoke_agent_stream",
        headers=headers,
        json={"user_query": query, "timeout_seconds": 30},
    )
    assert second.status_code == 200, second.text
    second_key = second.headers["X-FW-Turn-Key"]
    second_output = [
        json.loads(line) for line in second.text.splitlines() if line.strip()
    ][-1]
    assert second_output["type"] == "output"

    assert second_key != first_key
    assert second_output["turn_key"] != first_logical

    # The earlier turn is still there, unchanged by the resend.
    stored_after = client.get(f"/turns/{first_key}", headers=headers)
    assert stored_after.status_code == 200
    assert stored_after.json() == stored_before.json()
