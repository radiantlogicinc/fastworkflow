"""Artifacts on the turn page and on the step that returned them, in a real DOM.

The turn page uses the chat's own link and side panel rather than a second
renderer, and a step page shows the artifacts of exactly the dispatches that
step made. Attribution is by ``command_call_id`` only: the id a
``fw.agent.tool_call`` / ``fw.command.execute`` span carries, or the
``execution_records`` ref naming one of the step's spans. The seeded turn has a
step attributed each way, a step whose command returned nothing, and an
artifact-bearing output with no call id at all, which must appear on the turn
and on no step.

Real store, real server, real page; nothing is stubbed.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

import pytest

from fastworkflow import state_paths, tracing
from fastworkflow.observability import store as obs
from fastworkflow.run_chatbot import server as run_chatbot_server
from tests import test_run_chatbot_server as chatbot_fixtures

workflow_path = chatbot_fixtures.workflow_path

AGENT_TURN = "20260825T130000-agent"
CALL_TODO = "1" * 32
CALL_LIST = "2" * 32
CALL_EXPORT = "3" * 32
CSV_TEXT = "item,done\nmilk,no"


@pytest.fixture
def agent_turn_db(workflow_path) -> str:
    db_path = state_paths.observability_db(workflow_path)
    store = obs.ObservabilityStore(db_path)
    redactor = obs.Redactor()
    conv_id = store.mint_conversation_id("chan-agent")

    record = {
        "turn_output": {
            "turn_key": AGENT_TURN,
            "status": "completed",
            "success": True,
            "command_outputs": [
                {
                    "command_name": "add_todo",
                    "command_call_id": CALL_TODO,
                    "command_response": {
                        "response": "added",
                        "success": True,
                        "artifacts": {
                            "note": "hello inline artifact",
                            "report": {
                                "__fw_artifact_ref__": chatbot_fixtures.HTML_ARTIFACT_ID,
                                "size": len(chatbot_fixtures.HTML_PAYLOAD),
                                "content_type": "text/html",
                                "content_encoding": None,
                                "error": None,
                            },
                        },
                    },
                },
                {
                    "command_name": "list_todos",
                    "command_call_id": CALL_LIST,
                    "command_response": {"response": "milk", "success": True},
                },
                {
                    "command_name": "export_todos",
                    "command_call_id": CALL_EXPORT,
                    "command_response": {
                        "response": "exported",
                        "success": True,
                        "artifacts": {"csv": CSV_TEXT},
                    },
                },
                {
                    "command_name": "summarize",
                    "command_response": {
                        "response": "ok",
                        "success": True,
                        "artifacts": {"orphan": "no call id recorded"},
                    },
                },
            ],
        },
        "execution_records": [
            {"command_call_id": CALL_TODO, "parent_call_id": None,
             "command_ordinal": 0, "span_id": "s-exec1"},
            {"command_call_id": CALL_LIST, "parent_call_id": None,
             "command_ordinal": 1, "span_id": "s-exec2"},
            {"command_call_id": CALL_EXPORT, "parent_call_id": None,
             "command_ordinal": 2, "span_id": "s-exec3"},
        ],
    }
    artifact_rows = [
        {
            "artifact_id": chatbot_fixtures.HTML_ARTIFACT_ID,
            "turn_key": AGENT_TURN,
            "channel_id": "chan-agent",
            "span_id": None,
            "key": "report",
            "content_type": "text/html",
            "size_bytes": len(chatbot_fixtures.HTML_PAYLOAD),
            "sha256": "x",
            "inline_value": chatbot_fixtures.HTML_PAYLOAD.encode(),
            "error": None,
        },
    ]

    t0 = time.time_ns()
    ms = 1_000_000

    def span(span_id, name, parent, start, end, **extra):
        return tracing.Span(
            span_id=span_id, trace_id=AGENT_TURN, name=name,
            kind=extra.pop("kind", "internal"), channel_id="chan-agent",
            parent_span_id=parent, start_ns=t0 + start * ms,
            end_ns=t0 + end * ms, status="ok",
            attributes=extra.pop("attributes", {}), **extra,
        )

    spans = [
        span("s-root", "fw.turn", None, 0, 100,
             attributes={"user_message": "add milk and export"}),
        span("s-exec", "fw.agent.execute", "s-root", 1, 99,
             attributes={"agent_input": "add milk, then export"}),
        # Step 1: attributed through the spans' own command_call_id.
        span("s-step1", "fw.agent.step", "s-exec", 2, 20,
             attributes={"tool_name": "execute_workflow_query"}),
        span("s-tool1", "fw.agent.tool_call", "s-step1", 3, 19, kind="tool",
             command_name="add_todo",
             attributes={"raw_command": "add_todo milk",
                         "command_call_id": CALL_TODO}),
        span("s-exec1", "fw.command.execute", "s-tool1", 4, 18, kind="tool",
             command_name="add_todo",
             attributes={"raw_command": "add_todo milk",
                         "command_call_id": CALL_TODO}),
        # Step 2: a dispatch that returned no artifact.
        span("s-step2", "fw.agent.step", "s-exec", 21, 40,
             attributes={"tool_name": "execute_workflow_query"}),
        span("s-tool2", "fw.agent.tool_call", "s-step2", 22, 39, kind="tool",
             command_name="list_todos",
             attributes={"command_call_id": CALL_LIST}),
        span("s-exec2", "fw.command.execute", "s-tool2", 23, 38, kind="tool",
             command_name="list_todos",
             attributes={"command_call_id": CALL_LIST}),
        # Step 3: the spans carry no id; the record's execution_records ref
        # names this step's execute span.
        span("s-step3", "fw.agent.step", "s-exec", 41, 60,
             attributes={"tool_name": "execute_workflow_query"}),
        span("s-tool3", "fw.agent.tool_call", "s-step3", 42, 59, kind="tool",
             command_name="export_todos", attributes={}),
        span("s-exec3", "fw.command.execute", "s-tool3", 43, 58, kind="tool",
             command_name="export_todos", attributes={}),
        # Step 4: nothing joins it to any output, so it claims none.
        span("s-step4", "fw.agent.step", "s-exec", 61, 80,
             attributes={"tool_name": "execute_workflow_query"}),
        span("s-tool4", "fw.agent.tool_call", "s-step4", 62, 79, kind="tool",
             command_name="summarize", attributes={}),
    ]

    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        assert store.upsert_turn_row(
            conn,
            chatbot_fixtures._turn_row(
                AGENT_TURN, "chan-agent", conv_id, 1, "completed", 1, record),
            artifact_rows,
            redactor,
        )
        store.upsert_span_rows(conn, spans, redactor)
    return db_path


@pytest.fixture
def agent_turn_server(agent_turn_db, workflow_path):
    srv = run_chatbot_server.ChatbotServer(
        agent_turn_db, workflow_path=workflow_path, port=0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    thread.join(timeout=5)


def _run_dom(script_name: str, *args, timeout: int = 90):
    jsdom_root = os.environ.get("TEST_JSDOM_ROOT")
    if not jsdom_root:
        pytest.skip("Set TEST_JSDOM_ROOT to run DOM integration with jsdom")
    script = Path(__file__).with_name(script_name)
    result = subprocess.run(
        ["node", str(script), jsdom_root, *[str(a) for a in args]],
        capture_output=True, text=True, timeout=timeout,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_turn_and_step_artifacts_open_the_chat_panel_in_a_real_dom(agent_turn_server):
    _run_dom(
        "chatbot_turn_artifacts_dom.cjs",
        f"http://127.0.0.1:{agent_turn_server.port}/?token={agent_turn_server.token}",
        AGENT_TURN,
        chatbot_fixtures.HTML_ARTIFACT_ID,
    )
