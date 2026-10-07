"""Switching evidence sources without relaunching the page (fix-9eg.7.2).

The switching itself was already delivered — /api/select_workflow and the
picker. What was
missing is the boundary between sources: the page carried the previous
source's selected turn, experiment scoping, rendered detail and live chat
binding straight into the next one, so a workflow switch could leave the
composer pointed at the server it had just left.

Everything here runs against real chatbot servers over stores seeded in
tmp_path, with the page's own code in a browser DOM. Nothing is stubbed and
nothing reaches a model: the two workflows deliberately share a turn key, so a
detail pane that failed to reset would show the wrong source's evidence with a
key that still resolves.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path

import pytest

from fastworkflow import state_paths
from fastworkflow.observability import store as obs
from fastworkflow.run_chatbot import server as run_chatbot_server

# One key, two stores: the collision is the point.
SHARED_TURN = "20260901T090000-shared"
ARTIFACT_A = "a" * 32
ARTIFACT_PAYLOAD_A = "artifact that belongs to source A"


def _turn_row(turn_key: str, source: str, artifacts: dict) -> dict:
    return {
        "turn_key": turn_key,
        "channel_id": "chatbot",
        "conversation_id": 1,
        "ordinal": 1,
        "user_message": f"message recorded in {source}",
        "refined_user_message": None,
        "entry_workflow_name": source,
        "entry_context": "*",
        "status": "completed",
        "success": 1,
        "failure_reason": None,
        "answer": f"answer recorded in {source}",
        "conversation_summary": None,
        "conversation_traces": None,
        "started_at": "2026-09-01T09:00:00+00:00",
        "completed_at": "2026-09-01T09:00:02+00:00",
        "suspended_ms": 0,
        "continuation_of": None,
        "record_version": 1,
        "record_json": json.dumps(
            {
                "turn_output": {
                    "turn_key": turn_key,
                    "status": "completed",
                    "success": True,
                    "command_outputs": [
                        {
                            "command_name": f"command_of_{source}",
                            "command_response": {
                                "response": "done",
                                "success": True,
                                "artifacts": artifacts,
                            },
                        }
                    ],
                }
            }
        ),
    }


def _seed_workflow(
    root: Path, name: str, record_artifacts: dict, artifact_rows: list | None = None
) -> str:
    workflow = root / name
    (workflow / "_commands").mkdir(parents=True)
    db_path = state_paths.observability_db(str(workflow))
    store = obs.ObservabilityStore(db_path)
    redactor = obs.Redactor()
    store.mint_conversation_id("chatbot")
    store.record_conversation_label("chatbot", 1, f"Conversation in {name}", name)
    conn = store._connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        assert store.upsert_turn_row(
            conn,
            _turn_row(SHARED_TURN, name, record_artifacts),
            artifact_rows or [],
            redactor,
        )
        conn.commit()
    finally:
        conn.close()
    return str(workflow)


@pytest.fixture
def two_sources(tmp_path, monkeypatch):
    """Two live workflows that share a turn key."""
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    # Source A's artifact is offloaded, so showing it costs a real HTTP read
    # from A's store — which is what must not happen once B is open.
    source_a = _seed_workflow(
        tmp_path,
        "source_a",
        record_artifacts={
            "note": {
                "__fw_artifact_ref__": ARTIFACT_A,
                "size": len(ARTIFACT_PAYLOAD_A),
                "content_type": "text/plain",
                "content_encoding": None,
                "error": None,
            }
        },
        artifact_rows=[
            {
                "artifact_id": ARTIFACT_A,
                "turn_key": SHARED_TURN,
                "channel_id": "chatbot",
                "span_id": None,
                "key": "note",
                "content_type": "text/plain",
                "size_bytes": len(ARTIFACT_PAYLOAD_A),
                "sha256": "z",
                "inline_value": ARTIFACT_PAYLOAD_A.encode(),
                "error": None,
            }
        ],
    )
    source_b = _seed_workflow(
        tmp_path, "source_b", record_artifacts={"note": "artifact of source_b"}
    )

    return {
        "a": source_a,
        "b": source_b,
        "db_a": state_paths.observability_db(source_a),
    }


def _serve(**kwargs):
    server = run_chatbot_server.ChatbotServer(port=0, **kwargs)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


@pytest.fixture
def live_server(two_sources):
    """The page's own server, opened on source A."""
    server, thread = _serve(
        db_path=two_sources["db_a"], workflow_path=two_sources["a"]
    )
    yield server
    server.shutdown()
    thread.join(timeout=5)


def test_the_two_sources_really_collide(two_sources):
    """The fixture's premise: one turn key, two different records."""
    in_a = obs.ObservabilityStore(two_sources["db_a"]).get_turn(SHARED_TURN)
    in_b = obs.ObservabilityStore(
        state_paths.observability_db(two_sources["b"])
    ).get_turn(SHARED_TURN)
    assert in_a["turn_key"] == in_b["turn_key"] == SHARED_TURN
    assert in_a["answer"] != in_b["answer"]


def test_switching_sources_in_a_real_dom(two_sources, live_server):
    """Workflow → workflow, in a browser, over real servers.

    Asserts the boundary in both directions: everything scoped to the source
    being left is gone (selected turn, rendered detail, chat transcript and
    server binding), and a re-apply of the SAME session
    destroys none of it.
    """
    jsdom_root = os.environ.get("TEST_JSDOM_ROOT")
    if not jsdom_root:
        pytest.skip("Set TEST_JSDOM_ROOT to run DOM integration with jsdom")
    script = Path(__file__).with_name("chatbot_source_switch_dom.cjs")
    result = subprocess.run(
        [
            "node", str(script), jsdom_root,
            f"http://127.0.0.1:{live_server.port}/?token={live_server.token}",
            two_sources["b"],
            SHARED_TURN,
            ARTIFACT_A,
            ARTIFACT_PAYLOAD_A,
        ],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_finder_reports_a_scope_the_store_cannot_resolve(
    two_sources, live_server
):
    """fix-neo2: a refused scoped read must end the search, not hang it.

    The refusal is real — the chatbot server answers 400 for an attempt that
    is not a number — and the recovery is real: clearing the scope searches
    the same store successfully.
    """
    jsdom_root = os.environ.get("TEST_JSDOM_ROOT")
    if not jsdom_root:
        pytest.skip("Set TEST_JSDOM_ROOT to run DOM integration with jsdom")
    script = Path(__file__).with_name("chatbot_finder_stale_scope_dom.cjs")
    result = subprocess.run(
        [
            "node", str(script), jsdom_root,
            f"http://127.0.0.1:{live_server.port}/?token={live_server.token}",
            "not-an-attempt",
        ],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_without_the_boundary_the_old_source_survives_the_switch(
    two_sources, live_server, tmp_path
):
    """The same switch, against a private copy with the reset removed.

    A guard against a test that would pass either way: this asserts the
    defect, so if the boundary ever stops being the thing that clears the
    page, the test above is known to still be measuring it. The shared page
    file is never modified — the copy is loaded as a string with the document
    URL set to the real server, so its requests still go there.
    """
    jsdom_root = os.environ.get("TEST_JSDOM_ROOT")
    if not jsdom_root:
        pytest.skip("Set TEST_JSDOM_ROOT to run DOM integration with jsdom")
    source = run_chatbot_server.load_index_html().decode("utf-8")
    call = "    resetSourceScopedState();\n"
    assert source.count(call) == 1, "the reset call moved; update this copy"
    private_page = tmp_path / "index-without-the-boundary.html"
    private_page.write_text(source.replace(call, ""), encoding="utf-8")

    script = Path(__file__).with_name("chatbot_source_switch_mutation_dom.cjs")
    result = subprocess.run(
        [
            "node", str(script), jsdom_root, str(private_page),
            f"http://127.0.0.1:{live_server.port}/?token={live_server.token}",
            two_sources["b"], SHARED_TURN,
        ],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
