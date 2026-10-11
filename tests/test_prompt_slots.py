"""The exact prompt an over-cap LLM call was sent: stored as pieces, rebuilt on read.

An ``fw.llm.call`` whose ``messages`` exceed ``tracing.MAX_ATTR_BYTES`` keeps
only a cut envelope on the span. The prompt-slot enricher splits the prompt at
DSPy field markers, the SQLite sink stores each distinct piece once per turn in
``prompt_slots``, and ``ObservabilityStore.prompt_as_sent`` rebuilds the
messages and checks them against the recorded digest.

Real store, real tracing, real sink, real chatbot server (no mocks).
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import threading
import urllib.request
import uuid
from pathlib import Path

import pytest

from fastworkflow import state_paths, tracing
from fastworkflow.observability import prompt_slots
from fastworkflow.observability import store as obs
from fastworkflow.run_chatbot import server as run_chatbot_server

SYSTEM = "You are an Agent.\n" + ("Commands available in the current context.\n" * 500)


def _messages(steps: int) -> list[dict]:
    trajectory = "".join(
        f"[[ ## thought_{i} ## ]]\nthought {i}\n\n"
        f"[[ ## observation_{i} ## ]]\nObservation O{i + 1} (execute_workflow_query, in Identity)\n"
        f"Context is now 'ItemExplorer'\n\n"
        for i in range(steps)
    )
    user = (
        "[[ ## user_query ## ]]\nfind Sam Poe\n\n"
        f"[[ ## trajectory ## ]]\n{trajectory}"
        "Respond with the corresponding output fields."
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


class _Host:
    def __init__(self, sink, turn_key: str, channel_id: str = "prompt-slots") -> None:
        self.trace_sink = sink
        self.current_turn_key = turn_key
        self.observability_channel_id = channel_id
        self.trace_span_stack: list = []


class _RecordingSink:
    def __init__(self) -> None:
        self.spans: list[tracing.Span] = []

    def emit_span(self, span: tracing.Span) -> None:
        self.spans.append(span)

    def emit_turn_record(self, record) -> None:
        pass

    def record_conversation_label(self, *args) -> None:
        pass

    def emit_distillation_record(self, *args) -> None:
        pass


@pytest.fixture(autouse=True)
def _enrichment(monkeypatch):
    # Every sink prunes at startup; the turns recorded here must outlive the
    # next sink's startup prune until a test prunes on purpose.
    monkeypatch.setenv("FW_OBS_RETENTION_DAYS", "100000")
    prompt_slots.install_prompt_slot_enrichment()
    yield
    prompt_slots.uninstall_prompt_slot_enrichment()


@pytest.fixture
def db_path(tmp_path) -> str:
    return str(tmp_path / "observability.sqlite3")


def _record_calls(db_path: str, turn_key: str, prompts: list[list[dict]],
                  channel_id: str = "prompt-slots") -> list[str]:
    """Emit one fw.llm.call per prompt through real tracing; return span ids."""
    sink = obs.SQLiteTraceSink(db_path)
    host = _Host(sink, turn_key, channel_id)
    span_ids = []
    try:
        for messages in prompts:
            span = tracing.start_span(
                host,
                tracing.SPAN_LLM_CALL,
                kind=tracing.KIND_LLM,
                attributes={"model": "test", "messages": json.dumps(messages, ensure_ascii=False)},
                use_stack=False,
            )
            assert span is not None
            tracing.end_span(host, span, attributes={"output": "ok"})
            span_ids.append(span.span_id)
        assert sink.flush()
    finally:
        sink.close()
    return span_ids


def _slot_rows(db_path: str, turn_key: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM prompt_slots WHERE turn_key=?", (turn_key,)
        ).fetchone()[0]
    finally:
        conn.close()


def test_pieces_join_back_to_the_text_and_rebuild_verifies():
    messages_json = json.dumps(_messages(3), ensure_ascii=False)
    built = prompt_slots.build(messages_json)
    assert built is not None
    ref, texts = built
    for message in json.loads(messages_json):
        assert "".join(prompt_slots.split_text(message["content"])) == message["content"]

    rebuilt = prompt_slots.rebuild(ref, texts)

    assert rebuilt["verified"] is True
    assert rebuilt["messages"] == json.loads(messages_json)
    assert rebuilt["missing"] == [] and rebuilt["altered"] == []


def test_equivalent_json_formatting_still_verifies():
    messages = _messages(3)
    built = prompt_slots.build(json.dumps(messages, ensure_ascii=True, indent=2))
    assert built is not None
    ref, texts = built

    assert prompt_slots.rebuild(ref, texts)["verified"] is True


def test_an_over_cap_call_is_stored_as_pieces_and_rebuilt_byte_for_byte(db_path):
    turn_key = f"20260901T000000.000000Z-{uuid.uuid4().hex[:12]}"
    messages = _messages(6)
    assert len(json.dumps(messages).encode()) > tracing.MAX_ATTR_BYTES

    [span_id] = _record_calls(db_path, turn_key, [messages])

    [row] = obs.ReadOnlyObservabilityStore(db_path).get_spans(turn_key)
    attributes = json.loads(row["attributes"])
    assert attributes["messages"]["truncated"] is True
    assert tracing.ATTR_PROMPT_SLOT_TEXTS not in attributes
    assert SYSTEM not in row["attributes"]
    assert attributes[prompt_slots.REF_ATTRIBUTE]["contract"] == prompt_slots.PROMPT_SLOTS_CONTRACT

    prompt = obs.ReadOnlyObservabilityStore(db_path).prompt_as_sent(turn_key, span_id)

    assert prompt["available"] is True
    assert prompt["verified"] is True
    assert prompt["messages"] == messages


def test_pieces_shared_by_successive_steps_are_stored_once(db_path):
    turn_key = f"20260901T000000.000000Z-{uuid.uuid4().hex[:12]}"
    first, second = _messages(5), _messages(6)
    expected = set(prompt_slots.build(json.dumps(first, ensure_ascii=False))[1])
    expected |= set(prompt_slots.build(json.dumps(second, ensure_ascii=False))[1])

    span_ids = _record_calls(db_path, turn_key, [first, second])

    assert _slot_rows(db_path, turn_key) == len(expected)
    store = obs.ReadOnlyObservabilityStore(db_path)
    assert [store.prompt_as_sent(turn_key, s)["verified"] for s in span_ids] == [True, True]


def test_a_call_under_the_cap_keeps_its_messages_and_stores_no_pieces(db_path):
    turn_key = f"20260901T000000.000000Z-{uuid.uuid4().hex[:12]}"
    small = [{"role": "user", "content": "[[ ## user_query ## ]]\nhello"}]

    [span_id] = _record_calls(db_path, turn_key, [small])

    [row] = obs.ReadOnlyObservabilityStore(db_path).get_spans(turn_key)
    assert json.loads(json.loads(row["attributes"])["messages"]) == small
    assert _slot_rows(db_path, turn_key) == 0
    assert obs.ReadOnlyObservabilityStore(db_path).prompt_as_sent(turn_key, span_id) == {
        "available": False, "reason": "this call recorded no prompt pieces",
    }


def test_the_piece_texts_never_travel_inside_the_attribute_bag():
    sink = _RecordingSink()
    host = _Host(sink, f"turn-{uuid.uuid4().hex}")

    span = tracing.start_span(
        host, tracing.SPAN_LLM_CALL, kind=tracing.KIND_LLM,
        attributes={"messages": json.dumps(_messages(6))}, use_stack=False,
    )
    tracing.end_span(host, span)

    [emitted] = sink.spans
    assert tracing.ATTR_PROMPT_SLOT_TEXTS not in emitted.attributes
    assert emitted.prompt_slots and SYSTEM in emitted.prompt_slots.values()


def test_a_redacted_piece_is_reported_and_the_prompt_is_not_verified(db_path):
    turn_key = f"20260901T000000.000000Z-{uuid.uuid4().hex[:12]}"
    messages = _messages(6)
    messages[1]["content"] += "\n\n[[ ## observation_9 ## ]]\nkey sk-ABCDEFGHIJKLMNOPQRSTUV"

    [span_id] = _record_calls(db_path, turn_key, [messages])

    prompt = obs.ReadOnlyObservabilityStore(db_path).prompt_as_sent(turn_key, span_id)
    assert prompt["verified"] is False
    assert len(prompt["altered"]) == 1 and prompt["missing"] == []
    assert "sk-ABCDEFGHIJKLMNOPQRSTUV" not in json.dumps(prompt)
    assert "[REDACTED]" in prompt["messages"][1]["content"]


def test_a_store_created_before_the_table_gains_it_and_keeps_its_rows(db_path):
    turn_key = f"20260901T000000.000000Z-{uuid.uuid4().hex[:12]}"
    _record_calls(db_path, turn_key, [[{"role": "user", "content": "hi"}]])
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DROP TABLE prompt_slots")
        conn.commit()
        version = conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()

    obs.ObservabilityStore(db_path)

    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == version == obs.SCHEMA_VERSION
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='prompt_slots'"
        ).fetchone() is not None
    finally:
        conn.close()
    assert len(obs.ReadOnlyObservabilityStore(db_path).get_spans(turn_key)) == 1


def test_a_reader_of_a_store_without_the_table_reports_pieces_missing(db_path):
    turn_key = f"20260901T000000.000000Z-{uuid.uuid4().hex[:12]}"
    [span_id] = _record_calls(db_path, turn_key, [_messages(6)])
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DROP TABLE prompt_slots")
        conn.commit()
    finally:
        conn.close()

    prompt = obs.ReadOnlyObservabilityStore(db_path).prompt_as_sent(turn_key, span_id)

    assert prompt["available"] is True and prompt["verified"] is False
    assert prompt["missing"] and prompt["altered"] == []


def test_forget_channel_and_clear_conversations_erase_the_pieces(db_path):
    kept = f"20260901T000000.000000Z-{uuid.uuid4().hex[:12]}"
    forgotten = f"20260901T000001.000000Z-{uuid.uuid4().hex[:12]}"
    _record_calls(db_path, kept, [_messages(6)], channel_id="keep")
    _record_calls(db_path, forgotten, [_messages(6)], channel_id="forget")

    deleted = obs.ObservabilityStore(db_path).forget_channel("forget")

    assert deleted["prompt_slots"] > 0
    assert _slot_rows(db_path, forgotten) == 0
    assert _slot_rows(db_path, kept) > 0

    obs.ObservabilityStore(db_path).clear_conversations()
    assert _slot_rows(db_path, kept) == 0


def test_retention_prunes_the_pieces_of_turns_past_the_horizon(db_path):
    old = f"20200101T000000.000000Z-{uuid.uuid4().hex[:12]}"
    recent = f"29990101T000000.000000Z-{uuid.uuid4().hex[:12]}"
    _record_calls(db_path, old, [_messages(6)])
    _record_calls(db_path, recent, [_messages(6)])

    deleted = obs.ObservabilityStore(db_path).prune(retention_days=30)

    assert deleted["prompt_slots"] > 0
    assert _slot_rows(db_path, old) == 0
    assert _slot_rows(db_path, recent) > 0


def test_the_chatbot_serves_the_rebuilt_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    workflow = tmp_path / "my_workflow"
    workflow.mkdir()
    db_path = state_paths.observability_db(str(workflow))
    turn_key = f"20260901T000000.000000Z-{uuid.uuid4().hex[:12]}"
    messages = _messages(6)
    [span_id] = _record_calls(db_path, turn_key, [messages])

    srv = run_chatbot_server.ChatbotServer(db_path, workflow_path=str(workflow), port=0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        def get(path):
            req = urllib.request.Request(f"http://127.0.0.1:{srv.port}{path}")
            req.add_header("Authorization", f"Bearer {srv.token}")
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    return resp.status, json.loads(resp.read())
            except urllib.error.HTTPError as err:
                return err.code, None

        status, body = get(f"/api/prompt/{turn_key}/{span_id}")
        assert status == 200
        assert body["prompt"]["verified"] is True
        assert body["prompt"]["messages"] == messages
        assert get(f"/api/prompt/{turn_key}/not-a-span")[0] == 404
    finally:
        srv.shutdown()
        thread.join(timeout=5)


def test_every_llm_call_offers_the_prompt_as_sent(tmp_path):
    """Whole messages render in place; an over-cap call's are fetched rebuilt."""
    jsdom_root = os.environ.get("TEST_JSDOM_ROOT")
    if not jsdom_root:
        pytest.skip("Set TEST_JSDOM_ROOT to run DOM integration with jsdom")
    page = tmp_path / "index.html"
    page.write_bytes(run_chatbot_server.load_index_html())
    script = Path(__file__).with_name("chatbot_prompt_as_sent_dom.cjs")
    result = subprocess.run(
        ["node", str(script), jsdom_root, str(page)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
