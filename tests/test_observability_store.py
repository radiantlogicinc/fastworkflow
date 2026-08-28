"""Phase 2 observability: SQLite store + background writer (bead fix-kw7.3).

Store contract tests run against real SQLite in tmp_path (design §6 — no
mocks for stores/serialization); the WEC end-to-end tests reuse the fixture
patterns of tests/test_tracing_phase1.py (real todo_list_workflow, fakes only
at the NLU/agent boundary).
"""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import fastworkflow
from fastworkflow import TurnStatus, tracing
from fastworkflow import observability_store as obs
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.workflow_execution_context import WorkflowExecutionContext


@pytest.fixture
def todo_workflow_path() -> str:
    return str(Path(__file__).parent.joinpath("todo_list_workflow").resolve())


@pytest.fixture
def initialized_fastworkflow():
    fastworkflow.init({})
    from fastworkflow.command_routing import RoutingRegistry

    RoutingRegistry.clear_registry()
    yield
    RoutingRegistry.clear_registry()


@pytest.fixture
def db_path(tmp_path) -> str:
    return str(tmp_path / "observability.sqlite3")


@pytest.fixture
def sink(db_path):
    s = obs.SQLiteTraceSink(db_path)
    yield s
    s.close()


def _make_assistant_ctx(todo_workflow_path, monkeypatch, sink):
    wf = fastworkflow.Workflow.create(
        todo_workflow_path,
        workflow_id_str=f"obs-assist-{uuid.uuid4().hex}",
    )
    ctx = WorkflowExecutionContext(run_as_agent=False, trace_sink=sink)
    ctx.bind_app_workflow(wf)

    def fake_invoke(cls, session, command: str):
        return fastworkflow.CommandOutput(
            command_name=command.split()[0] if command else "",
            command_response=fastworkflow.CommandResponse(response=f"ok:{command}"),
        )

    monkeypatch.setattr(CommandExecutor, "invoke_command", classmethod(fake_invoke))
    return ctx, wf


def _make_agent_ctx(todo_workflow_path, monkeypatch, sink):
    ctx = WorkflowExecutionContext(run_as_agent=True, trace_sink=sink)
    wf = fastworkflow.Workflow.create(
        todo_workflow_path,
        workflow_id_str=f"obs-agent-{uuid.uuid4().hex}",
    )
    ctx.bind_app_workflow(wf)
    monkeypatch.setattr(
        "fastworkflow.workflow_agent.build_query_with_next_steps",
        lambda user_query, session, **kwargs: user_query,
    )
    monkeypatch.setattr(
        "fastworkflow.workflow_agent._what_can_i_do", lambda session: "commands"
    )
    monkeypatch.setattr(ctx, "_ensure_agent_initialized", lambda: None)
    monkeypatch.setattr(
        ctx,
        "_extract_conversation_summary",
        lambda user_query, actions, final: ("summary", "{}"),
    )
    return ctx, wf


def _rows(db_path: str, query: str, params=()):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(query, params).fetchall()]
    finally:
        conn.close()


# ----------------------------------------------------------------------
# Schema, versioning, file posture
# ----------------------------------------------------------------------


class TestSchema:
    def test_schema_created_with_version_and_vacuum(self, db_path):
        store = obs.ObservabilityStore(db_path)
        conn = sqlite3.connect(db_path)
        try:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
            # 2 = INCREMENTAL [R12], set at creation before any table
            assert conn.execute("PRAGMA auto_vacuum").fetchone()[0] == 2
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        finally:
            conn.close()
        assert {
            "conversations",
            "turns",
            "feedback",
            "spans",
            "artifacts",
            "train_runs",
            "diagnostics",
        } <= tables
        assert store.db_size_bytes() > 0

    def test_file_posture(self, db_path):
        obs.ObservabilityStore(db_path)
        mode = stat.S_IMODE(os.stat(db_path).st_mode)
        assert mode == 0o600  # [R4]
        dir_mode = stat.S_IMODE(os.stat(os.path.dirname(db_path)).st_mode)
        assert dir_mode == 0o700

    def test_refuses_newer_schema(self, db_path):
        obs.ObservabilityStore(db_path)
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA user_version = 99")
        conn.commit()
        conn.close()
        with pytest.raises(obs.IncompatibleObservabilityDB):
            obs.ObservabilityStore(db_path)  # [R11]


# ----------------------------------------------------------------------
# Identity: conversation-id minting [R1] and labels [R15]
# ----------------------------------------------------------------------


class TestConversationIdentity:
    def test_mint_is_sequential_per_channel(self, db_path):
        store = obs.ObservabilityStore(db_path)
        assert store.mint_conversation_id("chan-a") == 1
        assert store.mint_conversation_id("chan-a") == 2
        assert store.mint_conversation_id("chan-b") == 1
        assert store.mint_conversation_id("chan-a") == 3

    def test_record_conversation_label_upserts(self, db_path):
        store = obs.ObservabilityStore(db_path)
        conv = store.mint_conversation_id("chan-a")
        store.record_conversation_label("chan-a", conv, "Groceries", "Bought milk")
        store.record_conversation_label("chan-a", conv, "Groceries v2", "Bought more")
        rows = store.list_conversations("chan-a")
        assert len(rows) == 1
        assert rows[0]["topic"] == "Groceries v2"
        # Label for a conversation the store never minted still lands [R15]
        store.record_conversation_label("chan-c", 5, "Restored", "s")
        assert store.list_conversations("chan-c")[0]["conversation_id"] == 5


# ----------------------------------------------------------------------
# End-to-end: WEC turn -> DB rows
# ----------------------------------------------------------------------


class TestEndToEnd:
    def test_turn_and_spans_written(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch, db_path, sink
    ):
        ctx, _wf = _make_assistant_ctx(todo_workflow_path, monkeypatch, sink)
        conv = sink.store.mint_conversation_id("chan-e2e")
        ctx.bind_observability_identity(channel_id="chan-e2e", conversation_id=conv)

        turn_output = ctx.process_turn("add_todo buy milk")
        assert sink.flush()

        turns = _rows(db_path, "SELECT * FROM turns WHERE turn_key=?", (turn_output.turn_key,))
        assert len(turns) == 1
        row = turns[0]
        assert row["status"] == "completed"
        assert row["success"] == 1
        assert row["channel_id"] == "chan-e2e"
        assert row["conversation_id"] == conv
        assert row["ordinal"] == 1  # store-assigned [R1]
        assert row["user_message"] == "add_todo buy milk"
        record = json.loads(row["record_json"])
        assert record["turn_output"]["turn_key"] == turn_output.turn_key

        spans = _rows(db_path, "SELECT * FROM spans WHERE trace_id=?", (turn_output.turn_key,))
        names = {s["name"] for s in spans}
        assert tracing.SPAN_TURN in names
        assert tracing.SPAN_AGENT_TOOL_CALL in names
        root = next(s for s in spans if s["name"] == tracing.SPAN_TURN)
        assert root["end_ns"] is not None  # close upserted over the open emission
        assert root["status"] == "completed"

        # Second turn gets ordinal 2
        second = ctx.process_turn("list_todos")
        assert sink.flush()
        row2 = _rows(db_path, "SELECT ordinal FROM turns WHERE turn_key=?", (second.turn_key,))[0]
        assert row2["ordinal"] == 2

    def test_awaiting_then_terminal_transition(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch, db_path, sink
    ):
        ctx, _wf = _make_agent_ctx(todo_workflow_path, monkeypatch, sink)
        ctx.bind_observability_identity(channel_id="chan-susp")

        suspended = SimpleNamespace(suspended=True, clarification="Which task?")
        completed = SimpleNamespace(final_answer="All done")
        mock_agent = MagicMock()
        mock_agent.return_value = suspended
        mock_agent.resume.return_value = completed
        ctx._workflow_tool_agent = mock_agent
        ctx._intent_clarification_agent = MagicMock()

        first = ctx.process_turn("clean up")
        assert sink.flush()
        row = _rows(db_path, "SELECT status, success FROM turns WHERE turn_key=?", (first.turn_key,))[0]
        assert row["status"] == "awaiting_user"  # INSERT at first emission [R2]
        assert row["success"] == 0

        second = ctx.process_turn("the urgent one")
        assert second.turn_key == first.turn_key
        assert sink.flush()
        row = _rows(db_path, "SELECT status, success FROM turns WHERE turn_key=?", (first.turn_key,))[0]
        assert row["status"] == "completed"  # guarded transition [R2]

        # fw.ask_user span closed with the human wait
        ask = _rows(
            db_path,
            "SELECT * FROM spans WHERE trace_id=? AND name=?",
            (first.turn_key, tracing.SPAN_ASK_USER),
        )
        assert len(ask) == 1
        assert ask[0]["end_ns"] is not None
        assert ask[0]["kind"] == "human_wait"

    def test_terminal_row_is_write_once(self, db_path, sink):
        store = sink.store
        redactor = obs.Redactor()
        base = {
            "turn_key": "20260825T000000.000000Z-aaaaaaaaaaaa",
            "channel_id": "c",
            "conversation_id": None,
            "ordinal": None,
            "user_message": "m",
            "refined_user_message": None,
            "entry_workflow_name": "",
            "entry_context": "",
            "status": "completed",
            "success": 1,
            "failure_reason": None,
            "answer": "a",
            "conversation_summary": None,
            "conversation_traces": None,
            "started_at": None,
            "completed_at": None,
            "suspended_ms": 0,
            "continuation_of": None,
            "record_version": 1,
            "record_json": "{}",
        }
        conn = store._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            assert store.upsert_turn_row(conn, dict(base), [], redactor) is True
            # identical retry claims idempotent success
            assert store.upsert_turn_row(conn, dict(base), [], redactor) is True
            # conflicting content against a terminal row is refused
            conflicting = dict(base, record_json='{"x": 1}')
            assert store.upsert_turn_row(conn, conflicting, [], redactor) is False
            conn.commit()
        finally:
            conn.close()
        assert store.get_turn(base["turn_key"])["record_json"] == "{}"


# ----------------------------------------------------------------------
# Span idempotent upsert [R6]
# ----------------------------------------------------------------------


class TestSpanUpsert:
    def test_open_then_close_converges(self, db_path, sink):
        key = "20260825T000000.000000Z-bbbbbbbbbbbb"
        span_id = tracing.root_span_id(key)
        open_span = tracing.Span(
            span_id=span_id, trace_id=key, name="fw.turn", start_ns=100,
            status="open", attributes={"a": 1},
        )
        sink.emit_span(open_span)
        closed = tracing.Span(
            span_id=span_id, trace_id=key, name="fw.turn", start_ns=100,
            end_ns=200, status="completed", attributes={"a": 1, "b": 2},
        )
        sink.emit_span(closed)
        # A late open re-emission must not reopen the closed span
        sink.emit_span(open_span)
        assert sink.flush()

        rows = _rows(db_path, "SELECT * FROM spans WHERE span_id=?", (span_id,))
        assert len(rows) == 1
        assert rows[0]["end_ns"] == 200
        assert rows[0]["status"] == "completed"
        assert json.loads(rows[0]["attributes"])["b"] == 2


# ----------------------------------------------------------------------
# Size policy [R10], redaction [R20], traceback gate
# ----------------------------------------------------------------------


class TestSerializationPolicies:
    def _turn_result(self, artifacts: dict, channel="c"):
        response = fastworkflow.CommandResponse(response="done", artifacts=artifacts)
        output = fastworkflow.CommandOutput(
            command_name="x", command_response=response
        )
        turn_output = fastworkflow.TurnOutput(
            turn_key=fastworkflow.mint_turn_key(),
            status=TurnStatus.COMPLETED,
            answer="done",
            command_outputs=[output],
        )
        return fastworkflow.TurnResult(
            turn_output=turn_output, channel_id=channel, user_message="msg"
        )

    def test_oversized_artifact_offloaded_with_envelope(
        self, db_path, sink, monkeypatch
    ):
        monkeypatch.setenv("FW_OBS_INLINE_ARTIFACT_BYTES", "64")
        big = "y" * 500
        turn_result = self._turn_result({"big_blob": big, "small": "ok"})
        sink.emit_turn_record(turn_result)
        assert sink.flush()

        row = _rows(db_path, "SELECT * FROM turns")[0]
        record = json.loads(row["record_json"])
        artifacts = record["turn_output"]["command_outputs"][0]["command_response"]["artifacts"]
        assert artifacts["small"] == "ok"  # inline below the limit
        envelope = artifacts["big_blob"]
        assert envelope["__fw_artifact_ref__"]
        assert envelope["size"] > 64

        stored = _rows(db_path, "SELECT * FROM artifacts")[0]
        assert stored["artifact_id"] == envelope["__fw_artifact_ref__"]
        assert stored["key"] == "big_blob"
        assert json.loads(stored["inline_value"].decode()) == big

    def test_redaction_of_env_secret_and_shapes(self, db_path, monkeypatch):
        monkeypatch.setenv("LITELLM_API_KEY_TEST", "hunter2secretvalue")
        sink = obs.SQLiteTraceSink(db_path)
        try:
            turn_result = self._turn_result(
                {"leak": "the key is hunter2secretvalue and sk-abcdefghijklmnopqrstu"}
            )
            sink.emit_turn_record(turn_result)
            assert sink.flush()
        finally:
            sink.close()
        row = _rows(db_path, "SELECT record_json FROM turns")[0]
        assert "hunter2secretvalue" not in row["record_json"]
        assert "sk-abcdefghijklmnopqrstu" not in row["record_json"]
        assert "[REDACTED]" in row["record_json"]

    def test_traceback_suppressed_by_default(self, db_path, sink):
        turn_result = self._turn_result({"traceback": "Traceback (most recent...)"})
        sink.emit_turn_record(turn_result)
        assert sink.flush()
        row = _rows(db_path, "SELECT record_json FROM turns")[0]
        assert "most recent" not in row["record_json"]
        assert "FW_OBS_CAPTURE_TRACEBACKS" in row["record_json"]

    def test_unserializable_value_becomes_placeholder(self):
        turn_result = self._turn_result({"weird": object()})
        turn_row, _ = obs.serialize_turn_result(turn_result)
        record = json.loads(turn_row["record_json"])
        artifacts = record["turn_output"]["command_outputs"][0]["command_response"]["artifacts"]
        assert artifacts["weird"]["__fw_unserializable__"] == "object"


# ----------------------------------------------------------------------
# Conversation memory round trip (fix-24f.1)
#
# Every read below filters on a non-NULL conversation_summary. The serializer
# used to hardcode both memory columns to None, which made all of them return
# empty against real traffic while the seeded-row tests above stayed green —
# so these go through emit_turn_record rather than writing rows directly.
# ----------------------------------------------------------------------


class TestConversationMemoryRoundTrip:
    def _turn_result(self, summary, traces, conversation_id=1, channel="c"):
        output = fastworkflow.CommandOutput(
            command_name="x",
            command_response=fastworkflow.CommandResponse(response="done"),
        )
        turn_output = fastworkflow.TurnOutput(
            turn_key=fastworkflow.mint_turn_key(),
            status=TurnStatus.COMPLETED,
            answer="done",
            command_outputs=[output],
        )
        return fastworkflow.TurnResult(
            turn_output=turn_output,
            channel_id=channel,
            conversation_id=conversation_id,
            user_message="msg",
            conversation_summary=summary,
            conversation_traces=traces,
        )

    def test_stamped_turn_is_readable_as_memory(self, db_path, sink):
        sink.emit_turn_record(self._turn_result("first turn", '{"a": 1}'))
        sink.emit_turn_record(self._turn_result("second turn", '{"b": 2}'))
        assert sink.flush()

        store = obs.ObservabilityStore(db_path)
        assert store.count_usable_turns("c", 1) == 2
        window = store.get_memory_window("c", 1, max_turns=10)
        assert [entry["conversation summary"] for entry in window] == [
            "first turn",
            "second turn",
        ]
        assert window[0]["conversation_traces"] == '{"a": 1}'
        assert store.conversation_summaries("c", 1) == [
            {"conversation summary": "first turn"},
            {"conversation summary": "second turn"},
        ]
        assert store.conversation_label_state("c", 1) == ("", 2)
        assert store.get_last_completed_turn_key("c", 1) is not None
        assert [c["conversation_id"] for c in store.list_conversation_summaries("c", 10)] == [1]
        assert len(store.dump_all_conversations("c")[0]["turns"]) == 2

    def test_unstamped_turn_is_a_trace_not_memory(self, db_path, sink):
        """A turn that appended no history entry stays out of every memory read
        even though its row exists (ruling I4's usable-rows invariant)."""
        sink.emit_turn_record(self._turn_result(None, None))
        assert sink.flush()

        store = obs.ObservabilityStore(db_path)
        assert _rows(db_path, "SELECT * FROM turns")  # the row is there
        assert store.count_usable_turns("c", 1) == 0
        assert store.get_memory_window("c", 1, max_turns=10) == []
        assert store.list_conversation_summaries("c", 10) == []

    def test_terminal_emission_fills_columns_an_awaiting_user_row_left_null(
        self, db_path, sink
    ):
        suspended = self._turn_result(None, None)
        suspended.turn_output.status = TurnStatus.AWAITING_USER
        sink.emit_turn_record(suspended)
        assert sink.flush()
        store = obs.ObservabilityStore(db_path)
        assert store.count_usable_turns("c", 1) == 0

        resumed = self._turn_result("the resumed turn", "{}")
        resumed.turn_output.turn_key = suspended.turn_output.turn_key
        sink.emit_turn_record(resumed)
        assert sink.flush()

        assert store.count_usable_turns("c", 1) == 1
        window = store.get_memory_window("c", 1, max_turns=10)
        assert window[0]["conversation summary"] == "the resumed turn"

    def test_feedback_joins_into_the_memory_window(self, db_path, sink):
        turn_result = self._turn_result("a turn with feedback", "{}")
        sink.emit_turn_record(turn_result)
        assert sink.flush()

        store = obs.ObservabilityStore(db_path)
        store.upsert_feedback(
            turn_result.turn_output.turn_key, json.dumps({"nl_feedback": "helpful"})
        )
        window = store.get_memory_window("c", 1, max_turns=10)
        assert window[0]["feedback"] == {"nl_feedback": "helpful"}


# ----------------------------------------------------------------------
# Writer discipline [R13]: drops counted, failures never propagate
# ----------------------------------------------------------------------


class TestWriterDiscipline:
    def test_span_queue_overflow_drops_and_counts(self, db_path, monkeypatch):
        monkeypatch.setenv("FW_OBS_QUEUE_MAX", "1")
        sink = obs.SQLiteTraceSink(db_path)
        try:
            # Stall the writer by holding the DB write lock so the queue fills.
            blocker = sqlite3.connect(db_path, timeout=30.0)
            blocker.execute("BEGIN IMMEDIATE")
            for i in range(50):
                sink.emit_span(
                    tracing.Span(
                        span_id=f"s{i}", trace_id="t", name="fw.agent.tool_call",
                        start_ns=i, status="ok",
                    )
                )
            blocker.rollback()
            blocker.close()
            sink.flush()
        finally:
            sink.close()
        health = obs.ObservabilityStore(db_path).writer_health()
        assert health is not None
        assert health["spans_dropped"] > 0

    def test_store_failure_never_raises_to_emitter(self, db_path, sink, monkeypatch):
        def broken(*args, **kwargs):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(sink.store, "upsert_turn_row", broken)
        turn_output = fastworkflow.TurnOutput(
            turn_key=fastworkflow.mint_turn_key(), status=TurnStatus.COMPLETED
        )
        turn_result = fastworkflow.TurnResult(turn_output=turn_output, user_message="m")
        sink.emit_turn_record(turn_result)  # must not raise
        sink.flush()
        health = sink.store.writer_health()
        assert health["write_errors"] > 0
        assert "disk on fire" in (health["last_error"] or "")

    def test_unopenable_db_yields_no_sink_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "root"))
        workflow_path = str(tmp_path / "wf")
        os.makedirs(workflow_path, exist_ok=True)
        db = fastworkflow.state_paths.observability_db(workflow_path)
        obs.ObservabilityStore(db)  # create it
        os.chmod(db, 0o400)  # unwritable -> schema ensure fails
        try:
            assert obs.get_observability_sink(workflow_path) is None
        finally:
            os.chmod(db, 0o600)

    def test_close_drains_pending_writes(self, db_path):
        sink = obs.SQLiteTraceSink(db_path)
        for i in range(20):
            sink.emit_span(
                tracing.Span(
                    span_id=f"c{i}", trace_id="t", name="fw.agent.tool_call",
                    start_ns=i, status="ok",
                )
            )
        sink.close()
        assert len(_rows(db_path, "SELECT * FROM spans")) == 20
        # Emissions after close are dropped silently
        sink.emit_span(
            tracing.Span(span_id="late", trace_id="t", name="fw.turn", start_ns=1, status="open")
        )


# ----------------------------------------------------------------------
# Maintenance [R12] and erasure [R21]
# ----------------------------------------------------------------------


class TestMaintenance:
    def test_prune_deletes_old_spans_keeps_turns(self, db_path, sink):
        old_ns = int((time.time() - 90 * 86_400) * 1_000_000_000)
        sink.emit_span(
            tracing.Span(span_id="old", trace_id="t-old", name="fw.turn", start_ns=old_ns, status="ok")
        )
        sink.emit_span(
            tracing.Span(
                span_id="new", trace_id="t-new", name="fw.turn",
                start_ns=int(time.time() * 1e9), status="ok",
            )
        )
        assert sink.flush()

        deleted = sink.store.prune(retention_days=30)
        assert deleted["spans"] == 1
        remaining = _rows(db_path, "SELECT span_id FROM spans")
        assert [r["span_id"] for r in remaining] == ["new"]

    def test_forget_channel_erases_across_tables(
        self, initialized_fastworkflow, todo_workflow_path, monkeypatch, db_path, sink
    ):
        ctx, _wf = _make_assistant_ctx(todo_workflow_path, monkeypatch, sink)
        conv = sink.store.mint_conversation_id("chan-erase")
        ctx.bind_observability_identity(channel_id="chan-erase", conversation_id=conv)
        ctx.process_turn("add_todo x")

        ctx2, _wf2 = _make_assistant_ctx(todo_workflow_path, monkeypatch, sink)
        ctx2.bind_observability_identity(channel_id="chan-keep")
        ctx2.process_turn("list_todos")
        assert sink.flush()

        deleted = sink.store.forget_channel("chan-erase")
        assert deleted["turns"] == 1
        assert deleted["conversations"] == 1
        assert deleted["spans"] > 0

        assert _rows(db_path, "SELECT * FROM turns WHERE channel_id='chan-erase'") == []
        assert len(_rows(db_path, "SELECT * FROM turns WHERE channel_id='chan-keep'")) == 1
        assert _rows(db_path, "SELECT * FROM spans WHERE channel_id='chan-erase'") == []


# ----------------------------------------------------------------------
# Factory / FW_OBSERVABILITY gating [R4]
# ----------------------------------------------------------------------


class TestFactory:
    def test_gating(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "root"))
        workflow_path = str(tmp_path / "wf")
        os.makedirs(workflow_path, exist_ok=True)

        monkeypatch.setenv("FW_OBSERVABILITY", "0")
        assert obs.get_observability_sink(workflow_path) is None

        monkeypatch.delenv("FW_OBSERVABILITY", raising=False)
        # Entry points default ON; embedders default OFF
        assert obs.get_observability_sink(workflow_path, entry_point=False) is None
        sink = obs.get_observability_sink(workflow_path)
        try:
            assert sink is not None
            # Cached: same sink per DB path
            assert obs.get_observability_sink(workflow_path) is sink
        finally:
            if sink is not None:
                sink.close()

    def test_train_runs_roundtrip(self, db_path):
        store = obs.ObservabilityStore(db_path)
        store.record_train_run(
            "run-1", "fp", "2026-08-25T00:00:00Z", "2026-08-25T00:10:00Z",
            {"contexts": {"global": {"f1": 0.97}}},
        )
        runs = store.list_train_runs()
        assert len(runs) == 1
        assert json.loads(runs[0]["metrics_json"])["contexts"]["global"]["f1"] == 0.97


# ----------------------------------------------------------------------
# Ruling C2: conversation ids come from a per-channel counter that never
# decreases, so no erasure or prune can cause an id to be reused; and a mint
# that cannot run degrades instead of failing the caller.
#
# The Phase-A `legacy_floor` half of C2 is gone with the legacy store: it
# existed to stop a fresh observability DB re-issuing an id that already named
# one of the channel's per-channel-DB conversations, which mattered only while
# BOTH stores were written. Nothing reads those files now.
# ----------------------------------------------------------------------


class TestMinting:
    def _runtime(self, sink, channel_id):
        bound: dict = {}

        class _Ctx:
            trace_sink = sink

            def bind_observability_identity(self, **kwargs):
                bound.update(kwargs)

        runtime = SimpleNamespace(
            execution_context=_Ctx(),
            channel_id=channel_id,
        )
        return runtime, bound

    def test_reserve_conversation_id_binds_the_minted_id(self, db_path):
        from fastworkflow.run_fastapi_mcp.utils import reserve_conversation_id

        sink = obs.SQLiteTraceSink(db_path)
        try:
            runtime, bound = self._runtime(sink, "chanX")
            assert reserve_conversation_id(runtime) == 1
            assert bound == {"conversation_id": 1}
            assert reserve_conversation_id(runtime) == 2
        finally:
            sink.close()

    def test_an_id_is_never_reused_after_the_channel_is_forgotten(self, db_path):
        """Erasure must not roll the counter back (ruling C2).

        A MAX-derived mint would hand the next conversation an id that names a
        deleted one, so anything still holding the old id — a checkpoint, a
        client's history list — would silently point at the new conversation.
        """
        from fastworkflow.run_fastapi_mcp.utils import reserve_conversation_id

        sink = obs.SQLiteTraceSink(db_path)
        try:
            runtime, _bound = self._runtime(sink, "chanZ")
            first = reserve_conversation_id(runtime)
            second = reserve_conversation_id(runtime)
            sink.store.forget_channel("chanZ")
            after_erasure = reserve_conversation_id(runtime)
        finally:
            sink.close()

        assert (first, second) == (1, 2)
        assert after_erasure > second, (
            f"minting reused id {after_erasure} after the channel was forgotten"
        )

    def test_reserve_degrades_to_zero_when_the_mint_fails(self, db_path, monkeypatch):
        """A wedged DB must not fail /initialize or a turn.

        Zero is the same value a never-reserved channel carries, so every caller
        already handles it as "no active conversation".
        """
        from fastworkflow.run_fastapi_mcp.utils import reserve_conversation_id

        sink = obs.SQLiteTraceSink(db_path)
        try:
            def _wedged(*args, **kwargs):
                raise sqlite3.OperationalError("database is locked")

            monkeypatch.setattr(sink.store, "mint_conversation_id", _wedged)
            runtime, bound = self._runtime(sink, "chanY")
            conv_id = reserve_conversation_id(runtime)  # must not raise
        finally:
            sink.close()

        assert conv_id == 0
        assert bound == {}, "a failed mint bound an id onto the context anyway"


# ----------------------------------------------------------------------
# Gate 2 (§2.4, rulings I1/I6/C8/C9): sync-first turn records
# ----------------------------------------------------------------------


class TestSyncFirstTurnRecords:
    def _turn_result(self, summary="a turn", conversation_id=1, status=None):
        turn_output = fastworkflow.TurnOutput(
            turn_key=fastworkflow.mint_turn_key(),
            status=status or TurnStatus.COMPLETED,
            answer="ok",
        )
        return fastworkflow.TurnResult(
            turn_output=turn_output,
            channel_id="c",
            conversation_id=conversation_id,
            user_message="msg",
            conversation_summary=summary,
        )

    def test_a_healthy_emit_is_durable_before_it_returns(self, db_path, sink):
        turn_result = self._turn_result()
        assert sink.emit_turn_record(turn_result) is True
        # Deliberately NO flush: the ack promises the row is already there.
        assert obs.ObservabilityStore(db_path).get_turn(
            turn_result.turn_output.turn_key
        ) is not None
        assert sink.pending_retry_depth() == 0

    def test_an_open_breaker_degrades_to_the_queue_and_reports_it(self, db_path, sink):
        sink._sync_breaker_until = time.monotonic() + 300
        turn_result = self._turn_result()
        assert sink.emit_turn_record(turn_result) is False
        assert sink.pending_retry_depth() == 1
        # The row is not durable yet, which is exactly what the ack said.
        assert obs.ObservabilityStore(db_path).get_turn(
            turn_result.turn_output.turn_key
        ) is None
        assert sink.flush()
        assert obs.ObservabilityStore(db_path).get_turn(
            turn_result.turn_output.turn_key
        ) is not None

    def test_a_degraded_record_keeps_its_chronological_ordinal(self, db_path, sink):
        """Ruling I6: the ordinal is reserved synchronously before the enqueue.

        Without that, a turn written while the DB was briefly wedged would sort
        after turns that happened later.
        """
        assert sink.emit_turn_record(self._turn_result("first")) is True
        sink._sync_breaker_until = time.monotonic() + 300
        assert sink.emit_turn_record(self._turn_result("second")) is False
        sink._sync_breaker_until = 0.0
        assert sink.emit_turn_record(self._turn_result("third")) is True
        assert sink.flush()

        window = obs.ObservabilityStore(db_path).get_memory_window("c", 1, 10)
        assert [entry["conversation summary"] for entry in window] == [
            "first",
            "second",
            "third",
        ]

    def test_the_pending_ring_is_bounded(self, db_path, sink):
        sink._sync_breaker_until = time.monotonic() + 300
        for i in range(obs._PENDING_RETRY_MAX + 10):
            sink.emit_turn_record(self._turn_result(f"turn-{i}"))
        assert sink.pending_retry_depth() == obs._PENDING_RETRY_MAX
        health = sink._health
        assert health["records_dropped"] >= 10
        assert health["sync_fallbacks"] >= obs._PENDING_RETRY_MAX

    def test_the_breaker_rearms_only_after_a_successful_probe(self, db_path, sink):
        sink._trip_sync_breaker(RuntimeError("wedged"))
        assert sink._sync_available() is False

        # Cooldown elapsed, but the breaker stays shut until a probe succeeds.
        sink._sync_breaker_until = time.monotonic() - 1
        sink._maybe_rearm_sync_breaker()
        assert sink._sync_available() is True
        assert sink._health["sync_breaker_open"] is False
        assert sink.emit_turn_record(self._turn_result()) is True

    def test_writer_health_records_the_sync_path(self, db_path, sink):
        sink.emit_turn_record(self._turn_result())
        assert sink.flush()
        health = obs.ObservabilityStore(db_path).writer_health()
        assert health is not None
        assert health["sync_writes"] >= 1
        assert health["sync_write_ms_max"] >= 0
        assert health["sync_breaker_open"] is False

    def test_an_awaiting_user_emission_is_also_sync(self, db_path, sink):
        """Ruling I6: awaiting_user and terminal take the SAME path.

        Mixing them was what let one logical turn split across the sync and
        queued paths and produce spurious refused-terminal-write noise.
        """
        suspended = self._turn_result(
            summary=None, status=TurnStatus.AWAITING_USER
        )
        assert sink.emit_turn_record(suspended) is True
        row = obs.ObservabilityStore(db_path).get_turn(
            suspended.turn_output.turn_key
        )
        assert row is not None and row["status"] == "awaiting_user"
        # A suspended row is not a pending-retry obligation: it is not terminal.
        assert sink.pending_retry_depth() == 0
