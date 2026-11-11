"""Replaying recorded runs through a policy table (fastworkflow.policy_replay).

A real ObservabilityStore in tmp_path, seeded with one recorded attempt the way
a live turn writes it: the turn row (labelled with its experiment, task and
attempt) and its spans - fw.turn, the agent's steps, the tool call and the
command execution under it, and an fw.ask_user carrying the user's reply.
"""

from __future__ import annotations

import json

from fastworkflow import tracing
from fastworkflow.observability import store as obs
from fastworkflow.policy_decision import (
    ContractFacts,
    DecisionPhase,
    DecisionTable,
    PolicyOutcome,
    PolicyRow,
)
from fastworkflow.policy_replay import format_report, replay_attempt

EXPERIMENT, TASK = "exp-1", "t1"
FACTS = ContractFacts(write_commands=frozenset({"close_ticket"}))


def _never_listed(inputs) -> bool:
    return not inputs.ran("list_tickets")


TABLE = DecisionTable("replay-test/1", [
    PolicyRow("READ-FIRST", frozenset({DecisionPhase.BEFORE_SIDE_EFFECT}),
              PolicyOutcome.REWRITE, "write before listing",
              lambda i: FACTS.is_write(i.command_name) and _never_listed(i),
              "List the tickets first."),
    PolicyRow("CONFIRM", frozenset({DecisionPhase.BEFORE_SIDE_EFFECT}),
              PolicyOutcome.REWRITE, "write without a yes",
              lambda i: FACTS.is_write(i.command_name)
              and not (i.last_user_utterance() and i.last_user_utterance().name == "ask_user"),
              "Ask the user to confirm first."),
])


def _turn_row(turn_key: str, ordinal: int, message: str, answer: str, attempt: int) -> dict:
    return {
        "turn_key": turn_key, "channel_id": "c", "conversation_id": 1, "ordinal": ordinal,
        "user_message": message, "refined_user_message": None,
        "entry_workflow_name": "workflow", "entry_context": "context",
        "status": "completed", "success": 1, "failure_reason": None, "answer": answer,
        "conversation_summary": None, "conversation_traces": None,
        "started_at": "2026-10-01T00:00:00+00:00", "completed_at": "2026-10-01T00:01:00+00:00",
        "suspended_ms": 0, "continuation_of": None, "record_version": 1,
        "record_json": json.dumps({"turn_output": {"turn_key": turn_key}}),
        "experiment_id": EXPERIMENT, "task_id": TASK, "attempt": attempt,
    }


class _Spans:
    def __init__(self, turn_key: str, attempt: int):
        self.turn_key, self.attempt, self.spans, self.clock = turn_key, attempt, [], 0

    def add(self, name, attributes, *, parent=None, command_name=None) -> str:
        self.clock += 10
        span_id = f"{self.turn_key}-{len(self.spans)}"
        self.spans.append(tracing.Span(
            span_id=span_id, trace_id=self.turn_key, name=name, parent_span_id=parent,
            channel_id="c", command_name=command_name, start_ns=self.clock,
            end_ns=self.clock + 5, status=tracing.STATUS_OK, attributes=attributes,
            experiment_id=EXPERIMENT, task_id=TASK, attempt=self.attempt,
        ))
        return span_id

    def command(self, index: int, text: str, name: str, params: dict, response: str,
                *, blocked_by: str | None = None) -> None:
        step = self.add(tracing.SPAN_AGENT_STEP, {
            "step_index": index, "tool_name": "execute_workflow_query",
            "tool_args": {"command": text}, "observation": response,
            **({"policy_source": blocked_by} if blocked_by else {}),
        })
        if blocked_by:
            return
        call = self.add(tracing.SPAN_AGENT_TOOL_CALL,
                        {"raw_command": text, "response_text": response, "success": True},
                        parent=step)
        self.add(tracing.SPAN_COMMAND_EXECUTE,
                 {"raw_command": text, "parameters": params, "response_text": response},
                 parent=call, command_name=name)


def _store_with(tmp_path, turns: list[tuple[dict, _Spans]]):
    store = obs.ObservabilityStore(str(tmp_path / "observability.sqlite3"))
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        for row, spans in turns:
            assert store.upsert_turn_row(conn, row, [], obs.Redactor())
            store.upsert_span_rows(conn, spans.spans, obs.Redactor())
        conn.commit()
    return store


def _recorded_attempt(attempt: int, *, list_first: bool) -> tuple[dict, _Spans]:
    key = f"turn-{attempt}"
    spans = _Spans(key, attempt)
    spans.add(tracing.SPAN_TURN, {"user_message": "close T-1"})
    i = 0
    if list_first:
        spans.command(i, "list_tickets", "list_tickets", {}, "T-1, T-2")
        i += 1
    spans.add(tracing.SPAN_AGENT_STEP, {"step_index": i, "tool_name": "ask_user",
                                        "tool_args": {"clarification_request": "Close T-1?"},
                                        "clarification": "Close T-1?"})
    spans.add(tracing.SPAN_ASK_USER, {"agent_query": "Close T-1?", "user_response": "yes"})
    spans.command(i + 1, "close_ticket <id>T-1</id>", "close_ticket", {"id": "T-1"}, "closed")
    spans.add(tracing.SPAN_AGENT_STEP, {"step_index": i + 2, "tool_name": "finish"})
    return _turn_row(key, 1, "close T-1", "T-1 is closed.", attempt), spans


def test_a_good_attempt_replays_silently_and_a_bad_one_fires_where_it_went_wrong(tmp_path):
    store = _store_with(tmp_path, [_recorded_attempt(1, list_first=True),
                                   _recorded_attempt(2, list_first=False)])
    good = replay_attempt(store, EXPERIMENT, TASK, 1, TABLE, FACTS, passed=True)
    bad = replay_attempt(store, EXPERIMENT, TASK, 2, TABLE, FACTS, passed=False)
    assert good.steps == [] and good.error is None
    assert [(s.step_index, s.replayed, s.mark) for s in bad.steps] == [(1, "READ-FIRST", "ADDED")]
    assert bad.steps[0].rewrite == "List the tickets first."


def test_the_replay_uses_the_recorded_reply_to_the_question(tmp_path):
    """The confirmation the user gave (fw.ask_user) is what lets the write through."""
    row, spans = _recorded_attempt(1, list_first=True)
    spans.spans = [s for s in spans.spans if s.name != tracing.SPAN_ASK_USER]
    store = _store_with(tmp_path, [(row, spans)])
    replayed = replay_attempt(store, EXPERIMENT, TASK, 1, TABLE, FACTS)
    assert [s.replayed for s in replayed.steps] == ["CONFIRM"]


def test_a_step_blocked_in_the_recording_did_not_run(tmp_path):
    key = "turn-9"
    spans = _Spans(key, 9)
    spans.add(tracing.SPAN_TURN, {"user_message": "close T-1"})
    spans.command(0, "close_ticket <id>T-1</id>", "close_ticket", {"id": "T-1"},
                  "List the tickets first.", blocked_by="READ-FIRST")
    spans.command(1, "list_tickets", "list_tickets", {}, "T-1")
    store = _store_with(tmp_path, [(_turn_row(key, 1, "close T-1", "", 9), spans)])
    replayed = replay_attempt(store, EXPERIMENT, TASK, 9, TABLE, FACTS)
    assert [(s.recorded, s.replayed, s.mark) for s in replayed.steps] == [
        ("READ-FIRST", "READ-FIRST", "same")]


def test_an_attempt_with_no_turns_is_reported_not_guessed(tmp_path):
    store = _store_with(tmp_path, [])
    replayed = replay_attempt(store, EXPERIMENT, TASK, 5, TABLE, FACTS)
    assert replayed.error == "no turns recorded" and replayed.steps == []


def test_the_report_names_each_intervention_and_the_totals(tmp_path):
    from fastworkflow.policy_replay import ReplayReport

    store = _store_with(tmp_path, [_recorded_attempt(2, list_first=False)])
    report = ReplayReport(EXPERIMENT, TABLE.version, live=False, attempts=[
        replay_attempt(store, EXPERIMENT, TASK, 2, TABLE, FACTS, passed=False)])
    text = format_report(report)
    assert "=== task t1 attempt 2 (failed)" in text
    assert "[ADDED] close_ticket <id>T-1</id>" in text
    assert "on passing attempts: 0, on failing attempts: 1" in text


def test_distillation_does_not_count_a_blocked_command_as_an_action(tmp_path):
    """A blocked command still has its dispatch span (fw.command.execute, failed),
    but it never ran: the trace distillation compares leaves it out."""
    from fastworkflow import distillation_trace as dtrace

    key = "turn-7"
    spans = _Spans(key, 7)
    spans.add(tracing.SPAN_TURN, {"user_message": "close T-1"})
    step = spans.add(tracing.SPAN_AGENT_STEP, {
        "step_index": 0, "tool_name": "execute_workflow_query",
        "tool_args": {"command": "close_ticket"}, "policy_source": "READ-FIRST",
        "observation": "List the tickets first."})
    call = spans.add(tracing.SPAN_AGENT_TOOL_CALL, {"raw_command": "close_ticket"}, parent=step)
    spans.add(tracing.SPAN_COMMAND_EXECUTE,
              {"raw_command": "close_ticket", "parameters": {"id": "T-1"},
               "response_text": "List the tickets first.", "success": False},
              parent=call, command_name="close_ticket")
    spans.command(1, "list_tickets", "list_tickets", {}, "T-1")
    store = _store_with(tmp_path, [(_turn_row(key, 1, "close T-1", "", 7), spans)])
    attempt = dtrace.read_attempt(store, EXPERIMENT, TASK, 7)
    assert [a["command_name"] for a in attempt.all_actions()] == ["list_tickets"]
