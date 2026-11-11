"""The policy's model readings (fastworkflow.policy_intent), with DSPy's DummyLM.

The readers' contract matters more than the model: requests keep their ids and
are never dropped by omission, operations outside the workflow's vocabulary are
ignored, every failure leaves the state as it was, and no call is made when
there is nothing to read.
"""

from __future__ import annotations

import json

from dspy.utils import DummyLM

from fastworkflow.policy_decision import CommandRecord
from fastworkflow.policy_intent import (
    ASSISTANT,
    USER,
    IntentExtractor,
    QuestionJudge,
    TrackedRequest,
    apply_requests,
    describe,
    describe_tracked,
    for_rules,
    parse_described,
    question_history,
)

OPERATIONS = {"close": "close a ticket", "reopen": "reopen a closed ticket"}


def _reply(requests, consents=False) -> dict:
    return {"requests": json.dumps(requests), "consents_to_agent_proposal": str(consents)}


def test_a_new_request_is_read_in_the_workflows_vocabulary_only():
    lm = DummyLM([_reply([
        {"id": "", "op": "close", "scope": "all", "status": "firm", "detail": "my tickets"},
        {"id": "", "op": "delete_account", "status": "firm"},
    ])])
    reader = IntentExtractor(OPERATIONS, lm=lm)
    tracked, consents = reader.read([(USER, "close all my tickets")])
    assert [(r.id, r.op, r.scope, r.status) for r in tracked] == [("R1", "close", "all", "firm")]
    assert consents is False and reader.calls == 1


def test_a_request_keeps_its_id_and_is_never_dropped_by_omission():
    current = (TrackedRequest("R1", "close", "all", "firm", "my tickets"),
               TrackedRequest("R2", "reopen", "specific", "fallback", "T-1"))
    updated = apply_requests(current, [
        {"id": "R2", "op": "reopen", "scope": "specific", "status": "withdrawn", "detail": "T-1"},
        {"id": "R9", "op": "close", "scope": "specific", "status": "odd", "detail": "T-3"},
    ], OPERATIONS)
    assert [(r.id, r.status) for r in updated] == [("R1", "firm"), ("R2", "withdrawn"),
                                                   ("R3", "firm")]


def test_only_firm_requests_and_fallbacks_reach_the_rules():
    tracked = (TrackedRequest("R1", "close", "all", "firm"),
               TrackedRequest("R2", "reopen", "specific", "fallback"),
               TrackedRequest("R3", "close", "specific", "asks_if_possible"),
               TrackedRequest("R4", "reopen", "specific", "withdrawn"))
    intent = for_rules(tracked)
    assert [(r.op, r.conditional) for r in intent] == [("close", False), ("reopen", True)]
    assert parse_described(describe(intent)) == tuple(
        r if r.scope == "all" else r.__class__(r.op, "unspecified", r.conditional, r.detail)
        for r in intent)
    assert describe_tracked(tracked).startswith("R1:close[firm][all]")


def test_consent_is_read_against_the_assistants_proposal():
    lm = DummyLM([_reply([], consents=True)])
    reader = IntentExtractor(OPERATIONS, lm=lm)
    _, consents = reader.read([(ASSISTANT, "Shall I close T-1?"), (USER, "yes")])
    assert consents is True


def test_nothing_is_read_without_a_user_message_or_a_vocabulary():
    reader = IntentExtractor(OPERATIONS, lm=DummyLM([]))
    assert reader.read([(ASSISTANT, "Hello")]) == ((), False)
    assert IntentExtractor({}, lm=DummyLM([])).read([(USER, "close it")]) == ((), False)
    assert reader.calls == 0


def test_a_failed_reading_keeps_the_list_as_it_was():
    current = (TrackedRequest("R1", "close", "all", "firm"),)
    reader = IntentExtractor(OPERATIONS, lm=DummyLM([]))  # no answers: the call fails
    assert reader.read([(USER, "and reopen T-2")], current) == (current, False)
    assert reader.failures == 1


def test_a_message_is_read_once():
    reader = IntentExtractor(OPERATIONS, lm=DummyLM([_reply([])]))
    for _ in range(2):
        reader.read([(USER, "close T-1")])
    assert reader.calls == 1


def test_the_judge_is_not_called_with_nothing_to_judge_against():
    judge = QuestionJudge(lm=DummyLM([]))
    assert judge.judge("Which ticket?", earlier=[], lookups="") is None
    assert judge.calls == 0


def test_the_judge_reads_a_repeat():
    judge = QuestionJudge(lm=DummyLM([{
        "reasoning": "Same question, nothing happened since.",
        "repeats_without_new_information": "True",
        "asks_for_what_agent_can_look_up": "False",
        "reason": "asked already",
    }]))
    reading = judge.judge("Which ticket?", earlier=[{"question": "Which ticket?"}], lookups="")
    assert reading.repeats_without_new_information and reading.reason == "asked already"
    assert not reading.asks_for_what_agent_can_look_up


def test_question_history_records_what_happened_after_each_question():
    history = question_history([
        CommandRecord("@user", {}, "close my ticket"),
        CommandRecord("ask_user", {"clarification_request": "Which one?"}, "the printer one"),
        CommandRecord("find_ticket", {"q": "printer"}, "not found", success=False),
        CommandRecord("@user", {}, "it is T-7"),
    ])
    assert len(history) == 1
    assert history[0]["user_answer"] == "the printer one"
    assert history[0]["assistant_did_since"] == [
        "find_ticket(q=printer) -> FAILED: not found", "user said: it is T-7"]
