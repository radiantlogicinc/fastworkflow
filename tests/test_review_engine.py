"""The review engine (fix-3rsq.plan-review.1, stage 1): the state machine, fingerprints and cascading
invalidation, persistence and redaction, deterministic typed edits, the suspend/resume primitive on a
real turn, and parity of the action kind with the approval binding (ADR-EXP-025).

Integration tests. The proposals here are small typed test specs; the real round-1 and round-2 proposals
are stage 2. No test calls a model: the model-based edit interpreter is never registered, and a guard
fails any test that reaches it. The approval fixtures (the annotated library copy, the real agent) are
the ones tests/test_approval_binding.py uses.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from typing import Optional

import pytest
from pydantic import BaseModel, Field

from fastworkflow import approval_binding as ab
from fastworkflow import review_engine as re_
from fastworkflow.review_engine import (
    AWAITING, CONFIRMED, DECLINED, EXECUTED, EXPIRED, INVALIDATED, KIND_ACTION, KIND_PLAN,
    KIND_SUB_TASKS, REVISED, Change, Drop, EditParseError, IllegalTransition,
    Proposal, ReviewError, ReviewRecord, ReviewStore, Verdict, apply_edits, audit_attributes,
    fingerprint, link, open_review, parse_edits, propagate, proposal_of, register_proposal,
    reply_policy, reply_to_proposal, rendering_hash, set_model_interpreter, stale_upstream,
)
from fastworkflow.approval_binding import Surface
from fastworkflow.turn import TurnStatus
from fastworkflow import tracing
from tests.test_approval_binding import annotated_library, no_live_model, no_model_call_for_parameters, todo  # noqa: F401,F811
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

SECRET = "SECRET-VALUE-1f3a"


class Task(BaseModel):
    goal: str
    count: int = 1
    tags: list[str] = Field(default_factory=list)


@register_proposal
class TaskList(Proposal):
    kind = KIND_SUB_TASKS
    spec = "review_test_tasks"
    title = "Sub-tasks"
    item_template = "{index}. {goal} (x{count})"
    min_items = 1
    items: list[Task] = Field(default_factory=list)


class Goal(BaseModel):
    goal: str
    steps: list[str] = Field(default_factory=list)


@register_proposal
class GoalPlan(Proposal):
    kind = KIND_PLAN
    spec = "review_test_plan"
    title = "Plan"
    item_template = "{index}. {goal}: {steps}"
    min_items = 1
    items: list[Goal] = Field(default_factory=list)


def _tasks(*goals: str) -> TaskList:
    return TaskList(items=[Task(goal=goal) for goal in goals])


def _store(surface: Optional[Surface] = None) -> ReviewStore:
    store = ReviewStore(channel="c1", task="t1")
    store.surface = surface or Surface()
    return store


def _accept(store, record):
    """Reply accept on an open proposal record (through the real reply path)."""
    return reply_to_proposal(None, store, record, "accept")


def _confirmed_tasks(store, *goals: str) -> ReviewRecord:
    record = open_review(None, store, _tasks(*goals), surface=store.surface)
    assert _accept(store, record).done
    return record


def _force_store_state(record: ReviewRecord, state: str) -> None:
    """Move a record's state directly, for a graph the single-review-per-session rule cannot build in order.
    Used only where a test needs an upstream state that the normal flow reaches by another path."""
    record.state = state


# ----------------------------------------------------------------------
# The state machine
# ----------------------------------------------------------------------


def test_the_transition_table_allows_the_documented_moves_and_nothing_out_of_a_terminal_state():
    assert re_.ACTIVE_STATES == frozenset({AWAITING, REVISED})
    assert re_.TERMINAL_STATES == frozenset({DECLINED, EXPIRED, INVALIDATED, EXECUTED})
    store = _store()
    record = open_review(None, store, _tasks("a"), surface=store.surface)
    assert record.state == AWAITING and record.kind == KIND_SUB_TASKS
    re_.transition(None, store, record, REVISED, reason="t", surface=store.surface)
    re_.transition(None, store, record, REVISED, reason="t", surface=store.surface)   # a second revision
    re_.transition(None, store, record, CONFIRMED, reason="t", surface=store.surface)
    for illegal in (DECLINED, EXPIRED, AWAITING, REVISED):      # a confirmed review is not declined or reopened
        with pytest.raises(IllegalTransition):
            re_.transition(None, store, record, illegal, reason="t", surface=store.surface)
    assert record.state == CONFIRMED
    re_.transition(None, store, record, INVALIDATED, reason="withdrawn", surface=store.surface)   # legal
    for terminal in (DECLINED, EXPIRED, INVALIDATED, EXECUTED, AWAITING, REVISED):
        with pytest.raises(IllegalTransition):
            re_.transition(None, store, record, terminal, reason="t", surface=store.surface)


@pytest.mark.parametrize("start, end", [
    (AWAITING, EXECUTED),          # an action only executes after it is confirmed
    (REVISED, EXECUTED),
    (DECLINED, AWAITING),
    (EXPIRED, REVISED),
    (INVALIDATED, CONFIRMED),
    (EXECUTED, CONFIRMED),
])
def test_an_illegal_transition_is_refused_and_leaves_the_record_as_it_was(start, end):
    store = _store()
    record = open_review(None, store, _tasks("a"), surface=store.surface)
    record.state = start
    with pytest.raises(IllegalTransition):
        re_.transition(None, store, record, end, reason="t", surface=store.surface)
    assert record.state == start


def test_a_reply_moves_the_record_awaiting_to_revised_to_confirmed_and_the_store_releases_it():
    store = _store()
    record = open_review(None, store, _tasks("book a flight", "tell Ann"), surface=store.surface)
    reply = reply_to_proposal(None, store, record, "change 2: count=3")
    assert not reply.done and record.state == REVISED and record.revision == 1
    assert "2. tell Ann (x3)" in reply.text and store.active() is record
    done = reply_to_proposal(None, store, record, "accept")
    assert done.done and done.proposal is not None
    assert record.state == CONFIRMED and store.active() is None
    assert [item.count for item in done.proposal.items] == [1, 3]


def test_only_one_review_is_open_on_a_session_at_a_time():
    store = _store()
    open_review(None, store, _tasks("a"), surface=store.surface)
    with pytest.raises(ReviewError):
        open_review(None, store, _tasks("b"), surface=store.surface)


def test_a_declined_proposal_is_terminal_and_says_so():
    store = _store()
    record = open_review(None, store, _tasks("a"), surface=store.surface)
    reply = reply_to_proposal(None, store, record, "no")
    assert reply.done and reply.proposal is None and record.state == DECLINED
    assert "not accepted" in reply.text
    assert store.active() is None


def test_a_proposal_suspends_the_agent_call_on_its_prompt_when_asked_to():
    from fastworkflow.utils.react import AskUserSuspend

    store = _store()
    with pytest.raises(AskUserSuspend) as caught:
        open_review(None, store, _tasks("a", "b"), surface=store.surface, suspend=True)
    record = store.active()
    assert record is not None and record.suspended_agent and record.state == AWAITING
    assert str(caught.value) == record.rendering == re_.prompt_for(_tasks("a", "b"))


# ----------------------------------------------------------------------
# Fingerprints, rendering and cascading invalidation
# ----------------------------------------------------------------------


def test_the_fingerprint_is_the_same_across_processes_and_key_order():
    proposal = _tasks("alpha", "beta")
    here = proposal.fingerprint()
    code = ("import json,sys\n"
            "from fastworkflow import review_engine as r\n"
            "print(r.fingerprint(json.loads(sys.argv[1])))")
    binding = json.dumps(proposal.binding())
    other = subprocess.run([sys.executable, "-I", "-c", code, binding], capture_output=True, text=True, check=True)
    assert other.stdout.strip() == here
    reordered = {key: proposal.binding()[key] for key in reversed(list(proposal.binding()))}
    assert fingerprint(reordered) == here


def test_any_change_to_an_item_changes_the_fingerprint_and_the_rendering_hash():
    base = _tasks("a", "b")
    changed = apply_edits(base, [Change(index=2, field="count", value="2")])
    assert changed.fingerprint() != base.fingerprint()
    assert rendering_hash(changed.rendering()) != rendering_hash(base.rendering())
    assert apply_edits(base, [Change(index=2, field="count", value="1")]).fingerprint() == base.fingerprint()


def test_a_proposal_renders_from_its_template_and_a_missing_field_is_named_not_an_error():
    class Odd(Proposal):
        kind = KIND_SUB_TASKS
        spec = "review_test_odd"
        item_template = "{index}. {goal} / {nosuch}"
        items: list[Task] = Field(default_factory=list)

    assert "1. alpha / <nosuch?>" in Odd(items=[Task(goal="alpha")]).rendering()


def test_invalidating_an_upstream_invalidates_its_dependents_transitively():
    store = _store()
    first = _confirmed_tasks(store, "research")
    plan = open_review(None, store, GoalPlan(items=[Goal(goal="book")]), surface=store.surface, upstream=[first])
    assert plan.upstream == {first.id: first.fingerprint}
    assert _accept(store, plan).done and plan.state == CONFIRMED
    # Chain: first <- plan <- follow-up (open). Invalidate the root: both dependents go.
    follow = open_review(None, store, _tasks("follow"), surface=store.surface, upstream=[plan])
    re_.transition(None, store, first, INVALIDATED, reason="cancelled", surface=store.surface)
    invalidated = propagate(None, store, first, surface=store.surface)
    assert {r.id for r in invalidated} == {plan.id, follow.id}
    assert plan.state == INVALIDATED and follow.state == INVALIDATED
    assert store.active() is None
    assert stale_upstream(store, follow) is not None


def test_withdrawing_a_confirmed_upstream_invalidates_an_open_dependent_with_its_reason():
    store = _store()
    upstream = _confirmed_tasks(store, "x")
    follow = open_review(None, store, _tasks("y"), surface=store.surface, upstream=[upstream])
    re_.transition(None, store, upstream, INVALIDATED, reason="withdrawn", surface=store.surface)
    assert [r.id for r in propagate(None, store, upstream, surface=store.surface)] == [follow.id]
    assert follow.state == INVALIDATED and follow.turns_seen == 0
    assert stale_upstream(store, follow) == "an upstream review is invalidated"


def test_a_dependent_on_an_unconfirmed_upstream_cannot_be_linked():
    store = _store()
    upstream = open_review(None, store, _tasks("a"), surface=store.surface)
    with pytest.raises(ReviewError):
        link(ReviewRecord(id="x", token_hash="", command="c", annotation=None, raw_command="", binding={},
                          fingerprint="", rendering="", rendering_hash="", required_identity="channel",
                          created_turn=0, turns_seen=0, ttl_turns=2, ttl_seconds=900, created_at=time.time(),
                          state=AWAITING, origin="agent"), [upstream])


def test_a_confirmed_dependent_is_invalidated_when_its_upstream_is_withdrawn_after_confirmation():
    store = _store()
    upstream = _confirmed_tasks(store, "x")
    plan = open_review(None, store, GoalPlan(items=[Goal(goal="g")]), surface=store.surface, upstream=[upstream])
    assert _accept(store, plan).done and plan.state == CONFIRMED
    re_.transition(None, store, upstream, INVALIDATED, reason="withdrawn", surface=store.surface)
    propagate(None, store, upstream, surface=store.surface)
    assert plan.state == INVALIDATED


# ----------------------------------------------------------------------
# Persistence and redaction
# ----------------------------------------------------------------------


def test_a_proposal_record_round_trips_through_the_session_state_with_typed_items():
    store = _store()
    record = open_review(None, store, _tasks("alpha", "beta"), surface=store.surface)
    restored = ReviewStore()
    restored.load_state(store.to_state())
    again = restored.active()
    assert again is not None and again.kind == KIND_SUB_TASKS and again.fingerprint == record.fingerprint
    proposal = proposal_of(again)
    assert isinstance(proposal, TaskList) and [item.goal for item in proposal.items] == ["alpha", "beta"]
    assert fingerprint(again.binding) == again.fingerprint == record.fingerprint
    assert proposal.fingerprint() != record.fingerprint    # the record's fingerprint also covers channel and task


def test_a_record_saved_before_the_engine_still_loads_with_action_defaults():
    old = {"id": "r1", "token_hash": "h", "command": "Record/add_note", "annotation": "mandatory",
           "raw_command": "add_note", "binding": {}, "fingerprint": "f", "rendering": "text",
           "rendering_hash": "x", "required_identity": "channel", "created_turn": 1, "turns_seen": 0,
           "ttl_turns": 2, "ttl_seconds": 900, "created_at": time.time(), "state": "awaiting",
           "origin": "agent"}
    record = ReviewRecord.from_dict(old)
    assert record.kind == KIND_ACTION and record.revision == 0 and record.upstream == {}


def test_the_host_token_is_never_stored_and_never_in_the_persisted_session():
    host = Surface("host", verified=True, uid="alice")
    store = _store(host)
    record = open_review(None, store, _tasks("a"), surface=host)
    delivery = store.delivery
    assert delivery is not None and delivery["token"]
    persisted = json.dumps(store.to_state(), default=str)
    assert delivery["token"] not in persisted
    assert record.token_hash and record.token_hash != delivery["token"]
    assert re_.token_matches(delivery["token"], record.token_hash)


def test_audit_attributes_carry_hashes_names_and_counts_and_never_a_proposal_value():
    store = _store()
    record = open_review(None, store, _tasks(SECRET), surface=store.surface)
    attributes = audit_attributes(record, before="", after=AWAITING, reason="opened", surface=store.surface,
                                  mode="enforce", outcome="asked")
    assert SECRET not in json.dumps(attributes, default=str)
    assert attributes["approval_command"] == "review_test_tasks"     # the spec name tells a proposal row apart
    assert attributes["approval_fingerprint"] == record.fingerprint


def test_the_real_trace_sink_receives_one_audit_span_per_transition_and_no_raw_value(todo):
    # Spans are emitted inside an open turn, so the replies go through real turns on the session.
    store = todo.store
    store.records.clear()
    sink = todo.sink
    before = len(sink.named(tracing.SPAN_APPROVAL))
    record = open_review(todo.ctx, store, _tasks(SECRET), surface=store.surface)
    todo.ctx.process_turn("change 1: count=2")
    todo.ctx.process_turn("accept")
    spans = sink.named(tracing.SPAN_APPROVAL)[before:]
    assert [s.attributes["approval_state_after"] for s in spans] == [REVISED, CONFIRMED]
    assert all(s.attributes["approval_command"] == "review_test_tasks" for s in spans)
    assert all(SECRET not in json.dumps(s.attributes, default=str) for s in spans)
    assert record.state == CONFIRMED


# ----------------------------------------------------------------------
# Deterministic typed edits: parse, validate, apply
# ----------------------------------------------------------------------


def test_the_deterministic_reader_parses_changes_and_drops_separated_by_semicolons():
    assert parse_edits("change 2: goal=Book flights; drop 3") == [
        Change(index=2, field="goal", value="Book flights"), Drop(index=3)]
    assert parse_edits("CHANGE 1 count = 4\ndrop 2") == [Change(index=1, field="count", value="4"), Drop(index=2)]
    assert parse_edits("change 1: goal=a=b") == [Change(index=1, field="goal", value="a=b")]


@pytest.mark.parametrize("text", ["", "   ", "please make it better", "change x: goal=1", "drop", "change 1 goal"])
def test_text_the_reader_cannot_parse_is_refused_with_a_message(text):
    with pytest.raises(EditParseError) as caught:
        parse_edits(text)
    assert str(caught.value)


@pytest.mark.parametrize("text, expected", [
    ("change 9: goal=x", "There are only 2 steps."),
    ("change 1: nosuch=1", "Item 1 has no field 'nosuch'."),
    ("change 1: count=abc", "count of item 1 is not valid"),
    ("drop 1; drop 2", "The proposal needs at least 1 item(s)."),
    ("change 0: goal=x", "There are only 2 steps."),
])
def test_an_edit_that_does_not_validate_is_refused_with_a_message_and_nothing_changes(text, expected):
    store = _store()
    record = open_review(None, store, _tasks("a", "b"), surface=store.surface)
    before = record.fingerprint
    reply = reply_to_proposal(None, store, record, text)
    assert not reply.done and expected in reply.text
    assert record.state == AWAITING and record.fingerprint == before and record.revision == 0


def test_a_whole_reply_is_refused_when_one_of_its_edits_is_refused():
    store = _store()
    record = open_review(None, store, _tasks("a", "b"), surface=store.surface)
    reply = reply_to_proposal(None, store, record, "change 1: count=5; change 2: nosuch=1")
    assert not reply.done and record.revision == 0 and record.state == AWAITING


def test_list_fields_read_comma_separated_and_none_reads_as_no_value():
    tasks = _tasks("a")
    edited = apply_edits(tasks, [Change(index=1, field="tags", value="red, blue")])
    assert edited.items[0].tags == ["red", "blue"]
    class Opt(BaseModel):
        goal: str
        note: Optional[str] = None
    class OptList(Proposal):
        kind = KIND_SUB_TASKS
        spec = "review_test_opt"
        items: list[Opt] = Field(default_factory=list)
    opt = apply_edits(OptList(items=[Opt(goal="a", note="x")]), [Change(index=1, field="note", value="none")])
    assert opt.items[0].note is None


def test_the_original_proposal_is_never_modified_by_an_edit():
    tasks = _tasks("a")
    apply_edits(tasks, [Change(index=1, field="goal", value="changed")])
    assert tasks.items[0].goal == "a"


def test_a_free_text_reply_reprompts_when_no_reader_is_registered_and_never_calls_a_model():
    store = _store()
    record = open_review(None, store, _tasks("a"), surface=store.surface)
    reply = reply_to_proposal(None, store, record, "make the first one about trains")
    assert not reply.done and record.state == AWAITING and record.revision == 0
    assert "I cannot read" in reply.text and "accept" in reply.text


def test_the_policy_gives_graded_verdicts_for_proposal_replies():
    tasks = _tasks("a")
    policy = reply_policy(KIND_SUB_TASKS)
    assert policy.decide(tasks, "accept").verdict is Verdict.CONFIRM
    assert policy.decide(tasks, "yes").verdict is Verdict.CONFIRM
    assert policy.decide(tasks, "no").verdict is Verdict.DECLINE
    assert policy.decide(tasks, "drop 1").verdict is Verdict.REFUSE
    assert policy.decide(tasks, "change 1: count=2").verdict is Verdict.EDIT
    assert policy.decide(tasks, "hmm").verdict is Verdict.REPROMPT


def test_an_injected_reader_is_used_only_after_the_deterministic_reader_fails():
    calls = []

    def reader(text, proposal):
        calls.append(text)
        return [Change(index=1, field="goal", value="from the reader")]

    try:
        set_model_interpreter(reader)
        store = _store()
        record = open_review(None, store, _tasks("a"), surface=store.surface)
        reply = reply_to_proposal(None, store, record, "rename the first one")
        assert calls == ["rename the first one"] and not reply.done and record.revision == 1
        assert proposal_of(record).items[0].goal == "from the reader"
        calls.clear()
        reply_to_proposal(None, store, record, "change 1: goal=typed")
        assert calls == []                                    # the deterministic reader never reaches the model seam
    finally:
        set_model_interpreter(None)


def test_the_model_seam_is_not_registered_by_default():
    assert re_._model_interpreter is None


# ----------------------------------------------------------------------
# The real turn: the suspend and the reply through the agent's workflow (no agent is involved)
# ----------------------------------------------------------------------


def test_an_open_review_on_the_real_session_resumes_on_the_next_message(todo):
    store = todo.store
    store.records.clear()
    record = open_review(todo.ctx, store, _tasks("book a flight", "tell Ann"), surface=store.surface)
    out = todo.ctx.process_turn("change 2: count=3")
    assert out.status is TurnStatus.AWAITING_USER
    assert "2. tell Ann (x3)" in out.answer
    assert record.state == REVISED and record.revision == 1
    done = todo.ctx.process_turn("accept")
    assert record.state == CONFIRMED
    assert done.status is TurnStatus.COMPLETED
    assert store.active() is None


def test_a_proposal_reply_on_a_real_session_that_is_refused_says_why_and_keeps_the_review_open(todo):
    store = todo.store
    store.records.clear()
    record = open_review(todo.ctx, store, _tasks("a"), surface=store.surface)
    out = todo.ctx.process_turn("change 1: nosuch=1")
    assert out.status is TurnStatus.AWAITING_USER and "Item 1 has no field 'nosuch'" in out.answer
    assert record.state == AWAITING and record.revision == 0


# ----------------------------------------------------------------------
# Parity: the action kind keeps the approval binding's exact words and states
# ----------------------------------------------------------------------


def test_the_action_policy_reads_exactly_the_approval_words():
    policy = reply_policy(KIND_ACTION)
    assert policy.classify("yes") is Verdict.CONFIRM
    assert policy.classify("no") is Verdict.DECLINE
    assert policy.classify("always") is Verdict.ALWAYS
    for text in (" yes", "yes ", "Yes", "/yes", "sure", "yes but", "accept"):
        assert policy.classify(text) is Verdict.REPROMPT, text


def test_the_approval_binding_names_are_the_engine_names():
    assert ab.PendingAction is ReviewRecord
    assert ab.ApprovalStore is ReviewStore
    assert ab.ACTIVE_STATES is re_.ACTIVE_STATES
    assert ab.fingerprint is re_.fingerprint
    assert ab.render is re_.render


def test_a_proposal_review_gets_its_own_reply_cap_and_an_approval_keeps_two():
    """fix-3rsq.plan-review.2.5: the proposal kinds admit about 8 replies or 15 minutes; the approval (action) kind
    keeps TTL_TURNS=2 and TTL_SECONDS. Each open record carries the cap its kind is given."""
    assert re_.ttl_for_kind(KIND_ACTION) == (re_.TTL_TURNS, re_.TTL_SECONDS) == (2, 900)
    assert re_.ttl_for_kind(KIND_SUB_TASKS) == re_.ttl_for_kind(KIND_PLAN) == (8, 15 * 60)
    store = _store()
    record = open_review(None, store, _tasks("a"), surface=store.surface)
    assert (record.ttl_turns, record.ttl_seconds) == (8, 15 * 60)
