"""Proposal review reply cap (fix-kk46.4): unresolved replies count toward the cap; confirm or decline is
honoured on the cap turn; time and action caps are unchanged."""
from __future__ import annotations

import time

import pytest

from fastworkflow import approval_binding as ab
from fastworkflow.review_engine import CONFIRMED, DECLINED, EXPIRED, PROPOSAL_TTL_SECONDS, REVISED, open_review
from fastworkflow.turn import TurnStatus
from tests.test_approval_binding import (  # noqa: F401
    _added,
    _ask,
    _finish,
    annotated_library,
    no_live_model,
    no_model_call_for_parameters,
    todo,
)
from tests.test_review_engine import _tasks  # noqa: F401
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")


def _open_on_todo(todo):
    store = todo.store
    store.records.clear()
    record = open_review(todo.ctx, store, _tasks("book a flight", "tell Ann"), surface=store.surface)
    assert record.ttl_turns == 8 and record.kind == "sub_tasks"
    return record


def test_seven_unresolved_replies_then_yes_confirms_on_the_cap_turn(todo):
    record = _open_on_todo(todo)
    for number in range(7):
        todo.ctx.process_turn(f"change 2: count={number + 2}")
        assert record.state == REVISED
    assert record.turns_seen == 7
    done = todo.ctx.process_turn("accept")
    assert record.state == CONFIRMED and record.turns_seen == 7
    assert done.status is TurnStatus.COMPLETED


def test_eight_unresolved_replies_then_accept_on_the_cap_turn_is_honoured(todo):
    record = _open_on_todo(todo)
    for number in range(8):
        todo.ctx.process_turn(f"change 2: count={number + 2}")
    assert record.turns_seen == 8 and record.state == REVISED
    done = todo.ctx.process_turn("accept")
    assert record.state == CONFIRMED and record.turns_seen == 8
    assert done.status is TurnStatus.COMPLETED


def test_eight_unresolved_replies_then_decline_on_the_cap_turn_is_honoured(todo):
    record = _open_on_todo(todo)
    for number in range(8):
        todo.ctx.process_turn(f"change 2: count={number + 2}")
    assert record.turns_seen == 8
    todo.ctx.process_turn("no")
    assert record.state == DECLINED and record.turns_seen == 8


def test_eight_unresolved_replies_then_a_ninth_unresolved_reply_expires(todo):
    record = _open_on_todo(todo)
    for number in range(8):
        todo.ctx.process_turn(f"change 2: count={number + 2}")
    assert record.turns_seen == 8 and record.state == REVISED
    todo.ctx.process_turn("change 2: count=99")
    assert record.state == EXPIRED and record.turns_seen == 8
    assert todo.store.active() is None


def test_a_proposal_still_expires_when_wall_clock_runs_out(todo):
    record = _open_on_todo(todo)
    record.created_at = time.time() - PROPOSAL_TTL_SECONDS - 1
    todo.ctx.process_turn("accept")
    assert record.state == EXPIRED


def test_an_approval_still_expires_after_two_non_exact_replies(todo, annotated_library):
    _ask(todo, "add_note <text>Ann</text>", steps=_finish())
    approval = todo.store.active()
    assert approval is not None and approval.ttl_turns == ab.TTL_TURNS == 2
    todo.ctx.process_turn("maybe later")
    todo.ctx.process_turn("sure")
    todo.ctx.process_turn("yes")
    assert approval.state == EXPIRED
    assert _added(todo.chores, todo.base) == []
