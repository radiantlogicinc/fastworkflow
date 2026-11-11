"""A split of more than MAX_SUBTASKS items is reduced to the cap by joining (``split_merge.reduce_to_cap``); the review
shows the joined items, and a joined item left as drafted runs each of its parts as a job of its own, one level deep.
Real library copy with its trained heads and the real agent with a scripted model, as tests/test_round1_gate.py has them."""
from __future__ import annotations

import pytest

from fastworkflow import review_proposals
from fastworkflow.split_merge import JoinedSubtask, reduce_to_cap
from fastworkflow.turn import TurnStatus
from fastworkflow.workflow_agent import MAX_SUBTASKS
from tests.test_approval_binding import _finish  # noqa: F401
from tests.test_round1_gate import (  # noqa: F401 - the gated library context and the guards its fixtures need
    _open_review, _split_into, annotated_library, gated, initialized_fastworkflow, no_draft_extraction,
    no_live_model, no_model_call_for_parameters, todo)

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

NAMES = ["Ann Lee", "Bob Ray", "Cy Young", "Di Moss", "Ed Park", "Flo Hill", "Gus Boyd", "Hal Cox", "Ivy Dunn",
         "Jo Enns", "Kit Frye", "Lou Gray", "Max Hale"]


def _joined_split():
    items = [f"find member {name}" for name in NAMES]
    query = "find members " + ", ".join(NAMES)
    reduction = reduce_to_cap(query, items, MAX_SUBTASKS)
    assert len(items) == 13 and len(reduction.items) == MAX_SUBTASKS
    return query, reduction.items


def test_joining_keeps_the_parts_of_each_joined_item_in_order():
    _, joined = _joined_split()
    parts = [part for item in joined for part in (item.parts if isinstance(item, JoinedSubtask) else (item,))]
    assert parts == [f"find member {name}" for name in NAMES]
    assert all(isinstance(item, JoinedSubtask) for item in joined if item.count("find member") > 1)
    assert all(len(part) and not isinstance(part, JoinedSubtask) for item in joined
               if isinstance(item, JoinedSubtask) for part in item.parts)


def test_the_review_shows_the_cap_and_the_run_takes_every_part_as_a_job(gated, monkeypatch):
    query, joined = _joined_split()
    _split_into(monkeypatch, joined)
    out = gated.ctx.process_turn(query)
    assert out.status is TurnStatus.AWAITING_USER
    record = _open_review(gated)
    assert len(record.binding["items"]) == MAX_SUBTASKS            # the review shows the joined items
    assert any(item.get("parts") for item in record.binding["items"])


def test_an_unattended_turn_runs_thirteen_jobs_from_ten_items(gated, monkeypatch):
    query, joined = _joined_split()
    _split_into(monkeypatch, joined)
    gated.script.steps = _finish() * 13
    out = gated.ctx.process_turn(query, unattended=True)
    assert out.status is TurnStatus.COMPLETED
    assert gated.script.steps == []                                  # all thirteen jobs ran, none more
    plan = gated.ctx._turn_subtask_plan
    assert [item.goal for item in plan.items] == [f"find member {name}" for name in NAMES]
    assert all(not item.parts for item in plan.items)                # parts are leaves


def test_a_joined_item_whose_goal_was_edited_runs_as_one_job(gated):
    query, joined = _joined_split()
    draft = review_proposals.draft_round1_proposal(gated.wf, list(joined))
    index = next(n for n, item in enumerate(draft.items) if item.parts)
    edited_items = list(draft.items)
    edited_items[index] = edited_items[index].model_copy(update={"goal": "find member Ann Lee and read her record"})
    edited = type(draft)(items=edited_items)
    goals, plan = review_proposals.expand_joined_parts(gated.wf, [item.goal for item in edited.items], edited)
    assert len(goals) == len(plan.items) == 13 - len(draft.items[index].parts) + 1
    assert "find member Ann Lee and read her record" in goals
    unedited_goals, unedited = review_proposals.expand_joined_parts(
        gated.wf, [item.goal for item in draft.items], draft)
    assert len(unedited_goals) == 13


def test_a_joined_item_whose_inputs_were_edited_runs_as_one_job(gated):
    _, joined = _joined_split()
    draft = review_proposals.draft_round1_proposal(gated.wf, list(joined))
    index = next(n for n, item in enumerate(draft.items) if item.parts)
    items = list(draft.items)
    changed = items[index].model_copy(update={"inputs": [review_proposals.SubTaskInput(name="name", value="Zed Zane")]})
    items[index] = changed
    goals, plan = review_proposals.expand_joined_parts(
        gated.wf, [item.goal for item in items], type(draft)(items=items))
    assert changed.goal in goals and len(goals) == 13 - len(draft.items[index].parts) + 1


def test_a_split_within_the_cap_is_not_touched():
    items = [f"find member {name}" for name in NAMES[:5]]
    assert reduce_to_cap("find members " + ", ".join(NAMES[:5]), items, MAX_SUBTASKS).items == items
