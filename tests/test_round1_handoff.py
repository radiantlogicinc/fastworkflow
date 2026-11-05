"""Round 1, batch C (fix-3rsq.plan-review.2.7 free-text edits, fix-3rsq.plan-review.2.9 hand-off and caching).

Free-text edits: a reply that is not accept or no, and not a deterministic edit, is read by ONE typed model call into
structured edits (add, merge, split, change, drop), shown back for confirmation, and an unreadable reply asks again and
counts toward the review's reply cap. The model boundary is DummyLM on the planner's role (``_current_planner_lm``).

Hand-off: an accepted plan (or a draft that was not reviewed) drives execution. Each goal runs with its bound means: a
planning command through the phase runner with its inputs, a plain command with its inputs and a listing narrowed to it,
or from scratch. ``from sub-task N`` inputs take N's answer. A goal is complete only when the finish check says so.
An accept after a process restart resumes the review. The head candidates and the Jev match are asked once per turn.

Integration tests on the annotated library copy (tests/test_round1_gate.py's ``gated`` context): the real agent and its
tools, the scripted step choices of the approval tests, and DummyLM at the model boundary. No test calls a model.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from dspy.utils import DummyLM

from fastworkflow import review_engine, review_proposals, run_listing, workflow_agent
from fastworkflow import approval_binding as ab
from fastworkflow.review_engine import AWAITING, EXPIRED, REVISED
from fastworkflow.review_proposals import SubTaskProposal
from fastworkflow.turn import TurnStatus
from fastworkflow.workflow_agent import initialize_workflow_tool_agent
from tests.test_approval_binding import (  # noqa: F401 - the fixtures and the scripted agent
    Script, annotated_library, no_live_model, no_model_call_for_parameters, todo)
from tests.test_planning_command_phases import RecordingSink
from tests.test_round1_gate import (  # noqa: F401 - the gated context and the gate's helpers
    EMMA, FIND, JOKE, _confirmed, _marker, _open_review, _ran, _split_into, gated)
from tests.test_turn_result_capture import (  # noqa: F401 - the shared fixtures
    _make_agent_ctx, _set_agents, initialized_fastworkflow)

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")


@pytest.fixture(autouse=True)
def no_draft_extraction(monkeypatch):
    """The draft's argument extraction is a model call (fix-48np): off here, so these tests keep asserting that the
    review, the gate and the hand-off make no parameter-extraction model call (the guard fixture raises if one is
    made). Drafting with extraction on is covered by tests/test_drafted_subtask_inputs.py."""
    monkeypatch.setenv("FW_DRAFT_ARGUMENTS", "off")

BOOK_BY_TITLE = "Library/book_by_title"
FIND_MEMBER = "Library/find_member"
FIND_TEXT = "find_member <query>Ann Lee</query>"


class CountingLM(DummyLM):
    """A DummyLM that counts its calls (the model boundary of the Jev match and of the reader)."""

    def __init__(self, answers):
        super().__init__(answers)
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return super().__call__(*args, **kwargs)


def _reply_lm(*edits):
    """A reader's answer: the typed edits the model reads from the user's words."""
    return DummyLM([{"edits": list(edits)}])


def _planner_stub(monkeypatch, queries: list):
    """The planner's boundary: the query each run gets is recorded; a run's plan is named for the find only."""

    def fake(user_query, session, with_agent_inputs_and_trajectory=False, planner_lm=None, **kwargs):
        queries.append(user_query)
        session._initial_plan_text = "1. find_member" if "Ann Lee" in user_query else ""
        return user_query

    monkeypatch.setattr("fastworkflow.workflow_agent.build_query_with_next_steps", fake)


# ----------------------------------------------------------------------
# 2.7: free-text edits
# ----------------------------------------------------------------------


def test_a_free_text_edit_is_read_by_one_model_call_and_shown_back_until_confirmed(gated, monkeypatch):
    _split_into(monkeypatch, [EMMA, JOKE])
    gated.ctx.process_turn(f"{EMMA} and {JOKE}")
    record = _open_review(gated)
    lm = CountingLM([{"edits": [{"op": "change", "index": 2, "field": "means", "value": f"planning:{BOOK_BY_TITLE}"}]}])
    gated.ctx._current_planner_lm = lm
    out = gated.ctx.process_turn("use book_by_title for the second part")
    assert lm.calls == 1                                    # one model call for the reply
    assert out.status is TurnStatus.AWAITING_USER
    assert "I read your reply as these edits" in (out.answer or "")
    assert record.state == REVISED and record.revision == 1
    assert record.binding["items"][1]["means"] == {"kind": "planning", "name": BOOK_BY_TITLE}
    assert _confirmed(gated) == []                          # shown back, not applied as confirmed

    gated.script.steps = [("finish", {})] * 3               # the review frame, then the two goals
    done = gated.ctx.process_turn("accept")
    assert done.status is TurnStatus.COMPLETED
    accepted = review_proposals.accepted_subtask_proposal(gated.wf)
    assert accepted is not None and accepted.items[1].means.name == BOOK_BY_TITLE
    assert _marker(gated) == "reviewed"


def test_a_free_text_drop_is_read_and_the_confirmed_edit_applies(gated, monkeypatch):
    _split_into(monkeypatch, [EMMA, JOKE])
    gated.ctx.process_turn(f"{EMMA} and {JOKE}")
    record = _open_review(gated)
    gated.ctx._current_planner_lm = _reply_lm({"op": "drop", "index": 2})
    out = gated.ctx.process_turn("drop the last part")
    assert "I read your reply as these edits" in (out.answer or "")
    assert record.state == REVISED and [item["goal"] for item in record.binding["items"]] == [EMMA]
    gated.script.steps = [("finish", {})] * 2
    gated.ctx.process_turn("accept")
    accepted = review_proposals.accepted_subtask_proposal(gated.wf)
    assert accepted is not None and [item.goal for item in accepted.items] == [EMMA]


@pytest.mark.parametrize("edit, expected", [
    ({"op": "add", "after": 1, "value": "tell a story"}, [EMMA, "tell a story", JOKE]),
    ({"op": "merge", "indexes": [1, 2], "value": "both at once"}, ["both at once"]),
    ({"op": "split", "index": 1, "parts": ["part a", "part b"]}, ["part a", "part b", JOKE]),
])
def test_the_reader_reads_add_merge_and_split_as_typed_edits_the_engine_applies(gated, edit, expected):
    """The reader's output is the typed edit list: each op becomes the engine's edit, and the engine applies it."""
    from fastworkflow.review_proposals import SubTask

    proposal = SubTaskProposal(items=[SubTask(goal=EMMA), SubTask(goal=JOKE)])
    read = review_proposals.model_edit_reader(planner_lm=_reply_lm(edit))
    edits = read("the words of the reply", proposal)
    edited = review_engine.apply_edits(proposal, edits)
    assert [item.goal for item in edited.items] == expected


def test_a_reply_the_reader_cannot_read_asks_again_and_counts_toward_the_cap(gated, monkeypatch):
    _split_into(monkeypatch, [EMMA, JOKE])
    gated.ctx.process_turn(f"{EMMA} and {JOKE}")
    record = _open_review(gated)
    garbage = CountingLM([{"edits": "this is not a list of edits"}] * 20)
    gated.ctx._current_planner_lm = garbage
    for _ in range(record.ttl_turns):
        out = gated.ctx.process_turn("make it better somehow")
        assert out.status is TurnStatus.AWAITING_USER
        assert record.state == AWAITING and record.revision == 0
        assert "I could not read that as an edit" in (out.answer or "")
    assert record.turns_seen == record.ttl_turns and garbage.calls >= record.ttl_turns
    out = gated.ctx.process_turn("make it better somehow")  # the reply past the cap: the review has expired
    assert record.state == EXPIRED
    assert "expired" in (out.answer or "") and "nothing was run" in (out.answer or "")
    assert _ran(out) == []


# ----------------------------------------------------------------------
# 2.9: the hand-off
# ----------------------------------------------------------------------


def test_an_accepted_plan_runs_its_plain_goal_with_its_inputs_and_a_narrowed_listing(gated, monkeypatch):
    queries: list = []
    _planner_stub(monkeypatch, queries)
    listings: list = []
    _split_into(monkeypatch, [FIND, JOKE])
    gated.ctx.process_turn(f"{FIND} and {JOKE}")
    _open_review(gated)
    gated.ctx.process_turn("change 1: means=plain:" + FIND_MEMBER)
    gated.ctx.process_turn("change 1: inputs=query=Ann Lee")
    gated.ctx.process_turn("change 2: means=from_scratch")
    # The turn's own agent: while the review frame is on top, _workflow_tool_agent is the review frame's.
    turn_agent = gated.ctx._turn_frame().agent
    inner = gated.script

    def recording(module, trajectory, **inputs):
        if module is turn_agent.react and "available_commands" in inputs:
            listings.append(inputs["available_commands"])
        return inner(module, trajectory, **inputs)

    turn_agent._call_with_potential_trajectory_truncation = recording
    gated.script.steps = [("finish", {}), ("execute_workflow_query", {"command": FIND_TEXT}), ("finish", {}),
                          ("finish", {})]
    out = gated.ctx.process_turn("accept")
    assert out.status is TurnStatus.COMPLETED and gated.script.steps == []
    # The first goal's run carries its inputs; the plain goal's listing is narrowed to its command (not the whole job list).
    assert queries[0] == f"{FIND} (inputs: query=Ann Lee)"
    assert FIND_MEMBER.split("/")[-1] in listings[0]
    assert "count_books" not in listings[0] and "book_by_title" not in listings[0]
    assert queries[-1] == JOKE                              # the from-scratch goal runs as its own goal


def test_a_from_sub_task_input_takes_the_output_of_that_sub_task(gated, monkeypatch):
    queries: list = []
    _planner_stub(monkeypatch, queries)
    _split_into(monkeypatch, [JOKE, FIND])
    gated.ctx.process_turn(f"{JOKE} and {FIND}")
    _open_review(gated)
    gated.ctx.process_turn("change 1: means=from_scratch")
    gated.ctx.process_turn("change 2: means=plain:" + FIND_MEMBER)
    gated.ctx.process_turn("change 2: inputs=query=from sub-task 1")
    gated.script.steps = [("finish", {}), ("finish", {}), ("execute_workflow_query", {"command": FIND_TEXT}),
                          ("finish", {})]
    out = gated.ctx.process_turn("accept")
    assert out.status is TurnStatus.COMPLETED and gated.script.steps == []
    # Sub-task 1 ran first; its answer (the scripted model's final answer, "the answer") is the second goal's input.
    assert queries[0] == JOKE
    assert queries[-1] == f"{FIND} (inputs: query=the answer)"


def test_a_from_sub_task_input_takes_the_typed_output_of_a_run_that_produced_one(gated, monkeypatch):
    """find_member returns a typed Output (member_uids, names, total): the 'from sub-task 1' input of the next goal is
    that Output, not the agent's answer text (fix-3rsq.plan-review.2.11)."""
    queries: list = []

    def fake(user_query, session, with_agent_inputs_and_trajectory=False, planner_lm=None, **kwargs):
        # The plan names find_member for the goal whose own input is Ann Lee; the second goal's input carries the
        # typed output, which also holds "Ann Lee", so the stub keys on the input's own text.
        queries.append(user_query)
        session._initial_plan_text = "1. find_member" if "query=Ann Lee" in user_query else ""
        return user_query

    monkeypatch.setattr("fastworkflow.workflow_agent.build_query_with_next_steps", fake)
    _split_into(monkeypatch, [FIND, JOKE])
    gated.ctx.process_turn(f"{FIND} and {JOKE}")
    _open_review(gated)
    gated.ctx.process_turn("change 1: means=plain:" + FIND_MEMBER)
    gated.ctx.process_turn("change 1: inputs=query=Ann Lee")
    gated.ctx.process_turn("change 2: means=from_scratch")
    gated.ctx.process_turn("change 2: inputs=found=from sub-task 1")
    gated.script.steps = [("finish", {}), ("execute_workflow_query", {"command": FIND_TEXT}), ("finish", {}),
                          ("finish", {})]
    out = gated.ctx.process_turn("accept")
    assert out.status is TurnStatus.COMPLETED and gated.script.steps == []
    # The typed Output of find_member, as JSON: its member_uids and total are in the second goal's inputs.
    assert '"member_uids"' in queries[-1] and '"total"' in queries[-1]
    assert "found=the answer" not in queries[-1]


def test_a_goal_whose_command_succeeded_but_whose_finish_check_fails_is_not_complete(gated, monkeypatch):
    """Inv. 3: the goal's plan names find_member. A different command that succeeds does not satisfy it, the finish check
    sends the agent back, and the goal completes only once find_member has run."""
    queries: list = []
    _planner_stub(monkeypatch, queries)
    _split_into(monkeypatch, [FIND, JOKE])
    gated.ctx.process_turn(f"{FIND} and {JOKE}")
    _open_review(gated)
    gated.ctx.process_turn("change 1: means=plain:" + FIND_MEMBER)
    gated.ctx.process_turn("change 1: inputs=query=Ann Lee")
    from fastworkflow.observation_offloading import finish_check

    notes: list = []
    real_review = finish_check.review_finish

    def spy(**kwargs):
        note = real_review(**kwargs)
        notes.append(note)
        return note

    monkeypatch.setattr(finish_check, "review_finish", spy)
    gated.script.steps = [("finish", {}),                                   # the review frame
                          ("execute_workflow_query", {"command": "list_branches"}),
                          ("finish", {}),                                   # refused: find_member has not run
                          ("execute_workflow_query", {"command": FIND_TEXT}),
                          ("finish", {}),                                   # accepted: the plan is met
                          ("finish", {})]                                   # the second goal
    out = gated.ctx.process_turn("accept")
    assert out.status is TurnStatus.COMPLETED and gated.script.steps == []
    assert [note for note in notes if note] and "not every plan step is done yet" in next(n for n in notes if n)


# ----------------------------------------------------------------------
# 2.9 (a): an accept after a process restart resumes the review
# ----------------------------------------------------------------------


def _scripted_context(annotated_library, monkeypatch):
    """A fresh agent-mode context on the annotated library, scripted as the todo fixture scripts one (the same seams)."""
    from fastworkflow.context_identity import context_class_for
    from fastworkflow.examples.library.application.library import Library

    ctx, wf = _make_agent_ctx(str(annotated_library), monkeypatch)
    sink = RecordingSink()
    ctx.set_trace_sink(sink)
    ctx.push_active_workflow(wf)
    library = Library()
    wf.root_command_context = library
    wf.current_command_context = library
    ctx._cme_workflow.context["run_as_agent"] = True
    wf.context["run_as_agent"] = True
    monkeypatch.setattr(context_class_for(wf, "Book"), "instance_label_attr", "uid", raising=False)
    agent = initialize_workflow_tool_agent(ctx)
    _set_agents(ctx, agent)
    script = Script(agent)
    monkeypatch.setattr(agent, "_call_with_potential_trajectory_truncation", script)
    build_agent = ctx._build_react_agent

    def scripted_build():
        frame_agent = build_agent()
        script.agents.append(frame_agent)
        monkeypatch.setattr(frame_agent, "_call_with_potential_trajectory_truncation", script)
        return frame_agent

    monkeypatch.setattr(ctx, "_build_react_agent", scripted_build)
    monkeypatch.setattr(ctx, "_refine_user_query", lambda message, history: message)
    return SimpleNamespace(ctx=ctx, wf=wf, sink=sink, script=script, library=library)


def test_an_accept_after_a_process_restart_resumes_the_review_and_runs_its_goals(annotated_library, monkeypatch):
    queries: list = []
    _planner_stub(monkeypatch, queries)
    _split_into(monkeypatch, [EMMA, JOKE])
    first = _scripted_context(annotated_library, monkeypatch)
    first.ctx.process_turn(f"{EMMA} and {JOKE}")
    record = _open_review(SimpleNamespace(store=ab.store_for(first.wf)))
    first.ctx.process_turn("change 2: inputs=Ann Lee")
    assert record.state == REVISED
    state = json.loads(json.dumps(first.ctx.serialize_state(channel_id="restart-test")))

    # The process restarts: a new context and a new workflow, restored from the serialised state.
    second = _scripted_context(annotated_library, monkeypatch)
    second.ctx.apply_serialized_state(state)
    restored = ab.store_for(second.wf).active()
    assert restored is not None and restored.id == record.id and restored.state == REVISED
    assert second.ctx._turn_frame().subtask_review_open is True

    second.script.steps = [("finish", {})] * 3              # the review frame, then the two goals
    out = second.ctx.process_turn("accept")
    assert out.status is TurnStatus.COMPLETED and second.script.steps == []
    accepted = review_proposals.accepted_subtask_proposal(second.wf)
    assert accepted is not None and accepted.items[1].inputs[0].value == "Ann Lee"
    assert second.ctx._turn_subtask_plan == accepted
    spans = [span for span in second.sink.spans if span.name == "fw.turn"]
    assert spans and spans[-1].attributes.get("subtask_review") == "reviewed"


# ----------------------------------------------------------------------
# 2.9 (b): head candidates and the Jev match are asked once per turn
# ----------------------------------------------------------------------


def test_the_fast_path_check_the_draft_and_the_run_share_one_head_lookup_and_one_jev_match(gated, monkeypatch):
    """A single unsure sub-task: the fast-path check and the draft ask the heads for it once, and the Jev match runs once
    per scope in the draft; the accepted run asks neither again."""
    workflow_agent._planning_command_matcher.cache_clear()
    jev = CountingLM([{"job": "none"}] * 50)
    monkeypatch.setattr(workflow_agent, "_jev_lm", lambda: jev)
    head_asks: list = []
    real_head_answers = run_listing.head_answers

    def counting_head_answers(workflow, sub_task, command_info_root=None):
        head_asks.append(sub_task)
        return real_head_answers(workflow, sub_task, command_info_root)

    monkeypatch.setattr(run_listing, "head_answers", counting_head_answers)
    try:
        _split_into(monkeypatch, [JOKE])
        gated.ctx.process_turn(JOKE)
        _open_review(gated)
        after_draft_jev = jev.calls
        after_draft_heads = head_asks.count(JOKE)
        gated.script.steps = [("finish", {}), ("finish", {})]   # the review frame, then the goal
        out = gated.ctx.process_turn("accept")
        assert out.status is TurnStatus.COMPLETED
    finally:
        workflow_agent._planning_command_matcher.cache_clear()
    assert after_draft_jev > 0                              # the Jev match did run (once per scope)
    assert jev.calls == after_draft_jev                     # and the accepted run did not ask it again
    assert after_draft_heads == 1 and head_asks.count(JOKE) == 1
