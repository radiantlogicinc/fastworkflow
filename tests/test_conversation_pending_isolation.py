"""A conversation never answers another conversation's pending state (fix-newconv-pending).

A paused ask_user question and an open round-1 review are PARKED with the conversation that created them, and come
back when that conversation is bound again. A pending approval is CANCELLED on a conversation change, since it would run
its side effect on a bare "yes". stop_asking's tool removal ends with its conversation. Legacy state with no
conversation id is parked as unowned and never applied automatically. A caller that never binds a conversation id
behaves exactly as before.

Real components throughout: the library copy, its real agent and tools, and the scripted model steps of the approval
and round-1 tests (the model boundary). No test calls a model.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from fastworkflow import approval_binding as ab
from fastworkflow import review_proposals, workflow_agent
from fastworkflow.review_engine import AWAITING, INVALIDATED
from fastworkflow.turn import TurnStatus
from tests.test_approval_binding import (  # noqa: F401 - fixtures and the scripted agent
    _added, _ask, _finish, annotated_library, no_live_model, no_model_call_for_parameters, todo)
from tests.test_round1_gate import EMMA, JOKE, _open_review, _split_into, gated  # noqa: F401
from tests.test_round1_handoff import _scripted_context  # noqa: F401 - a fresh context on the annotated library
from tests.test_turn_result_capture import initialized_fastworkflow  # noqa: F401

pytestmark = pytest.mark.usefixtures("initialized_fastworkflow")

CONV_A = 1
CONV_B = 2
QUESTION = "Which Aaron do you mean?"


def _ask_aaron(todo, text: str = "who is Aaron?"):
    """One agent turn that asks the user a question (ask_user suspends the turn)."""
    todo.script.steps = [("ask_user", {"clarification_request": QUESTION})]
    return todo.ctx.process_turn(text)


def _listed_tools(agent) -> set[str]:
    return set(agent.react.signature.output_fields["next_tool_name"].annotation.__args__)


def _saved(ctx) -> dict:
    """The state as a restart reads it: through JSON, so only what survives a save is kept."""
    return json.loads(json.dumps(ctx.serialize_state(channel_id="pending-isolation")))


# ----------------------------------------------------------------------
# A paused question
# ----------------------------------------------------------------------


def test_a_first_message_in_a_new_conversation_starts_a_turn_and_is_not_the_answer(todo, annotated_library):
    todo.ctx.bind_observability_identity(conversation_id=CONV_A)
    asked = _ask_aaron(todo)
    assert asked.status is TurnStatus.AWAITING_USER and todo.ctx.awaiting_user

    todo.ctx.bind_observability_identity(conversation_id=CONV_B)
    todo.script.steps = [("finish", {})]
    first = todo.ctx.process_turn("who am I?")

    assert first.turn_key != asked.turn_key                     # a new turn, not the resume of A's
    assert first.status is TurnStatus.COMPLETED
    assert todo.script.steps == []                              # the agent was asked for its steps afresh
    assert not todo.ctx.awaiting_user
    assert todo.ctx.has_parked_pending()                        # A's question waits, parked, not dropped


def test_switching_back_to_the_first_conversation_makes_its_question_pending_again(todo, annotated_library):
    todo.ctx.bind_observability_identity(conversation_id=CONV_A)
    _ask_aaron(todo)
    todo.ctx.bind_observability_identity(conversation_id=CONV_B)
    todo.script.steps = [("finish", {})]
    todo.ctx.process_turn("who am I?")

    todo.ctx.bind_observability_identity(conversation_id=CONV_A)
    assert todo.ctx.awaiting_user
    assert todo.ctx.has_parked_pending() is False

    todo.script.steps = [("finish", {})]
    reply = todo.ctx.process_turn("Aaron Lee")
    assert reply.status is TurnStatus.COMPLETED and todo.script.steps == []   # the reply answered the question


def test_stop_asking_in_the_first_conversation_does_not_disable_ask_user_in_the_second(todo, annotated_library):
    todo.ctx.bind_observability_identity(conversation_id=CONV_A)
    _ask_aaron(todo)
    agent = todo.ctx._workflow_tool_agent
    workflow_agent._apply_reply_intent(agent, "stop_asking")
    assert "ask_user" in agent.disabled_tools and "ask_user" not in _listed_tools(agent)

    todo.ctx.bind_observability_identity(conversation_id=CONV_B)
    assert "ask_user" not in agent.disabled_tools
    assert "ask_user" in _listed_tools(agent)


# ----------------------------------------------------------------------
# A pending approval is cancelled, not parked
# ----------------------------------------------------------------------


def test_a_pending_approval_is_cancelled_when_the_second_conversation_starts_and_a_yes_runs_nothing(todo, annotated_library):
    todo.ctx.bind_observability_identity(conversation_id=CONV_A)
    asked = _ask(todo, "add_note <text>Ann</text>", steps=_finish())
    assert asked.status is TurnStatus.AWAITING_USER
    record = todo.store.active()
    assert record is not None and record.suspended_agent

    todo.ctx.bind_observability_identity(conversation_id=CONV_B)
    assert record.state == INVALIDATED and record.outcome == "cancelled"
    assert todo.store.active() is None

    todo.script.steps = [("finish", {})]
    todo.ctx.process_turn("yes")
    assert _added(todo.chores, todo.base) == []                 # the stored binding never ran
    assert not todo.ctx.has_parked_pending()                    # an approval is cancelled, never parked


def test_a_new_conversation_cancels_a_pending_approval_at_once(todo, annotated_library):
    todo.ctx.bind_observability_identity(conversation_id=CONV_A)
    _ask(todo, "add_note <text>Ann</text>", steps=_finish())
    record = todo.store.active()
    todo.ctx.park_pending_for_new_conversation()
    assert record.state == INVALIDATED and todo.store.active() is None
    assert not todo.ctx.awaiting_user
    assert _added(todo.chores, todo.base) == []


# ----------------------------------------------------------------------
# A round-1 review is parked
# ----------------------------------------------------------------------


def test_an_open_round_one_review_is_parked_and_never_answered_from_the_second_conversation(gated, monkeypatch):
    gated.ctx.bind_observability_identity(conversation_id=CONV_A)
    _split_into(monkeypatch, [EMMA, JOKE])
    gated.ctx.process_turn(f"{EMMA} and {JOKE}")
    record = _open_review(gated)
    review_id = record.id

    gated.ctx.bind_observability_identity(conversation_id=CONV_B)
    assert ab.active_reviews(gated.wf) == []                    # parked off the live store
    assert not gated.ctx._turn_frame().subtask_review_open

    _split_into(monkeypatch, [JOKE])                            # B's own request opens B's own review, if anything
    gated.script.steps = [("finish", {})]
    gated.ctx.process_turn("accept")
    assert review_proposals.accepted_subtask_proposal(gated.wf) is None
    assert review_id not in {r.id for r in ab.active_reviews(gated.wf)}

    gated.ctx.bind_observability_identity(conversation_id=CONV_A)
    assert gated.ctx._turn_frame().subtask_review_open is True
    assert review_id in {r.id for r in ab.active_reviews(gated.wf)}
    assert ab.active_record(gated.wf).state == AWAITING


# ----------------------------------------------------------------------
# Parked state survives a restart and comes back only with its conversation
# ----------------------------------------------------------------------


def test_parked_state_survives_a_restart_and_comes_back_only_with_its_conversation(todo, annotated_library, monkeypatch):
    todo.ctx.bind_observability_identity(conversation_id=CONV_A)
    _ask_aaron(todo)
    todo.ctx.bind_observability_identity(conversation_id=CONV_B)
    state = _saved(todo.ctx)
    assert state["conversation_id"] == CONV_B and state["parked_pending"]

    restarted = _scripted_context(annotated_library, monkeypatch)
    restarted.ctx.apply_serialized_state(state)
    restarted.ctx.bind_observability_identity(conversation_id=CONV_B)
    assert not restarted.ctx.awaiting_user
    assert restarted.ctx.has_parked_pending()

    restarted.ctx.bind_observability_identity(conversation_id=CONV_A)
    assert restarted.ctx.awaiting_user


def test_a_live_question_saved_under_one_conversation_is_parked_when_restored_under_another(todo, annotated_library, monkeypatch):
    todo.ctx.bind_observability_identity(conversation_id=CONV_A)
    _ask_aaron(todo)
    state = _saved(todo.ctx)

    restarted = _scripted_context(annotated_library, monkeypatch)
    restarted.ctx.apply_serialized_state(state)
    restarted.ctx.bind_observability_identity(conversation_id=CONV_B)
    assert not restarted.ctx.awaiting_user
    assert restarted.ctx.has_parked_pending()


# ----------------------------------------------------------------------
# Legacy state, and callers that never bind a conversation id
# ----------------------------------------------------------------------


def test_a_legacy_blob_without_a_conversation_id_is_parked_and_never_applied(todo, annotated_library, monkeypatch):
    todo.ctx.bind_observability_identity(conversation_id=CONV_A)
    _ask_aaron(todo)
    state = _saved(todo.ctx)
    del state["conversation_id"]
    state.pop("parked_pending", None)

    restarted = _scripted_context(annotated_library, monkeypatch)
    restarted.ctx.apply_serialized_state(state)
    restarted.ctx.bind_observability_identity(conversation_id=CONV_A)
    assert not restarted.ctx.awaiting_user                      # not applied to the conversation it was saved from
    assert restarted.ctx.has_parked_pending()

    restarted.script.steps = [("finish", {})]
    reply = restarted.ctx.process_turn("Aaron Lee")
    assert reply.status is TurnStatus.COMPLETED and restarted.script.steps == []   # a new turn, not the answer


def test_a_caller_that_never_binds_a_conversation_resumes_its_question_as_before(todo, annotated_library):
    asked = _ask_aaron(todo)
    assert asked.status is TurnStatus.AWAITING_USER

    todo.script.steps = [("finish", {})]
    reply = todo.ctx.process_turn("Aaron Lee")
    assert reply.status is TurnStatus.COMPLETED and todo.script.steps == []
    assert not todo.ctx.has_parked_pending()


def test_a_saved_question_with_no_conversation_resumes_after_a_restart_as_before(todo, annotated_library, monkeypatch):
    _ask_aaron(todo)
    state = _saved(todo.ctx)
    assert state["conversation_id"] is None

    restarted = _scripted_context(annotated_library, monkeypatch)
    restarted.ctx.apply_serialized_state(state)
    assert restarted.ctx.awaiting_user and not restarted.ctx.has_parked_pending()

    restarted.script.steps = [("finish", {})]
    reply = restarted.ctx.process_turn("Aaron Lee")
    assert reply.status is TurnStatus.COMPLETED and restarted.script.steps == []


# ----------------------------------------------------------------------
# New conversation settles the state when the button is pressed
# ----------------------------------------------------------------------


def test_new_conversation_parks_the_question_and_saves_it_before_any_next_message(todo, annotated_library, tmp_path):
    """The two calls the endpoint makes on the button (run_fastapi_mcp.__main__._settle_pending_for_new_conversation):
    park_pending_for_new_conversation, then persist_pending_after_turn on the real disk store."""
    from fastworkflow.run_fastapi_mcp.utils import persist_pending_after_turn
    from fastworkflow.session_state_store import DiskSessionStateStore

    todo.ctx.bind_observability_identity(conversation_id=CONV_A)
    _ask_aaron(todo)
    store = DiskSessionStateStore(str(tmp_path / "state"))
    runtime = SimpleNamespace(channel_id="ch-new-conversation", execution_context=todo.ctx)
    session_manager = SimpleNamespace(session_state_store=store)

    todo.ctx.park_pending_for_new_conversation()
    todo.ctx.clear_conversation_history()
    persist_pending_after_turn(session_manager, runtime, None)

    assert not todo.ctx.awaiting_user and todo.ctx.has_parked_pending()
    saved = store.load("ch-new-conversation")
    assert saved is not None and saved["parked_pending"]        # the saved state agrees with the live one
    assert saved["parked_pending"]["1"][0]["conversation_id"] == CONV_A
