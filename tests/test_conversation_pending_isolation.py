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


# ----------------------------------------------------------------------
# A deleted or archived conversation takes its parked state with it (fix-ayez)
# ----------------------------------------------------------------------


def test_dropping_a_conversation_removes_only_the_state_parked_for_it(todo, annotated_library):
    todo.ctx.bind_observability_identity(conversation_id=CONV_A)
    _ask_aaron(todo)
    todo.ctx.bind_observability_identity(conversation_id=CONV_B)
    assert todo.ctx.has_parked_pending()

    assert todo.ctx.drop_parked_pending(CONV_B) == 0            # B has nothing parked
    assert todo.ctx.has_parked_pending()
    assert todo.ctx.drop_parked_pending(CONV_A) == 1
    assert not todo.ctx.has_parked_pending()
    assert "parked_pending" in _saved(todo.ctx) and not _saved(todo.ctx)["parked_pending"]

    todo.ctx.bind_observability_identity(conversation_id=CONV_A)
    assert not todo.ctx.awaiting_user                           # nothing is restored for a dropped conversation


def test_the_persisted_blob_loses_a_dropped_conversations_parked_state(todo, annotated_library, tmp_path):
    from fastworkflow.run_fastapi_mcp.utils import drop_parked_pending_for_conversation
    from fastworkflow.session_state_store import DiskSessionStateStore

    todo.ctx.bind_observability_identity(conversation_id=CONV_A)
    _ask_aaron(todo)
    todo.ctx.bind_observability_identity(conversation_id=CONV_B)
    store = DiskSessionStateStore(str(tmp_path / "state"))
    store.save("ch-dropped", _saved(todo.ctx))
    session_manager = SimpleNamespace(session_state_store=store)

    # No loaded runtime: the saved blob is edited directly.
    assert drop_parked_pending_for_conversation(session_manager, "ch-dropped", CONV_A) == 1
    assert store.load("ch-dropped")["parked_pending"] == {}
    assert drop_parked_pending_for_conversation(session_manager, "ch-dropped", CONV_A) == 0

    # A loaded runtime: the live context and the saved blob are both changed through the persist helper.
    from fastworkflow.run_fastapi_mcp.utils import persist_pending_after_turn
    store.save("ch-dropped", _saved(todo.ctx))
    runtime = SimpleNamespace(channel_id="ch-dropped", execution_context=todo.ctx)
    assert drop_parked_pending_for_conversation(session_manager, "ch-dropped", CONV_A, runtime=runtime) == 1
    assert not todo.ctx.has_parked_pending()
    assert store.load("ch-dropped") is None                      # nothing live or parked is left to save


# ----------------------------------------------------------------------
# Parked state is pruned against the observability store (fix-ayez)
#
# The store is real (SQLiteTraceSink on a temp DB) and the ids are minted by it. A deleted conversation is a removed
# conversations row, which is what archive_conversation leaves behind.
# ----------------------------------------------------------------------

PRUNE_CHANNEL = "ch-prune"


@pytest.fixture
def obs_sink(tmp_path):
    from fastworkflow.observability.store import SQLiteTraceSink

    sink = SQLiteTraceSink(str(tmp_path / "observability.sqlite3"))
    try:
        yield sink
    finally:
        sink.close()


def _delete_conversation_row(store, channel_id: str, conversation_id: int) -> None:
    with store._connect() as conn:
        conn.execute("DELETE FROM conversations WHERE channel_id=? AND conversation_id=?",
                     (channel_id, conversation_id))
        conn.commit()


def _runtime_for(ctx, store=None):
    return SimpleNamespace(channel_id=PRUNE_CHANNEL, execution_context=ctx, observability_store=store)


def _persisted_blob(runtime, session_state_store):
    """Save through the real persist helper, as a turn or the New conversation button does, and read the blob back."""
    from fastworkflow.run_fastapi_mcp.utils import persist_pending_after_turn

    persist_pending_after_turn(SimpleNamespace(session_state_store=session_state_store), runtime, None)
    return session_state_store.load(PRUNE_CHANNEL)


def test_a_deleted_conversations_parked_state_is_pruned_at_the_next_persist(todo, annotated_library, obs_sink, tmp_path):
    from fastworkflow.session_state_store import DiskSessionStateStore

    store = obs_sink.store
    todo.ctx.set_trace_sink(obs_sink)
    conv_a = store.mint_conversation_id(PRUNE_CHANNEL)
    todo.ctx.bind_observability_identity(conversation_id=conv_a)
    _ask_aaron(todo)
    conv_b = store.mint_conversation_id(PRUNE_CHANNEL)
    todo.ctx.bind_observability_identity(conversation_id=conv_b)
    assert todo.ctx.parked_conversation_ids() == [conv_a]

    _delete_conversation_row(store, PRUNE_CHANNEL, conv_a)
    disk = DiskSessionStateStore(str(tmp_path / "state"))
    disk.save(PRUNE_CHANNEL, _saved(todo.ctx))
    saved = _persisted_blob(_runtime_for(todo.ctx, store), disk)

    assert not todo.ctx.has_parked_pending()
    assert saved is None                                        # nothing live or parked is left to save


def test_an_existing_conversations_parked_state_stays_through_a_persist(todo, annotated_library, obs_sink, tmp_path):
    from fastworkflow.session_state_store import DiskSessionStateStore

    store = obs_sink.store
    todo.ctx.set_trace_sink(obs_sink)
    conv_a = store.mint_conversation_id(PRUNE_CHANNEL)
    todo.ctx.bind_observability_identity(conversation_id=conv_a)
    _ask_aaron(todo)
    conv_b = store.mint_conversation_id(PRUNE_CHANNEL)
    todo.ctx.bind_observability_identity(conversation_id=conv_b)

    disk = DiskSessionStateStore(str(tmp_path / "state"))
    saved = _persisted_blob(_runtime_for(todo.ctx, store), disk)

    assert todo.ctx.parked_conversation_ids() == [conv_a]
    assert saved is not None and list(saved["parked_pending"]) == [str(conv_a)]


def test_unowned_and_none_parked_state_is_never_pruned(todo, annotated_library, obs_sink, tmp_path, monkeypatch):
    from fastworkflow.session_state_store import DiskSessionStateStore

    store = obs_sink.store
    # "none": a question asked with no conversation bound (no trace sink, so nothing mints one), parked by the New
    # conversation button.
    _ask_aaron(todo)
    todo.ctx.park_pending_for_new_conversation()
    assert "none" in todo.ctx._parked_pending

    # "unowned": legacy state with no conversation id, parked on restore.
    source = _scripted_context(annotated_library, monkeypatch)
    source.ctx.bind_observability_identity(conversation_id=CONV_A)
    _ask_aaron(source)
    state = _saved(source.ctx)
    del state["conversation_id"]
    legacy = _scripted_context(annotated_library, monkeypatch)
    legacy.ctx.set_trace_sink(obs_sink)
    legacy.ctx.apply_serialized_state(state)
    conv_a = store.mint_conversation_id(PRUNE_CHANNEL)
    legacy.ctx.bind_observability_identity(conversation_id=conv_a)
    assert "unowned" in legacy.ctx._parked_pending

    disk = DiskSessionStateStore(str(tmp_path / "state"))
    assert _persisted_blob(_runtime_for(todo.ctx, store), disk) is not None
    assert _prune_check(legacy.ctx, store) == 0
    assert _prune_check(todo.ctx, store) == 0
    assert "none" in todo.ctx._parked_pending and "unowned" in legacy.ctx._parked_pending


def _prune_check(ctx, store) -> int:
    from fastworkflow.run_fastapi_mcp.utils import prune_parked_pending

    return prune_parked_pending(_runtime_for(ctx, store))


class _UnreadableStore:
    def existing_conversation_ids(self, channel_id, conversation_ids):
        raise RuntimeError("database is locked")


def test_no_store_or_an_unreadable_store_prunes_nothing_and_raises_nothing(todo, annotated_library, tmp_path):
    from fastworkflow.session_state_store import DiskSessionStateStore

    todo.ctx.bind_observability_identity(conversation_id=CONV_A)
    _ask_aaron(todo)
    todo.ctx.bind_observability_identity(conversation_id=CONV_B)
    disk = DiskSessionStateStore(str(tmp_path / "state"))

    # No store at all (the runtime has no observability sink).
    assert _prune_check(todo.ctx, None) == 0
    saved = _persisted_blob(_runtime_for(todo.ctx, None), disk)
    assert todo.ctx.parked_conversation_ids() == [CONV_A] and saved["parked_pending"]

    # A store that cannot be read: a save must not fail, and nothing is dropped on a guess.
    assert _prune_check(todo.ctx, _UnreadableStore()) == 0
    saved = _persisted_blob(_runtime_for(todo.ctx, _UnreadableStore()), disk)
    assert todo.ctx.parked_conversation_ids() == [CONV_A] and saved["parked_pending"]


def test_a_restored_blob_loses_parked_state_of_deleted_conversations_only(todo, annotated_library, obs_sink, monkeypatch):
    from fastworkflow.run_fastapi_mcp.utils import _prune_parked_blob

    store = obs_sink.store
    conv_a = store.mint_conversation_id(PRUNE_CHANNEL)
    todo.ctx.bind_observability_identity(conversation_id=conv_a)
    _ask_aaron(todo)
    conv_b = store.mint_conversation_id(PRUNE_CHANNEL)
    todo.ctx.bind_observability_identity(conversation_id=conv_b)
    state = _saved(todo.ctx)
    _delete_conversation_row(store, PRUNE_CHANNEL, conv_a)

    assert _prune_parked_blob(state, PRUNE_CHANNEL, store, keep=state["conversation_id"]) == 1
    assert not state["parked_pending"]

    restarted = _scripted_context(annotated_library, monkeypatch)
    restarted.ctx.apply_serialized_state(state)
    restarted.ctx.bind_observability_identity(conversation_id=conv_a)
    assert not restarted.ctx.awaiting_user and not restarted.ctx.has_parked_pending()


def test_a_reused_conversation_id_does_not_get_the_old_parked_state(todo, annotated_library, obs_sink):
    from fastworkflow.run_fastapi_mcp.utils import reserve_conversation_id

    store = obs_sink.store
    todo.ctx.set_trace_sink(obs_sink)
    conv_a = store.mint_conversation_id(PRUNE_CHANNEL)
    todo.ctx.bind_observability_identity(conversation_id=conv_a)
    _ask_aaron(todo)
    conv_b = store.mint_conversation_id(PRUNE_CHANNEL)
    todo.ctx.bind_observability_identity(conversation_id=conv_b)
    assert todo.ctx.parked_conversation_ids() == [conv_a]

    # The archive cleanup when a channel has no conversations left: the rows and the counter go, so the store mints
    # from 1 again (the same id as conv_a, which was the channel's first conversation).
    _delete_conversation_row(store, PRUNE_CHANNEL, conv_a)
    _delete_conversation_row(store, PRUNE_CHANNEL, conv_b)
    with store._connect() as conn:
        conn.execute("DELETE FROM conversation_counters WHERE channel_id=?", (PRUNE_CHANNEL,))
        conn.commit()

    runtime = _runtime_for(todo.ctx, store)
    reused = reserve_conversation_id(runtime)
    assert reused == conv_a                                     # the id really is reused
    assert todo.ctx.bound_conversation_id() == reused
    assert not todo.ctx.awaiting_user and not todo.ctx.has_parked_pending()
