"""Integration tests for reading only the newest window of a conversation.

Two paths restore conversation history into a live session — `_create_user_runtime`
on a cold create and `POST /activate_conversation` — and both keep only
`MAX_CONVERSATION_TURNS_IN_MEMORY` turns. They used to obtain that window by
hydrating the whole conversation and slicing the rest away, so the constant
bounded what they KEPT rather than what they READ. A turn record carries its full
payload, so at production payload sizes restoring a long conversation meant
deserializing tens of megabytes to end up with twenty turns (fix-dzs.8).

Ported from the legacy `ConversationStore.get_conversation_window` when the
Phase-7 consolidation made the observability DB the single source of truth and
that module was deleted (bead `fix-gxr`). The bound is now a SQL `LIMIT` in
`ObservabilityStore.get_memory_window`, so "the excluded turns are never
deserialized" stopped being a property a test has to police — the rows never
leave SQLite. What still needs pinning is everything the two call sites depend
on: the window is the NEWEST turns in chronological order, and an empty result
is how the cold restore's step-back detects a reserved-but-empty conversation.

Runs against a real SQLite store under `tmp_path`, per the repo's
integration-tests-only rule.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import fastworkflow
from fastworkflow import TurnStatus
from fastworkflow import observability_store as obs

# Big enough that reading an excluded turn would be obvious in memory terms, small
# enough to keep the test fast. Production payloads are ~450 KB.
PAYLOAD = "x" * 4096

CHANNEL = "chan-window"


@pytest.fixture
def store(tmp_path: Path) -> obs.ObservabilityStore:
    return obs.ObservabilityStore(str(tmp_path / "observability.sqlite3"))


def _record_turn(store: obs.ObservabilityStore, conv_id: int, index: int) -> str:
    """One usable turn carrying a request-sized payload. Returns its turn_key."""
    turn_result = fastworkflow.TurnResult(
        turn_output=fastworkflow.TurnOutput(
            turn_key=fastworkflow.mint_turn_key(),
            status=TurnStatus.COMPLETED,
            answer="ok",
        ),
        channel_id=CHANNEL,
        conversation_id=conv_id,
        user_message="msg",
        conversation_summary=f"turn-{index}",
        conversation_traces=f"{index}:{PAYLOAD}",
    )
    turn_row, artifact_rows = obs.serialize_turn_result(turn_result)
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        store.upsert_turn_row(conn, turn_row, artifact_rows, obs.Redactor())
        conn.commit()
    return turn_result.turn_output.turn_key


def _conversation_of(store: obs.ObservabilityStore, turn_count: int) -> int:
    conv_id = store.mint_conversation_id(CHANNEL)
    for index in range(turn_count):
        _record_turn(store, conv_id, index)
    return conv_id


def _summaries(window: list[dict]) -> list[str]:
    return [entry["conversation summary"] for entry in window]


def test_the_window_is_the_newest_turns_in_order(store):
    """Newest N, but returned oldest-first: it is replayed as history."""
    conv_id = _conversation_of(store, 10)

    window = store.get_memory_window(CHANNEL, conv_id, 3)

    assert _summaries(window) == ["turn-7", "turn-8", "turn-9"]


def test_a_window_wider_than_the_conversation_returns_all_of_it(store):
    conv_id = _conversation_of(store, 4)

    window = store.get_memory_window(CHANNEL, conv_id, 50)

    assert _summaries(window) == [f"turn-{i}" for i in range(4)]


def test_the_window_carries_the_canonical_three_key_shape(store):
    """The restore feeds this straight into restore_history_from_turns."""
    conv_id = _conversation_of(store, 2)

    window = store.get_memory_window(CHANNEL, conv_id, 2)

    assert all(
        set(entry) == {"conversation summary", "conversation_traces", "feedback"}
        for entry in window
    )
    assert window[0]["conversation_traces"].startswith("0:")
    assert window[0]["feedback"] is None


def test_a_zero_width_window_returns_no_turns_without_failing(store):
    conv_id = _conversation_of(store, 3)

    assert store.get_memory_window(CHANNEL, conv_id, 0) == []


def test_a_conversation_with_no_usable_turns_reads_empty(store):
    """This is what the cold restore's step-back tests.

    Emptiness has to be zero USABLE turns rather than an absent row: the store
    mints a conversations row at reserve time, so a rotate leaves a row behind
    and row-presence would never step back (ruling I7).
    """
    reserved = store.mint_conversation_id(CHANNEL)

    assert store.get_memory_window(CHANNEL, reserved, 20) == []
    assert store.count_usable_turns(CHANNEL, reserved) == 0
    assert store.newest_conversation_ids(CHANNEL, limit=2) == [reserved]


def test_an_unusable_turn_is_excluded_from_the_window(store):
    """A cancelled turn has a row but no memory entry, so it is not history."""
    conv_id = _conversation_of(store, 2)
    cancelled = fastworkflow.TurnResult(
        turn_output=fastworkflow.TurnOutput(
            turn_key=fastworkflow.mint_turn_key(),
            status=TurnStatus.CANCELLED,
            answer="",
        ),
        channel_id=CHANNEL,
        conversation_id=conv_id,
        user_message="msg",
    )
    turn_row, artifact_rows = obs.serialize_turn_result(cancelled)
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        store.upsert_turn_row(conn, turn_row, artifact_rows, obs.Redactor())
        conn.commit()

    assert _summaries(store.get_memory_window(CHANNEL, conv_id, 20)) == [
        "turn-0",
        "turn-1",
    ]
    assert store.count_usable_turns(CHANNEL, conv_id) == 2


def test_the_summary_read_projects_only_the_summary_field(store):
    """Topic generation reads this; handing it whole turns would put a request
    payload in front of an LLM and reintroduce the growth the window prevents."""
    conv_id = _conversation_of(store, 5)

    summaries = store.conversation_summaries(CHANNEL, conv_id)

    assert [s["conversation summary"] for s in summaries] == [
        f"turn-{i}" for i in range(5)
    ]
    assert all(set(s) == {"conversation summary"} for s in summaries)
    assert len(str(summaries)) < 1024, "a turn payload reached the summary read"
