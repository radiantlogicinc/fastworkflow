"""Integration tests for reading only the newest window of a conversation.

Two paths restore conversation history into a live session — `_create_user_runtime`
on a cold create and `POST /activate_conversation` — and both keep only
`MAX_CONVERSATION_TURNS_IN_MEMORY` turns. They used to obtain that window by
hydrating the whole conversation and slicing the rest away, so the constant
bounded what they KEPT rather than what they READ. A turn record carries its full
payload, so at production payload sizes restoring a long conversation meant
deserializing tens of megabytes to end up with twenty turns (fix-dzs.8).

`get_conversation_window()` reads only the keys in the window. The distinction is
invisible in the returned value — a slice returns the same turns — so the test that
actually pins the behavior is the one counting deserializations
(`test_the_excluded_turns_are_never_deserialized`). Everything else here guards the
contracts the two call sites depend on: `None` for a missing record so the cold
restore's fallback to the previous id still fires, and a truthy record with no
turns for the reserved-but-empty state a rotate leaves behind.

Runs against a real SQLite-backed store under `tmp_path`, per the repo's
integration-tests-only rule. No LLM and no mocks: the read counter wraps the real
`KVStore.__getitem__` and delegates to it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fastworkflow.kvstore import KVStore
from fastworkflow.run_fastapi_mcp.conversation_store import ConversationStore

# Big enough that reading an excluded turn would be obvious in memory terms, small
# enough to keep the test fast. Production payloads are ~450 KB.
PAYLOAD = "x" * 4096


@pytest.fixture
def store(tmp_path: Path) -> ConversationStore:
    return ConversationStore("chan-window", str(tmp_path / "conversations"))


def _conversation_of(store: ConversationStore, turn_count: int) -> int:
    """A durable conversation whose turns are individually identifiable."""
    conv_id = store.reserve_next_conversation_id()
    store.append_conversation_turns(
        conv_id,
        [
            {
                "conversation summary": f"turn-{index}",
                "conversation_traces": PAYLOAD,
                "feedback": None,
            }
            for index in range(turn_count)
        ],
    )
    return conv_id


@pytest.fixture
def read_keys(monkeypatch) -> list[str]:
    """Every key the store actually deserializes, in order.

    Wraps the real ``KVStore.__getitem__`` — the single point where a stored row
    becomes a Python object — and delegates to it, so the store under test is
    unchanged apart from being observed.
    """
    keys: list[str] = []
    original = KVStore.__getitem__

    def counting_getitem(self, key):
        keys.append(str(key))
        return original(self, key)

    monkeypatch.setattr(KVStore, "__getitem__", counting_getitem)
    return keys


def _summaries(record: dict) -> list[str]:
    return [turn["conversation summary"] for turn in record["turns"]]


# ---------------------------------------------------------------------------
# What comes back
# ---------------------------------------------------------------------------

def test_the_window_is_the_newest_turns_in_order(store):
    conv_id = _conversation_of(store, 100)

    record = store.get_conversation_window(conv_id, 20)

    assert _summaries(record) == [f"turn-{i}" for i in range(80, 100)]


def test_a_window_wider_than_the_conversation_returns_all_of_it(store):
    conv_id = _conversation_of(store, 3)

    record = store.get_conversation_window(conv_id, 20)

    assert _summaries(record) == ["turn-0", "turn-1", "turn-2"]


def test_the_record_shape_matches_the_full_read(store):
    """The two getters must be interchangeable apart from how many turns come back.

    Both call sites read metadata off this record, and one of them (activate)
    only tests it for truthiness, so a differently-shaped record would fail
    somewhere other than here.
    """
    conv_id = _conversation_of(store, 30)
    store.update_conversation_topic_summary(conv_id, "Some Topic", "Some summary")

    windowed = store.get_conversation_window(conv_id, 20)
    full = store.get_conversation(conv_id)

    assert set(windowed) == set(full)
    assert windowed["topic"] == full["topic"] == "Some Topic"
    assert windowed["summary"] == full["summary"] == "Some summary"
    assert windowed["created_at"] == full["created_at"]
    # Bookkeeping stays hidden, exactly as the full read hides it.
    assert "appended_turn_count" not in windowed


# ---------------------------------------------------------------------------
# The point of the change
# ---------------------------------------------------------------------------

def test_the_excluded_turns_are_never_deserialized(store, read_keys):
    """THE test for this fix. A slice-based implementation fails only this one.

    Reading a 20-turn window out of a 100-turn conversation must deserialize the
    conversation record plus 20 turn records, not 100.
    """
    conv_id = _conversation_of(store, 100)
    read_keys.clear()

    store.get_conversation_window(conv_id, 20)

    turn_reads = [k for k in read_keys if ":turn:" in k]
    assert len(turn_reads) == 20, (
        f"read {len(turn_reads)} turn record(s) to return a window of 20; the "
        "excluded turns are being deserialized and thrown away"
    )
    assert turn_reads == [f"conv:{conv_id}:turn:{i}" for i in range(80, 100)]


def test_the_full_read_still_returns_every_turn(store, read_keys):
    """Regression guard: /admin/dump_all_conversations genuinely wants all of it."""
    conv_id = _conversation_of(store, 40)
    read_keys.clear()

    record = store.get_conversation(conv_id)

    assert _summaries(record) == [f"turn-{i}" for i in range(40)]
    assert len([k for k in read_keys if ":turn:" in k]) == 40


# ---------------------------------------------------------------------------
# Contracts the two restore call sites depend on
# ---------------------------------------------------------------------------

def test_a_missing_record_is_none_so_the_cold_restore_can_fall_back(store):
    """`_create_user_runtime` reads `not conversation` to mean "no record".

    It then retries at `conv_id - 1`, which is how a session started right after a
    rotate finds the conversation it should actually restore. Returning an empty
    record instead of None would break that fallback silently.
    """
    assert store.get_conversation_window(999, 20) is None


def test_a_reserved_but_empty_conversation_is_a_record_with_no_turns(store):
    """Distinguishable from a missing record, which is the state a rotate leaves.

    `reserve_next_conversation_id` moves the counter without writing a record, so
    this asserts the boundary: still no record until a turn is appended, then a
    truthy record whose window is empty.
    """
    conv_id = store.reserve_next_conversation_id()
    assert store.get_conversation_window(conv_id, 20) is None

    store.append_conversation_turns(
        conv_id,
        [{"conversation summary": "first", "conversation_traces": None, "feedback": None}],
    )
    record = store.get_conversation_window(conv_id, 20)
    assert record is not None
    assert _summaries(record) == ["first"]


def test_a_zero_width_window_returns_no_turns_without_failing(store):
    """Defensive: the callers pass a positive constant, but 0 must not read turns."""
    conv_id = _conversation_of(store, 5)

    record = store.get_conversation_window(conv_id, 0)

    assert record is not None
    assert record["turns"] == []
