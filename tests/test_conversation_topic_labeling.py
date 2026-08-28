"""Integration tests for how the observability store assigns conversation topics.

Topics are the user-facing handle on a stored conversation, so the topic must be
unique per channel *and* stable across rewrites. Those two requirements pull
against each other, and the collision helper is where they meet:

* **Uniqueness** — two different conversations on one channel must not end up
  sharing a topic, so the second one gets a numeric suffix. Matching is case- and
  whitespace-insensitive, because lookup normalizes the same way.
* **Stability** — rewriting a conversation's *own* topic, which happens whenever a
  topic is regenerated for unchanged content, must be idempotent. Counting the
  record's own stored topic as a collision made the topic oscillate between
  ``'T'`` and ``'T 1'`` on every refresh (fix-dzs.2).

Ported from the legacy ``ConversationStore`` when the Phase-7 consolidation made
the observability DB the single source of truth and that module was deleted
(bead `fix-gxr`). The behaviour under test is unchanged — it moved to
``ObservabilityStore.apply_label_txn``, the single label-write enforcement point,
which does the comparison in Python because SQLite's ``lower()`` is ASCII-only
(ruling I9).

Everything runs against a real SQLite store under ``tmp_path``, per the repo's
integration-tests-only rule. No LLM is involved: topics are passed in directly
rather than generated.

Blank topics are covered as well. An empty or whitespace-only topic is a *failed
generation*, not a title, and has to be stored as exactly ``""`` — the sentinel
that means "no successful title yet, retry on the next eligible trigger". It used
to collide with every *other* unlabeled conversation (they all store ``topic ==
""``) and be stored as ``' 1'``, which is neither a title nor the sentinel, so
such a conversation counted as labeled forever and a picker rendered it as ``' 1'``
(fix-dzs.3).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import fastworkflow
from fastworkflow import TurnStatus
from fastworkflow import observability_store as obs


@pytest.fixture
def store(tmp_path: Path) -> obs.ObservabilityStore:
    return obs.ObservabilityStore(str(tmp_path / "observability.sqlite3"))


CHANNEL = "chan"


def _conversation_with_a_turn(store: obs.ObservabilityStore, summary: str) -> int:
    """A durable conversation with one usable turn and no title yet.

    The turn is written through the real serializer so it carries the memory
    columns; a row without them is a trace, not conversation memory, and the
    label triggers would not see it (ruling I4).
    """
    conv_id = store.mint_conversation_id(CHANNEL)
    turn_result = fastworkflow.TurnResult(
        turn_output=fastworkflow.TurnOutput(
            turn_key=fastworkflow.mint_turn_key(),
            status=TurnStatus.COMPLETED,
            answer="ok",
        ),
        channel_id=CHANNEL,
        conversation_id=conv_id,
        user_message="msg",
        conversation_summary=summary,
    )
    turn_row, artifact_rows = obs.serialize_turn_result(turn_result)
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        store.upsert_turn_row(conn, turn_row, artifact_rows, obs.Redactor())
        conn.commit()
    return conv_id


def _topic(store: obs.ObservabilityStore, conv_id: int) -> str:
    return store.conversation_label_state(CHANNEL, conv_id)[0]


def _label(store: obs.ObservabilityStore, conv_id: int, topic: str, summary: str) -> str:
    """Write a label the way the labeling path does: blank passes as None.

    A blank generated topic must never be stored as a title, so the caller sends
    None and the store preserves whatever is there. Returns the stored topic.
    """
    return store.record_conversation_label(
        CHANNEL, conv_id, topic if topic.strip() else None, summary
    )


# ---------------------------------------------------------------------------
# Stability: a conversation cannot collide with itself
# ---------------------------------------------------------------------------

def test_rewriting_a_conversations_own_topic_is_idempotent(store):
    """Regenerating an unchanged topic must not add a suffix — nor take one off.

    The oscillation is the giveaway: counting the record's own topic as a
    collision suffixes the second write, and then the third write no longer
    matches (the stored topic is now ``'T 1'``) so the suffix disappears again.
    A single write is therefore not enough to catch this; three are.
    """
    conv_id = _conversation_with_a_turn(store, "asked about an order")

    for _ in range(3):
        _label(store, conv_id, "Order Status", "s")
        assert _topic(store, conv_id) == "Order Status"


def test_rewriting_own_topic_in_a_different_case_is_also_unsuffixed(store):
    """Self-exclusion is by identity, not by string equality."""
    conv_id = _conversation_with_a_turn(store, "asked about an order")
    _label(store, conv_id, "Order Status", "s")

    _label(store, conv_id, "ORDER STATUS", "s2")
    assert _topic(store, conv_id) == "ORDER STATUS"


def test_a_legitimate_suffix_does_not_drift_on_rewrite(store):
    """A conversation that genuinely needed a suffix keeps the same one."""
    first = _conversation_with_a_turn(store, "order one")
    second = _conversation_with_a_turn(store, "order two")

    _label(store, first, "Order Status", "s1")
    _label(store, second, "Order Status", "s2")
    assert _topic(store, second) == "Order Status 1"

    _label(store, second, "Order Status", "s3")
    assert _topic(store, second) == "Order Status 1"


# ---------------------------------------------------------------------------
# Uniqueness: two conversations must not share a topic
# ---------------------------------------------------------------------------

def test_a_different_conversation_with_the_same_topic_still_gets_suffixed(store):
    first = _conversation_with_a_turn(store, "order one")
    second = _conversation_with_a_turn(store, "order two")

    _label(store, first, "Order Status", "s1")
    _label(store, second, "Order Status", "s2")

    assert _topic(store, first) == "Order Status"
    assert _topic(store, second) == "Order Status 1"


def test_collision_detection_is_case_insensitive(store):
    first = _conversation_with_a_turn(store, "order one")
    second = _conversation_with_a_turn(store, "order two")

    _label(store, first, "Order Status", "s1")
    _label(store, second, "ORDER STATUS", "s2")

    assert _topic(store, second) == "ORDER STATUS 1"


def test_collision_detection_is_whitespace_insensitive(store):
    """The suffix is appended to the candidate verbatim, so padding survives."""
    first = _conversation_with_a_turn(store, "order one")
    second = _conversation_with_a_turn(store, "order two")

    _label(store, first, "Order Status", "s1")
    _label(store, second, "  Order Status  ", "s2")

    assert _topic(store, second) == "  Order Status   1"


def test_collisions_chain_past_an_existing_suffix(store):
    """A third conversation cannot land on the second one's suffixed title."""
    ids = [_conversation_with_a_turn(store, f"order {i}") for i in range(3)]
    for conv_id in ids:
        _label(store, conv_id, "Order Status", "s")

    assert [_topic(store, conv_id) for conv_id in ids] == [
        "Order Status",
        "Order Status 1",
        "Order Status 2",
    ]


def test_uniqueness_is_scoped_to_the_channel(store):
    """Two channels are two namespaces; one user's title cannot suffix another's."""
    mine = _conversation_with_a_turn(store, "order one")
    _label(store, mine, "Order Status", "s")

    theirs = store.mint_conversation_id("other-chan")
    stored = store.record_conversation_label(
        "other-chan", theirs, "Order Status", "s"
    )

    assert stored == "Order Status"
    assert store.conversation_label_state("other-chan", theirs)[0] == "Order Status"


# ---------------------------------------------------------------------------
# Blank topics: the retry sentinel, exempt from uniqueness
# ---------------------------------------------------------------------------

def test_a_blank_topic_stays_blank_when_other_unlabeled_conversations_exist(store):
    """Every unlabeled conversation stores ``""``, so uniquifying a blank would
    collide with all of them and yield ``' 1'`` — neither a title nor the
    sentinel a lazy-fill trigger watches for (fix-dzs.3)."""
    _conversation_with_a_turn(store, "one")
    _conversation_with_a_turn(store, "two")
    conv_id = _conversation_with_a_turn(store, "three")

    _label(store, conv_id, "", "a summary")

    assert _topic(store, conv_id) == ""


def test_a_whitespace_only_topic_is_treated_the_same_way(store):
    conv_id = _conversation_with_a_turn(store, "one")
    _label(store, conv_id, "   ", "a summary")
    assert _topic(store, conv_id) == ""


def test_no_stored_topic_is_ever_a_bare_numeric_suffix(store):
    """The failure mode this whole exemption exists for, stated directly."""
    ids = [_conversation_with_a_turn(store, f"turn {i}") for i in range(5)]
    for conv_id in ids:
        _label(store, conv_id, "", "s")

    for conv_id in ids:
        topic = _topic(store, conv_id)
        assert not topic.strip().isdigit(), f"stored topic is a bare suffix: {topic!r}"


def test_a_blank_topic_leaves_the_conversation_retryable(store):
    """Blank is the sentinel, so a later real topic must land unsuffixed."""
    conv_id = _conversation_with_a_turn(store, "one")
    _label(store, conv_id, "", "s1")
    assert _topic(store, conv_id) == ""

    _label(store, conv_id, "A Real Title", "s2")
    assert _topic(store, conv_id) == "A Real Title"


def test_the_summary_is_still_written_when_the_topic_is_blank(store):
    """A summary is useful without a title, and storing it does not disturb the
    blank topic that preserves the retry."""
    conv_id = _conversation_with_a_turn(store, "one")
    _label(store, conv_id, "", "a summary worth keeping")

    rows = store.list_conversations(CHANNEL)
    assert rows[0]["summary"] == "a summary worth keeping"
    assert _topic(store, conv_id) == ""


def test_a_blank_topic_does_not_destroy_a_title_already_stored(store):
    """One failed generation must not cost a conversation its title."""
    conv_id = _conversation_with_a_turn(store, "one")
    _label(store, conv_id, "Good Title", "s1")

    _label(store, conv_id, "   ", "s2")

    assert _topic(store, conv_id) == "Good Title"


# ---------------------------------------------------------------------------
# Ruling I9: the write reports the topic it actually stored, so a caller
# mirroring or logging the label propagates the SUFFIXED one, never its own
# candidate.
# ---------------------------------------------------------------------------

def test_the_write_returns_the_stored_suffixed_topic(store):
    first = _conversation_with_a_turn(store, "order one")
    second = _conversation_with_a_turn(store, "order two")

    assert _label(store, first, "Order Status", "s1") == "Order Status"
    assert _label(store, second, "Order Status", "s2") == "Order Status 1"
    assert _topic(store, second) == "Order Status 1"


def test_the_write_returns_the_preserved_topic_on_a_blank_generation(store):
    conv_id = _conversation_with_a_turn(store, "order one")
    _label(store, conv_id, "Good Title", "s1")

    # A failed (blank) generation keeps the stored title — and reports it, so a
    # caller re-asserts the real title rather than clearing it.
    assert _label(store, conv_id, "   ", "s2") == "Good Title"


def test_the_write_returns_blank_for_a_conversation_that_has_no_title_yet(store):
    conv_id = _conversation_with_a_turn(store, "order one")
    assert _label(store, conv_id, "", "s") == ""
