"""Integration tests for how ConversationStore assigns conversation topics.

Topics are the user-facing handle on a stored conversation: ``/activate_conversation``
looks a conversation up by topic, so the topic must be unique per channel *and*
stable across rewrites. Those two requirements pull against each other, and the
collision helper is where they meet:

* **Uniqueness** — two different conversations on one channel must not end up
  sharing a topic, so the second one gets a numeric suffix. Matching is case- and
  whitespace-insensitive, because lookup normalizes the same way.
* **Stability** — rewriting a conversation's *own* topic, which happens whenever a
  topic is regenerated for unchanged content, must be idempotent. Counting the
  record's own stored topic as a collision made the topic oscillate between
  ``'T'`` and ``'T 1'`` on every refresh (fix-dzs.2).

Both topic-writing entry points are covered: ``update_conversation_topic_summary``
(the finalize/rotate path) and ``save_conversation`` with an explicit
``conversation_id`` (which reuses the same helper and had the same defect).

Everything runs against a real SQLite-backed store under ``tmp_path``, per the
repo's integration-tests-only rule. No LLM is involved: topics are passed in
directly rather than generated.

Blank topics are covered as well. An empty or whitespace-only topic is a *failed
generation*, not a title, and has to be stored as exactly ``""`` — the sentinel
that means "no successful title yet, retry on the next eligible trigger". It used
to collide with every *other* unlabeled conversation (they all store ``topic ==
""``) and be stored as ``' 1'``, which is neither a title nor the sentinel, so
such a conversation counted as labeled forever and a picker rendered it as ``' 1'``
(fix-dzs.3).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pytest

from fastworkflow.run_fastapi_mcp.conversation_store import ConversationStore
from fastworkflow.utils.logging import logger as fastworkflow_logger


def _conversation_with_a_turn(store: ConversationStore, summary: str) -> int:
    """A durable conversation with one turn and the placeholder (blank) topic."""
    conv_id = store.reserve_next_conversation_id()
    store.append_conversation_turns(
        conv_id,
        [{"conversation summary": summary, "conversation_traces": None, "feedback": None}],
    )
    return conv_id


def _topic(store: ConversationStore, conv_id: int) -> str:
    return store.get_conversation(conv_id)["topic"]


# ---------------------------------------------------------------------------
# Stability: a conversation cannot collide with itself
# ---------------------------------------------------------------------------

def test_rewriting_a_conversations_own_topic_is_idempotent(tmp_path: Path):
    """Regenerating an unchanged topic must not add a suffix — nor take one off.

    The oscillation is the giveaway: counting the record's own topic as a
    collision suffixes the second write, and then the third write no longer
    matches (the stored topic is now ``'T 1'``) so the suffix disappears again.
    A single write is therefore not enough to catch this; three are.
    """
    store = ConversationStore("stability", str(tmp_path))
    conv_id = _conversation_with_a_turn(store, "where is my order")

    store.update_conversation_topic_summary(conv_id, "Order Status Question", "first")
    assert _topic(store, conv_id) == "Order Status Question"

    store.update_conversation_topic_summary(conv_id, "Order Status Question", "second")
    assert _topic(store, conv_id) == "Order Status Question"

    store.update_conversation_topic_summary(conv_id, "Order Status Question", "third")
    assert _topic(store, conv_id) == "Order Status Question"


def test_rewriting_own_topic_in_a_different_case_is_also_unsuffixed(tmp_path: Path):
    """Exclusion is by conversation id, not by string equality with the old topic.

    An LLM re-labelling the same conversation can return the same words in
    different case or spacing. That is still the record's own topic, so it must
    not be treated as somebody else's.
    """
    store = ConversationStore("recase", str(tmp_path))
    conv_id = _conversation_with_a_turn(store, "where is my order")

    store.update_conversation_topic_summary(conv_id, "Order Status Question", "first")
    store.update_conversation_topic_summary(conv_id, "  order STATUS question ", "second")

    assert _topic(store, conv_id) == "  order STATUS question "


def test_a_legitimate_suffix_does_not_drift_on_rewrite(tmp_path: Path):
    """A conversation that genuinely needed a suffix keeps the same one.

    Self-collision made this climb — ``'T 1'`` rewritten as ``'T'`` collided with
    both ``'T'`` and its own ``'T 1'`` and became ``'T 2'``.
    """
    store = ConversationStore("drift", str(tmp_path))
    first = _conversation_with_a_turn(store, "a")
    second = _conversation_with_a_turn(store, "b")

    store.update_conversation_topic_summary(first, "Support Topic", "s1")
    store.update_conversation_topic_summary(second, "Support Topic", "s2")
    assert _topic(store, second) == "Support Topic 1"

    store.update_conversation_topic_summary(second, "Support Topic", "s2 again")
    assert _topic(store, second) == "Support Topic 1"
    assert _topic(store, first) == "Support Topic"


# ---------------------------------------------------------------------------
# Uniqueness: the feature the helper exists for
# ---------------------------------------------------------------------------

def test_a_different_conversation_with_the_same_topic_still_gets_suffixed(tmp_path: Path):
    """The regression guard: excluding self must not disable collision handling."""
    store = ConversationStore("collide", str(tmp_path))
    first = _conversation_with_a_turn(store, "a")
    second = _conversation_with_a_turn(store, "b")
    third = _conversation_with_a_turn(store, "c")

    store.update_conversation_topic_summary(first, "Order Status Question", "s1")
    store.update_conversation_topic_summary(second, "Order Status Question", "s2")
    store.update_conversation_topic_summary(third, "Order Status Question", "s3")

    assert _topic(store, first) == "Order Status Question"
    assert _topic(store, second) == "Order Status Question 1"
    assert _topic(store, third) == "Order Status Question 2"


def test_collision_detection_is_case_insensitive(tmp_path: Path):
    """Lookup normalizes case, so a case-only difference is still a collision."""
    store = ConversationStore("case", str(tmp_path))
    first = _conversation_with_a_turn(store, "a")
    second = _conversation_with_a_turn(store, "b")

    store.update_conversation_topic_summary(first, "Order Status Question", "s1")
    store.update_conversation_topic_summary(second, "order status question", "s2")

    assert _topic(store, second) == "order status question 1"


def test_collision_detection_is_whitespace_insensitive(tmp_path: Path):
    """Leading/trailing whitespace is stripped for comparison only.

    The suffix is appended to the candidate verbatim, so the padding survives in
    the stored topic; comparison is what ignores it.
    """
    store = ConversationStore("space", str(tmp_path))
    first = _conversation_with_a_turn(store, "a")
    second = _conversation_with_a_turn(store, "b")

    padded = "  Order Status Question  "
    store.update_conversation_topic_summary(first, "Order Status Question", "s1")
    store.update_conversation_topic_summary(second, padded, "s2")

    assert _topic(store, second) == f"{padded} 1"
    assert store.get_conversation_by_topic(padded)[0] == first


# ---------------------------------------------------------------------------
# save_conversation shares the helper, so it shares both behaviours
# ---------------------------------------------------------------------------

def test_save_conversation_with_an_explicit_id_does_not_self_collide(tmp_path: Path):
    """Re-saving a named conversation with its own topic keeps that topic."""
    store = ConversationStore("explicit", str(tmp_path))
    conv_id = store.reserve_next_conversation_id()
    turns = [{"conversation summary": "a", "conversation_traces": None, "feedback": None}]

    store.save_conversation("Refund Request", "s", turns, conversation_id=conv_id)
    assert _topic(store, conv_id) == "Refund Request"

    store.save_conversation("Refund Request", "s", turns, conversation_id=conv_id)
    assert _topic(store, conv_id) == "Refund Request"

    store.save_conversation("Refund Request", "s", turns, conversation_id=conv_id)
    assert _topic(store, conv_id) == "Refund Request"


def test_save_conversation_with_an_explicit_id_still_suffixes_a_real_collision(tmp_path: Path):
    """A named conversation taking another conversation's topic is still a collision."""
    store = ConversationStore("explicit_collide", str(tmp_path))
    turns = [{"conversation summary": "a", "conversation_traces": None, "feedback": None}]

    first = store.reserve_next_conversation_id()
    store.save_conversation("Refund Request", "s1", turns, conversation_id=first)

    second = store.reserve_next_conversation_id()
    store.save_conversation("Refund Request", "s2", turns, conversation_id=second)

    assert _topic(store, first) == "Refund Request"
    assert _topic(store, second) == "Refund Request 1"


def test_save_conversation_without_an_id_still_suffixes_a_real_collision(tmp_path: Path):
    """The id-allocating path is unchanged: a fresh id has nothing to exclude."""
    store = ConversationStore("allocating", str(tmp_path))
    turns = [{"conversation summary": "a", "conversation_traces": None, "feedback": None}]

    first = store.save_conversation("Widget Order", "s1", turns)
    second = store.save_conversation("Widget Order", "s2", turns)

    assert first != second
    assert _topic(store, first) == "Widget Order"
    assert _topic(store, second) == "Widget Order 1"


# ---------------------------------------------------------------------------
# A blank topic is a failed generation, not a title to be uniquified
# ---------------------------------------------------------------------------

@pytest.fixture
def fastworkflow_logs(caplog):
    """caplog, wired to the logger that actually emits the blank-topic warning.

    ``fastworkflow.utils.logging`` sets ``propagate = False``, so its records never
    reach the root handler pytest installs and a plain ``caplog`` assertion would
    see nothing at all. Attaching caplog's own handler to that logger is enough.
    """
    caplog.set_level(logging.WARNING, logger="fastWorkflow")
    fastworkflow_logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        fastworkflow_logger.removeHandler(caplog.handler)


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def test_a_blank_topic_stays_blank_when_other_unlabeled_conversations_exist(
    tmp_path: Path, fastworkflow_logs
):
    """The fix-dzs.3 bug: an empty topic collided with every *other* blank one.

    A second unlabeled conversation is what makes this reproduce. Every unlabeled
    conversation stores ``topic == ""``, so an empty candidate matched all of them
    and came back suffixed as ``' 1'``. With only one conversation on the channel,
    fix-dzs.2's self-exclusion already hides the defect, which is why that weaker
    arrangement is not the test.

    The warning is part of the contract: storing ``""`` silently would leave an
    operator with no signal that generation is failing. It names the conversation
    and the channel and nothing else — conversation content must not reach a log.
    """
    store = ConversationStore("blank", str(tmp_path))
    target = _conversation_with_a_turn(store, "where is my order")
    _conversation_with_a_turn(store, "still unlabeled")  # the collision partner

    store.update_conversation_topic_summary(target, "", "a summary anyway")

    assert _topic(store, target) == ""

    warnings = _warnings(fastworkflow_logs)
    assert len(warnings) == 1
    assert str(target) in warnings[0]
    assert "a summary anyway" not in warnings[0]


def test_a_whitespace_only_topic_is_treated_the_same_way(tmp_path: Path):
    """``'   '`` is not a title either, and must not survive as stored whitespace.

    Stored whitespace would be a third state all over again: unusable as a title,
    and not equal to ``""``, so a sentinel check for blank would skip right past it.
    """
    store = ConversationStore("whitespace", str(tmp_path))
    target = _conversation_with_a_turn(store, "a")
    _conversation_with_a_turn(store, "b")

    store.update_conversation_topic_summary(target, "   ", "s")

    assert _topic(store, target) == ""


def test_no_stored_topic_is_ever_a_bare_numeric_suffix(tmp_path: Path):
    """Across a mix of blank and real topics, nothing is stored as ``' 1'``.

    Every conversation on the channel is checked, not just the one written, since
    the suffix was produced by comparing a candidate against all the *others*.
    """
    store = ConversationStore("suffix", str(tmp_path))
    ids = [_conversation_with_a_turn(store, s) for s in ("a", "b", "c", "d")]

    store.update_conversation_topic_summary(ids[0], "", "s0")
    store.update_conversation_topic_summary(ids[1], "  ", "s1")
    store.update_conversation_topic_summary(ids[2], "Order Status Question", "s2")
    store.update_conversation_topic_summary(ids[3], "Order Status Question", "s3")

    stored = [_topic(store, i) for i in ids]
    assert stored == ["", "", "Order Status Question", "Order Status Question 1"]
    for topic in stored:
        assert not topic.strip().isdigit(), f"stored topic is a bare suffix: {topic!r}"


def test_a_blank_topic_leaves_the_conversation_retryable(tmp_path: Path):
    """Blank is the retry sentinel, so a later real topic must land unsuffixed.

    ``' 1'`` broke this twice over: the conversation no longer looked unlabeled to
    whatever decides to regenerate, and the stored junk was itself what the next
    write had to displace.
    """
    store = ConversationStore("retry", str(tmp_path))
    target = _conversation_with_a_turn(store, "a")
    _conversation_with_a_turn(store, "b")

    store.update_conversation_topic_summary(target, "", "first attempt")
    assert _topic(store, target) == ""

    store.update_conversation_topic_summary(target, "Order Status Question", "second attempt")

    assert _topic(store, target) == "Order Status Question"
    assert store.get_conversation_by_topic("Order Status Question")[0] == target


def test_the_summary_is_still_written_when_the_topic_is_blank(tmp_path: Path):
    """Chosen policy: a failed title does not throw away a usable summary.

    A summary stands on its own, and writing it does not disturb the blank topic
    that preserves the retry. ``updated_at`` moves with it, because the record did
    change and that field is what orders ``list_conversations()``.
    """
    store = ConversationStore("summary", str(tmp_path))
    target = _conversation_with_a_turn(store, "a")
    _conversation_with_a_turn(store, "b")

    before = store.get_conversation(target)["updated_at"]
    time.sleep(0.002)  # updated_at has millisecond resolution

    store.update_conversation_topic_summary(target, "", "the conversation was about refunds")

    conv = store.get_conversation(target)
    assert conv["topic"] == ""
    assert conv["summary"] == "the conversation was about refunds"
    assert conv["updated_at"] > before


def test_a_blank_topic_does_not_destroy_a_title_already_stored(tmp_path: Path):
    """A failed regeneration must not demote a conversation that was labeled.

    Overwriting the field with ``""`` would satisfy "leaves the stored topic
    blank" for a fresh conversation while quietly losing the title of one already
    labeled, and would push a conversation that needs no work back into the retry
    pool. Declining to write the field at all does both jobs at once.
    """
    store = ConversationStore("keep", str(tmp_path))
    target = _conversation_with_a_turn(store, "a")
    _conversation_with_a_turn(store, "b")

    store.update_conversation_topic_summary(target, "Order Status Question", "s1")
    store.update_conversation_topic_summary(target, "", "s2")

    assert _topic(store, target) == "Order Status Question"
    assert store.get_conversation(target)["summary"] == "s2"


def test_the_other_topic_writer_inherits_the_blank_exemption(tmp_path: Path):
    """``save_conversation`` shares the helper, so it cannot suffix a blank either.

    The policy above lives at the finalize call site; this is the structural half
    of the fix. A caller reaching the store by another route still cannot produce
    a suffixed blank topic — only ``""``.
    """
    store = ConversationStore("other_writer", str(tmp_path))
    turns = [{"conversation summary": "a", "conversation_traces": None, "feedback": None}]
    _conversation_with_a_turn(store, "already unlabeled")

    conv_id = store.save_conversation("", "s", turns)

    assert _topic(store, conv_id) == ""
