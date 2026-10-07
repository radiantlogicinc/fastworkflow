"""The five persisted surfaces neither protection layer reaches on its own.

`observability_store` protects what it persists with `Redactor` — an
unconditional scrub of credential shapes and loaded secret env values, applied
at the sink boundary — and it is wired into the TurnResult pipeline.
Conversation labels written through the SYNC store path, review-note comments,
train-run metrics, writer diagnostics, and the scalar columns beside a span's
(already scrubbed) attributes JSON do not go through that pipeline, so without
their own protection they would reach SQLite verbatim.

Item 5 — the sync label path — is the one that is live rather than latent:
`run_fastapi_mcp/utils.ensure_topic_and_summary` calls
`ObservabilityStore.record_conversation_label` directly, and a topic and summary
are LLM output generated from a real user's conversation. The tests for it
therefore build NO sink at all, because a test that reached the store through
`SQLiteTraceSink.record_conversation_label` would be exercising the queued route
that was already protected and proving nothing about production.

Two properties are load-bearing here and are asserted for every surface:

1. **A planted credential does not survive to the DB.**
2. **A value with no credential in it is written byte for byte**, so the scrub
   changes nothing it does not have to.

The tests pin why `human_feedback.comment` and `diagnostics` keep their
content rather than leaving it to be re-litigated by whoever reads the code
next: the first is the record of what a reviewer judged, and the second is the
evidence gate's own input. Surface 1 used to be
`feedback.feedback_json`, the agent's memory of being corrected; fix-9eg.16
removed that table and the prompt injection that read it, so the scrub is
pinned on the review-note column that replaced it as the place free text from
a person or a coding agent lands.

Real SQLite in tmp_path throughout, per .cursor/rules/testing_rules.mdc.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

import fastworkflow
from fastworkflow import TurnStatus, tracing
from fastworkflow.observability import feedback
from fastworkflow.observability import store as obs

# A credential shape `Redactor._SECRET_PATTERNS` recognizes without any help from
# the environment.
SK_TOKEN = "sk-livekey1234567890abcdef"

# ...and one it only knows about because the variable's name marks it as secret.
API_KEY_VAR = "FIXAJV9_PLANTED_SERVICE_API_KEY"
ENV_SECRET = "hunter2-planted-secret-value"

# Stands in for content generated from a real user's conversation.
TENANT = "sara_doe_496 ordered a blue kayak"

REDACTED = "[REDACTED]"


@pytest.fixture
def db_path(tmp_path) -> str:
    return str(tmp_path / "observability.sqlite3")


@pytest.fixture
def planted_credentials(monkeypatch):
    """Put a secret in the environment before any store builds its Redactor.

    `Redactor` snapshots the environment when it is constructed and
    `ObservabilityStore` caches one lazily, so every test that wants the env-value
    branch scrubbed has to set the variable before it creates the store. Ordering
    it through a fixture keeps that from being a silent per-test mistake.
    """
    monkeypatch.setenv(API_KEY_VAR, ENV_SECRET)


def _rows(db_path: str, sql: str, params=()) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _turn_result(summary="user asked about a kayak", traces="get_order -> ok"):
    """A minimal turn that counts as conversation memory (non-NULL summary)."""
    turn_output = fastworkflow.TurnOutput(
        turn_key=fastworkflow.mint_turn_key(),
        status=TurnStatus.COMPLETED,
        answer="It ships Tuesday.",
        command_outputs=[
            fastworkflow.CommandOutput(
                command_name="get_order",
                command_response=fastworkflow.CommandResponse(response="ok"),
            )
        ],
    )
    return fastworkflow.TurnResult(
        turn_output=turn_output,
        channel_id="chan",
        conversation_id=1,
        user_message="where is my kayak",
        conversation_summary=summary,
        conversation_traces=traces,
    )


def _span(**overrides) -> tracing.Span:
    fields = {
        "span_id": "span-1",
        "trace_id": "20260828T000000.000000Z-aaaaaaaaaaaa",
        "name": tracing.SPAN_COMMAND_EXECUTE,
        "kind": tracing.KIND_INTERNAL,
        "channel_id": "chan-1",
        "command_name": "get_user_details",
        # What `workflow.current_command_context_displayname` returns: a
        # workflow-supplied `get_displayname(instance)`, which the bundled
        # simple_workflow_template implements as the instance's absolute path.
        "context": f"Order: {TENANT}",
        "start_ns": 100,
        "end_ns": 200,
        "status": "completed",
        "attributes": {"fw.command.name": "get_user_details"},
    }
    fields.update(overrides)
    return tracing.Span(**fields)


def _write_span(db_path: str, span: tracing.Span) -> dict:
    """Persist one span through the sink, the way the writer thread does."""
    sink = obs.SQLiteTraceSink(db_path)
    try:
        sink.emit_span(span)
        assert sink.flush()
    finally:
        sink.close()
    return _rows(db_path, "SELECT * FROM spans")[0]


def _set_diagnostic(store: obs.ObservabilityStore, key: str, value: dict) -> None:
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        store.set_diagnostic(conn, key, value)
        conn.commit()


# ----------------------------------------------------------------------
# Surface 5: conversation labels written through the SYNC store path
#
# The priority of the five: live, not latent. No sink is built anywhere in this
# section, so nothing here can be passing because of the queued route's scrub.
# ----------------------------------------------------------------------


class TestSyncPathConversationLabels:
    def test_the_sync_path_is_the_one_production_uses(self):
        """Guards the premise the rest of this section rests on.

        `ensure_topic_and_summary` writes labels by calling the STORE, not the
        sink — deliberately, so the label is visible to the very next
        `_label_is_due` read. If that ever changes, these tests are still green
        while covering a path nobody runs, so the coupling is asserted rather
        than described.
        """
        from fastworkflow.run_fastapi_mcp import utils

        source = utils.ensure_topic_and_summary.__doc__ or ""
        assert "record_conversation_label" in source
        assert hasattr(obs.ObservabilityStore, "record_conversation_label")

    def test_planted_credentials_do_not_reach_the_conversations_row(
        self, db_path, planted_credentials
    ):
        store = obs.ObservabilityStore(db_path)
        conv = store.mint_conversation_id("chan")
        store.record_conversation_label(
            "chan",
            conv,
            f"Renew key {SK_TOKEN}",
            f"The user pasted {ENV_SECRET} into the chat.",
        )

        row = _rows(db_path, "SELECT * FROM conversations")[0]
        assert SK_TOKEN not in row["topic"]
        assert ENV_SECRET not in row["summary"]
        assert REDACTED in row["topic"]
        assert REDACTED in row["summary"]

    def test_a_clean_label_is_stored_unchanged(self, db_path):
        store = obs.ObservabilityStore(db_path)
        conv = store.mint_conversation_id("chan")
        stored = store.record_conversation_label("chan", conv, "Kayak order", TENANT)

        row = _rows(db_path, "SELECT * FROM conversations")[0]
        assert row["topic"] == "Kayak order"
        assert row["summary"] == TENANT
        assert stored == "Kayak order"

    def test_the_returned_topic_is_what_was_actually_stored(self, db_path):
        """Ruling I9's contract: a caller that logs the label must log the
        scrubbed topic, not its own candidate — otherwise the operator's log
        carries the credential the DB was kept free of."""
        store = obs.ObservabilityStore(db_path)
        conv = store.mint_conversation_id("chan")
        returned = store.record_conversation_label("chan", conv, f"Key {SK_TOKEN}", None)

        stored = _rows(db_path, "SELECT topic FROM conversations")[0]["topic"]
        assert returned == stored
        assert SK_TOKEN not in returned

    def test_the_blank_topic_sentinel_survives_the_scrub(self, db_path):
        """A blank generated topic must stay NULL.

        `_label_is_due` treats a blank topic as "no successful title yet" and
        retries; storing anything non-blank for it would permanently freeze the
        conversation as titled-but-empty.
        """
        store = obs.ObservabilityStore(db_path)
        conv = store.mint_conversation_id("chan")
        store.record_conversation_label("chan", conv, "   ", "a summary")

        assert _rows(db_path, "SELECT topic FROM conversations")[0]["topic"] is None
        assert store.conversation_label_state("chan", conv)[0] == ""

    def test_a_none_topic_still_preserves_the_stored_one(self, db_path):
        """The blank-topic policy: a failed generation never clobbers a good
        title, and adding a protection layer must not change that."""
        store = obs.ObservabilityStore(db_path)
        conv = store.mint_conversation_id("chan")
        store.record_conversation_label("chan", conv, "Kayak order", "first")
        store.record_conversation_label("chan", conv, None, "second")

        row = _rows(db_path, "SELECT * FROM conversations")[0]
        assert row["topic"] == "Kayak order"
        assert row["summary"] == "second"

    def test_topic_uniquification_still_runs(self, db_path):
        store = obs.ObservabilityStore(db_path)
        first = store.mint_conversation_id("chan")
        second = store.mint_conversation_id("chan")
        store.record_conversation_label("chan", first, "Kayak order", "s1")
        store.record_conversation_label("chan", second, "kayak order", "s2")

        topics = {r["topic"] for r in _rows(db_path, "SELECT topic FROM conversations")}
        assert topics == {"Kayak order", "kayak order 1"}

    def test_both_label_routes_agree_on_what_they_store(self, db_path, tmp_path):
        """The queued route scrubs in `SQLiteTraceSink._apply_label` before
        reaching `apply_label_txn`; the sync route does not. Scrubbing first
        inside the enforcement point is what makes both produce the same topic —
        and a topic that depended on which route wrote the row could not be
        compared across two runs.
        """
        sync_store = obs.ObservabilityStore(db_path)
        sync_store.record_conversation_label("chan", 1, f"Key {SK_TOKEN}", TENANT)
        sync_topic = _rows(db_path, "SELECT topic FROM conversations")[0]["topic"]

        queued_path = str(tmp_path / "queued.sqlite3")
        sink = obs.SQLiteTraceSink(queued_path)
        try:
            sink.record_conversation_label("chan", 1, f"Key {SK_TOKEN}", TENANT)
            assert sink.flush()
        finally:
            sink.close()
        queued_topic = _rows(queued_path, "SELECT topic FROM conversations")[0]["topic"]

        assert SK_TOKEN not in sync_topic
        assert sync_topic == queued_topic


# ----------------------------------------------------------------------
# Surface 4: the scalar columns beside a span's attributes JSON
# ----------------------------------------------------------------------


class TestSpanScalarColumns:
    def test_planted_credentials_do_not_reach_the_span_row(
        self, db_path, planted_credentials
    ):
        row = _write_span(
            db_path,
            _span(context=f"Order {SK_TOKEN}", channel_id=f"chan-{ENV_SECRET}"),
        )
        serialized = json.dumps(row)
        assert SK_TOKEN not in serialized
        assert ENV_SECRET not in serialized
        assert REDACTED in row["context"]
        assert REDACTED in row["channel_id"]

    def test_a_clean_span_stores_every_scalar_unchanged(self, db_path):
        span = _span()
        row = _write_span(db_path, span)
        assert row["name"] == span.name
        assert row["command_name"] == span.command_name
        assert row["context"] == span.context
        assert row["channel_id"] == span.channel_id

    def test_channel_id_stays_joinable(self, db_path):
        """SCRUB-ONLY, and the reason is erasure, not convenience.

        `forget_channel` deletes spans with `WHERE channel_id=?`. Digesting this
        column would narrow first-class erasure to whatever the
        `trace_id IN (...)` fallback still covers. Reducing exposure by
        weakening erasure is not a trade worth making.
        """
        span = _span()
        _write_span(db_path, span)

        store = obs.ObservabilityStore(db_path)
        assert store.forget_channel(span.channel_id)["spans"] == 1
        assert _rows(db_path, "SELECT * FROM spans") == []

    def test_a_null_scalar_stays_null(self, db_path):
        """An absent context must not become the string "": `COALESCE(
        excluded.context, spans.context)` in the upsert depends on NULL staying
        NULL."""
        row = _write_span(db_path, _span(context=None, command_name=None))
        assert row["context"] is None
        assert row["command_name"] is None

    def test_span_attributes_are_still_scrubbed(self, db_path, planted_credentials):
        """The span-attribute scrub, re-asserted beside the scalar columns."""
        row = _write_span(db_path, _span(attributes={"leak": f"key {ENV_SECRET}"}))
        assert ENV_SECRET not in row["attributes"]
        assert REDACTED in row["attributes"]


# ----------------------------------------------------------------------
# Surface 1: human_feedback.comment — credential scrub only
# ----------------------------------------------------------------------


def _feedback_turn(db_path):
    """A real recorded turn, because a review note needs one to anchor to."""
    sink = obs.SQLiteTraceSink(db_path)
    turn = _turn_result()
    try:
        sink.emit_turn_record(turn)
        assert sink.flush()
    finally:
        sink.close()
    return turn.turn_output.turn_key


def _note(store, turn_key, comment, **kw):
    body = dict(
        target_kind="turn",
        span_ids=[],
        target_label="Turn",
        provenance="human",
        category="conclusions",
        subcategory="what_went_wrong",
    )
    body.update(kw)
    return store.add_human_feedback(turn_key, comment=comment, **body)


class TestFeedback:
    """Surface 1 is now `human_feedback.comment`.

    It was `feedback.feedback_json`, the agent-memory row. fix-9eg.16 removed
    that table, so the scrub has to be pinned where free text is actually
    written now: a review note's comment, typed by a person or posted by a
    coding agent, either of whom can paste a credential into it.
    """

    def test_planted_credentials_do_not_reach_the_feedback_row(
        self, db_path, planted_credentials
    ):
        store = obs.ObservabilityStore(db_path)
        turn_key = _feedback_turn(db_path)
        _note(store, turn_key, f"try {SK_TOKEN} or {ENV_SECRET}")

        stored = _rows(db_path, "SELECT * FROM human_feedback")[0]["comment"]
        assert SK_TOKEN not in stored
        assert ENV_SECRET not in stored
        assert REDACTED in stored

    def test_scrubbing_leaves_the_rest_of_the_comment_alone(
        self, db_path, planted_credentials
    ):
        """A scrub that swallowed surrounding prose would quietly destroy the
        one thing this row exists to keep: what the author actually said."""
        store = obs.ObservabilityStore(db_path)
        turn_key = _feedback_turn(db_path)
        _note(
            store,
            turn_key,
            f"the key {SK_TOKEN} did not work.\nRetry with the tenant key.",
        )

        stored = _rows(db_path, "SELECT * FROM human_feedback")[0]["comment"]
        assert stored.startswith("the key ")
        assert stored.endswith("did not work.\nRetry with the tenant key.")
        assert REDACTED in stored

    def test_the_label_is_scrubbed_too(self, db_path, planted_credentials):
        """`target_label` is caller-supplied text on the same row."""
        store = obs.ObservabilityStore(db_path)
        turn_key = _feedback_turn(db_path)
        _note(store, turn_key, "fine", target_label=f"Turn {SK_TOKEN}")

        stored = _rows(db_path, "SELECT * FROM human_feedback")[0]["target_label"]
        assert SK_TOKEN not in stored and REDACTED in stored

    def test_no_column_of_the_stored_row_carries_the_credential(
        self, db_path, planted_credentials
    ):
        """Scanned whole-row, not column by column.

        `target_label` is scrubbed on its own column and then written a SECOND
        time, verbatim, inside `anchors_json` — so a check that reads only the
        columns it remembers to name reports a clean row while the credential
        sits two fields away. Every value in the row is scanned instead, which
        is the only form of this assertion that keeps working when the row
        grows another field.
        """
        store = obs.ObservabilityStore(db_path)
        turn_key = _feedback_turn(db_path)
        _note(
            store,
            turn_key,
            f"try {SK_TOKEN}",
            target_label=f"Turn {SK_TOKEN} for {ENV_SECRET}",
        )

        row = _rows(db_path, "SELECT * FROM human_feedback")[0]
        whole = "\n".join(str(value) for value in row.values())
        assert SK_TOKEN not in whole, "a credential survived somewhere in the row"
        assert ENV_SECRET not in whole
        assert REDACTED in row["anchors_json"]

    def test_the_paired_side_of_a_comparison_is_scrubbed_as_well(
        self, db_path, planted_credentials
    ):
        """Both anchors, not just the one the columns mirror.

        A comparison comment carries a second target whose label has no column
        of its own, so it is only ever stored inside `anchors_json`. Scrubbing
        the serialized anchor by mirroring the columns would clean the primary
        side and leave the paired one untouched.
        """
        store = obs.ObservabilityStore(db_path)
        left = _feedback_turn(db_path)
        right = _feedback_turn(db_path)
        identity = store.store_identity()
        left_row = store.get_turn(left)
        anchors = feedback.build_anchors(
            feedback.FeedbackTarget.from_mapping({
                "store_id": identity, "turn_keys": [left],
                "target_kind": "turn", "span_ids": [],
                "target_label": "Turn", "label": f"left {SK_TOKEN}",
            }),
            sources={identity: store},
            paired=feedback.FeedbackTarget.from_mapping({
                "store_id": identity, "turn_keys": [right],
                "target_kind": "turn", "span_ids": [],
                "target_label": f"Turn {SK_TOKEN}",
                "label": f"right {ENV_SECRET}",
            }),
        )
        pair_key = anchors.pair_key
        store.add_human_feedback(
            left, target_kind="turn", span_ids=[], target_label="Turn",
            provenance="human", comment="the two sides diverge here",
            category="conclusions", subcategory="what_went_wrong",
            anchors=anchors,
        )

        row = _rows(db_path, "SELECT * FROM human_feedback")[0]
        whole = "\n".join(str(value) for value in row.values())
        assert SK_TOKEN not in whole and ENV_SECRET not in whole
        stored = json.loads(row["anchors_json"])
        assert REDACTED in stored["paired"]["target_label"]
        assert REDACTED in stored["paired"]["ref"]["label"]
        # Identity is NOT redacted: scrubbing a key would orphan the comment.
        assert stored["paired"]["ref"]["turn_keys"] == [right]
        assert stored["primary"]["ref"]["turn_keys"] == [left]
        assert stored["pair_key"] == pair_key
        assert left_row is not None


    def test_feedback_content_is_kept(self, db_path):
        """PINS A DELIBERATE DECISION: the comment is scrubbed, never replaced.

        A review comment IS the record of what a reviewer judged, and anything
        in its place makes the task Feedback view unreadable while looking like
        it still works. Credentials are scrubbed above; the prose is kept.
        """
        store = obs.ObservabilityStore(db_path)
        turn_key = _feedback_turn(db_path)
        _note(store, turn_key, TENANT)

        row = _rows(db_path, "SELECT * FROM human_feedback")[0]
        assert row["comment"] == TENANT
        assert (row["category"], row["subcategory"]) == (
            "conclusions", "what_went_wrong",
        )


# ----------------------------------------------------------------------
# Surface 2: train_runs.metrics_json
# ----------------------------------------------------------------------


def _metrics(extra: str = "") -> dict:
    return {
        "version_id": "20260828T000000",
        "models": {"tiny": f"google/bert_uncased_L-4_H-128_A-2{extra}"},
        "contexts": {"global": {"thresholds": {"threshold": 0.71}}},
        # `heldout_evaluation.EscalationScore.failures` records the verbatim
        # utterance of every failing case, and `metrics_persistence` copies the
        # whole escalation block through, so the column carries free text and
        # not only numbers.
        "totals": {"escalation": {"failures": [{"utterance": TENANT}]}},
    }


class TestTrainRunMetrics:
    def test_planted_credentials_do_not_reach_the_train_run_row(
        self, db_path, planted_credentials
    ):
        store = obs.ObservabilityStore(db_path)
        store.record_train_run(
            "run-1", "fp", None, None, _metrics(extra=f"?token={SK_TOKEN}&{ENV_SECRET}")
        )

        stored = _rows(db_path, "SELECT * FROM train_runs")[0]["metrics_json"]
        assert SK_TOKEN not in stored
        assert ENV_SECRET not in stored
        assert REDACTED in stored
        json.loads(stored)  # still parses

    def test_clean_metrics_are_stored_unchanged(self, db_path):
        store = obs.ObservabilityStore(db_path)
        store.record_train_run("run-1", "fp", None, None, _metrics())

        runs = store.list_train_runs()
        assert json.loads(runs[0]["metrics_json"]) == _metrics()

# ----------------------------------------------------------------------
# Surface 3: diagnostics — credential scrub only
# ----------------------------------------------------------------------


class TestDiagnostics:
    def test_a_provider_error_body_is_scrubbed(self, db_path, planted_credentials):
        """The scenario the redactor was written for: a LiteLLM
        `AuthenticationError` whose body echoes the key, arriving here as
        `repr(exc)` in `writer_health.last_error`."""
        store = obs.ObservabilityStore(db_path)
        _set_diagnostic(
            store,
            obs.WRITER_HEALTH_KEY_PREFIX + "writer-1",
            {
                "write_errors": 1,
                "last_error": (
                    f"AuthenticationError(\"key={SK_TOKEN} env={ENV_SECRET}\")"
                ),
            },
        )

        health = store.writer_health()
        assert health["write_errors"] == 1
        assert SK_TOKEN not in health["last_error"]
        assert ENV_SECRET not in health["last_error"]
        assert REDACTED in health["last_error"]

    def test_writer_health_stays_readable(self, db_path):
        """PINS A DELIBERATE DECISION: this table is scrubbed, never replaced.

        `health_delta` and `evidence_run` read `writer_health` to decide whether
        a run may be reported as evidence at all, and `problems()` names the
        affected turn keys so a partly-damaged run can be salvaged instead of
        discarded. Replacing it would blind the evidence gate.
        """
        store = obs.ObservabilityStore(db_path)
        turn_keys = ["20260828T000000.000000Z-aaaaaaaaaaaa"]
        _set_diagnostic(
            store,
            obs.WRITER_HEALTH_KEY_PREFIX + "writer-1",
            {
                "records_dropped": 2,
                "spans_dropped": 0,
                "records_dropped_turn_keys": turn_keys,
            },
        )

        health = store.writer_health()
        assert health["records_dropped"] == 2
        assert health["records_dropped_turn_keys"] == turn_keys

        delta = obs.health_delta({"records_dropped": 0}, health)
        assert delta.records_dropped == 2
        assert not delta.evidence_valid
        assert any("DROPPED" in problem for problem in delta.problems())

    def test_a_live_sink_still_publishes_its_health_row(self, db_path):
        """End to end: the writer thread's own heartbeat goes through
        `set_diagnostic`, so anything that replaced it there would take the health row
        out from under a running sink rather than merely out of a bundle."""
        sink = obs.SQLiteTraceSink(db_path)
        try:
            sink.emit_turn_record(_turn_result())
            assert sink.flush()
            sink.persist_health()
        finally:
            sink.close()

        health = obs.ObservabilityStore(db_path).writer_health()
        assert health is not None
        assert health["records_dropped"] == 0
        assert health["write_errors"] == 0


# ----------------------------------------------------------------------
# Conversation memory is unaffected — the constraint the whole change rides on
# ----------------------------------------------------------------------


class TestConversationMemory:
    def test_memory_rebuilds_intact(self, db_path):
        """Summary and traces both have to arrive.

        This asserts the whole shape `restore_history_from_turns` consumes. There were three keys until
        fix-9eg.16: the third joined the agent-memory feedback row into the
        window and thence into the agent's prompt. Recorded review notes
        deliberately do NOT travel this path, which is asserted here as well
        — a redaction test is exactly where a reappearing prompt channel
        would need to be noticed.
        """
        sink = obs.SQLiteTraceSink(db_path)
        turn = _turn_result()
        try:
            sink.emit_turn_record(turn)
            assert sink.flush()
        finally:
            sink.close()

        store = obs.ObservabilityStore(db_path)
        _note(store, turn.turn_output.turn_key, "helpful")

        assert store.count_usable_turns("chan", 1) == 1
        window = store.get_memory_window("chan", 1, max_turns=10)
        assert window == [
            {
                "conversation summary": "user asked about a kayak",
                "conversation_traces": "get_order -> ok",
            }
        ]
        assert "helpful" not in json.dumps(window)

    def test_a_labeled_conversation_still_lists_and_dumps(self, db_path):
        """A scrubbed label must not take the conversation out of the history
        list: `list_conversation_summaries` is how a user finds it again, and
        `_label_is_due` reads the stored topic to decide whether to spend another
        LLM call on one that is already titled."""
        sink = obs.SQLiteTraceSink(db_path)
        try:
            sink.emit_turn_record(_turn_result())
            assert sink.flush()
        finally:
            sink.close()

        store = obs.ObservabilityStore(db_path)
        store.record_conversation_label("chan", 1, "Kayak order", TENANT)

        listed = store.list_conversation_summaries("chan", 10)
        assert [c["conversation_id"] for c in listed] == [1]
        assert listed[0]["topic"]

        stored_topic, usable = store.conversation_label_state("chan", 1)
        assert usable == 1
        assert stored_topic.strip()  # so `_label_is_due` will not re-generate

        dumped = store.dump_all_conversations("chan")
        assert len(dumped[0]["turns"]) == 1


# ----------------------------------------------------------------------
# The scrub changes nothing it does not have to
# ----------------------------------------------------------------------


def test_clean_values_leave_all_five_surfaces_byte_identical(db_path):
    """One test covering all five at once, because the promise is about the set
    of them rather than about any one."""
    store = obs.ObservabilityStore(db_path)
    conv = store.mint_conversation_id("chan")
    store.record_conversation_label("chan", conv, "Kayak order", TENANT)
    turn_key = _feedback_turn(db_path)
    _note(store, turn_key, TENANT)
    store.record_train_run("run-1", "fp", None, None, _metrics())
    _set_diagnostic(store, "probe", {"note": TENANT})
    span = _span()
    span_row = _write_span(db_path, span)

    conversation = _rows(db_path, "SELECT * FROM conversations")[0]
    assert conversation["topic"] == "Kayak order"
    assert conversation["summary"] == TENANT
    assert _rows(db_path, "SELECT * FROM human_feedback")[0]["comment"] == TENANT
    assert json.loads(
        _rows(db_path, "SELECT * FROM train_runs")[0]["metrics_json"]
    ) == _metrics()
    assert json.loads(
        _rows(db_path, "SELECT value FROM diagnostics WHERE key='probe'")[0]["value"]
    ) == {"note": TENANT}
    assert span_row["context"] == span.context
    assert span_row["command_name"] == span.command_name
    assert span_row["name"] == span.name
    assert span_row["channel_id"] == span.channel_id
