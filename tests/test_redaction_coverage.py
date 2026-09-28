"""The five persisted surfaces neither protection layer reaches on its own.

`observability_store` has two independent protections, and between them they
can miss five write paths:

* `Redactor` — an unconditional, profile-independent scrub of credential shapes
  and loaded secret env values, applied at the sink boundary.
* `CapturePolicy` — a per-field classification layer whose `debug` default is
  inert and whose `evidence` profile is default-deny.

Both are wired into the TurnResult pipeline. Conversation labels written through
the SYNC store path, feedback, train-run metrics, writer diagnostics, and the
scalar columns beside a span's (already scrubbed) attributes JSON do not go
through that pipeline, so without their own protection they reach SQLite
verbatim under every profile.

Item 5 — the sync label path — is the one that is live rather than latent:
`run_fastapi_mcp/utils.ensure_topic_and_summary` calls
`ObservabilityStore.record_conversation_label` directly, and a topic and summary
are LLM output generated from a real user's conversation. The tests for it
therefore build NO sink at all, because a test that reached the store through
`SQLiteTraceSink.record_conversation_label` would be exercising the queued route
that was already protected and proving nothing about production.

Three properties are load-bearing here and are asserted for every surface:

1. **A planted credential does not survive to the DB**, under either profile.
2. **Withholding leaves a badge, never silence**: a viewer must be able to say
   "a value was here, this is its class, size and digest".
3. **The `debug` profile writes the same bytes it always did**, so turning
   capture policy on changes nothing for existing debug stores.

Two of the five are deliberately scrub-only, and the tests pin those decisions
rather than leaving them to be re-litigated by whoever reads the code next:
`feedback_json` is the agent's memory of being corrected, and `diagnostics` is
the evidence gate's own input.

Real SQLite in tmp_path throughout, per .cursor/rules/testing_rules.mdc.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

import fastworkflow
from fastworkflow import TurnStatus, tracing
from fastworkflow.observability import store as obs
from fastworkflow.observability.capture_policy import (
    CaptureFieldPolicy,
    evidence_policy,
    is_capture_envelope,
)

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


@pytest.fixture
def evidence_profile(monkeypatch):
    """Run the store under the default-deny profile."""
    monkeypatch.setenv(obs.CAPTURE_PROFILE_VAR, "evidence")


def _rows(db_path: str, sql: str, params=()) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _badge(stored: str) -> dict:
    """The policy envelope a withheld TEXT column now holds.

    Asserts the four things §12.0 delta 3 requires a viewer to be able to show, so
    every caller of this helper gets the "degrades, does not go dark" check for
    free rather than choosing which parts of it to remember.
    """
    assert isinstance(stored, str), f"not bindable as TEXT: {stored!r}"
    envelope = json.loads(stored)
    assert is_capture_envelope(envelope), envelope
    assert envelope["classification"], envelope
    assert envelope["original_bytes"] > 0, envelope
    assert envelope["digest"].startswith("sha256:"), envelope
    assert envelope["reason"], envelope
    assert envelope["policy_version"], envelope
    return envelope


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

    def test_debug_profile_stores_the_label_unchanged(self, db_path):
        store = obs.ObservabilityStore(db_path)
        conv = store.mint_conversation_id("chan")
        stored = store.record_conversation_label("chan", conv, "Kayak order", TENANT)

        row = _rows(db_path, "SELECT * FROM conversations")[0]
        assert row["topic"] == "Kayak order"
        assert row["summary"] == TENANT
        assert stored == "Kayak order"

    def test_evidence_profile_withholds_the_label_behind_a_badge(
        self, db_path, evidence_profile
    ):
        store = obs.ObservabilityStore(db_path)
        conv = store.mint_conversation_id("chan")
        store.record_conversation_label("chan", conv, f"Kayak for {TENANT}", TENANT)

        row = _rows(db_path, "SELECT * FROM conversations")[0]
        for column in ("topic", "summary"):
            badge = _badge(row[column])
            assert badge["classification"] == "user-text"
            assert badge["disposition"] == "omit"
        assert TENANT not in json.dumps(row)

    def test_the_returned_topic_is_what_was_actually_stored(
        self, db_path, evidence_profile
    ):
        """Ruling I9's contract: a caller that logs the label must log the badge,
        not its own candidate — otherwise the operator's log carries the value the
        DB was told to withhold."""
        store = obs.ObservabilityStore(db_path)
        conv = store.mint_conversation_id("chan")
        returned = store.record_conversation_label("chan", conv, TENANT, None)

        stored = _rows(db_path, "SELECT topic FROM conversations")[0]["topic"]
        assert returned == stored
        assert TENANT not in returned

    def test_the_blank_topic_sentinel_survives_the_policy(
        self, db_path, evidence_profile
    ):
        """A blank generated topic must stay NULL, not become a badge.

        `_label_is_due` treats a blank topic as "no successful title yet" and
        retries; a badge is non-blank, so policing a blank one would permanently
        freeze the conversation as titled-but-empty.
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

    def test_topic_uniquification_still_runs_under_the_default_profile(self, db_path):
        """Policing happens AFTER `_unique_topic_in_txn`, so the suffix can never
        land outside the envelope's closing brace."""
        store = obs.ObservabilityStore(db_path)
        first = store.mint_conversation_id("chan")
        second = store.mint_conversation_id("chan")
        store.record_conversation_label("chan", first, "Kayak order", "s1")
        store.record_conversation_label("chan", second, "kayak order", "s2")

        topics = {r["topic"] for r in _rows(db_path, "SELECT topic FROM conversations")}
        assert topics == {"Kayak order", "kayak order 1"}

    def test_a_withheld_topic_is_still_parseable_json(self, db_path, evidence_profile):
        """Two conversations, same title: under `evidence` they digest identically
        and uniquification stops distinguishing them — which is acceptable, but
        the column must still hold JSON rather than an envelope with ` 1` glued
        onto the end of it."""
        store = obs.ObservabilityStore(db_path)
        first = store.mint_conversation_id("chan")
        second = store.mint_conversation_id("chan")
        store.record_conversation_label("chan", first, "Kayak order", "s1")
        store.record_conversation_label("chan", second, "Kayak order", "s2")

        for row in _rows(db_path, "SELECT topic FROM conversations"):
            _badge(row["topic"])

    def test_both_label_routes_agree_on_what_they_store(
        self, db_path, tmp_path, evidence_profile
    ):
        """The queued route scrubs in `SQLiteTraceSink._apply_label` before
        reaching `apply_label_txn`; the sync route does not. Scrubbing first
        inside the enforcement point is what makes both produce the same
        envelope — and a digest that depended on which route wrote the row would
        be a digest nobody could compare across two runs.
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

        assert _badge(sync_topic)["digest"] == _badge(queued_topic)["digest"]


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

    def test_debug_profile_stores_every_scalar_unchanged(self, db_path):
        span = _span()
        row = _write_span(db_path, span)
        assert row["name"] == span.name
        assert row["command_name"] == span.command_name
        assert row["context"] == span.context
        assert row["channel_id"] == span.channel_id

    def test_evidence_profile_withholds_the_context_display_name(
        self, db_path, evidence_profile
    ):
        """`context` is the one scalar that can carry entity content: it comes
        from a workflow-supplied `get_displayname(instance)` hook."""
        row = _write_span(db_path, _span())
        badge = _badge(row["context"])
        assert badge["classification"] == "user-text"
        assert badge["disposition"] == "omit"
        assert TENANT not in json.dumps(row)

    def test_evidence_profile_keeps_the_closed_vocabularies_usable(
        self, db_path, evidence_profile
    ):
        """`name` and `command_name` are declared rather than withheld.

        Declaring them is what FW-REQ-002 clause 3 asks for; withholding them
        would break `list_turns(command_name=...)` and the `idx_spans_command`
        lookup behind the debug UI's command filter, for no reduction in
        exposure — both are closed vocabularies the workflow itself defines.
        """
        span = _span()
        row = _write_span(db_path, span)
        assert row["name"] == span.name
        assert row["command_name"] == span.command_name

        store = obs.ObservabilityStore(db_path)
        assert store.get_spans(span.trace_id)[0]["command_name"] == span.command_name

    def test_evidence_profile_leaves_channel_id_joinable(
        self, db_path, evidence_profile
    ):
        """SCRUB-ONLY, and the reason is erasure, not convenience.

        `forget_channel` deletes spans with `WHERE channel_id=?`. Digesting this
        column — which is what the `identifier` default would do — would narrow
        first-class erasure to whatever the `trace_id IN (...)` fallback still
        covers. Reducing exposure by weakening erasure is not a trade worth
        making.
        """
        span = _span()
        _write_span(db_path, span)

        store = obs.ObservabilityStore(db_path)
        assert store.forget_channel(span.channel_id)["spans"] == 1
        assert _rows(db_path, "SELECT * FROM spans") == []

    def test_a_null_scalar_stays_null(self, db_path, evidence_profile):
        """An absent context must not become the string "" or a badge for
        nothing: `COALESCE(excluded.context, spans.context)` in the upsert
        depends on NULL staying NULL."""
        row = _write_span(db_path, _span(context=None, command_name=None))
        assert row["context"] is None
        assert row["command_name"] is None

    def test_span_attributes_are_still_scrubbed(self, db_path, planted_credentials):
        """The span-attribute scrub, re-asserted because the capture-policy
        work rewrote the tuple bound around it."""
        row = _write_span(db_path, _span(attributes={"leak": f"key {ENV_SECRET}"}))
        assert ENV_SECRET not in row["attributes"]
        assert REDACTED in row["attributes"]


# ----------------------------------------------------------------------
# Surface 1: feedback_json — credential scrub, deliberately no capture policy
# ----------------------------------------------------------------------


class TestFeedback:
    def test_planted_credentials_do_not_reach_the_feedback_row(
        self, db_path, planted_credentials
    ):
        store = obs.ObservabilityStore(db_path)
        store.upsert_feedback(
            "turn-1",
            json.dumps({"nl_feedback": f"try {SK_TOKEN} or {ENV_SECRET}"}),
        )

        stored = _rows(db_path, "SELECT * FROM feedback")[0]["feedback_json"]
        assert SK_TOKEN not in stored
        assert ENV_SECRET not in stored

    def test_scrubbing_does_not_corrupt_the_json(self, db_path, planted_credentials):
        """Every credential pattern is confined to characters that cannot appear
        unescaped inside a JSON string, so a replacement can never cross a
        delimiter. Asserted rather than reasoned about, because
        `get_memory_window` calls `json.loads` on this column and falls back to
        handing the agent the raw text when it fails."""
        store = obs.ObservabilityStore(db_path)
        store.upsert_feedback(
            "turn-1",
            json.dumps(
                {
                    "binary_or_numeric_score": 1,
                    "nl_feedback": f"the key {SK_TOKEN} did not work",
                    "timestamp": 1756339200000,
                }
            ),
        )

        stored = _rows(db_path, "SELECT * FROM feedback")[0]["feedback_json"]
        parsed = json.loads(stored)
        assert parsed["binary_or_numeric_score"] == 1
        assert parsed["timestamp"] == 1756339200000
        assert REDACTED in parsed["nl_feedback"]

    @pytest.mark.parametrize("profile", ["debug", "evidence"])
    def test_feedback_content_survives_both_profiles(self, db_path, monkeypatch, profile):
        """PINS A DELIBERATE DECISION: no capture policy on this column.

        `get_memory_window` parses it and `restore_history_from_turns` puts the
        result straight into `dspy.History` — it is the agent's memory of being
        corrected, not evidence about the agent. Under `evidence` a badge would
        still parse, so the agent would silently receive an envelope dict where
        its feedback used to be and behave differently. Redacting this column
        belongs with conversation-memory redaction, which has to leave memory
        usable. If this test ever fails, the agent just got quieter.
        """
        monkeypatch.setenv(obs.CAPTURE_PROFILE_VAR, profile)
        store = obs.ObservabilityStore(db_path)
        store.upsert_feedback("turn-1", json.dumps({"nl_feedback": TENANT}))

        stored = _rows(db_path, "SELECT * FROM feedback")[0]["feedback_json"]
        assert json.loads(stored) == {"nl_feedback": TENANT}


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
        # whole escalation block through. This is why the column is classified
        # `opaque-payload` rather than treated as a bag of numbers.
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

    def test_debug_profile_stores_the_metrics_unchanged(self, db_path):
        store = obs.ObservabilityStore(db_path)
        store.record_train_run("run-1", "fp", None, None, _metrics())

        runs = store.list_train_runs()
        assert json.loads(runs[0]["metrics_json"]) == _metrics()

    def test_evidence_profile_withholds_the_metrics_behind_a_badge(
        self, db_path, evidence_profile
    ):
        store = obs.ObservabilityStore(db_path)
        store.record_train_run("run-1", "fp", "then", "now", _metrics())

        row = _rows(db_path, "SELECT * FROM train_runs")[0]
        badge = _badge(row["metrics_json"])
        assert badge["classification"] == "opaque-payload"
        assert badge["disposition"] == "omit"
        assert TENANT not in row["metrics_json"]
        # The provenance columns are unpoliced, so a bundle still knows a
        # training run happened and which sources produced it.
        assert row["run_id"] == "run-1"
        assert row["workflow_fingerprint"] == "fp"
        assert row["completed_at"] == "now"

    def test_an_evidence_deployment_can_re_admit_reviewed_metrics(
        self, db_path, evidence_profile
    ):
        """The escape hatch the write site promises has to actually exist."""
        store = obs.ObservabilityStore(db_path)
        # The lazy cache `_store_capture_policy` fills; set here to inject the
        # declared field policy a deployment would configure.
        store._capture_policy = evidence_policy(
            (
                CaptureFieldPolicy(
                    field_path=obs.POLICY_PATH_TRAIN_METRICS,
                    classification="controlled-vocabulary",
                    disposition="bounded-text",
                    redact_before_trace=False,
                ),
            )
        )
        store.record_train_run("run-1", "fp", None, None, _metrics())

        stored = _rows(db_path, "SELECT metrics_json FROM train_runs")[0]["metrics_json"]
        assert json.loads(stored) == _metrics()


# ----------------------------------------------------------------------
# Surface 3: diagnostics — credential scrub, deliberately no capture policy
# ----------------------------------------------------------------------


class TestDiagnostics:
    def test_a_provider_error_body_is_scrubbed(self, db_path, planted_credentials):
        """The scenario the redactor was written for: a LiteLLM
        `AuthenticationError` whose body echoes the key, arriving here as
        `repr(exc)` in `writer_health.last_error`."""
        store = obs.ObservabilityStore(db_path)
        _set_diagnostic(
            store,
            "writer_health",
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

    @pytest.mark.parametrize("profile", ["debug", "evidence"])
    def test_writer_health_stays_readable_under_both_profiles(
        self, db_path, monkeypatch, profile
    ):
        """PINS A DELIBERATE DECISION: no capture policy on this table.

        `health_delta` and `evidence_run` read `writer_health` to decide whether
        a run may be reported as evidence at all, and `problems()` names the
        affected turn keys so a partly-damaged run can be salvaged instead of
        discarded. Withholding it under `evidence` would blind the evidence gate
        under the one profile that exists to make the gate mean something.
        """
        monkeypatch.setenv(obs.CAPTURE_PROFILE_VAR, profile)
        store = obs.ObservabilityStore(db_path)
        turn_keys = ["20260828T000000.000000Z-aaaaaaaaaaaa"]
        _set_diagnostic(
            store,
            "writer_health",
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

    @pytest.mark.parametrize("profile", ["debug", "evidence"])
    def test_a_live_sink_still_publishes_its_health_row(
        self, db_path, monkeypatch, profile
    ):
        """End to end: the writer thread's own heartbeat goes through
        `set_diagnostic`, so a policy there would take the health row out from
        under a running sink rather than merely out of a bundle."""
        monkeypatch.setenv(obs.CAPTURE_PROFILE_VAR, profile)
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
    @pytest.mark.parametrize("profile", ["debug", "evidence"])
    def test_memory_rebuilds_intact_under_both_profiles(
        self, db_path, monkeypatch, profile
    ):
        """Summary, traces AND feedback all have to arrive.

        `test_capture_policy_wiring` already pins the first two through
        `_POLICY_EXEMPT_TURN_COLUMNS`; this adds the third, which is the one this
        change could have broken, and asserts the whole 3-key shape
        `restore_history_from_turns` consumes.
        """
        monkeypatch.setenv(obs.CAPTURE_PROFILE_VAR, profile)
        sink = obs.SQLiteTraceSink(db_path)
        turn = _turn_result()
        try:
            sink.emit_turn_record(turn)
            assert sink.flush()
        finally:
            sink.close()

        store = obs.ObservabilityStore(db_path)
        store.upsert_feedback(
            turn.turn_output.turn_key, json.dumps({"nl_feedback": "helpful"})
        )

        assert store.count_usable_turns("chan", 1) == 1
        window = store.get_memory_window("chan", 1, max_turns=10)
        assert window == [
            {
                "conversation summary": "user asked about a kayak",
                "conversation_traces": "get_order -> ok",
                "feedback": {"nl_feedback": "helpful"},
            }
        ]

    @pytest.mark.parametrize("profile", ["debug", "evidence"])
    def test_a_labeled_conversation_still_lists_and_dumps(
        self, db_path, monkeypatch, profile
    ):
        """A withheld label must not take the conversation out of the history
        list: `list_conversation_summaries` is how a user finds it again, and
        `_label_is_due` reads the stored topic to decide whether to spend another
        LLM call on one that is already titled."""
        monkeypatch.setenv(obs.CAPTURE_PROFILE_VAR, profile)
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
        assert listed[0]["topic"]  # non-blank, whatever the profile did to it

        stored_topic, usable = store.conversation_label_state("chan", 1)
        assert usable == 1
        assert stored_topic.strip()  # so `_label_is_due` will not re-generate

        dumped = store.dump_all_conversations("chan")
        assert len(dumped[0]["turns"]) == 1


# ----------------------------------------------------------------------
# Phase 0: the default profile changes nothing but the scrub
# ----------------------------------------------------------------------


def test_the_default_profile_leaves_all_five_surfaces_byte_identical(db_path):
    """One test covering all five at once, because the Phase-0 promise is about
    the set of them rather than about any one."""
    assert obs.resolve_capture_policy().profile == "debug"

    store = obs.ObservabilityStore(db_path)
    conv = store.mint_conversation_id("chan")
    store.record_conversation_label("chan", conv, "Kayak order", TENANT)
    store.upsert_feedback("turn-1", json.dumps({"nl_feedback": TENANT}))
    store.record_train_run("run-1", "fp", None, None, _metrics())
    _set_diagnostic(store, "probe", {"note": TENANT})
    span = _span()
    span_row = _write_span(db_path, span)

    conversation = _rows(db_path, "SELECT * FROM conversations")[0]
    assert conversation["topic"] == "Kayak order"
    assert conversation["summary"] == TENANT
    assert json.loads(
        _rows(db_path, "SELECT * FROM feedback")[0]["feedback_json"]
    ) == {"nl_feedback": TENANT}
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


def test_every_declared_policy_path_is_reachable_from_a_write_site():
    """A path constant nobody can spell is a policy nobody can override.

    Guards against a constant being renamed here while the write site keeps its
    own literal, which would leave a deployment's `CaptureFieldPolicy` silently
    matching nothing.
    """
    paths = {
        obs.POLICY_PATH_SPAN_NAME,
        obs.POLICY_PATH_SPAN_COMMAND_NAME,
        obs.POLICY_PATH_SPAN_CONTEXT,
        obs.POLICY_PATH_CONVERSATION_TOPIC,
        obs.POLICY_PATH_CONVERSATION_SUMMARY,
        obs.POLICY_PATH_TRAIN_METRICS,
        # ido-zlm. The sixth surface is not a column of this database at all:
        # the offload evidence sidecar's raw command response, written by
        # `observation_offloading.archive.persist` through
        # `protect_offload_observation`. It is listed here for the same reason
        # as the other five -- a deployment re-admitting it under the evidence
        # profile has to be able to spell the path.
        obs.POLICY_PATH_OFFLOAD_OBSERVATION,
    }
    assert len(paths) == 7
    # The turn-column paths `_policed_column` builds must not collide with them.
    turn_paths = {f"turn.{column}" for column, _ in obs._POLICED_TURN_COLUMNS}
    assert not paths & turn_paths
