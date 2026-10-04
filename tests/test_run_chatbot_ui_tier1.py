"""UI tier 1 (bead fix-49m.6): what the debug UI surfaces from the ported store.

Four things the observability port recorded now reach the run_chatbot debug
UI, drilling down inside `#detail` rather than through new panels:

(a) an `fw.llm.call` whose `usage.completion_tokens` equals the flat
    `call_kwargs.max_tokens` is chipped "cut at limit", with per-turn and
    per-attempt tallies where turns and attempts are listed;
(b) the evidence-run verdict -- valid, or invalid with the STORED reasons --
    plus each segment's writer-health delta, on the experiment page and on
    every attempt row;
(c) a value the capture policy withheld renders as "withheld by policy:
    <reason>" with its digest, never as a bare hash that looks like data;
(d) the attempt card carries a collapsed "server configuration" section from
    the stamped `runtime_snapshot`, and says "not recorded" when it is null.

Server tests run against a real store in tmp_path seeded through the store's
own write methods (no mocks), read back through the stdlib server; the same
store is then archived and read through a workspace manifest, because the
read-only workspace must render all four from an archive too. The SPA is a
single self-contained file, so its render functions are pinned by presence.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from fastworkflow import state_paths, tracing
from fastworkflow.observability import store as obs
from fastworkflow.observability.capture_policy import evidence_policy
from fastworkflow.experiment.runner import ExperimentController
from fastworkflow.observability.workspace import (
    WORKSPACE_SCHEMA,
    UnknownWorkspaceStore,
    load_observability_workspace,
)
from fastworkflow.run_chatbot import server as run_chatbot_server
from fastworkflow.run_chatbot.server import (
    annotate_turn_rows,
    count_llm_calls_cut_at_limit,
    evidence_verdict,
    llm_call_cut_at_limit,
)


EXP = "exp-ui"
TASK = "task-1"
BARE_EXP = "exp-bare"
TURN_CUT = "20260906T100000-cut"          # attempt 1: one call cut at its cap
TURN_PLAIN = "20260906T100100-plain"      # attempt 2: nothing cut
TURN_POLICED = "20260906T100200-policed"  # chatbot channel, columns withheld
SECRET_MESSAGE = "my secret grocery list"
SECRET_ANSWER = "added milk to the secret list"
SECRET_FAILURE = "provider said: key sk-live-000 rejected"
SECRET_TOPIC = "Grocery secrets"
SECRET_SUMMARY = "About the milk nobody should see"
SNAPSHOT = {
    "capture_profile": "evidence",
    "configuration_valid": True,
    "effective_features": {"decision_signals_v1": "shadow"},
    "pid": 4242,
    "workflow_fingerprint": "sha256:abc",
    "workflow_model_version": None,
}
PROBLEM = (
    "1 turn record(s) DROPPED, affecting: tk-lost. "
    "The run is not valid evidence (§12.4)."
)
WARNING = (
    "2 span(s) dropped, affecting: tk-thin. These turns have incomplete "
    "detail; the run remains valid."
)
DELTA_INVALID = {
    "records_dropped": 1,
    "spans_dropped": 0,
    "write_errors": 0,
    "refused_terminal_writes": 0,
    "busy_retries": 0,
    "sync_fallbacks": 0,
    "records_dropped_turn_keys": ["tk-lost"],
    "spans_dropped_turn_keys": [],
    "dropped_turn_keys_elided": 0,
    "incomparable": False,
}
DELTA_VALID = dict(
    DELTA_INVALID,
    records_dropped=0,
    spans_dropped=2,
    records_dropped_turn_keys=[],
    spans_dropped_turn_keys=["tk-thin"],
)


# ----------------------------------------------------------------------
# The pure helpers
# ----------------------------------------------------------------------


def _llm(usage, call_kwargs, *, name="fw.llm.call", text=True):
    attributes = {"module": "Predict"}
    if usage is not None:
        attributes["usage"] = json.dumps(usage) if text else usage
    if call_kwargs is not None:
        attributes["call_kwargs"] = json.dumps(call_kwargs) if text else call_kwargs
    return {"name": name, "attributes": attributes}


class TestTokenLimitDetection:
    def test_completion_equal_to_the_flat_cap_is_cut(self):
        assert llm_call_cut_at_limit(
            _llm({"completion_tokens": 512, "prompt_tokens": 9}, {"max_tokens": 512})
        )

    def test_attributes_may_arrive_as_row_text_or_decoded(self):
        span = _llm({"completion_tokens": 8}, {"max_tokens": 8})
        assert llm_call_cut_at_limit(span)
        assert llm_call_cut_at_limit(dict(span, attributes=json.dumps(span["attributes"])))
        assert llm_call_cut_at_limit(
            _llm({"completion_tokens": 8}, {"max_tokens": 8}, text=False)
        )

    @pytest.mark.parametrize(
        "usage, call_kwargs",
        [
            (None, {"max_tokens": 512}),                       # no usage
            ({"completion_tokens": 512}, None),                # no call_kwargs
            ({"prompt_tokens": 512}, {"max_tokens": 512}),     # no completion count
            ({"completion_tokens": 512}, {"timeout": 9.5}),    # no cap
            ({"completion_tokens": 100}, {"max_tokens": 512}), # under the cap
            # The pre-flattening shape: a cap one level down is not a cap.
            ({"completion_tokens": 512}, {"kwargs": {"max_tokens": 512}}),
            ({"completion_tokens": "512"}, {"max_tokens": "512"}),  # strings
            ({"completion_tokens": 512.0}, {"max_tokens": 512}),    # float
            ({"completion_tokens": True}, {"max_tokens": 1}),       # bool
            ({"completion_tokens": 0}, {"max_tokens": 0}),          # no cap at all
            ("not a mapping", {"max_tokens": 512}),
        ],
    )
    def test_never_a_false_positive(self, usage, call_kwargs):
        assert not llm_call_cut_at_limit(_llm(usage, call_kwargs))

    def test_only_llm_call_spans_count(self):
        assert not llm_call_cut_at_limit(
            _llm({"completion_tokens": 5}, {"max_tokens": 5}, name="fw.turn")
        )
        assert count_llm_calls_cut_at_limit(
            [
                _llm({"completion_tokens": 5}, {"max_tokens": 5}),
                _llm({"completion_tokens": 5}, {"max_tokens": 5}, name="fw.turn"),
                _llm({"completion_tokens": 4}, {"max_tokens": 5}),
                {"name": "fw.llm.call", "attributes": "not json"},
            ]
        ) == 1


class TestEvidenceVerdict:
    def test_no_segments_is_unrecorded_not_valid(self):
        verdict = evidence_verdict([])
        assert verdict["state"] == "unrecorded"
        assert verdict["segments"] == [] and verdict["problems"] == []
        assert evidence_verdict(None)["state"] == "unrecorded"

    def test_invalid_segment_quotes_its_stored_reasons_verbatim(self):
        verdict = evidence_verdict(
            [
                {
                    "seq": 1,
                    "evidence_run_id": "evr-1",
                    "valid": 0,
                    "started_at": "t0",
                    "completed_at": "t1",
                    "record": {
                        "problems": [PROBLEM],
                        "writer_health_delta": DELTA_INVALID,
                        "in_process": True,
                    },
                }
            ]
        )
        assert verdict["state"] == "invalid"
        assert verdict["problems"] == [PROBLEM]
        segment = verdict["segments"][0]
        assert segment["valid"] is False
        assert segment["problems"] == [PROBLEM]
        assert segment["writer_health_delta"] == DELTA_INVALID
        assert segment["in_process"] is True
        assert (segment["seq"], segment["evidence_run_id"]) == (1, "evr-1")

    def test_a_valid_segment_with_dropped_spans_is_valid_with_warnings(self):
        verdict = evidence_verdict(
            [{"seq": 1, "evidence_run_id": "evr-1", "valid": 1,
              "record": {"problems": [WARNING], "writer_health_delta": DELTA_VALID}}]
        )
        assert verdict["state"] == "valid"
        assert verdict["problems"] == []
        assert verdict["warnings"] == [WARNING]

    def test_the_valid_column_outranks_a_cleaner_later_record(self):
        """`record_evidence_segment` keeps `valid=0` once set even when the seq
        is rewritten clean; the badge must follow the column, and then has no
        stored reason to quote -- which it reports as such, not as valid."""
        verdict = evidence_verdict(
            [{"seq": 1, "evidence_run_id": "evr-1", "valid": 0,
              "record": {"valid": True, "problems": []}}]
        )
        assert verdict["state"] == "invalid"
        assert verdict["problems"] == []

    def test_an_unreadable_record_still_yields_the_column_verdict(self):
        verdict = evidence_verdict(
            [{"seq": 1, "evidence_run_id": "evr-1", "valid": 1, "record": None}]
        )
        assert verdict["state"] == "valid"
        assert verdict["segments"][0]["writer_health_delta"] is None


# ----------------------------------------------------------------------
# A seeded store: one claimed experiment, policed columns, evidence segments
# ----------------------------------------------------------------------


def _turn_row(turn_key, channel_id, *, conversation_id=None, ordinal=None,
              user_message=None, answer="done", failure_reason=None,
              record=None, claim=None):
    row = {
        "turn_key": turn_key,
        "channel_id": channel_id,
        "conversation_id": conversation_id,
        "ordinal": ordinal,
        "user_message": user_message if user_message is not None else f"run {turn_key}",
        "refined_user_message": None,
        "entry_workflow_name": "ui-tier1",
        "entry_context": "test",
        "status": "completed",
        "success": 1,
        "failure_reason": failure_reason,
        "answer": answer,
        "conversation_summary": None,
        "conversation_traces": None,
        "started_at": "2026-09-06T10:00:00+00:00",
        "completed_at": "2026-09-06T10:00:01+00:00",
        "suspended_ms": 0,
        "continuation_of": None,
        "record_version": 1,
        "experiment_id": None,
        "task_id": None,
        "attempt": None,
        "claim_epoch": None,
        "server_incarnation": None,
        "record_json": json.dumps(
            record or {"turn_output": {"turn_key": turn_key, "success": True}}
        ),
    }
    if claim is not None:
        row.update(
            experiment_id=claim.experiment_id,
            task_id=claim.task_id,
            attempt=claim.attempt,
            claim_epoch=claim.epoch,
            server_incarnation=claim.server_incarnation,
        )
    return row


def _llm_span(span_id, trace_id, parent, t0, usage, call_kwargs):
    attributes = {"module": "Predict", "module_chain": "Predict"}
    if usage is not None:
        attributes["usage"] = json.dumps(usage)
    if call_kwargs is not None:
        attributes["call_kwargs"] = json.dumps(call_kwargs)
    return tracing.Span(
        span_id=span_id, trace_id=trace_id, name="fw.llm.call", kind="client",
        channel_id="registered:task-1:1", parent_span_id=parent,
        start_ns=t0, end_ns=t0 + 1_000_000, status="ok", attributes=attributes,
    )


@pytest.fixture
def workflow_path(tmp_path, monkeypatch) -> str:
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    # The store pins its capture regime at creation from this variable, and
    # `record_conversation_label` polices the label through it: seeding under
    # `evidence` is what puts real envelopes in the conversation columns.
    monkeypatch.setenv(obs.CAPTURE_PROFILE_VAR, "evidence")
    wf = tmp_path / "ui_workflow"
    wf.mkdir()
    return str(wf)


@pytest.fixture
def seeded_db(workflow_path) -> str:
    db_path = state_paths.observability_db(workflow_path)
    store = obs.ObservabilityStore(db_path)
    policy = evidence_policy()
    controller = ExperimentController(
        db_path, store.store_identity(), migrate=False, external=True
    )

    # -- (d) one attempt bound by a server that stamped its snapshot, one by a
    #    server that had none ------------------------------------------------
    controller.create_experiment(
        EXP, "ui tier 1", declared_tasks=1, declared_attempts=2,
        declarations=[(TASK, 1, "job-1"), (TASK, 2, "job-2")],
    )
    claims = {}
    for attempt, snapshot in ((1, SNAPSHOT), (2, None)):
        channel = f"registered:{TASK}:{attempt}"
        bootstrap = controller.register_attempt(
            EXP, TASK, attempt, f"job-{attempt}", channel
        )
        claims[attempt] = controller.claim_attempt(
            bootstrap, server_incarnation=f"server-{attempt}",
            runtime_snapshot=snapshot,
        )
        store.start_attempt(EXP, TASK, attempt, channel, source_key=f"job-{attempt}")
        store.finish_attempt(
            EXP, TASK, attempt, outcome="pass" if attempt == 1 else "fail",
            outcome_source="grader",
        )

    # -- (b) two evidence segments: one invalid with its reason, one valid
    #    with a dropped-span warning ----------------------------------------
    store.record_evidence_segment(
        EXP, 1, "evr-1",
        {"valid": False, "problems": [PROBLEM],
         "writer_health_delta": DELTA_INVALID, "in_process": True},
    )
    store.record_evidence_segment(
        EXP, 2, "evr-2",
        {"valid": True, "problems": [WARNING],
         "writer_health_delta": DELTA_VALID, "in_process": False},
    )

    # An experiment with no evidence run and an attempt no server stamped.
    store.create_experiment(BARE_EXP, "bare", declared_tasks=1, declared_attempts=1)
    store.start_attempt(BARE_EXP, "t0", 1, "chan-bare")
    store.finish_attempt(BARE_EXP, "t0", 1, outcome="pass", outcome_source="grader")

    # -- (c) a chatbot turn whose policed columns hold envelopes -------------
    conv_id = store.mint_conversation_id("chatbot")
    store.record_conversation_label("chatbot", conv_id, SECRET_TOPIC, SECRET_SUMMARY)
    policed_record = {
        "turn_output": {
            "turn_key": TURN_POLICED,
            "success": True,
            "command_outputs": [
                {
                    "command_name": "add_todo",
                    "command_parameters": {
                        "description": policy.apply(
                            "command.add_todo.parameters.description",
                            "buy the secret milk", classification="user-text",
                        )
                    },
                    "command_response": {
                        "response": policy.apply(
                            "command.add_todo.response", SECRET_ANSWER,
                            classification="user-text",
                        ),
                        "success": True,
                        "artifacts": {
                            "note": policy.apply(
                                "command.add_todo.artifacts.note",
                                "an inline secret note", classification="user-text",
                            )
                        },
                    },
                }
            ],
        }
    }

    # -- (a) spans: one call cut at its cap and four that must not count -----
    t0 = time.time_ns()
    spans = [
        tracing.Span(
            span_id="s-root", trace_id=TURN_CUT, name="fw.turn", kind="internal",
            channel_id="registered:task-1:1", start_ns=t0, end_ns=t0 + 9_000_000,
            status="ok", attributes={},
        ),
        _llm_span("s-cut", TURN_CUT, "s-root", t0 + 1_000_000,
                  {"prompt_tokens": 10, "completion_tokens": 512, "total_tokens": 522},
                  {"max_tokens": 512, "timeout": 30.0}),
        _llm_span("s-under", TURN_CUT, "s-root", t0 + 3_000_000,
                  {"prompt_tokens": 10, "completion_tokens": 100, "total_tokens": 110},
                  {"max_tokens": 512}),
        _llm_span("s-no-usage", TURN_CUT, "s-root", t0 + 5_000_000,
                  None, {"max_tokens": 512}),
        _llm_span("s-no-cap", TURN_CUT, "s-root", t0 + 6_000_000,
                  {"completion_tokens": 512}, None),
        _llm_span("s-nested", TURN_CUT, "s-root", t0 + 7_000_000,
                  {"completion_tokens": 512}, {"kwargs": {"max_tokens": 512}}),
        _llm_span("s-plain", TURN_PLAIN, None, t0,
                  {"completion_tokens": 100}, {"max_tokens": 512}),
    ]

    redactor = store._store_redactor()
    conn = store._connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        assert store.upsert_turn_row(
            conn, _turn_row(TURN_CUT, "registered:task-1:1", claim=claims[1]),
            [], redactor,
        )
        assert store.upsert_turn_row(
            conn, _turn_row(TURN_PLAIN, "registered:task-1:2", claim=claims[2]),
            [], redactor,
        )
        assert store.upsert_turn_row(
            conn,
            _turn_row(
                TURN_POLICED, "chatbot", conversation_id=conv_id, ordinal=1,
                user_message=obs._policed_column(
                    policy, "user_message", "user-text", SECRET_MESSAGE
                ),
                answer=obs._policed_column(policy, "answer", "user-text", SECRET_ANSWER),
                failure_reason=obs._policed_column(
                    policy, "failure_reason", "opaque-payload", SECRET_FAILURE
                ),
                record=policed_record,
            ),
            [], redactor,
        )
        store.upsert_span_rows(conn, spans, redactor)
        conn.commit()
    finally:
        conn.close()
    return db_path


def _serve(db_path, **kwargs):
    srv = run_chatbot_server.ChatbotServer(db_path, port=0, **kwargs)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    return srv, thread


@pytest.fixture
def server(seeded_db, workflow_path):
    srv, thread = _serve(seeded_db, workflow_path=workflow_path)
    yield srv
    srv.shutdown()
    thread.join(timeout=5)


def _get(server, path):
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.port}{path}",
        headers={"Authorization": f"Bearer {server.token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _get_json(server, path):
    status, body = _get(server, path)
    assert status == 200, f"{path} -> {status}: {body[:300]!r}"
    return json.loads(body)


def _assert_withheld(value, secret):
    """The persisted form of a withheld value: an envelope, never the text."""
    envelope = json.loads(value) if isinstance(value, str) else value
    assert isinstance(envelope, dict) and envelope["__fw_capture__"] is True
    assert envelope["disposition"] == "omit"
    assert envelope["reason"] == "omitted by evidence profile default"
    # `capture_policy._digest_bytes`: "sha256:" + 16 hex, a labelled digest.
    assert envelope["digest"].startswith("sha256:") and len(envelope["digest"]) == 23
    assert envelope["original_bytes"] == len(secret.encode("utf-8"))
    assert secret not in json.dumps(envelope)


# ----------------------------------------------------------------------
# The live routes
# ----------------------------------------------------------------------


class TestLiveRoutes:
    def test_listed_turns_carry_their_cut_at_limit_tally(self, server):
        turns = {t["turn_key"]: t for t in _get_json(server, "/api/turns")["turns"]}
        assert turns[TURN_CUT]["llm_calls_cut_at_limit"] == 1
        assert turns[TURN_PLAIN]["llm_calls_cut_at_limit"] == 0
        assert turns[TURN_POLICED]["llm_calls_cut_at_limit"] == 0
        # The spans the SPA chips come back with both sides of the equality.
        spans = {s["span_id"]: s for s in _get_json(server, f"/api/spans/{TURN_CUT}")["spans"]}
        cut = spans["s-cut"]["attributes"]
        assert json.loads(cut["usage"])["completion_tokens"] == 512
        assert json.loads(cut["call_kwargs"])["max_tokens"] == 512
        assert "kwargs" not in json.loads(cut["call_kwargs"])

    def test_experiment_detail_carries_the_verdict_with_stored_reasons(self, server):
        exp = _get_json(server, f"/api/experiment/{EXP}")["experiment"]
        verdict = exp["evidence"]
        assert verdict["state"] == "invalid"
        assert verdict["problems"] == [PROBLEM]
        assert verdict["warnings"] == [WARNING]
        by_seq = {s["seq"]: s for s in verdict["segments"]}
        assert by_seq[1]["valid"] is False and by_seq[1]["evidence_run_id"] == "evr-1"
        assert by_seq[1]["writer_health_delta"] == DELTA_INVALID
        assert by_seq[2]["valid"] is True and by_seq[2]["problems"] == [WARNING]
        assert by_seq[2]["writer_health_delta"] == DELTA_VALID
        assert by_seq[2]["in_process"] is False
        # The raw segments the verdict was built from are still there.
        assert [s["evidence_run_id"] for s in exp["evidence_runs"]] == ["evr-1", "evr-2"]

        bare = _get_json(server, f"/api/experiment/{BARE_EXP}")["experiment"]
        assert bare["evidence"] == {
            "state": "unrecorded", "segments": [], "problems": [], "warnings": []
        }

    def test_attempt_rows_carry_verdict_tally_and_snapshot(self, server):
        rows = _get_json(server, f"/api/experiment/{EXP}/attempts?task={TASK}")["attempts"]
        by_attempt = {r["attempt"]: r for r in rows}
        assert set(by_attempt) == {1, 2}
        for row in rows:
            assert row["evidence"]["state"] == "invalid"
            assert row["evidence"]["problems"] == [PROBLEM]
            assert "runtime_snapshot_json" not in row
        assert by_attempt[1]["llm_calls_cut_at_limit"] == 1
        assert by_attempt[1]["turn_count"] == 1
        assert by_attempt[1]["runtime_snapshot"] == SNAPSHOT
        assert by_attempt[2]["llm_calls_cut_at_limit"] == 0
        assert by_attempt[2]["runtime_snapshot"] is None

        bare = _get_json(server, f"/api/experiment/{BARE_EXP}/attempts")["attempts"]
        assert bare[0]["evidence"]["state"] == "unrecorded"
        assert bare[0]["runtime_snapshot"] is None
        assert bare[0]["llm_calls_cut_at_limit"] == 0

    def test_unknown_experiment_attempts_still_404(self, server):
        assert _get(server, "/api/experiment/nope/attempts")[0] == 404

    def test_policed_columns_reach_the_ui_as_envelopes_not_text(self, server):
        turn = _get_json(server, f"/api/turn/{TURN_POLICED}")["turn"]
        _assert_withheld(turn["user_message"], SECRET_MESSAGE)
        _assert_withheld(turn["answer"], SECRET_ANSWER)
        _assert_withheld(turn["failure_reason"], SECRET_FAILURE)
        output = turn["record"]["turn_output"]["command_outputs"][0]
        _assert_withheld(output["command_response"]["response"], SECRET_ANSWER)
        _assert_withheld(output["command_parameters"]["description"], "buy the secret milk")
        _assert_withheld(output["command_response"]["artifacts"]["note"], "an inline secret note")

        listed = {t["turn_key"]: t for t in _get_json(server, "/api/turns?channel=chatbot")["turns"]}
        _assert_withheld(listed[TURN_POLICED]["user_message"], SECRET_MESSAGE)

        conv = _get_json(server, "/api/conversations?channel=chatbot")["conversations"][0]
        _assert_withheld(conv["topic"], SECRET_TOPIC)
        _assert_withheld(conv["summary"], SECRET_SUMMARY)

    def test_no_secret_text_anywhere_in_the_policed_responses(self, server):
        for path in (
            f"/api/turn/{TURN_POLICED}",
            "/api/turns?channel=chatbot",
            "/api/conversations?channel=chatbot",
        ):
            _, body = _get(server, path)
            for secret in (SECRET_MESSAGE, SECRET_ANSWER, SECRET_FAILURE,
                           SECRET_TOPIC, SECRET_SUMMARY, "sk-live-000"):
                assert secret.encode() not in body, (path, secret)


class TestAnnotateTurnRows:
    def test_annotation_reads_only_through_the_store(self, seeded_db):
        store = obs.ReadOnlyObservabilityStore(seeded_db)
        turns = store.list_turns(limit=10)
        annotate_turn_rows(store, turns)
        assert {t["turn_key"]: t["llm_calls_cut_at_limit"] for t in turns} == {
            TURN_CUT: 1, TURN_PLAIN: 0, TURN_POLICED: 0
        }


# ----------------------------------------------------------------------
# fix-tk5: the same stamps, from a bounded number of span queries
# ----------------------------------------------------------------------
#
# `annotate_turn_rows` called `get_spans` once per listed turn, so the rail's
# 500-turn refresh issued ~500 queries. The page's spans now come from one
# bulk read. The stamps have to be *identical* to what the per-turn path
# produced, so these compare the two over a many-turn store rather than
# re-asserting expected values.


@pytest.fixture
def wide_db(workflow_path) -> str:
    """30 chatbot turns; every third one has an LLM call cut at its cap."""
    db_path = state_paths.observability_db(workflow_path)
    store = obs.ObservabilityStore(db_path)
    conv_id = store.mint_conversation_id("chatbot")
    t0 = time.time_ns()
    rows, spans = [], []
    for index in range(30):
        key = f"20260907T{index:06d}-wide"
        rows.append(_turn_row(key, "chatbot", conversation_id=conv_id, ordinal=index))
        base = t0 + index * 10_000_000
        spans.append(
            tracing.Span(
                span_id=f"{key}-root", trace_id=key, name="fw.turn", kind="internal",
                channel_id="chatbot", start_ns=base, end_ns=base + 5_000_000,
                status="ok", attributes={},
            )
        )
        spans.append(
            _llm_span(
                f"{key}-llm", key, f"{key}-root", base + 1_000_000,
                {"prompt_tokens": 10,
                 "completion_tokens": 512 if index % 3 == 0 else 40,
                 "total_tokens": 60},
                {"max_tokens": 512, "model": "test/model"},
            )
        )
    redactor = store._store_redactor()
    conn = store._connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        for row in rows:
            assert store.upsert_turn_row(conn, row, [], redactor)
        store.upsert_span_rows(conn, spans, redactor)
        conn.commit()
    finally:
        conn.close()
    return db_path


def _stamps_the_old_way(store, turns):
    """What `annotate_turn_rows` did before fix-tk5: one `get_spans` per turn."""
    return [
        run_chatbot_server.turn_span_stamps(store.get_spans(turn["turn_key"]))
        for turn in turns
    ]


class TestBulkSpanStamping:
    def test_stamps_are_identical_to_the_per_turn_path(self, wide_db):
        store = obs.ReadOnlyObservabilityStore(wide_db)
        turns = store.list_turns(channel_id="chatbot", limit=100)
        assert len(turns) == 30
        expected = _stamps_the_old_way(store, turns)
        annotate_turn_rows(store, turns)
        for turn, stamp in zip(turns, expected):
            assert {key: turn[key] for key in stamp} == stamp
        # Not vacuous: ten of the thirty really were cut at their cap.
        assert sum(turn["llm_calls_cut_at_limit"] for turn in turns) == 10

    def test_one_bulk_read_and_no_per_turn_read(self, wide_db):
        store = obs.ReadOnlyObservabilityStore(wide_db)
        turns = store.list_turns(channel_id="chatbot", limit=100)
        bulk_calls = []
        real = store.spans_for_turns

        def counted(keys):
            bulk_calls.append(1)
            return real(keys)

        store.spans_for_turns = counted
        store.get_spans = lambda key: pytest.fail(
            "a listed page must not read spans one turn at a time"
        )
        annotate_turn_rows(store, turns)
        assert bulk_calls == [1]

    def test_the_turns_route_answers_the_same_tallies(self, wide_db, workflow_path):
        store = obs.ReadOnlyObservabilityStore(wide_db)
        turns = store.list_turns(channel_id="chatbot", limit=100)
        expected = {
            turn["turn_key"]: stamp["llm_calls_cut_at_limit"]
            for turn, stamp in zip(turns, _stamps_the_old_way(store, turns))
        }
        srv, thread = _serve(wide_db, workflow_path=workflow_path)
        try:
            listed = _get_json(srv, "/api/turns?channel=chatbot&limit=100")["turns"]
        finally:
            srv.shutdown()
            thread.join(timeout=5)
        assert {t["turn_key"]: t["llm_calls_cut_at_limit"] for t in listed} == expected


# ----------------------------------------------------------------------
# The same store, archived, through the read-only workspace
# ----------------------------------------------------------------------


@pytest.fixture
def workspace_manifest(seeded_db, tmp_path):
    archive = obs.ObservabilityStore(seeded_db, migrate=False).archive_to(
        str(tmp_path / "sealed.sqlite3")
    )
    manifest = tmp_path / "workspace.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": WORKSPACE_SCHEMA,
                "workspace_id": "workspace-ui",
                "label": "UI tier 1 archive",
                "stores": [
                    {
                        "store_id": "sealed",
                        "label": "sealed",
                        "path": Path(archive["path"]).name,
                        "mode": "sealed",
                        "sha256": archive["sha256"],
                        "store_identity": archive["store_identity"],
                    }
                ],
                "experiments": [
                    {
                        "experiment_id": "logical",
                        "label": "logical",
                        "segments": [
                            {"segment_id": "seg-ui", "store_id": "sealed",
                             "local_experiment_id": EXP},
                            {"segment_id": "seg-bare", "store_id": "sealed",
                             "local_experiment_id": BARE_EXP},
                        ],
                    }
                ],
                "projected_attempts": [
                    {
                        "logical_attempt": {
                            "experiment_id": "historical", "task_id": "joined",
                            "attempt": 1,
                        },
                        "attempt_refs": [
                            {
                                "store_id": "sealed",
                                "local_experiment_id": EXP,
                                "task_id": TASK,
                                "attempt": 1,
                                "turn_ref": {"store_id": "sealed",
                                             "logical_turn_key": TURN_CUT},
                            }
                        ],
                        "turn_refs": [
                            {"store_id": "sealed", "logical_turn_key": TURN_CUT}
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest, archive


def test_workspace_reader_exposes_the_archived_evidence_runs(workspace_manifest):
    manifest, _ = workspace_manifest
    workspace = load_observability_workspace(manifest)
    runs = workspace.evidence_runs("sealed", EXP)
    assert [r["evidence_run_id"] for r in runs] == ["evr-1", "evr-2"]
    assert runs[0]["record"]["problems"] == [PROBLEM]
    assert workspace.evidence_runs("sealed", "no-such-experiment") == []
    with pytest.raises(UnknownWorkspaceStore):
        workspace.evidence_runs("", EXP)
    with pytest.raises(UnknownWorkspaceStore):
        workspace.evidence_runs("other", EXP)


class TestWorkspaceRoutes:
    @pytest.fixture
    def ws_server(self, workspace_manifest):
        manifest, archive = workspace_manifest
        srv, thread = _serve(archive["path"], workspace_manifest_path=str(manifest))
        yield srv
        srv.shutdown()
        thread.join(timeout=5)

    def test_segments_carry_the_archived_verdicts(self, ws_server):
        segments = {
            s["segment_id"]: s
            for s in _get_json(ws_server, "/api/workspace/experiment/logical/segments")["segments"]
        }
        assert segments["seg-ui"]["evidence"]["state"] == "invalid"
        assert segments["seg-ui"]["evidence"]["problems"] == [PROBLEM]
        assert segments["seg-ui"]["evidence"]["segments"][0]["writer_health_delta"] == DELTA_INVALID
        assert segments["seg-bare"]["evidence"]["state"] == "unrecorded"

    def test_attempts_carry_verdict_tally_snapshot_and_per_turn_counts(self, ws_server):
        rows = _get_json(ws_server, "/api/workspace/experiment/logical/attempts")["attempts"]
        by_key = {(r["local_experiment_id"], r["attempt"]): r for r in rows}
        ui1 = by_key[(EXP, 1)]
        assert ui1["evidence"]["state"] == "invalid"
        assert ui1["evidence"]["problems"] == [PROBLEM]
        assert ui1["llm_calls_cut_at_limit"] == 1
        assert ui1["runtime_snapshot"] == SNAPSHOT
        # Tier 2 (fix-aou) stamps decision signals and cost on the same refs;
        # the tier-1 facts are pinned as a subset rather than the whole dict.
        assert len(ui1["turn_refs"]) == 1
        assert {
            "store_id": "sealed", "logical_turn_key": TURN_CUT, "llm_calls_cut_at_limit": 1
        }.items() <= ui1["turn_refs"][0].items()
        ui2 = by_key[(EXP, 2)]
        assert ui2["llm_calls_cut_at_limit"] == 0
        assert ui2["runtime_snapshot"] is None
        bare = by_key[(BARE_EXP, 1)]
        assert bare["evidence"]["state"] == "unrecorded"
        assert bare["runtime_snapshot"] is None

    def test_projected_attempts_are_badged_and_tallied_once(self, ws_server):
        rows = _get_json(
            ws_server, "/api/workspace/projected_attempts?experiment=historical"
        )["projected_attempts"]
        assert len(rows) == 1
        row = rows[0]
        # The same turn is referenced twice (attempt_ref.turn_ref and
        # turn_refs) and must be counted once.
        assert row["llm_calls_cut_at_limit"] == 1
        assert row["resolved_turns"][0]["llm_calls_cut_at_limit"] == 1
        source = row["resolved_sources"][0]
        assert source["evidence"]["state"] == "invalid"
        assert source["evidence"]["problems"] == [PROBLEM]
        assert source["resolved_turn"]["llm_calls_cut_at_limit"] == 1
        assert source["resolved_attempt"]["runtime_snapshot"] == SNAPSHOT

    def test_scoped_turn_read_keeps_the_envelopes(self, ws_server):
        turn = _get_json(ws_server, f"/api/workspace/turn/sealed/{TURN_POLICED}")["turn"]
        _assert_withheld(turn["user_message"], SECRET_MESSAGE)
        _assert_withheld(
            turn["record"]["turn_output"]["command_outputs"][0]["command_response"]["response"],
            SECRET_ANSWER,
        )
        spans = _get_json(ws_server, f"/api/workspace/trace/sealed/{TURN_CUT}")["spans"]
        assert count_llm_calls_cut_at_limit(spans) == 1

    def test_archive_stays_byte_identical_after_the_reads(self, ws_server, workspace_manifest):
        import hashlib

        _, archive = workspace_manifest
        _get_json(ws_server, "/api/workspace/experiment/logical/attempts")
        _get_json(ws_server, "/api/workspace/projected_attempts?experiment=historical")
        assert hashlib.sha256(Path(archive["path"]).read_bytes()).hexdigest() == archive["sha256"]


# ----------------------------------------------------------------------
# The SPA: the render functions exist and drill inside #detail
# ----------------------------------------------------------------------


class TestPage:
    def test_render_functions_and_strings_are_present(self):
        page = run_chatbot_server.load_index_html()
        # (a) token-limit chip and tallies
        assert b"function llmCallCutAtLimit(span)" in page
        assert b"function tokenLimitChip(count)" in page
        assert b'"cut at limit"' in page
        assert b'" cut at limit"' in page
        # Rail turn rows are label-only (owner decision 2026-09-29); the chips live in the turn view.
        assert b"appendTokenLimitChip(subLine, extra.llm_calls_cut_at_limit)" in page # attempt rows (tally lives on the evidence row)
        assert b"appendTokenLimitChip(outcomeLine, row.llm_calls_cut_at_limit)" in page  # workspace card
        assert b"cut: (llmCallCutAtLimit(span) ? 1 : 0) + sumCut(children)" in page
        # (b) evidence verdict, reasons quoted, deltas rendered by key
        assert b"function evidenceBadge(verdict)" in page
        assert b"function renderEvidenceVerdict(container, verdict, opts)" in page
        assert b"function renderEvidenceSegments(container, segments)" in page
        assert b"renderEvidenceVerdict(card, exp.evidence)" in page
        assert b"renderEvidenceVerdict(verdictBox, extra.evidence, { segments: false })" in page
        assert b"renderEvidenceVerdict(segBox, segment.evidence)" in page
        assert b"writer-health delta" in page
        assert b"evidence INVALID" in page and b"no evidence run recorded" in page
        # (c) capture envelopes never print as bare digests
        assert b"function captureEnvelope(value)" in page
        assert b"function policedText(value)" in page
        assert b"function appendPoliced(parent, value)" in page
        assert b'"withheld by policy: "' in page
        assert b'"cut by policy: "' in page
        assert b'var CAPTURE_MARKER = "__fw_capture__"' in page
        for site in (
            b"return policedText(node.label);",  # rail labels carry conversation topics
            b'el("h2", null, policedText(node.label))',
            b"policedText(turn.user_message)",
            b"appendPoliced(um, turn.user_message",
            b"appendPoliced(ans, turn.answer)",
            b"policedText(turn.failure_reason)",
            b"policedText(span.command_name)",
            b"policedText(span.name)",
            b"policedText(span.context)",
            b"tmBubble(\"user\", policedText(turn.user_message))",
            b"answer: policedText(turn.answer)",
        ):
            assert site in page, site
        # The restored-conversation "Continuing ..." bubble was removed; if a
        # restored topic is shown again it must still go through the policy.
        assert b"latest.topic" not in page or b"policedText(latest.topic)" in page
        # pretty() itself walks envelopes, which covers record_json dumps,
        # artifacts and span attribute sections.
        assert b"var env = captureEnvelope(item);" in page
        # (d) the collapsed server-configuration section on the attempt card
        assert b"function renderServerConfiguration(container, attempt)" in page
        assert b'"server configuration"' in page
        assert b'"not recorded for this attempt"' in page
        assert b"renderServerConfiguration(item, extra)" in page
        assert b"renderServerConfiguration(box," in page
        # Inside #detail, not a new panel; the rules the page already keeps.
        assert b"innerHTML" not in page
        assert b"https://" not in page
        assert page.count(b'id="detail"') == 1

    def test_page_still_serves_with_its_hash_sourced_csp(self, server):
        status, body = _get(server, "/")
        assert status == 200
        assert b"renderServerConfiguration" in body


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
