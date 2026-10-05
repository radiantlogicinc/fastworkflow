"""UI tier 2 (bead fix-aou; ido-01b.15): the execution ledger, decision-signal
chips and low-confidence filter, provenance and comparability, and cost
roll-ups, all inside `#detail` and all derived in the read layer from store
reads only.

(a) a turn's execution ledger joins the record's `execution_records` refs
    with the trace's `fw.command.execute` spans on `command_call_id`; a
    resumed turn is one ledger, not a restart;
(b) per-turn chips from `decision_uncertainty` (least confident top-k
    margin), `fw.ask_user` (asked) and `consequence` (worst class), and a
    rail filter that counts only turns with a RECORDED margin below the
    user's threshold;
(c) a collapsed provenance section from the evidence-run records, the
    experiment row and the attempts' runtime snapshots, and a comparability
    check on the compare route that quotes every differing field without
    blocking the view;
(d) cost per turn and per attempt from `fw.llm.call.cost`, with "not
    recorded" -- never zero -- for calls that carried none.

Server tests seed a real store in tmp_path through the store's own write
methods, read it back through the stdlib server, then archive it and read it
again through a workspace manifest. The SPA is one self-contained file, so
its render functions are pinned by presence.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from fastworkflow import state_paths, tracing
from fastworkflow.observability import store as obs
from fastworkflow.experiment.runner import ExperimentController
from fastworkflow.observability.workspace import (
    WORKSPACE_SCHEMA,
    UnknownWorkspaceStore,
    load_observability_workspace,
)
from fastworkflow.run_chatbot import server as run_chatbot_server
from fastworkflow.run_chatbot.server import (
    LOW_CONFIDENCE_DEFAULT_MARGIN,
    annotate_turn_detail,
    cost_rollup,
    execution_ledger,
    experiment_provenance,
    is_low_confidence,
    llm_call_cost,
    merge_cost_rollups,
    provenance_differences,
    turn_decision_signals,
)


EXP = "exp-tier2"
BASE = "exp-tier2-baseline"
ODD = "exp-tier2-odd-shape"
TASK = "task-1"
TURN_A = "20260907T100000-plain"    # attempt 1: three dispatches, asked outside
TURN_B = "20260907T100100-resumed"  # attempt 2: resumed; one dispatch pre-resume
SNAPSHOT_1 = {
    "capture_profile": "debug",
    "capture_policy_version": "1",
    "workflow_fingerprint": "sha256:same",
    "workflow_model_version": "20260905T132341Z-a0605e",
    "workflow_scope_rule_version": 1,
    "effective_features": {"decision_signals_v1": "shadow"},
}
SNAPSHOT_2 = dict(SNAPSHOT_1, workflow_model_version="20260906T000000Z-ffffff")
OBSERVABILITY = {
    "schema_version": 1,
    "enabled": True,
    "capture_profile": "debug",
    "capture_policy_version": "1",
    "span_contract_version": 3,
    "span_contract_versions": {"fw.turn": 1, "fw.llm.call": 1},
    "db_schema_version": 3,
    "config": {"FW_OBS_CAPTURE_PROFILE": "debug", "FW_OBS_RETENTION_DAYS": "30"},
    "dspy_history_enabled": True,
    "evidence_grade": True,
}
OBSERVABILITY_BASE = dict(
    OBSERVABILITY,
    span_contract_version=2,
    span_contract_versions={"fw.turn": 1},
    config={"FW_OBS_CAPTURE_PROFILE": "debug", "FW_OBS_RETENTION_DAYS": "7"},
)


def _uncertainty(margin=None, confidence=None):
    if margin is None:
        return {
            "decision_kind": "command-identity",
            "signals": [],
            "candidate_count": 1,
            "reducible": None,
            "signals_absent_reason": "deterministic-resolution",
        }
    signals = [
        {
            "signal_id": "nlu.classifier.topk_margin",
            "signal_version": "intent-classifier/test",
            "kind": "classifier-topk-margin",
            "value": margin,
            "calibration_ref": None,
        }
    ]
    if confidence is not None:
        signals.insert(
            0,
            {
                "signal_id": "nlu.classifier.confidence",
                "signal_version": "intent-classifier/test",
                "kind": "classifier-confidence",
                "value": confidence,
                "calibration_ref": None,
            },
        )
    return {
        "decision_kind": "command-identity",
        "signals": signals,
        "candidate_count": 1,
        "reducible": None,
        "signals_absent_reason": None,
    }


def _span(span_id, trace_id, name, parent, t0, *, dur=1_000_000, status="ok",
          command_name=None, context=None, attributes=None, kind="tool"):
    return {
        "span_id": span_id,
        "trace_id": trace_id,
        "parent_span_id": parent,
        "name": name,
        "kind": kind,
        "channel_id": "chan",
        "command_name": command_name,
        "context": context,
        "start_ns": t0,
        "end_ns": None if dur is None else t0 + dur,
        "status": status,
        "attributes": attributes or {},
    }


def _execute(span_id, trace_id, parent, t0, call_id, command, *, context="global",
             parent_call=None, consequence=None, child_calls=None, status="ok",
             dur=1_000_000, success=True):
    attributes = {
        "raw_command": command,
        "command_call_id": call_id,
        "parent_call_id": parent_call,
        "success": success,
        "span_contract_version": 1,
    }
    if consequence is not None:
        attributes["consequence"] = {
            "consequence_class": consequence,
            "effect_kind": "unknown",
            "assessor_version": "default/1",
        }
    if child_calls is not None:
        attributes["child_calls"] = child_calls
    return _span(span_id, trace_id, "fw.command.execute", parent, t0, dur=dur,
                 status=status, command_name=command, context=context,
                 attributes=attributes)


def _intent(span_id, trace_id, parent, t0, uncertainty):
    return _span(span_id, trace_id, "fw.nlu.intent", parent, t0, kind="internal",
                 attributes={"decision_uncertainty": uncertainty, "stage": "INTENT_DETECTION"})


def _llm(span_id, trace_id, parent, t0, cost, usage=None):
    attributes = {"module": "Predict", "module_chain": "Predict", "model": "test/model"}
    if cost is not None:
        attributes["cost"] = cost
    if usage is not None:
        attributes["usage"] = json.dumps(usage)
    return _span(span_id, trace_id, "fw.llm.call", parent, t0, kind="client",
                 attributes=attributes)


def _ask(span_id, trace_id, parent, t0):
    return _span(span_id, trace_id, "fw.ask_user", parent, t0, dur=None,
                 status="open", kind="human_wait",
                 attributes={"agent_query": "which one?", "attempt": 0})


def _ref(call_id, ordinal, span_id=None, parent=None):
    return {
        "contract_version": 1,
        "command_call_id": call_id,
        "parent_call_id": parent,
        "command_ordinal": ordinal,
        "span_id": span_id,
    }


# The two turns' traces, as dicts (what the routes hand the helpers after
# decoding) and as tracing.Span rows for the store.
T0 = 1_788_000_000_000_000_000


def turn_a_spans():
    return [
        _span("a-root", TURN_A, "fw.turn", None, T0, dur=50_000_000, kind="internal"),
        _span("a-exec", TURN_A, "fw.agent.execute", "a-root", T0 + 1_000_000,
              dur=40_000_000, kind="internal"),
        _span("a-tc1", TURN_A, "fw.agent.tool_call", "a-exec", T0 + 2_000_000,
              command_name="open_directory", context="global",
              attributes={"command_call_id": "call-1",
                          "consequence": {"consequence_class": "low"}}),
        _execute("a-ex1", TURN_A, "a-tc1", T0 + 2_100_000, "call-1", "open_directory",
                 consequence="high", dur=900_000,
                 child_calls=[{"call_id": "call-1a", "parent_call_id": "call-1",
                               "command_name": "wildcard"}]),
        _intent("a-nlu1", TURN_A, "a-ex1", T0 + 2_200_000, _uncertainty(0.9, 0.95)),
        _span("a-tc2", TURN_A, "fw.agent.tool_call", "a-exec", T0 + 10_000_000,
              command_name="find_identity", context="DirectoryExplorer",
              attributes={"command_call_id": "call-2"}),
        _execute("a-ex2", TURN_A, "a-tc2", T0 + 10_100_000, "call-2", "find_identity",
                 context="DirectoryExplorer", consequence="medium", dur=2_000_000),
        _intent("a-nlu2", TURN_A, "a-ex2", T0 + 10_200_000, _uncertainty(0.1, 0.4)),
        _intent("a-nlu3", TURN_A, "a-ex2", T0 + 10_300_000, _uncertainty()),
        _llm("a-llm1", TURN_A, "a-exec", T0 + 20_000_000, 0.0015,
             {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}),
        _llm("a-llm2", TURN_A, "a-exec", T0 + 25_000_000, None,
             {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}),
        _ask("a-ask", TURN_A, "a-root", T0 + 45_000_000),
    ]


def turn_a_record():
    return {
        "turn_output": {"turn_key": TURN_A, "success": True},
        "execution_records": [
            _ref("call-1", 0, "a-ex1"),
            _ref("call-1a", 1, None, parent="call-1"),
            _ref("call-2", 2, "a-ex2"),
        ],
        "routing_events": [],
    }


def turn_b_spans():
    return [
        _span("b-root", TURN_B, "fw.turn", None, T0, dur=90_000_000, kind="internal"),
        # Before the suspension: cut short by the ask-user control signal,
        # which tracing.status_for_dispatch_exception writes as `cancelled`.
        _execute("b-ex1", TURN_B, "b-root", T0 + 1_000_000, "call-b1", "before_resume",
                 status="cancelled", dur=3_000_000),
        _ask("b-ask", TURN_B, "b-ex1", T0 + 2_000_000),
        # After the resume, in a new process with a fresh recorder.
        _execute("b-ex2", TURN_B, "b-root", T0 + 60_000_000, "call-b2", "after_resume"),
        _intent("b-nlu", TURN_B, "b-ex2", T0 + 60_100_000, _uncertainty()),
        _llm("b-llm", TURN_B, "b-root", T0 + 70_000_000, None),
    ]


def turn_b_record():
    return {
        "turn_output": {"turn_key": TURN_B, "success": True},
        "execution_records": [_ref("call-b2", 0, "b-ex2")],
        "routing_events": [],
    }


# ----------------------------------------------------------------------
# (a) the execution ledger
# ----------------------------------------------------------------------


class TestExecutionLedger:
    def test_joins_record_refs_and_execute_spans_on_command_call_id(self):
        ledger = execution_ledger(turn_a_record(), turn_a_spans())
        rows = ledger["rows"]
        assert [r["command_call_id"] for r in rows] == ["call-1", "call-1a", "call-2"]
        assert [r["position"] for r in rows] == [1, 2, 3]
        first = rows[0]
        assert first["command_name"] == "open_directory"
        assert first["context"] == "global"
        assert first["status"] == "ok"
        assert first["success"] is True
        assert first["duration_ns"] == 900_000
        assert first["span_id"] == "a-ex1"
        assert first["in_record"] and first["span_recorded"] and not first["child_call"]
        assert ledger["record_rows"] == 3 and ledger["span_rows"] == 2
        assert ledger["rows_not_in_record"] == 0
        assert ledger["rows_without_span"] == 1

    def test_a_child_call_sits_under_its_parent_with_the_name_the_span_filed(self):
        rows = execution_ledger(turn_a_record(), turn_a_spans())["rows"]
        child = rows[1]
        assert child["command_call_id"] == "call-1a"
        assert child["parent_call_id"] == "call-1"
        assert child["child_call"] is True
        assert child["command_name"] == "wildcard"
        assert child["span_recorded"] is False
        assert child["status"] is None          # no span: no status, never invented
        assert child["duration_ns"] is None

    def test_an_ask_user_outside_any_dispatch_is_counted_on_the_turn(self):
        ledger = execution_ledger(turn_a_record(), turn_a_spans())
        assert ledger["asked_user_outside_dispatch"] == 1
        assert all(r["asked_user"] == 0 for r in ledger["rows"])

    def test_a_resumed_turn_is_one_ledger_not_a_restart(self):
        ledger = execution_ledger(turn_b_record(), turn_b_spans())
        rows = ledger["rows"]
        assert [r["command_call_id"] for r in rows] == ["call-b1", "call-b2"]
        before, after = rows
        # The pre-resume dispatch: known from its span only, status quoted
        # from the control-signal rule, and the ask-user attributed to it.
        assert before["in_record"] is False and before["span_recorded"] is True
        assert before["status"] == "cancelled"
        assert before["asked_user"] == 1
        assert before["command_name"] == "before_resume"
        # The post-resume dispatch: in the record, ordinal 0 of the NEW
        # recorder, yet second in the ledger because time orders it.
        assert after["in_record"] is True and after["command_ordinal"] == 0
        assert ledger["rows_not_in_record"] == 1
        assert ledger["asked_user_outside_dispatch"] == 0

    def test_a_ref_whose_span_did_not_carry_the_id_still_joins_by_span_id(self):
        spans = [_execute("x-ex", "t", None, T0, "ignored", "cmd")]
        spans[0]["attributes"].pop("command_call_id")
        rows = execution_ledger(
            {"execution_records": [_ref("call-x", 0, "x-ex")]}, spans
        )["rows"]
        assert len(rows) == 1
        assert rows[0]["command_call_id"] == "call-x"
        assert rows[0]["command_name"] == "cmd" and rows[0]["span_recorded"]

    def test_record_only_and_span_only_and_empty(self):
        record_only = execution_ledger(turn_a_record(), [])
        assert [r["command_call_id"] for r in record_only["rows"]] == [
            "call-1", "call-1a", "call-2"
        ]
        assert record_only["rows_without_span"] == 3
        span_only = execution_ledger(None, turn_a_spans())
        assert [r["command_call_id"] for r in span_only["rows"]] == [
            "call-1", "call-1a", "call-2"
        ]
        assert span_only["rows_not_in_record"] == 3
        assert execution_ledger({"execution_records": "garbage"}, [])["rows"] == []
        assert execution_ledger({}, [])["rows"] == []

    def test_attributes_may_arrive_as_row_text(self):
        spans = [dict(s, attributes=json.dumps(s["attributes"])) for s in turn_a_spans()]
        rows = execution_ledger(turn_a_record(), spans)["rows"]
        assert rows[0]["command_name"] == "open_directory"
        assert rows[1]["command_name"] == "wildcard"


# ----------------------------------------------------------------------
# (b) decision signals and the low-confidence rule
# ----------------------------------------------------------------------


class TestDecisionSignals:
    def test_the_least_confident_margin_asked_and_worst_consequence(self):
        signals = turn_decision_signals(turn_a_spans())
        assert signals["intent_margin_min"] == 0.1
        assert signals["intent_margin_decisions"] == 2
        assert signals["intent_decisions"] == 3
        assert signals["intent_decisions_without_margin"] == 1
        assert signals["asked_user"] == 1
        # execute spans decide (high, medium); the tool_call's `low` is the
        # same dispatch seen from above and does not count twice
        assert signals["consequence_max"] == "high"
        assert signals["consequence_assessed"] == 2

    def test_no_signal_is_no_chip_not_zero(self):
        signals = turn_decision_signals(turn_b_spans())
        assert signals["intent_margin_min"] is None
        assert signals["intent_margin_decisions"] == 0
        assert signals["intent_decisions"] == 1       # deterministic, explained
        assert signals["consequence_max"] is None
        assert signals["consequence_assessed"] == 0
        assert signals["asked_user"] == 1
        empty = turn_decision_signals([])
        assert empty["intent_margin_min"] is None and empty["asked_user"] == 0

    def test_malformed_values_are_not_margins(self):
        bad = _uncertainty(0.5)
        bad["signals"][0]["value"] = "0.5"
        spans = [_intent("n1", "t", None, T0, bad)]
        assert turn_decision_signals(spans)["intent_margin_min"] is None
        bad["signals"][0]["value"] = True
        assert turn_decision_signals(spans)["intent_margin_min"] is None
        bad["signals"][0]["value"] = float("nan")
        assert turn_decision_signals(spans)["intent_margin_min"] is None
        odd = _execute("e", "t", None, T0, "c", "cmd", consequence="absurd")
        assert turn_decision_signals([odd])["consequence_max"] is None

    def test_tool_call_consequence_is_the_fallback_when_no_execute_span_exists(self):
        spans = [_span("tc", "t", "fw.agent.tool_call", None, T0,
                       attributes={"consequence": {"consequence_class": "critical"}})]
        assert turn_decision_signals(spans)["consequence_max"] == "critical"

    def test_low_confidence_requires_a_recorded_margin_below_the_threshold(self):
        assert is_low_confidence({"intent_margin_min": 0.1}, 0.2)
        assert not is_low_confidence({"intent_margin_min": 0.2}, 0.2)
        assert not is_low_confidence({"intent_margin_min": None}, 0.2)
        assert not is_low_confidence({}, 0.2)
        assert not is_low_confidence({"intent_margin_min": "0.0"}, 0.2)
        assert 0 < LOW_CONFIDENCE_DEFAULT_MARGIN < 1


# ----------------------------------------------------------------------
# (d) cost roll-ups
# ----------------------------------------------------------------------


class TestCost:
    def test_only_a_non_negative_number_on_an_llm_call_is_a_cost(self):
        assert llm_call_cost(_llm("s", "t", None, T0, 0.0015)) == 0.0015
        assert llm_call_cost(_llm("s", "t", None, T0, 0)) == 0.0
        assert llm_call_cost(_llm("s", "t", None, T0, None)) is None
        assert llm_call_cost(_llm("s", "t", None, T0, "0.0015")) is None
        assert llm_call_cost(_llm("s", "t", None, T0, True)) is None
        assert llm_call_cost(_llm("s", "t", None, T0, -1.0)) is None
        assert llm_call_cost(_llm("s", "t", None, T0, float("nan"))) is None
        not_llm = _span("s", "t", "fw.agent.step", None, T0, attributes={"cost": 1.0})
        assert llm_call_cost(not_llm) is None

    def test_rollup_total_is_none_never_zero_when_nothing_was_recorded(self):
        assert cost_rollup(turn_a_spans()) == {
            "calls": 2, "recorded": 1, "unrecorded": 1, "total": 0.0015
        }
        assert cost_rollup(turn_b_spans()) == {
            "calls": 1, "recorded": 0, "unrecorded": 1, "total": None
        }
        assert cost_rollup([]) == {"calls": 0, "recorded": 0, "unrecorded": 0, "total": None}
        text = [dict(s, attributes=json.dumps(s["attributes"])) for s in turn_a_spans()]
        assert cost_rollup(text)["total"] == 0.0015

    def test_merge_keeps_the_unrecorded_count_beside_the_sum(self):
        merged = merge_cost_rollups([cost_rollup(turn_a_spans()), cost_rollup(turn_b_spans())])
        assert merged == {"calls": 3, "recorded": 1, "unrecorded": 2, "total": 0.0015}
        assert merge_cost_rollups([cost_rollup([]), cost_rollup(turn_b_spans())])["total"] is None


# ----------------------------------------------------------------------
# (c) provenance and comparability
# ----------------------------------------------------------------------


def _detail(experiment_id, observability_records, **columns):
    detail = {
        "experiment_id": experiment_id,
        "capture_profile": "debug",
        "capture_policy_version": "1",
        "workflow_name": "wf",
        "benchmark_id": None,
        "benchmark_version": None,
        "benchmark_digest_sha256": None,
        "evidence_runs": [
            {"seq": seq, "record": {"observability": record} if record is not None else None}
            for seq, record in enumerate(observability_records, start=1)
        ],
    }
    detail.update(columns)
    return detail


def _field(provenance, key, source=None):
    matches = [
        f for f in provenance["fields"]
        if f["key"] == key and (source is None or f["source"] == source)
    ]
    assert len(matches) == 1, (key, source, matches)
    return matches[0]


class TestProvenance:
    def test_fields_come_from_the_three_sources_keys_verbatim(self):
        provenance = experiment_provenance(
            _detail(EXP, [OBSERVABILITY]),
            [{"task_id": TASK, "attempt": 1, "runtime_snapshot": SNAPSHOT_1}],
        )
        assert _field(provenance, "capture_profile", "experiment")["value"] == "debug"
        assert _field(provenance, "span_contract_version", "evidence_run")["value"] == 3
        assert _field(provenance, "span_contract_versions.fw.llm.call")["value"] == 1
        assert _field(provenance, "config.FW_OBS_RETENTION_DAYS")["value"] == "30"
        assert _field(provenance, "capture_policy_version", "evidence_run")["value"] == "1"
        assert _field(provenance, "workflow_fingerprint", "runtime_snapshot")["value"] == "sha256:same"
        assert _field(provenance, "effective_features.decision_signals_v1")["value"] == "shadow"
        assert provenance["inconsistent"] == 0

    def test_git_revision_is_listed_as_not_recorded_when_nothing_recorded_it(self):
        provenance = experiment_provenance(_detail(EXP, [OBSERVABILITY]), [])
        revision = _field(provenance, "git_revision")
        assert revision["recorded"] is False and revision["value"] is None
        assert provenance["unrecorded"] >= 1
        # A benchmark pin the row lacks is "not recorded" too, not "".
        assert _field(provenance, "benchmark_id")["recorded"] is False

    def test_git_revision_is_quoted_from_wherever_a_harness_put_it(self):
        detail = _detail(EXP, [OBSERVABILITY])
        detail["evidence_runs"][0]["record"]["engine"] = {"source_revision": "abc123"}
        assert _field(experiment_provenance(detail, []), "git_revision")["value"] == "abc123"
        detail["evidence_runs"][0]["record"].pop("engine")
        detail["evidence_runs"][0]["record"]["provenance"] = {
            "engine": {"source_revision": "def456"}
        }
        assert _field(experiment_provenance(detail, []), "git_revision")["value"] == "def456"

    def test_a_field_two_segments_disagree_on_lists_every_value_with_its_origin(self):
        provenance = experiment_provenance(
            _detail(EXP, [OBSERVABILITY, OBSERVABILITY_BASE]),
            [
                {"task_id": TASK, "attempt": 1, "runtime_snapshot": SNAPSHOT_1},
                {"task_id": TASK, "attempt": 2, "runtime_snapshot": SNAPSHOT_2},
            ],
        )
        version = _field(provenance, "span_contract_version", "evidence_run")
        assert version["consistent"] is False and version["value"] is None
        assert version["values"] == [
            {"where": "evidence segment #1", "value": 3},
            {"where": "evidence segment #2", "value": 2},
        ]
        model = _field(provenance, "workflow_model_version", "runtime_snapshot")
        assert model["consistent"] is False
        assert [v["where"] for v in model["values"]] == [
            f"attempt {TASK}#1", f"attempt {TASK}#2"
        ]
        assert _field(provenance, "workflow_fingerprint", "runtime_snapshot")["consistent"]
        assert provenance["inconsistent"] >= 2

    def test_an_unreadable_record_or_no_snapshot_yields_only_what_exists(self):
        provenance = experiment_provenance(
            _detail(EXP, [None]), [{"task_id": TASK, "attempt": 1, "runtime_snapshot": None}]
        )
        sources = {f["source"] for f in provenance["fields"]}
        assert sources == {"experiment", "evidence_run"}     # git_revision only
        assert _field(provenance, "git_revision")["recorded"] is False

    def test_differences_quote_both_sides_and_skip_what_neither_recorded(self):
        treatment = experiment_provenance(_detail(EXP, [OBSERVABILITY]), [])
        baseline = experiment_provenance(_detail(BASE, [OBSERVABILITY_BASE]), [])
        differences = provenance_differences(treatment, baseline)
        by_key = {(d["source"], d["key"]): d for d in differences}
        assert by_key[("evidence_run", "span_contract_version")] == {
            "key": "span_contract_version", "source": "evidence_run",
            "treatment": 3, "baseline": 2,
        }
        assert by_key[("evidence_run", "config.FW_OBS_RETENTION_DAYS")]["baseline"] == "7"
        # present on one side only: recorded vs not
        assert by_key[("evidence_run", "span_contract_versions.fw.llm.call")] == {
            "key": "span_contract_versions.fw.llm.call", "source": "evidence_run",
            "treatment": 1, "baseline": None,
        }
        # git_revision: unrecorded on both sides, so nothing to quote
        assert ("evidence_run", "git_revision") not in by_key
        assert ("experiment", "capture_profile") not in by_key
        assert provenance_differences(treatment, treatment) == []

    def test_an_inconsistent_side_compares_as_its_list_of_values(self):
        treatment = experiment_provenance(_detail(EXP, [OBSERVABILITY, OBSERVABILITY_BASE]), [])
        baseline = experiment_provenance(_detail(BASE, [OBSERVABILITY]), [])
        diff = {
            d["key"]: d for d in provenance_differences(treatment, baseline)
        }["span_contract_version"]
        assert diff["treatment"] == [3, 2] and diff["baseline"] == 3


# ----------------------------------------------------------------------
# Through the server, over a real store
# ----------------------------------------------------------------------


def _turn_row(turn_key, channel_id, *, record, claim=None, suspended_ms=0,
              conversation_id=None, ordinal=1):
    row = {
        "turn_key": turn_key,
        "channel_id": channel_id,
        "conversation_id": conversation_id,
        "ordinal": ordinal,
        "user_message": f"run {turn_key}",
        "refined_user_message": None,
        "entry_workflow_name": "ui-tier2",
        "entry_context": "test",
        "status": "completed",
        "success": 1,
        "failure_reason": None,
        "answer": "done",
        "conversation_summary": None,
        "conversation_traces": None,
        "started_at": "2026-09-07T10:00:00+00:00",
        "completed_at": "2026-09-07T10:00:01+00:00",
        "suspended_ms": suspended_ms,
        "continuation_of": None,
        "record_version": 1,
        "experiment_id": None,
        "task_id": None,
        "attempt": None,
        "claim_epoch": None,
        "server_incarnation": None,
        "record_json": json.dumps(record),
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


def _as_span(span):
    return tracing.Span(
        span_id=span["span_id"], trace_id=span["trace_id"], name=span["name"],
        kind=span["kind"], channel_id=span["channel_id"],
        parent_span_id=span["parent_span_id"], command_name=span["command_name"],
        context=span["context"], start_ns=span["start_ns"], end_ns=span["end_ns"],
        status=span["status"], attributes=span["attributes"],
    )


@pytest.fixture
def workflow_path(tmp_path, monkeypatch) -> str:
    monkeypatch.setenv("FASTWORKFLOW_STATE_ROOT", str(tmp_path / "state"))
    monkeypatch.setenv(obs.CAPTURE_PROFILE_VAR, "debug")
    wf = tmp_path / "ui_workflow"
    wf.mkdir()
    return str(wf)


@pytest.fixture
def seeded_db(workflow_path) -> str:
    db_path = state_paths.observability_db(workflow_path)
    store = obs.ObservabilityStore(db_path)
    controller = ExperimentController(
        workflow_path, store.store_identity(), migrate=False, external=True
    )

    controller.create_experiment(
        EXP, "ui tier 2", declared_tasks=1, declared_attempts=2,
        declarations=[(TASK, 1, "job-1"), (TASK, 2, "job-2")],
    )
    claims = {}
    for attempt, snapshot in ((1, SNAPSHOT_1), (2, SNAPSHOT_2)):
        channel = f"registered:{TASK}:{attempt}"
        bootstrap = controller.register_attempt(EXP, TASK, attempt, f"job-{attempt}", channel)
        claims[attempt] = controller.claim_attempt(
            bootstrap, server_incarnation=f"server-{attempt}", runtime_snapshot=snapshot,
        )
        store.start_attempt(EXP, TASK, attempt, channel, source_key=f"job-{attempt}")
        store.finish_attempt(EXP, TASK, attempt, outcome="pass", outcome_source="grader")
    store.record_evidence_segment(
        EXP, 1, "evr-tier2",
        {"valid": True, "problems": [], "observability": OBSERVABILITY, "in_process": False},
    )

    # A comparable baseline whose evidence run was captured under a different
    # span contract and retention setting, and a third experiment of another
    # shape that the store refuses to compare.
    for experiment_id, attempts, observability in (
        (BASE, 2, OBSERVABILITY_BASE), (ODD, 1, OBSERVABILITY)
    ):
        store.create_experiment(
            experiment_id, experiment_id, declared_tasks=1, declared_attempts=attempts,
        )
        for attempt in range(1, attempts + 1):
            store.start_attempt(experiment_id, TASK, attempt, f"{experiment_id}:{attempt}")
            store.finish_attempt(
                experiment_id, TASK, attempt, outcome="pass", outcome_source="grader"
            )
        store.record_evidence_segment(
            experiment_id, 1, f"evr-{experiment_id}",
            {"valid": True, "problems": [], "observability": observability},
        )

    redactor = store._store_redactor()
    conn = store._connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        assert store.upsert_turn_row(
            conn, _turn_row(TURN_A, f"registered:{TASK}:1", record=turn_a_record(),
                            claim=claims[1]),
            [], redactor,
        )
        assert store.upsert_turn_row(
            conn, _turn_row(TURN_B, f"registered:{TASK}:2", record=turn_b_record(),
                            claim=claims[2], suspended_ms=5_000),
            [], redactor,
        )
        store.upsert_span_rows(
            conn, [_as_span(s) for s in turn_a_spans() + turn_b_spans()], redactor
        )
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


def _get_json(server, path, expect=200):
    status, body = _get(server, path)
    assert status == expect, f"{path} -> {status}: {body[:300]!r}"
    return json.loads(body)


class TestLiveRoutes:
    def test_listed_turns_carry_signals_and_cost(self, server):
        turns = {t["turn_key"]: t for t in _get_json(server, "/api/turns?limit=50")["turns"]}
        a, b = turns[TURN_A], turns[TURN_B]
        assert a["decision_signals"]["intent_margin_min"] == 0.1
        assert a["decision_signals"]["asked_user"] == 1
        assert a["decision_signals"]["consequence_max"] == "high"
        assert a["llm_cost"] == {"calls": 2, "recorded": 1, "unrecorded": 1, "total": 0.0015}
        assert a["llm_calls_cut_at_limit"] == 0
        assert b["decision_signals"]["intent_margin_min"] is None
        assert b["llm_cost"]["total"] is None and b["llm_cost"]["unrecorded"] == 1

    def test_low_confidence_filter_counts_only_recorded_margins_below(self, server):
        keys = lambda path: [t["turn_key"] for t in _get_json(server, path)["turns"]]  # noqa: E731
        assert keys("/api/turns?low_confidence_below=0.2") == [TURN_A]
        assert keys("/api/turns?low_confidence_below=0.1") == []      # not strictly below
        assert keys("/api/turns?low_confidence_below=0.05") == []
        assert keys("/api/turns?low_confidence_below=1") == [TURN_A]  # B has no margin
        for bad in ("abc", "-0.1", "nan", "inf"):
            status, _ = _get(server, f"/api/turns?low_confidence_below={bad}")
            assert status == 400, bad

    def test_the_opened_turn_carries_its_ledger_signals_and_cost(self, server):
        turn = _get_json(server, f"/api/turn/{TURN_A}")["turn"]
        rows = turn["execution_ledger"]["rows"]
        assert [r["command_call_id"] for r in rows] == ["call-1", "call-1a", "call-2"]
        assert turn["execution_ledger"]["asked_user_outside_dispatch"] == 1
        assert turn["decision_signals"]["intent_margin_min"] == 0.1
        assert turn["llm_cost"]["total"] == 0.0015
        resumed = _get_json(server, f"/api/turn/{TURN_B}")["turn"]
        assert resumed["suspended_ms"] == 5_000
        rows = resumed["execution_ledger"]["rows"]
        assert [(r["command_call_id"], r["in_record"], r["status"], r["asked_user"])
                for r in rows] == [("call-b1", False, "cancelled", 1), ("call-b2", True, "ok", 0)]
        assert resumed["execution_ledger"]["rows_not_in_record"] == 1

    def test_experiment_detail_carries_provenance_with_git_revision_unrecorded(self, server):
        exp = _get_json(server, f"/api/experiment/{EXP}")["experiment"]
        provenance = exp["provenance"]
        assert _field(provenance, "git_revision")["recorded"] is False
        assert _field(provenance, "capture_profile", "experiment")["value"] == "debug"
        assert _field(provenance, "span_contract_version", "evidence_run")["value"] == 3
        assert _field(provenance, "span_contract_versions.fw.turn")["value"] == 1
        model = _field(provenance, "workflow_model_version", "runtime_snapshot")
        assert model["consistent"] is False and len(model["values"]) == 2
        assert _field(provenance, "workflow_fingerprint", "runtime_snapshot")["value"] == "sha256:same"

    def test_attempt_rows_carry_cost_beside_the_token_tally(self, server):
        rows = {r["attempt"]: r for r in _get_json(server, f"/api/experiment/{EXP}/attempts")["attempts"]}
        assert rows[1]["llm_cost"] == {"calls": 2, "recorded": 1, "unrecorded": 1, "total": 0.0015}
        assert rows[2]["llm_cost"] == {"calls": 1, "recorded": 0, "unrecorded": 1, "total": None}
        assert rows[1]["llm_calls_cut_at_limit"] == 0 and rows[1]["turn_count"] == 1

    def test_compare_quotes_provenance_differences_without_blocking(self, server, seeded_db):
        store = obs.ObservabilityStore(seeded_db, migrate=False)
        for experiment_id in (EXP, BASE, ODD):
            assert store.complete_experiment(experiment_id) == "complete"
        cmp = _get_json(server, f"/api/experiment/{EXP}/compare?baseline={BASE}")
        assert cmp["comparable"] is True
        by_key = {d["key"]: d for d in cmp["provenance_differences"]}
        assert by_key["span_contract_version"] == {
            "key": "span_contract_version", "source": "evidence_run",
            "treatment": 3, "baseline": 2,
        }
        assert by_key["config.FW_OBS_RETENTION_DAYS"]["baseline"] == "7"
        # snapshots exist on the treatment only: recorded vs not recorded
        assert by_key["workflow_fingerprint"] == {
            "key": "workflow_fingerprint", "source": "runtime_snapshot",
            "treatment": "sha256:same", "baseline": None,
        }
        # The store's own refusal (differing shape) still 409s, and the
        # differences ride along on that answer too.
        refused = _get_json(server, f"/api/experiment/{EXP}/compare?baseline={ODD}", expect=409)
        assert refused["comparable"] is False
        assert any("declared shapes differ" in p for p in refused["problems"])
        assert {d["key"] for d in refused["provenance_differences"]} >= {"workflow_fingerprint"}
        assert "span_contract_version" not in {d["key"] for d in refused["provenance_differences"]}

    def test_unknown_experiment_still_404s(self, server):
        status, _ = _get(server, "/api/experiment/no-such/compare?baseline=" + BASE)
        assert status == 404


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
                "workspace_id": "workspace-ui-tier2",
                "label": "UI tier 2 archive",
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
                            {"segment_id": "seg-main", "store_id": "sealed",
                             "local_experiment_id": EXP},
                        ],
                    }
                ],
                "projected_attempts": [
                    {
                        "logical_attempt": {
                            "experiment_id": "historical", "task_id": "joined", "attempt": 1,
                        },
                        "attempt_refs": [
                            {
                                "store_id": "sealed", "local_experiment_id": EXP,
                                "task_id": TASK, "attempt": 1,
                                "turn_ref": {"store_id": "sealed", "logical_turn_key": TURN_A},
                            }
                        ],
                        "turn_refs": [{"store_id": "sealed", "logical_turn_key": TURN_A}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest, archive


def test_workspace_readers_expose_the_archived_experiment_and_attempts(workspace_manifest):
    manifest, _ = workspace_manifest
    workspace = load_observability_workspace(manifest)
    experiment = workspace.experiment("sealed", EXP)
    assert experiment["experiment_id"] == EXP and experiment["capture_profile"] == "debug"
    assert [s["evidence_run_id"] for s in experiment["evidence_runs"]] == ["evr-tier2"]
    assert workspace.experiment("sealed", "no-such") is None
    rows = workspace.attempts_in_store("sealed", EXP)
    assert [r["attempt"] for r in rows] == [1, 2]
    assert rows[0]["runtime_snapshot"] == SNAPSHOT_1
    assert workspace.attempts_in_store("sealed", "no-such") == []
    for bad in ("", "other"):
        with pytest.raises(UnknownWorkspaceStore):
            workspace.experiment(bad, EXP)
        with pytest.raises(UnknownWorkspaceStore):
            workspace.attempts_in_store(bad, EXP)


class TestWorkspaceRoutes:
    @pytest.fixture
    def ws_server(self, workspace_manifest):
        manifest, archive = workspace_manifest
        srv, thread = _serve(archive["path"], workspace_manifest_path=str(manifest))
        yield srv
        srv.shutdown()
        thread.join(timeout=5)

    def test_segments_carry_the_archived_provenance(self, ws_server):
        segments = _get_json(ws_server, "/api/workspace/experiment/logical/segments")["segments"]
        assert len(segments) == 1
        provenance = segments[0]["provenance"]
        assert _field(provenance, "git_revision")["recorded"] is False
        assert _field(provenance, "span_contract_version", "evidence_run")["value"] == 3
        assert _field(provenance, "workflow_model_version", "runtime_snapshot")["consistent"] is False
        assert segments[0]["evidence"]["state"] == "valid"

    def test_attempts_and_turn_refs_carry_signals_and_cost(self, ws_server):
        rows = {r["attempt"]: r for r in _get_json(
            ws_server, "/api/workspace/experiment/logical/attempts")["attempts"]}
        assert rows[1]["llm_cost"]["total"] == 0.0015
        assert rows[2]["llm_cost"]["total"] is None and rows[2]["llm_cost"]["unrecorded"] == 1
        ref = rows[1]["turn_refs"][0]
        assert ref["decision_signals"]["intent_margin_min"] == 0.1
        assert ref["decision_signals"]["consequence_max"] == "high"
        assert ref["llm_cost"]["recorded"] == 1 and ref["llm_calls_cut_at_limit"] == 0
        assert rows[2]["turn_refs"][0]["decision_signals"]["intent_margin_min"] is None

    def test_projected_attempts_are_stamped_once(self, ws_server):
        projected = _get_json(
            ws_server, "/api/workspace/projected_attempts?experiment=historical"
        )["projected_attempts"]
        assert len(projected) == 1
        row = projected[0]
        assert row["llm_cost"] == {"calls": 2, "recorded": 1, "unrecorded": 1, "total": 0.0015}
        assert row["resolved_turns"][0]["decision_signals"]["asked_user"] == 1
        assert row["resolved_sources"][0]["resolved_turn"]["llm_cost"]["total"] == 0.0015

    def test_scoped_turn_read_carries_the_ledger(self, ws_server):
        turn = _get_json(ws_server, f"/api/workspace/turn/sealed/{TURN_B}")["turn"]
        rows = turn["execution_ledger"]["rows"]
        assert [(r["command_call_id"], r["in_record"]) for r in rows] == [
            ("call-b1", False), ("call-b2", True)
        ]
        assert turn["decision_signals"]["asked_user"] == 1
        assert turn["llm_cost"]["total"] is None
        status, _ = _get(ws_server, f"/api/turn/{TURN_B}")
        assert status == 400   # unscoped reads stay refused in workspace mode

    def test_archive_stays_byte_identical_after_the_reads(self, ws_server, workspace_manifest):
        _, archive = workspace_manifest
        _get_json(ws_server, "/api/workspace/experiment/logical/segments")
        _get_json(ws_server, "/api/workspace/experiment/logical/attempts")
        _get_json(ws_server, f"/api/workspace/turn/sealed/{TURN_A}")
        assert hashlib.sha256(Path(archive["path"]).read_bytes()).hexdigest() == archive["sha256"]


def test_annotate_turn_detail_reads_only_what_it_is_handed():
    turn = {"record": turn_a_record()}
    annotate_turn_detail(turn, turn_a_spans())
    assert len(turn["execution_ledger"]["rows"]) == 3
    assert turn["decision_signals"]["intent_margin_min"] == 0.1
    assert turn["llm_cost"]["total"] == 0.0015
    assert turn["llm_calls_cut_at_limit"] == 0


# ----------------------------------------------------------------------
# The page
# ----------------------------------------------------------------------


class TestPage:
    def test_render_functions_and_strings_are_present(self):
        page = run_chatbot_server.load_index_html()
        # (a) the ledger at turn level, rows opening their span
        assert b"function renderExecutionLedger(container, turn, openSpan)" in page
        assert b'"Execution ledger"' in page
        assert b"renderExecutionLedger(ledgerCard, state.turn, openSpanInTree)" in page
        assert b"renderExecutionLedger(ledgerCard, turn, null)" in page
        # The turn-wide record lives on the Execution stage, found by its
        # phase key rather than its title, with the ledger folded behind a link.
        assert b"function turnRecordHome()" in page
        assert b'child.kind === "phase" && child.phase === PHASE_EXECUTION' in page
        assert b'el("details", "card ledgerDisclosure")' in page
        assert b'"no span recorded"' in page
        assert b"the record lists only dispatches since the last resume" in page
        assert b"function openSpanInTree(spanId)" in page
        # (b) decision chips remain; hierarchy navigation replaced the sidebar filters
        assert b"function signalChips(signals)" in page
        assert b"var LOW_CONFIDENCE_DEFAULT_MARGIN = 0.2" in page
        assert b'id="fLowConf"' not in page and b'id="fLowConfMargin"' not in page
        assert b'id="convList"' in page
        # Rail rows are label-only (owner decision 2026-09-29); signals show in the turn view.
        assert b"appendSignalChips(container, turn.decision_signals)" in page # turn header
        assert b"appendSignalChips(sub, stamps.decision_signals)" in page     # workspace links
        assert b'"asked the user"' in page and b'"consequence "' in page
        # (c) the collapsed provenance fold and the comparability check
        assert b"function renderProvenance(container, provenance, label)" in page
        assert b"function renderProvenanceDifferences(container, differences)" in page
        # Experiment detail stays concise; provenance remains available to the
        # API, workspace segments, and baseline comparison diagnostics.
        assert b'renderProvenance(card, exp.provenance, "provenance")' not in page
        assert b"renderProvenance(segBox, segment.provenance," in page
        assert page.count(b"renderProvenanceDifferences(container, cmp.provenance_differences)") == 2
        assert b'"not recorded"' in page
        assert b"the comparison is shown regardless" in page
        # (d) cost beside tokens, never zero for unknown
        assert b"function fmtCostAmount(cost)" in page
        assert b'"cost not recorded"' in page
        assert b"function fmtCost(duration, tokens, cost)" in page
        # A level charges its own call only when no other call is already
        # charged for the same provider response, whether that other call nests
        # inside it or sits beside it. Charging both doubled the tree's total
        # (fix-9eg.5, then shared-response-cost for the sibling shape). This
        # assertion previously pinned the nesting-only form.
        assert b"cost: addCost(own.cost, sumCost(children))" in page
        assert b"function chargeFolds(byId)" in page
        assert b'own = { tokens: sharedTokens(), cost: sharedCost(), cache: spanCache(span) };' in page
        # Rail rows carry no cost chip (label-only rail, owner decision 2026-09-29).
        # The attempt row's chips come from the EVIDENCE view keyed by attempt
        # (`extra`), not from the decision view's run row (`row`) — that one
        # carries Best run and comparability and no cost at all, so reading
        # `row.llm_cost` there would have shown nothing for every attempt.
        assert b"appendCostChip(subLine, extra.llm_cost)" in page
        assert b"appendCostChip(outcomeLine, row.llm_cost)" in page
        assert b'row("LLM cost", fmtCostAmount(turn.llm_cost))' in page
        # Inside #detail, not a new panel; the rules the page already keeps.
        assert b"innerHTML" not in page
        assert b"https://" not in page
        assert page.count(b'id="detail"') == 1

    def test_page_still_serves_with_its_hash_sourced_csp(self, server):
        status, body = _get(server, "/")
        assert status == 200
        assert b"renderExecutionLedger" in body


def test_execution_stage_layout_dom(server):
    """TURN_A runs an agent loop, so its record moves to the Execution stage;
    TURN_B dispatches commands directly and keeps it on the turn page."""
    dependency = os.environ.get("TEST_JSDOM_ROOT")
    if not dependency:
        pytest.skip("Set TEST_JSDOM_ROOT to run DOM integration with jsdom")
    script = Path(__file__).with_name("chatbot_execution_stage_layout_dom.cjs")
    result = subprocess.run(
        [
            "node",
            str(script),
            dependency,
            f"http://127.0.0.1:{server.port}/?token={server.token}",
            TURN_A,
            TURN_B,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
