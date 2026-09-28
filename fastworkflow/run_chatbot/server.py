"""fastWorkflow Chatbot debug mode: stdlib-only localhost read-only HTTP layer.

Design invariants (docs/fastworkflow_observability_studio_design.md §3.4):

- Access control: binds 127.0.0.1 only; a per-launch random bearer
  token (``secrets.token_urlsafe``) embedded in the printed URL (Jupyter
  pattern) is required on EVERY request (Authorization header or ``?token=``),
  compared in constant time (``hmac.compare_digest``); a strict Host/Origin
  allowlist rejects everything non-loopback with 403. Loopback hosts
  (``127.0.0.1`` / ``localhost`` / ``[::1]``) pass on ANY port — port
  forwarders (WSL relays, IDE port forwards) legitimately re-expose the
  server on a different local port — while the loopback-only rule is what
  defeats DNS rebinding, and the token stays the authentication.
- Rendering safety: the SPA page ships with a restrictive CSP
  (inline script allowed only via its own sha256 hashes — the page is one
  self-contained file); artifact responses carry
  ``default-src 'none'; sandbox`` so direct navigation is inert; the read
  layer only calls ObservabilityStore methods (parameterized queries).
- Read discipline: per-request store reads — every ObservabilityStore
  method opens its own short-lived connection, no held cursors — so WAL
  checkpointing by the writer never starves.
- Packaging: stdlib-only. This module must never import
  fastapi/uvicorn or any third-party HTTP dependency.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import importlib.resources
import json
import logging
import os
import re
import secrets
import signal
import sqlite3
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from collections.abc import Iterable, Iterator, Mapping
from typing import Any, Optional
from urllib.parse import parse_qs, quote, unquote, urlsplit

from fastworkflow import state_paths
from fastworkflow.benchmark import setup as benchmark_setup
from fastworkflow.experiment.setup import ExperimentSetups, SetupConflict
from fastworkflow.benchmark.catalog import (
    BenchmarkAlreadyExistsError,
    BenchmarkManifestError,
    list_benchmarks,
    list_versions,
    load_analysis,
    load_version,
    write_analysis,
    write_version,
)
from fastworkflow.observability.store import (
    FEATURE_EXPERIMENTS_V1,
    ExperimentNotFound,
    IncompatibleObservabilityDB,
    ObservabilityStore,
    ReadOnlyObservabilityStore,
)
from fastworkflow.observability.workspace import (
    WORKSPACE_SCHEMA,
    ObservabilityWorkspace,
    UnknownLogicalExperiment,
    UnknownWorkspaceStore,
    WorkspaceBusyError,
    WorkspaceError,
    WorkspaceIntegrityError,
    load_observability_workspace,
)
from fastworkflow.review.sidecar import (
    ReviewAuthorizationError,
    ReviewNotFoundError,
    ReviewSidecar,
    ReviewValidationError,
    project_review_trace,
    project_review_turn,
)
from fastworkflow.run_chatbot import launcher

logger = logging.getLogger(__name__)

# Tolerates attributes (e.g. a future type="module") so adding one cannot
# silently produce an empty hash list — which would fail closed and brick the
# page. ChatbotServer.__init__ additionally asserts extraction succeeded.
_SCRIPT_RE = re.compile(rb"<script\b[^>]*>(.*?)</script>", re.DOTALL)

# Content types that may contain active content: only ever rendered inside a
# sandboxed iframe by the SPA; direct responses are additionally sandboxed via
# CSP (see _artifact_headers) [R22].
_HTMLISH_TYPES = ("text/html", "application/xhtml+xml", "image/svg+xml")

# ----------------------------------------------------------------------
# Derived fields for the SPA (fix-49m.6)
# ----------------------------------------------------------------------
#
# Three things the debug UI shows are not columns: whether an `fw.llm.call`
# stopped at its output cap, how many such calls a turn or an attempt made,
# and the verdict an experiment's evidence segments add up to. They are derived
# here, in the read layer, from ObservabilityStore reads only [R12] -- never
# from a query of this module's own -- so the workspace's archived stores
# render them through the very same functions. (The fourth, the attempt's
# runtime snapshot, IS a column: `_decode_attempt_row` already exposes it.)

SPAN_LLM_CALL = "fw.llm.call"

EVIDENCE_VALID = "valid"
EVIDENCE_INVALID = "invalid"
EVIDENCE_UNRECORDED = "unrecorded"


def _mapping_attr(value: Any) -> Optional[dict[str, Any]]:
    """A span attribute as a dict, or None.

    `usage` and `call_kwargs` are persisted as JSON text
    (`dspy_logger._json_text`); the `attributes` column itself is text straight
    off the row and a dict once a route has decoded it. Both forms are
    accepted; anything that is not a mapping answers None.
    """
    if isinstance(value, (str, bytes)):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    return value if isinstance(value, dict) else None


def _exact_int(value: Any) -> Optional[int]:
    """An int, or None. bool is excluded on purpose: `True == 1` would let a
    malformed payload read as a one-token call cut at a one-token cap."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def llm_call_cut_at_limit(span: Mapping[str, Any]) -> bool:
    """Whether an `fw.llm.call` produced exactly `call_kwargs.max_tokens` tokens.

    A completion that stops exactly at the cap stopped because of the cap, not
    because the model finished. `call_kwargs` is flat -- `call_kwargs.max_tokens`
    (`tests/test_dspy_call_kwargs_shape.py`). Missing usage, a missing cap, a
    non-integral value on either side or a non-positive cap all answer False:
    the chip must never be a false positive, so absence of evidence is "no".
    """
    if span.get("name") != SPAN_LLM_CALL:
        return False
    attributes = _mapping_attr(span.get("attributes"))
    if attributes is None:
        return False
    usage = _mapping_attr(attributes.get("usage"))
    call_kwargs = _mapping_attr(attributes.get("call_kwargs"))
    if usage is None or call_kwargs is None:
        return False
    produced = _exact_int(usage.get("completion_tokens"))
    cap = _exact_int(call_kwargs.get("max_tokens"))
    if produced is None or cap is None or cap <= 0:
        return False
    return produced == cap


def count_llm_calls_cut_at_limit(spans: Iterable[Mapping[str, Any]]) -> int:
    return sum(1 for span in spans if llm_call_cut_at_limit(span))


def annotate_turn_rows(
    store: ReadOnlyObservabilityStore, turns: list[dict[str, Any]]
) -> None:
    """Stamp each listed turn with its cut-at-limit tally, decision signals
    and cost roll-up, from one read of its spans (tiers 1 and 2).

    The spans for the whole page come from one bulk read, not one query per
    listed turn: the rail asks for up to 500 turns and refreshes on a timer,
    so per-turn reads meant ~500 round trips a refresh. The stamps
    themselves are unchanged -- `turn_span_stamps` still sees exactly the rows
    `get_spans` would have handed it for that turn.
    """
    spans_by_turn = store.spans_for_turns(turn["turn_key"] for turn in turns)
    for turn in turns:
        turn.update(turn_span_stamps(spans_by_turn.get(turn["turn_key"]) or []))


def evidence_verdict(evidence_runs: Optional[Iterable[Mapping[str, Any]]]) -> dict[str, Any]:
    """The verdict the UI badges an experiment and its attempts with.

    Built from the experiment's stored evidence segments
    (`experiment.get("evidence_runs")`, one per `evidence_run()`): the `valid`
    column decides -- it is monotone in invalidity, so it outranks whatever the
    latest record says -- and the reasons are the record's `problems` list,
    quoted verbatim. A valid segment can still carry problems (dropped spans
    leave a run valid); those are reported as `warnings`, so a reader sees
    "valid, with incomplete detail on these turns" rather than a clean badge.
    An experiment with no segment is `unrecorded`, which is neither verdict.
    """
    segments: list[dict[str, Any]] = []
    problems: list[str] = []
    warnings: list[str] = []
    for segment in evidence_runs or []:
        record = segment.get("record")
        if not isinstance(record, dict):
            record = {}
        stored = record.get("problems")
        stored_problems = (
            [str(problem) for problem in stored] if isinstance(stored, list) else []
        )
        valid = bool(segment.get("valid"))
        delta = record.get("writer_health_delta")
        segments.append(
            {
                "seq": segment.get("seq"),
                "evidence_run_id": segment.get("evidence_run_id"),
                "valid": valid,
                "problems": stored_problems,
                "writer_health_delta": delta if isinstance(delta, dict) else None,
                "in_process": record.get("in_process"),
                "started_at": segment.get("started_at"),
                "completed_at": segment.get("completed_at"),
            }
        )
        (warnings if valid else problems).extend(stored_problems)
    if not segments:
        state = EVIDENCE_UNRECORDED
    elif all(segment["valid"] for segment in segments):
        state = EVIDENCE_VALID
    else:
        state = EVIDENCE_INVALID
    return {
        "state": state,
        "segments": segments,
        "problems": problems,
        "warnings": warnings,
    }


def annotate_attempt_rows(
    store: ReadOnlyObservabilityStore,
    rows: list[dict[str, Any]],
    verdict: dict[str, Any],
) -> None:
    """Stamp attempt rows with their token-limit tally and the evidence verdict.

    The verdict is the experiment's: a segment records the interval a batch of
    attempts ran in, not which attempt it covered, so the honest per-attempt
    badge is the experiment's verdict and not a guess at a mapping.
    """
    for row in rows:
        turns = store.list_turns(
            experiment_id=row["experiment_id"],
            task_id=row["task_id"],
            attempt=int(row["attempt"]),
            limit=10_000,
        )
        row["turn_count"] = len(turns)
        spans_by_turn = store.spans_for_turns(turn["turn_key"] for turn in turns)
        stamps = [
            turn_span_stamps(spans_by_turn.get(turn["turn_key"]) or [])
            for turn in turns
        ]
        row["llm_calls_cut_at_limit"] = sum(
            stamp["llm_calls_cut_at_limit"] for stamp in stamps
        )
        row["llm_cost"] = merge_cost_rollups(stamp["llm_cost"] for stamp in stamps)
        row["evidence"] = verdict


def _workspace_segment_verdicts(
    workspace: ObservabilityWorkspace, experiment_id: str
) -> dict[str, dict[str, Any]]:
    return {
        segment["segment_id"]: evidence_verdict(
            workspace.evidence_runs(
                segment["store_id"], segment["local_experiment_id"]
            )
        )
        for segment in workspace.segments(experiment_id)
    }


def _workspace_span_cache(
    workspace: ObservabilityWorkspace, refs: Iterable[tuple[Any, Any]]
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """``{(store_id, logical_turn_key): spans}`` for many refs, one bulk read
    per store instead of one `trace` call per ref.

    Refs missing either half are dropped here rather than raising: the caller
    already treats an unresolvable ref as "nothing to tally", and a store that
    a manifest no longer names must not break the rest of the answer.
    """
    by_store: dict[str, list[str]] = {}
    for store_id, key in refs:
        if store_id and key:
            by_store.setdefault(str(store_id), []).append(str(key))
    cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for store_id, keys in by_store.items():
        for key, spans in workspace.traces(store_id, keys).items():
            cache[(store_id, key)] = spans
    return cache


def annotate_workspace_attempts(
    workspace: ObservabilityWorkspace,
    rows: list[dict[str, Any]],
    verdict_by_segment: Mapping[str, dict[str, Any]],
) -> None:
    """The workspace twin of `annotate_attempt_rows`, over scoped trace reads.

    Every ref on every row is read in one pass per store; the stamps
    are the ones `trace` would have produced ref by ref."""
    spans_by_ref = _workspace_span_cache(
        workspace,
        (
            (ref.get("store_id"), ref.get("logical_turn_key"))
            for row in rows
            for ref in row.get("turn_refs") or []
        ),
    )
    for row in rows:
        total = 0
        costs = []
        for ref in row.get("turn_refs") or []:
            ref.update(turn_span_stamps(
                spans_by_ref.get((ref["store_id"], ref["logical_turn_key"])) or []
            ))
            total += ref["llm_calls_cut_at_limit"]
            costs.append(ref["llm_cost"])
        row["turn_count"] = len(row.get("turn_refs") or [])
        row["llm_calls_cut_at_limit"] = total
        row["llm_cost"] = merge_cost_rollups(costs)
        row["evidence"] = verdict_by_segment.get(
            row.get("segment_id"), evidence_verdict([])
        )


def annotate_projected_attempts(
    workspace: ObservabilityWorkspace, rows: list[dict[str, Any]]
) -> None:
    """Projected history rows: tally the resolved turns once each, and badge
    every resolved source with the verdict its own store persisted.

    The spans behind those tallies are read in one pass per store up front;
    a projection that resolves the same turn from several sources
    then costs one lookup, not one query, per mention."""

    def _resolved_turns(row: Mapping[str, Any]) -> Iterator[Any]:
        yield from row.get("resolved_turns") or []
        for source in row.get("resolved_sources") or []:
            yield source.get("resolved_turn")

    spans_by_ref = _workspace_span_cache(
        workspace,
        (
            (turn.get("store_id"), turn.get("logical_turn_key"))
            for row in rows
            for turn in _resolved_turns(row)
            if isinstance(turn, dict)
        ),
    )
    for row in rows:
        seen: set[tuple[str, str]] = set()
        total = 0
        costs: list[dict[str, Any]] = []

        def tally(turn: Any) -> None:
            nonlocal total
            if not isinstance(turn, dict):
                return
            store_id = turn.get("store_id")
            key = turn.get("logical_turn_key")
            if not store_id or not key:
                return
            turn.update(turn_span_stamps(spans_by_ref.get((store_id, key)) or []))
            if (store_id, key) not in seen:
                seen.add((store_id, key))
                total += turn["llm_calls_cut_at_limit"]
                costs.append(turn["llm_cost"])

        for turn in row.get("resolved_turns") or []:
            tally(turn)
        for source in row.get("resolved_sources") or []:
            tally(source.get("resolved_turn"))
            local_id = source.get(
                "local_experiment_id", source.get("experiment_id")
            )
            if source.get("store_id") and local_id is not None:
                source["evidence"] = evidence_verdict(
                    workspace.evidence_runs(str(source["store_id"]), str(local_id))
                )
        row["llm_calls_cut_at_limit"] = total
        row["llm_cost"] = merge_cost_rollups(costs)


# ----------------------------------------------------------------------
# Derived fields for the SPA, tier 2 (fix-aou)
# ----------------------------------------------------------------------
#
# Four more things the debug UI shows that are not columns, derived here from
# ObservabilityStore reads only [R12] so the workspace's archives render them
# through the same functions:
#
# (a) a turn's execution ledger -- every dispatch, joined on `command_call_id`
#     between the turn record's `execution_records` refs and the trace's
#     `fw.command.execute` spans (and the span-less inner hops those spans
#     file under `child_calls`);
# (b) the turn's decision signals -- the least confident intent resolution's
#     top-k margin, whether the user was asked, and the worst consequence
#     class any dispatch was assessed at -- and the low-confidence filter
#     they feed;
# (c) an experiment's provenance, flattened from the evidence-run records,
#     the experiment row and the attempts' runtime snapshots, plus the
#     field-by-field difference between two experiments' provenance;
# (d) cost roll-ups from the `cost` attribute on `fw.llm.call`.
#
# Every one of them says "not recorded" for an absence rather than inventing
# a value: a missing margin is no chip and not a low-confidence turn, and a
# missing cost is never a zero.

SPAN_COMMAND_EXECUTE = "fw.command.execute"
SPAN_AGENT_TOOL_CALL = "fw.agent.tool_call"
SPAN_ASK_USER = "fw.ask_user"
SPAN_NLU_INTENT = "fw.nlu.intent"

# decision_signals.SignalKind member for the classifier's top-1 minus top-2
# probability; the polarity table there says higher is more confident.
SIGNAL_TOPK_MARGIN = "classifier-topk-margin"
# decision_signals._CONSEQUENCE_ORDER, worst last. Restated as data here so
# that reading a stored record does not import the capture module.
CONSEQUENCE_ORDER = ("none", "low", "medium", "high", "critical")

# There is no calibrated threshold on record: decision_signals is capture-only
# by design (FW-REQ-021 clause 4), and the trained `ambiguous_threshold.json`
# files bound classifier CONFIDENCE, not the margin. So the filter's default
# is a viewing aid the UI names as such, and the user may set another.
LOW_CONFIDENCE_DEFAULT_MARGIN = 0.2

PROVENANCE_NOT_RECORDED = "not recorded"

# The experiment row's own provenance-bearing columns.
_EXPERIMENT_PROVENANCE_COLUMNS = (
    "capture_profile",
    "capture_policy_version",
    "workflow_name",
    "benchmark_id",
    "benchmark_version",
    "benchmark_digest_sha256",
)
# The attempt's stamped runtime snapshot (runtime_readiness) keys that pin
# what ran; `effective_features` is flattened one level.
_SNAPSHOT_PROVENANCE_KEYS = (
    "workflow_fingerprint",
    "workflow_model_version",
    "workflow_model_legacy_layout",
    "workflow_scope_rule_version",
    "command_surface_count",
    "capture_profile",
    "capture_policy_version",
)


def _span_attributes(span: Mapping[str, Any]) -> dict[str, Any]:
    return _mapping_attr(span.get("attributes")) or {}


def _finite_number(value: Any) -> Optional[float]:
    """A finite float, or None. bool is excluded: True is not a cost of 1."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _text_or_none(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


# -- (a) execution ledger ----------------------------------------------------


def execution_ledger(
    record: Any, spans: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """The ordered ledger of every dispatch in one turn.

    Joined on `command_call_id` from two sources that the substrate keeps on
    purpose in two tiers (turn.py, "Additive turn-level capture"): the turn
    record's `execution_records` refs (durable skeleton: id, parent, ordinal,
    span_id) and the trace's `fw.command.execute` spans (best-effort detail:
    name, context, status, timing, and the `child_calls` ledger of span-less
    inner hops). A dispatch known to either appears once.

    `status` is the span's own status column, which is what
    `tracing.status_for_dispatch_exception` wrote: ok, error, or cancelled for
    a control signal (an ask-user suspension or a cancellation). It is quoted,
    never restated; a dispatch with no span has no status and says so.

    A resumed turn is one ledger, not a restart: the recorder is per process
    (see ``workflow_execution_context``), so the terminal record lists
    only the dispatches since the last resume, while the earlier ones persist
    as spans under the same trace. Rows are ordered by their span's start
    time, a span-less child directly after its parent, and record-only rows
    after the timed ones in record order -- so the pre-suspension dispatches
    come first and `in_record` false tells the reader which rows the record
    itself no longer lists.
    """
    span_list = list(spans)
    refs: list[dict[str, Any]] = []
    if isinstance(record, dict):
        raw = record.get("execution_records")
        if isinstance(raw, list):
            refs = [
                ref
                for ref in raw
                if isinstance(ref, dict) and _text_or_none(ref.get("command_call_id"))
            ]
    by_span_id: dict[str, Mapping[str, Any]] = {
        span["span_id"]: span for span in span_list if _text_or_none(span.get("span_id"))
    }

    entries: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    def entry_for(call_id: str) -> dict[str, Any]:
        if call_id not in entries:
            entries[call_id] = {
                "command_call_id": call_id,
                "parent_call_id": None,
                "command_ordinal": None,
                "span_id": None,
                "command_name": None,
                "context": None,
                "status": None,
                "success": None,
                "start_ns": None,
                "duration_ns": None,
                "in_record": False,
                "span_recorded": False,
                "child_call": False,
                "asked_user": 0,
            }
            order.append(call_id)
        return entries[call_id]

    def apply_span(entry: dict[str, Any], span: Mapping[str, Any]) -> None:
        attributes = _span_attributes(span)
        entry["span_id"] = span.get("span_id")
        entry["span_recorded"] = True
        entry["command_name"] = _text_or_none(span.get("command_name")) or entry["command_name"]
        entry["context"] = _text_or_none(span.get("context")) or entry["context"]
        entry["status"] = _text_or_none(span.get("status"))
        if isinstance(attributes.get("success"), bool):
            entry["success"] = attributes["success"]
        parent = attributes.get("parent_call_id")
        if _text_or_none(parent):
            entry["parent_call_id"] = parent
        start = _exact_int(span.get("start_ns"))
        end = _exact_int(span.get("end_ns"))
        entry["start_ns"] = start
        entry["duration_ns"] = end - start if start is not None and end is not None else None

    for ref in refs:
        entry = entry_for(str(ref["command_call_id"]))
        entry["in_record"] = True
        if _text_or_none(ref.get("parent_call_id")):
            entry["parent_call_id"] = ref["parent_call_id"]
        ordinal = _exact_int(ref.get("command_ordinal"))
        if ordinal is not None:
            entry["command_ordinal"] = ordinal
        if _text_or_none(ref.get("span_id")):
            entry["span_id"] = ref["span_id"]

    execute_spans = [s for s in span_list if s.get("name") == SPAN_COMMAND_EXECUTE]
    for span in execute_spans:
        attributes = _span_attributes(span)
        call_id = _text_or_none(attributes.get("command_call_id"))
        if call_id is None:
            continue
        entry = entry_for(call_id)
        if not entry["span_recorded"]:
            apply_span(entry, span)
        children = attributes.get("child_calls")
        for child in children if isinstance(children, list) else []:
            if not isinstance(child, dict):
                continue
            child_id = _text_or_none(child.get("call_id"))
            if child_id is None:
                continue
            child_entry = entry_for(child_id)
            child_entry["child_call"] = True
            child_entry["parent_call_id"] = (
                _text_or_none(child.get("parent_call_id")) or call_id
            )
            if _text_or_none(child.get("command_name")):
                child_entry["command_name"] = child["command_name"]
    # A ref whose span_id names an execute span that did not carry the id.
    for entry in entries.values():
        if entry["span_recorded"] or not entry["span_id"]:
            continue
        span = by_span_id.get(entry["span_id"])
        if span is not None and span.get("name") == SPAN_COMMAND_EXECUTE:
            apply_span(entry, span)

    # Whether a dispatch produced an ask-user entry: an fw.ask_user span whose
    # ancestry reaches that dispatch's execute span. One raised outside any
    # dispatch (the agent loop asking, which is where the trial's all sat) is
    # counted on the turn instead, never attributed to a row by guesswork.
    asked_outside = 0
    for span in span_list:
        if span.get("name") != SPAN_ASK_USER:
            continue
        cursor = span.get("parent_span_id")
        owner: Optional[str] = None
        hops = 0
        while cursor and cursor in by_span_id and hops < 10_000:
            parent = by_span_id[cursor]
            if parent.get("name") == SPAN_COMMAND_EXECUTE:
                owner = _text_or_none(_span_attributes(parent).get("command_call_id"))
                break
            cursor = parent.get("parent_span_id")
            hops += 1
        if owner is not None and owner in entries:
            entries[owner]["asked_user"] += 1
        else:
            asked_outside += 1

    def start_of(entry: dict[str, Any]) -> Optional[int]:
        if entry["start_ns"] is not None:
            return entry["start_ns"]
        parent = entries.get(entry["parent_call_id"]) if entry["parent_call_id"] else None
        if parent is not None and parent["start_ns"] is not None:
            return parent["start_ns"]
        return None

    def sort_key(call_id: str) -> tuple[Any, ...]:
        entry = entries[call_id]
        start = start_of(entry)
        # A span-less child borrows its parent's start and sorts just after
        # it; with no timestamp anywhere the record's ordinal is the order.
        nested = (
            start is not None
            and entry["start_ns"] is None
            and entry["parent_call_id"] is not None
        )
        ordinal = entry["command_ordinal"]
        return (
            0 if start is not None else 1,
            start if start is not None else 0,
            1 if nested else 0,
            ordinal if ordinal is not None else order.index(call_id),
        )

    rows = []
    for position, call_id in enumerate(sorted(order, key=sort_key), start=1):
        row = dict(entries[call_id])
        row["position"] = position
        rows.append(row)
    return {
        "rows": rows,
        "record_rows": len(refs),
        "span_rows": len(execute_spans),
        "rows_not_in_record": sum(1 for row in rows if not row["in_record"]),
        "rows_without_span": sum(1 for row in rows if not row["span_recorded"]),
        "asked_user_outside_dispatch": asked_outside,
    }


# -- (b) decision signals ------------------------------------------------------


def turn_decision_signals(spans: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """The turn's intent-resolution chips, from what the NLU spans recorded.

    `intent_margin_min` is the smallest `classifier-topk-margin` among the
    turn's `fw.nlu.intent` decisions -- the least confident resolution -- and
    None when no decision carried one (an exact-prefix match records
    `signals_absent_reason: deterministic-resolution` and no number, which is
    not confidence 1.0 and not low confidence either). `asked_user` counts
    `fw.ask_user` spans. `consequence_max` is the worst `consequence_class`
    any `fw.command.execute` assessed (falling back to `fw.agent.tool_call`
    when a trace has no execute spans), None when none was assessed.
    """
    margins: list[float] = []
    intent_decisions = 0
    decisions_without_margin = 0
    asked = 0
    execute_classes: list[str] = []
    tool_call_classes: list[str] = []
    for span in spans:
        name = span.get("name")
        attributes = _span_attributes(span)
        if name == SPAN_NLU_INTENT:
            uncertainty = _mapping_attr(attributes.get("decision_uncertainty"))
            if uncertainty is None:
                continue
            intent_decisions += 1
            found = False
            signals = uncertainty.get("signals")
            for signal in signals if isinstance(signals, list) else []:
                if not isinstance(signal, dict) or signal.get("kind") != SIGNAL_TOPK_MARGIN:
                    continue
                value = _finite_number(signal.get("value"))
                if value is not None:
                    margins.append(value)
                    found = True
            if not found:
                decisions_without_margin += 1
        elif name == SPAN_ASK_USER:
            asked += 1
        elif name in (SPAN_COMMAND_EXECUTE, SPAN_AGENT_TOOL_CALL):
            consequence = _mapping_attr(attributes.get("consequence"))
            cls = consequence.get("consequence_class") if consequence else None
            if isinstance(cls, str) and cls in CONSEQUENCE_ORDER:
                (execute_classes if name == SPAN_COMMAND_EXECUTE else tool_call_classes).append(cls)
    classes = execute_classes or tool_call_classes
    return {
        "intent_margin_min": min(margins) if margins else None,
        "intent_margin_decisions": len(margins),
        "intent_decisions": intent_decisions,
        "intent_decisions_without_margin": decisions_without_margin,
        "asked_user": asked,
        "consequence_max": (
            max(classes, key=CONSEQUENCE_ORDER.index) if classes else None
        ),
        "consequence_assessed": len(classes),
    }


def is_low_confidence(signals: Mapping[str, Any], threshold: float) -> bool:
    """Below the threshold on a RECORDED margin only: no signal, not counted."""
    margin = _finite_number(signals.get("intent_margin_min"))
    return margin is not None and margin < threshold


# -- (d) cost roll-ups ----------------------------------------------------------


def llm_call_cost(span: Mapping[str, Any]) -> Optional[float]:
    """The `cost` an `fw.llm.call` recorded (dspy_logger copies the DSPy
    history entry's `cost`), or None when it recorded none. A negative or
    non-numeric value is not a cost and answers None too."""
    if span.get("name") != SPAN_LLM_CALL:
        return None
    cost = _finite_number(_span_attributes(span).get("cost"))
    return cost if cost is not None and cost >= 0 else None


def cost_rollup(spans: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum of recorded costs over the LLM calls, with the unrecorded count
    beside it. `total` is None -- never 0 -- when no call recorded a cost."""
    calls = recorded = 0
    total = 0.0
    for span in spans:
        if span.get("name") != SPAN_LLM_CALL:
            continue
        calls += 1
        cost = llm_call_cost(span)
        if cost is not None:
            recorded += 1
            total += cost
    return {
        "calls": calls,
        "recorded": recorded,
        "unrecorded": calls - recorded,
        "total": total if recorded else None,
    }


def merge_cost_rollups(rollups: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    calls = recorded = unrecorded = 0
    total = 0.0
    for rollup in rollups:
        calls += int(rollup.get("calls") or 0)
        recorded += int(rollup.get("recorded") or 0)
        unrecorded += int(rollup.get("unrecorded") or 0)
        part = _finite_number(rollup.get("total"))
        if part is not None:
            total += part
    return {
        "calls": calls,
        "recorded": recorded,
        "unrecorded": unrecorded,
        "total": total if recorded else None,
    }


def turn_span_stamps(spans: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Everything a listed turn is stamped with, from one read of its spans."""
    span_list = list(spans)
    return {
        "llm_calls_cut_at_limit": count_llm_calls_cut_at_limit(span_list),
        "decision_signals": turn_decision_signals(span_list),
        "llm_cost": cost_rollup(span_list),
    }


def annotate_turn_detail(turn: dict[str, Any], spans: Iterable[Mapping[str, Any]]) -> None:
    """The opened turn: its ledger, chips and cost, from the trace the route
    already reads. Live and workspace routes both call this."""
    span_list = list(spans)
    turn["execution_ledger"] = execution_ledger(turn.get("record"), span_list)
    turn.update(turn_span_stamps(span_list))


# -- (c) provenance and comparability -------------------------------------------


def _flatten_provenance(prefix: str, value: Any, out: dict[str, Any]) -> None:
    """Nested provenance maps flatten to dotted keys; scalars and lists stay
    as they are, so a per-emitter contract version reads as
    `span_contract_versions.fw.turn: 1`."""
    if isinstance(value, dict):
        for key in sorted(value):
            _flatten_provenance(f"{prefix}.{key}" if prefix else str(key), value[key], out)
    else:
        out[prefix] = value


def _git_revision_of(record: Mapping[str, Any]) -> Optional[str]:
    """The engine's source revision, wherever a harness put it in the record.

    The evidence-run record (`EvidenceRun.as_record`) carries the
    ObservabilityProvenance only; the EngineProvenance with
    `source_revision` lives in the harness's RuntimeProvenance bundle, which
    this store never persists. These are the places a record might hold it;
    none of the trial's records did.
    """
    candidates = (
        record.get("engine"),
        (record.get("provenance") or {}).get("engine")
        if isinstance(record.get("provenance"), dict)
        else None,
        (record.get("runtime") or {}).get("engine")
        if isinstance(record.get("runtime"), dict)
        else None,
    )
    for candidate in candidates:
        if isinstance(candidate, dict):
            for key in ("source_revision", "git_revision"):
                if _text_or_none(candidate.get(key)):
                    return candidate[key]
    for key in ("source_revision", "git_revision"):
        if _text_or_none(record.get(key)):
            return record[key]
    return None


def benchmark_pin_check(
    workflow_folderpath: Optional[str], detail: Mapping[str, Any]
) -> Optional[dict[str, Any]]:
    """Check an experiment's benchmark pin against the catalogue file itself.

    The pin (`benchmark_id@version` plus the digest recorded when the run was
    declared) is stored in the experiment row; the version file it names lives
    in the workflow folder. Showing the recorded digest alone tells a reader
    nothing about whether the corpus still says what it said — that needs the
    file, which is why a sealed workspace now carries the folder.

    Four honest answers, never a hidden one: `match`, `mismatch` (both digests
    quoted verbatim, the reader decides what it means), `catalogue_unavailable`
    (with the reason: no folder declared, folder gone, benchmark or version
    missing), and `pin_incomplete` (the run recorded an id and version but no
    digest, so there is nothing to compare). ``None`` only when nothing was
    pinned at all.
    """
    benchmark_id = _text_or_none(detail.get("benchmark_id"))
    version = _text_or_none(detail.get("benchmark_version"))
    if not benchmark_id or not version:
        return None
    pinned = _text_or_none(detail.get("benchmark_digest_sha256"))
    check: dict[str, Any] = {
        "benchmark_id": benchmark_id,
        "benchmark_version": version,
        "pinned_digest": pinned,
        "catalogue_digest": None,
        "workflow_folderpath": workflow_folderpath,
        "status": "catalogue_unavailable",
        "detail": "",
    }
    if not workflow_folderpath:
        check["detail"] = (
            "this workspace manifest names no workflow folder, so the "
            "benchmark catalogue cannot be read"
        )
        return check
    if not os.path.isdir(workflow_folderpath):
        check["detail"] = (
            f"the workflow folder named by this workspace is not on this "
            f"machine: {workflow_folderpath}"
        )
        return check
    try:
        loaded = load_version(workflow_folderpath, benchmark_id, version)
    except (BenchmarkManifestError, OSError) as exc:
        check["detail"] = f"{benchmark_id}@{version} cannot be read: {exc}"
        return check
    check["catalogue_digest"] = loaded.get("digest_sha256")
    if not pinned:
        check["status"] = "pin_incomplete"
        check["detail"] = (
            "the experiment recorded no benchmark digest; the catalogue file "
            "is shown but nothing was pinned to compare it against"
        )
        return check
    if check["catalogue_digest"] == pinned:
        check["status"] = "match"
        check["detail"] = "the catalogue file still matches the pinned digest"
    else:
        check["status"] = "mismatch"
        check["detail"] = (
            f"pinned {pinned}, catalogue {check['catalogue_digest']}"
        )
    return check


def experiment_provenance(
    detail: Mapping[str, Any], attempts: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """An experiment's provenance, one field per row, keys verbatim.

    Three sources, each named on its field: the experiment row's own columns;
    the evidence-run records' `observability` block (ObservabilityProvenance:
    capture policy and span-contract versions, per-emitter versions, DB
    schema, the FW_OBS_* config in effect); and the attempts' runtime
    snapshots (workflow fingerprint and model version). A field two segments
    or two attempts disagree on is reported with every value and where each
    came from, never collapsed to one. `git_revision` is listed even when
    nothing recorded it, because its absence is the fact a reader needs.
    """
    fields: list[dict[str, Any]] = []

    def add(key: str, source: str, observations: list[tuple[str, Any]]) -> None:
        recorded = [(where, value) for where, value in observations if value is not None]
        if not recorded:
            fields.append(
                {"key": key, "source": source, "recorded": False, "value": None,
                 "consistent": True, "values": []}
            )
            return
        distinct: list[Any] = []
        for _, value in recorded:
            if value not in distinct:
                distinct.append(value)
        fields.append(
            {
                "key": key,
                "source": source,
                "recorded": True,
                "value": distinct[0] if len(distinct) == 1 else None,
                "consistent": len(distinct) == 1,
                "values": [{"where": where, "value": value} for where, value in recorded],
            }
        )

    for column in _EXPERIMENT_PROVENANCE_COLUMNS:
        add(column, "experiment", [("experiment", detail.get(column))])

    per_key: dict[str, list[tuple[str, Any]]] = {}
    revisions: list[tuple[str, Any]] = []
    for segment in detail.get("evidence_runs") or []:
        record = segment.get("record") if isinstance(segment, dict) else None
        if not isinstance(record, dict):
            continue
        where = f"evidence segment #{segment.get('seq')}"
        observability = record.get("observability")
        flat: dict[str, Any] = {}
        if isinstance(observability, dict):
            _flatten_provenance("", observability, flat)
        for key, value in flat.items():
            per_key.setdefault(key, []).append((where, value))
        revisions.append((where, _git_revision_of(record)))
    for key in sorted(per_key):
        add(key, "evidence_run", per_key[key])
    add("git_revision", "evidence_run", revisions)

    snapshot_keys: dict[str, list[tuple[str, Any]]] = {}
    for row in attempts:
        snapshot = row.get("runtime_snapshot")
        if not isinstance(snapshot, dict):
            continue
        where = f"attempt {row.get('task_id')}#{row.get('attempt')}"
        for key in _SNAPSHOT_PROVENANCE_KEYS:
            if key in snapshot:
                snapshot_keys.setdefault(key, []).append((where, snapshot[key]))
        features = snapshot.get("effective_features")
        if isinstance(features, dict):
            for name in sorted(features):
                snapshot_keys.setdefault(f"effective_features.{name}", []).append(
                    (where, features[name])
                )
    for key in sorted(snapshot_keys):
        add(key, "runtime_snapshot", snapshot_keys[key])

    return {
        "fields": fields,
        "recorded": sum(1 for field in fields if field["recorded"]),
        "unrecorded": sum(1 for field in fields if not field["recorded"]),
        "inconsistent": sum(1 for field in fields if not field["consistent"]),
    }


def provenance_differences(
    treatment: Mapping[str, Any], baseline: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Every provenance field the two experiments do not agree on, verbatim.

    A field recorded on one side and not the other differs; a field recorded
    on neither does not (there is nothing to quote). A field a side's own
    segments disagree on is compared as the list of its observed values.
    """

    def by_key(provenance: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
        return {
            (field["source"], field["key"]): field
            for field in provenance.get("fields") or []
        }

    def value_of(field: Optional[Mapping[str, Any]]) -> Any:
        if field is None or not field.get("recorded"):
            return None
        if field.get("consistent"):
            return field.get("value")
        return [entry.get("value") for entry in field.get("values") or []]

    left, right = by_key(treatment), by_key(baseline)
    differences = []
    for source, key in sorted(set(left) | set(right)):
        t_value = value_of(left.get((source, key)))
        b_value = value_of(right.get((source, key)))
        if t_value is None and b_value is None:
            continue
        if t_value == b_value:
            continue
        differences.append(
            {"key": key, "source": source, "treatment": t_value, "baseline": b_value}
        )
    return differences


def load_index_html() -> bytes:
    """The single self-contained SPA page, shipped as package data."""
    resource = (
        importlib.resources.files("fastworkflow.run_chatbot") / "static" / "index.html"
    )
    return resource.read_bytes()


def _inline_script_hashes(page: bytes) -> list[str]:
    """CSP sha256 sources for the page's own inline <script> blocks.

    The SPA is one self-contained file (no external requests), so
    ``script-src 'self'`` alone would block its inline script. Hash-sourcing
    keeps the policy restrictive: only the exact scripts shipped in the page
    execute; record-derived text can never inject a runnable script.
    """
    return [
        "'sha256-" + base64.b64encode(hashlib.sha256(m).digest()).decode() + "'"
        for m in _SCRIPT_RE.findall(page)
    ]


def _looks_like_workflow(path: str) -> bool:
    """A fastWorkflow workflow dir: authored commands or trained artifacts."""
    return os.path.isdir(os.path.join(path, "_commands")) or os.path.isdir(
        os.path.join(path, "___command_info")
    )


# Mirrors model_pipeline_training.GLOBAL_CONTEXT_FOLDER without importing
# that module — it pulls in torch/transformers, which the chatbot must not.
_GLOBAL_CONTEXT_FOLDER = "global"
_CME_CONTEXT_NAMES: Optional[set[str]] = None


def _cme_context_names() -> set[str]:
    """Internal command_metadata_extraction context names (cached).

    App workflow ``routing_definition.json`` lists these too; they are trained
    in the CME workflow, not per app, so they must not count as missing.
    """
    global _CME_CONTEXT_NAMES
    if _CME_CONTEXT_NAMES is not None:
        return _CME_CONTEXT_NAMES
    names: set[str] = set()
    try:
        import fastworkflow

        internal = fastworkflow.get_internal_workflow_path(
            "command_metadata_extraction"
        )
    except Exception:
        _CME_CONTEXT_NAMES = names
        return names
    json_path = os.path.join(internal, "command_context_model.json")
    try:
        with open(json_path, encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            names.update(str(key) for key in data)
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    commands = os.path.join(internal, "_commands")
    try:
        for entry in os.listdir(commands):
            full = os.path.join(commands, entry)
            if (
                os.path.isdir(full)
                and not entry.startswith(".")
                and entry != "__pycache__"
            ):
                names.add(entry)
    except OSError:
        pass
    _CME_CONTEXT_NAMES = names
    return names


def _workflow_is_trained(path: str) -> bool:
    """Filesystem check matching ``is_workflow_trained`` without importing torch.

    ``___command_info`` appearing is not enough: train writes that directory
    immediately, before any ``threshold.json`` exists.
    """
    command_info_root = os.path.join(path, "___command_info")
    routing_def_path = os.path.join(command_info_root, "routing_definition.json")
    if not os.path.isfile(routing_def_path):
        return False
    try:
        with open(routing_def_path, encoding="utf-8") as handle:
            routing_definition = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    contexts = routing_definition.get("contexts") or {}
    if not isinstance(contexts, dict) or not contexts:
        return False
    contexts_to_check = (set(contexts) - _cme_context_names()) | {"*"}
    for context_name in contexts_to_check:
        folder = _GLOBAL_CONTEXT_FOLDER if context_name == "*" else context_name
        threshold_path = os.path.join(command_info_root, folder, "threshold.json")
        if not os.path.isfile(threshold_path):
            return False
    return True


def _rel_under(path: str, root: str) -> str:
    """Path relative to ``root``, posix slashes, or ``""`` if outside ``root``."""
    try:
        rel = os.path.relpath(os.path.abspath(path), os.path.abspath(root))
    except ValueError:
        return ""
    if rel == ".":
        return ""
    if rel == ".." or rel.startswith(".." + os.sep):
        return ""
    return rel.replace("\\", "/")


def _workflow_entry(path: str, source: str, rel: str = "") -> dict[str, Any]:
    path = os.path.abspath(path)
    if launcher.is_bundled_example_path(path):
        source = "examples"
    trained = _workflow_is_trained(path)
    training = launcher.is_train_running(path)
    bundled = source == "examples" or launcher.is_bundled_example_path(path)
    return {
        "path": path,
        "name": os.path.basename(path),
        "rel": rel or os.path.basename(path),
        "trained": trained,
        "training": training,
        "source": source,
        "trainable": (not bundled) and (not trained) and (not training),
    }


_SKIP_DIR_NAMES = {
    "__pycache__",
    "node_modules",
    "site-packages",
    "dist",
    "build",
    "venv",
    "_commands",
    "___command_info",
    "___workflow_contexts",
    "___convo_info",
}

# Nested project layouts (apps/team/workflow) sit deeper than the old
# two-level scan; five is enough to find them without walking the world.
_MAX_WF_SCAN_DEPTH = 5
_MAX_WF_CANDIDATES = 100


def list_workflow_candidates() -> list[dict[str, Any]]:
    """Workflow dirs the developer most likely wants: the bundled examples,
    plus a bounded nested scan below the launch directory.

    Each entry carries ``rel`` (path relative to the launch directory, or a
    ``Bundled examples/`` prefix when the workflow lives outside it) so the
    picker can group them under folders instead of a flat list.
    """
    seen: dict[str, dict[str, Any]] = {}
    cwd = os.getcwd()

    def add(path: str, source: str, rel: str) -> None:
        path = os.path.abspath(path)
        if path not in seen and _looks_like_workflow(path):
            seen[path] = _workflow_entry(path, source, rel)

    add(cwd, "local", _rel_under(cwd, cwd))

    def walk(current: str, depth: int) -> None:
        if depth > _MAX_WF_SCAN_DEPTH or len(seen) >= _MAX_WF_CANDIDATES:
            return
        try:
            names = sorted(os.listdir(current))
        except OSError:
            return
        for name in names:
            if len(seen) >= _MAX_WF_CANDIDATES:
                return
            if name.startswith(".") or name in _SKIP_DIR_NAMES:
                continue
            full = os.path.join(current, name)
            if not os.path.isdir(full):
                continue
            rel = _rel_under(full, cwd)
            if _looks_like_workflow(full):
                add(full, "local", rel)
            walk(full, depth + 1)

    walk(cwd, 1)
    try:
        import fastworkflow

        examples = os.path.join(
            os.path.dirname(os.path.abspath(fastworkflow.__file__)), "examples"
        )
        for entry in sorted(os.listdir(examples)):
            full = os.path.join(examples, entry)
            rel = _rel_under(full, cwd)
            if not rel:
                rel = "Bundled examples/" + entry
            add(full, "examples", rel)
    except Exception:
        pass
    # A directory that both looks like a workflow and contains other workflows
    # (the library package has _commands/ plus examples/) is a folder, not a
    # leaf the developer would pick.
    for path in list(seen):
        if any(other != path and other.startswith(path + os.sep) for other in seen):
            seen.pop(path, None)
    candidates = list(seen.values())
    candidates.sort(
        key=lambda w: (w["source"] != "local", not w["trained"], w["name"].lower())
    )
    return candidates[:_MAX_WF_CANDIDATES]


_MANIFEST_PROBE_BYTES = 4096
_MANIFEST_SCHEMA_RE = re.compile(
    r'"(?:schema|schema_version)"\s*:\s*"' + re.escape(WORKSPACE_SCHEMA) + r'"'
)


def _declares_workspace_schema(path: str) -> bool:
    """Whether this file's head declares the v1 workspace manifest schema.

    The picker used to offer every ``*.json`` in the browsed directory, so
    from a project root it filled with score dumps and trajectory files that
    cannot be opened. The cheap, honest discriminator is the one key
    `ObservabilityWorkspace.load` itself insists on: ``schema`` (or the older
    ``schema_version``) equal to :data:`WORKSPACE_SCHEMA`.

    At most :data:`_MANIFEST_PROBE_BYTES` are read, so browsing a directory of
    900 KB result files costs one short read each and never loads one into
    memory. That prefix is parsed as JSON when it happens to be a whole small
    document -- which checks the key really is at the *top* level -- and
    otherwise scanned for the schema declaration, since a truncated prefix
    cannot be parsed. A manifest whose schema key sits past the probe window
    reads as "not a manifest": absence of evidence is "no", the same rule the
    rest of this module's derivations use, and the developer can still type
    the path.

    Any read error (permissions, a directory racing in, undecodable bytes)
    answers False rather than raising: the picker must render.
    """
    try:
        with open(path, "rb") as handle:
            head = handle.read(_MANIFEST_PROBE_BYTES)
    except OSError:
        return False
    try:
        text = head.decode("utf-8", errors="replace")
    except Exception:  # pragma: no cover - decode with errors= cannot raise
        return False
    try:
        value = json.loads(text)
    except ValueError:
        # Truncated at the probe window (or malformed): fall back to spotting
        # the schema declaration textually.
        return bool(_MANIFEST_SCHEMA_RE.search(text))
    if not isinstance(value, dict):
        return False
    return value.get("schema", value.get("schema_version")) == WORKSPACE_SCHEMA


def _local_workspace_manifests(base: str, name: str) -> list[dict[str, str]]:
    """This directory's own ``*.json`` file as a manifest offer, or nothing.

    Offered when it is named ``workspace.json`` -- the well-known name, offered
    on its name alone so a manifest that fails validation is still reachable
    and reports why -- or when its head declares the workspace schema, which
    covers a manifest someone renamed. Everything else is omitted entirely:
    not offered and not labelled, because a row the picker cannot open is
    worse than no row.
    """
    if not name.lower().endswith(".json"):
        return []
    full = os.path.join(base, name)
    if name.lower() != "workspace.json" and not _declares_workspace_schema(full):
        return []
    return [{"name": name, "label": name, "path": full}]


def _nested_workspace_manifests(base: str, name: str) -> list[dict[str, str]]:
    """``workspace.json`` one level under ``base/name``, labelled by that folder.

    What an owner points the picker at is the collection folder
    (``evaluation/collections/``); the manifest lives two levels down, at
    ``<collection>/workspace/workspace.json`` or ``<collection>/workspace.json``.
    Listing only the current directory made those invisible, so opening a
    sealed collection meant knowing and typing the path.

    Only the exact name ``workspace.json`` is looked for, never arbitrary
    ``*.json`` one level down: a collection folder holds many unrelated JSON
    files (scores, summaries, seal records), and offering those as manifests
    would fill the picker with entries that cannot be opened. Nothing is read
    — existence and the folder name are the whole probe.
    """
    found: list[dict[str, str]] = []
    for relative in ("workspace.json", os.path.join("workspace", "workspace.json")):
        candidate = os.path.join(base, name, relative)
        if os.path.isfile(candidate):
            found.append(
                {
                    "name": os.path.join(name, relative),
                    "label": name,
                    "path": os.path.abspath(candidate),
                }
            )
    return found


def browse_directories(dir_path: str) -> dict[str, Any]:
    """One level of the local filesystem for the workflow picker: directories
    only, never file contents; each entry flagged when it is a workflow.

    Workspace manifests come from two places: this directory's own
    ``workspace.json`` plus any other ``*.json`` here whose head declares the
    workspace schema (`_local_workspace_manifests`), and the well-known
    ``workspace.json`` one level down inside each subdirectory. Stray JSON is
    omitted, never offered-and-broken.
    """
    base = os.path.abspath(dir_path or os.getcwd())
    if not os.path.isdir(base):
        return {"error": f"not a directory: {base}"}
    entries = []
    workspace_manifests: list[dict[str, str]] = []
    nested_manifests: list[dict[str, str]] = []
    try:
        names = sorted(os.listdir(base))
    except OSError as exc:
        return {"error": f"cannot list {base}: {exc}"}
    for name in names:
        if name.startswith("."):
            continue
        full = os.path.join(base, name)
        if not os.path.isdir(full):
            workspace_manifests.extend(_local_workspace_manifests(base, name))
            continue
        nested_manifests.extend(_nested_workspace_manifests(base, name))
        is_workflow = _looks_like_workflow(full)
        entry = {
            "name": name,
            "path": full,
            "is_workflow": is_workflow,
            "trained": _workflow_is_trained(full) if is_workflow else False,
        }
        if is_workflow:
            entry["training"] = launcher.is_train_running(full)
        entries.append(entry)
        if len(entries) >= 300:
            break
    parent = os.path.dirname(base)
    return {
        "dir": base,
        "parent": parent if parent != base else None,
        "entries": entries,
        # This directory's own JSON first, then what was found one level down:
        # the same 300 cap covers both, so a directory of many collections
        # cannot make the answer unbounded.
        "workspace_manifests": (workspace_manifests + nested_manifests)[:300],
    }


def _free_server_port(preferred: int) -> tuple[int, bool]:
    """(port to use, moved?) — the preferred port when it is free, otherwise a
    free ephemeral one. Anything may be squatting the default 8000 (an old
    server, another chatbot, an unrelated app); spawning onto a busy port is
    worse than moving: uvicorn takes seconds to fail its bind, and meanwhile
    the chat would connect to WHATEVER is already answering there — possibly a
    different workflow's server entirely."""
    import socket

    for candidate, moved in ((preferred, False), (0, True)):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", candidate))
                return probe.getsockname()[1], moved
        except OSError:
            continue
    return preferred, False


def _autodetect_env_files(workflow_path: str) -> tuple[str, str]:
    """Best-effort env-file discovery for the spawned server, in order:
    workflow-local files, then the bundled ``examples/`` shared files (when
    the workflow lives there). Missing files resolve to "" so the chatbot can
    offer a file picker or create workflow-local files from the templates."""
    wf = os.path.abspath(workflow_path)
    roots = [wf]
    parent = os.path.dirname(wf)
    if os.path.basename(parent) == "examples":
        roots.append(parent)

    def first_existing(filename: str) -> str:
        for root in roots:
            candidate = os.path.join(root, filename)
            if os.path.isfile(candidate):
                return candidate
        return ""

    return first_existing("fastworkflow.env"), first_existing(
        "fastworkflow.passwords.env"
    )


def _env_template_text(filename: str) -> str:
    resource = importlib.resources.files("fastworkflow") / "examples" / filename
    return resource.read_text(encoding="utf-8")


def _write_env_file(path: str, content: str) -> None:
    """Atomically write one workflow-local env file with owner-only access."""
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", dir=parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
        os.replace(temp_path, path)
        os.chmod(path, 0o600)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass
        raise


class ChatbotServer:
    """The chatbot's local web layer.

    Ordinary observability reads use ``ReadOnlyObservabilityStore``. Explicit,
    token-gated control-plane actions select a workflow, configure missing env
    files, or clear recorded conversations.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        workflow_path: str = "",
        port: int = 0,
        token: Optional[str] = None,
        spawn_options: Optional[dict] = None,
        workspace_manifest_path: Optional[str] = None,
    ) -> None:
        self.db_path = db_path or ""
        self.workflow_path = workflow_path
        self.workspace: Optional[ObservabilityWorkspace] = (
            load_observability_workspace(workspace_manifest_path)
            if workspace_manifest_path
            else None
        )
        self.workspace_manifest_path = (
            str(self.workspace.manifest_path) if self.workspace is not None else ""
        )
        self._review_sidecar: Optional[ReviewSidecar] = None
        self._review_sidecar_lock = threading.Lock()
        # Auto-spawn posture for the workflow's FastAPI server; see
        # run_chatbot_main. no_server=True keeps the chatbot debug-only.
        self.spawn_options = dict(spawn_options or {"no_server": True})
        self.server_proc = None  # the spawned FastAPI server (subprocess.Popen)
        self.server_url: Optional[str] = None
        self.spawn_error: Optional[str] = None
        self.server_note: Optional[str] = None  # e.g. "port 8000 busy; using 40123"
        self.env_file_path = ""
        self.passwords_file_path = ""
        self.env_setup_required = False
        # Single-user dev tool: the channel is an implementation detail the
        # developer never types, and it is FIXED rather than minted per launch.
        # A per-launch channel scattered every restart's conversations into its
        # own top-level group in the debug rail, so yesterday's turns were a
        # different "channel" from today's for no reason a developer could see.
        # Conversations still separate them; the channel no longer does.
        self.channel_id = "chatbot"
        self.user_id = "developer"
        self._activate_lock = threading.Lock()
        self._train_lock = threading.Lock()
        # Per-launch bearer token [R5]; overridable only for tests.
        self.token = token if token is not None else secrets.token_urlsafe(32)
        self.index_html = load_index_html()
        script_hashes = _inline_script_hashes(self.index_html)
        if b"<script" in self.index_html and not script_hashes:
            # Fail loudly at launch rather than serving a page whose own
            # script the CSP will block with no server-side signal.
            raise RuntimeError(
                "CSP hash extraction found no inline <script> blocks in the "
                "bundled SPA; the page would be blocked by its own policy"
            )
        script_srcs = " ".join(script_hashes)
        # connect-src: 'self' for the debug-mode read API, plus loopback-only
        # origins so TEST MODE can call the local FastAPI server
        # (/initialize, /invoke_agent, /invoke_assistant). Never a non-loopback
        # host — the SPA can only ever talk to servers on this machine [R19][R22].
        self.page_csp = (
            "default-src 'none'; "
            f"script-src 'self'{' ' + script_srcs if script_srcs else ''}; "
            "style-src 'self' 'unsafe-inline'; "
            "connect-src 'self' http://127.0.0.1:* http://localhost:*; "
            "img-src 'self' data:; "
            "frame-src 'self'"
        )

        server = self

        class _Handler(_ChatbotRequestHandler):
            chatbot = server

        # 127.0.0.1 only — never configurable to a wider bind [R5][R18].
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]

    def open_store(self) -> Optional[ReadOnlyObservabilityStore]:
        """Per-request READ-ONLY store handle, or None while the DB is absent
        or unopenable. The viewer never creates, migrates, or writes the DB
        it inspects — a missing DB (e.g. test-mode cold start before the
        first turn) serves empty views instead of an error.
        """
        if not self.db_path:
            return None  # no workflow selected yet
        try:
            return ReadOnlyObservabilityStore(self.db_path)
        except IncompatibleObservabilityDB:
            raise  # a newer-schema DB is a real error, surfaced per-request
        except Exception:
            return None

    def open_review_sidecar(self) -> ReviewSidecar:
        """Return the manifest-bound review store for the active workspace."""
        if self.workspace is None or not self.workspace_manifest_path:
            raise ReviewValidationError(
                "review assignments require an active observability workspace"
            )
        with self._review_sidecar_lock:
            if self._review_sidecar is None:
                self._review_sidecar = ReviewSidecar.from_workspace_manifest(
                    self.workspace_manifest_path
                )
            return self._review_sidecar

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/?token={self.token}"

    # -- workflow activation + server lifecycle -------------------------

    def session_payload(self) -> dict[str, Any]:
        """What the SPA needs to run without asking the developer anything.
        Reflects the LIVE child state — the SPA polls this, so a server that
        dies mid-session is reported honestly, not as 'running'."""
        running = self.server_proc is not None and self.server_proc.poll() is None
        # An omitted spawn still reports server_url when --server-port named an
        # existing server, so the Advanced panel can be prefilled.
        expose_url = running or bool(
            self.spawn_options.get("no_server") and self.server_url
        )
        payload = {
            "workflow_path": self.workflow_path,
            "workflow_name": (
                os.path.basename(os.path.abspath(self.workflow_path))
                if self.workflow_path
                else ""
            ),
            "db_path": self.db_path,
            "server_url": self.server_url if expose_url else None,
            "server_running": running,
            "server_exit_code": (
                self.server_proc.returncode
                if self.server_proc is not None and not running
                else None
            ),
            "server_note": self.server_note,
            "env_setup_required": self.env_setup_required,
            "env_file_path": self.env_file_path or None,
            "passwords_file_path": self.passwords_file_path or None,
            "channel_id": self.channel_id,
            "user_id": self.user_id,
            "jwt_mode": (
                "signed"
                if self.spawn_options.get("expect_encrypted_jwt")
                else "unsigned"
            ),
            "spawn_error": self.spawn_error,
        }
        if self.workspace is not None:
            payload.update(
                {
                    "workspace_mode": True,
                    "workspace": self.workspace.summary(),
                    "workflow_path": "",
                    "workflow_name": "",
                    "db_path": "",
                    "server_url": None,
                    "server_running": False,
                    "env_setup_required": False,
                    "read_only": True,
                }
            )
        else:
            payload.update({"workspace_mode": False, "read_only": False})
        return payload

    def activate_workspace(self, manifest_path: str) -> dict[str, Any]:
        """Load a manifest selected through the token-gated browser picker."""
        with self._activate_lock:
            workspace = load_observability_workspace(manifest_path)
            if self.server_proc is not None and self.server_proc.poll() is None:
                launcher.terminate_server(self.server_proc)
            self.server_proc = None
            self.server_url = None
            self.workflow_path = ""
            self.db_path = ""
            self.workspace = workspace
            self.workspace_manifest_path = str(workspace.manifest_path)
            self._review_sidecar = None
            return self.session_payload()

    def activate_workflow(self, workflow_path: str) -> dict[str, Any]:
        """Point the chatbot at a workflow and (unless disabled) make sure its
        FastAPI server is running. Selecting a different workflow replaces the
        spawned server. Never raises: failures land in ``spawn_error`` and the
        chatbot stays usable as a trace viewer."""
        from fastworkflow import state_paths

        with self._activate_lock:
            workflow_path = os.path.abspath(workflow_path)
            same_workflow = os.path.abspath(self.workflow_path or "") == workflow_path
            self.workflow_path = workflow_path
            self.db_path = state_paths.observability_db(workflow_path)
            self.spawn_error = None
            self.server_note = None
            if self.spawn_options.get("no_server"):
                external = self.spawn_options.get("server_port")
                if external:
                    self.server_url = f"http://127.0.0.1:{int(external)}"
                return self.session_payload()
            if (
                same_workflow
                and self.server_proc is not None
                and self.server_proc.poll() is None
            ):
                return self.session_payload()  # already serving this workflow

            from fastworkflow.run_chatbot import launcher

            if self.server_proc is not None and self.server_proc.poll() is None:
                launcher.terminate_server(self.server_proc)
            self.server_proc = None
            self.server_url = None

            env_file = self.spawn_options.get("env_file_path") or ""
            passwords_file = self.spawn_options.get("passwords_file_path") or ""
            if not env_file or not passwords_file:
                auto_env, auto_passwords = _autodetect_env_files(workflow_path)
                env_file = env_file or auto_env
                passwords_file = passwords_file or auto_passwords
            self.env_file_path = env_file
            self.passwords_file_path = passwords_file
            self.env_setup_required = not (
                env_file
                and passwords_file
                and os.path.isfile(env_file)
                and os.path.isfile(passwords_file)
            )
            if self.env_setup_required:
                self.server_proc = None
                self.server_url = None
                return self.session_payload()

            preferred_port = int(
                self.spawn_options.get("server_port") or PREFERRED_SPAWN_PORT
            )
            server_port, moved = _free_server_port(preferred_port)
            self.server_note = (
                f"port {preferred_port} was busy; the server runs on {server_port} instead"
                if moved
                else None
            )
            if moved:
                logger.warning(
                    f"Chatbot server port {preferred_port} is busy; "
                    f"spawning the FastAPI server on {server_port} instead"
                )

            expect_encrypted = bool(self.spawn_options.get("expect_encrypted_jwt"))
            plan = launcher.plan_server_spawn(
                workflow_path=workflow_path,
                env_file_path=env_file,
                passwords_file_path=passwords_file,
                chatbot_origin=f"http://127.0.0.1:{self.port}",
                server_port=server_port,
                expect_encrypted_jwt=expect_encrypted,
                # Loopback-only + loopback-pinned CORS + a chatbot that mints
                # its own tokens via /initialize: unsigned dev JWTs are the
                # default posture for the AUTO-spawned server (owner decision
                # amending R19's opt-in flag; --expect-encrypted-jwt restores
                # signed mode).
                allow_unsigned_jwt=not expect_encrypted,
            )
            if not plan.ok:
                self.spawn_error = plan.reason
                return self.session_payload()
            try:
                self.server_proc = launcher.spawn_server(plan)
            except OSError as exc:
                self.spawn_error = f"could not start the FastAPI server: {exc}"
                return self.session_payload()
            time.sleep(1.0)  # one early liveness check: died-at-startup is common
            if self.server_proc.poll() is not None:
                self.spawn_error = (
                    "the FastAPI server exited immediately "
                    f"(exit code {self.server_proc.returncode}) — its output is in "
                    "the chatbot's terminal; check that the --server-port is free "
                    "and the env files are valid"
                )
                self.server_proc = None
                return self.session_payload()
            self.server_url = plan.server_url
            return self.session_payload()

    def configure_env_files(
        self,
        *,
        env_content: Optional[str] = None,
        passwords_content: Optional[str] = None,
        create_from_templates: bool = False,
    ) -> dict[str, Any]:
        """Install missing workflow-local env files, then activate the workflow."""
        if not self.workflow_path:
            raise ValueError("select a workflow before configuring env files")
        if launcher.is_bundled_example_path(self.workflow_path):
            # Writing into the packaged examples dir would land a passwords
            # file in site-packages — or, in a repo checkout, in a directory
            # git does not ignore. The bundled examples read the shared
            # examples/fastworkflow*.env templates instead.
            raise ValueError(
                "bundled examples cannot take workflow-local env files; copy "
                "the example to your own folder first, or edit the shared "
                "templates beside the examples directory"
            )
        max_bytes = 512 * 1024
        for label, content in (
            ("environment", env_content),
            ("passwords", passwords_content),
        ):
            if content is not None and not isinstance(content, str):
                raise TypeError(f"{label} file content must be text")
            if content is not None and len(content.encode("utf-8")) > max_bytes:
                raise ValueError(f"{label} file is larger than {max_bytes} bytes")

        env_target = os.path.join(self.workflow_path, "fastworkflow.env")
        passwords_target = os.path.join(
            self.workflow_path, "fastworkflow.passwords.env"
        )
        if create_from_templates:
            if not os.path.isfile(env_target):
                _write_env_file(env_target, _env_template_text("fastworkflow.env"))
            if not os.path.isfile(passwords_target):
                _write_env_file(
                    passwords_target,
                    _env_template_text("fastworkflow.passwords.env"),
                )
        if env_content is not None:
            _write_env_file(env_target, env_content)
        if passwords_content is not None:
            _write_env_file(passwords_target, passwords_content)

        self.spawn_options["env_file_path"] = ""
        self.spawn_options["passwords_file_path"] = ""
        return self.activate_workflow(self.workflow_path)

    def start_train(self, workflow_path: str) -> dict[str, Any]:
        """Spawn a detached ``fastworkflow train`` and return immediately.

        The child outlives this chatbot process. Shutdown does not signal it.
        Status is the pid file + ``_workflow_is_trained`` on later polls.
        """
        workflow_path = os.path.abspath(workflow_path)
        with self._train_lock:
            env_file = self.spawn_options.get("env_file_path") or ""
            passwords_file = self.spawn_options.get("passwords_file_path") or ""
            if not env_file or not passwords_file:
                auto_env, auto_passwords = _autodetect_env_files(workflow_path)
                env_file = env_file or auto_env
                passwords_file = passwords_file or auto_passwords
            plan = launcher.plan_train_spawn(
                workflow_path=workflow_path,
                env_file_path=env_file,
                passwords_file_path=passwords_file,
                already_trained=_workflow_is_trained(workflow_path),
            )
            if not plan.ok:
                raise ValueError(plan.reason)
            pid = launcher.spawn_detached_train(plan)
            return {
                "ok": True,
                "pid": pid,
                "training": True,
                "trained": False,
                "log_path": plan.log_path,
            }

    def serve_forever(self) -> None:
        self.httpd.serve_forever()

    def shutdown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        if self.server_proc is not None and self.server_proc.poll() is None:
            from fastworkflow.run_chatbot import launcher

            launcher.terminate_server(self.server_proc)
            self.server_proc = None


# The one sentence that explains an unreadable evidence store, shared by the
# two payloads that report it: the navigation warning band -- the sidebar is
# where a reader meets the failure -- and the /experiments payload.
STORE_UNAVAILABLE = "Recorded experiments are unavailable in the selected evidence store: "


class _ChatbotRequestHandler(BaseHTTPRequestHandler):
    """Token-gated request handler. Observability queries are GET-only;
    explicit control-plane POSTs select a workflow, configure env, start
    train, or clear recorded conversations."""

    chatbot: ChatbotServer  # bound by ChatbotServer.__init__
    protocol_version = "HTTP/1.1"
    server_version = "fastWorkflowChatbot"
    sys_version = ""

    # -- plumbing --------------------------------------------------------

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass  # quiet; the terminal belongs to the launch banner

    def _send(
        self,
        status: int,
        body: bytes,
        content_type: str,
        extra_headers: Optional[dict[str, str]] = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status=status)

    # -- access control [R5][R18] ---------------------------------------
    #
    # The allowlist admits any LOOPBACK authority — 127.0.0.1 / localhost /
    # [::1], any port — and nothing else. Loopback-only is what defeats DNS
    # rebinding (a rebound request arrives with the attacker's hostname in
    # Host); the port is deliberately NOT pinned, because port forwarders
    # (VS Code Remote / WSL relays) legitimately re-expose the server on a
    # different local port and the browser's Host names THAT port. The bearer
    # token remains the authentication on every request either way.

    @staticmethod
    def _is_loopback_authority(authority: str) -> bool:
        authority = authority.strip().lower()
        if not authority:
            return False
        if authority.startswith("["):  # bracketed IPv6, e.g. [::1]:8901
            hostname = authority.split("]", 1)[0].lstrip("[")
        else:
            hostname = authority.rsplit(":", 1)[0] if ":" in authority else authority
        return hostname in ("127.0.0.1", "localhost", "::1")

    def _host_origin_allowed(self) -> bool:
        host = (self.headers.get("Host") or "").strip().lower()
        if not self._is_loopback_authority(host):
            logger.warning(
                f"Chatbot refused a request with non-loopback Host {host!r} [R18]"
            )
            return False
        origin = (self.headers.get("Origin") or "").strip().lower()
        if origin:
            scheme, sep, authority = origin.partition("://")
            if (
                scheme != "http"
                or not sep
                or not self._is_loopback_authority(authority)
            ):
                logger.warning(
                    f"Chatbot refused a request with non-loopback Origin {origin!r} [R18]"
                )
                return False
        return True

    def _token_valid(self, query: dict[str, list[str]]) -> bool:
        presented = ""
        auth = self.headers.get("Authorization") or ""
        if auth.startswith("Bearer "):
            presented = auth[len("Bearer ") :].strip()
        elif query.get("token"):
            presented = query["token"][0]
        return hmac.compare_digest(
            presented.encode("utf-8"), self.chatbot.token.encode("utf-8")
        )

    def _review_capability(self) -> str:
        """Return the separately presented rater capability."""
        return (self.headers.get("X-Review-Capability") or "").strip()

    # -- routing ---------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        try:
            self._handle_get()
        except BrokenPipeError:
            pass
        except Exception as exc:
            try:
                self._error(500, f"internal error: {type(exc).__name__}")
            except Exception:
                pass

    def _handle_get(self) -> None:
        split = urlsplit(self.path)
        path = split.path
        query = parse_qs(split.query)

        if not self._host_origin_allowed():
            self._error(
                403,
                "forbidden: only loopback hosts (127.0.0.1 / localhost / [::1]) "
                f"may access the chatbot; got Host={self.headers.get('Host')!r}, "
                f"Origin={self.headers.get('Origin')!r}",
            )
            return
        # EVERY request is token-gated, the page included (Jupyter pattern).
        if not self._token_valid(query):
            self._error(401, "unauthorized: missing or invalid token")
            return

        if path in ("/", "/index.html"):
            self._send(
                200,
                self.chatbot.index_html,
                "text/html; charset=utf-8",
                {"Content-Security-Policy": self.chatbot.page_csp},
            )
            return
        if path == "/trace" or path.startswith("/trace/"):
            self._handle_trace_navigation(path, query)
            return
        if path.startswith("/api/"):
            self._handle_api(path, query)
            return
        self._error(404, "not found")

    def _handle_trace_navigation(
        self, path: str, query: dict[str, list[str]]
    ) -> None:
        """Convert a durable scoped turn reference into SPA hash navigation."""
        if self.chatbot.workspace is None:
            self._error(404, "no observability workspace is loaded")
            return
        if path != "/trace":
            self._error(
                400,
                "unscoped /trace/<key> links are refused; provide store_id and "
                "logical_turn_key to /trace",
            )
            return
        store_id = (query.get("store_id") or [""])[0]
        logical_turn_key = (query.get("logical_turn_key") or [""])[0]
        if not store_id or not logical_turn_key:
            self._error(
                400,
                "trace navigation requires store_id and logical_turn_key",
            )
            return
        try:
            if self.chatbot.workspace.turn(store_id, logical_turn_key) is None:
                self._error(404, "turn not found in the named store")
                return
        except UnknownWorkspaceStore as exc:
            self._error(404, str(exc.args[0] if exc.args else exc))
            return
        fragment = (
            "store="
            + quote(store_id, safe="")
            + "&turn="
            + quote(logical_turn_key, safe="")
        )
        location = "/?token=" + quote(self.chatbot.token, safe="") + "#" + fragment
        self._send(
            303,
            b"",
            "text/plain; charset=utf-8",
            {"Location": location},
        )

    # Writes: ordinary observability browsing stays read-only. The explicit
    # control-plane POSTs (select workflow, configure env, train, clear
    # conversations) carry the same host/origin + token gates as GETs.
    def _refuse_write(self) -> None:
        self._send_json(
            {"error": "method not allowed: observability data is read-only"}, 405
        )

    def do_POST(self) -> None:  # noqa: N802
        try:
            split = urlsplit(self.path)
            review_answer_path = (
                split.path.startswith("/api/review/assignments/")
                and split.path.endswith("/answers")
            )
            review_adjudication_path = (
                split.path.startswith("/api/review/assignments/")
                and split.path.endswith("/adjudications")
            )
            setup_post_path = (
                split.path == "/api/experiment-setups"
                or split.path.startswith("/api/experiment-setups/")
            )
            benchmark_setup_path = split.path == "/api/benchmark-setup"
            benchmark_experiment_path = (split.path.startswith("/api/benchmarks/")
                                         and split.path.endswith("/experiments"))
            benchmark_post_path = split.path == "/api/benchmarks" or (
                split.path.startswith("/api/benchmarks/")
                and split.path.endswith("/versions")
            )
            if (
                not benchmark_setup_path
                and not benchmark_experiment_path
                and not setup_post_path
                and not review_answer_path
                and not review_adjudication_path
                and not benchmark_post_path
                and split.path
                not in {
                "/api/human-feedback",
                "/api/select_workflow",
                "/api/select_workspace",
                "/api/configure_env",
                "/api/clear_conversations",
                "/api/review/assignments",
                "/api/train",
                }
            ):
                self._refuse_write()
                return
            query = parse_qs(split.query)
            if not self._host_origin_allowed():
                self._error(403, "forbidden: host/origin not allowed")
                return
            if not self._token_valid(query):
                self._error(401, "unauthorized: missing or invalid token")
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, TypeError):
                self._error(400, "invalid JSON body")
                return
            if split.path == "/api/human-feedback":
                if not isinstance(body, dict):
                    self._error(400, "body must be a JSON object")
                    return
                self._handle_human_feedback(query, body)
                return
            if benchmark_setup_path or benchmark_experiment_path:
                folder = self._benchmark_workflow_path(write=True)
                if folder is None:
                    return
                try:
                    if not isinstance(body, dict):
                        raise ValueError("body must be an object")
                    if benchmark_setup_path:
                        self._send_json({"version": benchmark_setup.save_benchmark(folder, body)}, status=201)
                    else:
                        benchmark_id = unquote(split.path[len("/api/benchmarks/"):-len("/experiments")])
                        record = benchmark_setup.create_experiment(
                            folder, benchmark_id, body.get("version"),
                            body.get("description", ""),
                        )
                        self._send_json({"experiment": record}, status=201)
                except benchmark_setup.BenchmarkSetupConflict as exc:
                    self._error(409, str(exc))
                except (BenchmarkManifestError, ValueError, TypeError) as exc:
                    self._error(400, str(exc))
                return
            if setup_post_path:
                self._handle_setup(split.path, body=body, write=True)
                return
            if review_answer_path or review_adjudication_path:
                if not isinstance(body, dict):
                    self._error(400, "body must be a JSON object")
                    return
                if self.chatbot.workspace is None:
                    self._error(
                        409,
                        "review answers require an active observability workspace",
                    )
                    return
                suffix = (
                    "/answers" if review_answer_path else "/adjudications"
                )
                encoded_id = split.path[
                    len("/api/review/assignments/") : -len(suffix)
                ].rstrip("/")
                if not encoded_id:
                    self._error(404, "not found")
                    return
                assignment_id = unquote(encoded_id)
                capability = self._review_capability()
                try:
                    sidecar = self.chatbot.open_review_sidecar()
                    # Authorize the path before appending an immutable revision.
                    role = "rater" if review_answer_path else "adjudicator"
                    sidecar.authorize_capability(assignment_id, capability, role)
                    capture = (
                        sidecar.capture_answer
                        if review_answer_path
                        else sidecar.capture_adjudication
                    )
                    captured = capture(
                        capability,
                        str(body.get("row_id") or ""),
                        str(body.get("question_id") or ""),
                        body.get("answer"),
                    )
                except ReviewAuthorizationError as exc:
                    self._error(403, str(exc))
                    return
                except ReviewValidationError as exc:
                    self._error(400, str(exc))
                    return
                except ReviewNotFoundError as exc:
                    self._error(404, str(exc.args[0] if exc.args else exc))
                    return
                key = "answer" if review_answer_path else "adjudication"
                self._send_json({key: captured})
                return
            if split.path == "/api/review/assignments":
                if self.chatbot.workspace is None:
                    self._error(
                        409,
                        "review assignments require an active observability workspace",
                    )
                    return
                try:
                    created = self.chatbot.open_review_sidecar().create_assignment(body)
                except ReviewValidationError as exc:
                    self._error(400, str(exc))
                    return
                except sqlite3.IntegrityError:
                    self._error(409, "an assignment with this id already exists")
                    return
                self._send_json(created, status=201)
                return
            if benchmark_post_path:
                self._handle_benchmark_post(split.path, body)
                return
            if self.chatbot.workspace is not None:
                self._error(
                    403,
                    "workspace mode is read-only; live and destructive actions "
                    "are disabled",
                )
                return
            if split.path == "/api/select_workspace":
                path = str(body.get("path") or "").strip()
                if not path or not os.path.isfile(path):
                    self._error(400, f"not a file: {path!r}")
                    return
                try:
                    session = self.chatbot.activate_workspace(path)
                except (OSError, ValueError, WorkspaceError) as exc:
                    self._error(400, str(exc))
                    return
                self._send_json({"session": session})
                return
            if split.path == "/api/select_workflow":
                path = str(body.get("path") or "").strip()
                if not path or not os.path.isdir(path):
                    self._error(400, f"not a directory: {path!r}")
                    return
                if not _looks_like_workflow(path):
                    self._error(
                        400,
                        f"{path} does not look like a fastWorkflow workflow "
                        "(no _commands/ or ___command_info/ inside)",
                    )
                    return
                self._send_json({"session": self.chatbot.activate_workflow(path)})
                return
            if split.path == "/api/configure_env":
                try:
                    session = self.chatbot.configure_env_files(
                        env_content=body.get("env_content"),
                        passwords_content=body.get("passwords_content"),
                        create_from_templates=bool(body.get("create_from_templates")),
                    )
                except (OSError, TypeError, ValueError) as exc:
                    self._error(400, str(exc))
                    return
                self._send_json({"session": session})
                return
            if split.path == "/api/train":
                path = str(body.get("path") or "").strip()
                if not path or not os.path.isdir(path):
                    self._error(400, f"not a directory: {path!r}")
                    return
                if not _looks_like_workflow(path):
                    self._error(
                        400,
                        f"{path} does not look like a fastWorkflow workflow "
                        "(no _commands/ or ___command_info/ inside)",
                    )
                    return
                try:
                    result = self.chatbot.start_train(path)
                except OSError as exc:
                    self._error(500, f"could not start training: {exc}")
                    return
                except ValueError as exc:
                    reason = str(exc)
                    lowered = reason.lower()
                    status = (
                        409
                        if ("already running" in lowered or "in progress" in lowered)
                        else 400
                    )
                    self._error(status, reason)
                    return
                self._send_json(result)
                return

            if body.get("confirm") != "clear all conversations":
                self._error(
                    400,
                    "confirmation required: confirm='clear all conversations'",
                )
                return
            if not self.chatbot.db_path or not os.path.exists(self.chatbot.db_path):
                self._send_json({"deleted": {}})
                return
            deleted = run_clear_conversations(
                self.chatbot.db_path, self.chatbot.workflow_path
            )
            self._send_json({"deleted": deleted})
        except BrokenPipeError:
            pass
        except Exception as exc:
            try:
                self._error(500, f"internal error: {type(exc).__name__}")
            except Exception:
                pass

    def do_PUT(self) -> None:  # noqa: N802
        """Admitted PUT: the benchmark's sibling analysis file."""
        try:
            split = urlsplit(self.path)
            benchmark_analysis_path = (
                split.path.startswith("/api/benchmarks/")
                and split.path.endswith("/analysis")
            )
            if not benchmark_analysis_path:
                self._refuse_write()
                return
            query = parse_qs(split.query)
            if not self._host_origin_allowed():
                self._error(403, "forbidden: host/origin not allowed")
                return
            if not self._token_valid(query):
                self._error(401, "unauthorized: missing or invalid token")
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, TypeError):
                self._error(400, "invalid JSON body")
                return
            if not isinstance(body, dict):
                self._error(400, "body must be a JSON object")
                return
            self._handle_benchmark_analysis_put(split.path, body)
        except BrokenPipeError:
            pass
        except Exception as exc:
            try:
                self._error(500, f"internal error: {type(exc).__name__}")
            except Exception:
                pass

    def do_DELETE(self) -> None:  # noqa: N802
        try:
            split = urlsplit(self.path)
            prefix = "/api/benchmark-experiments/"
            if not split.path.startswith(prefix):
                self._refuse_write()
                return
            query = parse_qs(split.query)
            if not self._host_origin_allowed():
                self._error(403, "forbidden: host/origin not allowed")
                return
            if not self._token_valid(query):
                self._error(401, "unauthorized: missing or invalid token")
                return
            folder = self._benchmark_workflow_path(write=True)
            if folder is None:
                return
            experiment_id = unquote(split.path[len(prefix):])
            try:
                record = benchmark_setup.delete_empty_experiment(folder, experiment_id)
            except (KeyError, benchmark_setup.ExperimentDeleted):
                self._error(404, "experiment not found")
                return
            except benchmark_setup.BenchmarkSetupConflict as exc:
                self._error(409, str(exc))
                return
            except (ValueError, TypeError) as exc:
                self._error(400, str(exc))
                return
            self._send_json({"deleted": experiment_id, "benchmark_id": record["benchmark_id"]})
        except BrokenPipeError:
            pass
        except Exception:
            self._error(500, "Could not delete this experiment. Refresh and try again.")

    def do_PATCH(self) -> None:  # noqa: N802
        """Two admitted PATCH surfaces: an experiment's editable annotations,
        and the author's description on a registration not yet claimed.

        The Host/Origin and bearer-token gates are applied per verb method with
        no shared chokepoint -- `_handle_get` and `do_POST` each run their own --
        so this repeats them rather than inheriting anything. A do_PATCH written
        without them would be an ungated cross-origin write.
        """
        try:
            split = urlsplit(self.path)
            registration_path = split.path.startswith("/api/benchmark-experiments/")
            if not split.path.startswith("/api/experiment/") and not registration_path:
                self._refuse_write()
                return
            query = parse_qs(split.query)
            if not self._host_origin_allowed():
                self._error(403, "forbidden: host/origin not allowed")
                return
            if not self._token_valid(query):
                self._error(401, "unauthorized: missing or invalid token")
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, TypeError):
                self._error(400, "invalid JSON body")
                return
            if not isinstance(body, dict):
                self._error(400, "body must be a JSON object")
                return
            if self.chatbot.workspace is not None:
                self._error(
                    403,
                    "workspace mode is read-only; experiment annotations cannot be changed",
                )
                return
            if registration_path:
                self._handle_registration_patch(split.path, body)
                return
            self._handle_experiment_patch(
                split.path,
                body,
                (query.get("benchmark_experiment") or [None])[0],
            )
        except BrokenPipeError:
            pass
        except Exception as exc:
            try:
                self._error(500, f"internal error: {type(exc).__name__}")
            except Exception:
                pass

    # -- API endpoints ---------------------------------------------------

    def _handle_api(self, path: str, query: dict[str, list[str]]) -> None:
        # Authoring and navigation do not depend on execution evidence. In
        # particular, an incompatible selected store must not trap the user
        # by breaking session loading and the workflow picker.
        q = lambda name: query.get(name, [None])[0]  # noqa: E731
        if path == "/api/navigation":
            self._handle_navigation()
            return
        if path == "/api/human-feedback":
            self._handle_human_feedback(query)
            return
        if path.startswith("/api/benchmark-experiments/"):
            self._handle_benchmark_registration(unquote(path[len("/api/benchmark-experiments/"):]))
            return
        if path == "/api/session":
            self._send_json({"session": self.chatbot.session_payload()})
            return
        if path == "/api/workflows":
            self._send_json({"workflows": list_workflow_candidates()})
            return
        if path == "/api/browse":
            self._send_json(browse_directories(q("dir") or ""))
            return
        if path == "/api/experiment-setups" or path.startswith("/api/experiment-setups/"):
            self._handle_setup(path)
            return
        if path == "/api/benchmarks" or path.startswith("/api/benchmarks/"):
            self._handle_benchmarks(path)
            return

        # Per-request read-only store; never migrate incompatible evidence.
        try:
            source = q("benchmark_experiment")
            store = self._registered_store(source) if source else self.chatbot.open_store()
        except (IncompatibleObservabilityDB, ValueError, KeyError) as exc:
            self._error(409, str(exc))
            return
        if path.startswith("/api/review/assignments/"):
            self._handle_review_assignment(path)
        elif path == "/api/workspace" or path.startswith("/api/workspace/"):
            self._handle_workspace(path, q)
        elif self.chatbot.workspace is not None and path in {
            "/api/turns",
            "/api/experiments",
        }:
            self._error(
                400,
                "workspace reads must be scoped by store_id; unscoped search is refused",
            )
        elif self.chatbot.workspace is not None and (
            path.startswith("/api/turn/")
            or path.startswith("/api/spans/")
            or path.startswith("/api/experiment/")
            or path == "/api/feedback"
            or path.startswith("/api/feedback/")
        ):
            self._error(
                400,
                "workspace reads must use the store-aware /api/workspace routes",
            )
        elif path == "/api/meta":
            self._send_json(
                {
                    "workflow_path": self.chatbot.workflow_path,
                    "workflow_name": (
                        self.chatbot.workflow_path.rstrip("/\\").rsplit("/", 1)[-1]
                        if self.chatbot.workflow_path
                        else ""
                    ),
                    "db_path": self.chatbot.db_path,
                    "db_available": store is not None,
                    "db_size_bytes": store.db_size_bytes() if store else 0,
                }
            )
        elif store is None:
            if path == "/api/health":
                self._send_json(
                    {"writer_health": None, "db_size_bytes": 0, "db_available": False}
                )
            elif path in ("/api/channels", "/api/conversations", "/api/turns"):
                self._send_json({"channels": [], "conversations": [], "turns": []})
            elif path == "/api/feedback":
                self._send_json({"feedback": []})
            elif path == "/api/experiments":
                # An empty state, not "observability DB not found": a cold start
                # has no experiments, which is a fact about the DB rather than
                # an error the operator can act on.
                self._send_json({"experiments": []})
            else:
                self._error(404, "observability DB not found")
        elif path == "/api/channels":
            self._send_json({"channels": store.list_channels()})
        elif path == "/api/conversations":
            self._send_json(
                {
                    "conversations": store.list_conversations(
                        channel_id=q("channel"),
                        limit=self._int(q("limit"), 100),
                        offset=self._int(q("offset"), 0),
                    )
                }
            )
        elif path == "/api/turns":
            success = q("success")
            attempt_filter = None
            if q("attempt") is not None:
                try:
                    attempt_filter = int(q("attempt"))
                except ValueError:
                    self._error(400, "attempt must be an integer")
                    return
            # (b) the rail's low-confidence filter: a turn whose least
            # confident intent decision recorded a top-k margin below this.
            # Applied to the annotated page, after the store's own filters;
            # a turn with no recorded margin is never counted.
            low_confidence_below = None
            if q("low_confidence_below") is not None:
                low_confidence_below = _finite_number(
                    self._float_or_none(q("low_confidence_below"))
                )
                if low_confidence_below is None or low_confidence_below < 0:
                    self._error(
                        400, "low_confidence_below must be a non-negative number"
                    )
                    return
            turns = store.list_turns(
                channel_id=q("channel"),
                conversation_id=(
                    self._int(q("conversation"), None)
                    if q("conversation") is not None
                    else None
                ),
                status=q("status"),
                success=(
                    None
                    if success is None
                    else success in ("1", "true", "True")
                ),
                command_name=q("command"),
                context=q("context"),
                experiment_id=q("experiment"),
                task_id=q("task"),
                attempt=attempt_filter,
                limit=self._int(q("limit"), 100),
                offset=self._int(q("offset"), 0),
            )
            annotate_turn_rows(store, turns)
            if low_confidence_below is not None:
                turns = [
                    turn
                    for turn in turns
                    if is_low_confidence(turn["decision_signals"], low_confidence_below)
                ]
            self._send_json({"turns": turns})
        elif path.startswith("/api/turn/"):
            turn_key = path[len("/api/turn/") :]
            turn = store.get_turn(turn_key)
            if turn is None:
                self._error(404, "turn not found")
                return
            try:
                turn["record"] = json.loads(turn.pop("record_json"))
            except (ValueError, KeyError):
                turn["record"] = None
            annotate_turn_detail(turn, store.get_spans(turn_key))
            self._send_json({"turn": turn})
        elif path == "/api/feedback":
            self._send_json(
                {
                    "feedback": store.list_feedback(
                        channel_id=q("channel"),
                        limit=self._int(q("limit"), 100),
                    )
                }
            )
        elif path.startswith("/api/feedback/"):
            turn_key = unquote(path[len("/api/feedback/") :])
            feedback = store.get_feedback(turn_key)
            if feedback is None:
                self._error(404, "feedback not found")
                return
            self._send_json({"feedback": feedback})
        elif path.startswith("/api/spans/"):
            trace_id = path[len("/api/spans/") :]
            spans = store.get_spans(trace_id)
            for span in spans:
                try:
                    span["attributes"] = json.loads(span["attributes"])
                except (ValueError, TypeError, KeyError):
                    pass
            self._send_json({"spans": spans})
        elif path == "/api/experiments" or path.startswith("/api/experiment/"):
            self._handle_experiments(store, path, q)
        elif path.startswith("/api/artifact/"):
            self._serve_artifact(store, path[len("/api/artifact/") :])
        elif path == "/api/health":
            self._send_json(
                {
                    "writer_health": store.writer_health(),
                    "db_size_bytes": store.db_size_bytes(),
                    "db_available": True,
                }
            )
        else:
            self._error(404, "not found")

    def _handle_review_assignment(self, path: str) -> None:
        """Return a workspace-scoped assignment or its answer export."""
        if self.chatbot.workspace is None:
            self._error(404, "no observability workspace is loaded")
            return
        encoded_id = path[len("/api/review/assignments/") :]
        if "/rows/" in encoded_id:
            self._handle_review_evidence(encoded_id)
            return
        export = encoded_id.endswith("/export")
        progress = encoded_id.endswith("/progress")
        if export:
            encoded_id = encoded_id[: -len("/export")]
        elif progress:
            encoded_id = encoded_id[: -len("/progress")]
        if not encoded_id:
            self._error(404, "not found")
            return
        assignment_id = unquote(encoded_id)
        try:
            sidecar = self.chatbot.open_review_sidecar()
            if export:
                sidecar.authorize_capability(
                    assignment_id,
                    self._review_capability(),
                    "adjudicator",
                )
                assignment = sidecar.export_assignment(assignment_id)
            elif progress:
                assignment = sidecar.assignment_progress(
                    assignment_id, self._review_capability()
                )
            else:
                assignment = sidecar.get_assignment(assignment_id)
        except ReviewAuthorizationError as exc:
            self._error(403, str(exc))
            return
        except ReviewValidationError as exc:
            self._error(400, str(exc))
            return
        except ReviewNotFoundError:
            self._error(404, "review assignment not found")
            return
        if export:
            self._send_json({"export": assignment})
        elif progress:
            self._send_json({"progress": assignment})
        else:
            self._send_json({"assignment": assignment})

    def _handle_review_evidence(self, encoded_path: str) -> None:
        """Return the capability-gated evidence projection for one assigned row."""
        workspace = self.chatbot.workspace
        if workspace is None:
            self._error(404, "no observability workspace is loaded")
            return
        encoded_id, separator, rest = encoded_path.partition("/rows/")
        encoded_row_id, operation_separator, operation = rest.partition("/")
        if (
            not separator
            or not operation_separator
            or not encoded_id
            or not encoded_row_id
            or operation not in {"turn", "trace"}
        ):
            self._error(404, "not found")
            return
        assignment_id = unquote(encoded_id)
        row_id = unquote(encoded_row_id)
        try:
            progress = self.chatbot.open_review_sidecar().assignment_progress(
                assignment_id, self._review_capability()
            )
            row = next(
                (
                    candidate
                    for candidate in progress["assignment"]["rows"]
                    if candidate["id"] == row_id
                ),
                None,
            )
            if row is None:
                self._error(404, "review row not found")
                return
            turn_ref = row["turn_ref"]
            store_id = turn_ref.get("store_id")
            logical_turn_key = turn_ref.get("logical_turn_key")
            if not store_id or not logical_turn_key:
                self._error(400, "review row does not contain a scoped workspace turn")
                return
            blinded = bool(progress["assignment"]["blinded"])
            if operation == "turn":
                turn = workspace.turn(store_id, logical_turn_key)
                if turn is None:
                    self._error(404, "turn not found in the named store")
                    return
                self._send_json(
                    {"turn": project_review_turn(turn, blinded=blinded)}
                )
            else:
                self._send_json(
                    {
                        "spans": project_review_trace(
                            workspace.trace(store_id, logical_turn_key),
                            blinded=blinded,
                        )
                    }
                )
        except ReviewAuthorizationError as exc:
            self._error(403, str(exc))
        except ReviewValidationError as exc:
            self._error(400, str(exc))
        except ReviewNotFoundError:
            self._error(404, "review assignment not found")
        except UnknownWorkspaceStore as exc:
            self._error(404, str(exc.args[0] if exc.args else exc))
        except WorkspaceIntegrityError as exc:
            self._error(409, str(exc))
        except WorkspaceBusyError as exc:
            self._error(503, str(exc))

    def _handle_workspace(self, path: str, q: Any) -> None:
        """Read-only HTTP projection of a validated multi-store workspace."""
        workspace = self.chatbot.workspace
        if workspace is None:
            self._error(404, "no observability workspace is loaded")
            return
        try:
            if path in {"/api/workspace", "/api/workspace/summary"}:
                self._send_json({"workspace": workspace.summary()})
                return
            if path == "/api/workspace/stores":
                self._send_json({"stores": workspace.stores()})
                return
            if path == "/api/workspace/experiments":
                self._send_json({"experiments": workspace.experiments()})
                return
            if path == "/api/workspace/projected_attempts":
                attempt = None
                if q("attempt") is not None:
                    try:
                        attempt = int(q("attempt"))
                    except ValueError:
                        self._error(400, "attempt must be an integer")
                        return
                projected = workspace.projected_attempts(
                    experiment_id=q("experiment"),
                    task_id=q("task"),
                    attempt=attempt,
                )
                annotate_projected_attempts(workspace, projected)
                self._send_json({"projected_attempts": projected})
                return
            experiment_prefix = "/api/workspace/experiment/"
            if path.startswith(experiment_prefix):
                rest = path[len(experiment_prefix) :]
                encoded_id, separator, operation = rest.partition("/")
                experiment_id = unquote(encoded_id)
                if not separator or operation not in {
                    "segments",
                    "tasks",
                    "attempts",
                }:
                    self._error(404, "not found")
                    return
                if operation == "segments":
                    segments = workspace.segments(experiment_id)
                    for segment in segments:
                        segment["evidence"] = evidence_verdict(
                            workspace.evidence_runs(
                                segment["store_id"], segment["local_experiment_id"]
                            )
                        )
                        local = workspace.experiment(
                            segment["store_id"], segment["local_experiment_id"]
                        )
                        segment["provenance"] = experiment_provenance(
                            local or {},
                            workspace.attempts_in_store(
                                segment["store_id"], segment["local_experiment_id"]
                            ),
                        )
                        # Provenance lists the pin as recorded; this checks it
                        # against the catalogue file the manifest points at.
                        # Attached to the segment because the pin belongs to
                        # the local experiment row, and two segments of one
                        # logical experiment may have been pinned differently.
                        segment["benchmark_pin"] = benchmark_pin_check(
                            workspace.workflow_folderpath, local or {}
                        )
                    self._send_json({"segments": segments})
                elif operation == "tasks":
                    self._send_json({"tasks": workspace.tasks(experiment_id)})
                else:
                    rows = workspace.attempts(experiment_id, task_id=q("task"))
                    annotate_workspace_attempts(
                        workspace,
                        rows,
                        _workspace_segment_verdicts(workspace, experiment_id),
                    )
                    self._send_json({"attempts": rows})
                return
            for noun in ("turn", "trace", "spans"):
                prefix = f"/api/workspace/{noun}/"
                if not path.startswith(prefix):
                    continue
                rest = path[len(prefix) :]
                encoded_store, separator, encoded_key = rest.partition("/")
                if not separator or not encoded_store or not encoded_key:
                    self._error(
                        400,
                        f"{noun} reads require both store_id and logical_turn_key",
                    )
                    return
                store_id = unquote(encoded_store)
                logical_turn_key = unquote(encoded_key)
                if noun == "turn":
                    turn = workspace.turn(store_id, logical_turn_key)
                    if turn is None:
                        self._error(404, "turn not found in the named store")
                        return
                    annotate_turn_detail(
                        turn, workspace.trace(store_id, logical_turn_key)
                    )
                    self._send_json({"turn": turn})
                else:
                    self._send_json(
                        {"spans": workspace.trace(store_id, logical_turn_key)}
                    )
                return
            self._error(404, "not found")
        except (UnknownWorkspaceStore, UnknownLogicalExperiment) as exc:
            self._error(404, str(exc.args[0] if exc.args else exc))
        except WorkspaceIntegrityError as exc:
            self._error(409, str(exc))
        except WorkspaceBusyError as exc:
            self._error(503, str(exc))

    def _handle_setup(self, path, *, body=None, write=False):
        # Setup reviews are live workflow authoring records, never sealed evidence.
        if self.chatbot.workspace is not None:
            self._error(
                403,
                "Select a live workflow to review experiment setups; sealed workspaces are read-only",
            )
            return
        workflow_path = self.chatbot.workflow_path
        if not workflow_path:
            self._error(409, "Select a workflow before reviewing experiment setups")
            return
        setups = ExperimentSetups(workflow_path)
        rest = path[len("/api/experiment-setups") :].strip("/")
        parts = rest.split("/") if rest else []
        try:
            if write and not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
            if not parts:
                if write:
                    result = setups.save(body.get("spec"), body.get("expected_revision"))
                    self._send_json({"setup": result}, status=201)
                else:
                    self._send_json({"setups": setups.list()})
            elif len(parts) == 1 and not write:
                self._send_json({"setup": setups.get(unquote(parts[0]))})
            elif len(parts) == 2 and parts[1] == "decisions" and write:
                result = setups.decide(
                    unquote(parts[0]),
                    body.get("revision"),
                    body.get("digest"),
                    body.get("decision"),
                    body.get("reviewer"),
                    body.get("comment", ""),
                )
                self._send_json({"setup": result}, status=201)
            elif len(parts) == 2 and parts[1] == "export" and not write:
                current = setups.get(unquote(parts[0]))
                self._send_json(
                    setups.approved(
                        current["experiment_id"], current["revision"], current["digest"]
                    )
                )
            else:
                self._error(404, "not found")
        except SetupConflict as exc:
            self._error(409, str(exc))
        except KeyError:
            self._error(404, "setup not found")
        except (ValueError, TypeError) as exc:
            self._error(400, str(exc))

    def _benchmark_workflow_path(self, *, write: bool = False) -> Optional[str]:
        """Workflow folder for versioned benchmark corpus files.

        In workspace mode this is the folder the manifest named at seal time,
        and it is served for READS only: the corpus a sealed run was pinned to
        is part of reading that run's evidence, and refusing it left the pin as
        a digest with nothing behind it. Writes stay refused exactly as before
        — the folder is a live checkout that a read-only workspace must not
        touch, and `write=True` returns None before the manifest is consulted.

        A manifest with no folder keeps its 409, and so does one whose folder
        is gone: nothing was found to read, and the reason is quoted.
        """
        if self.chatbot.workspace is not None:
            if write:
                self._error(
                    403,
                    "workspace mode is read-only; benchmark corpus files cannot be changed",
                )
                return None
            declared = self.chatbot.workspace.workflow_folderpath
            if not declared:
                self._error(
                    409,
                    "benchmarks are available in live workflow mode only",
                )
                return None
            if not os.path.isdir(declared):
                self._error(
                    409,
                    "the workflow folder named by this workspace is not on "
                    f"this machine: {declared}",
                )
                return None
            return declared
        workflow_path = (self.chatbot.workflow_path or "").strip()
        if not workflow_path:
            self._error(
                409,
                "select a workflow before using benchmarks",
            )
            return None
        return workflow_path

    def _handle_navigation(self):
        from .navigation import build_navigation, read_source
        benchmarks, registrations, sources, warnings = [], [], [], []
        folder = self.chatbot.workflow_path
        workspace = self.chatbot.workspace
        if workspace is not None:
            folder = workspace.summary().get("workflow_folderpath")
        if folder and os.path.isdir(folder):
            for bid in list_benchmarks(folder):
                versions = list_versions(folder, bid)
                row = {"benchmark_id": bid, "versions": versions}
                if versions:
                    try:
                        row.update(load_version(folder, bid, versions[-1]))
                    except BenchmarkManifestError as exc:
                        warnings.append(str(exc))
                benchmarks.append(row)
                if workspace is None:
                    registrations.extend(benchmark_setup.registered_experiments(folder, bid))
        if workspace is not None:
            for descriptor in workspace.stores():
                sid = descriptor["store_id"]
                with workspace.registry.open(sid) as store:
                    sources.append(read_source(store, {"store_id": sid}))
            self._send_json({"root": build_navigation(benchmarks, [], sources, warnings)})
            return
        try:
            store = self.chatbot.open_store()
            if store:
                sources.append({"store": store, "source": None})
        except (IncompatibleObservabilityDB, OSError, sqlite3.Error) as exc:
            warnings.append(STORE_UNAVAILABLE + str(exc))
        for record in registrations:
            if not record.get("store"):
                continue
            try:
                store = self._registered_store(record["experiment_id"])
                detail = store.get_experiment(record["experiment_id"])
                if detail is not None:
                    record["archived"] = bool(detail.get("archived"))
                sources.append({"store": store,
                    "source": {"benchmark_experiment": record["experiment_id"]},
                    "experiment_id": record["experiment_id"]})
            except (ValueError, KeyError, OSError, sqlite3.Error, IncompatibleObservabilityDB) as exc:
                record["warning"] = str(exc)
        self._send_json({"root": build_navigation(benchmarks, registrations, sources, warnings)})

    def _handle_human_feedback(self, query, body=None):
        """Owner-authenticated annotations in the selected evidence database."""
        q = lambda key: (query.get(key) or [None])[0]
        writing = body is not None
        if writing and not isinstance(body, dict):
            self._error(400, "body must be a JSON object")
            return
        turn_key = q("turn_key")
        if not turn_key:
            self._error(400, "turn_key is required")
            return
        try:
            workspace = self.chatbot.workspace
            if workspace is not None:
                if writing:
                    self._error(403, "workspace evidence is read-only; annotate the working database")
                    return
                with workspace.registry.open(q("store_id") or "") as store:
                    if store.get_turn(turn_key) is None:
                        self._error(404, "turn not found")
                        return
                    self._send_json({"feedback": store.list_human_feedback(turn_key), "read_only": True})
                return
            source = q("benchmark_experiment")
            store = self._registered_store(source) if source else self.chatbot.open_store()
            turn = store.get_turn(turn_key) if store else None
            if turn is None:
                self._error(404, "turn not found")
                return
            if source and turn.get("experiment_id") != source:
                self._error(400, "turn does not belong to the selected experiment")
                return
            if writing:
                required = {"target_kind", "span_ids", "target_label", "provenance"}
                allowed = required | {"comment", "went_wrong", "worked", "should_change"}
                if not required <= set(body) or not set(body) <= allowed:
                    raise ValueError(
                        "provide target_kind, span_ids, target_label, provenance, "
                        "and comment or the went_wrong / worked / should_change fields"
                    )
                ObservabilityStore.open_for_annotation(store.db_path).add_human_feedback(turn_key, **body)
            self._send_json({"feedback": store.list_human_feedback(turn_key), "read_only": False},
                            status=201 if writing else 200)
        except (IncompatibleObservabilityDB, UnknownWorkspaceStore) as exc:
            self._error(409, str(exc))
        except (ValueError, TypeError, KeyError) as exc:
            self._error(400, str(exc))

    def _registered_store(self, experiment_id):
        if self.chatbot.workspace is not None or not self.chatbot.workflow_path:
            raise ValueError("registered experiments require a selected live workflow")
        record = benchmark_setup.load_experiment(self.chatbot.workflow_path, experiment_id)
        target = record.get("store")
        if not target:
            raise ValueError("experiment has not started")
        store = ReadOnlyObservabilityStore(target["db_path"])
        if store.store_identity() != target["store_id"]:
            raise ValueError("registered experiment evidence store identity changed")
        detail = store.get_experiment(experiment_id)
        if not detail or any(detail.get(key) != record[key] for key in
                ("benchmark_id", "benchmark_version", "benchmark_digest_sha256")):
            raise ValueError("recorded experiment does not match its benchmark registration")
        return store

    def _handle_benchmark_registration(self, experiment_id):
        folder = self._benchmark_workflow_path()
        if folder is None:
            return
        try:
            record, manifest = benchmark_setup.experiment_manifest(folder, experiment_id)
        except (KeyError, benchmark_setup.ExperimentDeleted):
            self._error(404, "experiment not found")
            return
        except (ValueError, BenchmarkManifestError) as exc:
            self._error(409, str(exc))
            return
        recorded, warning = False, None
        if record.get("store"):
            try:
                self._registered_store(experiment_id)
                recorded = True
            except (ValueError, OSError, sqlite3.Error, IncompatibleObservabilityDB) as exc:
                warning = str(exc)
        self._send_json({"experiment": record, "benchmark": manifest,
                         "recorded": recorded, "warning": warning,
                         "can_delete": record.get("store") is None and self.chatbot.workspace is None})

    def _handle_registration_patch(self, path: str, body: dict[str, Any]) -> None:
        """`PATCH /api/benchmark-experiments/<id>` -- the author's description.

        The registration file is setup data, not evidence, and this route can
        only reach one whose runner has not claimed it:
        `update_experiment_description` refuses a bound registration under the
        same lock the binding takes.
        """
        experiment_id = unquote(path[len("/api/benchmark-experiments/"):]).rstrip("/")
        if not experiment_id:
            self._error(404, "not found")
            return
        if "description" not in body:
            self._error(400, 'nothing to patch: send {"description": "..."}')
            return
        folder = self._benchmark_workflow_path(write=True)
        if folder is None:
            return
        try:
            record = benchmark_setup.update_experiment_description(
                folder, experiment_id, body.get("description")
            )
        except (KeyError, benchmark_setup.ExperimentDeleted):
            self._error(404, "experiment not found")
            return
        except benchmark_setup.BenchmarkSetupConflict as exc:
            self._error(409, str(exc))
            return
        except (ValueError, TypeError) as exc:
            self._error(400, str(exc))
            return
        self._send_json({"experiment": record})

    def _benchmark_experiments(self, benchmark_id):
        from .navigation import newest_experiments_first

        folder = self._benchmark_workflow_path()
        if folder is None:
            return
        rows, warning = [], None
        if self.chatbot.workspace is not None:
            workspace = self.chatbot.workspace
            for logical in workspace.experiments():
                matches = [workspace.experiment(segment["store_id"], segment["local_experiment_id"])
                           for segment in workspace.segments(logical["experiment_id"])]
                benchmark_rows = [
                    row for row in matches
                    if row and row.get("benchmark_id") == benchmark_id
                ]
                versions = sorted(
                    {row["benchmark_version"] for row in benchmark_rows}
                )
                if versions:
                    rows.append(
                        dict(
                            logical,
                            benchmark_version=", ".join(versions),
                            workspace=True,
                            archived=all(
                                bool(row.get("archived")) for row in benchmark_rows
                            ),
                            created_at=max(
                                row.get("created_at") or "" for row in benchmark_rows
                            ),
                        )
                    )
        else:
            registrations = benchmark_setup.registered_experiments(folder, benchmark_id)
            rows = [dict(row, registered=True, status="registered") for row in registrations]
            for row in rows:
                if not row.get("store"):
                    row["archived"] = False
                    continue
                try:
                    detail = self._registered_store(row["experiment_id"]).get_experiment(
                        row["experiment_id"]
                    )
                    row["archived"] = bool(detail and detail.get("archived"))
                except (
                    ValueError,
                    KeyError,
                    OSError,
                    sqlite3.Error,
                    IncompatibleObservabilityDB,
                ):
                    row["archived"] = False
            try:
                store = self.chatbot.open_store()
                if store:
                    offset = 0
                    while True:
                        batch = store.list_experiments(limit=200, offset=offset)
                        for row in batch:
                            if row.get("benchmark_id") == benchmark_id:
                                if any(r["experiment_id"] == row["experiment_id"] for r in rows):
                                    continue
                                rows.append(row)
                        if len(batch) < 200:
                            break
                        offset += len(batch)
            except IncompatibleObservabilityDB as exc:
                warning = STORE_UNAVAILABLE + str(exc)
        self._send_json(
            {"experiments": newest_experiments_first(rows), "warning": warning}
        )

    def _handle_benchmarks(self, path: str) -> None:
        """Read workflow-local benchmark catalogs from ``<workflow>/benchmarks/``."""
        workflow_path = self._benchmark_workflow_path(write=False)
        if workflow_path is None:
            return
        if path == "/api/benchmarks":
            payload = []
            for benchmark_id in list_benchmarks(workflow_path):
                payload.append(
                    {
                        "benchmark_id": benchmark_id,
                        "versions": list_versions(workflow_path, benchmark_id),
                    }
                )
            for row in payload:
                if row["versions"]:
                    try:
                        manifest = load_version(workflow_path, row["benchmark_id"], row["versions"][-1])
                        if "title" in manifest:
                            row["title"] = manifest["title"]
                    except BenchmarkManifestError:
                        pass
            self._send_json({"benchmarks": payload})
            return

        rest = path[len("/api/benchmarks/") :]
        benchmark_id, _, tail = rest.partition("/")
        benchmark_id = unquote(benchmark_id)
        if not benchmark_id:
            self._error(404, "not found")
            return
        if tail == "":
            self._send_json(
                {
                    "benchmark_id": benchmark_id,
                    "versions": list_versions(workflow_path, benchmark_id),
                }
            )
            return
        if tail == "experiments":
            self._benchmark_experiments(benchmark_id)
            return
        if tail == "analysis":
            try:
                analysis = load_analysis(workflow_path, benchmark_id)
            except BenchmarkManifestError as exc:
                self._error(400, str(exc))
                return
            self._send_json({"benchmark_id": benchmark_id, "analysis": analysis})
            return
        version_prefix, _, version = tail.partition("/")
        if version_prefix != "versions" or not version:
            self._error(404, "not found")
            return
        version = unquote(version)
        try:
            manifest = load_version(workflow_path, benchmark_id, version)
        except BenchmarkManifestError as exc:
            self._error(404, str(exc))
            return
        self._send_json({"version": manifest})

    def _handle_benchmark_post(self, path: str, body: Any) -> None:
        """Create one immutable benchmark version file under the workflow folder."""
        workflow_path = self._benchmark_workflow_path(write=True)
        if workflow_path is None:
            return
        if not isinstance(body, dict):
            self._error(400, "body must be a JSON object")
            return
        spec = dict(body)
        if path.startswith("/api/benchmarks/") and path.endswith("/versions"):
            encoded_id = path[len("/api/benchmarks/") : -len("/versions")].rstrip("/")
            url_benchmark_id = unquote(encoded_id)
            if not url_benchmark_id:
                self._error(404, "not found")
                return
            body_benchmark_id = spec.get("benchmark_id")
            if body_benchmark_id is not None and body_benchmark_id != url_benchmark_id:
                self._error(
                    400,
                    "benchmark_id in body does not match the URL path",
                )
                return
            spec.setdefault("benchmark_id", url_benchmark_id)
        try:
            written = write_version(workflow_path, spec)
        except BenchmarkAlreadyExistsError as exc:
            self._error(409, str(exc))
            return
        except BenchmarkManifestError as exc:
            self._error(400, str(exc))
            return
        self._send_json({"version": written}, status=201)

    @staticmethod
    def _analysis_payload_from_body(body: dict[str, Any]) -> Any:
        """Accept a bare object or ``{"analysis": ...}`` wrapper."""
        if "analysis" in body:
            if set(body) - {"analysis"}:
                raise ValueError(
                    "unexpected fields: this route updates analysis only"
                )
            return body["analysis"]
        return body

    def _handle_benchmark_analysis_put(self, path: str, body: dict[str, Any]) -> None:
        """``PUT /api/benchmarks/<id>/analysis`` — mutable sibling analysis file."""
        workflow_path = self._benchmark_workflow_path(write=True)
        if workflow_path is None:
            return
        encoded_id = path[len("/api/benchmarks/") : -len("/analysis")].rstrip("/")
        benchmark_id = unquote(encoded_id)
        if not benchmark_id:
            self._error(404, "not found")
            return
        try:
            payload = self._analysis_payload_from_body(body)
        except ValueError as exc:
            self._error(400, str(exc))
            return
        try:
            written = write_analysis(workflow_path, benchmark_id, payload)
        except BenchmarkManifestError as exc:
            self._error(400, str(exc))
            return
        self._send_json({"benchmark_id": benchmark_id, "analysis": written})

    def _handle_experiments(
        self,
        store: ReadOnlyObservabilityStore,
        path: str,
        q: Any,
    ) -> None:
        """The `/api/experiment*` GET surface.

        The noun choice is deliberate: this surface never calls an experiment
        attempt a "run". An experiment has tasks, a task has attempts, and an
        attempt resolves to the channel/conversation/turn keys the existing
        trace views already render — so nothing here re-implements a viewer.

        A DB written before the experiment tables existed 404s with a reason a
        human can act on, rather than raising `no such table` behind a generic
        500.
        """
        if not store.has_feature(FEATURE_EXPERIMENTS_V1):
            self._error(404, "this database predates experiment recording")
            return
        if path == "/api/experiments":
            self._send_json(
                {
                    "experiments": store.list_experiments(
                        status=q("status"),
                        arm=q("arm"),
                        limit=self._int(q("limit"), 100),
                        offset=self._int(q("offset"), 0),
                    )
                }
            )
            return

        rest = path[len("/api/experiment/") :]
        # Split first, THEN decode: decoding first would let a %2F inside an id
        # invent a sub-path segment. The SPA sends encodeURIComponent(id) and
        # create_experiment accepts any caller-supplied id, so an id containing
        # a space or a slash would otherwise 404 forever.
        experiment_id, _, sub = rest.partition("/")
        experiment_id = unquote(experiment_id)
        if not experiment_id:
            self._error(404, "not found")
            return
        if sub == "":
            detail = store.get_experiment(experiment_id)
            if detail is None:
                self._error(404, "experiment not found")
                return
            detail["evidence"] = evidence_verdict(detail.get("evidence_runs"))
            detail["provenance"] = experiment_provenance(
                detail, store.experiment_attempt_rows(experiment_id)
            )
            self._send_json({"experiment": detail})
        elif sub == "tasks":
            if store.get_experiment(experiment_id) is None:
                self._error(404, "experiment not found")
                return
            self._send_json({"tasks": store.experiment_tasks(experiment_id)})
        elif sub == "attempts":
            detail = store.get_experiment(experiment_id)
            if detail is None:
                self._error(404, "experiment not found")
                return
            rows = store.experiment_attempt_rows(experiment_id, task_id=q("task"))
            annotate_attempt_rows(
                store, rows, evidence_verdict(detail.get("evidence_runs"))
            )
            self._send_json({"attempts": rows})
        elif sub == "score":
            try:
                self._send_json({"score": store.experiment_scores(experiment_id)})
            except ExperimentNotFound:
                self._error(404, "experiment not found")
        elif sub == "compare":
            # Resolve the experiment BEFORE branching on the baseline: folding
            # a missing experiment into `(detail or {})` reported it as 400
            # "no baseline" rather than 404 "experiment not found", which sends
            # the reader looking for the wrong thing.
            detail = store.get_experiment(experiment_id)
            if detail is None:
                self._error(404, "experiment not found")
                return
            baseline = q("baseline") or detail.get("baseline_experiment_id")
            if not baseline:
                self._error(
                    400,
                    "no baseline: pass ?baseline=<experiment_id> or set "
                    "baseline_experiment_id on the experiment",
                )
                return
            try:
                comparison = store.compare_experiments(experiment_id, baseline)
            except ExperimentNotFound as exc:
                self._error(404, f"experiment not found: {exc.experiment_id}")
                return
            # (c) the comparability check rides along on BOTH answers: a
            # provenance difference is quoted, never a refusal, so the 409 the
            # store already issues for a differing benchmark pin stays the
            # only thing that blocks the view.
            baseline_detail = store.get_experiment(baseline)
            comparison["provenance_differences"] = provenance_differences(
                experiment_provenance(
                    detail, store.experiment_attempt_rows(experiment_id)
                ),
                experiment_provenance(
                    baseline_detail or {}, store.experiment_attempt_rows(baseline)
                ),
            )
            # 409, not 200-with-a-flag: an incomparable pair is a refusal, and a
            # client that renders whatever it got would render a comparison of
            # two runs that share no task.
            status = 200 if comparison.get("comparable") else 409
            self._send_json(comparison, status=status)
        else:
            self._error(404, "not found")

    def _handle_experiment_patch(
        self,
        path: str,
        body: dict[str, Any],
        source_experiment_id: Optional[str] = None,
    ) -> None:
        """`PATCH /api/experiment/<id>` -- editable annotations.

        Admitted on the annotation argument in `[DR30]`: the invariant
        protected is "recorded observability data stays read-only over HTTP"
        (studio design §3.4, the access-control section), and `notes` plus
        `archived` are annotation columns that cannot alter any span, turn,
        artifact, attempt outcome or score.
        """
        rest = path[len("/api/experiment/") :]
        # partition, not split-and-discard: the GET side validates its sub-path
        # and 404s on an unknown one, and a write route that silently accepted
        # /api/experiment/<id>/anything would make every GET sub-path an
        # undocumented alias for the notes PATCH.
        experiment_id, _, sub = rest.partition("/")
        experiment_id = unquote(experiment_id)
        if not experiment_id or sub:
            self._error(404, "not found")
            return
        if source_experiment_id is not None and source_experiment_id != experiment_id:
            self._error(400, "benchmark experiment does not match the URL experiment")
            return
        try:
            store = (
                self._registered_store(source_experiment_id)
                if source_experiment_id
                else self.chatbot.open_store()
            )
        except (ValueError, KeyError, OSError, sqlite3.Error, IncompatibleObservabilityDB) as exc:
            self._error(409, str(exc))
            return
        if store is None or not store.has_feature(FEATURE_EXPERIMENTS_V1):
            self._error(404, "this database predates experiment recording")
            return
        if "analysis" in body:
            self._error(400, "analysis is not a field of an experiment; use notes")
            return
        if set(body) - {"notes", "archived"}:
            self._error(
                400, "unexpected fields: this route updates notes and archived only"
            )
            return
        if not body:
            self._error(
                400, 'nothing to patch: send {"notes": "..."} or {"archived": true}'
            )
            return
        notes = body.get("notes")
        if "notes" in body and notes is not None and not isinstance(notes, str):
            self._error(400, "notes must be a string or null")
            return
        archived = body.get("archived")
        if "archived" in body and not isinstance(archived, bool):
            self._error(400, "archived must be true or false")
            return
        try:
            # `[DR53]`: the feature check above ran through the per-request
            # READ-ONLY handle, so a PATCH against a pre-experiments snapshot
            # cannot be what creates the tables in it.
            writable = ObservabilityStore.open_for_annotation(store.db_path)
            if "notes" in body:
                writable.update_experiment_notes(experiment_id, notes)
            if "archived" in body:
                writable.update_experiment_archived(experiment_id, archived)
        except ExperimentNotFound:
            self._error(404, "experiment not found")
            return
        except (OSError, sqlite3.Error) as exc:
            self._error(
                500, f"could not update experiment annotations: {type(exc).__name__}"
            )
            return
        self._send_json({"experiment": store.get_experiment(experiment_id)})

    def _serve_artifact(
        self, store: ReadOnlyObservabilityStore, artifact_id: str
    ) -> None:
        """Offloaded artifact content, with its stored content-type.

        HTML-ish content is only ever *rendered* inside a sandboxed iframe by
        the SPA; the raw response is additionally neutralized with
        ``CSP: default-src 'none'; sandbox`` so navigating to the URL directly
        cannot run scripts either.
        """
        artifact = store.get_artifact(artifact_id)
        if artifact is None:
            self._error(404, "artifact not found")
            return
        content_type = artifact.get("content_type") or "application/octet-stream"
        value = artifact.get("inline_value") or b""
        if isinstance(value, str):
            value = value.encode("utf-8")
        base_type = content_type.split(";")[0].strip().lower()
        headers = {
            "Content-Security-Policy": "default-src 'none'; sandbox",
            "Content-Disposition": "inline",
        }
        if base_type in _HTMLISH_TYPES:
            headers["X-FW-Artifact-Htmlish"] = "1"
        self._send(200, bytes(value), content_type, headers)

    @staticmethod
    def _float_or_none(value: Optional[str]) -> Optional[float]:
        try:
            return float(value) if value is not None else None
        except ValueError:
            return None

    @staticmethod
    def _int(value: Optional[str], default: Any) -> Any:
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default


# ----------------------------------------------------------------------
# CLI entry points (used by `fastworkflow run_chatbot`; kept import-light)
# ----------------------------------------------------------------------


def _open_in_browser(url: str) -> None:
    """Open the user's default browser; never noisy, never fatal.

    On WSL there is usually no Linux browser — stdlib ``webbrowser`` falls
    through to xdg-open, which sprays a 'not found' line per candidate and
    gives up — while the WINDOWS default browser is one hop away. Prefer
    ``wslview`` (wslu) then ``powershell.exe Start-Process``; the printed URL
    in the banner is always the fallback.
    """
    import shutil
    import subprocess

    if "PYTEST_CURRENT_TEST" in os.environ:
        return  # pytest is the only skip path; there is no --no-browser flag

    is_wsl = False
    try:
        with open("/proc/version", "r", encoding="utf-8") as f:
            is_wsl = "microsoft" in f.read().lower()
    except OSError:
        pass
    if is_wsl:
        for cmd in (
            ["wslview", url],
            # The token is token_urlsafe (A-Za-z0-9_-), so the single-quoted
            # PowerShell literal cannot be escaped out of.
            ["powershell.exe", "-NoProfile", "-Command", f"Start-Process '{url}'"],
        ):
            if shutil.which(cmd[0]) is None:
                continue
            try:
                subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                return
            except OSError:
                continue
        return  # no opener available; the banner URL is the path
    import webbrowser

    try:
        webbrowser.open(url)
    except Exception:
        pass


def run_prune(db_path: str) -> dict[str, int]:
    """Library maintenance utility: bounded prune + vacuum.

    Not wired to any CLI flag or HTTP route — pruning runs automatically at
    sink startup; this exists for scripts/tests that need it on demand.
    """
    return ObservabilityStore(db_path).prune()


def run_forget_channel(
    db_path: str, channel_id: str, workflow_path: str = ""
) -> dict[str, int]:
    """Library erasure utility: delete one channel everywhere.

    Not wired to any CLI flag or HTTP route — the chatbot UI exposes the
    all-channel Clear-conversations action instead; this remains the
    single-channel primitive for scripts/tests (e.g. a deletion request for
    one API channel). Also deletes the LEGACY per-channel conversation DB
    (``conversations/<channel_id>.sqlite3`` + sidecars) while the Phase-A
    dual-write period lasts — without this, "forgotten" conversations remain
    fully readable in the legacy store.

    The channel's offload evidence -- the archived execute responses its
    turns produced -- lives in the same database and is erased by
    ``forget_channel`` in the same transaction as its turn records, experiment
    runs included; there is no second evidence file to sweep.
    """
    deleted = ObservabilityStore(db_path).forget_channel(channel_id)
    if workflow_path and channel_id == os.path.basename(channel_id):
        legacy_db = os.path.join(
            state_paths.conversations_dir(workflow_path), f"{channel_id}.sqlite3"
        )
        removed = 0
        for path in (legacy_db, f"{legacy_db}-wal", f"{legacy_db}-shm"):
            try:
                os.remove(path)
                removed += 1
            except FileNotFoundError:
                pass
        deleted["legacy_conversation_db_files"] = removed
    return deleted


def run_clear_conversations(db_path: str, workflow_path: str = "") -> dict[str, int]:
    """Erase all conversation/turn observability for one workflow.

    Including every offload evidence row -- the archived execute responses
    of the conversations being cleared, and of experiment runs, whose
    experiment records are cleared too. ``clear_conversations`` deletes them
    in the same transaction as the turn records.
    """
    deleted = ObservabilityStore(db_path).clear_conversations()
    if workflow_path:
        legacy_dir = state_paths.conversations_dir(workflow_path)
        removed = 0
        try:
            names = os.listdir(legacy_dir)
        except FileNotFoundError:
            names = []
        for name in names:
            if not name.endswith((".sqlite3", ".sqlite3-wal", ".sqlite3-shm")):
                continue
            path = os.path.join(legacy_dir, name)
            if not os.path.isfile(path):
                continue
            try:
                os.remove(path)
                removed += 1
            except FileNotFoundError:
                pass
        deleted["legacy_conversation_db_files"] = removed
    return deleted


PREFERRED_SPAWN_PORT = 8000


def spawn_options_from_cli_args(args) -> dict:
    """Map ``run_chatbot`` CLI flags to ChatbotServer spawn_options.

    Passing ``--server-port`` means an existing FastAPI server: do not spawn.
    Omitting it auto-spawns a loopback server (preferred port
    ``PREFERRED_SPAWN_PORT``; a busy port still moves at activate time).
    """
    external_port = getattr(args, "server_port", None)
    return {
        "no_server": external_port is not None,
        "server_port": (
            int(external_port) if external_port is not None else PREFERRED_SPAWN_PORT
        ),
        "expect_encrypted_jwt": bool(getattr(args, "expect_encrypted_jwt", False)),
    }


def run_chatbot_main(args) -> int:
    """Entry point for the `fastworkflow run_chatbot` subcommand.

    UX contract:
    The chatbot opens with a workflow picker (bundled examples + a directory
    browser). Selecting one discovers its env files or asks the developer to
    install them, then starts the FastAPI server unless ``--server-port``
    named an existing server.
    """
    workspace_manifest_path = getattr(args, "workspace_manifest", None)
    spawn_options = spawn_options_from_cli_args(args)
    if workspace_manifest_path:
        # Workspace inspection never starts or connects to a live workflow server.
        spawn_options = {"no_server": True}
    try:
        server = ChatbotServer(
            port=0,
            spawn_options=spawn_options,
            workspace_manifest_path=workspace_manifest_path,
        )
    except (OSError, WorkspaceError, ValueError) as exc:
        print(f"Error: cannot start the chatbot ({exc}).")
        return 1

    # -- banner ---------------------------------------------------------
    print("fastWorkflow Chatbot")
    if server.workspace is not None:
        print(
            "  read-only workspace: "
            + server.workspace.label
            + " ("
            + str(server.workspace.manifest_path)
            + ")"
        )
    else:
        print("  pick a workflow in the browser (bundled examples")
        print("  and local folders are listed; you can browse anywhere).")
    print(f"\n  Open in your browser:\n\n    {server.url}\n")
    print("Press Ctrl+C to stop.", flush=True)
    _open_in_browser(server.url)

    # A service-manager SIGTERM must run the same cleanup as Ctrl+C — without
    # this, `kill <chatbot-pid>` orphans the spawned FastAPI server (which may
    # be running with unsigned JWTs and loaded API keys).
    def _raise_system_exit(_signum, _frame):
        raise SystemExit(0)

    try:
        signal.signal(signal.SIGTERM, _raise_system_exit)
    except (ValueError, OSError):
        pass  # not the main thread / unsupported platform: keep going

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server.server_proc is not None:
            print("Stopping the spawned FastAPI server...", flush=True)
        server.shutdown()  # also terminates the spawned server
    return 0
