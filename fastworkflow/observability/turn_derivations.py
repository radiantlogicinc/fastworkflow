"""Pure per-turn derivations over recorded spans.

The execution ledger, the decision-signal chips and their low-confidence test,
the cut-at-limit test for an `fw.llm.call`, and the per-call cost helpers.
Moved verbatim from ``run_chatbot.turn_annotations`` (fix-cnoc), which
re-exports every name, so that ``observability.comparison`` and
``observability.diagnosis`` import them at module level instead of reaching
into the debug-UI package through an import cycle.

Stdlib only: nothing here imports fastworkflow, which is what keeps this module
below both of those. `cost_rollup` is not here; it lives in
``observability.comparison`` beside the `usage_rollup` it delegates to.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any, Optional


SPAN_LLM_CALL = "fw.llm.call"


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
    `fw.ask_user` spans.
    """
    margins: list[float] = []
    intent_decisions = 0
    decisions_without_margin = 0
    asked = 0
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
            for uncertainty_signal in signals if isinstance(signals, list) else []:
                if (not isinstance(uncertainty_signal, dict)
                        or uncertainty_signal.get("kind") != SIGNAL_TOPK_MARGIN):
                    continue
                value = _finite_number(uncertainty_signal.get("value"))
                if value is not None:
                    margins.append(value)
                    found = True
            if not found:
                decisions_without_margin += 1
        elif name == SPAN_ASK_USER:
            asked += 1
    return {
        "intent_margin_min": min(margins) if margins else None,
        "intent_margin_decisions": len(margins),
        "intent_decisions": intent_decisions,
        "intent_decisions_without_margin": decisions_without_margin,
        "asked_user": asked,
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
