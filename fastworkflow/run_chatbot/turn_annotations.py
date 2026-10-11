"""Turn, attempt, and diagnosis derivations for the chatbot read layer.

Moved verbatim from ``run_chatbot.server``. No handler state.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any, Optional

from fastworkflow.observability.comparison import (
    ExecutionRef,
    cost_rollup,
    project_execution,
    usage_rollup,
)
from fastworkflow.observability.diagnosis import (
    InvalidTurnQuery,
    TurnQuery,
    diagnose_turn,
)
from fastworkflow.observability.store import ReadOnlyObservabilityStore
# Every name that moved to turn_derivations (fix-cnoc) is imported back, private
# ones included, so each old `turn_annotations.<name>` import keeps working.
from fastworkflow.observability.turn_derivations import (
    CONSEQUENCE_ORDER,
    LOW_CONFIDENCE_DEFAULT_MARGIN,
    SIGNAL_TOPK_MARGIN,
    SPAN_AGENT_TOOL_CALL,
    SPAN_ASK_USER,
    SPAN_COMMAND_EXECUTE,
    SPAN_LLM_CALL,
    SPAN_NLU_INTENT,
    _exact_int,
    _finite_number,
    _mapping_attr,
    _span_attributes,
    _text_or_none,
    count_llm_calls_cut_at_limit,
    execution_ledger,
    is_low_confidence,
    llm_call_cost,
    llm_call_cut_at_limit,
    merge_cost_rollups,
    turn_decision_signals,
)

# ----------------------------------------------------------------------
# Derived fields for the SPA (fix-49m.6)
# ----------------------------------------------------------------------
#
# Three things the debug UI shows are not columns: whether an `fw.llm.call`
# stopped at its output cap, how many such calls a turn or an attempt made,
# and the verdict an experiment's evidence segments add up to. They are derived
# in the read layer (the per-call cut-at-limit test in
# `observability/turn_derivations.py`, the rest here), from ObservabilityStore
# reads only [R12] -- never from a query of this module's own -- so a sealed
# experiment's archive renders them through the very same functions.
# (The fourth, the attempt's runtime snapshot, IS a column:
# `_decode_attempt_row` already exposes it.)

EVIDENCE_VALID = "valid"
EVIDENCE_INVALID = "invalid"
EVIDENCE_UNRECORDED = "unrecorded"


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

    Rows that already carry the stamps (the complete-dataset scan computes
    them while spans are in hand) are left alone so this never re-decodes the
    page's attribute JSON.
    """
    need = [turn for turn in turns if "llm_calls_cut_at_limit" not in turn]
    if not need:
        return
    spans_by_turn = store.spans_for_turns(turn["turn_key"] for turn in need)
    for turn in need:
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


# ----------------------------------------------------------------------
# Derived fields for the SPA, tier 2 (fix-aou)
# ----------------------------------------------------------------------
#
# Four more things the debug UI shows that are not columns, derived here from
# ObservabilityStore reads only [R12] so sealed archives render them through
# the same functions:
#
# (a) a turn's execution ledger -- every dispatch, joined on `command_call_id`
#     between the turn record's `execution_records` refs and the trace's
#     `fw.command.execute` spans (and the span-less inner hops those spans
#     file under `child_calls`);
# (b) the turn's decision signals -- the least confident intent resolution's
#     top-k margin and whether the user was asked -- and the low-confidence
#     filter they feed;
# (c) an experiment's provenance, flattened from the evidence-run records,
#     the experiment row and the attempts' runtime snapshots, plus the
#     field-by-field difference between two experiments' provenance;
# (d) cost roll-ups from the `cost` attribute on `fw.llm.call`.
#
# Every one of them says "not recorded" for an absence rather than inventing
# a value: a missing margin is no chip and not a low-confidence turn, and a
# missing cost is never a zero.
#
# The pure derivations behind (a), (b) and (d) live in
# `observability/turn_derivations.py` -- except `cost_rollup`, which lives in
# `observability/comparison.py` beside the `usage_rollup` it delegates to --
# and are re-exported from this module; `turn_span_stamps` below composes them.

PROVENANCE_NOT_RECORDED = "not recorded"

# How many `train_runs` rows one training-history request reads. The table has
# one row per published training run, so a workflow accumulates them slowly;
# the default is `ObservabilityStore.list_train_runs`' own and the ceiling is
# what stops a caller asking for every metrics blob in the store at once.
TRAINING_RUN_DEFAULT_LIMIT = 50
TRAINING_RUN_MAX_LIMIT = 200

# The experiment row's own provenance-bearing columns.
_EXPERIMENT_PROVENANCE_COLUMNS = (
    "workflow_name",
    "benchmark_id",
    "benchmark_version",
    "benchmark_digest_sha256",
)
# The attempt's stamped runtime snapshot (runtime_readiness) keys that pin
# what ran. Snapshots stored before feature modes were retired may also carry
# `effective_features`; the provenance reader flattens it if present.
_SNAPSHOT_PROVENANCE_KEYS = (
    "workflow_fingerprint",
    "workflow_model_version",
    "workflow_model_legacy_layout",
)


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
    already reads."""
    span_list = list(spans)
    turn["execution_ledger"] = execution_ledger(turn.get("record"), span_list)
    turn.update(turn_span_stamps(span_list))


def annotate_turn_diagnosis(
    turn: dict[str, Any],
    spans: Iterable[Mapping[str, Any]],
    *,
    store_id: Optional[str] = None,
    low_confidence_below: Optional[float] = None,
) -> None:
    """Stamp an opened turn with its diagnosis (`fix-9eg.18.2/.18.3`).

    Deliberately separate from `annotate_turn_detail` rather than folded into
    it: the ledger, chips and cost are what every reader of a turn gets, and
    widening that shape would change a payload this slice does not own.

    The steps come from the ledger `annotate_turn_detail` already projected --
    `project_execution` runs that same `execution_ledger` -- so the markers
    describe the dispatch sequence the trace view renders rather than a second
    one derived here.
    """
    span_list = [dict(span) for span in spans]
    turn_key = str(turn.get("logical_turn_key") or turn.get("turn_key") or "")
    ref = ExecutionRef(store_id=store_id or "", turn_keys=(turn_key,))
    projection = project_execution(
        ref,
        _SingleTurnReader(turn, span_list),
        ledger=execution_ledger,
        cost_rollup=cost_rollup,
    )
    turn["diagnosis"] = diagnose_turn(
        turn_key,
        turn,
        turn.get("record"),
        span_list,
        projection.steps,
        ref=ref,
        low_confidence_below=low_confidence_below,
    ).as_dict()


# Wire values are parsed EXACTLY or refused. The rail's older parsing fell back
# to a default when a value did not convert, so `?limit=abc` quietly served 100
# rows and `?attempt=1.5` quietly served every attempt -- a filter that silently
# means something else is worse than one that fails, because the operator reads
# the result as an answer about the dataset.
_EXACT_INT_RE = re.compile(r"^[+-]?[0-9]+$")
_EXACT_NUMBER_RE = re.compile(
    r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$"
)
# `nan` and `inf` match no pattern here on purpose: both are floats a caller can
# write and neither is a threshold. See `TurnQuery`, which refuses them again.
_WIRE_TRUE = frozenset({"1", "true"})
_WIRE_FALSE = frozenset({"0", "false"})


def _wire_int(value: Optional[str], name: str) -> Optional[int]:
    if value is None:
        return None
    text = value.strip()
    if not _EXACT_INT_RE.match(text):
        raise InvalidTurnQuery(f"{name} must be an integer, not {value!r}")
    return int(text)


def _wire_number(value: Optional[str], name: str) -> Optional[float]:
    if value is None:
        return None
    text = value.strip()
    if not _EXACT_NUMBER_RE.match(text):
        raise InvalidTurnQuery(f"{name} must be a finite number, not {value!r}")
    return float(text)


def _wire_bool(value: Optional[str], name: str) -> Optional[bool]:
    if value is None:
        return None
    text = value.strip().casefold()
    if text in _WIRE_TRUE:
        return True
    if text in _WIRE_FALSE:
        return False
    raise InvalidTurnQuery(f"{name} must be true or false, not {value!r}")


def _defaulted(value: Optional[int], fallback: int) -> int:
    return fallback if value is None else value


def _wire_markers(value: Optional[str], name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    # Unknown names are refused by TurnQuery, which owns the vocabulary; this
    # only rejects a request that named the parameter and then said nothing,
    # which is far likelier to be a client bug than an empty filter.
    if not names:
        raise InvalidTurnQuery(f"{name} was given with no marker names")
    return names


def turn_query_from_params(q: Any, *, default_limit: int = 100) -> TurnQuery:
    """Build one `TurnQuery` from the wire, or refuse with `InvalidTurnQuery`.

    One function for every route that searches turns -- the rail and an agent
    read -- so a filter means the same thing wherever it is asked. The
    store-level names are the rail's existing ones (`channel`, `conversation`,
    `command`, ...), kept so an existing link keeps working; the diagnostic ones
    are new.
    """
    return TurnQuery(
        channel_id=q("channel"),
        conversation_id=_wire_int(q("conversation"), "conversation"),
        status=q("status"),
        success=_wire_bool(q("success"), "success"),
        command_name=q("command"),
        context=q("context"),
        experiment_id=q("experiment"),
        task_id=q("task"),
        attempt=_wire_int(q("attempt"), "attempt"),
        markers_any=_wire_markers(q("markers_any"), "markers_any"),
        markers_all=_wire_markers(q("markers_all"), "markers_all"),
        low_confidence_below=_wire_number(
            q("low_confidence_below"), "low_confidence_below"
        ),
        text_contains=q("text") or None,
        # An explicit bound is honoured even when it is refusable: `limit=0` is
        # a mistake to be told about, not a value to be replaced by the
        # default, which is what makes `or default_limit` the wrong idiom here.
        limit=_defaulted(_wire_int(q("limit"), "limit"), default_limit),
        offset=_defaulted(_wire_int(q("offset"), "offset"), 0),
        scan_limit=_wire_int(q("scan_limit"), "scan_limit"),
        resume_after=q("resume_after") or None,
        include_record=_wire_bool(q("include_record"), "include_record"),
    )


# A store minted before store identities existed answers `store_identity()` with
# None. Its turns are still worth diagnosing, so the projection carries this in
# place of a name rather than inventing one that would collide with a real
# store; anchors built from it are refused by the feedback writer, which is the
# honest outcome for evidence nobody can address.
UNIDENTIFIED_STORE_ID = "unidentified-store"


def diagnostic_store_id(store: Any) -> str:
    """The id a diagnosis anchors against: the store's own recorded identity."""
    try:
        identity = store.store_identity()
    except AttributeError:
        identity = None
    return str(identity) if identity else UNIDENTIFIED_STORE_ID


class _SingleTurnReader:
    """The `ExecutionReader` shape over one turn the route already read.

    Exists so diagnosing an opened turn costs no further store reads: the route
    has the row and the trace in hand, and re-opening the store to fetch them
    again is the kind of duplicate read `project_execution` takes a reader to
    avoid.
    """

    def __init__(self, turn: Mapping[str, Any], spans: list[dict[str, Any]]) -> None:
        self._turn = turn
        self._spans = spans
        self._key = str(turn.get("logical_turn_key") or turn.get("turn_key") or "")

    def turn(self, store_id: str, turn_key: str) -> Optional[Mapping[str, Any]]:
        return self._turn if turn_key == self._key else None

    def trace(self, store_id: str, turn_key: str) -> list[dict[str, Any]]:
        return self._spans if turn_key == self._key else []
