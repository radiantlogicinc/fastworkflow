"""Diagnostic projection and complete-dataset discovery over recorded evidence.

The backend half of `fix-9eg.18.1/.18.2/.18.3`. Two things live here, and they
are here together because they are the same claim read at two zoom levels:

- **Discovery** (`search_turns`, `.18.1`): filtering, text search, counting and
  paging that apply to the WHOLE authorized dataset rather than to whichever
  page a browser happened to load first. `run_chatbot/server.py` today reads one
  `list_turns` page and then drops rows from it that fail the low-confidence
  test, so a turn matching on page 4 is invisible and the answer reads as "no
  matches" rather than "none on this page". The scan below walks the store's own
  filtered set in bounded chunks, applies the diagnostic predicates to every row
  it walks, and returns a bounded page whose `total_matched` is the count over
  the whole scan -- so the number beside the list and the list agree.

- **Diagnosis** (`turn_markers`, `diagnose_turn`, `diagnose_execution`, `.18.2`
  and `.18.3`): the markers a human highlights and a coding agent filters on --
  context navigation, unsuccessful executor steps, intent ambiguity/error,
  parameter-extraction invalidity/retries, awaiting-user suspensions, repeated
  dispatches and suspected loops -- each carrying the exact span that evidences
  it and the executor step it nests under.

**Everything here is read-only and quotes recorded evidence.** Four rules,
each of which the real corpus shows is not academic:

1. **`resolved=false` on an intent decision is not a failure.** In the pilot
   store 1714 of 8558 `fw.nlu.intent` spans record `resolved=false` with
   `ambiguous=false`: that is the CME wildcard command walking the parent
   context chain, which is how routing is SUPPOSED to work. Flagging it would
   mark almost every turn as broken. Only a recorded `ambiguous=true` or a span
   whose own status is `error` marks an intent problem; the unresolved attempts
   are still reported, as a neutral count, because they are what a reader
   follows to understand the walk.

2. **Repetition is not failure.** `repeated_command` is a neutral observation
   with its occurrences attached. `suspected_loop` is only added when repetition
   coincides with SEPARATELY RECORDED trouble (an unsuccessful or errored
   dispatch, an ambiguity, an invalid extraction, a retry round) in the same
   turn. The heuristic's bounds travel in the output (`LoopPolicy.as_dict`), so
   a reader never has to guess what "suspected" was measured against.

3. **Absent is not false.** `success` is None on the 85 execute spans in the
   pilot store that ended by exception (status `error`, no `success` attribute),
   and `retry_round` is absent on 1701 of 5648 extraction spans. Those stay
   unknown and are counted in `coverage`, never defaulted. A marker's absence
   means "not evidenced", which is why `partial_evidence` exists as its own
   marker: a turn whose evidence is incomplete is findable rather than quietly
   clean-looking.

4. **Nothing is inferred that a producer did not record.** No cost is imputed,
   no event is synthesized, no span is invented for a dispatch that has none.
   Where the evidence cannot answer, the field says so and `coverage` counts it.

**Not a second ledger.** The dispatch sequence comes from
`observability/turn_derivations.py`'s `execution_ledger`, through
`comparison.default_ledger_projection()` and `comparison.project_execution`, so
a step this module diagnoses is the same step the comparison view aligns and the
same row the debug UI lists. Anchors are `comparison.EvidenceAnchor`, so a
comment recorded against a diagnosed step is recorded against the same evidence
the comparison slice would anchor it to.

**Scope is the caller's, never resolved here.** Every entry point takes an
already-opened store and the `store_id` it was registered under. Nothing in this module opens a path, searches a second store,
or widens an experiment/task/attempt scope it was handed.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterable, Mapping, Optional, Protocol, Sequence

from fastworkflow import tracing
from fastworkflow.observability.comparison import (
    ExecutionProjection,
    ExecutionReader,
    ExecutionRef,
    ExecutionStep,
    LedgerProjection,
    # The ancestry walk and the "attributes may be a dict or a JSON string"
    # normalization already exist there and are exercised by the comparison
    # tests; a second copy would be a second thing to keep true.
    _SpanTree,
    anchor_for_step,
    default_ledger_projection,
    project_execution,
)
from fastworkflow.observability.derived_cache import (
    DerivedTurnCache,
    DerivedTurnEntry,
    estimate_entry_bytes,
)
from fastworkflow.observability.turn_derivations import (
    is_low_confidence,
    turn_decision_signals,
)

# ----------------------------------------------------------------------
# Vocabulary
# ----------------------------------------------------------------------

# Markers. One string per distinct claim, because a UI chip, an API filter and a
# test assertion must all name the same thing.
MARKER_CONTEXT_NAVIGATION = "context_navigation"
MARKER_STEP_UNSUCCESSFUL = "step_unsuccessful"
MARKER_STEP_ERROR = "step_error"
MARKER_INTENT_AMBIGUOUS = "intent_ambiguous"
MARKER_INTENT_ERROR = "intent_error"
MARKER_PARAM_INVALID = "parameter_extraction_invalid"
MARKER_PARAM_ERROR = "parameter_extraction_error"
MARKER_PARAM_RETRY = "parameter_extraction_retry"
MARKER_AWAITING_USER = "awaiting_user"
MARKER_REPEATED_COMMAND = "repeated_command"
MARKER_SUSPECTED_LOOP = "suspected_loop"
MARKER_LOW_CONFIDENCE = "low_confidence"
MARKER_PARTIAL_EVIDENCE = "partial_evidence"

# Deterministic order, so two callers listing a turn's markers produce the same
# sequence and a facet block is stable between refreshes.
MARKER_ORDER: tuple[str, ...] = (
    MARKER_CONTEXT_NAVIGATION,
    MARKER_STEP_UNSUCCESSFUL,
    MARKER_STEP_ERROR,
    MARKER_INTENT_AMBIGUOUS,
    MARKER_INTENT_ERROR,
    MARKER_PARAM_INVALID,
    MARKER_PARAM_ERROR,
    MARKER_PARAM_RETRY,
    MARKER_AWAITING_USER,
    MARKER_REPEATED_COMMAND,
    MARKER_SUSPECTED_LOOP,
    MARKER_LOW_CONFIDENCE,
    MARKER_PARTIAL_EVIDENCE,
)

# Navigation states. `unknown` is a first-class answer: an execute span that
# recorded no context type (the exception path does not) cannot say whether the
# context moved, and neither can two equal types -- a context type names a
# class, not an instance -- so saying "unchanged" there would be an invention.
NAV_CHANGED = "changed"
NAV_UNKNOWN = "unknown"

# Phase-event kinds: the nested decisions a dispatch made.
PHASE_INTENT = "intent_detection"
PHASE_PARAM = "parameter_extraction"
PHASE_ASK_USER = "ask_user"

# Marker basis. `spans` is what a complete scan can afford on every row; `record`
# adds the facts only the turn record carries (a dispatch with no span at all,
# and per-`CommandOutput` success). Declared on every projection so a reader
# never has to guess which one produced a marker set -- and so the list view and
# the detail view can be compared honestly rather than assumed identical.
BASIS_SPANS = "spans"
BASIS_SPANS_AND_RECORD = "spans+record"

# Span names this build declares a contract for. A span outside it is counted in
# `coverage.unrecognized_span_names` rather than ignored: the pilot store holds
# `fw.context.checkpoint`, `fw.evidence.read` and `fw.context.checkpoint_probe`
# spans that no emitter in this tree writes, and a reader deserves to know its
# evidence contains kinds this build cannot interpret.
KNOWN_SPAN_NAMES: frozenset[str] = frozenset(tracing.SPAN_CONTRACTS)

# Span statuses this build's emitters write. The pilot store also holds
# `awaiting_user`, `completed`, `failed` and `open`, written by other producers
# or left open by a process that died. `open` is why a missing end is a coverage
# fact and not a zero duration.
_TERMINAL_ERROR_STATUSES: frozenset[str] = frozenset({tracing.STATUS_ERROR, "failed"})
_OPEN_STATUSES: frozenset[str] = frozenset({"open"})

_DISPATCH_SPAN_NAMES: frozenset[str] = frozenset(
    {tracing.SPAN_COMMAND_EXECUTE, tracing.SPAN_AGENT_TOOL_CALL}
)

# How many supporting spans a marker carries inline before it reports a count
# instead. A list row must stay small enough to page 200 of them; the detail
# projection carries every event anyway, so nothing is lost.
MAX_MARKER_ANCHORS = 8

# One `list_turns` round trip. The scan itself is NOT capped by default: a cap
# on the scan is a cap on what is discoverable, and a row beyond it is invisible
# no matter how the caller pages, which is the defect this module exists to fix
# rather than a safety valve. Memory stays bounded by the PAGE, not the dataset
# -- matched rows outside the page are counted and dropped.
#
# A caller that needs a bounded per-request cost sets `TurnQuery.scan_limit` and
# follows `TurnSearchPage.next_scan_cursor`, which resumes exactly where the
# previous segment stopped and does eventually reach the end of the store.
SCAN_CHUNK = 500

# Span attribute JSON is the memory hog of a complete scan (hundreds of MB for
# a large store when 500 turns' spans are materialised together). `list_turns`
# still walks in SCAN_CHUNK keyset pages; spans are fetched in much smaller
# sub-batches so peak resident span JSON stays bounded (~tens of MB).
SPAN_FETCH_BATCH = 8

# A page larger than this is refused rather than silently served: a caller
# asking for a million rows in one response has made a mistake, and answering it
# would defeat the bounded-page property the rest of this module maintains.
MAX_PAGE_LIMIT = 1_000


class DiagnosisError(RuntimeError):
    """Base for refusals this module raises rather than guessing."""


class InvalidTurnQuery(DiagnosisError, ValueError):
    """A query this module will not answer, stated rather than coerced."""


# ----------------------------------------------------------------------
# Small local helpers
# ----------------------------------------------------------------------


def _text_or_none(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def _exact_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _exact_int_or_none(value: Any) -> Optional[int]:
    """A recorded non-negative integer, or None.

    `bool` is excluded because `True` is not round 1, and a float is excluded
    because attempt 2.0 is a value some other producer wrote and this module
    will not decide what it meant.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def _bool_or_none(value: Any) -> Optional[bool]:
    """A recorded boolean, or None. An int column (`turns.success`) counts; a
    string does not -- `"false"` is a value some other producer wrote and this
    module will not decide what it meant."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    return None


def _non_empty_list(value: Any) -> Optional[list[Any]]:
    return value if isinstance(value, list) and value else None


def _digest(value: Any) -> Optional[str]:
    """A stable digest of a recorded parameter mapping, or None.

    Only ever a digest: parameters carry entity content, and a repeat signature
    must be comparable without the content travelling with it.
    """
    if value is None:
        return None
    try:
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# The markers that count as recorded trouble for the loop heuristic. An
# awaiting-user suspension is deliberately not among them (see `turn_markers`),
# and neither is `context_navigation`: moving context is what navigation
# commands are for.
TROUBLE_MARKERS: frozenset[str] = frozenset(
    {
        MARKER_STEP_UNSUCCESSFUL,
        MARKER_STEP_ERROR,
        MARKER_INTENT_AMBIGUOUS,
        MARKER_INTENT_ERROR,
        MARKER_PARAM_INVALID,
        MARKER_PARAM_ERROR,
        MARKER_PARAM_RETRY,
    }
)


def marker_is_trouble(markers: Iterable[str]) -> bool:
    return any(marker in TROUBLE_MARKERS for marker in markers)


def _capped(span_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "span_ids": list(span_ids[:MAX_MARKER_ANCHORS]),
        "count": len(span_ids),
        "truncated": len(span_ids) > MAX_MARKER_ANCHORS,
    }


DecisionSignalProjection = Callable[[Iterable[Mapping[str, Any]]], Mapping[str, Any]]
LowConfidenceTest = Callable[[Mapping[str, Any], float], bool]


def default_decision_signals() -> DecisionSignalProjection:
    """`observability/turn_derivations.py`'s `turn_decision_signals` -- the one implementation.

    Imported from that stdlib-only module for the reason
    `comparison.default_ledger_projection` gives: this module must stay usable
    without the HTTP layer, and the dependency runs one way at import time. The
    low-confidence filter has to mean the same thing in a complete scan as it
    does on the page the browser already had, and the only way to guarantee that
    is to run the same function.
    """
    # No cycle: turn_derivations imports nothing from fastworkflow, and
    # turn_annotations (which imports TurnQuery, diagnose_turn, and
    # InvalidTurnQuery from here) re-exports it.
    return turn_decision_signals


def default_low_confidence_test() -> LowConfidenceTest:
    """`observability/turn_derivations.py`'s `is_low_confidence`: below the threshold on a
    RECORDED margin only. A turn whose decisions recorded no margin is not low
    confidence, it is unmeasured -- the distinction `.18.1`'s acceptance names."""
    # No cycle: turn_derivations imports nothing from fastworkflow, and
    # turn_annotations (which imports TurnQuery, diagnose_turn, and
    # InvalidTurnQuery from here) re-exports it.
    return is_low_confidence


# ----------------------------------------------------------------------
# The loop heuristic, stated as data
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class LoopPolicy:
    """Bounds for the repeated-dispatch heuristic, carried in every output.

    `.18.3` requires the heuristic be bounded AND documented in the output, so
    this travels with the result rather than living in a docstring a reader of
    the JSON never sees.

    - `min_repeats`: how many occurrences of one signature make a repeat group.
    - `window_steps`: occurrences must fall within this many consecutive
      dispatches. Two identical commands at the start and end of a long turn are
      a user doing the same thing twice, not a loop.
    - `require_recorded_trouble`: a repeat group is only ever `suspected_loop`
      when the turn ALSO recorded a failure, error, ambiguity, invalid
      extraction or retry round. Off, the marker becomes "repeated at all",
      which is the false positive `.18.3` names.
    - `max_groups`: a bound on reported groups, so a pathological turn cannot
      produce an unbounded payload. Truncation is reported, not hidden.
    """

    name: str = "repeated-dispatch-window/1"
    min_repeats: int = 3
    window_steps: int = 6
    require_recorded_trouble: bool = True
    max_groups: int = 20

    def __post_init__(self) -> None:
        if self.min_repeats < 2:
            raise InvalidTurnQuery("min_repeats must be at least 2")
        if self.window_steps < self.min_repeats:
            raise InvalidTurnQuery("window_steps must be at least min_repeats")
        if self.max_groups < 1:
            raise InvalidTurnQuery("max_groups must be at least 1")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "min_repeats": self.min_repeats,
            "window_steps": self.window_steps,
            "require_recorded_trouble": self.require_recorded_trouble,
            "max_groups": self.max_groups,
        }


DEFAULT_LOOP_POLICY = LoopPolicy()


# ----------------------------------------------------------------------
# Records
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseEvent:
    """One recorded nested decision, and the executor step it belongs to.

    `owner_span_id` is the nearest `fw.command.execute` / `fw.agent.tool_call`
    ANCESTOR, found by walking recorded parent links -- not guessed from timing.
    In the pilot store every `fw.nlu.intent` and `fw.nlu.param_extraction` span
    has such an ancestor and every `fw.ask_user` span has none (the agent loop
    asks, not a dispatch), so `owner_span_id is None` is a real and ordinary
    state meaning "recorded outside any dispatch", not a lookup failure.

    `detail` holds only keys the span actually recorded. An absent key is absent
    here too.
    """

    kind: str
    span_id: str
    parent_span_id: Optional[str]
    owner_span_id: Optional[str]
    owner_call_id: Optional[str]
    start_ns: Optional[int]
    status: Optional[str]
    markers: tuple[str, ...]
    detail: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "owner_span_id": self.owner_span_id,
            "owner_call_id": self.owner_call_id,
            "start_ns": self.start_ns,
            "status": self.status,
            "markers": list(self.markers),
            "detail": dict(self.detail),
        }


@dataclass(frozen=True)
class RepeatOccurrence:
    span_id: Optional[str]
    command_call_id: Optional[str]
    start_ns: Optional[int]
    index: int
    status: Optional[str]
    success: Optional[bool]

    def as_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "command_call_id": self.command_call_id,
            "start_ns": self.start_ns,
            "index": self.index,
            "status": self.status,
            "success": self.success,
        }


@dataclass(frozen=True)
class RepeatGroup:
    """One command dispatched repeatedly within the policy's window.

    A neutral observation on its own. `suspected_loop` is the heuristic's
    verdict and carries `trouble` -- the separately recorded evidence that
    raised it -- so a reader can check the reasoning rather than trust it.
    `basis` says what the signature was built from, because two dispatches
    matching on command and context only is a weaker statement than two matching
    on parameters as well.
    """

    signature_id: str
    command_name: Optional[str]
    context: Optional[str]
    parameters_digest: Optional[str]
    basis: str
    occurrences: tuple[RepeatOccurrence, ...]
    span_ids: tuple[str, ...]
    suspected_loop: bool
    trouble: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.occurrences)

    def as_dict(self) -> dict[str, Any]:
        return {
            "signature_id": self.signature_id,
            "command_name": self.command_name,
            "context": self.context,
            "parameters_digest": self.parameters_digest,
            "basis": self.basis,
            "count": self.count,
            "occurrences": [occ.as_dict() for occ in self.occurrences],
            "span_ids": list(self.span_ids),
            "suspected_loop": self.suspected_loop,
            "trouble": list(self.trouble),
        }


@dataclass(frozen=True)
class TurnMarkers:
    """The compact per-turn summary a list row carries and a filter tests.

    Cheap enough to compute for every turn in a complete scan (spans only, one
    chunked query per page of turn keys), and the SAME object the detail
    projection reports, so a turn found by the list is described identically
    when it is opened.
    """

    turn_key: str
    basis: str
    markers: tuple[str, ...]
    counts: Mapping[str, int]
    evidence: Mapping[str, Any]
    navigation: Mapping[str, Any]
    coverage: Mapping[str, Any]
    repeats: tuple[RepeatGroup, ...]
    decision_signals: Mapping[str, Any]
    loop_policy: Mapping[str, Any]

    def has(self, marker: str) -> bool:
        return marker in self.markers

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn_key": self.turn_key,
            "basis": self.basis,
            "markers": list(self.markers),
            "counts": dict(self.counts),
            "evidence": dict(self.evidence),
            "navigation": dict(self.navigation),
            "coverage": dict(self.coverage),
            "repeats": [group.as_dict() for group in self.repeats],
            "decision_signals": dict(self.decision_signals),
            "loop_policy": dict(self.loop_policy),
        }


@dataclass(frozen=True)
class StepDiagnosis:
    """One executor step with its markers and the evidence under it.

    The step itself is `comparison.ExecutionStep`, unchanged -- this adds the
    diagnosis beside it rather than re-deriving the dispatch. `anchor` is the
    `EvidenceAnchor` a comment attaches to, so highlighting a step and
    commenting on it address the same span.
    """

    step: ExecutionStep
    markers: tuple[str, ...]
    navigation: Mapping[str, Any]
    events: tuple[PhaseEvent, ...]
    repeat_signature_id: Optional[str]
    anchor: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.step.as_dict(),
            "markers": list(self.markers),
            "navigation": dict(self.navigation),
            "events": [event.as_dict() for event in self.events],
            "repeat_signature_id": self.repeat_signature_id,
            "anchor": dict(self.anchor),
        }


@dataclass(frozen=True)
class TurnDiagnosis:
    """A turn's full diagnostic projection: summary, steps, nested events.

    `markers` is the record-aware superset; `span_markers` is exactly what the
    list scan computed for the same turn. They are reported side by side
    on purpose -- `markers_only_in_record` is the honest name for "the list
    filter cannot find this one", and `coverage.steps_without_span` says why.
    """

    turn_key: str
    basis: str
    markers: tuple[str, ...]
    span_markers: TurnMarkers
    steps: tuple[StepDiagnosis, ...]
    events: tuple[PhaseEvent, ...]
    repeats: tuple[RepeatGroup, ...]
    counts: Mapping[str, int]
    coverage: Mapping[str, Any]
    turn_anchor: Mapping[str, Any]
    loop_policy: Mapping[str, Any]

    @property
    def markers_only_in_record(self) -> tuple[str, ...]:
        return tuple(m for m in self.markers if m not in self.span_markers.markers)

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn_key": self.turn_key,
            "basis": self.basis,
            "markers": list(self.markers),
            "span_markers": self.span_markers.as_dict(),
            "markers_only_in_record": list(self.markers_only_in_record),
            "steps": [step.as_dict() for step in self.steps],
            "events": [event.as_dict() for event in self.events],
            "repeats": [group.as_dict() for group in self.repeats],
            "counts": dict(self.counts),
            "coverage": dict(self.coverage),
            "turn_anchor": dict(self.turn_anchor),
            "loop_policy": dict(self.loop_policy),
        }


@dataclass(frozen=True)
class ExecutionDiagnosis:
    """Every turn of one scoped execution, diagnosed, plus the roll-up.

    Wraps a `comparison.ExecutionProjection` rather than replacing it: `steps`,
    `artifacts`, `cost` and `unavailable` stay where they were, and this adds
    the diagnostic layer. A turn the projection could not read is still listed
    in `projection.unavailable`, and its absence is a coverage fact here.
    """

    ref: ExecutionRef
    projection: ExecutionProjection
    turns: tuple[TurnDiagnosis, ...]
    markers: tuple[str, ...]
    counts: Mapping[str, int]
    coverage: Mapping[str, Any]
    loop_policy: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref.as_dict(),
            "markers": list(self.markers),
            "counts": dict(self.counts),
            "coverage": dict(self.coverage),
            "turns": [turn.as_dict() for turn in self.turns],
            "unavailable": list(self.projection.unavailable),
            "loop_policy": dict(self.loop_policy),
        }


# ----------------------------------------------------------------------
# Span-level diagnosis
# ----------------------------------------------------------------------


def _dispatch_navigation(attributes: Mapping[str, Any]) -> dict[str, Any]:
    """Where a dispatch started and ended, from its recorded context types.

    **A matching context type does not prove the context did not move.** The
    recorded value is the active context's class name, so `Project` and
    `Project` are equally consistent with "the same project throughout" and
    "moved from one project to another"; that reports `unknown` with
    `basis: same_context_type` rather than claiming `unchanged`. A DIFFERENT
    type is provable, and is the primary evidence for navigation.

    `recorded_flags` are navigation keys a producer stamped explicitly
    (`auto_navigated` and friends appear on two spans in the pilot corpus; no
    emitter in this tree writes them today). A recorded `auto_navigated=true` is
    a producer STATING that it navigated, so it establishes navigation on its
    own and `basis` says so -- evidence from a producer this build no longer
    contains is still evidence, and discarding it would lose history. A flag
    recorded false is not taken as proof of the negative.

    `basis` names which evidence decided, so a reader can weigh a `changed` that
    came from a type difference against one that came from a producer's flag.
    """
    before = _text_or_none(attributes.get(tracing.ATTR_CONTEXT_BEFORE))
    after = _text_or_none(attributes.get(tracing.ATTR_CONTEXT_AFTER))
    flags = {
        key: attributes[key]
        for key in (
            "auto_navigated",
            "auto_navigation_rule",
            "auto_navigation_step",
            "entered_context",
        )
        if key in attributes
    }
    if before is not None and after is not None and before != after:
        state, basis = NAV_CHANGED, "context_type_change"
    elif flags.get("auto_navigated") is True:
        state, basis = NAV_CHANGED, "recorded_flag"
    elif before is None or after is None:
        state, basis = NAV_UNKNOWN, "no_context_recorded"
    else:
        state, basis = NAV_UNKNOWN, "same_context_type"
    return {
        "state": state,
        "basis": basis,
        "from": before,
        "to": after,
        "recorded_flags": flags,
    }


def _dispatch_markers(
    status: Optional[str], attributes: Mapping[str, Any]
) -> tuple[list[str], Optional[bool]]:
    """Markers for one dispatch span, and the success it recorded.

    Status and `success` are two different recorded facts and get two different
    markers. In the pilot store 178 execute spans are `error` with
    `success=false` (both), and 85 are `error` with no `success` at all (status
    only) -- collapsing them would lose the distinction between "the command
    reported failure" and "the dispatch raised".
    """
    markers: list[str] = []
    success = _bool_or_none(attributes.get("success"))
    if success is False:
        markers.append(MARKER_STEP_UNSUCCESSFUL)
    if status in _TERMINAL_ERROR_STATUSES:
        markers.append(MARKER_STEP_ERROR)
    return markers, success


def _intent_markers(
    status: Optional[str], attributes: Mapping[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    """Markers and reported detail for one `fw.nlu.intent` span.

    `resolved=false` deliberately produces NO marker (see rule 1 in the module
    docstring); it is reported in `detail` so the parent-chain walk stays
    readable. `ambiguous` is the recorded flag for "the runtime could not pick",
    and is the only attribute-driven intent marker.
    """
    markers: list[str] = []
    ambiguous = _bool_or_none(attributes.get("ambiguous"))
    resolved = _bool_or_none(attributes.get("resolved"))
    if ambiguous is True:
        markers.append(MARKER_INTENT_AMBIGUOUS)
    if status in _TERMINAL_ERROR_STATUSES:
        markers.append(MARKER_INTENT_ERROR)
    detail = {
        key: attributes[key]
        for key in (
            "stage",
            "context",
            "matcher_layer",
            "escalation_outcome",
            "candidate_count",
            "command_name",
            "is_cme_command",
        )
        if key in attributes
    }
    detail["ambiguous"] = ambiguous
    detail["resolved"] = resolved
    return markers, detail


def _param_markers(
    status: Optional[str], attributes: Mapping[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    """Markers and reported detail for one `fw.nlu.param_extraction` span.

    `parameters_valid=false` is the recorded verdict; `missing_fields` /
    `invalid_fields` are its supporting detail and also stand alone, because a
    producer that recorded the fields without the verdict has still evidenced an
    invalid extraction.

    `retry_round` is a BOOLEAN (`bool(stored_params)` in
    `parameter_extraction.py`): it says "this extraction resumed from stored
    parameters", not which round it was. Since the contract's v2 (`fix-8ko2`) a
    producer also records `retry_round_ordinal`, the 0-based attempt number,
    and that is reported when it is present.

    It is never reconstructed when it is absent, and absent is a real answer
    rather than an old-producer artefact: a v2 producer deliberately omits the
    ordinal when it cannot determine the round, which happens whenever a
    session was restored from a continuation that persisted the stored
    parameters without their round (`fix-7gp9`). Counting extraction spans
    instead would look like an ordinal and be wrong exactly when it matters --
    a dropped span makes the count smaller than the round it purports to name.
    So an unrecorded round stays None and `coverage` reports how many did, via
    `extractions_without_retry_round`.
    """
    markers: list[str] = []
    valid = _bool_or_none(attributes.get("parameters_valid"))
    missing = _non_empty_list(attributes.get("missing_fields"))
    invalid = _non_empty_list(attributes.get("invalid_fields"))
    if valid is False or missing or invalid:
        markers.append(MARKER_PARAM_INVALID)
    if status in _TERMINAL_ERROR_STATUSES:
        markers.append(MARKER_PARAM_ERROR)
    retry = _bool_or_none(attributes.get("retry_round"))
    ordinal = _exact_int_or_none(attributes.get("retry_round_ordinal"))
    if retry is True or (ordinal is not None and ordinal > 0):
        markers.append(MARKER_PARAM_RETRY)
    detail: dict[str, Any] = {
        "parameters_valid": valid,
        "retry_round": retry,
        "retry_round_ordinal": ordinal,
        "missing_field_count": len(missing) if missing else (0 if "missing_fields" in attributes else None),
        "invalid_field_count": len(invalid) if invalid else (0 if "invalid_fields" in attributes else None),
    }
    for key in ("command_name", "extraction_method"):
        if key in attributes:
            detail[key] = attributes[key]
    return markers, detail


def _phase_events(tree: _SpanTree, spans: Sequence[Mapping[str, Any]]) -> list[PhaseEvent]:
    """Every nested decision, attributed to its executor step by ancestry."""
    events: list[PhaseEvent] = []
    for span in spans:
        name = span.get("name")
        if name == tracing.SPAN_NLU_INTENT:
            kind = PHASE_INTENT
        elif name == tracing.SPAN_NLU_PARAM_EXTRACTION:
            kind = PHASE_PARAM
        elif name == tracing.SPAN_ASK_USER:
            kind = PHASE_ASK_USER
        else:
            continue
        span_id = _text_or_none(span.get("span_id"))
        if span_id is None:
            continue
        attributes = tree.attributes(span)
        status = _text_or_none(span.get("status"))
        if kind == PHASE_INTENT:
            markers, detail = _intent_markers(status, attributes)
        elif kind == PHASE_PARAM:
            markers, detail = _param_markers(status, attributes)
        else:
            markers = [MARKER_AWAITING_USER]
            detail = {
                key: attributes[key]
                for key in ("attempt", "human_wait_ms")
                if key in attributes
            }
            # A recorded `user_response` means the suspension was answered. Its
            # absence on a still-open ask is a fact about the turn, not a
            # failure, so it is reported as a flag rather than a marker.
            detail["answered"] = "user_response" in attributes
        owner_span_id: Optional[str] = None
        owner_call_id: Optional[str] = None
        # ancestry() yields the span itself first; the dispatch is above it.
        for ancestor in tree.ancestry(_text_or_none(span.get("parent_span_id"))):
            if ancestor.get("name") in _DISPATCH_SPAN_NAMES:
                owner_span_id = _text_or_none(ancestor.get("span_id"))
                owner_call_id = _text_or_none(
                    tree.attributes(ancestor).get(tracing.ATTR_COMMAND_CALL_ID)
                )
                break
        events.append(
            PhaseEvent(
                kind=kind,
                span_id=span_id,
                parent_span_id=_text_or_none(span.get("parent_span_id")),
                owner_span_id=owner_span_id,
                owner_call_id=owner_call_id,
                start_ns=_exact_int(span.get("start_ns")),
                status=status,
                markers=tuple(markers),
                detail=detail,
            )
        )
    events.sort(key=lambda event: (event.start_ns is None, event.start_ns or 0, event.span_id))
    return events


def _repeat_groups(
    dispatches: Sequence[Mapping[str, Any]],
    *,
    trouble_by_span: Mapping[str, Sequence[str]],
    policy: LoopPolicy,
) -> list[RepeatGroup]:
    """Repeated dispatch signatures within the policy's window.

    `dispatches` are in execution order and each carries its signature parts.
    Occurrences of one signature count toward a group only when they fall within
    `window_steps` consecutive dispatches of each other, so the same command run
    once at the top of a turn and once at the bottom is not a repeat.

    `trouble_by_span` is CO-LOCATED evidence: the trouble recorded on a
    dispatch's own span or on a decision nested under it. A group is only
    `suspected_loop` when its OWN occurrences carry some. The looser rule --
    "the turn recorded trouble somewhere" -- was measured against the pilot
    store and called 67 of 152 turns a suspected loop, which is the
    false-positive `.18.3` names: a long agent turn almost always contains both
    a repeated tool call and an unrelated failure, and joining them is a
    coincidence, not evidence.
    """
    ordered = list(dispatches)
    by_signature: dict[tuple[Any, ...], list[int]] = {}
    for index, item in enumerate(ordered):
        signature = (item["command_name"], item["context"], item["parameters_digest"])
        by_signature.setdefault(signature, []).append(index)

    groups: list[RepeatGroup] = []
    for signature, indices in by_signature.items():
        if len(indices) < policy.min_repeats:
            continue
        # Longest run of occurrences whose span in dispatch order fits the window.
        best: list[int] = []
        start = 0
        for end in range(len(indices)):
            while indices[end] - indices[start] >= policy.window_steps:
                start += 1
            if end - start + 1 > len(best):
                best = indices[start : end + 1]
        if len(best) < policy.min_repeats:
            continue
        command_name, context, parameters_digest = signature
        if parameters_digest is not None:
            basis = "command+context+parameters"
        elif context is not None:
            basis = "command+context"
        elif command_name is not None:
            basis = "command"
        else:
            basis = "unknown"
        occurrences = tuple(
            RepeatOccurrence(
                span_id=ordered[i]["span_id"],
                command_call_id=ordered[i]["command_call_id"],
                start_ns=ordered[i]["start_ns"],
                index=i,
                status=ordered[i]["status"],
                success=ordered[i]["success"],
            )
            for i in best
        )
        span_ids = tuple(occ.span_id for occ in occurrences if occ.span_id)
        trouble = tuple(
            dict.fromkeys(
                reason
                for span_id in span_ids
                for reason in trouble_by_span.get(span_id, ())
            )
        )
        suspected = bool(trouble) or not policy.require_recorded_trouble
        groups.append(
            RepeatGroup(
                signature_id=_digest([command_name, context, parameters_digest]) or "unknown",
                command_name=command_name,
                context=context,
                parameters_digest=parameters_digest,
                basis=basis,
                occurrences=occurrences,
                span_ids=span_ids,
                suspected_loop=suspected,
                trouble=trouble,
            )
        )
    groups.sort(key=lambda group: (-group.count, group.signature_id))
    return groups[: policy.max_groups]


def turn_markers(
    turn_key: str,
    spans: Iterable[Mapping[str, Any]],
    *,
    turn_row: Optional[Mapping[str, Any]] = None,
    low_confidence_below: Optional[float] = None,
    loop_policy: LoopPolicy = DEFAULT_LOOP_POLICY,
    decision_signals: Optional[DecisionSignalProjection] = None,
    low_confidence_test: Optional[LowConfidenceTest] = None,
) -> TurnMarkers:
    """The span-basis diagnostic summary for one turn.

    This is what a complete scan can afford on every row and what a list filter
    tests, so it is deliberately computable from the trace alone: the turn row
    is optional and only contributes its recorded status/success, never a
    default for them.

    `low_confidence_below` is applied here rather than after paging, using
    `observability/turn_derivations.py`'s own `turn_decision_signals` and `is_low_confidence`
    -- the defect `.18.1` exists to fix is precisely that the server applied
    that test to an already-cut page.
    """
    # Decode each span's attributes ONCE into the local copy. `_SpanTree`
    # accepts an already-decoded mapping, so every later `attributes()` call is
    # a dict lookup instead of a second `json.loads` of the same column -- which
    # matters when a complete scan reads a whole store's traces.
    span_list: list[dict[str, Any]] = []
    decoder = _SpanTree(())
    for span in spans:
        copy = dict(span)
        copy["attributes"] = decoder.attributes(span)
        span_list.append(copy)
    tree = _SpanTree(span_list)

    markers: set[str] = set()
    evidence: dict[str, list[str]] = {}
    counts: dict[str, int] = {}
    # Trouble recorded ON a dispatch or on a decision nested under it, keyed by
    # the dispatch's span id. The loop heuristic reads only this.
    trouble_by_span: dict[str, list[str]] = {}

    def note(marker: str, span_id: Optional[str]) -> None:
        markers.add(marker)
        if span_id:
            evidence.setdefault(marker, []).append(span_id)

    dispatches: list[dict[str, Any]] = []
    navigation_changed: list[dict[str, Any]] = []
    nav_unknown = 0
    nav_same_type = 0
    unrecognized: set[str] = set()
    open_spans = 0
    dispatch_spans = 0
    intent_spans = 0
    param_spans = 0
    intent_unresolved = 0
    retry_unknown = 0
    round_unknown = 0
    success_unknown = 0

    for span in span_list:
        name = span.get("name")
        if isinstance(name, str) and name not in KNOWN_SPAN_NAMES:
            unrecognized.add(name)
        if _text_or_none(span.get("status")) in _OPEN_STATUSES or span.get("end_ns") is None:
            open_spans += 1
        if name not in _DISPATCH_SPAN_NAMES:
            continue
        dispatch_spans += 1
        span_id = _text_or_none(span.get("span_id"))
        attributes = tree.attributes(span)
        status = _text_or_none(span.get("status"))
        step_markers, success = _dispatch_markers(status, attributes)
        if success is None:
            success_unknown += 1
        for marker in step_markers:
            note(marker, span_id)
        if span_id and step_markers:
            trouble_by_span.setdefault(span_id, []).extend(step_markers)
        navigation = _dispatch_navigation(attributes)
        if navigation["state"] == NAV_CHANGED:
            note(MARKER_CONTEXT_NAVIGATION, span_id)
            navigation_changed.append(
                {
                    "span_id": span_id,
                    "command_name": _text_or_none(span.get("command_name")),
                    "from": navigation["from"],
                    "to": navigation["to"],
                }
            )
        elif navigation["basis"] == "no_context_recorded":
            nav_unknown += 1
        else:
            nav_same_type += 1
        dispatches.append(
            {
                "span_id": span_id,
                "command_call_id": _text_or_none(
                    attributes.get(tracing.ATTR_COMMAND_CALL_ID)
                ),
                "command_name": _text_or_none(span.get("command_name"))
                or _text_or_none(attributes.get("raw_command")),
                "context": _text_or_none(span.get("context")),
                "parameters_digest": _digest(attributes.get("parameters")),
                "start_ns": _exact_int(span.get("start_ns")),
                "status": status,
                "success": success,
            }
        )

    for event in _phase_events(tree, span_list):
        if event.kind == PHASE_INTENT:
            intent_spans += 1
            if event.detail.get("resolved") is False:
                intent_unresolved += 1
        elif event.kind == PHASE_PARAM:
            param_spans += 1
            # An extraction whose round is unrecorded either way. A v2 producer
            # records the ordinal, a v1 one records only the flag, and a span
            # carrying neither is the case this counts.
            if (
                event.detail.get("retry_round") is None
                and event.detail.get("retry_round_ordinal") is None
            ):
                retry_unknown += 1
            # Narrower and separately reported: WHICH attempt this was is
            # unrecorded. Every v1 span is in here, and so is a v2 span whose
            # producer could not determine the round -- a session rehydrated
            # from a continuation that persisted the stored parameters but not
            # their round records the flag and omits the ordinal, on purpose.
            # A consumer asking "which retry was this" needs that distinction;
            # folding it into the flag count would answer a different question.
            if event.detail.get("retry_round_ordinal") is None:
                round_unknown += 1
        for marker in event.markers:
            note(marker, event.span_id)
        # `awaiting_user` is a suspension, not trouble: a turn that asked the
        # user three times is a conversation, and counting it as loop evidence
        # would mark ordinary clarification as a defect.
        if event.owner_span_id and marker_is_trouble(event.markers):
            trouble_by_span.setdefault(event.owner_span_id, []).extend(
                m for m in event.markers if m != MARKER_AWAITING_USER
            )

    # A suspended turn is an awaiting-user state whether or not an ask_user span
    # survived. The status is the turn row's own recorded column.
    row_status = _text_or_none((turn_row or {}).get("status"))
    if row_status in ("awaiting_user", "suspended"):
        markers.add(MARKER_AWAITING_USER)

    dispatches.sort(key=lambda item: (item["start_ns"] is None, item["start_ns"] or 0))
    repeats = _repeat_groups(
        dispatches, trouble_by_span=trouble_by_span, policy=loop_policy
    )
    for group in repeats:
        markers.add(MARKER_REPEATED_COMMAND)
        evidence.setdefault(MARKER_REPEATED_COMMAND, []).extend(group.span_ids)
        if group.suspected_loop:
            markers.add(MARKER_SUSPECTED_LOOP)
            evidence.setdefault(MARKER_SUSPECTED_LOOP, []).extend(group.span_ids)

    signals_fn = decision_signals or default_decision_signals()
    signals = dict(signals_fn(span_list))
    if low_confidence_below is not None:
        test = low_confidence_test or default_low_confidence_test()
        if test(signals, low_confidence_below):
            markers.add(MARKER_LOW_CONFIDENCE)

    coverage = {
        "spans": len(span_list),
        "dispatch_spans": dispatch_spans,
        "intent_spans": intent_spans,
        "parameter_extraction_spans": param_spans,
        "dispatches_without_success": success_unknown,
        "dispatches_without_context_type": nav_unknown,
        # Both types recorded and equal: this build cannot prove the context
        # stayed put. Reported separately from `dispatches_without_context_type`
        # because it is a property of what a context type can say, not a
        # per-turn capture gap, and rolling the two together would put a
        # `partial_evidence` chip on essentially every turn.
        "dispatches_with_same_context_type": nav_same_type,
        "extractions_without_retry_flag": retry_unknown,
        "extractions_without_retry_round": round_unknown,
        "open_or_unended_spans": open_spans,
        "unrecognized_span_names": sorted(unrecognized),
    }
    # Partial evidence is a finding, not a footnote: a turn whose trace is
    # missing the very fields a marker is derived from can look clean for the
    # wrong reason, and `.18.2`/`.18.3` both require that stay visible.
    #
    # `extractions_without_retry_flag` is deliberately NOT one of the triggers,
    # even though it is exactly such a gap. `retry_round` is absent on 1701 of
    # 5648 extraction spans in the pilot store, and including it here marked 143
    # of 152 turns partial -- a marker that matches 94% of the dataset is not a
    # filter, it is noise, and it would bury the 57 turns whose navigation or
    # success is genuinely unreadable. The gap stays in `coverage`, where a
    # reader asking about retries finds it, instead of on a chip that would
    # stop meaning anything. Making it filterable needs the producer to record
    # the round ordinal (`fix-8ko2`).
    #
    # `extractions_without_retry_round` is excluded for the same reason and one
    # more: it is legitimately unknown for every session restored from a
    # continuation, because the persisted state carries the stored parameters
    # but not their round (`fix-7gp9`). Chipping that would mark every resumed
    # conversation partial.
    if (
        nav_unknown
        or success_unknown
        or open_spans
        or unrecognized
        or (dispatch_spans == 0 and len(span_list) > 0)
    ):
        markers.add(MARKER_PARTIAL_EVIDENCE)

    counts.update(
        {
            "context_navigations": len(navigation_changed),
            "intent_unresolved_attempts": intent_unresolved,
            "dispatches": dispatch_spans,
            "intent_spans": intent_spans,
            "parameter_extraction_spans": param_spans,
            "repeat_groups": len(repeats),
        }
    )
    for marker in MARKER_ORDER:
        counts[marker] = len(evidence.get(marker, ()))

    return TurnMarkers(
        turn_key=turn_key,
        basis=BASIS_SPANS,
        markers=tuple(m for m in MARKER_ORDER if m in markers),
        counts=counts,
        evidence={
            marker: _capped(tuple(dict.fromkeys(span_ids)))
            for marker, span_ids in sorted(evidence.items())
        },
        navigation={
            "changed": len(navigation_changed),
            "unknown": nav_unknown,
            "same_type": nav_same_type,
            "transitions": navigation_changed[:MAX_MARKER_ANCHORS],
        },
        coverage=coverage,
        repeats=tuple(repeats),
        decision_signals=signals,
        loop_policy=loop_policy.as_dict(),
    )


# ----------------------------------------------------------------------
# Turn and execution diagnosis (record-aware)
# ----------------------------------------------------------------------


def _record_command_success(record: Any) -> dict[str, bool]:
    """`command_call_id -> success`, from the turn record's CommandOutputs.

    The durable half of the pair: a dispatch whose span was dropped still has
    its outcome here, which is how `.18.2`'s "a completed turn containing an
    unsuccessful command" stays findable when the trace is incomplete.
    """
    out: dict[str, bool] = {}
    if not isinstance(record, Mapping):
        return out
    turn_output = record.get("turn_output")
    if not isinstance(turn_output, Mapping):
        return out
    outputs = turn_output.get("command_outputs")
    for output in outputs if isinstance(outputs, list) else []:
        if not isinstance(output, Mapping):
            continue
        call_id = _text_or_none(output.get("command_call_id"))
        response = output.get("command_response")
        success = None
        if isinstance(response, Mapping):
            success = _bool_or_none(response.get("success"))
        if success is None:
            success = _bool_or_none(output.get("success"))
        if call_id and success is not None:
            out[call_id] = success
    return out


def diagnose_turn(
    turn_key: str,
    turn_row: Optional[Mapping[str, Any]],
    record: Any,
    spans: Iterable[Mapping[str, Any]],
    steps: Sequence[ExecutionStep],
    *,
    ref: Optional[ExecutionRef] = None,
    low_confidence_below: Optional[float] = None,
    loop_policy: LoopPolicy = DEFAULT_LOOP_POLICY,
) -> TurnDiagnosis:
    """Diagnose one turn's steps, given the ledger steps already projected.

    `steps` come from `comparison.project_execution`, which runs the server's
    `execution_ledger`; nothing here re-derives the dispatch sequence.
    """
    span_list = [dict(span) for span in spans]
    tree = _SpanTree(span_list)
    summary = turn_markers(
        turn_key,
        span_list,
        turn_row=turn_row,
        low_confidence_below=low_confidence_below,
        loop_policy=loop_policy,
    )
    events = _phase_events(tree, span_list)
    events_by_owner: dict[Optional[str], list[PhaseEvent]] = {}
    for event in events:
        events_by_owner.setdefault(event.owner_call_id, []).append(event)

    groups_by_signature = {group.signature_id: group for group in summary.repeats}
    signature_by_span: dict[str, str] = {}
    for signature_id, group in groups_by_signature.items():
        for span_id in group.span_ids:
            signature_by_span[span_id] = signature_id

    record_success = _record_command_success(record)
    markers: set[str] = set(summary.markers)
    steps_without_span = 0
    record_only_failures: list[str] = []

    diagnosed: list[StepDiagnosis] = []
    for step in steps:
        step_markers: list[str] = []
        attributes: Mapping[str, Any] = {}
        if step.span_id and step.span_id in tree.by_id:
            attributes = tree.attributes(tree.by_id[step.span_id])
        else:
            steps_without_span += 1
        span_markers, _ = _dispatch_markers(step.status, attributes)
        step_markers.extend(span_markers)
        recorded = record_success.get(step.command_call_id)
        if recorded is None:
            recorded = step.response_success
        if recorded is False and MARKER_STEP_UNSUCCESSFUL not in step_markers:
            step_markers.append(MARKER_STEP_UNSUCCESSFUL)
            if not step.span_recorded:
                record_only_failures.append(step.command_call_id)
        navigation = _dispatch_navigation(attributes)
        if navigation["state"] == NAV_CHANGED:
            step_markers.append(MARKER_CONTEXT_NAVIGATION)
        step_events = tuple(events_by_owner.get(step.command_call_id, ()))
        for event in step_events:
            step_markers.extend(event.markers)
        signature_id = signature_by_span.get(step.span_id or "")
        if signature_id:
            step_markers.append(MARKER_REPEATED_COMMAND)
            if groups_by_signature[signature_id].suspected_loop:
                step_markers.append(MARKER_SUSPECTED_LOOP)
        ordered = tuple(m for m in MARKER_ORDER if m in set(step_markers))
        markers.update(ordered)
        anchor = (
            anchor_for_step(ref, step).as_dict()
            if ref is not None
            else {
                "store_id": None,
                "turn_key": step.turn_key,
                "target_kind": "step" if step.span_id else "turn",
                "span_ids": [step.span_id] if step.span_id else [],
                "command_call_id": step.command_call_id,
                "step_position": step.position,
                "anchorable": bool(step.span_id),
            }
        )
        diagnosed.append(
            StepDiagnosis(
                step=step,
                markers=ordered,
                navigation=navigation,
                events=step_events,
                repeat_signature_id=signature_id,
                anchor=anchor,
            )
        )

    if steps_without_span or record_only_failures:
        markers.add(MARKER_PARTIAL_EVIDENCE)

    coverage = dict(summary.coverage)
    coverage.update(
        {
            "steps": len(steps),
            "steps_without_span": steps_without_span,
            "record_only_failures": len(record_only_failures),
            "events_outside_any_dispatch": len(events_by_owner.get(None, ())),
        }
    )
    counts = dict(summary.counts)
    counts["steps"] = len(steps)
    counts["events"] = len(events)

    # The turn anchor is built here rather than through
    # `comparison.anchor_for_turn`, which needs a `TurnProjection` this function
    # is not given; the shape and the fields are that function's.
    turn_anchor: Mapping[str, Any] = (
        {
            "ref_id": ref.ref_id(),
            "store_id": ref.store_id,
            "turn_key": turn_key,
            "target_kind": "turn",
            "span_ids": [],
            "anchorable": True,
        }
        if ref is not None
        else {
            "store_id": None,
            "turn_key": turn_key,
            "target_kind": "turn",
            "span_ids": [],
            "anchorable": True,
        }
    )

    return TurnDiagnosis(
        turn_key=turn_key,
        basis=BASIS_SPANS_AND_RECORD,
        markers=tuple(m for m in MARKER_ORDER if m in markers),
        span_markers=summary,
        steps=tuple(diagnosed),
        events=tuple(events),
        repeats=summary.repeats,
        counts=counts,
        coverage=coverage,
        turn_anchor=turn_anchor,
        loop_policy=loop_policy.as_dict(),
    )


def diagnose_execution(
    ref: ExecutionRef,
    reader: ExecutionReader,
    *,
    ledger: Optional[LedgerProjection] = None,
    cost_rollup: Optional[Callable[..., Mapping[str, Any]]] = None,
    low_confidence_below: Optional[float] = None,
    loop_policy: LoopPolicy = DEFAULT_LOOP_POLICY,
    projection: Optional[ExecutionProjection] = None,
) -> ExecutionDiagnosis:
    """Diagnose one scoped recorded execution, end to end.

    Read-only. The dispatch sequence, artifacts, timing and cost come from
    `comparison.project_execution` (which runs the server's `execution_ledger`),
    and a caller that already holds a projection passes it in rather than paying
    for a second read.

    A turn the projection could not read stays in `projection.unavailable` and
    is counted in `coverage.turns_unavailable`; the readable turns are still
    diagnosed, because half an execution is worth inspecting.
    """
    if projection is None:
        projection = project_execution(
            ref,
            reader,
            ledger=ledger or default_ledger_projection(),
            cost_rollup=cost_rollup,
        )
    steps_by_turn: dict[str, list[ExecutionStep]] = {}
    for step in projection.steps:
        steps_by_turn.setdefault(step.turn_key, []).append(step)

    turns: list[TurnDiagnosis] = []
    markers: set[str] = set()
    counts: dict[str, int] = {}
    for turn in projection.turns:
        row = reader.turn(ref.store_id, turn.turn_key)
        spans = reader.trace(ref.store_id, turn.turn_key)
        diagnosis = diagnose_turn(
            turn.turn_key,
            row,
            (row or {}).get("record"),
            spans,
            steps_by_turn.get(turn.turn_key, []),
            ref=ref,
            low_confidence_below=low_confidence_below,
            loop_policy=loop_policy,
        )
        turns.append(diagnosis)
        markers.update(diagnosis.markers)
        for key, value in diagnosis.counts.items():
            if isinstance(value, int):
                counts[key] = counts.get(key, 0) + value

    coverage = {
        "turns": len(turns),
        "turns_unavailable": len(projection.unavailable),
        "steps": len(projection.steps),
        "steps_without_span": sum(
            int(turn.coverage.get("steps_without_span") or 0) for turn in turns
        ),
        "open_or_unended_spans": sum(
            int(turn.coverage.get("open_or_unended_spans") or 0) for turn in turns
        ),
        "unrecognized_span_names": sorted(
            {
                name
                for turn in turns
                for name in turn.coverage.get("unrecognized_span_names") or ()
            }
        ),
    }
    if projection.unavailable:
        markers.add(MARKER_PARTIAL_EVIDENCE)

    return ExecutionDiagnosis(
        ref=ref,
        projection=projection,
        turns=tuple(turns),
        markers=tuple(m for m in MARKER_ORDER if m in markers),
        counts=counts,
        coverage=coverage,
        loop_policy=loop_policy.as_dict(),
    )


# ----------------------------------------------------------------------
# Complete-dataset discovery
# ----------------------------------------------------------------------


class TurnSearchSource(Protocol):
    """What a complete scan needs from a store, and nothing more.

    Satisfied by `ObservabilityStore` and `ReadOnlyObservabilityStore`, so the
    same scan serves the live debug view and a sealed archive without a
    second code path. `get_turn` is only called when the caller asked for
    record-basis markers.
    """

    def list_turns(self, **kwargs: Any) -> list[dict[str, Any]]: ...

    def spans_for_turns(
        self, turn_keys: Iterable[str]
    ) -> dict[str, list[dict[str, Any]]]: ...

    def get_turn(self, turn_key: str) -> Optional[Mapping[str, Any]]: ...


# Store-level predicates: the ones `ObservabilityStore.list_turns` can answer in
# SQL. Named here so the scan passes exactly these through and nothing silently
# becomes a client-side filter.
def _require_bound(
    name: str, value: Any, *, minimum: Optional[int] = None, maximum: Optional[int] = None
) -> None:
    """An EXACT integer within bounds, or a refusal naming what arrived.

    `bool` is excluded because `True` is not a limit of 1, and a float is
    excluded because `2.5` rows is not a page size -- coercing either produces
    a query that quietly means something the caller did not write, and letting
    it through produces a `TypeError` from SQLite three layers down.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidTurnQuery(f"{name} must be an exact integer, not {value!r}")
    if minimum is not None and value < minimum:
        raise InvalidTurnQuery(f"{name} must be at least {minimum}, not {value}")
    if maximum is not None and value > maximum:
        raise InvalidTurnQuery(f"{name} must be at most {maximum}, not {value}")


# Markers the turn record can evidence when the trace cannot: a dispatch with no
# span still recorded its outcome in a `CommandOutput`. `partial_evidence` rides
# along because a record-only failure is itself a sign of an incomplete trace.
RECORD_SENSITIVE_MARKERS: frozenset[str] = frozenset(
    {MARKER_STEP_UNSUCCESSFUL, MARKER_PARTIAL_EVIDENCE}
)


_STORE_FILTERS = (
    "channel_id",
    "conversation_id",
    "status",
    "success",
    "command_name",
    "context",
    "experiment_id",
    "task_id",
    "attempt",
)


@dataclass(frozen=True)
class TurnQuery:
    """One discovery request, for a human list or an agent read.

    The store-level fields are `list_turns`'s own and are passed through
    untouched, so experiment/task/attempt scope is the store's answer and never
    re-derived here. The diagnostic fields are applied by the scan to EVERY row
    the store's filters admit, not to a page of them:

    - `markers_any` / `markers_all`: marker names (see `MARKER_ORDER`).
    - `low_confidence_below`: the recorded top-k margin test, run with the
      server's own function.
    - `text_contains`: case-insensitive substring over the listed text columns
      (`TEXT_SEARCH_FIELDS`).

    `limit` bounds the page in both modes; `offset` pages the MATCHED sequence
    of a complete scan, in `turn_key DESC` order -- the same order `list_turns`
    returns, so a page here is a page there.

    `scan_limit` and `resume_after` are the other mode: how much one call may
    walk, and where the next picks up. A segmented call stops as soon as the
    page is full, so its cursor never runs ahead of what the caller received.
    Mixing the two modes is refused rather than silently reinterpreted.
    """

    channel_id: Optional[str] = None
    conversation_id: Optional[int] = None
    status: Optional[str] = None
    success: Optional[bool] = None
    command_name: Optional[str] = None
    context: Optional[str] = None
    experiment_id: Optional[str] = None
    task_id: Optional[str] = None
    attempt: Optional[int] = None
    markers_any: tuple[str, ...] = ()
    markers_all: tuple[str, ...] = ()
    low_confidence_below: Optional[float] = None
    text_contains: Optional[str] = None
    limit: int = 100
    offset: int = 0
    scan_limit: Optional[int] = None
    resume_after: Optional[str] = None
    include_record: Optional[bool] = None
    loop_policy: LoopPolicy = DEFAULT_LOOP_POLICY

    def __post_init__(self) -> None:
        _require_bound("limit", self.limit, minimum=1, maximum=MAX_PAGE_LIMIT)
        _require_bound("offset", self.offset, minimum=0)
        if self.scan_limit is not None:
            _require_bound("scan_limit", self.scan_limit, minimum=1)
        if self.conversation_id is not None:
            _require_bound("conversation_id", self.conversation_id)
        if self.attempt is not None:
            _require_bound("attempt", self.attempt, minimum=0)
        if self.low_confidence_below is not None:
            threshold = self.low_confidence_below
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
                raise InvalidTurnQuery(
                    f"low_confidence_below must be a number, not {threshold!r}"
                )
            # NaN and the infinities are numbers that make every comparison a
            # lie: NaN matches nothing however low a margin was recorded, and
            # inf matches everything including a perfectly confident decision.
            if threshold != threshold or threshold in (float("inf"), float("-inf")):
                raise InvalidTurnQuery(
                    f"low_confidence_below must be finite, not {threshold!r}"
                )
            if threshold < 0:
                raise InvalidTurnQuery("low_confidence_below must not be negative")
        for name, values in (("markers_any", self.markers_any), ("markers_all", self.markers_all)):
            if isinstance(values, str) or not isinstance(values, (tuple, list)):
                raise InvalidTurnQuery(f"{name} must be a sequence of marker names")
            if bad := [value for value in values if not isinstance(value, str)]:
                raise InvalidTurnQuery(
                    f"{name} must contain marker names; got {bad[0]!r}"
                )
        if self.resume_after is not None and (
            not isinstance(self.resume_after, str) or not self.resume_after
        ):
            raise InvalidTurnQuery("resume_after must be a non-empty turn key")
        if self.offset and (self.scan_limit is not None or self.resume_after is not None):
            # Offset pages a complete scan; the cursor continues a segmented
            # one. Honouring both at once cannot be made complete: an offset
            # large enough to empty a segment's page would consume that
            # segment's matches -- counted, skipped, and then left behind the
            # cursor -- so they would be returned by nobody.
            raise InvalidTurnQuery(
                "offset pages a complete scan and scan_limit/resume_after "
                "segment one; they may not be combined. Page a complete scan "
                "with offset, or walk a segmented scan by passing each "
                "next_scan_cursor back as resume_after"
            )
        named = set(self.markers_any) | set(self.markers_all)
        if unknown := sorted(named - set(MARKER_ORDER)):
            raise InvalidTurnQuery(
                f"unknown marker(s): {', '.join(unknown)}; known markers are "
                + ", ".join(MARKER_ORDER)
            )

    @property
    def reads_record(self) -> bool:
        """Whether the scan reads each turn's record as well as its trace.

        `include_record=None` means AUTO, and auto is True exactly when the
        query tests a marker the record can evidence but the trace cannot. A
        dispatch whose span was dropped still recorded its `CommandOutput`, so
        a `step_unsuccessful` search that read only traces would answer "no
        failures" about a turn that recorded one -- the same class of false
        empty result as filtering after paging. A caller that wants the cheaper
        trace-only scan asks for it explicitly with `include_record=False`, and
        the page's `basis` says which one answered.
        """
        if self.include_record is not None:
            return bool(self.include_record)
        return bool(
            (set(self.markers_any) | set(self.markers_all)) & RECORD_SENSITIVE_MARKERS
        )

    @property
    def needs_spans(self) -> bool:
        """Whether a row's markers must be computed to answer this query.

        A query with no diagnostic predicate still gets markers when the caller
        wants facets; this only says whether the PREDICATE needs them.
        """
        return bool(
            self.markers_any
            or self.markers_all
            or self.low_confidence_below is not None
        )

    def store_filters(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in _STORE_FILTERS}

    def as_dict(self) -> dict[str, Any]:
        out = self.store_filters()
        out.update(
            {
                "markers_any": list(self.markers_any),
                "markers_all": list(self.markers_all),
                "low_confidence_below": self.low_confidence_below,
                "text_contains": self.text_contains,
                "limit": self.limit,
                "offset": self.offset,
                "scan_limit": self.scan_limit,
                "resume_after": self.resume_after,
                "include_record": self.include_record,
                "reads_record": self.reads_record,
                "loop_policy": self.loop_policy.as_dict(),
            }
        )
        return out


# The text columns `text_contains` searches. `record_json` is deliberately not
# among them: it is the largest column in the store and searching it would turn
# a bounded scan into a full-text read of every turn.
TEXT_SEARCH_FIELDS: tuple[str, ...] = (
    "turn_key",
    "user_message",
    "answer",
    "failure_reason",
    "entry_context",
    "entry_workflow_name",
)


# What a page's counts are counts OF. `dataset` means the scan started at the
# top and reached the end, so `total_matched` is the dataset's answer. `segment`
# means the call walked one bounded stretch -- because it was resumed, or
# because it stopped early -- and its counts are that stretch's, to be summed
# across the walk rather than read as a total.
SCOPE_DATASET = "dataset"
SCOPE_SEGMENT = "segment"


@dataclass(frozen=True)
class TurnSearchPage:
    """One bounded page of a scan, and the counts of what that scan walked.

    `facets` counts each marker over the rows matching everything EXCEPT the
    marker and low-confidence predicates, which is what makes "12 turns
    navigated context" a usable next click rather than a restatement of the
    filter already applied.

    A facet is None, never 0, when the query could not evaluate it:
    `low_confidence` has no meaning without a threshold, and a zero there would
    read as "no turn was unconfident" when the truth is that nobody asked.

    Two ways to walk the whole dataset, and `counts_scope` says which happened:

    `dataset` -- the default. One call scans to the end, `total_matched` is the
    dataset's total, `total_matched_exact` is True, `next_scan_cursor` is None,
    and further pages come from `offset`.

    `segment` -- the caller bounded per-request cost with `scan_limit`, or
    resumed. `total_matched` and `facets` count THIS segment; the dataset's
    totals are the sums across the walk, exact once a segment comes back with
    `next_scan_cursor` None. `offset` is refused in this mode, because the
    continuation is the cursor: pass `next_scan_cursor` back as
    `TurnQuery.resume_after` until it is None, and every match is returned
    exactly once -- including when the page limit is smaller than the scan
    limit, which is when a cursor that trailed the scan rather than the page
    used to step over matches it had already counted.
    """

    rows: tuple[dict[str, Any], ...]
    total_matched: int
    total_matched_exact: bool
    total_scanned: int
    scan_truncated: bool
    facets: Mapping[str, Optional[int]]
    order: str
    basis: str
    query: Mapping[str, Any]
    counts_scope: str = SCOPE_DATASET
    next_scan_cursor: Optional[str] = None
    store_id: Optional[str] = None

    @property
    def has_more(self) -> bool:
        """Whether anything is left: more matched rows, or more to scan.

        A truncated scan counts as more even when this page showed every match
        it found, because the rest of the dataset has not been looked at. This
        is what a caller must consult before saying "no matches": an empty page
        of an unfinished walk means not yet, not none.
        """
        if self.scan_truncated:
            return True
        return self.query["offset"] + len(self.rows) < self.total_matched

    @property
    def scan_complete(self) -> bool:
        """Whether the walk is over, so a count or an empty result is final."""
        return not self.scan_truncated

    def as_dict(self) -> dict[str, Any]:
        return {
            "turns": [dict(row) for row in self.rows],
            "total_matched": self.total_matched,
            "total_matched_exact": self.total_matched_exact,
            "counts_scope": self.counts_scope,
            "total_scanned": self.total_scanned,
            "scan_truncated": self.scan_truncated,
            "scan_complete": self.scan_complete,
            "next_scan_cursor": self.next_scan_cursor,
            "has_more": self.has_more,
            "facets": dict(self.facets),
            "order": self.order,
            "basis": self.basis,
            "store_id": self.store_id,
            "query": dict(self.query),
        }


def turn_matches(
    row: Mapping[str, Any],
    markers: Optional[TurnMarkers],
    query: TurnQuery,
) -> bool:
    """The diagnostic half of the predicate, in one place.

    The store's own filters are already applied by `list_turns`; this is
    everything a browser used to do to a page after the fact. Exported so an
    HTTP route, an agent read and a test assert the same rule rather than three
    that agree today.

    `markers` may be None only when the query needs no marker: asking a
    marker-bearing query about a row whose markers were never computed would
    otherwise silently answer False.
    """
    if query.text_contains:
        needle = query.text_contains.casefold()
        if not any(
            needle in value.casefold()
            for field_name in TEXT_SEARCH_FIELDS
            for value in (row.get(field_name),)
            if isinstance(value, str)
        ):
            return False
    if not query.needs_spans:
        return True
    if markers is None:
        raise InvalidTurnQuery(
            "this query tests markers but the row carries none; compute "
            "turn_markers before calling turn_matches"
        )
    if query.markers_any and not any(markers.has(m) for m in query.markers_any):
        return False
    if query.markers_all and not all(markers.has(m) for m in query.markers_all):
        return False
    if query.low_confidence_below is not None and not markers.has(MARKER_LOW_CONFIDENCE):
        return False
    return True


def search_turns(
    source: TurnSearchSource,
    query: TurnQuery,
    *,
    store_id: Optional[str] = None,
    with_facets: bool = True,
    derived_cache: Optional[DerivedTurnCache] = None,
    page_stamps: Optional[
        Callable[[Iterable[Mapping[str, Any]]], Mapping[str, Any]]
    ] = None,
) -> TurnSearchPage:
    """Filter, search, count and page over the COMPLETE authorized dataset.

    The scan walks `list_turns` in `SCAN_CHUNK`-sized pages -- the store's own
    filtered set, in its own `turn_key DESC` order -- and evaluates the
    diagnostic predicates on every row it walks. Only the requested page of
    MATCHED rows is retained; the rest are counted and dropped, so memory is
    bounded by the page, not by the dataset. There is no ceiling on the scan by
    default: a ceiling is a limit on what can be discovered at all, which is the
    bug, not a guard against it.

    Chunks are fetched by KEYSET (`before_turn_key`), not by offset, so a turn
    recorded while the scan is running cannot shift later chunks into repeating
    a row or skipping one -- which would corrupt `total_matched` in a way no
    caller could detect.

    Spans are read in `SPAN_FETCH_BATCH`-sized sub-batches through
    `iter_spans_for_turn_batches` when the source provides it (still one SQL
    query per sub-batch, not one query per turn), and are not read at all when
    neither the predicate nor facets need them. When `page_stamps` is supplied,
    page stamps (cut-at-limit, decision signals, cost) are computed for EVERY
    fetched turn while its decoded spans are in hand -- not only for rows that
    land on the page -- and stored in `derived_cache` with the markers, so
    later pages are warm instead of refetching and re-decoding those spans;
    then the span rows are dropped. Without `page_stamps`, rows carry only the
    diagnostic projection they always have -- agent-facing callers stay
    unchanged.

    An optional `derived_cache` skips the span fetch entirely on a freshness
    hit, but only when the source has a stable filesystem identity (and the
    caller-supplied `store_id` when present). Without a cache, or without a
    stable identity, behaviour matches the uncached scan.

    Scope is whatever the caller opened. This never resolves a path, never
    consults a second store, and passes the experiment/task/attempt filters to
    the store verbatim -- a store that records no experiments answers those
    filters with nothing, which is `list_turns`'s own documented behavior and is
    left alone.

    Ordering and continuation: `turn_key DESC`, the route's own order. A turn
    recorded DURING a scan may or may not be seen depending on where the cursor
    had reached -- that is inherent to reading a store while it is written and
    is why a caller needing a frozen view scans a sealed archive -- but it
    cannot corrupt the rows that were seen.
    """
    filters = query.store_filters()
    reads_record = query.reads_record
    need_markers = query.needs_spans or with_facets or reads_record
    want_stamps = page_stamps is not None
    rows: list[dict[str, Any]] = []
    total_matched = 0
    total_scanned = 0
    scan_truncated = False
    facets: dict[str, Optional[int]] = {marker: 0 for marker in MARKER_ORDER}
    if query.low_confidence_below is None:
        facets[MARKER_LOW_CONFIDENCE] = None
    page_start = query.offset
    page_end = query.offset + query.limit

    # The facet field: everything this query narrows EXCEPT the marker and
    # low-confidence predicates, so a facet count answers "how many more would
    # this chip find" rather than restating the filter already applied.
    facet_query = TurnQuery(**{**filters, "text_contains": query.text_contains})

    # Keyset, not offset: each chunk asks for rows strictly after the last key
    # seen, so a turn recorded mid-scan cannot shift a later page into repeating
    # a row or skipping one.
    #
    # `cursor` trails the last row this call has ACCOUNTED FOR -- counted in
    # `total_matched`, counted in `facets`, and either placed on the page or
    # deliberately skipped by `offset`. That invariant is the whole continuation
    # contract: resuming from it re-walks exactly the rows this call did not
    # account for, so a segmented walk returns every match exactly once.
    #
    # It is why a segmented scan STOPS when the page fills instead of scanning
    # on. Scanning past a full page would count and drop matches the caller
    # never received, and then hand back a cursor positioned beyond them -- they
    # would be counted once and shown never. An unbounded scan has no such
    # problem: it reaches every match in this one call, and paging it with
    # `offset` revisits them all.
    #
    # EITHER segmenting control puts the call in that mode. `resume_after` alone
    # is a caller already walking by cursor -- `scan_limit` only says how far
    # one call may walk when the page does NOT fill -- so treating a resume
    # without a bound as an unbounded scan would silently end the walk at the
    # first full page and strand every match behind it.
    segmented = query.scan_limit is not None or query.resume_after is not None
    cursor = query.resume_after
    page_full = False
    store_ident = _stable_store_identity(source, store_id)
    cacheable = (
        derived_cache is not None
        and store_ident is not None
        and callable(getattr(source, "span_stats_for_turns", None))
    )
    policy_key = _loop_policy_cache_key(query.loop_policy)
    iter_batches = getattr(source, "iter_spans_for_turn_batches", None)

    while not page_full:
        chunk_size = SCAN_CHUNK
        if query.scan_limit is not None:
            remaining = query.scan_limit - total_scanned
            if remaining <= 0:
                break
            chunk_size = min(SCAN_CHUNK, remaining)
        chunk = source.list_turns(
            **filters, limit=chunk_size, before_turn_key=cursor
        )
        if not chunk:
            break

        if not need_markers:
            for row in chunk:
                total_scanned += 1
                turn_key = _text_or_none(row.get("turn_key"))
                if turn_key is None:
                    continue
                cursor = turn_key
                if not turn_matches(row, None, query):
                    continue
                position = total_matched
                total_matched += 1
                if page_start <= position < page_end:
                    rows.append(dict(row))
                    if segmented and len(rows) >= query.limit:
                        page_full = True
                        break
            continue

        row_by_key = {
            key: row
            for row in chunk
            for key in [_text_or_none(row.get("turn_key"))]
            if key is not None
        }
        chunk_keys = list(row_by_key)
        markers_by_key: dict[str, TurnMarkers] = {}
        stamps_by_key: dict[str, Mapping[str, Any]] = {}
        fetch_keys: list[str] = []
        cache_keys: dict[str, Hashable] = {}

        # The freshness probe yields the cache key both to look an entry up and
        # to store one after a fetch, so it runs whenever caching is possible --
        # a cold cache still pays it, because every miss is put below.
        if cacheable:
            assert derived_cache is not None and store_ident is not None
            stats_by_key = source.span_stats_for_turns(chunk_keys)  # type: ignore[attr-defined]
            for key in chunk_keys:
                span_count, max_rowid = stats_by_key.get(key, (0, 0))
                cache_key = _derived_cache_key(
                    store_ident,
                    key,
                    span_count=int(span_count),
                    max_rowid=int(max_rowid),
                    turn_row=row_by_key.get(key),
                    low_confidence_below=query.low_confidence_below,
                    policy_key=policy_key,
                    want_stamps=want_stamps,
                )
                cache_keys[key] = cache_key
                hit = derived_cache.get(cache_key)
                if hit is not None:
                    markers_by_key[key] = hit.markers
                    if want_stamps:
                        stamps_by_key[key] = hit.stamps
                else:
                    fetch_keys.append(key)
        else:
            fetch_keys = list(chunk_keys)

        if fetch_keys:
            if callable(iter_batches):
                batch_iter = iter_batches(fetch_keys, batch_size=SPAN_FETCH_BATCH)
            else:
                batch_iter = [(fetch_keys, source.spans_for_turns(fetch_keys))]
            decoder = _SpanTree(())
            for _batch_keys, spans_by_turn in batch_iter:
                for key in _batch_keys:
                    raw_spans = spans_by_turn.get(key, [])
                    # Decode attributes once; turn_markers and page stamps both
                    # accept already-decoded mappings, so a second json.loads
                    # of the same column is avoided.
                    decoded = []
                    for span in raw_spans:
                        copy = dict(span)
                        copy["attributes"] = decoder.attributes(span)
                        decoded.append(copy)
                    markers = turn_markers(
                        key,
                        decoded,
                        turn_row=row_by_key.get(key),
                        low_confidence_below=query.low_confidence_below,
                        loop_policy=query.loop_policy,
                    )
                    stamps: Mapping[str, Any] = {}
                    if want_stamps:
                        assert page_stamps is not None
                        stamps = page_stamps(decoded)
                        stamps_by_key[key] = stamps
                    markers_by_key[key] = markers
                    cache_key = cache_keys.get(key)
                    if cacheable and cache_key is not None and derived_cache is not None:
                        derived_cache.put(
                            cache_key,
                            DerivedTurnEntry(
                                markers=markers,
                                stamps=dict(stamps) if want_stamps else {},
                                approx_bytes=estimate_entry_bytes(
                                    markers, stamps if want_stamps else {}
                                ),
                            ),
                        )
                    decoded.clear()
                spans_by_turn.clear()

        for row in chunk:
            total_scanned += 1
            turn_key = _text_or_none(row.get("turn_key"))
            if turn_key is None:
                continue
            # Advanced before the row is judged, not after: a scanned row is
            # accounted whether or not it matched, and a later segment must not
            # walk it again.
            cursor = turn_key
            markers = markers_by_key.get(turn_key)
            if markers is None:
                markers = turn_markers(
                    turn_key,
                    (),
                    turn_row=row,
                    low_confidence_below=query.low_confidence_below,
                    loop_policy=query.loop_policy,
                )
            if reads_record:
                markers = _widen_with_record(markers, _read_record(source, turn_key))

            if with_facets and turn_matches(row, markers, facet_query):
                for marker in markers.markers:
                    if facets.get(marker) is not None:
                        facets[marker] = (facets.get(marker) or 0) + 1

            if not turn_matches(row, markers, query):
                continue
            position = total_matched
            total_matched += 1
            if page_start <= position < page_end:
                annotated = dict(row)
                annotated["diagnosis"] = markers.as_dict()
                # The rail's existing chips read these two names; keeping
                # them means a row from this scan drops into the current UI
                # without the client learning a new shape.
                annotated["decision_signals"] = dict(markers.decision_signals)
                annotated["markers"] = list(markers.markers)
                if want_stamps:
                    assert page_stamps is not None
                    stamps = stamps_by_key.get(turn_key)
                    if stamps is None:
                        stamps = page_stamps(())
                    annotated.update(stamps)
                rows.append(annotated)
                if segmented and len(rows) >= query.limit:
                    page_full = True
                    break

    # When markers were not needed but the caller asked for page stamps, stamp
    # the page now in small batches so facets=false still avoids a second full
    # decode in annotate_turn_rows, without materialising the whole store.
    if want_stamps and not need_markers and rows:
        assert page_stamps is not None
        _stamp_page_rows(
            source,
            rows,
            stamps_fn=page_stamps,
            derived_cache=derived_cache if cacheable else None,
            store_ident=store_ident,
            policy_key=policy_key,
            low_confidence_below=query.low_confidence_below,
        )

    # Whether anything the store's filters admit still lies beyond the cursor.
    # Asked of the store rather than inferred from the loop, because "the chunk
    # came back short" and "the bound was reached" are different reasons to stop
    # and only one of them means the dataset is exhausted.
    bound_reached = (
        query.scan_limit is not None and total_scanned >= query.scan_limit
    )
    if segmented and cursor is not None and (page_full or bound_reached):
        scan_truncated = bool(
            source.list_turns(**filters, limit=1, before_turn_key=cursor)
        )

    # Counts describe what this call walked. They describe the DATASET only when
    # the walk started at the top (no `resume_after`) and reached the end (not
    # truncated); a segmented walk's dataset total is the sum of its segments,
    # and is exact once a segment reports `next_scan_cursor` None.
    covers_dataset = query.resume_after is None and not scan_truncated

    return TurnSearchPage(
        rows=tuple(rows),
        total_matched=total_matched,
        total_matched_exact=covers_dataset,
        counts_scope=SCOPE_DATASET if covers_dataset else SCOPE_SEGMENT,
        total_scanned=total_scanned,
        scan_truncated=scan_truncated,
        next_scan_cursor=cursor if scan_truncated else None,
        facets=facets,
        order="turn_key DESC",
        basis=BASIS_SPANS_AND_RECORD if reads_record else BASIS_SPANS,
        query=query.as_dict(),
        store_id=store_id,
    )


def _stable_store_identity(
    source: TurnSearchSource, store_id: Optional[str]
) -> Optional[Hashable]:
    """A filesystem-stable cache namespace, or None when caching must be skipped.

    `id(source)` is never used: per-request readers recycle object ids, and
    sealed archive stores are byte-copies that share turn keys with their
    live originals.
    """
    db_path = getattr(source, "db_path", None)
    if not isinstance(db_path, str) or not db_path:
        return None
    try:
        real = os.path.realpath(db_path)
    except OSError:
        return None
    if store_id:
        return (real, str(store_id))
    return (real,)


def _turn_row_cache_fields(turn_row: Optional[Mapping[str, Any]]) -> tuple[Any, ...]:
    """Every turn-row field `turn_markers` reads (today: only `status`)."""
    row = turn_row or {}
    return (_text_or_none(row.get("status")),)


def _derived_cache_key(
    store_ident: Hashable,
    turn_key: str,
    *,
    span_count: int,
    max_rowid: int,
    turn_row: Optional[Mapping[str, Any]],
    low_confidence_below: Optional[float],
    policy_key: tuple[tuple[str, Any], ...],
    want_stamps: bool,
) -> tuple[Any, ...]:
    return (
        store_ident,
        turn_key,
        span_count,
        max_rowid,
        _turn_row_cache_fields(turn_row),
        low_confidence_below,
        policy_key,
        want_stamps,
    )


def _loop_policy_cache_key(policy: LoopPolicy) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(policy.as_dict().items()))


def _stamp_page_rows(
    source: TurnSearchSource,
    rows: list[dict[str, Any]],
    *,
    stamps_fn: Callable[[Iterable[Mapping[str, Any]]], Mapping[str, Any]],
    derived_cache: Optional[DerivedTurnCache],
    store_ident: Optional[Hashable],
    policy_key: tuple[tuple[str, Any], ...],
    low_confidence_below: Optional[float],
) -> None:
    """Attach page stamps to already-selected rows, in memory-bounded batches."""
    need = [row for row in rows if "llm_calls_cut_at_limit" not in row]
    if not need:
        return
    keys = [str(row["turn_key"]) for row in need if row.get("turn_key")]
    stamps_by_key: dict[str, Mapping[str, Any]] = {}
    fetch_keys = list(keys)
    row_by_key = {str(row["turn_key"]): row for row in need if row.get("turn_key")}
    if (
        derived_cache is not None
        and store_ident is not None
        and len(derived_cache) > 0
        and callable(getattr(source, "span_stats_for_turns", None))
    ):
        stats_by_key = source.span_stats_for_turns(keys)  # type: ignore[attr-defined]
        fetch_keys = []
        for key in keys:
            span_count, max_rowid = stats_by_key.get(key, (0, 0))
            cache_key = _derived_cache_key(
                store_ident,
                key,
                span_count=int(span_count),
                max_rowid=int(max_rowid),
                turn_row=row_by_key.get(key),
                low_confidence_below=low_confidence_below,
                policy_key=policy_key,
                want_stamps=True,
            )
            hit = derived_cache.get(cache_key)
            if hit is not None and hit.stamps:
                stamps_by_key[key] = hit.stamps
            else:
                fetch_keys.append(key)
    if fetch_keys:
        iter_batches = getattr(source, "iter_spans_for_turn_batches", None)
        if callable(iter_batches):
            batch_iter = iter_batches(fetch_keys, batch_size=SPAN_FETCH_BATCH)
        else:
            batch_iter = [(fetch_keys, source.spans_for_turns(fetch_keys))]
        for batch_keys, spans_by_turn in batch_iter:
            for key in batch_keys:
                stamps_by_key[key] = stamps_fn(spans_by_turn.get(key, []))
            spans_by_turn.clear()
    for row in need:
        key = str(row.get("turn_key") or "")
        row.update(stamps_by_key.get(key) or stamps_fn(()))



def _read_record(source: TurnSearchSource, turn_key: str) -> Any:
    """One turn's decoded record, or None when it has none this reader can use."""
    row = source.get_turn(turn_key)
    raw = (row or {}).get("record_json")
    if not isinstance(raw, str):
        return None
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return None


def _widen_with_record(markers: TurnMarkers, record: Any) -> TurnMarkers:
    """Add the failure markers only the turn record evidences.

    A dispatch whose span was dropped still recorded its `CommandOutput`, so an
    unsuccessful command in a completed turn stays findable. Only ever ADDS: the
    span basis is a subset of this, which is what lets a list row and a detail
    view be compared instead of merely hoped to agree.
    """
    failures = [call_id for call_id, ok in _record_command_success(record).items() if not ok]
    if not failures:
        return markers
    if markers.has(MARKER_STEP_UNSUCCESSFUL):
        return markers
    counts = dict(markers.counts)
    counts[MARKER_STEP_UNSUCCESSFUL] = counts.get(MARKER_STEP_UNSUCCESSFUL, 0) + len(failures)
    coverage = dict(markers.coverage)
    coverage["record_only_failures"] = len(failures)
    new_markers = set(markers.markers) | {MARKER_STEP_UNSUCCESSFUL, MARKER_PARTIAL_EVIDENCE}
    return TurnMarkers(
        turn_key=markers.turn_key,
        basis=BASIS_SPANS_AND_RECORD,
        markers=tuple(m for m in MARKER_ORDER if m in new_markers),
        counts=counts,
        evidence=dict(markers.evidence)
        | {"record_only_failures": {"command_call_ids": failures[:MAX_MARKER_ANCHORS]}},
        navigation=markers.navigation,
        coverage=coverage,
        repeats=markers.repeats,
        decision_signals=markers.decision_signals,
        loop_policy=markers.loop_policy,
    )
