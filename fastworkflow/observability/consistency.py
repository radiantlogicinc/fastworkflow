"""How alike are the repeated runs of one task (`fix-9eg.17.5`).

Three descriptive metrics over evidence that is already recorded, and nothing
else. There is no target here, no rubric, no verdict and no winner: a run that
agrees with its siblings is CONSISTENT, which is not the same as correct, and
this module never says otherwise. Consistency is a shape a human or a coding
agent reads and then judges; the judging happens through the existing feedback
taxonomy on the existing comparison, not here.

The three:

1. **Planning similarity.** Cosine between embeddings of the recorded,
   user-visible plan text -- the `plan` attribute of the `fw.planner.plan` and
   `fw.planner.replan` spans, in recorded order, with turn boundaries kept
   explicit. It is the plan the planner PUBLISHED, never the chain-of-thought
   that produced it: `reasoning` is deliberately not on those spans, so there
   is nothing hidden to leak in and nothing to reconstruct.
2. **Final-answer similarity.** Cosine over the answer of the LAST recorded
   turn, kept apart from planning similarity and from the artifact comparison.
   Similar wording is not proof of equal facts, equal artifacts or correctness.
3. **Execution-step variation.** Canonical executed-step counts from the
   existing execution ledger, per run and across runs.

WHAT A STEP IS, stated once because every count below depends on it. The
canonical count is the number of DISTINCT `command_call_id` rows the execution
ledger files for the run (`step_count_rule = ledger_distinct_dispatch`). The
ledger is already the one implementation of "what dispatches happened", and it
joins the record's refs, the `fw.command.execute` spans and each parent's
`child_calls` list onto one row per dispatch -- so a span opened and closed, a
dispatch named by both the record and its span, and a wrapper that also appears
in its parent's child list each count ONCE. Failures, retries and navigation
commands are dispatches and are counted; nothing is filtered by outcome,
because a success-only count would make a flailing run look tidy. The
root/child split is published beside the count so the composition is visible
rather than assumed.

WHAT IS NEVER GUESSED. Absent text is `absent`, never an empty string that
would make two silent runs look identical. A value the capture policy withheld
is `withheld` and a bounded one is `truncated`, both distinct from absent. A
turn the store cannot produce makes the run's evidence INCOMPLETE and its step
count `partial`, which keeps it out of the distribution while leaving it on
screen. A run with no recorded turns has an UNKNOWN step count, not zero. One
repeat is `insufficient_repeats`, not perfect consistency.

BEST RUN DOES NOT MOVE THE AGGREGATE. `summary` is over all eligible runs and
every eligible pair; `reference_rows` is the selected best run against each
other run. They are computed from the same pair table and published apart, so
choosing a different best run changes which rows are highlighted and changes
nothing about the distribution.

COMPARING TWO EXPERIMENTS is refused unless both sides carry the same
`metric_identity` -- metric version, text projection version and embedding
model fingerprint. A cosine from one embedding model minus a cosine from
another is a number with no meaning, and printing it with a caveat is worse
than not printing it.

EMBEDDINGS ARE LOCAL OR ABSENT. `LocalTextEmbedder` loads an already-installed
HuggingFace model with `local_files_only=True`. There is no download, no paid
provider and no LLM judge, and there is no lexical stand-in either: a missing
model yields an actionable `unavailable` state carrying the command that would
install it, while the step-count metrics -- which need no model -- still
compute. Derived vectors are cached by content/model/projection identity
OUTSIDE the evidence store, so nothing here writes to sealed evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from fastworkflow.observability import comparison as comparison_module
from fastworkflow.observability.capture_policy import CAPTURE_ENVELOPE_MARKER

# Versions that gate a numeric delta between two experiments. Bump the metric
# version when a number's MEANING changes and the projection version when the
# text fed to the model changes; either one makes older figures incomparable,
# which is the whole reason they are published rather than kept internal.
METRICS_VERSION = "consistency/metrics/1"
TEXT_PROJECTION_VERSION = "consistency/text/1"

# The canonical counting rule, named so a payload can be read without this file.
STEP_COUNT_RULE = "ledger_distinct_dispatch"

# Restated rather than imported from `fastworkflow.tracing`, the way
# `comparison.py` restates `fw.command.execute`: reading stored evidence should
# not pull in the emitter. `test_consistency.py` asserts these still equal the
# tracing constants, so the copy cannot drift silently.
SPAN_PLANNER_PLAN = "fw.planner.plan"
SPAN_PLANNER_REPLAN = "fw.planner.replan"
PLAN_SPAN_NAMES = (SPAN_PLANNER_PLAN, SPAN_PLANNER_REPLAN)

# The model this repo already has on disk for sentence-level similarity. It is
# named, not discovered: a fallback to "whatever else is cached" would change
# the meaning of every stored number without changing its label.
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# The checkpoint's own `max_seq_length`: one pass through the encoder covers
# this many of the text's tokens. It is NOT a cap on the text. A plan or a
# final answer longer than one window is embedded whole, window by window, and
# the windows are combined by the rule below.
DEFAULT_WINDOW_TOKENS = 256
# The work bound. 64 windows is ~16k tokens of one plan or one answer, far past
# anything this product records; past it the text really is cut, and every
# similarity computed from it is labelled partial rather than reported as a
# similarity over the whole output.
DEFAULT_MAX_WINDOWS = 64
# Named in the payload and folded into the fingerprint: two cosines produced by
# different long-text rules are not comparable, and the difference is invisible
# in the number itself.
AGGREGATION_METHOD = "token_weighted_mean_of_window_vectors_l2"
LONG_TEXT_RULE = (
    "text longer than one encoder window is split into consecutive "
    "non-overlapping windows; each window is mean-pooled over its attention "
    "mask and L2-normalized, and the windows are combined by a token-count-"
    "weighted mean that is L2-normalized again. A text that fits one window "
    "is unaffected by this rule."
)

# Bounds. A task with 40 repeats is 780 pairs and megabytes of text; a request
# handler must not spend minutes on one. Exceeding either bound produces an
# explicitly CAPPED report naming what was left out, never a quiet sample.
DEFAULT_MAX_RUNS = 30
DEFAULT_MAX_PAIRS = 300

# Text states. `absent` is "the evidence records none", which is not `withheld`
# ("the capture policy removed it") and not `unreadable` ("the turn could not
# be read"). Collapsing the three would make three different investigations
# look like one.
TEXT_PRESENT = "present"
TEXT_ABSENT = "absent"
TEXT_WITHHELD = "withheld"
TEXT_UNREADABLE = "unreadable"

# Count states. `partial` is a real count over an incomplete read; it is shown
# and excluded from the distribution, because it is not the run's count.
COUNT_KNOWN = "known"
COUNT_PARTIAL = "partial"
COUNT_UNKNOWN = "unknown"

# Coverage of a computed figure.
COVERAGE_FULL = "full"
COVERAGE_PARTIAL = "partial"
COVERAGE_CAPPED = "capped"

METRIC_COMPUTED = "computed"
METRIC_UNKNOWN = "unknown"
METRIC_UNAVAILABLE = "unavailable"

SPREAD_FORMULA = "spread = max - min over the computed pairwise cosine values"
SD_FORMULA = (
    "population SD = sqrt(sum((x - mean)^2) / n) over the runs with a known "
    "step count"
)
NORMALIZED_STEP_FORMULA = (
    "normalized step-count difference = 100 * abs(a - b) / max(a, b); both "
    "counts known and zero is 0%"
)


class ConsistencyError(RuntimeError):
    """A consistency request that cannot be answered as asked."""


class EmbeddingUnavailable(ConsistencyError):
    """No local embedding model, stated with what would fix it.

    Deliberately not a fallback. A lexical stand-in would answer every question
    this module is asked, with numbers that look like the real ones and mean
    something else.
    """

    def __init__(self, reason: str, remedy: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.remedy = remedy


# ----------------------------------------------------------------------
# Small numerics, written out because their definitions are the contract
# ----------------------------------------------------------------------


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity, defined here so a test can check it without a model.

    Refuses vectors of different length rather than zipping to the shorter one:
    two embeddings of different width come from two different models, and a
    truncated dot product of them is a number about nothing.
    """
    if len(left) != len(right):
        raise ConsistencyError(
            f"cosine needs two vectors of the same width, got {len(left)} "
            f"and {len(right)}"
        )
    if not left:
        raise ConsistencyError("cosine is undefined for an empty vector")
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
    right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ConsistencyError("cosine is undefined for a zero vector")
    # Clamped: a normalized dot product can land on 1.0000000000000002 in
    # float64, and a similarity above 1 reads as a bug in the metric.
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


def population_sd(values: Sequence[float]) -> Optional[float]:
    """Population standard deviation, or None for no values.

    Population, not sample: these are ALL the runs there are, not a draw from a
    larger pool, and dividing by n-1 would be claiming an inference nobody made.
    One value has a spread of 0.0 -- which is true and is why `n = 1` is
    reported as insufficient repeats separately, rather than being smuggled in
    as a perfect score.
    """
    if not values:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def normalized_step_difference(left: int, right: int) -> float:
    """`100 * abs(a-b) / max(a,b)`, with both-zero defined as 0%."""
    largest = max(left, right)
    if largest == 0:
        return 0.0
    return 100.0 * abs(left - right) / largest


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


# ----------------------------------------------------------------------
# Text projection: what actually gets embedded
# ----------------------------------------------------------------------


def _clean_text(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    return text or None


def _span_attributes(span: Mapping[str, Any]) -> dict[str, Any]:
    """A span's attributes, decoded or not, as the reader handed them over.

    The workspace reader decodes the JSON column and `ObservabilityStore`
    returns the raw text; both are legitimate and `server.py` and
    `comparison.py` each accept both for the same reason. Six lines here beats
    reaching into either module's private helper.
    """
    raw = span.get("attributes")
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except (ValueError, TypeError):
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _capture_envelope(value: Any) -> Optional[Mapping[str, Any]]:
    """The capture policy's envelope, if this value is one.

    A withheld or bounded value arrives as a dict carrying
    `CAPTURE_ENVELOPE_MARKER`, not as text. Treating it as absent would report
    "this run recorded no plan" about a run that recorded one and had it
    removed, which sends a reader looking in the wrong place.
    """
    if isinstance(value, Mapping) and value.get(CAPTURE_ENVELOPE_MARKER):
        return value
    return None


@dataclass(frozen=True)
class PlanSegment:
    """One recorded plan emission, with the boundary it sits behind."""

    turn_index: int
    turn_key: str
    span_name: str
    replan_trigger: Optional[str]
    text: str

    @property
    def boundary(self) -> str:
        kind = "replan" if self.span_name == SPAN_PLANNER_REPLAN else "plan"
        trigger = f" after {self.replan_trigger}" if self.replan_trigger else ""
        return f"[turn {self.turn_index + 1} {kind}{trigger}]"

    def as_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "turn_key": self.turn_key,
            "span_name": self.span_name,
            "replan_trigger": self.replan_trigger,
            "boundary": self.boundary,
            "chars": len(self.text),
        }


@dataclass(frozen=True)
class ProjectedText:
    """One side of one similarity metric, as text plus why it is what it is.

    `state` is the honest four-way answer; `text` is set only when the state is
    `present`. Two runs that both recorded nothing are two `absent` values and
    form no pair -- not two equal empty strings with a cosine of 1.
    """

    kind: str
    state: str
    text: Optional[str] = None
    reason: Optional[str] = None
    segments: tuple[PlanSegment, ...] = ()
    withheld_segments: int = 0
    truncated_source: bool = False
    # Something this text is made of was named by the reference and could not
    # be read: a pruned turn, a withheld plan emission. The surviving text is
    # still worth comparing, but it is NOT the whole of what the run recorded,
    # and every similarity computed from it is partial coverage.
    evidence_incomplete: bool = False
    incomplete_reason: Optional[str] = None

    @property
    def present(self) -> bool:
        return self.state == TEXT_PRESENT and bool(self.text)

    @property
    def partial_source(self) -> bool:
        """The text is present but is less than the run recorded."""
        return bool(
            self.truncated_source or self.withheld_segments or self.evidence_incomplete
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "state": self.state,
            "reason": self.reason,
            "chars": len(self.text or ""),
            "segments": [segment.as_dict() for segment in self.segments],
            "segment_count": len(self.segments),
            "withheld_segments": self.withheld_segments,
            # The capture policy bounded the SOURCE value. Distinct from the
            # embedding-window truncation reported per vector below: one is
            # evidence that was never stored whole, the other is a limit of the
            # model, and a reader chasing a low similarity needs to know which.
            "source_bounded": self.truncated_source,
            "evidence_incomplete": self.evidence_incomplete,
            "incomplete_reason": self.incomplete_reason,
            "partial_source": self.partial_source,
        }


def canonical_plan_text(segments: Sequence[PlanSegment]) -> str:
    """The exact string a plan sequence is embedded as.

    Order and turn boundaries are content: a run that planned A then B is not
    the run that planned B then A, and two plans emitted in one turn are not
    the same shape as the same two split across a suspension and a resume. So
    each segment is prefixed by its own boundary line and the segments are
    joined in recorded order. Nothing volatile goes in -- no timestamps, no
    span ids, no call ids -- because those differ between two runs of the same
    plan and would depress every similarity by a constant nobody can see.
    """
    return "\n".join(
        f"{segment.boundary}\n{segment.text}" for segment in segments
    )


def _plan_segments(
    turn_index: int, turn_key: str, spans: Sequence[Mapping[str, Any]]
) -> tuple[list[PlanSegment], int, bool]:
    """Plan emissions of one turn, in recorded order.

    Returns the segments, how many were withheld by the capture policy, and
    whether any surviving one was stored bounded.
    """
    ordered = sorted(
        (span for span in spans if span.get("name") in PLAN_SPAN_NAMES),
        key=lambda span: (
            span.get("start_ns") if isinstance(span.get("start_ns"), int) else 0,
            str(span.get("span_id") or ""),
        ),
    )
    segments: list[PlanSegment] = []
    withheld = 0
    bounded = False
    for span in ordered:
        attributes = _span_attributes(span)
        raw = attributes.get("plan")
        envelope = _capture_envelope(raw)
        if envelope is not None:
            prefix = _clean_text(envelope.get("prefix"))
            if prefix is None:
                withheld += 1
                continue
            bounded = True
            raw = prefix
        text = _clean_text(raw)
        if text is None:
            # An emitted plan with no text is a real recorded state (the
            # planner returned nothing), and it carries no content to compare.
            continue
        trigger = attributes.get("replan_trigger")
        segments.append(
            PlanSegment(
                turn_index=turn_index,
                turn_key=turn_key,
                span_name=str(span.get("name")),
                replan_trigger=trigger if isinstance(trigger, str) else None,
                text=text,
            )
        )
    return segments, withheld, bounded


# ----------------------------------------------------------------------
# One run's evidence
# ----------------------------------------------------------------------


@dataclass
class RunEvidence:
    """Everything the three metrics need about one attempt, and its coverage."""

    attempt: int
    experiment_id: Optional[str]
    task_id: Optional[str]
    ref: Optional[comparison_module.ExecutionRef]
    plan: ProjectedText
    answer: ProjectedText
    step_count: Optional[int]
    step_count_state: str
    step_detail: dict[str, Any] = field(default_factory=dict)
    turns_named: int = 0
    turns_readable: int = 0
    unavailable: tuple[str, ...] = ()
    execution_status: Optional[str] = None
    outcome: Optional[str] = None
    evidence_state: Optional[str] = None
    evidence_label: Optional[str] = None
    projection_error: Optional[str] = None
    # Whether "the last turn" is a recorded fact for this run or the
    # reference's order standing in for one. Published, because the final
    # answer and the plan sequence are both claims about order.
    turn_order: Optional[str] = None
    turn_order_reason: Optional[str] = None

    @property
    def evidence_complete(self) -> bool:
        return not self.unavailable and self.projection_error is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempt": self.attempt,
            "experiment_id": self.experiment_id,
            "task_id": self.task_id,
            "ref_id": None if self.ref is None else self.ref.ref_id(),
            "execution_ref": None if self.ref is None else self.ref.as_dict(),
            "execution_status": self.execution_status,
            "outcome": self.outcome,
            "evidence_state": self.evidence_state,
            "evidence_label": self.evidence_label,
            "step_count": self.step_count,
            "step_count_state": self.step_count_state,
            "step_detail": dict(self.step_detail),
            "plan": self.plan.as_dict(),
            "answer": self.answer.as_dict(),
            "evidence": {
                "turns_named": self.turns_named,
                "turns_readable": self.turns_readable,
                "unavailable": list(self.unavailable),
                "complete": self.evidence_complete,
                "error": self.projection_error,
                "turn_order": self.turn_order,
                "turn_order_reason": self.turn_order_reason,
            },
        }


class _CachingReader:
    """One read per (store, turn) for a whole report.

    A consistency report projects each run twice -- once through the shared
    `project_execution` for the ledger, once here for the plan spans -- and
    would otherwise pull every trace out of sqlite twice per request. Read-only
    and per-report, so it cannot serve a later request stale evidence.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._turns: dict[tuple[str, str], Any] = {}
        self._traces: dict[tuple[str, str], list[Mapping[str, Any]]] = {}

    def turn(self, store_id: str, turn_key: str) -> Any:
        key = (store_id, turn_key)
        if key not in self._turns:
            self._turns[key] = self._inner.turn(store_id, turn_key)
        return self._turns[key]

    def trace(self, store_id: str, turn_key: str) -> list[Mapping[str, Any]]:
        key = (store_id, turn_key)
        if key not in self._traces:
            self._traces[key] = list(self._inner.trace(store_id, turn_key))
        return self._traces[key]


def _absent(kind: str, reason: str, state: str = TEXT_ABSENT) -> ProjectedText:
    return ProjectedText(kind=kind, state=state, reason=reason)


def collect_run_evidence(
    row: Mapping[str, Any],
    reader: Any,
    *,
    ledger: Any = None,
) -> RunEvidence:
    """Project one attempt row into the evidence the metrics read.

    `row` is an attempt row as `best_run.task_run_summary` (live) or
    `selection_api._workspace_attempts` (sealed) produces it, so this module
    never resolves a store, a turn key or a reference of its own.

    A row with no reference is not an error: an attempt that finished with
    nothing recorded is a real outcome, and it appears with an UNKNOWN step
    count and absent text rather than being dropped from the list.
    """
    attempt = int(row["attempt"])
    raw_ref = row.get("execution_ref")
    common = {
        "attempt": attempt,
        "experiment_id": row.get("experiment_id"),
        "task_id": row.get("task_id"),
        "execution_status": row.get("execution_status"),
        "outcome": row.get("outcome"),
        "evidence_state": row.get("evidence_state"),
        "evidence_label": row.get("evidence_label"),
    }
    if not row.get("comparable") or not raw_ref:
        label = row.get("evidence_label") or "this attempt recorded no turns"
        return RunEvidence(
            ref=None,
            plan=_absent("plan", label),
            answer=_absent("final_answer", label),
            step_count=None,
            step_count_state=COUNT_UNKNOWN,
            **common,
        )

    ref = comparison_module.ExecutionRef.from_mapping(raw_ref)
    try:
        projection = comparison_module.project_execution(ref, reader, ledger=ledger)
    except comparison_module.ComparisonError as exc:
        # The reference and the evidence disagree, or a pass could not be
        # resolved. That is a fact about this run, not a failure of the report:
        # the other runs still have metrics and this one says why it has none.
        reason = str(exc)
        return RunEvidence(
            ref=ref,
            plan=_absent("plan", reason, TEXT_UNREADABLE),
            answer=_absent("final_answer", reason, TEXT_UNREADABLE),
            step_count=None,
            step_count_state=COUNT_UNKNOWN,
            turns_named=len(ref.turn_keys),
            projection_error=reason,
            **common,
        )

    readable = {turn.turn_key for turn in projection.turns}
    segments: list[PlanSegment] = []
    withheld = 0
    bounded = False
    for turn in projection.turns:
        found, missing, was_bounded = _plan_segments(
            turn.turn_index, turn.turn_key, reader.trace(ref.store_id, turn.turn_key)
        )
        segments.extend(found)
        withheld += missing
        bounded = bounded or was_bounded

    # Turns the reference NAMES but the store could not return. Their plan
    # emissions are not absent, they are unread: the surviving plan is a
    # fragment of the recorded sequence, and every comparison against it has
    # to say so rather than treat the fragment as the run's plan.
    unread_turns = [key for key in ref.turn_keys if key not in readable]
    plan_gaps: list[str] = []
    if unread_turns:
        plan_gaps.append(
            f"{len(unread_turns)} turn(s) named by this run could not be read, "
            "so any plan they recorded is missing from this sequence"
        )
    if projection.turn_order == comparison_module.TURN_ORDER_REFERENCE:
        # The segments are all here; what is not recoverable is the order they
        # were emitted in, and a plan SEQUENCE compared in an order the
        # evidence did not record is a fragment of the truth, not the run's
        # planning.
        plan_gaps.append(
            "the recorded order of this run's turns is not derivable from the "
            "evidence, so this plan sequence is shown in the reference's order"
        )
    if withheld:
        plan_gaps.append(
            f"{withheld} recorded plan emission(s) were removed by the capture "
            "policy"
        )

    if segments:
        plan = ProjectedText(
            kind="plan",
            state=TEXT_PRESENT,
            text=canonical_plan_text(segments),
            segments=tuple(segments),
            withheld_segments=withheld,
            truncated_source=bounded,
            evidence_incomplete=bool(plan_gaps),
            incomplete_reason="; ".join(plan_gaps) or None,
        )
    elif withheld:
        plan = ProjectedText(
            kind="plan",
            state=TEXT_WITHHELD,
            reason=(
                f"{withheld} recorded plan emission(s) were removed by the "
                "capture policy, so this run's plan text cannot be compared"
            ),
            withheld_segments=withheld,
        )
    else:
        plan = _absent(
            "plan",
            "no fw.planner.plan or fw.planner.replan span recorded a plan for "
            "this run",
        )

    answer = _final_answer(projection, ref, readable)

    complete = not projection.unavailable
    step_detail = {
        "root": sum(1 for step in projection.steps if not step.child_call),
        "child": sum(1 for step in projection.steps if step.child_call),
        "failed": sum(
            1
            for step in projection.steps
            if step.success is False or step.status == "error"
        ),
        "asked_user": sum(step.asked_user for step in projection.steps),
        "turns": len(projection.turns),
    }
    return RunEvidence(
        ref=ref,
        plan=plan,
        answer=answer,
        step_count=len(projection.steps),
        step_count_state=COUNT_KNOWN if complete else COUNT_PARTIAL,
        step_detail=step_detail,
        turns_named=len(ref.turn_keys),
        turns_readable=len(readable),
        unavailable=tuple(projection.unavailable),
        turn_order=projection.turn_order,
        turn_order_reason=projection.turn_order_reason,
        **common,
    )


def _final_answer(
    projection: comparison_module.ExecutionProjection,
    ref: comparison_module.ExecutionRef,
    readable: set[str],
) -> ProjectedText:
    """The CHRONOLOGICALLY last turn's answer, or an explicit absence.

    Not "the last non-empty answer". A run whose final turn answered nothing
    produced no final answer, and borrowing an earlier turn's would compare one
    run's ending against another run's middle.

    Nor "the last turn the reference happens to name last". A reference's turn
    vector is an identity, frozen when the run was filed, and an archived one
    can name its turns newest-first; reading its tail as the run's ending gave
    archived runs their OPENING answer under the name "final answer"
    (fix-6v3n). The ending is whichever turn the evidence recorded last, which
    is what `project_execution` now orders `projection.turns` by.

    Nor "the last READABLE turn's answer". If the reference names a turn the
    store can no longer return, that turn's place in the run is unknown -- it
    may be the one that ended it. The turn before it is the middle of the run,
    not its conclusion, and substituting it would silently compare a run's
    ending against another run's penultimate step and report the result as
    final-answer similarity.
    """
    if not projection.turns:
        return _absent(
            "final_answer",
            "no turn of this run could be read, so it has no final answer",
            TEXT_UNREADABLE,
        )
    unread = [key for key in ref.turn_keys if key not in readable]
    if unread:
        return _absent(
            "final_answer",
            f"{len(unread)} turn(s) named by this run could not be read "
            f"({', '.join(sorted(unread))}), so which turn ended it is "
            "unknown; an earlier turn's answer is the middle of this run, not "
            "its ending",
            TEXT_UNREADABLE,
        )
    if projection.turn_order == comparison_module.TURN_ORDER_REFERENCE:
        return _absent(
            "final_answer",
            "the recorded order of this run's turns is not derivable from the "
            "evidence, so which of them ended the run is unknown",
            TEXT_UNREADABLE,
        )
    last = projection.turns[-1]
    envelope = _capture_envelope(last.answer)
    if envelope is not None:
        return ProjectedText(
            kind="final_answer",
            state=TEXT_WITHHELD,
            reason="the capture policy removed this turn's answer",
        )
    text = _clean_text(last.answer)
    if text is None:
        return _absent(
            "final_answer",
            f"the last recorded turn ({last.turn_key}) recorded no answer text",
        )
    return ProjectedText(kind="final_answer", state=TEXT_PRESENT, text=text)


# ----------------------------------------------------------------------
# The local embedding facility
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class EmbeddedText:
    vector: tuple[float, ...]
    token_count: int
    truncated: bool
    # How many model-capacity windows the text needed. One means the text fit
    # the encoder in a single pass; more means it was embedded whole by the
    # documented aggregation rather than cut down to its first paragraph.
    windows: int = 1


class VectorCache:
    """Derived vectors, addressed by content and model identity.

    Outside the evidence store, always: this writes under a directory the
    caller names (the workflow's own state directory in the server), never into
    a database that holds recorded evidence, and never into a sealed archive.
    The key is the model fingerprint plus a digest of the exact text, so a
    changed projection or a changed model cannot read a stale vector -- it
    simply misses.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    def _path(self, fingerprint: str, text: str) -> Path:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return self.root / fingerprint / digest[:2] / f"{digest}.json"

    def get(self, fingerprint: str, text: str) -> Optional[EmbeddedText]:
        path = self._path(fingerprint, text)
        try:
            payload = json.loads(path.read_text("utf-8"))
        except (OSError, ValueError):
            return None
        vector = payload.get("vector")
        if not isinstance(vector, list) or not vector:
            return None
        return EmbeddedText(
            vector=tuple(float(value) for value in vector),
            token_count=int(payload.get("token_count") or 0),
            truncated=bool(payload.get("truncated")),
            windows=int(payload.get("windows") or 1),
        )

    def put(self, fingerprint: str, text: str, embedded: EmbeddedText) -> None:
        path = self._path(fingerprint, text)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Written beside and renamed: a half-written vector read by a
            # concurrent request would be a silently wrong similarity.
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "vector": list(embedded.vector),
                        "token_count": embedded.token_count,
                        "truncated": embedded.truncated,
                        "windows": embedded.windows,
                    }
                ),
                "utf-8",
            )
            temporary.replace(path)
        except OSError:
            # A cache that cannot be written is a slower report, not a failed
            # one. The vectors are derived and reproducible by construction.
            return


class LocalTextEmbedder:
    """An already-installed HuggingFace encoder, loaded offline.

    Mean pooling over the attention mask and L2 normalization -- the recipe the
    `sentence-transformers` package applies to this checkpoint -- implemented
    against plain `transformers` because that is what this environment has.
    The pooling and normalization are published in `describe()` and folded into
    the fingerprint, so a future change to either makes old numbers visibly
    incomparable instead of quietly different.

    `local_files_only=True` on both loads is the no-download guarantee. A model
    that is not in the cache raises `EmbeddingUnavailable` carrying the command
    that would put it there; nothing here reaches the network.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_EMBEDDING_MODEL,
        *,
        window_tokens: int = DEFAULT_WINDOW_TOKENS,
        max_windows: int = DEFAULT_MAX_WINDOWS,
    ) -> None:
        self.model_id = model_id
        self.window_tokens = window_tokens
        self.max_windows = max_windows
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - both are hard deps here
            raise EmbeddingUnavailable(
                f"the local embedding stack is not importable ({exc})",
                "install this project's dependencies with `poetry install`",
            ) from exc
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_id, local_files_only=True
            )
            self._model = AutoModel.from_pretrained(model_id, local_files_only=True)
        except Exception as exc:
            raise EmbeddingUnavailable(
                f"the local embedding model {model_id!r} is not installed in "
                f"this machine's HuggingFace cache ({type(exc).__name__}: {exc})",
                "fetch it once on a machine with network access, e.g. "
                f"`python -c \"from transformers import AutoModel, AutoTokenizer; "
                f"AutoModel.from_pretrained('{model_id}'); "
                f"AutoTokenizer.from_pretrained('{model_id}')\"`; consistency "
                "similarity stays unavailable until then and no download is "
                "attempted automatically",
            ) from exc
        self._torch = torch
        self._model.eval()
        self.revision = _resolved_revision(model_id)
        self.dimension = int(self._model.config.hidden_size)
        # What this tokenizer wraps a sequence in, learned by asking it to
        # encode nothing: `[CLS] [SEP]` for a BERT family checkpoint. Asked
        # rather than hardcoded, and asked rather than reached for through
        # `build_inputs_with_special_tokens`, which transformers 5 removed.
        template = list(
            self._tokenizer("", add_special_tokens=True, verbose=False)["input_ids"]
        )
        half = len(template) // 2
        self._opening = template[:half]
        self._closing = template[half:]

    def describe(self) -> dict[str, Any]:
        return {
            "available": True,
            "model_id": self.model_id,
            "revision": self.revision,
            "dimension": self.dimension,
            "pooling": "mean_over_attention_mask",
            "normalized": True,
            "window_tokens": self.window_tokens,
            "max_windows": self.max_windows,
            "long_text_rule": LONG_TEXT_RULE,
            "aggregation": AGGREGATION_METHOD,
            "local_only": True,
            "revision_resolved": self.revision is not None,
            "fingerprint": self.fingerprint(),
        }

    def fingerprint(self) -> Optional[str]:
        """What two numbers must share before they may be subtracted.

        The aggregation method and the window size are in here alongside the
        checkpoint: two cosines computed by different long-text rules are as
        incomparable as two computed by different models, and the difference
        is far easier to miss.

        An UNRESOLVED revision does not become the string `unknown-revision`:
        it makes the fingerprint `None`, which refuses cache reuse and refuses
        cross-experiment deltas outright. A fingerprint that pins the model id
        but not the weights behind it would let two different checkpoints
        share a cache entry and have their numbers subtracted.
        """
        if self.revision is None:
            return None
        material = "\x1f".join(
            [
                self.model_id,
                self.revision,
                "mean_over_attention_mask",
                "l2",
                AGGREGATION_METHOD,
                str(self.window_tokens),
                str(self.max_windows),
                TEXT_PROJECTION_VERSION,
            ]
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    def embed(self, text: str) -> EmbeddedText:
        """The WHOLE text, in as many model-capacity windows as it takes.

        An encoder has a fixed window; a recorded plan or final answer does
        not. Embedding only the first window would compare two runs by their
        openings and call the result their similarity, which is the specific
        dishonesty this method exists to avoid. So the text is tokenized whole,
        split into consecutive non-overlapping windows, each window pooled and
        normalized exactly as a single short text would be, and the windows
        combined by a TOKEN-COUNT-WEIGHTED mean that is normalized again. A
        text that fits one window therefore gets bit-for-bit what the
        single-pass recipe gave it, and a longer one gets every part of itself
        represented in proportion to its length.

        The aggregation is deterministic -- same text, same vector -- and is
        named in the fingerprint, so a future change to it cannot be mistaken
        for a change in behaviour by anything comparing two numbers.

        `max_windows` is the only cut, and it is a work bound rather than a
        model bound: a text past it is marked `truncated`, which makes every
        similarity computed from it explicitly partial.
        """
        torch = self._torch
        # Tokenized once, without the special tokens, because the windows each
        # get their own pair and counting the text's own tokens is what makes
        # the weighting and the bound mean what they say.
        content = self._tokenizer(
            text, add_special_tokens=False, truncation=False, verbose=False
        )["input_ids"]
        token_count = len(content)
        windows = [
            content[start:start + self.window_tokens]
            for start in range(0, len(content), self.window_tokens)
        ] or [[]]
        over_bound = len(windows) > self.max_windows
        windows = windows[: self.max_windows]

        batch = self._tokenizer.pad(
            {
                "input_ids": [
                    self._opening + list(window) + self._closing
                    for window in windows
                ]
            },
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = self._model(**batch)
        hidden = output.last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

        weights = torch.tensor(
            [float(max(len(window), 1)) for window in windows], dtype=pooled.dtype
        ).unsqueeze(-1)
        combined = (pooled * weights).sum(dim=0, keepdim=True) / weights.sum()
        combined = torch.nn.functional.normalize(combined, p=2, dim=1)
        return EmbeddedText(
            vector=tuple(float(value) for value in combined[0].tolist()),
            token_count=token_count,
            truncated=over_bound,
            windows=len(windows),
        )


def _resolved_revision(model_id: str) -> Optional[str]:
    """The cached snapshot this model id resolves to, or None.

    Best effort and never fatal: the revision sharpens the fingerprint, and a
    machine whose cache layout this cannot read still gets a fingerprint -- one
    that says `unknown-revision`, which is honest about what it does not pin.
    """
    try:
        from transformers.utils import cached_file

        path = cached_file(model_id, "config.json", local_files_only=True)
    except Exception:
        return None
    return Path(path).parent.name if path else None


class _EmbeddingSession:
    """Vectors for one report: cache first, model second, counted either way."""

    def __init__(
        self, embedder: Optional[LocalTextEmbedder], cache: Optional[VectorCache]
    ) -> None:
        self.embedder = embedder
        self.cache = cache
        self.memo: dict[str, EmbeddedText] = {}
        self.cache_hits = 0
        self.computed = 0

    @property
    def available(self) -> bool:
        return self.embedder is not None

    def vector(self, text: str) -> EmbeddedText:
        if self.embedder is None:
            raise EmbeddingUnavailable(
                "no local embedding model is loaded", "see the report's remedy"
            )
        existing = self.memo.get(text)
        if existing is not None:
            return existing
        # No fingerprint, no cache: an entry keyed by an identity that does
        # not pin the weights could be handed to a different checkpoint. The
        # loader refuses such an embedder outright, so this is the belt to
        # that braces, and it degrades to recomputing rather than to guessing.
        fingerprint = self.embedder.fingerprint()
        if self.cache is not None and fingerprint is not None:
            cached = self.cache.get(fingerprint, text)
            if cached is not None and len(cached.vector) == self.embedder.dimension:
                self.cache_hits += 1
                self.memo[text] = cached
                return cached
        embedded = self.embedder.embed(text)
        self.computed += 1
        if self.cache is not None and fingerprint is not None:
            self.cache.put(fingerprint, text, embedded)
        self.memo[text] = embedded
        return embedded


# ----------------------------------------------------------------------
# The metrics
# ----------------------------------------------------------------------


def _similarity(
    session: _EmbeddingSession,
    left: ProjectedText,
    right: ProjectedText,
    unavailable: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    """One pair's cosine for one kind of text, or why there is none."""
    if not left.present or not right.present:
        missing = []
        if not left.present:
            missing.append(f"left: {left.state}")
        if not right.present:
            missing.append(f"right: {right.state}")
        return {
            "state": METRIC_UNKNOWN,
            "cosine": None,
            "reason": "; ".join(missing),
            "left_state": left.state,
            "right_state": right.state,
        }
    if not session.available:
        return {
            "state": METRIC_UNAVAILABLE,
            "cosine": None,
            "reason": (unavailable or {}).get("reason"),
            "remedy": (unavailable or {}).get("remedy"),
            "left_state": left.state,
            "right_state": right.state,
        }
    left_vector = session.vector(left.text or "")
    right_vector = session.vector(right.text or "")
    truncated = left_vector.truncated or right_vector.truncated
    bounded = left.truncated_source or right.truncated_source
    # A surviving plan whose sibling emission was withheld, or whose run has a
    # turn nobody can read, is a partial source even though its own text is
    # whole. Reporting that cosine as full coverage is exactly the claim this
    # module must not make.
    incomplete = left.partial_source or right.partial_source
    partial = truncated or bounded or incomplete
    return {
        "state": METRIC_COMPUTED,
        "cosine": cosine(left_vector.vector, right_vector.vector),
        "coverage": COVERAGE_PARTIAL if partial else COVERAGE_FULL,
        # Said in words, because "0.91" with a quiet `partial` beside it still
        # gets quoted as the similarity of the whole output.
        "coverage_note": (
            "at least one side is less than the whole of what its run "
            "recorded (some of it was withheld, bounded, unreadable or past "
            "the embedding bound), so this is not a similarity of the full "
            "recorded text"
            if partial
            else None
        ),
        "truncated_for_model": truncated,
        "source_bounded": bounded,
        "source_incomplete": incomplete,
        "left_tokens": left_vector.token_count,
        "right_tokens": right_vector.token_count,
        "left_windows": left_vector.windows,
        "right_windows": right_vector.windows,
        "left_state": left.state,
        "right_state": right.state,
    }


def _step_pair(left: RunEvidence, right: RunEvidence) -> dict[str, Any]:
    """Counts, signed delta and the normalized difference, or an explicit gap."""
    if left.step_count is None or right.step_count is None:
        return {
            "state": METRIC_UNKNOWN,
            "left": left.step_count,
            "right": right.step_count,
            "left_state": left.step_count_state,
            "right_state": right.step_count_state,
            "delta": None,
            "normalized_difference_pct": None,
            "reason": "at least one run has no known executed-step count",
            "formula": NORMALIZED_STEP_FORMULA,
        }
    partial = COUNT_PARTIAL in (left.step_count_state, right.step_count_state)
    return {
        "state": METRIC_COMPUTED,
        "left": left.step_count,
        "right": right.step_count,
        "left_state": left.step_count_state,
        "right_state": right.step_count_state,
        # Signed as RIGHT MINUS LEFT, the same direction the cross-experiment
        # deltas use (candidate minus baseline): left is the side being
        # compared against, so 2 then 4 reads "+2, the other run took two more
        # steps". abs() alone would lose which way, and flipping the direction
        # between a pair row and an experiment delta would be worse than
        # either choice on its own.
        "delta": right.step_count - left.step_count,
        "normalized_difference_pct": normalized_step_difference(
            left.step_count, right.step_count
        ),
        "coverage": COVERAGE_PARTIAL if partial else COVERAGE_FULL,
        "formula": NORMALIZED_STEP_FORMULA,
        "note": (
            "equal or near-equal counts do not imply the same commands in the "
            "same order; open the step comparison to see what ran"
        ),
    }


def _pair_row(
    session: _EmbeddingSession,
    left: RunEvidence,
    right: RunEvidence,
    unavailable: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "left_attempt": left.attempt,
        "right_attempt": right.attempt,
        "left_experiment_id": left.experiment_id,
        "right_experiment_id": right.experiment_id,
        "planning": _similarity(session, left.plan, right.plan, unavailable),
        "final_answer": _similarity(session, left.answer, right.answer, unavailable),
        "steps": _step_pair(left, right),
    }
    if left.ref is not None and right.ref is not None:
        # The identity the existing comparison and pair review already use, so
        # a click from here opens THAT pair and any comment written on it is
        # counted against the same key rather than a consistency-only one.
        row["review_pair_key"] = comparison_module.review_pair_key(left.ref, right.ref)
    else:
        row["review_pair_key"] = None
    return row


def _summarize_similarity(
    rows: Sequence[Mapping[str, Any]], kind: str, capped: bool
) -> dict[str, Any]:
    values: list[float] = []
    unknown = 0
    unavailable = 0
    partial = 0
    reasons: dict[str, int] = {}
    for row in rows:
        metric = row[kind]
        state = metric["state"]
        if state == METRIC_COMPUTED:
            values.append(float(metric["cosine"]))
            if metric.get("coverage") == COVERAGE_PARTIAL:
                partial += 1
        elif state == METRIC_UNAVAILABLE:
            unavailable += 1
        else:
            unknown += 1
            reason = metric.get("reason") or "unknown"
            reasons[reason] = reasons.get(reason, 0) + 1
    distribution = _distribution(values)
    coverage = COVERAGE_FULL
    if capped:
        coverage = COVERAGE_CAPPED
    elif partial:
        coverage = COVERAGE_PARTIAL
    return {
        "pairs_considered": len(rows),
        "pairs_computed": len(values),
        "pairs_unknown": unknown,
        "pairs_unavailable": unavailable,
        "pairs_partial_coverage": partial,
        "unknown_reasons": reasons,
        "mean": distribution["mean"],
        "min": distribution["min"],
        "max": distribution["max"],
        "spread": (
            None
            if distribution["max"] is None
            else distribution["max"] - distribution["min"]
        ),
        "spread_formula": SPREAD_FORMULA,
        "coverage": coverage,
    }


def _summarize_steps(runs: Sequence[RunEvidence]) -> dict[str, Any]:
    known = [run.step_count for run in runs if run.step_count_state == COUNT_KNOWN]
    partial = [run.attempt for run in runs if run.step_count_state == COUNT_PARTIAL]
    unknown = [run.attempt for run in runs if run.step_count_state == COUNT_UNKNOWN]
    values = [float(value) for value in known if value is not None]
    return {
        "rule": STEP_COUNT_RULE,
        "per_run": [
            {
                "attempt": run.attempt,
                "count": run.step_count,
                "state": run.step_count_state,
                "detail": dict(run.step_detail),
            }
            for run in runs
        ],
        "runs_with_known_count": len(known),
        # Named, not just counted: "3 runs have no count" sends nobody
        # anywhere, and these are exactly the runs worth opening.
        "attempts_partial": partial,
        "attempts_unknown": unknown,
        "mean": (sum(values) / len(values)) if values else None,
        "min": min(known) if known else None,
        "max": max(known) if known else None,
        "population_sd": population_sd(values),
        "sd_formula": SD_FORMULA,
    }


def _embedding_state(
    embedder: Optional[LocalTextEmbedder], unavailable: Optional[Mapping[str, Any]]
) -> dict[str, Any]:
    if embedder is not None:
        return embedder.describe()
    detail = dict(unavailable or {})
    return {
        "available": False,
        "reason": detail.get("reason", "no local embedding model was loaded"),
        "remedy": detail.get("remedy"),
        "model_id": detail.get("model_id", DEFAULT_EMBEDDING_MODEL),
        "fingerprint": None,
    }


def configured_model_id() -> str:
    """Which local checkpoint to use, overridable for an offline machine.

    An override that names a model this machine does not have is an
    `unavailable` report, not a download and not a silent fall back to the
    default -- the operator asked for a specific model and getting a different
    one's numbers under that name is worse than getting none.
    """
    override = os.environ.get("FASTWORKFLOW_CONSISTENCY_EMBEDDING_MODEL")
    return override.strip() if override and override.strip() else DEFAULT_EMBEDDING_MODEL


def load_embedder(
    model_id: Optional[str] = None,
) -> tuple[Optional[LocalTextEmbedder], Optional[dict[str, Any]]]:
    """`(embedder, None)` or `(None, why-not)`. Never raises, never downloads.

    A model that loads but whose EXACT cached revision cannot be resolved is
    treated as unavailable rather than used. Its numbers would be labelled
    with a model identity that does not pin the weights, which is what would
    let a cached vector from one checkpoint be reused for another and let two
    experiments' cosines be subtracted across a silent model change.
    """
    model_id = model_id or configured_model_id()
    try:
        embedder = LocalTextEmbedder(model_id)
    except EmbeddingUnavailable as exc:
        return None, {
            "reason": exc.reason,
            "remedy": exc.remedy,
            "model_id": model_id,
        }
    if embedder.fingerprint() is None:
        return None, {
            "reason": (
                f"the local model {model_id!r} loaded, but its exact cached "
                "revision could not be resolved, so its numbers could not be "
                "labelled with a model identity that pins the weights behind "
                "them"
            ),
            "remedy": (
                "install the model through the standard HuggingFace cache "
                "(a `snapshots/<revision>/` layout) rather than a bare "
                "directory, so the revision is readable offline; similarity "
                "stays unavailable until then and step-count consistency is "
                "unaffected"
            ),
            "model_id": model_id,
        }
    return embedder, None


_SHARED_EMBEDDERS: dict[
    str, tuple[Optional[LocalTextEmbedder], Optional[dict[str, Any]]]
] = {}


def shared_embedder(
    model_id: Optional[str] = None,
) -> tuple[Optional[LocalTextEmbedder], Optional[dict[str, Any]]]:
    """`load_embedder` once per process per model id.

    A request handler must not re-read ~90 MB of weights per page view, and the
    loaded model is immutable and read-only, so one per process is the whole
    optimization. A FAILURE is memoized too: a machine without the model would
    otherwise pay the full failed resolution on every request to say the same
    sentence. `reset_shared_embedders()` clears both for tests.
    """
    model_id = model_id or configured_model_id()
    existing = _SHARED_EMBEDDERS.get(model_id)
    if existing is None:
        existing = load_embedder(model_id)
        _SHARED_EMBEDDERS[model_id] = existing
    return existing


def reset_shared_embedders() -> None:
    """Forget the per-process embedders. For tests that change the model id."""
    _SHARED_EMBEDDERS.clear()


def task_consistency(
    *,
    experiment_id: str,
    task_id: str,
    runs: Sequence[RunEvidence],
    best_attempt: Optional[int] = None,
    embedder: Optional[LocalTextEmbedder] = None,
    embedding_unavailable: Optional[Mapping[str, Any]] = None,
    cache: Optional[VectorCache] = None,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    runs_capped: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """The whole consistency report for one task of one experiment.

    Every run given is listed, whatever its status. A run is ELIGIBLE for the
    aggregate when it has a reference at all; the per-metric eligibility is
    narrower and is reported per pair, so "8 of 10 pairs had a plan on both
    sides" is visible rather than hidden inside a mean.
    """
    session = _EmbeddingSession(embedder, cache)
    eligible = [run for run in runs if run.ref is not None]
    pairs_possible = len(eligible) * (len(eligible) - 1) // 2
    capped = pairs_possible > max_pairs

    rows: list[dict[str, Any]] = []
    for index, left in enumerate(eligible):
        for right in eligible[index + 1:]:
            if len(rows) >= max_pairs:
                break
            rows.append(_pair_row(session, left, right, embedding_unavailable))
        if len(rows) >= max_pairs:
            break

    reference = None
    reference_rows: list[dict[str, Any]] = []
    if best_attempt is not None:
        chosen = next(
            (run for run in eligible if run.attempt == best_attempt), None
        )
        if chosen is None:
            reference = {
                "kind": "best_run",
                "attempt": best_attempt,
                "usable": False,
                "reason": (
                    "the recorded best run has no readable execution, so it "
                    "cannot be the reference for these comparisons"
                ),
            }
        else:
            reference = {"kind": "best_run", "attempt": best_attempt, "usable": True}
            # Recomputed against the chosen run rather than filtered out of
            # `rows`, so the reference side is always the LEFT side and the
            # signed step delta reads "the OTHER run took N more/fewer steps
            # than the best one".
            # `rows` is untouched by this, which is what keeps the aggregate
            # independent of the choice.
            reference_rows = [
                _pair_row(session, chosen, other, embedding_unavailable)
                for other in eligible
                if other.attempt != chosen.attempt
            ]
    else:
        reference = {
            "kind": "none",
            "usable": False,
            "reason": "no best run has been chosen for this task",
        }

    insufficient = len(eligible) < 2
    payload: dict[str, Any] = {
        "experiment_id": experiment_id,
        "task_id": task_id,
        "metric_identity": {
            "metrics_version": METRICS_VERSION,
            "text_projection_version": TEXT_PROJECTION_VERSION,
            "step_count_rule": STEP_COUNT_RULE,
            "embedding": _embedding_state(embedder, embedding_unavailable),
        },
        "runs": [run.as_dict() for run in runs],
        "runs_listed": len(runs),
        "runs_eligible": len(eligible),
        # One run is one run. Reporting it as perfectly consistent would be the
        # single most misleading number this module could produce.
        "insufficient_repeats": insufficient,
        "insufficient_repeats_note": (
            "a single recorded run has nothing to be consistent with; repeat "
            "the task to get a distribution"
            if insufficient
            else None
        ),
        "pairs": rows,
        "coverage": {
            "pairs_possible": pairs_possible,
            "pairs_reported": len(rows),
            "pairs_capped": capped,
            "max_pairs": max_pairs,
            "runs_capped": bool(runs_capped),
            "runs_cap_detail": dict(runs_capped or {}),
            "vectors_computed": session.computed,
            "vectors_from_cache": session.cache_hits,
        },
        "summary": {
            "runs": len(eligible),
            "planning_similarity": _summarize_similarity(rows, "planning", capped),
            "final_answer_similarity": _summarize_similarity(
                rows, "final_answer", capped
            ),
            # Over ALL listed runs, not just the pair-eligible ones: a run
            # that finished having recorded nothing still happened, and the
            # honest place to say so is next to the distribution it is missing
            # from. The arithmetic below is over known counts either way, so
            # this widens the coverage statement without moving a number.
            "step_counts": _summarize_steps(runs),
            "independence_note": (
                "computed over every eligible pair of runs; choosing a "
                "different best run does not change it"
            ),
        },
        "reference": reference,
        "reference_rows": reference_rows,
        "interpretation": (
            "Descriptive only. Agreement between runs is consistency, not "
            "correctness: several runs can agree and all be wrong."
        ),
    }
    return payload


def compare_task_consistency(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Two experiments' consistency on the SAME task, with honest deltas.

    Refuses a numeric delta when the two sides were not measured the same way.
    Metric version, text projection and embedding fingerprint must all match;
    if they do not, the two reports are still shown side by side and the
    difference is explained instead of subtracted.
    """
    left_task = baseline.get("task_id")
    right_task = candidate.get("task_id")
    if left_task != right_task:
        raise ConsistencyError(
            f"consistency is compared within one task; {left_task!r} and "
            f"{right_task!r} are different tasks and pooling them would "
            "average two different questions"
        )

    left_identity = baseline.get("metric_identity") or {}
    right_identity = candidate.get("metric_identity") or {}
    mismatches = []
    for key in ("metrics_version", "text_projection_version", "step_count_rule"):
        if left_identity.get(key) != right_identity.get(key):
            mismatches.append(
                f"{key}: {left_identity.get(key)!r} vs {right_identity.get(key)!r}"
            )
    left_model = (left_identity.get("embedding") or {}).get("fingerprint")
    right_model = (right_identity.get("embedding") or {}).get("fingerprint")
    similarity_comparable = (
        not mismatches and bool(left_model) and left_model == right_model
    )
    if left_model != right_model:
        mismatches.append(
            f"embedding fingerprint: {left_model!r} vs {right_model!r}"
        )
    elif not left_model:
        # Both sides agree on nothing: neither has a model identity at all.
        # Equal absence is not agreement, and saying so beats a subtraction
        # whose stated reason would have been an empty string.
        mismatches.append(
            "embedding fingerprint: neither side has a resolved model "
            "identity, so their similarity figures are not comparable"
        )

    def side(report: Mapping[str, Any]) -> dict[str, Any]:
        summary = report.get("summary") or {}
        return {
            "experiment_id": report.get("experiment_id"),
            "runs": summary.get("runs"),
            "insufficient_repeats": report.get("insufficient_repeats"),
            "planning_similarity": summary.get("planning_similarity"),
            "final_answer_similarity": summary.get("final_answer_similarity"),
            "step_counts": summary.get("step_counts"),
            "coverage": report.get("coverage"),
            "metric_identity": report.get("metric_identity"),
        }

    result: dict[str, Any] = {
        "task_id": left_task,
        "baseline": side(baseline),
        "candidate": side(candidate),
        "metric_identity_matches": not mismatches,
        "mismatches": mismatches,
        "note": (
            "Descriptive comparison of two recorded populations, not a "
            "significance test. Each side's repeat count and coverage are "
            "reported so different n is visible rather than averaged away."
        ),
    }
    result["deltas"] = _deltas(baseline, candidate, similarity_comparable, mismatches)
    return result


def _deltas(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    similarity_comparable: bool,
    mismatches: Sequence[str],
) -> dict[str, Any]:
    left = baseline.get("summary") or {}
    right = candidate.get("summary") or {}

    def delta(path: str, key: str, comparable: bool) -> dict[str, Any]:
        if not comparable:
            return {
                "state": METRIC_UNKNOWN,
                "delta": None,
                "reason": (
                    "these two were not measured the same way, so the "
                    "difference of their numbers would not be a difference in "
                    "behaviour: " + "; ".join(mismatches)
                ),
            }
        left_value = (left.get(path) or {}).get(key)
        right_value = (right.get(path) or {}).get(key)
        if left_value is None or right_value is None:
            return {
                "state": METRIC_UNKNOWN,
                "delta": None,
                "baseline": left_value,
                "candidate": right_value,
                "reason": "at least one side has no value for this metric",
            }
        return {
            "state": METRIC_COMPUTED,
            "baseline": left_value,
            "candidate": right_value,
            "delta": right_value - left_value,
        }

    # Step counts need no embedding, so they survive a model mismatch; the
    # similarity deltas do not, and are refused separately rather than the
    # whole comparison being thrown away.
    steps_comparable = not [
        item for item in mismatches if not item.startswith("embedding fingerprint")
    ]
    return {
        "planning_similarity_mean": delta(
            "planning_similarity", "mean", similarity_comparable
        ),
        "final_answer_similarity_mean": delta(
            "final_answer_similarity", "mean", similarity_comparable
        ),
        "step_count_population_sd": delta(
            "step_counts", "population_sd", steps_comparable
        ),
        "step_count_mean": delta("step_counts", "mean", steps_comparable),
    }


def collect_task_evidence(
    attempts: Iterable[Mapping[str, Any]],
    reader: Any,
    *,
    ledger: Any = None,
    max_runs: int = DEFAULT_MAX_RUNS,
) -> tuple[list[RunEvidence], Optional[dict[str, Any]]]:
    """Every attempt row projected, bounded, with the bound reported.

    The cap keeps the most recent attempts, because a task repeated 60 times is
    normally being watched at its latest end -- and when it bites, the returned
    detail names exactly which attempts were left out so nobody reads the
    result as all of them.
    """
    rows = list(attempts)
    capped: Optional[dict[str, Any]] = None
    if len(rows) > max_runs:
        ordered = sorted(rows, key=lambda row: int(row["attempt"]))
        dropped = ordered[: len(ordered) - max_runs]
        rows = ordered[len(ordered) - max_runs:]
        capped = {
            "attempts_recorded": len(ordered),
            "attempts_reported": len(rows),
            "attempts_omitted": [int(row["attempt"]) for row in dropped],
            "rule": f"the {max_runs} highest attempt numbers of this task",
        }
    caching = _CachingReader(reader)
    return (
        [collect_run_evidence(row, caching, ledger=ledger) for row in rows],
        capped,
    )


__all__ = [
    "COUNT_KNOWN",
    "COUNT_PARTIAL",
    "COUNT_UNKNOWN",
    "COVERAGE_CAPPED",
    "COVERAGE_FULL",
    "COVERAGE_PARTIAL",
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_MAX_PAIRS",
    "DEFAULT_MAX_RUNS",
    "AGGREGATION_METHOD",
    "DEFAULT_MAX_WINDOWS",
    "DEFAULT_WINDOW_TOKENS",
    "LONG_TEXT_RULE",
    "METRICS_VERSION",
    "METRIC_COMPUTED",
    "METRIC_UNAVAILABLE",
    "METRIC_UNKNOWN",
    "NORMALIZED_STEP_FORMULA",
    "PLAN_SPAN_NAMES",
    "SD_FORMULA",
    "SPAN_PLANNER_PLAN",
    "SPAN_PLANNER_REPLAN",
    "SPREAD_FORMULA",
    "STEP_COUNT_RULE",
    "TEXT_ABSENT",
    "TEXT_PRESENT",
    "TEXT_PROJECTION_VERSION",
    "TEXT_UNREADABLE",
    "TEXT_WITHHELD",
    "ConsistencyError",
    "EmbeddedText",
    "EmbeddingUnavailable",
    "LocalTextEmbedder",
    "PlanSegment",
    "ProjectedText",
    "RunEvidence",
    "VectorCache",
    "canonical_plan_text",
    "collect_run_evidence",
    "collect_task_evidence",
    "compare_task_consistency",
    "configured_model_id",
    "cosine",
    "load_embedder",
    "normalized_step_difference",
    "population_sd",
    "reset_shared_embedders",
    "shared_embedder",
    "task_consistency",
]
