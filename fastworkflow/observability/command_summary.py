"""Recorded command dispatches, grouped by the name the evidence recorded.

`fix-9eg.3.1.3` and `fix-9eg.3.1.4`. A small pure reducer over the canonical
dispatch observations `comparison.project_execution` already produced, so the
page and an agent read ONE tally of one recorded execution rather than two that
can disagree.

What it will not do, because the epic's metric boundaries turn on it:

- A turn containing command X and a failure does not prove X failed. Every
  count here is per DISPATCH, keyed on `command_call_id`, and the exact
  contributing dispatches travel with the group so a reader can check the
  attribution rather than trust it.
- `response_success` -- what the turn record's `CommandOutput` recorded for that
  dispatch -- is the canonical outcome, and it is true, false or UNKNOWN. A
  dispatch nothing recorded an outcome for is not a success and not a failure.
- The execute span's own `success`/`status` is a second, best-effort fact. It
  is reported BESIDE the canonical one and, where the two disagree, the
  disagreement is listed rather than resolved: overwriting one with the other
  would delete the only evidence that the producers disagreed.
- Command names are grouped exactly as recorded. Nothing is normalised, and a
  dispatch whose name was never recorded goes to its own unknown bucket instead
  of being guessed at or dropped.
- Unreadable turns are coverage, not zeroes. A projection that could not read
  half a run says so on the summary; the counts describe what was examined.
- Timing is a second metric over the same dispatches, not a field of the first.
  It publishes min, median and max of the durations that were validly recorded
  -- no sum and no mean, because a parent dispatch's recorded duration already
  contains its children's, so anything additive over them would be an elapsed
  time these observations cannot support. Untimed is untimed, never zero, and
  the underlying per-dispatch durations stay on the contributors so a later
  merge takes the median of the durations rather than a median of medians.

The observations carry their own run/turn/step identity, so the same reducer
takes a bounded explicit list of runs later without changing shape. What is
EXPOSED today is one run per compared side; nothing here assumes that, and
nothing here goes looking for more.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

from fastworkflow.observability.comparison import (
    COVERAGE_COMPLETE,
    COVERAGE_NONE,
    COVERAGE_PARTIAL,
)

METRIC = "command_outcomes"
METRIC_VERSION = "command_outcomes/1"
UNIT = "dispatches"

# Timing is a SECOND metric over the same observations (`fix-9eg.3.1.4`), with
# its own unit, its own coverage and its own version: folding it into the
# outcome metric would make a timing change read as an outcome change.
DURATION_METRIC = "command_dispatch_duration"
DURATION_VERSION = "command_dispatch_duration/1"
DURATION_UNIT = "ns"
# A dispatch's recorded duration covers everything it did, INCLUDING the inner
# dispatches recorded beneath it. So these observations overlap, and no sum of
# them is elapsed time, critical-path time or human-wait-subtracted active
# time. No total is published here at all, and the basis travels with the
# numbers so a caller cannot acquire one by accident.
DURATION_BASIS = "inclusive_dispatch"

OUTCOME_TRUE = "true"
OUTCOME_FALSE = "false"
OUTCOME_UNKNOWN = "unknown"

# What the execute span said, kept apart from the canonical record outcome.
SPAN_OK = "ok"
SPAN_FAILED = "failed"
SPAN_UNRECORDED = "unrecorded"


@dataclass(frozen=True)
class CommandObservation:
    """One recorded dispatch, with the identity that makes it exactly one.

    `(store_id, turn_key, command_call_id)` is the identity. A ledger that
    names the same dispatch twice -- a record ref and its span, a resumed turn
    whose record re-lists a call -- is one dispatch, and counting it twice
    would invent activity that never happened.

    Everything else is quoted from the projection: `response_success` from the
    turn record's `CommandOutput`, `span_success`/`status` from the execute
    span, `duration_ns` as the span measured it.
    """

    store_id: Optional[str]
    turn_key: str
    command_call_id: str
    command_name: Optional[str] = None
    context: Optional[str] = None
    turn_index: Optional[int] = None
    position: Optional[int] = None
    parent_call_id: Optional[str] = None
    span_id: Optional[str] = None
    experiment_id: Optional[str] = None
    task_id: Optional[str] = None
    attempt: Optional[int] = None
    pass_id: Optional[str] = None
    status: Optional[str] = None
    span_success: Optional[bool] = None
    response_success: Optional[bool] = None
    duration_ns: Optional[int] = None
    in_record: bool = False
    span_recorded: bool = False
    child_call: bool = False

    @property
    def identity(self) -> tuple[Optional[str], str, str]:
        return (self.store_id, self.turn_key, self.command_call_id)

    @property
    def run_identity(self) -> tuple[Any, ...]:
        return (
            self.store_id,
            self.experiment_id,
            self.task_id,
            self.attempt,
            self.pass_id,
        )

    def as_ref(self) -> dict[str, Any]:
        """The exact contributing dispatch, addressable by a reader.

        Everything a caller needs to open this dispatch where it was recorded
        and to check the group's arithmetic against it: the run it belongs to,
        the turn and step identity, and both recorded outcomes.
        """
        return {
            "store_id": self.store_id,
            "experiment_id": self.experiment_id,
            "task_id": self.task_id,
            "attempt": self.attempt,
            "pass_id": self.pass_id,
            "turn_key": self.turn_key,
            "turn_index": self.turn_index,
            "position": self.position,
            "command_call_id": self.command_call_id,
            "parent_call_id": self.parent_call_id,
            "span_id": self.span_id,
            "command_name": self.command_name,
            "context": self.context,
            "response_success": self.response_success,
            "span_success": self.span_success,
            "status": self.status,
            "span_outcome": span_outcome(self),
            "conflict": is_conflicting(self),
            # The underlying observation, kept per dispatch rather than only as
            # the group's order statistics: a later merge of several runs must
            # take the median of the DURATIONS, never a median of medians.
            "duration_ns": recorded_duration_ns(self),
            "duration_recorded": valid_duration(self),
            # Recorded, but not usable as a duration. Distinct from nothing
            # being recorded, because a broken measurement is a fact about the
            # producer and a missing one is not.
            "duration_invalid": (
                self.duration_ns is not None and not valid_duration(self)
            ),
            "in_record": self.in_record,
            "span_recorded": self.span_recorded,
            "child_call": self.child_call,
        }


def valid_duration(observation: CommandObservation) -> bool:
    """A recorded duration that can be counted: a whole, finite, non-negative
    count of nanoseconds.

    A missing one is unknown. A negative or non-finite one is a broken
    measurement -- a clock that went backwards across a resume, an end_ns the
    producer never wrote. Neither is zero, and neither is quietly counted as
    one, because a zero in a minimum is a claim that something took no time.

    A FRACTIONAL one is refused rather than rounded. The projection computes
    `duration_ns` as `end_ns - start_ns` over integer nanoseconds, so a
    fraction did not come from the recorder; accepting one would force a choice
    between truncating it in the statistics (making the group's minimum
    disagree with the contributor it came from) and carrying a sub-nanosecond
    precision this metric does not claim. Refused, counted as invalid, and
    named in the coverage gaps.
    """
    value = observation.duration_ns
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if not math.isfinite(value):
        return False
    if isinstance(value, float) and not value.is_integer():
        return False
    return value >= 0


def recorded_duration_ns(observation: CommandObservation) -> Optional[int]:
    """The countable duration as ONE type, so no two readers of this summary
    see different numbers for the same dispatch."""
    return int(observation.duration_ns) if valid_duration(observation) else None


def span_outcome(observation: CommandObservation) -> str:
    """What the execute span says on its own, in its own words.

    `success` is the attribute the dispatch wrote; `status` is the column
    `tracing.status_for_dispatch_exception` wrote. A cancelled dispatch (an
    ask-user suspension) is not a failure, so only `error` counts as one.
    """
    if observation.span_success is True:
        return SPAN_OK
    if observation.span_success is False:
        return SPAN_FAILED
    status = (observation.status or "").strip().lower()
    if status == "error":
        return SPAN_FAILED
    if status == "ok":
        return SPAN_OK
    return SPAN_UNRECORDED


def is_conflicting(observation: CommandObservation) -> bool:
    """The record and the span disagree about this one dispatch.

    Reported, never resolved. Both facts stay on the contributor row and the
    canonical count stays the record's, because the record is the durable
    source -- but a caller that sees a conflict knows not to read either number
    as settled.
    """
    if observation.response_success is None:
        return False
    outcome = span_outcome(observation)
    if outcome == SPAN_UNRECORDED:
        return False
    return (observation.response_success is True) != (outcome == SPAN_OK)


def observations_from_projection(projection: Any) -> list[CommandObservation]:
    """The canonical dispatches of one projected side, as observations.

    Read off `projection.steps`, which is the ledger's join of the turn record
    and the trace -- the same rows the comparison aligns and the same rows a
    drill-down opens. Steps the pass selector put aside (`unassigned_steps`)
    are NOT here. That set is mixed: another pass's dispatches and dispatches
    the span tree stamped with no pass at all both land in it, and counting
    either under the selected pass is the attribution error this module exists
    to avoid. They are reported as coverage instead, without a guess about
    which kind they were.
    """
    ref = getattr(projection, "ref", None)
    store_id = getattr(ref, "store_id", None)
    return [
        CommandObservation(
            store_id=store_id,
            turn_key=step.turn_key,
            command_call_id=step.command_call_id,
            command_name=step.command_name,
            context=step.context,
            turn_index=step.turn_index,
            position=step.position,
            parent_call_id=step.parent_call_id,
            span_id=step.span_id,
            experiment_id=getattr(ref, "experiment_id", None),
            task_id=getattr(ref, "task_id", None),
            attempt=getattr(ref, "attempt", None),
            pass_id=step.pass_id or getattr(ref, "pass_id", None),
            status=step.status,
            span_success=step.success,
            response_success=step.response_success,
            duration_ns=step.duration_ns,
            in_record=step.in_record,
            span_recorded=step.span_recorded,
            child_call=step.child_call,
        )
        for step in getattr(projection, "steps", ())
    ]


def _dedupe(
    observations: Iterable[CommandObservation],
) -> tuple[list[CommandObservation], int]:
    seen: dict[tuple[Optional[str], str, str], CommandObservation] = {}
    duplicates = 0
    for observation in observations:
        if observation.identity in seen:
            duplicates += 1
            continue
        seen[observation.identity] = observation
    return list(seen.values()), duplicates


def _duration_stats(members: Sequence[CommandObservation]) -> dict[str, Any]:
    """Min, median and max over the VALID recorded durations of one group.

    Three order statistics and no fourth. There is deliberately no sum and no
    mean: a parent dispatch's recorded duration already contains its children's
    (`DURATION_BASIS`), so adding these would produce a number that looks like
    elapsed time and is not one. The order statistics are true of the
    observations themselves whether or not they overlap.

    An even count averages the two middle observations, which is the ordinary
    median and can land on a half nanosecond; it is left exact rather than
    rounded, because rounding a median is a second opinion about the data.

    `min_ns`/`median_ns`/`max_ns` are None -- never 0 -- for a group nothing
    timed, and a one-observation group is all three at once, which is correct
    and is why no minimum sample size is imposed.
    """
    # The SAME values the contributors publish, by construction: one accessor,
    # so a group's minimum can never be a truncated version of the observation
    # a reader is looking at.
    timed = sorted(
        value for value in (recorded_duration_ns(m) for m in members)
        if value is not None
    )
    # FULL identity, not the bare call id: a call id is unique only within the
    # turn that minted it, and a group already spans several turns of one run.
    # Matching on the id alone would report an identically named parent in
    # another turn -- or another store -- as an overlap inside this group.
    grouped = {member.identity for member in members}
    return {
        "metric": DURATION_METRIC,
        "metric_version": DURATION_VERSION,
        "unit": DURATION_UNIT,
        "basis": DURATION_BASIS,
        # Said as data and not only in the page's wording, because the agent
        # reading this JSON is the caller most likely to add them up.
        "additive": False,
        "timed": len(timed),
        # Never zero-filled: a dispatch nothing timed is untimed, and the
        # statistics above describe the timed ones only.
        "untimed": len(members) - len(timed),
        "invalid": sum(
            1 for m in members
            if m.duration_ns is not None and not valid_duration(m)
        ),
        "min_ns": timed[0] if timed else None,
        "median_ns": statistics.median(timed) if timed else None,
        "max_ns": timed[-1] if timed else None,
        # The overlap, counted where it is visible: dispatches in this group
        # whose parent is also in it. A parent whose child sits in another
        # group still contains that child's time, which is what `basis` says
        # and why no count here can make the numbers additive.
        "nested_within_group": sum(
            1 for m in members
            if m.parent_call_id is not None
            and (m.store_id, m.turn_key, m.parent_call_id) in grouped
        ),
        "child_dispatches": sum(1 for m in members if m.child_call),
    }


def _group(name: Optional[str], members: Sequence[CommandObservation]) -> dict[str, Any]:
    outcomes = {OUTCOME_TRUE: 0, OUTCOME_FALSE: 0, OUTCOME_UNKNOWN: 0}
    spans = {SPAN_OK: 0, SPAN_FAILED: 0, SPAN_UNRECORDED: 0}
    for member in members:
        if member.response_success is True:
            outcomes[OUTCOME_TRUE] += 1
        elif member.response_success is False:
            outcomes[OUTCOME_FALSE] += 1
        else:
            outcomes[OUTCOME_UNKNOWN] += 1
        spans[span_outcome(member)] += 1
    conflicts = [member.as_ref() for member in members if is_conflicting(member)]
    return {
        "command_name": name,
        # The bucket, named. A reader must be able to tell "no name was
        # recorded for these dispatches" from a command literally called
        # nothing.
        "unknown_command": name is None,
        "dispatches": len(members),
        "response_success": dict(outcomes),
        "span_outcome": dict(spans),
        "conflicts": conflicts,
        "conflict_count": len(conflicts),
        "coverage": {
            "in_record": sum(1 for m in members if m.in_record),
            "not_in_record": sum(1 for m in members if not m.in_record),
            "span_recorded": sum(1 for m in members if m.span_recorded),
            "without_span": sum(1 for m in members if not m.span_recorded),
            "child_calls": sum(1 for m in members if m.child_call),
        },
        "duration": _duration_stats(members),
        # EXACT members, not a filter a caller would have to re-apply. A
        # turn-level "show me this command" query would hand back every failure
        # in the turns this group touched; these are the dispatches that were
        # counted, and nothing else.
        "contributors": [member.as_ref() for member in members],
    }


def _capture_state(
    *,
    unreadable: Sequence[Any],
    examined: Optional[int],
    dispatching_turns: int,
    dispatches: int,
    without_span: int,
    not_in_record: int,
    outcomes_unrecorded: int,
    untimed_with_span: int,
    invalid_durations: int,
    outside_pass: int,
) -> tuple[str, list[str]]:
    """How completely the examined turns were RECORDED, said in their own words.

    Enumeration and capture are two questions and the second is the one that
    misleads. A run whose every named turn was readable can still be missing
    the spans, the record entries or the outcomes of the dispatches inside it,
    and reporting that as complete invites a reader to take "0 failures" for a
    finding rather than for a silence.

    A readable turn that recorded NO dispatch is a gap of the same kind: a turn
    that genuinely dispatched nothing and a turn whose dispatches were never
    captured look identical from here, so it is named rather than resolved.

    `none` when nothing was observed at all: there is no population to describe
    the capture of, and calling that complete is the specific claim this
    function exists to refuse.
    """
    gaps: list[str] = []
    if unreadable:
        gaps.append(
            f"{len(unreadable)} turn(s) named by this scope could not be read"
        )
    if examined is not None and examined > dispatching_turns:
        gaps.append(
            f"{examined - dispatching_turns} readable turn(s) recorded no "
            "dispatch at all, which a turn that dispatched nothing and a turn "
            "whose dispatches were not captured both look like"
        )
    if without_span:
        gaps.append(f"{without_span} dispatch(es) recorded no span")
    if not_in_record:
        gaps.append(f"{not_in_record} dispatch(es) are not in the turn record")
    if outcomes_unrecorded:
        gaps.append(f"{outcomes_unrecorded} dispatch(es) recorded no outcome")
    # Counted apart from the span-less dispatches above, which have no duration
    # by definition: this is a dispatch the trace DID record and still did not
    # time, which is a different gap in the same evidence.
    if untimed_with_span:
        gaps.append(
            f"{untimed_with_span} recorded span(s) carry no usable duration"
        )
    if invalid_durations:
        gaps.append(
            f"{invalid_durations} dispatch(es) recorded a duration that is not "
            "usable (negative or not finite) and are counted as untimed"
        )
    if outside_pass:
        # NOT "belong to another pass". `project_execution` sets aside every
        # step whose resolved pass id differs from the selected one, and a step
        # the span tree stamped with NO pass resolves to None -- so this set is
        # other passes AND unattributed dispatches mixed together. Which one a
        # given step is cannot be read off the count, and inferring it would be
        # the confident wrong attribution the pass machinery exists to avoid.
        gaps.append(
            f"{outside_pass} dispatch(es) are not attributed to this selected "
            "pass -- another recorded pass, or no recorded pass at all -- and "
            "are not counted here"
        )
    if not dispatches:
        return COVERAGE_NONE, gaps
    return (COVERAGE_PARTIAL if gaps else COVERAGE_COMPLETE), gaps


def summarize_commands(
    observations: Iterable[CommandObservation],
    *,
    requested_scope: Optional[Mapping[str, Any]] = None,
    coverage: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Group recorded dispatches by their recorded command name.

    `requested_scope` is what the caller asked for; the observed scope is
    derived from the observations themselves, so a summary always says which
    runs it actually saw rather than which one it was assumed to be about.

    `coverage` carries what the caller knows and the observations cannot: turns
    that could not be read, steps a pass selector put aside. Merged into the
    summary's coverage so partial evidence stays visible next to the counts.
    """
    members, duplicates = _dedupe(observations)
    by_name: dict[Optional[str], list[CommandObservation]] = {}
    for member in members:
        by_name.setdefault(member.command_name, []).append(member)

    groups = [_group(name, rows) for name, rows in by_name.items()]
    # Busiest first, then by name, with the unknown bucket last so it never
    # sorts as though it were a command called "".
    groups.sort(
        key=lambda group: (
            -group["dispatches"],
            group["command_name"] is None,
            group["command_name"] or "",
        )
    )

    totals = {OUTCOME_TRUE: 0, OUTCOME_FALSE: 0, OUTCOME_UNKNOWN: 0}
    for member in members:
        if member.response_success is True:
            totals[OUTCOME_TRUE] += 1
        elif member.response_success is False:
            totals[OUTCOME_FALSE] += 1
        else:
            totals[OUTCOME_UNKNOWN] += 1

    observed_runs = sorted(
        {member.run_identity for member in members},
        key=lambda identity: tuple("" if part is None else str(part) for part in identity),
    )
    extra = dict(coverage or {})
    unreadable = list(extra.pop("unreadable_turns", []) or [])
    outside_pass = int(extra.pop("steps_outside_pass", 0) or 0)
    raw_examined = extra.pop("turns_examined", None)
    dispatching_turns = len({member.turn_key for member in members})
    # The caller's count of readable turns where it has one. Without it the
    # examined population is unknown rather than assumed to be the turns that
    # happened to dispatch something.
    examined = None if raw_examined is None else int(raw_examined)
    timed = sum(1 for member in members if valid_duration(member))
    invalid_durations = sum(
        1 for member in members
        if member.duration_ns is not None and not valid_duration(member)
    )
    without_span = sum(1 for member in members if not member.span_recorded)
    not_in_record = sum(1 for member in members if not member.in_record)
    capture, gaps = _capture_state(
        unreadable=unreadable,
        examined=examined,
        dispatching_turns=dispatching_turns,
        dispatches=len(members),
        without_span=without_span,
        not_in_record=not_in_record,
        outcomes_unrecorded=totals[OUTCOME_UNKNOWN],
        untimed_with_span=sum(
            1 for member in members
            if member.span_recorded and not valid_duration(member)
        ),
        invalid_durations=invalid_durations,
        outside_pass=outside_pass,
    )

    return {
        "metric": METRIC,
        "metric_version": METRIC_VERSION,
        "unit": UNIT,
        "scope": {
            "requested": dict(requested_scope or {}),
            "observed": {
                "runs": [
                    {
                        "store_id": identity[0],
                        "experiment_id": identity[1],
                        "task_id": identity[2],
                        "attempt": identity[3],
                        "pass_id": identity[4],
                    }
                    for identity in observed_runs
                ],
                "run_count": len(observed_runs),
                "turns": examined if examined is not None else dispatching_turns,
            },
        },
        "groups": groups,
        "totals": {
            "dispatches": len(members),
            "groups": len(groups),
            "response_success": dict(totals),
            "unknown_command_dispatches": len(by_name.get(None, [])),
            "conflicts": sum(group["conflict_count"] for group in groups),
            # Timing coverage for the side as a whole. No min/median/max here:
            # a figure across unlike commands answers no question anyone asked,
            # and publishing one invites exactly the cross-command comparison
            # the groups exist to keep separate.
            "duration": {
                "metric": DURATION_METRIC,
                "metric_version": DURATION_VERSION,
                "unit": DURATION_UNIT,
                "basis": DURATION_BASIS,
                "additive": False,
                "timed": timed,
                "untimed": len(members) - timed,
                "invalid": invalid_durations,
            },
        },
        "coverage": {
            # ENUMERATION: were all the turns this scope named readable at all.
            # False whenever one was not, in which case the counts above are
            # about what WAS read -- a zero in them is a zero among the
            # examined dispatches and never a claim about the missing turns.
            "enumeration_complete": not unreadable,
            # Readable turns, INCLUDING the ones that recorded no dispatch: a
            # turn that answered without dispatching anything was still
            # examined, and counting only the dispatch-bearing ones would
            # shrink the examined population to the part that had activity.
            "turns_examined": examined,
            "turns_with_dispatches": dispatching_turns,
            "turns_without_dispatches": (
                None if examined is None else examined - dispatching_turns
            ),
            "turns_unreadable": len(unreadable),
            "unreadable_turns": unreadable,
            # Dispatches the projection did not attribute to the SELECTED pass:
            # some belong to another recorded pass, some were never stamped
            # with one at all, and this count does not claim to tell them
            # apart. Excluded from every count above, and said so rather than
            # silently dropped.
            "steps_outside_pass": outside_pass,
            "duplicate_observations": duplicates,
            # CAPTURE: a separate question from enumeration, because reading
            # every turn does not mean every dispatch in them was fully
            # recorded. Nothing readable is what `none` means; `partial` is
            # anything below.
            "capture": capture,
            "capture_gaps": gaps,
            "dispatches_without_span": without_span,
            "dispatches_not_in_record": not_in_record,
            "outcomes_unrecorded": totals[OUTCOME_UNKNOWN],
            **extra,
        },
    }


def summarize_projection(
    projection: Any, *, requested_scope: Optional[Mapping[str, Any]] = None
) -> dict[str, Any]:
    """The command summary of one already-projected side.

    Derived from the projection the caller already has -- no second read, no
    new query and no scan. The requested scope defaults to the reference's own
    identity, which is the one run this side is.
    """
    ref = getattr(projection, "ref", None)
    unavailable = list(getattr(projection, "unavailable", ()) or ())
    scope = dict(requested_scope or {})
    if not scope and ref is not None:
        scope = {
            "store_id": getattr(ref, "store_id", None),
            "experiment_id": getattr(ref, "experiment_id", None),
            "task_id": getattr(ref, "task_id", None),
            "attempt": getattr(ref, "attempt", None),
            "pass_id": getattr(ref, "pass_id", None),
            "turns": list(getattr(ref, "turn_keys", ()) or ()),
        }
    return summarize_commands(
        observations_from_projection(projection),
        requested_scope=scope,
        coverage={
            "unreadable_turns": unavailable,
            # The turns that WERE read, whether or not they dispatched
            # anything. `observations_from_projection` can only see the ones
            # that did.
            "turns_examined": len(getattr(projection, "turns", ()) or ()),
            "steps_outside_pass": len(getattr(projection, "unassigned_steps", ()) or ()),
        },
    )


__all__ = [
    "METRIC",
    "METRIC_VERSION",
    "UNIT",
    "COVERAGE_COMPLETE",
    "COVERAGE_NONE",
    "COVERAGE_PARTIAL",
    "DURATION_BASIS",
    "DURATION_METRIC",
    "DURATION_UNIT",
    "DURATION_VERSION",
    "OUTCOME_TRUE",
    "OUTCOME_FALSE",
    "OUTCOME_UNKNOWN",
    "SPAN_OK",
    "SPAN_FAILED",
    "SPAN_UNRECORDED",
    "CommandObservation",
    "is_conflicting",
    "observations_from_projection",
    "recorded_duration_ns",
    "span_outcome",
    "summarize_commands",
    "summarize_projection",
    "valid_duration",
]
