"""Answer-time rehydration: the extract step reads the evidence, not the pointers.

The ReAct loop's trajectory is a *working* surface: compaction swaps a 3 KB
listing for a 250 B offload label, which is what keeps the agent's peak prompt
small enough to finish long runs.

The extract step is a different reader with a different job. It has no tools,
it runs once, and ``trajectory`` is its only evidence input -- so at answer time
the same compaction that helped the loop is what leaves the writer holding
pointers, answering a deliverable slot with "see Observation O23" about a
listing sitting in that turn's own archive.

So immediately before the extract call, and only there, this module builds the
extractor's OWN copy of the trajectory with the evidence put back: an **offload
label** becomes the raw archived observation it names, printed with its alias
line and the context clause recorded for that alias, exactly as the agent first
saw it. Nothing else in the trajectory is rewritten; an observation whose full
text is still inline has lost nothing and is left alone.

Rules this module does not bend:

* **Nothing is chosen by a model and nothing is invented.** Every byte added here
  was produced by a command in this turn and stored under a digest.
* **The ReAct loop's own trajectory object is never touched.** ``rehydrate``
  works on a copy; the archive and the handle store are opened read-only.
* **The budget is a hard bound.** Most recent first, stop at the first
  replacement that would not fit, and say which aliases were left as pointers so
  the extractor can name them unresolved instead of guessing.

Rehydration is what the extract step does, for every workflow; there is no flag
to turn it off. The budget is derived from the model's context window
(``fastworkflow.context_budget``); ``FW_ANSWER_REHYDRATION_MAX_BYTES`` remains
as a tuning override.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from fastworkflow import context_budget
from fastworkflow.observation_offloading.archive import (
    RuntimeHandleArchive,
    RuntimeHandleScope,
)
from fastworkflow.observation_offloading.compact import record_foreign_line
from fastworkflow.observation_offloading.labels import (
    annotated_observation,
    is_offload_label,
    label_alias,
    owns_line,
    printed_alias,
)
from fastworkflow.observation_offloading.state import (
    context_clause_of,
    default_scope,
    stored_handles,
)

logger = logging.getLogger(__name__)

#: The tuning override for the extraction byte budget, in UTF-8 bytes of the
#: whole extractor trajectory. The budget itself is a fraction of the model's
#: context window (``context_budget.ANSWER_REHYDRATION``).
ANSWER_REHYDRATION_MAX_BYTES_ENV = context_budget.ANSWER_REHYDRATION.override_env
#: ~250 KB of UTF-8 at the reference window -- about the 80k-token answer-time
#: prompt measured on real answering runs. It is a ceiling,
#: not a target: a run whose evidence is smaller produces a smaller prompt.
DEFAULT_MAX_BYTES = context_budget.REFERENCE_ANSWER_REHYDRATION_MAX_BYTES
#: Below this a budget could not hold one page of evidence, so it is refused and
#: the derived budget stands rather than silently producing a pointer-only prompt.
MIN_MAX_BYTES = context_budget.ANSWER_REHYDRATION.floor

#: The key the drop line is appended under. Deliberately not an observation key:
#: it is a statement about the trajectory, not a tool result, and the extractor
#: must never read it as evidence about the workflow.
NOT_REHYDRATED_KEY = "answer_rehydration_note"
NOT_REHYDRATED_PREFIX = (
    "Not rehydrated for the answer (evidence exists under these observations): "
)

KIND_LABEL = "label"        # (a)


# ---------------------------------------------------------------------------
# The budget
# ---------------------------------------------------------------------------

def max_bytes_from_env() -> int:
    """The effective extraction byte budget for this run.

    A fraction of the model's context window, unless
    ``FW_ANSWER_REHYDRATION_MAX_BYTES`` overrides it. See
    ``fastworkflow.context_budget``.
    """
    return context_budget.answer_rehydration_max_bytes()


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

@dataclass
class RehydrationReport:
    """What the extractor's copy gained, and what it did not."""

    budget_bytes: int = 0
    bytes_before: int = 0
    bytes_after: int = 0
    counts: dict[str, int] = field(default_factory=lambda: {KIND_LABEL: 0})
    rehydrated: list[dict[str, Any]] = field(default_factory=list)
    dropped_aliases: list[str] = field(default_factory=list)
    unresolved_aliases: list[str] = field(default_factory=list)
    note_line: str = ""
    stopped_on: str = ""

    @property
    def bytes_added(self) -> int:
        return self.bytes_after - self.bytes_before

    def as_event(self) -> dict[str, Any]:
        return {
            "budget_bytes": self.budget_bytes,
            "bytes_before": self.bytes_before,
            "bytes_after": self.bytes_after,
            "bytes_added": self.bytes_added,
            "rehydrated_total": len(self.rehydrated),
            "rehydrated_labels": self.counts[KIND_LABEL],
            "dropped_aliases": list(self.dropped_aliases),
            "unresolved_aliases": list(self.unresolved_aliases),
            "stopped_on": self.stopped_on,
            "aliases": [
                {"alias": item["alias"], "kind": item["kind"],
                 "added_bytes": item["added_bytes"]}
                for item in self.rehydrated
            ],
        }


# ---------------------------------------------------------------------------
# Measuring
# ---------------------------------------------------------------------------

def trajectory_bytes(trajectory: Mapping[str, Any]) -> int:
    """UTF-8 bytes of every value in the trajectory.

    The same quantity for the copy and the original, which is all a budget needs
    to be comparable. It is deliberately not the formatted prompt: the adapter's
    field framing is a constant per key and would make the budget depend on the
    adapter in force rather than on the evidence.
    """
    return sum(len(str(value).encode("utf-8")) for value in trajectory.values())


def _alias_ordinal(alias: str) -> int:
    try:
        return int(str(alias)[1:])
    except (TypeError, ValueError):
        return 0


def _step_indexes(trajectory: Mapping[str, Any]) -> list[int]:
    indexes: set[int] = set()
    for key in trajectory:
        name = str(key)
        if not name.startswith("observation_"):
            continue
        suffix = name.removeprefix("observation_")
        if suffix.isdigit():
            indexes.add(int(suffix))
    return sorted(indexes)


def _candidates(
    trajectory: Mapping[str, Any],
    executes: Optional[list[tuple[int, int]]] = None,
    scope: Optional[RuntimeHandleScope] = None,
) -> list[tuple[int, str, str]]:
    """``(step_index, alias, text)`` for every aliased execute observation.

    Most recent first. The alias comes off the observation itself -- the printed
    alias line, or the label's own alias -- so this never has to recompute execute
    ordinals or know how many steps the loop truncated away. An execute step with
    no alias on it (an error string, a refusal) is not a candidate: there is
    nothing stored to put back.

    When the agent's execute ledger is given (``executes``), a step whose line
    names any alias but its own is not a candidate either: on a step that was
    never annotated that line is the backend's text, and resolving it would put
    another observation's evidence in this step's place. Without the ledger the
    alias read off the text is taken as it stands.
    """
    ledger = {index: f"O{ordinal}" for index, ordinal in (executes or [])}
    found: list[tuple[int, str, str]] = []
    for index in _step_indexes(trajectory):
        if str(trajectory.get(f"tool_name_{index}") or "") != "execute_workflow_query":
            continue
        text = trajectory.get(f"observation_{index}")
        if not isinstance(text, str) or not text:
            continue
        alias = label_alias(text) if is_offload_label(text) else printed_alias(text)
        if not alias:
            continue
        expected = ledger.get(index)
        if expected is not None and not owns_line(text, expected):
            record_foreign_line("rehydration", index, text, expected, scope)
            continue
        found.append((index, alias, text))
    found.reverse()
    return found


# ---------------------------------------------------------------------------
# (a) The archived observation behind a label
# ---------------------------------------------------------------------------

def archived_observation(
    alias: str,
    *,
    scope: RuntimeHandleScope,
    archive: RuntimeHandleArchive,
) -> Optional[str]:
    """The raw archived text for *alias*, hot cache first, then SQLite.

    The same resolution ``search_memory`` uses, for the same reason: both tiers
    hold the identical digest-verified bytes, and the hot one saves a read.
    """
    handle = stored_handles(scope).get(alias)
    if handle is None:
        try:
            handle = archive.get(scope, alias)
        except Exception:  # noqa: BLE001 - a read failure is a miss, never a turn failure
            logger.debug("answer rehydration could not read %s", alias, exc_info=True)
            return None
    if handle is None:
        return None
    text = handle.get("text")
    return text if isinstance(text, str) else None


def rehydrated_label(
    alias: str,
    *,
    scope: RuntimeHandleScope,
    archive: RuntimeHandleArchive,
) -> Optional[str]:
    """The label's observation, re-printed exactly as the agent first saw it.

    The archive stores the command response WITHOUT the presentation line, so
    the line is rebuilt here from the alias and the context clause recorded for
    it at dispatch. An alias with no recorded clause prints the plain alias
    line: the clause is presentation and its absence is never guessed at.

    ``annotated_observation`` joins the two, exactly as the compaction hook did
    when the step completed, so a response whose own first line is shaped like
    an alias line is quoted here too and reads back as the same response.
    """
    text = archived_observation(alias, scope=scope, archive=archive)
    if text is None:
        return None
    return annotated_observation(
        alias,
        # The archive that holds the text also holds the subject recorded
        # for it (ido-dhw), so a label rehydrated in a process that never
        # ran the turn prints the same clause the agent first saw.
        context_clause_of(scope, alias, selected_archive=archive) or "",
        text,
    )


# ---------------------------------------------------------------------------
# (b) and (c) The stored rows behind a result handle
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------

def rehydrate(
    trajectory: Mapping[str, Any],
    *,
    scope: Optional[RuntimeHandleScope] = None,
    archive: Optional[RuntimeHandleArchive] = None,
    budget: Optional[int] = None,
    executes: Optional[list[tuple[int, int]]] = None,
) -> tuple[dict[str, Any], RehydrationReport]:
    """The extractor's copy of *trajectory*, with the evidence behind it put back.

    ``executes`` is the agent's own ``(step_index, ordinal)`` ledger. With it, a
    line on an execute step that names any alias but that step's own is never
    resolved (see ``_candidates``); without it behaviour is unchanged.

    Returns ``(trajectory_copy, report)``. The input mapping is never mutated:
    the copy is what the extract call receives, so the ReAct loop keeps the
    trajectory it ran on and the turn record is unchanged.

    Order is most recent first, and the walk STOPS at the first replacement that
    would take the copy over ``budget``. Everything older than that stop stays as
    it was -- still a label -- and every alias left that way is named in one
    deterministic line appended to the copy, so the extractor can report those
    slots as unresolved instead of inventing them.
    """
    selected_scope = scope or default_scope()
    if archive is None:
        from fastworkflow.observation_offloading import state as offload_state

        archive = offload_state.archive()
    budget_bytes = int(budget if budget is not None else max_bytes_from_env())

    copy: dict[str, Any] = dict(trajectory)
    report = RehydrationReport(budget_bytes=budget_bytes)
    report.bytes_before = trajectory_bytes(trajectory)
    used = report.bytes_before
    seen_labels: set[str] = set()
    candidates = _candidates(trajectory, executes, selected_scope)
    stopped = False

    for position, (index, alias, text) in enumerate(candidates):
        if stopped:
            # ``ido-1tu``/F34. The note says evidence EXISTS under these
            # observations and was not put back, so only an alias that had
            # something to put back belongs in it. A plain inline observation
            # whose full text is already in the trajectory lost nothing to the
            # stop, and listing it would tell the extractor to treat present
            # evidence as unresolved.
            if _evidence_behind(
                alias, text, scope=selected_scope, archive=archive,
                report=report, seen_labels=seen_labels,
            ):
                report.dropped_aliases.append(alias)
            continue
        kind = ""
        # (a) The label, if this observation is one. ``base`` is the text the
        # rows are appended to below: the archived observation for a label, the
        # observation's own text otherwise.
        base = text
        if is_offload_label(text):
            if alias in seen_labels:
                # ``ido-1tu``/F34. One alias names one archived observation,
                # however many steps print its label. A more recent step
                # already carries that text in full, so restoring it again
                # would spend the budget twice on bytes the extractor is
                # holding -- the rule ``seen_blocks`` applies to a handle's
                # rows, applied to a label.
                continue
            restored = rehydrated_label(
                alias, scope=selected_scope, archive=archive
            )
            if restored is None:
                report.unresolved_aliases.append(alias)
                continue
            kind = KIND_LABEL
            base = restored

        if not kind:
            # Not an offload label: nothing to put back.
            continue
        replacement: str = base

        text_bytes = len(text.encode("utf-8"))
        added = len(replacement.encode("utf-8")) - text_bytes
        if used + added > budget_bytes:
            stopped = True
            report.stopped_on = alias
            report.dropped_aliases.append(alias)
            continue
        copy[f"observation_{index}"] = replacement
        used += added
        report.counts[kind] += 1
        if kind == KIND_LABEL:
            # ``ido-1tu``/F34. This alias's archived text is now in the copy;
            # an older step printing the same label needs nothing further.
            seen_labels.add(alias)
        report.rehydrated.append({
            "alias": alias,
            "kind": kind,
            "step_index": index,
            "recency_rank": position,
            "added_bytes": added,
            "utf8_bytes": len(replacement.encode("utf-8")),
        })

    clauses: list[str] = []
    if report.dropped_aliases:
        report.dropped_aliases.sort(key=_alias_ordinal)
        clauses.append(NOT_REHYDRATED_PREFIX + ", ".join(report.dropped_aliases))
    if clauses:
        report.note_line = " ".join(clauses)
        copy[NOT_REHYDRATED_KEY] = report.note_line
    report.bytes_after = trajectory_bytes(copy)
    return copy, report


def _evidence_behind(
    alias: str,
    text: str,
    *,
    scope: RuntimeHandleScope,
    archive: RuntimeHandleArchive,
    report: RehydrationReport,
    seen_labels: set[str],
) -> bool:
    """Would the walk have put anything back for this observation?

    Asked only after the budget stopped the walk, and it answers exactly what
    the walk above would have done for the same observation, so the dropped
    list names the aliases that really lost evidence and nothing else:

    * an offload label counts when the archive still holds its text, and not
      when a more recent step already restored the same alias (``seen_labels``)
      or the archive cannot be read -- an alias with nothing behind it is
      ``unresolved``, not dropped, and the note's "evidence exists" would be
      untrue of it;
    * any other observation counts for nothing: its whole text is in the
      trajectory the extractor is reading.
    """
    if is_offload_label(text):
        if alias in seen_labels:
            return False
        if archived_observation(alias, scope=scope, archive=archive) is None:
            report.unresolved_aliases.append(alias)
            return False
        return True
    return False


__all__ = [
    "ANSWER_REHYDRATION_MAX_BYTES_ENV",
    "DEFAULT_MAX_BYTES",
    "KIND_LABEL",
    "MIN_MAX_BYTES",
    "NOT_REHYDRATED_KEY",
    "NOT_REHYDRATED_PREFIX",
    "RehydrationReport",
    "archived_observation",
    "max_bytes_from_env",
    "rehydrate",
    "rehydrated_label",
    "trajectory_bytes",
]
