"""Distil a student run against a recorded teacher run.

The live teacher/student double pass in `distillation.py` welds both models into
one process, so there is no way to get a teacher right and only then distil
against it. This module separates them: run a teacher, iterate on it, then run a
student and compare the two recordings.

Both sides are rebuilt from observability spans by `distillation_trace`, never
from live in-memory state. The judge is a diff judge, and a live trajectory
compared against a rebuilt one would differ in formatting for reasons that have
nothing to do with the agents — differences the judge would then blame on the
student. Rebuilding both through one code path also makes a comparison
re-runnable over a recorded pair, which is what lets the judge prompts be tuned
without re-running any agent.

The comparison unit is a whole task, not a turn. `compare_trajectories` takes
flat action lists and is not turn-scoped; per-turn comparison would additionally
require turn N to mean the same thing on both sides, which stops being true as
soon as the two conversations diverge.

Nothing here is specific to any harness: ``experiment_id`` and ``task_id`` are
opaque strings and the task list is discovered from the store.

The comparison logic and judge prompts are `distillation.py`'s, called
unchanged. This module decides only *what to compare* and *what to do with the
result*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import dspy

from fastworkflow import distillation_trace as dtrace
from fastworkflow.distillation import DistillationSession
from fastworkflow.observability import feedback as obs_feedback
from fastworkflow.observability.store import FEEDBACK_COMMENT_MAX_CHARS
from fastworkflow.utils import dspy_utils
from fastworkflow.utils.logging import logger

#: Written on every feedback row this module produces. Already an accepted
#: value in the store's ``FEEDBACK_PROVENANCES``.
PROVENANCE = "distillation_agent"

#: The judge's decline, recorded as the analysis when it extracted no rule and
#: left no reasoning. A divergence the judge assessed as context-justified is
#: different evidence from one nobody looked at, and only the annotation can
#: tell them apart later.
_NO_RULE_NOTE = (
    "(no rule extracted — the judge assessed this difference as justified by "
    "context rather than a student mistake)"
)

#: What went wrong, when the narrative call failed or returned nothing.
_NARRATIVE_UNAVAILABLE = (
    "(the narrative could not be generated; the raw divergence is recorded "
    "under Observations / Analysis · Observation)"
)


def _fit_comment(text: str) -> str:
    """*text* within the store's per-note limit, saying so when it was cut.

    A divergence summary lists every differing action with its parameters, so a
    long run can exceed the limit, and the store would reject the whole note.
    Keeping the head with an explicit marker loses less than losing the note.
    """
    text = (text or "").strip()
    if len(text) <= FEEDBACK_COMMENT_MAX_CHARS:
        return text
    marker = f"\n\n[truncated: first {{shown:,}} of {len(text):,} characters shown]"
    shown = FEEDBACK_COMMENT_MAX_CHARS - len(marker.format(shown=FEEDBACK_COMMENT_MAX_CHARS))
    return text[:shown].rstrip() + marker.format(shown=shown)


class DivergenceNarrativeSignature(dspy.Signature):
    """Explain, to a developer reading a trace, how a student agent's run
differed from a teacher agent's run on the same task.

You are given a mechanically-produced divergence summary: a literal diff of the
two runs' actions or plans. It is complete but unreadable — long list dumps,
every step, no interpretation. Your job is to say what actually happened.

IMPORTANT — the two conversations are not identical. Both agents were given the
same opening message, but the simulated user then responded to each agent
separately, so from the second turn on they may have been asked different
things. Read both conversations before attributing a difference to the student.
A student that did something else because it was asked something else has not
made a mistake.

Write two things:

what_went_wrong: 2-4 sentences naming the substantive difference in the
student's behaviour. Lead with the difference that matters most. Name specific
commands. State the consequence where there is one (a wrong order of calls, a
missing lookup, redundant work after the task was already done). Do not quote
the raw dump back; do not restate every step; do not speculate beyond the
evidence. If the difference is cosmetic, or explained by the two agents being
asked different things, say that plainly instead of inventing a fault.

what_worked: 1-2 sentences on what the student did correctly, where there is
something real to report. Both agents reaching the same outcome, or following
the same sequence up to a point, is worth saying. Return EMPTY if the student
did nothing notably right.

Write for someone who will read this next to the trace it describes. No
preamble, no headings, no bullet points.
    """

    task_opening: str = dspy.InputField(
        desc="The task's opening user message — identical for both agents"
    )
    teacher_conversation: str = dspy.InputField(
        desc="Every user message the teacher received, in order"
    )
    student_conversation: str = dspy.InputField(
        desc="Every user message the student received, in order. May diverge from "
        "the teacher's after the opening."
    )
    divergence_kind: str = dspy.InputField(
        desc="'executed actions' or 'plans' — what the summary is comparing"
    )
    divergence_summary: str = dspy.InputField(
        desc="The mechanical diff of teacher vs student"
    )
    extracted_insights: str = dspy.InputField(
        desc="Rules already extracted from this divergence, or empty if the "
        "judge assessed it as context-justified. Do not repeat them; they are "
        "shown to the reader separately."
    )
    what_went_wrong: str = dspy.OutputField(desc="2-4 sentences, prose")
    what_worked: str = dspy.OutputField(desc="1-2 sentences, or EMPTY")


def _conversation(attempt: dtrace.ReconstructedAttempt) -> str:
    """Every user message of an attempt, numbered, oldest first."""
    return "\n".join(
        f"[{i}] {turn.user_message}"
        for i, turn in enumerate(attempt.turns)
        if turn.user_message
    )


def _narrate(
    *,
    teacher: dtrace.ReconstructedAttempt,
    student: dtrace.ReconstructedAttempt,
    kind: str,
    divergence_summary: str,
    insights: list[str],
) -> tuple[str, str]:
    """Turn a mechanical diff into prose for the annotation.

    Separate from the judge signatures on purpose: those produce rules that feed
    the agent's prompt and must not shift, this produces prose for a person.
    Different consumer, different prompt, and this one can fail without costing
    the insights.

    Both conversations go in because they diverge after the opening message —
    without the student's, a difference caused by a different question reads as
    a student mistake.

    On any failure, ``what_went_wrong`` says the narrative is unavailable rather
    than repeating the raw summary: the summary is already recorded as its own
    note (Observations / Analysis · Observation), so nothing is lost and nothing
    is shown twice.
    """
    try:
        lm = dspy_utils.get_lm("LLM_DISTILLATION", "LITELLM_API_KEY_DISTILLATION")
        with dspy.context(lm=lm):
            result = dspy.ChainOfThought(DivergenceNarrativeSignature)(
                task_opening=teacher.first_user_message or student.first_user_message,
                teacher_conversation=_conversation(teacher),
                student_conversation=_conversation(student),
                divergence_kind=kind,
                divergence_summary=divergence_summary,
                extracted_insights="\n".join(insights),
            )
        went_wrong = (result.what_went_wrong or "").strip()
        worked = (result.what_worked or "").strip()
        if worked.upper() == "EMPTY":
            worked = ""
        return went_wrong or _NARRATIVE_UNAVAILABLE, worked
    except Exception as exc:
        logger.warning(
            f"Could not narrate the {kind} divergence "
            f"({type(exc).__name__}: {exc}); the raw diff is still recorded"
        )
        return _NARRATIVE_UNAVAILABLE, ""


@dataclass
class TaskComparison:
    """What one task's teacher-vs-student comparison found."""

    task_id: str
    teacher_attempt: int
    student_attempt: int
    execution_diverged: bool = False
    execution_summary: str = ""
    planning_diverged: bool = False
    planning_summary: str = ""
    execution_insights: list[str] = field(default_factory=list)
    planning_insights: list[str] = field(default_factory=list)
    shared_actions: list[str] = field(default_factory=list)
    feedback_written: int = 0
    skipped_reason: Optional[str] = None
    # Which runs the pair came from, for pool/self distillation (compare_pool):
    # the failing trial's run, the reference's run, and whether the reference
    # is a passing trial of the same model ("self") or of a teacher ("teacher").
    # Empty for a plain teacher-vs-student comparison.
    student_experiment: str = ""
    reference_experiment: str = ""
    reference_source: str = ""
    # Each side's graded outcome (True pass, False fail, None ungraded), for the
    # report and provenance: teacher/student pairs are chosen without looking at
    # outcomes, so a passing student trial or a failing reference stays visible.
    student_passed: Optional[bool] = None
    reference_passed: Optional[bool] = None

    @property
    def diverged(self) -> bool:
        return self.execution_diverged or self.planning_diverged

    @property
    def insight_count(self) -> int:
        return len(self.execution_insights) + len(self.planning_insights)


def _action_names(actions: list[dict[str, Any]]) -> list[str]:
    return [a.get("command_name", "") for a in actions if a.get("command_name")]


def _shared_actions(
    teacher: list[dict[str, Any]], student: list[dict[str, Any]]
) -> list[str]:
    """Commands both sides ran — the 'what worked' half of the feedback."""
    student_names = set(_action_names(student))
    seen: list[str] = []
    for name in _action_names(teacher):
        if name in student_names and name not in seen:
            seen.append(name)
    return seen


def _origin_note(
    teacher_experiment_id: str,
    teacher: dtrace.ReconstructedAttempt,
    student: dtrace.ReconstructedAttempt,
    source: str = "",
) -> str:
    """Names the compared pair, so a feedback row is self-describing."""
    sizes = f"({len(teacher.turns)} turns there vs {len(student.turns)} here)."
    if source == "self" or (not source and teacher_experiment_id == student.experiment_id):
        where = (
            "in the same run" if teacher_experiment_id == student.experiment_id
            else f"in run {teacher_experiment_id}"
        )
        return (
            f"Self-distillation: this failing trial (attempt {student.attempt}) was "
            f"compared with passing trial {teacher.attempt} of the same model on "
            f"task {teacher.task_id}, {where} {sizes}"
        )
    if source == "teacher":
        return (
            f"Teacher distillation: this failing trial was compared with a passing "
            f"teacher trial - run {teacher_experiment_id}, task {teacher.task_id} "
            f"attempt {teacher.attempt} {sizes}"
        )
    return (
        f"Distillation: this run diverged from teacher run {teacher_experiment_id}, "
        f"task {teacher.task_id} attempt {teacher.attempt} "
        f"({len(teacher.turns)} teacher turns vs {len(student.turns)} here)."
    )


def _anchor(
    student: dtrace.ReconstructedAttempt, phase: str
) -> tuple[Optional[str], list[str]]:
    """The turn to hang a feedback row on, anchored at turn level.

    Deliberately turn-level (empty span list) rather than phase-level. The
    viewer renders a row only when the stored id set is byte-equal to the set it
    computes for the node being looked at, and it computes that set by walking
    its own rendered tree — not the span table. Any id list we assemble here is
    a guess at that tree's shape, and a guess that is one span out renders on no
    node at all, silently. A turn anchor is the one the viewer always agrees
    with: it compares `[] == []`.

    Nothing is lost by it. The comparison is per-task, so the finding is about
    the attempt rather than one span, and `target_label` still names which
    phase diverged.

    The phase argument selects which turn to attach to: the first turn that
    actually did work of that kind, so an execution finding lands on the turn
    that executed rather than on turn 1 by default.
    """
    for turn in student.turns:
        if turn.span_ids.get(phase):
            return turn.turn_key, []
    return (student.turns[0].turn_key if student.turns else None), []


def _write_feedback(
    store: Any,
    student: dtrace.ReconstructedAttempt,
    *,
    phase: str,
    label: str,
    divergence_summary: str,
    judge_reasoning: str,
    went_wrong: str,
    worked: str,
    insights: list[str],
    insights_subcategory: str,
) -> int:
    """Record one phase's divergence as coding-agent-readable annotations.

    The store takes one comment per note under one (category, subcategory)
    pair, so a finding is written as up to five notes on the same turn, all
    carrying the phase as their label, in the taxonomy's own order:

    * the divergence summary (the mechanical diff both LLMs were given)
      -> Observations / Analysis · Observation
    * the judge's reasoning (why each difference is a mistake or justified)
      -> Observations / Analysis · Analysis. With no reasoning and no rules,
      the judge's decline (``_NO_RULE_NOTE``) is recorded here instead.
    * ``went_wrong`` -> Conclusions · What went wrong
    * ``worked`` -> Conclusions · What went right
    * the extracted rules -> Recommendations · ``insights_subcategory``
      (What not to do for execution anti-patterns, What to do for planning
      rules); absent when there are none, so Recommendations only ever holds
      real rules.

    An empty part is skipped rather than sent, because the store rejects a note
    without text. Returns how many notes were written.

    Never raises: a comparison that cannot be annotated is still a comparison,
    and losing the insight files to a feedback-write failure would be worse.
    Each note is attempted on its own, so one rejected note does not cost the
    others.
    """
    turn_key, span_ids = _anchor(student, phase)
    if not turn_key:
        return 0
    analysis = judge_reasoning or ("" if insights else _NO_RULE_NOTE)
    notes = (
        ("divergence_summary", obs_feedback.CATEGORY_OBSERVATIONS_ANALYSIS,
         obs_feedback.SUBCATEGORY_OBSERVATION, divergence_summary),
        ("judge_reasoning", obs_feedback.CATEGORY_OBSERVATIONS_ANALYSIS,
         obs_feedback.SUBCATEGORY_ANALYSIS, analysis),
        ("went_wrong", obs_feedback.CATEGORY_CONCLUSIONS,
         obs_feedback.SUBCATEGORY_WHAT_WENT_WRONG, went_wrong),
        ("worked", obs_feedback.CATEGORY_CONCLUSIONS,
         obs_feedback.SUBCATEGORY_WHAT_WENT_RIGHT, worked),
        ("insights", obs_feedback.CATEGORY_RECOMMENDATIONS,
         insights_subcategory, "\n".join(insights)),
    )
    written = 0
    for part, category, subcategory, text in notes:
        comment = _fit_comment(text)
        if not comment:
            continue
        try:
            store.add_human_feedback(
                turn_key,
                target_kind="turn",
                span_ids=span_ids,
                target_label=label,
                provenance=PROVENANCE,
                comment=comment,
                category=category,
                subcategory=subcategory,
            )
            written += 1
        except Exception as exc:
            logger.warning(
                f"Could not record distillation feedback ({part}) on turn {turn_key}: "
                f"{type(exc).__name__}: {exc}"
            )
    return written


def compare_attempt(
    session: DistillationSession,
    store: Any,
    *,
    teacher_experiment_id: str,
    teacher: dtrace.ReconstructedAttempt,
    student: dtrace.ReconstructedAttempt,
    write_insights: bool = True,
    write_feedback: bool = True,
    feedback_on_empty: bool = True,
    reference_source: str = "",
) -> TaskComparison:
    """Compare one task's teacher and student attempts.

    ``teacher`` is the reference side: a teacher run's attempt, or a passing
    trial of the same model (``reference_source`` "self"; see compare_pool).

    The judge is given the task's opening message as ``user_query`` — identical
    on both sides by construction — and each side's whole-attempt trajectory and
    action list.
    """
    result = TaskComparison(
        task_id=student.task_id,
        teacher_attempt=teacher.attempt,
        student_attempt=student.attempt,
    )

    teacher_actions = teacher.all_actions()
    student_actions = student.all_actions()
    user_query = teacher.first_user_message or student.first_user_message

    # --- execution -----------------------------------------------------
    result.execution_diverged, result.execution_summary = session.compare_trajectories(
        teacher_actions, student_actions
    )
    if result.execution_diverged:
        result.shared_actions = _shared_actions(teacher_actions, student_actions)
        result.execution_insights = session.extract_insights(
            teacher.merged_trajectory(),
            student.merged_trajectory(),
            result.execution_summary,
            user_query,
        )
        if write_insights and result.execution_insights:
            session.append_insights(result.execution_insights)
        if write_feedback and (result.execution_insights or feedback_on_empty):
            went_wrong, worked = _narrate(
                teacher=teacher,
                student=student,
                kind="executed actions",
                divergence_summary=result.execution_summary,
                insights=result.execution_insights,
            )
            if not worked and result.shared_actions:
                worked = "Teacher and student both executed: " + ", ".join(
                    result.shared_actions
                )
            result.feedback_written += _write_feedback(
                store,
                student,
                phase=dtrace.PHASE_EXECUTION,
                label="Execution",
                divergence_summary=result.execution_summary,
                judge_reasoning=session.last_execution_reasoning,
                went_wrong=(
                    f"{_origin_note(teacher_experiment_id, teacher, student, reference_source)}\n\n"
                    f"{went_wrong}"
                ),
                worked=worked,
                # Execution insights are anti-patterns
                # (execution_agent_anti_patterns.md).
                insights=result.execution_insights,
                insights_subcategory=obs_feedback.SUBCATEGORY_WHAT_NOT_TO_DO,
            )

    # --- planning ------------------------------------------------------
    teacher_steps = teacher.all_planning_steps()
    student_steps = student.all_planning_steps()
    result.planning_diverged, result.planning_summary = session.compare_planning_traces(
        teacher_steps, student_steps
    )
    if result.planning_diverged:
        result.planning_insights = session.extract_planning_insights(
            teacher_steps,
            student_steps,
            result.planning_summary,
            user_query,
            teacher_actions,
            student_actions,
        )
        if write_insights and result.planning_insights:
            session.append_planning_insights(result.planning_insights)
        if write_feedback and (result.planning_insights or feedback_on_empty):
            went_wrong, worked = _narrate(
                teacher=teacher,
                student=student,
                kind="plans",
                divergence_summary=result.planning_summary,
                insights=result.planning_insights,
            )
            result.feedback_written += _write_feedback(
                store,
                student,
                phase=dtrace.PHASE_PLANNING,
                label="Planning",
                divergence_summary=result.planning_summary,
                judge_reasoning=session.last_planning_reasoning,
                went_wrong=(
                    f"{_origin_note(teacher_experiment_id, teacher, student, reference_source)}\n\n"
                    f"{went_wrong}"
                ),
                worked=worked,
                # Planning insights are prescriptive rules
                # (planning_agent_insights.md).
                insights=result.planning_insights,
                insights_subcategory=obs_feedback.SUBCATEGORY_WHAT_TO_DO,
            )

    return result


def compare_task(
    store: Any,
    *,
    teacher_experiment_id: str,
    student_experiment_id: str,
    workflow_folderpath: str,
    task_id: str,
    teacher_attempt: Optional[int] = None,
    student_attempt: Optional[int] = None,
    write_insights: bool = True,
    write_feedback: bool = True,
    feedback_on_empty: bool = True,
) -> TaskComparison:
    """Compare one task across the two runs.

    Attempts default to the lowest recorded on each side, so a teacher run with
    several trials distils against its first unless told otherwise.
    """
    def _pick(experiment_id: str, requested: Optional[int]) -> Optional[int]:
        if requested is not None:
            return requested
        attempts = [
            attempt
            for tid, attempt in dtrace.attempts_in_experiment(store, experiment_id)
            if tid == task_id
        ]
        return min(attempts) if attempts else None

    t_attempt = _pick(teacher_experiment_id, teacher_attempt)
    s_attempt = _pick(student_experiment_id, student_attempt)
    if t_attempt is None or s_attempt is None:
        return TaskComparison(
            task_id=task_id,
            teacher_attempt=t_attempt or 0,
            student_attempt=s_attempt or 0,
            skipped_reason=(
                "task not recorded in "
                + ("teacher" if t_attempt is None else "student")
                + " run"
            ),
        )

    try:
        teacher = dtrace.read_attempt(store, teacher_experiment_id, task_id, t_attempt)
        student = dtrace.read_attempt(store, student_experiment_id, task_id, s_attempt)
    except dtrace.TraceUnreadable as exc:
        return TaskComparison(
            task_id=task_id,
            teacher_attempt=t_attempt,
            student_attempt=s_attempt,
            skipped_reason=str(exc),
        )

    session = DistillationSession(workflow_folderpath=workflow_folderpath)
    return compare_attempt(
        session,
        store,
        teacher_experiment_id=teacher_experiment_id,
        teacher=teacher,
        student=student,
        write_insights=write_insights,
        write_feedback=write_feedback,
        feedback_on_empty=feedback_on_empty,
    )


def compare_runs(
    store: Any,
    *,
    teacher_experiment_id: str,
    student_experiment_id: str,
    workflow_folderpath: str,
    task_ids: Optional[list[str]] = None,
    teacher_attempt: Optional[int] = None,
    student_attempts: Any = None,
    write_insights: bool = True,
    write_feedback: bool = True,
    feedback_on_empty: bool = True,
) -> list[TaskComparison]:
    """Compare every task the two runs share, in task order.

    Per task, ``teacher_attempt`` (default: the lowest recorded) is compared with
    each of ``student_attempts``: None for the lowest recorded, ``"all"`` for
    every recorded attempt, or a list of attempt numbers. The same choice applies
    to every task; a task missing the named attempt is reported as skipped.
    Pairs are what was asked for, whatever their outcomes - see ``run_pairs``.

    Tasks are iterated sequentially because `extract_insights` re-reads the
    insights file on each call: the judge therefore sees insights extracted from
    earlier tasks in this same pass and can decline to repeat them, exactly as
    the live path behaves.

    Set ``write_insights=False, write_feedback=False`` for a dry run — the
    insight files are append-only, so a repeated pass would otherwise duplicate
    entries.
    """
    _warn_on_capture_mismatch(store, teacher_experiment_id, student_experiment_id)

    teacher = _attempts_by_task(store, teacher_experiment_id)
    student = _attempts_by_task(store, student_experiment_id)
    pairs = run_pairs(
        teacher, student, task_ids=task_ids,
        teacher_attempt=teacher_attempt, student_attempts=student_attempts,
    )
    if not pairs:
        logger.warning(
            f"No tasks in common between {teacher_experiment_id} and "
            f"{student_experiment_id}; nothing to distil"
        )
        return []

    outcomes = {
        **{(teacher_experiment_id, *k): v
           for k, v in _outcomes(store, teacher_experiment_id).items()},
        **{(student_experiment_id, *k): v
           for k, v in _outcomes(store, student_experiment_id).items()},
    }
    results: list[TaskComparison] = []
    for pair in pairs:
        if pair.skipped_reason:
            result = TaskComparison(
                task_id=pair.task_id,
                teacher_attempt=pair.reference_attempt,
                student_attempt=pair.student_attempt,
                skipped_reason=pair.skipped_reason,
            )
        else:
            result = compare_task(
                store,
                teacher_experiment_id=teacher_experiment_id,
                student_experiment_id=student_experiment_id,
                workflow_folderpath=workflow_folderpath,
                task_id=pair.task_id,
                teacher_attempt=pair.reference_attempt,
                student_attempt=pair.student_attempt,
                write_insights=write_insights,
                write_feedback=write_feedback,
                feedback_on_empty=feedback_on_empty,
            )
        result.student_experiment = student_experiment_id
        result.reference_experiment = teacher_experiment_id
        result.student_passed = outcomes.get(
            (student_experiment_id, pair.task_id, pair.student_attempt))
        result.reference_passed = outcomes.get(
            (teacher_experiment_id, pair.task_id, pair.reference_attempt))
        results.append(result)
    return results


def _attempts_by_task(store: Any, experiment_id: str) -> dict[str, list[int]]:
    """task_id -> the attempt numbers recorded in one run, ascending."""
    by_task: dict[str, list[int]] = {}
    for task_id, attempt in dtrace.attempts_in_experiment(store, experiment_id):
        by_task.setdefault(str(task_id), []).append(int(attempt))
    return {task: sorted(set(attempts)) for task, attempts in by_task.items()}


def _outcomes(store: Any, experiment_id: str) -> dict[tuple[str, int], Optional[bool]]:
    """(task_id, attempt) -> passed, from the run's graded attempt rows."""
    try:
        rows = store.experiment_attempt_rows(experiment_id)
    except Exception:  # noqa: BLE001 - outcomes are informational here
        return {}
    return {
        (str(r.get("task_id")), int(r.get("attempt") or 0)): attempt_passed(r)
        for r in rows
    }


def run_pairs(
    teacher: dict[str, list[int]],
    student: dict[str, list[int]],
    *,
    task_ids: Optional[list[str]] = None,
    teacher_attempt: Optional[int] = None,
    student_attempts: Any = None,
) -> list["PoolPair"]:
    """The teacher-vs-student pairs to compare, and the tasks skipped, in task order.

    Pure: no store, no model. ``teacher`` / ``student`` map task_id to the
    attempts each run recorded. Tasks are those both runs recorded, or
    ``task_ids``. Per task the reference is ``teacher_attempt`` (default the
    lowest teacher attempt) and the compared trials are ``student_attempts``:
    None for the lowest student attempt, ``"all"`` for every one, or a list of
    numbers (in the order given). Outcomes play no part: the caller chose.
    """
    if task_ids is None:
        tasks = sorted(set(teacher) & set(student))
    else:
        tasks = list(dict.fromkeys(str(t) for t in task_ids))
    pairs: list[PoolPair] = []
    for task_id in tasks:
        t_recorded, s_recorded = teacher.get(task_id, []), student.get(task_id, [])
        if not t_recorded or not s_recorded:
            pairs.append(PoolPair(
                task_id=task_id,
                skipped_reason="task not recorded in "
                + ("teacher" if not t_recorded else "student") + " run",
            ))
            continue
        reference = t_recorded[0] if teacher_attempt is None else int(teacher_attempt)
        if reference not in t_recorded:
            pairs.append(PoolPair(
                task_id=task_id, reference_attempt=reference,
                skipped_reason=f"teacher attempt {reference} not recorded for this task "
                               f"(recorded: {', '.join(map(str, t_recorded))})",
            ))
            continue
        if student_attempts is None:
            chosen = [s_recorded[0]]
        elif student_attempts == "all":
            chosen = list(s_recorded)
        else:
            chosen = list(dict.fromkeys(int(a) for a in student_attempts))
        for attempt in chosen:
            if attempt not in s_recorded:
                pairs.append(PoolPair(
                    task_id=task_id, student_attempt=attempt, reference_attempt=reference,
                    skipped_reason=f"student attempt {attempt} not recorded for this task "
                                   f"(recorded: {', '.join(map(str, s_recorded))})",
                ))
                continue
            pairs.append(PoolPair(
                task_id=task_id, student_attempt=attempt, reference_attempt=reference,
            ))
    return pairs


#: An attempt with no recorded outcome counts as passing at or above this reward.
PASS_REWARD = 1.0


def attempt_passed(row: dict[str, Any]) -> Optional[bool]:
    """Whether an attempt row passed; None when it was not graded.

    The store's own ``outcome`` ("pass" / "fail") decides when present, as it
    does for the experiment views; otherwise a recorded ``reward`` at or above
    ``PASS_REWARD`` is a pass.
    """
    outcome = str(row.get("outcome") or "").strip().lower()
    if outcome in ("pass", "fail"):
        return outcome == "pass"
    reward = row.get("reward")
    if reward is None:
        return None
    try:
        return float(reward) >= PASS_REWARD
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class PoolPair:
    """One pair to distil, or one task skipped (``skipped_reason`` set)."""

    task_id: str
    student_experiment: str = ""
    student_attempt: int = 0
    reference_experiment: str = ""
    reference_attempt: int = 0
    reference_source: str = ""
    skipped_reason: Optional[str] = None


def graded_attempts(
    store: Any, experiment_ids: list[str]
) -> dict[str, list[tuple[str, int, bool]]]:
    """task_id -> [(experiment_id, attempt, passed)], runs in the order given.

    Ungraded attempts are left out: such a trial can be neither a reference
    nor a failure to learn from.
    """
    by_task: dict[str, list[tuple[str, int, bool]]] = {}
    for experiment_id in experiment_ids:
        rows = sorted(
            store.experiment_attempt_rows(experiment_id),
            key=lambda r: (str(r.get("task_id")), int(r.get("attempt") or 0)),
        )
        for row in rows:
            passed = attempt_passed(row)
            if row.get("task_id") is None or row.get("attempt") is None or passed is None:
                continue
            by_task.setdefault(str(row["task_id"]), []).append(
                (experiment_id, int(row["attempt"]), passed)
            )
    return by_task


def pool_pairs(
    students: dict[str, list[tuple[str, int, bool]]],
    teachers: dict[str, list[tuple[str, int, bool]]],
    *,
    task_ids: Optional[list[str]] = None,
    max_pairs_per_task: Optional[int] = None,
) -> list[PoolPair]:
    """The pairs compare_pool distils, and the tasks it skips, in task order.

    Pure: no store, no model. See compare_pool for the rules.
    """
    pairs: list[PoolPair] = []
    for task_id in (sorted(students) if task_ids is None else list(task_ids)):
        trials = students.get(task_id, [])
        fails = [(e, a) for e, a, passed in trials if not passed]
        reference = next(((e, a, "self") for e, a, passed in trials if passed), None)
        if reference is None:
            reference = next(
                ((e, a, "teacher") for e, a, passed in teachers.get(task_id, []) if passed),
                None,
            )
        if not fails or reference is None:
            passed_n = sum(1 for _, _, p in trials if p)
            why = (
                "task not graded in the student runs" if not trials
                else "no failing student trial to learn about" if not fails
                else "no passing reference (student or teacher) to learn from"
            )
            pairs.append(PoolPair(
                task_id=task_id,
                student_experiment=fails[0][0] if fails else "",
                student_attempt=fails[0][1] if fails else 0,
                reference_experiment=reference[0] if reference else "",
                reference_attempt=reference[1] if reference else 0,
                reference_source=reference[2] if reference else "",
                skipped_reason=f"{why} ({passed_n} passed, {len(fails)} failed)",
            ))
            continue
        for fail_experiment, fail_attempt in fails[:max_pairs_per_task]:
            pairs.append(PoolPair(
                task_id=task_id,
                student_experiment=fail_experiment,
                student_attempt=fail_attempt,
                reference_experiment=reference[0],
                reference_attempt=reference[1],
                reference_source=reference[2],
            ))
    return pairs


def compare_pool(
    store: Any,
    *,
    student_experiments: list[str],
    teacher_experiments: Optional[list[str]] = None,
    workflow_folderpath: str,
    task_ids: Optional[list[str]] = None,
    max_pairs_per_task: Optional[int] = None,
    write_insights: bool = True,
    write_feedback: bool = True,
    feedback_on_empty: bool = True,
) -> list[TaskComparison]:
    """Distil every failing student trial against a PASSING reference.

    The reference for a task is, in order of preference:

    1. a passing trial of the same (student) model, from ``student_experiments``
       in the order given - its insights are about the model's own variance,
       which is what a verifier can correct;
    2. otherwise a passing trial from ``teacher_experiments`` - covers tasks
       the student never solves.

    A teacher trial is never a reference unless it passed: comparing against a
    failing "teacher" tells the judge a wrong trajectory is the correct one.
    Tasks with no failing student trial, or no passing reference anywhere, are
    reported as skipped. Each result carries ``reference_source`` ("self" or
    "teacher") so every insight can be traced to its source.

    ``max_pairs_per_task`` caps how many failing trials per task are distilled
    (lowest attempts first); the judge sees earlier pairs' insights, so later
    pairs of the same task mostly repeat them. Pairs run sequentially for that
    reason, as in compare_runs.
    """
    pairs = pool_pairs(
        graded_attempts(store, student_experiments),
        graded_attempts(store, list(teacher_experiments or [])),
        task_ids=task_ids,
        max_pairs_per_task=max_pairs_per_task,
    )
    results: list[TaskComparison] = []
    session: Optional[DistillationSession] = None
    references: dict[tuple[str, str, int], Any] = {}
    for pair in pairs:
        result = TaskComparison(
            task_id=pair.task_id,
            teacher_attempt=pair.reference_attempt,
            student_attempt=pair.student_attempt,
            skipped_reason=pair.skipped_reason,
            student_experiment=pair.student_experiment,
            reference_experiment=pair.reference_experiment,
            reference_source=pair.reference_source,
        )
        if pair.skipped_reason:
            results.append(result)
            continue
        try:
            key = (pair.reference_experiment, pair.task_id, pair.reference_attempt)
            if key not in references:
                references[key] = dtrace.read_attempt(store, *key)
            failed = dtrace.read_attempt(
                store, pair.student_experiment, pair.task_id, pair.student_attempt)
        except dtrace.TraceUnreadable as exc:
            result.skipped_reason = str(exc)
            results.append(result)
            continue
        if session is None:
            session = DistillationSession(workflow_folderpath=workflow_folderpath)
        compared = compare_attempt(
            session,
            store,
            teacher_experiment_id=pair.reference_experiment,
            teacher=references[key],
            student=failed,
            write_insights=write_insights,
            write_feedback=write_feedback,
            feedback_on_empty=feedback_on_empty,
            reference_source=pair.reference_source,
        )
        compared.student_experiment = pair.student_experiment
        compared.reference_experiment = pair.reference_experiment
        compared.reference_source = pair.reference_source
        compared.student_passed, compared.reference_passed = False, True
        results.append(compared)
    return results


def compare_self(
    store: Any,
    *,
    experiment_id: str,
    workflow_folderpath: str,
    task_ids: Optional[list[str]] = None,
    max_pairs_per_task: Optional[int] = None,
    write_insights: bool = True,
    write_feedback: bool = True,
    feedback_on_empty: bool = True,
) -> list[TaskComparison]:
    """Distil a run against itself: each failing trial against a passing one.

    Same model, same task, several trials. Where some passed and some failed,
    the lowest-numbered passing trial is the reference and every failing trial
    is compared with it. The judge's "correct approach" vs "may have issues"
    framing is exactly passing vs failing. compare_pool with no teacher runs.
    """
    return compare_pool(
        store,
        student_experiments=[experiment_id],
        teacher_experiments=[],
        workflow_folderpath=workflow_folderpath,
        task_ids=task_ids,
        max_pairs_per_task=max_pairs_per_task,
        write_insights=write_insights,
        write_feedback=write_feedback,
        feedback_on_empty=feedback_on_empty,
    )


def provenance_records(results: list[TaskComparison]) -> list[dict[str, Any]]:
    """One JSON-ready record per compared pair: which runs, and what it produced.

    Written by ``fastworkflow distill --provenance FILE`` so every insight can be
    traced back to the pair (and reference source) that produced it.
    """
    return [
        {
            "task_id": r.task_id,
            "student_experiment": r.student_experiment,
            "student_attempt": r.student_attempt,
            "reference_experiment": r.reference_experiment,
            "reference_attempt": r.teacher_attempt,
            "reference_source": r.reference_source,
            "student_passed": r.student_passed,
            "reference_passed": r.reference_passed,
            "skipped_reason": r.skipped_reason,
            "execution_insights": list(r.execution_insights),
            "planning_insights": list(r.planning_insights),
        }
        for r in results
    ]


def _warn_on_capture_mismatch(
    store: Any, teacher_experiment_id: str, student_experiment_id: str
) -> None:
    """Two runs recorded under different capture profiles are not comparable.

    A withheld field would reach the judge as an absence, which reads as "the
    student did less" — a fabricated divergence. Warn rather than refuse: the
    trace layer already reports what it could not read.
    """
    try:
        teacher = store.get_experiment(teacher_experiment_id) or {}
        student = store.get_experiment(student_experiment_id) or {}
    except Exception:
        return
    t_profile = teacher.get("capture_profile")
    s_profile = student.get("capture_profile")
    if t_profile and s_profile and t_profile != s_profile:
        logger.warning(
            f"Teacher run was captured under profile {t_profile!r} but the student "
            f"under {s_profile!r}; withheld fields can read as missing work"
        )
