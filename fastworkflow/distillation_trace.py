"""Rebuild a recorded run's agent trace from observability spans.

Distillation compares what a teacher did against what a student did. When both
already ran, their traces live in the observability store, and this module turns
them back into the three shapes `distillation.py` compares:

* the ReAct ``trajectory`` dict (``thought_i`` / ``tool_name_i`` / ``tool_args_i``
  / ``observation_i``), consumed by ``_format_trajectory_for_llm``
* the ``actions`` list (``command_name`` + ``parameters``), consumed by
  ``compare_trajectories``
* the ``PlanningStep`` list, consumed by ``compare_planning_traces``

Nothing here knows what produced the run. ``experiment_id`` and ``task_id`` are
opaque strings and tasks/attempts are discovered from the store, so any harness
that records through fastWorkflow can be distilled, not just one benchmark.

Reconstruction is not a straight read; four recorded details need care, each
documented at the function that handles it: dense step re-indexing
(`_trajectory_from_steps`), truncation envelopes (`unwrap`), plan line-splitting
(`_planning_steps_from_spans`) and reading the user message off the span rather
than the turn row (`_user_message`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from fastworkflow import tracing
from fastworkflow.distillation import PlanningStep
from fastworkflow.utils.logging import logger

# Phases the comparer anchors feedback to, mapped to the span names that carry
# them. The labels match the chatbot's phase nodes so an annotation lands on a
# node the viewer already renders.
PHASE_PLANNING = "planning"
PHASE_EXECUTION = "execution"
PHASE_COMMANDS = "commands"
PHASE_STEPS = "steps"


class TraceUnreadable(RuntimeError):
    """A recorded trace cannot be rebuilt into something comparable."""


@dataclass
class ReconstructedTurn:
    """One recorded turn, in the shapes `distillation.py` compares."""

    turn_key: str
    ordinal: int
    user_message: str
    trajectory: dict[str, Any]
    actions: list[dict[str, Any]]
    planning_steps: list[PlanningStep]
    span_ids: dict[str, list[str]] = field(default_factory=dict)
    degraded: list[str] = field(default_factory=list)


@dataclass
class ReconstructedAttempt:
    """Every turn of one (experiment, task, attempt), oldest first."""

    experiment_id: str
    task_id: str
    attempt: int
    turns: list[ReconstructedTurn]

    @property
    def first_user_message(self) -> str:
        """The opening message — what the judge is given as ``user_query``."""
        return self.turns[0].user_message if self.turns else ""

    def all_actions(self) -> list[dict[str, Any]]:
        """Every action across the attempt, in order.

        `compare_trajectories` takes flat action lists and is not turn-scoped,
        so a whole attempt compares as one unit.
        """
        return [action for turn in self.turns for action in turn.actions]

    def all_planning_steps(self) -> list[PlanningStep]:
        """Every planning step across the attempt, renumbered end to end."""
        steps: list[PlanningStep] = []
        for turn in self.turns:
            for step in turn.planning_steps:
                steps.append(
                    PlanningStep(
                        step_number=len(steps),
                        user_query=step.user_query,
                        generated_plan=step.generated_plan,
                        reasoning=step.reasoning,
                    )
                )
        return steps

    def merged_trajectory(self) -> dict[str, Any]:
        """Every ReAct step across the attempt as one densely-indexed dict.

        `_format_trajectory_for_llm` walks ``idx = 0, 1, 2 …`` and stops at the
        first missing key, so per-turn indices have to be rebased rather than
        concatenated.
        """
        merged: dict[str, Any] = {}
        idx = 0
        for turn in self.turns:
            for local in _step_indices(turn.trajectory):
                for key in ("thought", "tool_name", "tool_args", "action", "observation"):
                    value = turn.trajectory.get(f"{key}_{local}")
                    if value is not None:
                        merged[f"{key}_{idx}"] = value
                idx += 1
        return merged

    def span_ids_for(self, phase: str) -> list[str]:
        """Every span id of one phase across the attempt."""
        return [sid for turn in self.turns for sid in turn.span_ids.get(phase, [])]

    def turn_for_span(self, span_id: str) -> Optional[ReconstructedTurn]:
        """The turn a span belongs to — how feedback stays turn-anchored."""
        return next(
            (
                turn
                for turn in self.turns
                if any(span_id in ids for ids in turn.span_ids.values())
            ),
            None,
        )


# ----------------------------------------------------------------------
# Attribute helpers
# ----------------------------------------------------------------------


def unwrap(value: Any) -> Any:
    """Undo `tracing.cap_attr_value`'s truncation envelope.

    Oversize attribute strings are stored as
    ``{"truncated": True, "original_length": N, "sha256": ..., "value": prefix}``.
    Feeding that dict into a judge prompt would render as a Python repr, so the
    prefix is returned with the truncation stated in words instead.
    """
    if isinstance(value, dict) and value.get("truncated") is True and "value" in value:
        original = value.get("original_length")
        suffix = (
            f"... [truncated from {original} bytes]"
            if isinstance(original, int)
            else "... [truncated]"
        )
        return f"{value['value']}{suffix}"
    return value


def _attributes(span: dict[str, Any]) -> dict[str, Any]:
    """A span row's attributes as a dict, whatever the row carried."""
    raw = span.get("attributes")
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _attr(span: dict[str, Any], key: str, default: Any = None) -> Any:
    value = _attributes(span).get(key, default)
    return unwrap(value) if value is not None else default


def _by_name(spans: Iterable[dict[str, Any]], *names: str) -> list[dict[str, Any]]:
    """Spans of the given names, in recorded order (already ``start_ns``)."""
    wanted = set(names)
    return [span for span in spans if span.get("name") in wanted]


def _step_indices(trajectory: dict[str, Any]) -> list[int]:
    """The dense step indices present in a trajectory dict."""
    indices = {
        int(key.rsplit("_", 1)[1])
        for key in trajectory
        if key.startswith("thought_") and key.rsplit("_", 1)[1].isdigit()
    }
    return sorted(indices)


# ----------------------------------------------------------------------
# Per-shape reconstruction
# ----------------------------------------------------------------------


def _trajectory_from_steps(spans: list[dict[str, Any]]) -> dict[str, Any]:
    """The ReAct trajectory from ``fw.agent.step`` spans.

    Indices are assigned by recorded order, NOT from the ``step_index``
    attribute: a turn resumed after ask_user restarts its local counter, so the
    recorded indices can repeat or gap — and `_format_trajectory_for_llm` stops
    at the first missing key, which would silently truncate the trace the judge
    sees.
    """
    trajectory: dict[str, Any] = {}
    for idx, span in enumerate(_by_name(spans, tracing.SPAN_AGENT_STEP)):
        attrs = _attributes(span)
        for key in ("thought", "tool_name", "tool_args", "observation"):
            if key in attrs:
                trajectory[f"{key}_{idx}"] = unwrap(attrs[key])
        if "tool_name" in attrs:
            trajectory[f"action_{idx}"] = (
                f"{unwrap(attrs.get('tool_name'))}: {unwrap(attrs.get('tool_args'))}"
            )
    return trajectory


def _blocked_by_policy(span: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> bool:
    """Whether a command span belongs to a step a policy rule rewrote.

    Such a command was stopped before it ran (fastworkflow.policy_runtime): its
    span records the dispatch, but it is not an action the agent took, so the
    comparison must not see it - the live action log leaves it out too.
    """
    parent = by_id.get(span.get("parent_span_id"))
    for _ in range(4):
        if parent is None:
            return False
        if parent.get("name") == tracing.SPAN_AGENT_STEP:
            return bool(_attributes(parent).get("policy_source"))
        parent = by_id.get(parent.get("parent_span_id"))
    return False


def _actions_from_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The action log from ``fw.command.execute`` spans.

    Only this span carries ``parameters``, and ``command_name`` + ``parameters``
    are the only fields `_action_signature` reads — so the rebuilt list compares
    byte-identically against one captured live.
    """
    actions: list[dict[str, Any]] = []
    by_id = {span.get("span_id"): span for span in spans if span.get("span_id")}
    for span in _by_name(spans, tracing.SPAN_COMMAND_EXECUTE):
        if _blocked_by_policy(span, by_id):
            continue
        attrs = _attributes(span)
        actions.append(
            {
                "command": unwrap(attrs.get("raw_command", "")) or "",
                "command_name": span.get("command_name") or "",
                "parameters": attrs.get("parameters") or {},
                "response": unwrap(attrs.get("response_text", "")) or "",
            }
        )
    return actions


def _planner_reasoning(
    planner_span: dict[str, Any], children: dict[str, list[dict[str, Any]]]
) -> str:
    """The planner's chain-of-thought, from its child ``fw.llm.call`` span.

    `PlanningStep.reasoning` is not on the planner span's contract, but
    `dspy_logger` back-fills ``reasoning`` onto the ``fw.llm.call`` span opened
    inside the planner's module frame. Joining to the child recovers it without
    a span-contract change — and works on runs recorded before this module
    existed, which is what makes already-recorded teachers usable.
    """
    for child in children.get(planner_span.get("span_id"), ()):
        if child.get("name") == tracing.SPAN_LLM_CALL:
            reasoning = _attr(child, "reasoning")
            if reasoning:
                return reasoning if isinstance(reasoning, str) else str(reasoning)
    return ""


def _planning_steps_from_spans(
    spans: list[dict[str, Any]],
    children: dict[str, list[dict[str, Any]]],
    refined_user_message: str,
) -> list[PlanningStep]:
    """Planning steps from ``fw.planner.plan`` / ``.replan`` spans.

    The plan is split on LINES, not whitespace. The live path does
    ``next_steps.split()`` (`workflow_agent.py`), which shreds a numbered plan
    into single words and makes `compare_planning_traces` fire on any wording
    difference. That split cannot be fixed at source — the same list is joined
    back into the agent's prompt, so changing it would alter agent behaviour and
    invalidate recorded runs. Splitting correctly here is safe because BOTH
    sides of a comparison are rebuilt by this function, so the comparison stays
    symmetric and `compare_planning_traces` itself is untouched.
    """
    steps: list[PlanningStep] = []
    planner_spans = _by_name(spans, tracing.SPAN_PLANNER_PLAN, tracing.SPAN_PLANNER_REPLAN)
    for idx, span in enumerate(planner_spans):
        plan_text = _attr(span, "plan", "") or ""
        lines = [line.strip() for line in str(plan_text).splitlines() if line.strip()]
        trigger = _attr(span, "replan_trigger")
        # The initial plan answers the refined user message; a replan answers
        # whatever re-triggered it, which is all the span records.
        query = refined_user_message if idx == 0 and not trigger else f"[replan: {trigger}]"
        steps.append(
            PlanningStep(
                step_number=idx,
                user_query=query,
                generated_plan=lines,
                reasoning=_planner_reasoning(span, children),
            )
        )
    return steps


def _user_message(turn_row: dict[str, Any], spans: list[dict[str, Any]]) -> tuple[str, bool]:
    """The turn's user message, preferring the span over the turn row.

    Under the ``evidence`` capture profile the ``turns.user_message`` column is
    withheld, but span attributes are only credential-scrubbed, never withheld —
    so the ``fw.turn`` span still carries it. Returns ``(message, degraded)``.
    """
    for span in _by_name(spans, tracing.SPAN_TURN):
        message = _attr(span, "user_message")
        if isinstance(message, str) and message:
            return message, False
    fallback = turn_row.get("user_message")
    if isinstance(fallback, str) and fallback:
        return fallback, False
    return "", True


def _children_by_parent(spans: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    children: dict[str, list[dict[str, Any]]] = {}
    for span in spans:
        parent = span.get("parent_span_id")
        if parent:
            children.setdefault(parent, []).append(span)
    return children


def _span_ids_by_phase(spans: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Span ids grouped by phase, as LEAF ids, sorted.

    Must match how the viewer anchors an annotation, because it renders a
    feedback row only when the stored id set equals the set it computes for the
    node being viewed — an exact string comparison, not a subset test. Its
    `feedbackAnchor` walks a phase node and keeps only the spans that have no
    children, so recording the parent id instead would store a row that renders
    on no node at all.
    """
    children: dict[str, list[dict[str, Any]]] = {}
    for span in spans:
        parent = span.get("parent_span_id")
        if parent:
            children.setdefault(parent, []).append(span)

    def leaves(span: dict[str, Any]) -> list[str]:
        kids = children.get(span.get("span_id"), [])
        if not kids:
            return [span["span_id"]] if span.get("span_id") else []
        return [sid for kid in kids for sid in leaves(kid)]

    def ids(*names: str) -> list[str]:
        collected: set[str] = set()
        for span in _by_name(spans, *names):
            collected.update(leaves(span))
        return sorted(collected)

    return {
        PHASE_PLANNING: ids(tracing.SPAN_PLANNER_PLAN, tracing.SPAN_PLANNER_REPLAN),
        PHASE_EXECUTION: ids(tracing.SPAN_AGENT_EXECUTE),
        PHASE_COMMANDS: ids(tracing.SPAN_COMMAND_EXECUTE),
        PHASE_STEPS: ids(tracing.SPAN_AGENT_STEP),
    }


# ----------------------------------------------------------------------
# Public reads
# ----------------------------------------------------------------------


def read_turn(turn_row: dict[str, Any], spans: list[dict[str, Any]]) -> ReconstructedTurn:
    """Rebuild one turn from its row and its spans."""
    children = _children_by_parent(spans)
    message, message_missing = _user_message(turn_row, spans)
    refined = turn_row.get("refined_user_message") or message

    degraded: list[str] = []
    if message_missing:
        degraded.append("user_message")

    return ReconstructedTurn(
        turn_key=turn_row.get("turn_key", ""),
        ordinal=int(turn_row.get("ordinal") or 0),
        user_message=message,
        trajectory=_trajectory_from_steps(spans),
        actions=_actions_from_spans(spans),
        planning_steps=_planning_steps_from_spans(
            spans, children, refined if isinstance(refined, str) else ""
        ),
        span_ids=_span_ids_by_phase(spans),
        degraded=degraded,
    )


def read_attempt(
    store: Any, experiment_id: str, task_id: str, attempt: int
) -> ReconstructedAttempt:
    """Rebuild every turn of one attempt, oldest first.

    Raises `TraceUnreadable` when the attempt recorded no turns — comparing an
    empty trace against a full one would report every teacher action as a
    student omission, which is a fabricated divergence, not a finding.
    """
    turn_rows = store.list_turns(
        experiment_id=experiment_id, task_id=task_id, attempt=attempt, limit=10000
    )
    if not turn_rows:
        raise TraceUnreadable(
            f"no turns recorded for experiment {experiment_id!r} task {task_id!r} "
            f"attempt {attempt}"
        )
    turn_rows.sort(key=lambda row: (row.get("ordinal") or 0, row.get("turn_key") or ""))

    spans_by_turn = store.spans_for_turns([row["turn_key"] for row in turn_rows])
    turns = [
        read_turn(row, spans_by_turn.get(row["turn_key"], [])) for row in turn_rows
    ]

    degraded = sorted({field for turn in turns for field in turn.degraded})
    if degraded:
        logger.warning(
            f"Distillation trace for {experiment_id}/{task_id}#{attempt} is missing "
            f"{', '.join(degraded)}; the capture profile may be withholding it"
        )
    return ReconstructedAttempt(
        experiment_id=experiment_id,
        task_id=task_id,
        attempt=attempt,
        turns=turns,
    )


def attempts_in_experiment(store: Any, experiment_id: str) -> list[tuple[str, int]]:
    """Every ``(task_id, attempt)`` an experiment recorded, in task order.

    Discovered from the store rather than supplied, so the comparer needs no
    knowledge of how the harness enumerated its tasks.
    """
    rows = store.experiment_attempt_rows(experiment_id)
    pairs = {
        (str(row["task_id"]), int(row["attempt"]))
        for row in rows
        if row.get("task_id") is not None and row.get("attempt") is not None
    }
    return sorted(pairs)
