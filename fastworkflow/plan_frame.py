"""
Per-call planning state for a turn: the PlanFrame stack (design 3.5).

A PlanFrame holds the sub-task state of one planning-command call: one frame per
planning-command call, with the turn frame at index 0 of
WorkflowExecutionContext._frames. run_planning_command pushes a frame for each
call and pops it when the call ends.

to_state / from_state carry a frame across a suspension. The turn frame is
written to the existing top-level keys of the session blob instead, so its
values are not written through these methods.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Sequence

import fastworkflow
from fastworkflow.utils.logging import logger

#: The env setting for the turn-wide step budget of nested frames.
TURN_MAX_STEPS_ENV = "FW_TURN_MAX_STEPS"
#: The default turn-wide step budget.
DEFAULT_TURN_MAX_STEPS = 400
#: The raw values already warned about. turn_max_steps() runs once per nested
#: command, so a bad setting would otherwise log on every one of them.
_warned_turn_max_steps: set[str] = set()


def turn_max_steps() -> int:
    """The turn-wide step budget: ``FW_TURN_MAX_STEPS``, default 400.

    Read through ``fastworkflow.get_env_var``, which takes the workflow's env file
    before the process environment. A value that is not a positive integer logs a
    warning (once per distinct raw value) and the default applies.
    """
    raw = str(fastworkflow.get_env_var(TURN_MAX_STEPS_ENV, str, default=""))
    if raw == "":
        return DEFAULT_TURN_MAX_STEPS
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value > 0:
        return value
    if raw not in _warned_turn_max_steps:
        _warned_turn_max_steps.add(raw)
        logger.warning(
            f"{TURN_MAX_STEPS_ENV}={raw!r} is not a positive integer; "
            f"using the default {DEFAULT_TURN_MAX_STEPS}"
        )
    return DEFAULT_TURN_MAX_STEPS


@dataclass
class PlanFrame:
    kind: Literal["turn", "planning", "spawned"]   # "spawned" from Phase 18 (Spawned frames, below)
    command_name: Optional[str]          # qualified planning command name; None for the turn frame and a spawned frame
    depth: int                           # 0 for the turn frame
    alias: str                           # "" for turn; "<parent alias>/<stem>#<n>" otherwise
    subtasks: list[str] = field(default_factory=list)
    origins: list[int] = field(default_factory=list)
    subtask_index: int = 0
    subtask_answers: list[str] = field(default_factory=list)
    # Parallel to subtask_answers: each run's typed Output as JSON, "" when the run produced none (fix-3rsq.plan-review.2.11).
    subtask_outputs: list[str] = field(default_factory=list)
    subtask_exhausted: bool = False
    evidence_floors: dict[int, int] = field(default_factory=dict)   # not serialised (as today)
    initial_plan_text: str = ""          # not serialised (as today); stays: the planner writes the run's plan (Phase 6 decision 2026-10-30)
    phase_run: Optional[dict[str, Any]] = None
    phase_live: Optional[tuple[Any, list[Any]]] = None   # not serialised (as today)
    phased_planning_command_call: Any = None        # not serialised (as today)
    # The planning command call in progress as (command, call): what a restart after a failed first phase plans
    # again (WorkflowExecutionContext._restart_after_empty_first_phase). Not serialised: no restart after a restore.
    phase_call: Any = None
    child_step_ranges: list[list[int]] = field(default_factory=list)  # [first, last] per child run
    invoking_step: Optional[int] = None  # the parent step index this frame runs inside
    #: The step number the call's first step takes (its run's range starts here).
    first_step: Optional[int] = None
    #: The context the parent's step ran in: the clause its now-in note compares against. The run is
    #: held under the context the call ended on, not this one.
    ran_in_clause: str = ""
    ran_in_context_name: str = ""
    #: The arguments the call was run with, to rebuild its command input when it finishes after a suspension.
    call_args: dict[str, Any] = field(default_factory=dict)
    #: The command the parent's step ran, as the action log holds it, and the note on the
    #: arguments that were folded into it: a step that suspended never wrote either.
    command_text: str = ""
    folded_note: str = ""
    agent: Any = None                    # per-frame fastWorkflowReAct, built lazily
    #: How many planning-command calls this frame has made, per command stem: the
    #: ``n`` of a child's alias (``<alias>/<stem>#<n>``). Not serialised, so a restored
    #: frame counts from its restore; alias numbers are for tracing, and step indexes
    #: stay unique regardless.
    child_counts: dict[str, int] = field(default_factory=dict)
    #: The turn frame's sub-task review gate (fix-3rsq.plan-review.2.6/.2.9): the review is open (the turn is awaiting
    #: it), and the turn's sub-task plan (review_proposals.SubTaskProposal: the accepted proposal, or the draft). Only the
    #: turn frame uses them; the serialised turn state carries them as top-level keys.
    subtask_review_open: bool = False
    framework_entry: bool = False
    subtask_plan: Any = None

    @classmethod
    def turn(cls) -> "PlanFrame":
        """The turn frame: the frame at index 0, with no planning call behind it."""
        return cls(kind="turn", command_name=None, depth=0, alias="")

    def to_state(self) -> dict[str, Any]:
        """The fields a nested frame carries across a suspension, as fresh values.

        Lists and dicts are new objects, never the frame's own: validate_state
        forbids a mutable container reachable by two paths.
        """
        return {
            "kind": self.kind,
            "command_name": self.command_name,
            "depth": self.depth,
            "alias": self.alias,
            "subtasks": list(self.subtasks),
            "origins": list(self.origins),
            "subtask_index": self.subtask_index,
            "subtask_answers": list(self.subtask_answers),
            "subtask_outputs": list(self.subtask_outputs),
            "subtask_exhausted": self.subtask_exhausted,
            "phase_run": copy.deepcopy(self.phase_run),
            "child_step_ranges": [list(step_range) for step_range in self.child_step_ranges],
            "invoking_step": self.invoking_step,
            "first_step": self.first_step,
            "ran_in_clause": self.ran_in_clause,
            "ran_in_context_name": self.ran_in_context_name,
            "call_args": copy.deepcopy(self.call_args),
            "command_text": self.command_text,
            "folded_note": self.folded_note,
        }

    @classmethod
    def from_state(cls, d: dict[str, Any]) -> "PlanFrame":
        """The inverse of to_state. A key missing from ``d`` takes its default."""
        subtasks = list(d.get("subtasks") or [])
        return cls(
            kind=d.get("kind", "turn"),
            command_name=d.get("command_name"),
            depth=int(d.get("depth", 0)),
            alias=d.get("alias", ""),
            subtasks=subtasks,
            # Absent origins: each entry is its own origin, as apply_serialized_state reads them.
            origins=list(d.get("origins") or range(len(subtasks))),
            subtask_index=int(d.get("subtask_index") or 0),
            subtask_answers=list(d.get("subtask_answers") or []),
            subtask_outputs=list(d.get("subtask_outputs") or []),
            subtask_exhausted=bool(d.get("subtask_exhausted")),
            phase_run=copy.deepcopy(d.get("phase_run") or None),
            child_step_ranges=[list(step_range) for step_range in d.get("child_step_ranges") or []],
            invoking_step=d.get("invoking_step"),
            first_step=None if d.get("first_step") is None else int(d["first_step"]),
            ran_in_clause=str(d.get("ran_in_clause") or ""),
            ran_in_context_name=str(d.get("ran_in_context_name") or ""),
            call_args=copy.deepcopy(d.get("call_args") or {}),
            command_text=str(d.get("command_text") or ""),
            folded_note=str(d.get("folded_note") or ""),
        )


def runs_outside_ranges(runs: Sequence[dict[str, Any]],
                        ranges: Sequence[Sequence[int]]) -> list[dict[str, Any]]:
    """The runs whose ``step_index`` lies in no range of *ranges* (each ``[first, last]``,
    inclusive), in their order. A run with no step index is kept."""
    return [run for run in runs
            if run.get("step_index") is None
            or not any(first <= run["step_index"] <= last for first, last in ranges)]
