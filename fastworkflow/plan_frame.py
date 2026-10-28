"""
Per-call planning state for a turn: the PlanFrame stack (design 3.5).

A PlanFrame holds the sub-task state of one planning-command call: one frame per
planning-command call, with the turn frame at index 0 of
WorkflowExecutionContext._frames. Nothing pushes a nested frame yet, so for now
the stack is always the turn frame alone.

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


def turn_max_steps() -> int:
    """The turn-wide step budget: ``FW_TURN_MAX_STEPS``, default 400.

    Read through ``fastworkflow.get_env_var``, which takes the workflow's env file
    before the process environment. A value that is not a positive integer logs a
    warning and the default applies.
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
    subtask_exhausted: bool = False
    evidence_floors: dict[int, int] = field(default_factory=dict)   # not serialised (as today)
    initial_plan_text: str = ""          # not serialised (as today); goes in Phase 6: a run's plan is its latest plan step (3.6)
    phase_run: Optional[dict[str, Any]] = None
    phase_live: Optional[tuple[Any, list[Any]]] = None   # not serialised (as today)
    phased_skill_call: Any = None        # not serialised (as today)
    child_step_ranges: list[list[int]] = field(default_factory=list)  # [first, last] per child run
    invoking_step: Optional[int] = None  # the parent step index this frame runs inside
    agent: Any = None                    # per-frame fastWorkflowReAct, built lazily
    fast_path: bool = False              # the whole plan of its sub-task (3.6)

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
            "subtask_exhausted": self.subtask_exhausted,
            "phase_run": copy.deepcopy(self.phase_run),
            "child_step_ranges": [list(step_range) for step_range in self.child_step_ranges],
            "invoking_step": self.invoking_step,
            "fast_path": self.fast_path,
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
            subtask_exhausted=bool(d.get("subtask_exhausted")),
            phase_run=copy.deepcopy(d.get("phase_run") or None),
            child_step_ranges=[list(step_range) for step_range in d.get("child_step_ranges") or []],
            invoking_step=d.get("invoking_step"),
            fast_path=bool(d.get("fast_path")),
        )


def runs_outside_ranges(runs: Sequence[dict[str, Any]],
                        ranges: Sequence[Sequence[int]]) -> list[dict[str, Any]]:
    """The runs whose ``step_index`` lies in no range of *ranges* (each ``[first, last]``,
    inclusive), in their order. A run with no step index is kept."""
    return [run for run in runs
            if run.get("step_index") is None
            or not any(first <= run["step_index"] <= last for first, last in ranges)]
