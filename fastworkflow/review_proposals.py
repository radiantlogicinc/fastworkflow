"""The framework's review proposals: round 1 (sub-tasks) and round 2 (plan), as typed Pydantic objects.

Plans are GOALS with bound steps, never a bare command list (arch 19.0 invariant 1): a ``PlanGoal`` states
the goal and the commands that are the means to it. Each proposal is registered under its spec name, which
is the name of the planning command that reviews it (``review_subtasks``, ``review_plan``).

``from_text`` reads a proposal from the one-line text a planning command passes to ``review_proposal``:

* sub-tasks: ``goal; goal; ...`` (inputs and choices are then set by edits or by the candidates builder);
* plan: ``goal = step | step; goal = step``.

``build_round1_proposal`` is stage 3: it builds the round-1 proposal from a split result and per-sub-task
candidates. It is pure and used only by tests; no live path calls it yet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, Field

from fastworkflow.review_engine import KIND_PLAN, KIND_SUB_TASKS, PROPOSAL_SPECS, Proposal, register_proposal

FROM_SCRATCH = "from_scratch"

#: The framework's proposal classes by spec name (the review command's lookup).
PROPOSAL_CLASSES = PROPOSAL_SPECS


class SubTask(BaseModel):
    """One sub-task: its goal, its inputs (a value, or ``from sub-task N``), and its choice of means."""

    goal: str
    inputs: str = ""
    choice: str = FROM_SCRATCH   # a planning command name, a plain command name, or from_scratch


@register_proposal
class SubTaskProposal(Proposal):
    kind = KIND_SUB_TASKS
    spec = "review_subtasks"
    title = "Sub-tasks"
    item_template = "{index}. {goal} (inputs: {inputs}; means: {choice})"
    min_items = 1
    items: list[SubTask] = Field(default_factory=list)

    @classmethod
    def from_text(cls, text: str) -> "SubTaskProposal":
        goals = [part.strip() for part in text.split(";") if part.strip()]
        return cls(items=[SubTask(goal=goal) for goal in goals])


class PlanGoal(BaseModel):
    """A goal, and the bound steps (commands with their values) that are its means."""

    goal: str
    steps: list[str] = Field(default_factory=list)


@register_proposal
class PlanProposal(Proposal):
    kind = KIND_PLAN
    spec = "review_plan"
    title = "Plan"
    item_template = "{index}. {goal}: {steps}"
    min_items = 1
    items: list[PlanGoal] = Field(default_factory=list)

    @classmethod
    def from_text(cls, text: str) -> "PlanProposal":
        goals: list[PlanGoal] = []
        for part in text.split(";"):
            if not part.strip():
                continue
            goal, _, steps = part.partition("=")
            goals.append(PlanGoal(goal=goal.strip(),
                                  steps=[step.strip() for step in steps.split("|") if step.strip()]))
        return cls(items=goals)


# ----------------------------------------------------------------------
# Stage 3: the round-1 builder (pure; tests only)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One candidate means for a sub-task: a command or planning command name, and whether it is a planning
    command. Built from the shortlist (run_listing) by the caller."""

    name: str
    is_planning: bool
    score: float = 0.0


def build_round1_proposal(subtasks: list[str],
                          candidates: dict[int, list[Candidate]],
                          *, top_k: int = 3, inputs: Optional[dict[int, str]] = None) -> SubTaskProposal:
    """The round-1 proposal for a split result: one sub-task per goal, each given its best means.

    A planning command among a sub-task's top-k candidates is preferred over a plain command (the
    planning command already knows the steps); otherwise the best candidate; otherwise ``from_scratch``.
    ``candidates`` is keyed by the sub-task's 1-based number. Inputs default to none.
    """
    items: list[SubTask] = []
    for number, goal in enumerate(subtasks, 1):
        ranked = sorted(candidates.get(number, []), key=lambda c: -c.score)[:top_k]
        planning = next((c for c in ranked if c.is_planning), None)
        choice = (planning or (ranked[0] if ranked else None))
        items.append(SubTask(goal=goal, inputs=(inputs or {}).get(number, ""),
                             choice=choice.name if choice else FROM_SCRATCH))
    return SubTaskProposal(items=items)


#: The review command's argument values for each proposal (a value is not a command name).
PROPOSAL_KINDS = {1: SubTaskProposal, 2: PlanProposal}
