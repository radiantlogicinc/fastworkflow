"""The planner's structured plan: what the agent is told to do, and what the finish check verifies.

The planner returns the steps and the subjects the request names as data rather
than prose, because the finish check (``fastworkflow.observation_offloading.finish_check``) has to know,
for every step, which commands it runs, whether it is optional or waits on the
user, and which named subjects it concerns. Reading that back out of free text
depended on how one planner happened to format its list; the fields below do not.

The agent still receives the plan as a numbered list (``render``), so what the
agent reads is the same kind of text it always read.

When the structured call fails, ``build_query_with_next_steps`` falls back to the
plain-text planner and ``parse_text_plan`` recovers what it can: numbered steps,
lettered sub-steps, and the backticked command names that are real commands.
Subjects cannot be recovered from text, so a fallback plan has none and the
finish check then asks only whether each step ran at all.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

from pydantic import BaseModel, Field

import fastworkflow

#: Appended to the planner's instructions when it returns a structured plan.
STRUCTURED_PLAN_GUIDE = """
    Return the plan as structured steps:
    - text: one short sentence per step.
    - commands: the workflow command names the step runs (empty when it only reasons, compares, reports or asks the user).
    - parts: only when a step runs several commands in sequence, one sub-step per command, each with its own commands.
      Alternatives (one command OR another, whichever applies) are not parts: list them together in the step's commands.
    - optional: true for a step or sub-step that is optional or only needed in some cases.
    - needs_user: true for a step that asks the user, or that only runs after the user confirms, approves or selects something.
    Also list the subjects: every specific item the request names and asks about (a person, an account, a record...),
    with its name exactly as written in the request and one lowercase word for its kind.
    """


def workflow_command_names(workflow_path: str) -> Optional[set[str]]:
    """Every command name the workflow declares, in any context; None if unreadable."""
    try:
        definition = fastworkflow.RoutingRegistry.get_definition(workflow_path)
        return {name.split("/")[-1] for names in definition.contexts.values() for name in names}
    except Exception:  # noqa: BLE001 - only a filter for backticked words
        return None


class PlanSubject(BaseModel):
    """A specific item the request names and asks about."""

    name: str = Field(description="The name exactly as written in the request")
    kind: str = Field(description="One lowercase word for what it is, e.g. person, account, permission, control, order")


class PlanPart(BaseModel):
    """One sub-step of a step that runs several commands."""

    text: str = Field(description="The sub-step as one short sentence")
    commands: list[str] = Field(default_factory=list, description="Workflow command names this sub-step runs")
    optional: bool = Field(default=False, description="True if the sub-step is optional or only needed in some cases")


class PlanStep(BaseModel):
    """One step of the plan."""

    text: str = Field(description="The step as one short sentence")
    commands: list[str] = Field(default_factory=list, description="Workflow command names this step runs; empty if it only reasons, reports or asks the user")
    parts: list[PlanPart] = Field(default_factory=list, description="Sub-steps, only when the step runs several commands in sequence")
    optional: bool = Field(default=False, description="True if the whole step is optional or only needed in some cases")
    needs_user: bool = Field(default=False, description="True if the step asks the user, or only runs after the user confirms, approves or selects something")


class TurnPlan(BaseModel):
    """A turn's plan, with where it came from."""

    steps: list[PlanStep] = Field(default_factory=list)
    subjects: list[PlanSubject] = Field(default_factory=list)
    #: "structured" when the planner returned these fields, "text" when they were
    #: parsed from a plain-text plan after the structured call failed.
    source: str = "structured"


def render(steps: Iterable[PlanStep]) -> str:
    """The numbered list the agent reads."""
    lines = []
    for number, step in enumerate(steps, 1):
        flags = [flag for flag, on in (("optional", step.optional), ("needs the user", step.needs_user)) if on]
        suffix = f" ({', '.join(flags)})" if flags else ""
        lines.append(f"{number}. {step.text.strip()}{suffix}")
        for letter, part in zip("abcdefghijklmnopqrstuvwxyz", step.parts):
            optional = " (optional)" if part.optional else ""
            lines.append(f"   {letter}. {part.text.strip()}{optional}")
    return "\n".join(lines)


def step_text(step: PlanStep) -> str:
    """One step with its sub-steps, as a single line of plan text."""
    parts = " ".join(
        f"{letter}. {part.text.strip()}" for letter, part in zip("abcdefghijklmnopqrstuvwxyz", step.parts)
    )
    return f"{step.text.strip()} {parts}".strip()


def command_parts(step: PlanStep) -> list[PlanPart]:
    """The sub-steps that run a command and are required.

    A step with fewer than two such sub-steps is checked as one unit.
    """
    parts = [part for part in step.parts if part.commands and not part.optional]
    return parts if len(parts) >= 2 else []


def is_checked(step: PlanStep) -> bool:
    """Whether the finish check holds the agent to this step."""
    return not (step.optional or step.needs_user)


# ---------------------------------------------------------------------------
# Fallback: a plain-text plan
# ---------------------------------------------------------------------------

_STEP_RE = re.compile(r"^\s*\**\s*(\d+)[.)]\s*")
_PART_RE = re.compile(r"(?:(?<=\s)|^)([a-l])[.)]\s")
_COMMAND_RE = re.compile(r"`([a-z][a-z0-9_]*)")
_OPTIONAL_LEAD_RE = re.compile(
    r"\boptional(?:ly)?\b(?! (?:filter|parameter|argument)s?\b)(?!ly with a filter)|\bif needed\b",
    re.IGNORECASE,
)
_APPROVAL_RE = re.compile(
    r"\bafter (?:the )?(?:user(?:'s)? )?(?:approval|confirmation)\b|\bpending (?:user )?approval\b"
    r"|\buser confirmation required\b|\bonce confirmed\b|\bif (?:the )?user (?:confirms|approves|selects)\b"
    r"|\bafter (?:the )?user (?:selects|confirms|approves)\b|\(after confirmation\)",
    re.IGNORECASE,
)
_LEAD_CHARS = 40


def _lead(text: str) -> str:
    title = re.match(r"^\s*(?:[a-l][.)]\s*)?\*\*(.+?)\*\*", text)
    return title.group(1) if title else text[:_LEAD_CHARS]


def _commands(text: str, known: Optional[set[str]]) -> list[str]:
    found = list(dict.fromkeys(_COMMAND_RE.findall(text)))
    return [c for c in found if known is None or c in known]


def parse_text_plan(text: str, known_commands: Optional[set[str]] = None) -> TurnPlan:
    """Recover steps from a numbered plain-text plan.

    ``known_commands`` filters backticked words to real command names, so a
    backticked field name (``permission_uid``) is not taken for a command.
    """
    steps: list[str] = []
    for line in (text or "").splitlines():
        if _STEP_RE.match(line):
            steps.append(_STEP_RE.sub("", line, count=1).strip())
        elif steps and line.strip():
            steps[-1] += " " + line.strip()
    if not steps and (text or "").strip():
        # A plan squeezed onto one line: "1. a 2. b".
        steps = [s.strip() for s in re.split(r"(?:^|\s)\d+[.)]\s", text) if s.strip()]

    plan_steps = []
    for raw in steps:
        marks = list(_PART_RE.finditer(raw))
        head = raw[: marks[0].start()] if len(marks) >= 2 else raw
        parts = []
        if len(marks) >= 2:
            for index, mark in enumerate(marks):
                end = marks[index + 1].start() if index + 1 < len(marks) else len(raw)
                part_text = raw[mark.end(): end].strip()
                parts.append(PlanPart(
                    text=part_text,
                    commands=_commands(part_text, known_commands),
                    optional=bool(_OPTIONAL_LEAD_RE.search(_lead(part_text)) or _APPROVAL_RE.search(part_text)),
                ))
        lead = _lead(head)
        plan_steps.append(PlanStep(
            text=head.strip() or raw.strip(),
            commands=_commands(raw, known_commands),
            parts=parts,
            optional=bool(_OPTIONAL_LEAD_RE.search(lead)),
            needs_user=bool(_APPROVAL_RE.search(lead)),
        ))
    return TurnPlan(steps=plan_steps, subjects=[], source="text")
