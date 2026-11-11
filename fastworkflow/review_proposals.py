"""The framework's review proposals: round 1 (sub-tasks) and round 2 (plan), as typed Pydantic objects.

Plans are GOALS with bound steps, never a bare command list (arch 19.0 invariant 1): a ``PlanGoal`` states
the goal and the commands that are the means to it. Each proposal is registered under its spec name, which
is the name of the planning command that reviews it (``review_subtasks``, ``review_plan``).

``from_text`` reads a proposal from the one-line text a planning command passes to ``review_proposal``:

* sub-tasks: ``goal; goal; ...`` (outcomes, inputs and means are then set by edits or by the candidates builder);
* plan: ``goal = step | step; goal = step``.

``build_round1_proposal`` is stage 3: it builds the round-1 proposal from a split result and per-sub-task
candidates. It is pure. ``draft_round1_proposal`` feeds it real candidates (the heads, the shortlist, optional Jev) for
the turn's gate, and ``fast_path_for`` is the gate's fast-path predicate (fix-3rsq.plan-review.2.2, .2.3). The gate
itself is in workflow_execution_context (``_subtask_gate``).
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional

import dspy
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from fastworkflow import turn_cache
from fastworkflow.review_engine import (
    KIND_PLAN, KIND_SUB_TASKS, PROPOSAL_SPECS, Add, Change, Drop, EditParseError, EditRefused, Merge, Proposal,
    Split, _SafeValues, register_proposal)

logger = logging.getLogger(__name__)

FROM_SCRATCH = "from_scratch"
MEANS_PLANNING = "planning"
MEANS_PLAIN = "plain"
MEANS_FROM_SCRATCH = "from_scratch"

#: The framework's proposal classes by spec name (the review command's lookup).
PROPOSAL_CLASSES = PROPOSAL_SPECS


class Means(BaseModel):
    """How a sub-task is done: a prebuilt planning command, a plain command, or built from scratch (round 2).

    Text forms (the reply reader and the rendering): ``planning:NAME``, ``plain:NAME`` and ``from_scratch``.
    """

    kind: Literal["planning", "plain", "from_scratch"] = MEANS_FROM_SCRATCH
    name: str = ""

    @model_validator(mode="after")
    def _name_matches_kind(self):
        if self.kind == MEANS_FROM_SCRATCH and self.name:
            raise ValueError("a from_scratch sub-task names no command")
        if self.kind != MEANS_FROM_SCRATCH and not self.name:
            raise ValueError(f"a {self.kind} sub-task names its command")
        return self

    def render(self) -> str:
        return MEANS_FROM_SCRATCH if self.kind == MEANS_FROM_SCRATCH else f"{self.kind}:{self.name}"

    @classmethod
    def from_text(cls, text: str) -> "Means":
        """The text form of a means (``planning:NAME``, ``plain:NAME``, ``from_scratch``); anything else refused."""
        text = text.strip()
        if text == MEANS_FROM_SCRATCH:
            return cls(kind=MEANS_FROM_SCRATCH)
        kind, _, name = text.partition(":")
        if kind in (MEANS_PLANNING, MEANS_PLAIN) and name.strip():
            return cls(kind=kind, name=name.strip())
        raise ValueError("means must be from_scratch, planning:NAME or plain:NAME")

    def __str__(self) -> str:
        return self.render()


_FROM_SUBTASK = re.compile(r"^(?:(?P<name>[A-Za-z_]\w*)\s*=\s*)?from sub-task\s+(?P<number>\d+)$", re.IGNORECASE)
_NAMED_VALUE = re.compile(r"^(?P<name>[A-Za-z_]\w*)\s*=\s*(?P<value>.*)$", re.DOTALL)
_NAMED_COLON = re.compile(r"^(?P<name>[A-Za-z_]\w*)\s*:\s*(?P<value>.*)$", re.DOTALL)
_ENTITY_INPUT_KEYS = frozenset({"name", "unit", "person", "right", "group", "application", "feed", "control"})


class SubTaskInput(BaseModel):
    """One input of a sub-task: a value the user gave, or ``from sub-task N`` (an earlier sub-task's output).

    Exactly one of ``value`` and ``from_subtask`` is set. ``name`` is the input's name when it has one.
    """

    name: str = ""
    value: Optional[str] = None
    from_subtask: Optional[int] = Field(default=None, ge=1)
    #: The framework drafted it (``draft_inputs``: the shared extractor on the sub-task's text, or a declared
    #: producer that an earlier sub-task runs), not the user. Shown like any input; see ``resolve_inputs``.
    drafted: bool = False

    @model_validator(mode="after")
    def _exactly_one_source(self):
        if (self.value is None) == (self.from_subtask is None):
            raise ValueError("an input is a value or 'from sub-task N', not both and not neither")
        return self

    def render(self) -> str:
        text = f"from sub-task {self.from_subtask}" if self.from_subtask is not None else str(self.value)
        return f"{self.name}={text}" if self.name else text

    @classmethod
    def from_text(cls, text: str) -> "SubTaskInput":
        """Read one input's text: ``name=value``, ``name=from sub-task N``, ``from sub-task N`` or a bare value."""
        text = text.strip()
        if match := _FROM_SUBTASK.match(text):
            return cls(name=match.group("name") or "", from_subtask=int(match.group("number")))
        if match := _NAMED_VALUE.match(text):
            return cls(name=match.group("name"), value=_strip_outer_quotes(match.group("value")))
        if match := _NAMED_COLON.match(text):
            return cls(name=match.group("name"), value=_strip_outer_quotes(match.group("value")))
        return cls(value=_strip_outer_quotes(text))


class SubTask(BaseModel):
    """One sub-task of a request: its goal, its desired outcome, its inputs, and its means.

    ``goal`` is the sub-task's prompt (what the user asked for this piece). ``outcome`` is the desired outcome, as
    text: a predicate about the world ("Ann holds a loan on Dune"), not a command. ``means`` is the bound means: a
    planning command, a plain command, or ``from_scratch``.

    Arch 19.0 invariants 1 and 3: a sub-task is a goal with bound means, never a bare command list, and no goal is
    completed because its bound command returned success. A goal is complete only when the finish check
    (``observation_offloading/finish_check``) confirms its outcome; ``CommandOutput.success`` never marks one done.
    Round 1 keeps ``outcome`` as the user reviewed it and does not verify it.
    """

    goal: str
    outcome: str = ""
    inputs: list[SubTaskInput] = Field(default_factory=list)
    means: Means = Field(default_factory=Means)
    #: What the framework's check said about the means, shown on the line: a refused planning-command argument
    #: (``check_planned_arguments``). Empty when the check passed or no check ran. Not a completion: an advisory note.
    check: str = ""
    #: The means the draft drafted this sub-task's inputs for (``draft_inputs`` ran on its goal), or "" when it did not
    #: (drafting off, the draft failed, or the user added the sub-task; a split or merged one keeps its first item's).
    #: fix-48np.1: a sub-task whose means is still the drafted one is not drafted again at run time
    #: (``reviewed_arguments``). Not shown.
    drafted_means: str = Field(default="", json_schema_extra={"review_hidden": True})
    #: The sub-tasks this one joins, when the split had more than MAX_SUBTASKS and ``split_merge.reduce_to_cap`` joined some
    #: (``JoinedSubtask``): the review shows the joined item, and a run whose item is still as drafted runs each part as
    #: a job of its own (``expand_joined_parts``). ``parts_fingerprint`` is the item as drafted; an edit of the goal, outcome,
    #: inputs or means makes it differ, and the item then runs as the one job it reads. Not shown.
    parts: list[str] = Field(default_factory=list, json_schema_extra={"review_hidden": True})
    parts_fingerprint: str = Field(default="", json_schema_extra={"review_hidden": True})
    #: The item this one came from before this reply (set by ``apply_edits``; not serialized).
    _before_item: Optional["SubTask"] = PrivateAttr(default=None)
    #: This reply asked for look_up on this item (set by ``apply_edits``; not serialized).
    _look_up: bool = PrivateAttr(default=False)

    @field_validator("inputs", mode="before")
    @classmethod
    def _inputs_from_text(cls, value: Any) -> Any:
        """Edits and the one-line form give inputs as text: each piece is read by ``SubTaskInput.from_text``."""
        if isinstance(value, str):
            value = [value] if value.strip() else []
        return [SubTaskInput.from_text(item) if isinstance(item, str) else item for item in (value or [])]

    @field_validator("means", mode="before")
    @classmethod
    def _means_from_text(cls, value: Any) -> Any:
        return Means.from_text(value) if isinstance(value, str) else value


_MULTI_INPUT_SPLIT = re.compile(
    r"(?:[,;]|\s+and\s+)\s*(?=[A-Za-z_]\w*\s*[=:]|from sub-task\s+\d+)",
    re.IGNORECASE,
)
_META_INPUT_KEYS = frozenset({"means", "goal", "outcome", "desired_outcome", "check"})


def _strip_outer_quotes(value: str) -> str:
    stripped = (value or "").strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in "\"'":
        return stripped[1:-1].strip()
    return stripped


def extract_quoted_names(text: str) -> list[str]:
    """Literal entity names the user wrapped in quotes (not contractions like don't)."""
    if not text:
        return []
    found: list[str] = []
    for match in re.finditer(r'"([^"\n]{2,})"', text):
        name = match.group(1).strip()
        if name and name not in found:
            found.append(name)
    for match in re.finditer(r"(?<![A-Za-z0-9])'([^'\n]{2,})'(?![A-Za-z0-9])", text):
        name = match.group(1).strip()
        if name and name not in found:
            found.append(name)
    return found


def _split_input_assignments(stripped: str) -> list[str]:
    """Split key=value fragments without breaking quoted values."""
    if not (_NAMED_VALUE.match(stripped) or _NAMED_COLON.match(stripped) or _FROM_SUBTASK.match(stripped)):
        return [stripped]
    pieces: list[str] = []
    start = 0
    in_quote: Optional[str] = None
    index = 0
    while index < len(stripped):
        char = stripped[index]
        if in_quote:
            if char == in_quote:
                in_quote = None
            index += 1
            continue
        if char in "\"'":
            in_quote = char
            index += 1
            continue
        if char in ",;":
            rest = stripped[index + 1 :]
            if re.match(r"\s*(?=[A-Za-z_]\w*\s*[=:]|from sub-task\s+\d+)", rest, re.I):
                piece = stripped[start:index].strip()
                if piece:
                    pieces.append(piece)
                start = index + 1
            index += 1
            continue
        match = re.match(r"\s+and\s+(?=[A-Za-z_]\w*\s*[=:])", stripped[index:], re.I)
        if match:
            piece = stripped[start:index].strip()
            if piece:
                pieces.append(piece)
            start = index + match.end()
            index = start
            continue
        index += 1
    tail = stripped[start:].strip()
    if tail:
        pieces.append(tail)
    return pieces if pieces else [stripped]


def _input_edit_part_key(part: str) -> Optional[str]:
    stripped = (part or "").strip()
    if match := _NAMED_VALUE.match(stripped):
        return match.group("name").lower()
    if match := _NAMED_COLON.match(stripped):
        return match.group("name").lower()
    return None


def _input_edit_parts(stripped: str) -> list[str]:
    """Split an ``inputs`` edit into separate key=value pieces; commas inside values stay intact."""
    if _NAMED_VALUE.match(stripped) or _NAMED_COLON.match(stripped) or _FROM_SUBTASK.match(stripped):
        parts = _split_input_assignments(stripped)
        if len(parts) > 1:
            kept: list[str] = []
            for part in parts:
                text = part.strip()
                if not text:
                    continue
                key = _input_edit_part_key(text)
                if key and key in _META_INPUT_KEYS:
                    continue
                kept.append(text)
            return kept if kept else [stripped]
    return [stripped]


#: An input's value that clears the input: ``kind=none`` removes ``kind``.
_CLEARING_VALUES = frozenset({"none", "null"})


def _is_clearing_change(edit: Change) -> bool:
    """A one-input clearing change (``name=none``): it is applied on its own, never merged with other input edits."""
    if edit.field != "inputs":
        return False
    parts = _input_edit_parts((edit.value or "").strip())
    return (len(parts) == 1 and _input_edit_part_key(parts[0]) is not None
            and (SubTaskInput.from_text(parts[0]).value or "").strip().lower() in _CLEARING_VALUES)


def _merge_one_subtask_input(existing: list[SubTaskInput], text: str) -> list[SubTaskInput]:
    """Update or add one input by name and keep the item's other inputs."""
    new_input = SubTaskInput.from_text(text)
    result = list(existing)
    if new_input.name:
        if (new_input.value or "").strip().lower() in _CLEARING_VALUES:
            # ``name=none`` clears that one input (``inputs=none`` clears them all).
            return [each for each in result if each.name != new_input.name]
        for position, each in enumerate(result):
            if each.name == new_input.name:
                result[position] = new_input
                return result
        result.append(new_input)
        return result
    for position, each in enumerate(result):
        if not each.name:
            result[position] = new_input
            return result
    result.append(new_input)
    return result


def merge_subtask_input_edit(existing: list[SubTaskInput], text: str) -> list[SubTaskInput]:
    """One ``inputs`` change: update or add inputs and keep the item's other inputs (commas stay in values)."""
    stripped = (text or "").strip()
    if stripped.lower() == "none":
        return []
    result = list(existing)
    for part in _input_edit_parts(stripped):
        result = _merge_one_subtask_input(result, part)
    return result


def _inputs_user_kept(before: SubTask | None, after: SubTask) -> list[SubTaskInput]:
    """Inputs *after* carries that differ from *before* (set in the same reply as a goal or means change)."""
    if before is None:
        return list(after.inputs)
    before_named = {each.name: each for each in before.inputs if each.name}
    kept: list[SubTaskInput] = []
    for each in after.inputs:
        if each.name:
            if each.name not in before_named or each != before_named[each.name]:
                kept.append(each)
        else:
            unnamed_before = [inp for inp in before.inputs if not inp.name]
            if not unnamed_before or each != unnamed_before[0]:
                kept.append(each)
    return kept


def _overlay_inputs(
    drafted: SubTask,
    kept: list[SubTaskInput],
    fallback_existing: Optional[list[SubTaskInput]] = None,
) -> SubTask:
    if fallback_existing and kept:
        merged = list(fallback_existing)
        for each in kept:
            merged = merge_subtask_input_edit(merged, each.render())
        return drafted.model_copy(update={"inputs": merged})
    if not kept:
        return drafted
    merged = merge_subtask_input_edit(drafted.inputs, kept[0].render())
    for each in kept[1:]:
        merged = merge_subtask_input_edit(merged, each.render())
    return drafted.model_copy(update={"inputs": merged})


def _input_user_line(inp: SubTaskInput) -> str:
    if inp.from_subtask is not None:
        text = f"from sub-task {inp.from_subtask}"
    else:
        text = inp.value or ""
    return f"{inp.name}: {text}" if inp.name else text


@register_proposal
class SubTaskProposal(Proposal):
    kind = KIND_SUB_TASKS
    spec = "review_subtasks"
    title = "Sub-tasks"
    item_template = "{index}. {goal} (inputs: {inputs}; means: {means}); desired outcome: {outcome}{check}"
    min_items = 1
    items: list[SubTask] = Field(default_factory=list)

    def user_rendering(self, workflow_folderpath: Optional[str] = None) -> str:
        from fastworkflow.command_writes import write_commands

        writes = write_commands(workflow_folderpath) if workflow_folderpath else frozenset()
        lines = [f"{self.title}:"]
        for index, item in enumerate(self.items, 1):
            input_bits = [_input_user_line(each) for each in item.inputs]
            suffix = f" ({', '.join(input_bits)})" if input_bits else ""
            write_badge = ""
            if item.means.kind != MEANS_FROM_SCRATCH and item.means.name in writes:
                write_badge = " - changes data"
            lines.append(f"{index}. {item.goal}{suffix}{write_badge}")
        return "\n".join(lines)

    def model_rendering(self) -> str:
        import string

        lines = [f"{self.title}:"]
        for index, item in enumerate(self.items, 1):
            values = self.item_values(item)
            values["index"] = index
            try:
                lines.append(string.Formatter().vformat(self.item_template, (), _SafeValues(values)))
            except (ValueError, AttributeError):
                lines.append(f"{index}. " + ", ".join(f"{k}={v}" for k, v in values.items() if k != "index"))
        return "\n".join(lines)

    def rendering(self) -> str:
        return self.user_rendering()

    def revised(self, host: Any, before: Proposal) -> "SubTaskProposal":
        """With drafting on, re-draft a sub-task when its goal, means, add or look_up changed (fix-48np.1, fix-kk46.3).

        Inputs the user set in the same reply are kept: re-draft, then overlay those inputs.
        """
        workflow = getattr(host, "_app_workflow", None)
        if workflow is None or not drafting_on():
            return self
        before_items = list(before.items) if isinstance(before, SubTaskProposal) else []
        planner_lm = getattr(host, "_current_planner_lm", None)
        items: list[SubTask] = []
        for position, item in enumerate(self.items):
            before_item = getattr(item, "_before_item", None)
            if before_item is None and before_items:
                if item in before_items:
                    before_item = before_items[before_items.index(item)]
                else:
                    matched_swapped = next(
                        (b for b in before_items if (b.goal, b.means, b.inputs) == (item.goal, item.means, item.inputs)),
                        None,
                    )
                    if matched_swapped is not None:
                        item = matched_swapped.model_copy()
                        before_item = matched_swapped
                    elif len(self.items) == len(before_items):
                        before_item = before_items[position]
            kept = _inputs_user_kept(before_item, item)
            if getattr(item, "_look_up", False):
                item = redraft_for_look_up(workflow, item, items, planner_lm=planner_lm)
                item = _overlay_inputs(item, kept)
            elif before_item is None and item.means.kind == MEANS_FROM_SCRATCH:
                picked = build_round1_proposal(
                    [item.goal], {1: _candidates_for_subtask(workflow, item.goal)}).items[0].means
                item = item.model_copy(update={"means": picked})
                if item.means.kind != MEANS_FROM_SCRATCH:
                    item = drafted_item(workflow, item, item.goal, items, planner_lm=planner_lm)
                    item = _overlay_inputs(item, kept)
            elif (before_item is not None and item.goal != before_item.goal
                  and item.means == before_item.means):
                picked = build_round1_proposal(
                    [item.goal], {1: _candidates_for_subtask(workflow, item.goal)}).items[0].means
                if picked.kind != MEANS_FROM_SCRATCH:
                    item = item.model_copy(update={"means": picked})
                item = drafted_item(workflow, item, item.goal, items, planner_lm=planner_lm)
                overlay_fallback = (
                    before_item.inputs
                    if (kept and item.means == before_item.means)
                    else None
                )
                item = _overlay_inputs(item, kept, fallback_existing=overlay_fallback)
            elif (before_item is not None and item.means.kind != MEANS_FROM_SCRATCH
                  and (item.means != before_item.means or item.drafted_means != item.means.name)):
                item = drafted_item(workflow, item, item.goal, items, planner_lm=planner_lm)
                item = _overlay_inputs(item, kept)
            items.append(item)
        return self.model_copy(update={"items": items})

    def item_values(self, item: Any) -> dict[str, Any]:
        values = super().item_values(item)
        values["inputs"] = ", ".join(each.render() for each in item.inputs) or "none"
        values["means"] = item.means.render()
        values["outcome"] = item.outcome or "not stated"
        values["check"] = f" [check: {item.check}]" if item.check else ""
        return values

    @classmethod
    def from_text(cls, text: str) -> "SubTaskProposal":
        """A proposal from its text: ``goal; goal; ...`` (the goals only), or the JSON of a draft (a list of sub-task
        objects, or ``{"items": [...]}``), which is how the gate passes its candidates' draft to the review."""
        stripped = (text or "").strip()
        # A step passes the draft as text quoted by repr (single quotes around JSON that holds double quotes).
        if len(stripped) >= 2 and stripped[0] in "'\"" and stripped[-1] == stripped[0]:
            stripped = stripped[1:-1].strip()
        if stripped[:1] in ("{", "["):
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError:
                data = None
            if isinstance(data, (dict, list)):
                return cls(items=data["items"] if isinstance(data, dict) else data)
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
                          *, top_k: int = 3, inputs: Optional[dict[int, list[SubTaskInput]]] = None,
                          outcomes: Optional[dict[int, str]] = None) -> SubTaskProposal:
    """The round-1 proposal for a split result: one sub-task per goal, each given its best means.

    A planning command among a sub-task's top-k candidates is preferred over a plain command (the
    planning command already knows the steps); otherwise the best candidate; otherwise ``from_scratch``.
    ``candidates`` is keyed by the sub-task's 1-based number. Inputs and outcomes default to none.
    """
    items: list[SubTask] = []
    for number, goal in enumerate(subtasks, 1):
        ranked = sorted(candidates.get(number, []), key=lambda c: -c.score)[:top_k]
        planning = next((c for c in ranked if c.is_planning), None)
        choice = (planning or (ranked[0] if ranked else None))
        if choice is None:
            means = Means()
        else:
            means = Means(kind=MEANS_PLANNING if choice.is_planning else MEANS_PLAIN, name=choice.name)
        items.append(SubTask(goal=goal, outcome=(outcomes or {}).get(number, ""),
                             inputs=list((inputs or {}).get(number, [])), means=means))
    return SubTaskProposal(items=items)


#: The review command's argument values for each proposal (a value is not a command name).
PROPOSAL_KINDS = {1: SubTaskProposal, 2: PlanProposal}


# ----------------------------------------------------------------------
# The gate's predicates and candidates (fix-3rsq.plan-review.2.2, .2.3)
# ----------------------------------------------------------------------

#: The ``items`` text the gate passes the review with: the draft itself is handed over through ``draft_scope`` (a step's
#: text is parsed as a command's arguments, which would cut a JSON draft at its first space).
DRAFT_ITEMS = "draft"
_draft_var: contextvars.ContextVar[Optional["SubTaskProposal"]] = contextvars.ContextVar(
    "fw_review_draft", default=None)


@contextlib.contextmanager
def draft_scope(proposal: "SubTaskProposal"):
    """The draft a review opened with ``DRAFT_ITEMS`` puts to the user, for the length of the framework call that opens it."""
    token = _draft_var.set(proposal)
    try:
        yield
    finally:
        _draft_var.reset(token)


def draft_for(items_text: str) -> Optional["SubTaskProposal"]:
    """The gate's draft when a review is asked with ``DRAFT_ITEMS`` inside ``draft_scope``; None otherwise."""
    return _draft_var.get() if items_text == DRAFT_ITEMS else None


#: The fast-path reasons (what let one sub-task run without a review).
FAST_EXACT_NAME = "exact_name"
FAST_HEAD_CONFIDENT = "head_confident"


@dataclass(frozen=True)
class FastPath:
    """A sub-task that runs without a review: its command (qualified name), whether it is a planning command, and why."""

    name: str
    is_planning: bool
    reason: str


def fast_path_for(workflow, subtasks: list[str]) -> Optional[FastPath]:
    """The fast-path predicate (owner decision 1): one sub-task, with an exact command name or a confident head match.

    "Confident" is the signal intent detection routes alone on: the trained head's classifier was confident, so
    ``CommandRouter.predict_with_details`` answered one label (its ``confident`` flag), which intent detection takes
    without asking. Its threshold is the head's own, not a new one. An exact command name is intent detection's
    exact-prefix match (``run_listing.exact_command_name``). Several sub-tasks, an unsure head, or no match is None.
    No model call: the heads run locally.
    """
    from fastworkflow import planning_commands, run_listing

    if len(subtasks) != 1:
        return None
    sub_task = subtasks[0]
    planning = set(planning_commands.resolve_field_sources(workflow.folderpath).planning_commands)
    if name := run_listing.exact_command_name(workflow, sub_task):
        return FastPath(name, name in planning, FAST_EXACT_NAME)
    confident = run_listing.head_candidates(workflow, sub_task).confident_labels
    if len(confident) == 1:
        (name,) = confident
        return FastPath(name, name in planning, FAST_HEAD_CONFIDENT)
    return None


def planning_intent_confirmation(workflow, path: FastPath, sub_task: str) -> Optional[str]:
    """HOOK for fix-3rsq.plan-review.6: the one-line intent confirmation a planning command taken on the fast path
    gets ("I'll run X: step -> step. OK?"). Round 1 asks nothing and returns None; the gate does not call it yet."""
    return None


def _candidates_for_subtask(workflow, sub_task: str, *, exclude: Optional[set[str]] = None) -> list[Candidate]:
    """Per-sub-task candidate means for the round-1 builder (``draft_round1_proposal``, look_up, add)."""
    from fastworkflow import planning_commands, run_listing
    from fastworkflow.workflow_agent import core_command_names

    folder = workflow.folderpath
    planning = set(planning_commands.resolve_field_sources(folder).planning_commands)
    core = core_command_names()
    excluded = exclude or set()
    heads = run_listing.head_candidates(workflow, sub_task)
    _, _, probabilities = planning_match(workflow, sub_task)
    scores = {**heads.execution_scores, **heads.planning_scores}
    for name, probability in probabilities.items():
        scores[name] = max(scores.get(name, 0.0), probability)
    named = run_listing.gather(workflow, sub_task)
    names = (run_listing.gather(workflow, sub_task, probabilities or None) | heads.labels
             | heads.planning_scores.keys() | heads.execution_scores.keys())
    return [Candidate(name, is_planning=name in planning, score=scores.get(name, 0.0))
            for name in sorted(names)
            if name not in core and name not in excluded
            and (name in named or scores.get(name, 0.0) >= run_listing.UNSURE_CANDIDATE_FLOOR)]


def draft_round1_proposal(workflow, subtasks: list[str], planner_lm=None) -> SubTaskProposal:
    """The draft round-1 proposal for a split result, from real candidates (fix-3rsq.plan-review.2.3).

    Per sub-task, the candidates are: the heads' planning and execution top-k (``run_listing.head_candidates``), the
    names the gather reads (``run_listing.gather``: the names the sub-task writes, the heads' labels and, when
    ``JEV_API_KEY`` is set, Jev's planning-command distribution), and Jev's confident or unsure planning commands. Jev is
    the existing matcher (``workflow_agent.match_planning_command``): it runs no model without the key and then leaves the
    distribution empty. ``build_round1_proposal`` picks each sub-task's means (a planning command in the top-k first).
    Each sub-task with a command for its means gets its inputs drafted from its text (``draft_inputs``, fix-48np:
    the shared parameter extractor, and for a planning command the typed fill fallback on ``planner_lm``; the values
    the review shows are the ones the run uses, fix-48np.1). A planning pick's refusal is shown on its line: the
    drafted values' refusal, or, with no draft, ``check_planned_arguments`` of no arguments.
    """
    candidates = {number: _candidates_for_subtask(workflow, sub_task)
                    for number, sub_task in enumerate(subtasks, 1)}
    proposal = build_round1_proposal(subtasks, candidates)
    items: list[SubTask] = []
    for item, sub_task in zip(proposal.items, subtasks):
        drafted = drafted_item(workflow, item, sub_task, items, planner_lm=planner_lm)
        if parts := list(getattr(sub_task, "parts", ()) or ()):
            drafted = drafted.model_copy(update={"parts": parts})
            drafted = drafted.model_copy(update={"parts_fingerprint": _fingerprint(drafted)})
        items.append(drafted)
    return SubTaskProposal(items=items)


def _fingerprint(item: SubTask) -> str:
    """The item as the draft made it: what an edit of the goal, outcome, inputs or means changes."""
    return json.dumps([item.goal, item.outcome, item.means.render(), [each.model_dump() for each in item.inputs]],
                      sort_keys=True, default=str)


def expand_joined_parts(workflow, subtasks: list[str], plan: Optional[SubTaskProposal], planner_lm=None
                        ) -> tuple[list[str], Optional[SubTaskProposal]]:
    """``(subtasks, plan)`` with each joined item that is still as drafted replaced by its parts, in order.

    One level only: each part is drafted on its own (``draft_round1_proposal``, as a split sub-task is) and is a leaf. An
    item whose goal, outcome, inputs or means the review changed runs as the one job it now reads. ``from sub-task N``
    inputs of the other items follow the renumbering (N names the first part of a joined item N). Nothing changes when
    no item has parts, so a split within the cap is run exactly as before.
    """
    if plan is None or len(plan.items) != len(subtasks) or not any(item.parts for item in plan.items):
        return subtasks, plan
    first_number: dict[int, int] = {}
    expanded: list[SubTask] = []
    goals: list[str] = []
    for number, item in enumerate(plan.items, 1):
        first_number[number] = len(expanded) + 1
        if item.parts and item.parts_fingerprint == _fingerprint(item):
            for leaf in draft_round1_proposal(workflow, list(item.parts), planner_lm=planner_lm).items:
                expanded.append(leaf.model_copy(update={"parts": [], "parts_fingerprint": ""}))
                goals.append(leaf.goal)
        else:
            expanded.append(item.model_copy(update={"parts": [], "parts_fingerprint": ""}))
            goals.append(subtasks[number - 1])
    renumbered = []
    for item in expanded:
        inputs = [each.model_copy(update={"from_subtask": first_number.get(each.from_subtask, each.from_subtask)})
                  if each.from_subtask is not None else each for each in item.inputs]
        renumbered.append(item.model_copy(update={"inputs": inputs}))
    return goals, SubTaskProposal(items=renumbered)


def drafted_item(workflow, item: SubTask, sub_task: str, earlier: list[SubTask], planner_lm=None) -> SubTask:
    """*item* with its inputs drafted from *sub_task* for its means (``draft_inputs``), its planning check, and the means
    it was drafted for (``drafted_means``). With no draft (drafting off, from scratch, a failed draft) the inputs are
    empty and a planning pick's check is ``check_planned_arguments`` of no arguments."""
    from fastworkflow import planning_commands, run_listing

    check = ""
    inputs: list[SubTaskInput] = []
    drafted = (draft_inputs(workflow, item, sub_task, earlier, planner_lm=planner_lm)
               if item.means.kind != MEANS_FROM_SCRATCH else None)
    if drafted is not None:
        inputs = list(drafted.inputs)
    if item.means.kind == MEANS_PLANNING:
        command = planning_commands.planning_command(workflow.folderpath, item.means.name)
        if drafted is not None:
            check = drafted.refusal or ""
        elif command is not None:
            refusal = run_listing.check_planned_arguments(command, {}, sub_task).refusal
            check = refusal or ""
    return item.model_copy(update={"check": check, "inputs": inputs,
                                   "drafted_means": item.means.name if drafted is not None else ""})


#: ``FW_DRAFT_ARGUMENTS`` (on by default since fix-48np.1): the draft fills each sub-task's inputs from the shared
#: extractor, and a bound sub-task then runs with the inputs the review shows (``reviewed_values``). ``off`` (or 0,
#: false, no) leaves the inputs empty and the run as before fix-48np.
DRAFT_ARGUMENTS_ENV_VAR = "FW_DRAFT_ARGUMENTS"


def drafting_on() -> bool:
    """Whether ``FW_DRAFT_ARGUMENTS`` is on (the default): the draft fills inputs, and the run takes them as its
    arguments. Only ``off``, ``0``, ``false`` or ``no`` turns it off."""
    import fastworkflow

    value = fastworkflow.get_env_var(DRAFT_ARGUMENTS_ENV_VAR, default="on")
    return str(value if value is not None else "on").strip().lower() not in ("off", "0", "false", "no")


@dataclass(frozen=True)
class DraftedInputs:
    """A sub-task's drafted inputs (each ``drafted``), and why its values do not stand (None when they do)."""

    inputs: tuple[SubTaskInput, ...] = ()
    refusal: Optional[str] = None


def _input_text(value: Any) -> str:
    """An argument as an input's text, in the form ``check_arguments`` reads back (a bool true/false, a list as JSON)."""
    from enum import Enum

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, (list, tuple)):
        return json.dumps([str(v) for v in value])
    return str(value)


def _producers(workflow, means: Means, earlier: list[SubTask]) -> dict[str, int]:
    """``{field: N}``: the fields of *means*'s ``Input`` whose declared producer (``from_command_output``, resolved as
    ``available_from``) an earlier sub-task N runs, as its command or as a step of its planning command; the latest N."""
    from fastworkflow import ModuleType, RoutingRegistry, planning_commands

    folder = workflow.folderpath
    params_class = RoutingRegistry.get_definition(folder).get_command_class(
        means.name, ModuleType.COMMAND_PARAMETERS_CLASS)
    if params_class is None:
        return {}
    callees = planning_commands.resolve_field_sources(folder).callees

    def runs(item: SubTask) -> set[str]:
        if item.means.kind == MEANS_FROM_SCRATCH:
            return set()
        names = {item.means.name, *callees.get(item.means.name, ())}
        return names | {name.rpartition("/")[2] for name in names}

    out: dict[str, int] = {}
    for name, info in params_class.model_fields.items():
        extra = info.json_schema_extra if isinstance(info.json_schema_extra, dict) else {}
        producers = {str(ref).rpartition(".")[0] for ref in extra.get("available_from") or ()}
        producers |= {p.rpartition("/")[2] for p in producers}
        for number, item in enumerate(earlier, 1):
            if producers & runs(item):
                out[name] = number
    return out


def _given(info: Any, value: Any) -> bool:
    from pydantic_core import PydanticUndefined

    from fastworkflow.utils.signatures import NOT_FOUND

    if value is None or (isinstance(value, str) and value.strip() in ("", NOT_FOUND, "None")):
        return False
    if isinstance(value, (list, tuple)) and not value:
        return False
    default = info.default
    return default is PydanticUndefined or default is None or value != default


def draft_inputs(workflow, item: SubTask, sub_task: str, earlier: list[SubTask],
                 planner_lm=None) -> Optional[DraftedInputs]:
    """The inputs of *item*'s means drafted from *sub_task* (fix-48np, stage 1), or None when no draft was made.

    A field whose declared producer (``from_command_output``) an earlier sub-task runs is ``from sub-task N``, not extracted. The
    other values are the shared extractor's (``parameter_extraction.extract_arguments``: the extractor execution
    commands use, with its checks, and for a planning command ``check_arguments``, the checks the typed planning call
    runs), each one the extractor gave and the checks kept that is not the field's default. For a planning command
    whose required value is still missing or refused after the extractor's retry, the typed fill fallback runs, as it
    does in ``typed_planning_call`` (fix-48np.1: one planner call, on *planner_lm* or LLM_PLANNER; the same checks
    after it), so the draft and the run follow the same rules. Values that are still refused, or a required value
    that is still missing, give the refusal and no values. None when the switch is off (``FW_DRAFT_ARGUMENTS=off``; on
    by default) or the extraction fails: the draft is advisory and never fails the turn.
    """
    from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import extract_arguments

    if not drafting_on():
        return None
    try:
        from_earlier = _producers(workflow, item.means, earlier)
        result = extract_arguments(workflow, item.means.name, sub_task)
    except Exception as exc:  # noqa: BLE001 - the draft is advisory: no inputs, as before
        # A warning, not a silent pass: a failure here (a provider error, a test's no-model guard) stays visible.
        logger.warning("could not draft the inputs of %s (%s: %s); the draft shows none",
                       item.means.name, type(exc).__name__, exc)
        return None
    inputs = [SubTaskInput(name=name, from_subtask=number, drafted=True) for name, number in from_earlier.items()]
    if result.params is None:
        return DraftedInputs(inputs=tuple(inputs))
    covered = result.missing and set(result.missing) <= set(from_earlier) and result.refusal is not None \
        and result.refusal.startswith("missing required argument")
    fields = type(result.params).model_fields
    values = {name: getattr(result.params, name, None) for name in fields}
    if not result.valid and not covered:
        filled = _typed_fallback_values(workflow, item, sub_task, values, result.refusal, planner_lm)
        if filled is None:
            return DraftedInputs(inputs=tuple(inputs), refusal=result.refusal)
        values = filled
    for name, info in fields.items():
        value = values.get(name)
        if name in from_earlier or not _given(info, value):
            continue
        inputs.append(SubTaskInput(name=name, value=_input_text(value), drafted=True))
    return DraftedInputs(inputs=tuple(inputs))


def _typed_fallback_values(workflow, item: SubTask, sub_task: str, values: dict, refusal: Optional[str],
                           planner_lm=None) -> Optional[dict]:
    """The checked arguments of a planning means after the typed fill fallback (``typed_fallback_arguments``, the
    fallback ``typed_planning_call`` runs), or None when the means is not a planning command, the fallback fills
    nothing, or the checks still refuse: the extractor's refusal then stands. Never raises: the draft is advisory."""
    from fastworkflow import planning_commands

    if item.means.kind != MEANS_PLANNING:
        return None
    command = planning_commands.planning_command(workflow.folderpath, item.means.name)
    if command is None or not command.params:
        return None
    from fastworkflow.utils import dspy_utils
    from fastworkflow.workflow_agent import CommandsSystemPreludeAdapter, typed_planner_commands_text

    raw = {p.name: planning_commands._given_or_none(values.get(p.name)) for p in command.params}
    try:
        lm = planner_lm if planner_lm is not None else dspy_utils.get_lm("LLM_PLANNER", "LITELLM_API_KEY_PLANNER")
        with dspy.context(lm=lm, adapter=CommandsSystemPreludeAdapter()):
            args, notes = planning_commands.typed_fallback_arguments(
                command, sub_task, typed_planner_commands_text(workflow), raw,
                planning_commands.PlanningCommandInputError(refusal or ""), workflow)
    except planning_commands.PlanningCommandCallError:
        return None                                   # still refused (or the hook raised): the refusal stands
    except Exception as exc:  # noqa: BLE001 - the draft is advisory: the extractor's refusal stands
        logger.warning("the typed fill fallback of %s failed (%s: %s); the draft shows the refusal",
                       item.means.name, type(exc).__name__, exc)
        return None
    logger.info("drafted %s with the typed fill fallback (%s)", item.means.name, "; ".join(notes))
    return args


def accepted_subtask_proposal(workflow) -> Optional[SubTaskProposal]:
    """The sub-task proposal the user accepted in the turn's review, read from the session's review store, or None.

    The hand-off hook (fix-3rsq.plan-review.2.9): the accepted record is CONFIRMED and was opened in the current turn.
    The gate also keeps it on the context as ``_turn_subtask_plan`` (the draft for a turn that was not reviewed).
    """
    from fastworkflow import approval_binding, review_engine

    store = approval_binding.store_for(workflow)
    if store is None:
        return None
    for record in reversed(store.records):
        if (record.command == "review_subtasks" and record.state == review_engine.CONFIRMED
                and record.created_turn == store.turn_seq):
            proposal = review_engine.proposal_of(record)
            return proposal if isinstance(proposal, SubTaskProposal) else None
    return None


# ----------------------------------------------------------------------
# Per-turn candidates (fix-3rsq.plan-review.2.9 note b): the Jev match is asked once per turn and text
# ----------------------------------------------------------------------


def planning_match(workflow, text: str, exclude: Optional[set[str]] = None
                    ) -> tuple[str, Any, dict[str, float]]:
    """``(template, command, distribution)`` of the planning-command match for *text*: the same call the turn's planner
    makes (``workflow_agent.planner_planning_command_for``), memoised for the turn so the fast-path check, the draft and
    the sub-task's run share one Jev call. The distribution is the match's options (empty without JEV_API_KEY).

    *exclude* is a set of command names left out of the match (included in the turn-cache key; fix-kk46.3 look_up).
    """
    from fastworkflow import workflow_agent
    from fastworkflow.workflow_agent import PLANNING_COMMAND_MATCH_THRESHOLD, load_planning_command_map

    folder = workflow.folderpath
    excluded = frozenset(exclude or ())

    def compute():
        distribution: dict[str, float] = {}
        template, command = workflow_agent.planner_planning_command_for(
            text, folder, scopes=workflow_agent.planning_scopes(workflow), distribution=distribution)
        if excluded and command is not None and command.name in excluded:
            candidates = [(name, score) for name, score in distribution.items()
                          if name not in excluded and name != "none"]
            command = None
            template = ""
            if candidates:
                name, score = max(candidates, key=lambda pair: pair[1])
                if score >= PLANNING_COMMAND_MATCH_THRESHOLD:
                    command = load_planning_command_map(folder).get(name)
                    if command is not None:
                        template = workflow_agent._template_text(command.template)
        return template, command, distribution

    template, command, distribution = turn_cache.memo(("planning_match", folder, text, excluded), compute)
    filtered = {name: score for name, score in distribution.items() if name not in excluded}
    return template, command, filtered


def redraft_for_look_up(workflow, item: SubTask, earlier: list[SubTask], *, planner_lm=None) -> SubTask:
    """Goal-preserving re-draft for ``look_up`` (fix-kk46.3): write commands re-match with writes excluded; read-only
    commands keep the means and re-draft inputs from the goal."""
    from fastworkflow.command_writes import write_commands

    writes = write_commands(workflow.folderpath)
    if item.means.kind != MEANS_FROM_SCRATCH and item.means.name in writes:
        exclude = set(writes)
        _, command, _ = planning_match(workflow, item.goal, exclude=exclude)
        if command is not None:
            item = item.model_copy(update={"means": Means(kind=MEANS_PLANNING, name=command.name)})
            return drafted_item(workflow, item, item.goal, earlier, planner_lm=planner_lm)
        new_means = build_round1_proposal(
            [item.goal], {1: _candidates_for_subtask(workflow, item.goal, exclude=exclude)}).items[0].means
        if new_means.kind == MEANS_FROM_SCRATCH:
            return item.model_copy(update={"means": new_means, "inputs": [], "drafted_means": "", "check": ""})
        item = item.model_copy(update={"means": new_means})
        return drafted_item(workflow, item, item.goal, earlier, planner_lm=planner_lm)
    return drafted_item(workflow, item, item.goal, earlier, planner_lm=planner_lm)


# ----------------------------------------------------------------------
# The free-text reader (fix-3rsq.plan-review.2.7): one model call turns the user's words into typed edits
# ----------------------------------------------------------------------


class ProposedEdit(BaseModel):
    """One edit the reader reads from the reply, numbered as the proposal is shown.

    ``change`` sets ``field`` (goal, outcome, inputs or means) to ``value``; ``drop`` removes ``index``; ``add``
    inserts a sub-task with goal ``value`` after ``after`` (0: first); ``merge`` joins ``indexes`` into one with goal
    ``value``; ``split`` cuts ``index`` into one sub-task per ``parts`` goal.
    """

    model_config = ConfigDict(extra="forbid")
    op: Literal["change", "drop", "add", "merge", "split"]
    index: Optional[int] = None
    indexes: list[int] = Field(default_factory=list)
    after: Optional[int] = None
    field: str = ""
    value: str = ""
    parts: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalize_llm_output(self) -> "ProposedEdit":
        if self.index == 0:
            self.index = None
        if (self.index is None and self.op in ("change", "drop", "split")
                and len(self.indexes) == 1 and self.indexes[0] >= 1):
            self.index = self.indexes[0]
        if self.op == "merge" and 0 in self.indexes:
            self.indexes = [each + 1 for each in self.indexes]
        if self.op == "add" and not self.value.strip() and self.parts:
            self.value = self.parts[0].strip()
        return self


class PendingItemSlot(EditRefused):
    """Edits parsed from free text that need an item index on a multi-item proposal."""

    def __init__(self, message: str, proposed: list[ProposedEdit]):
        super().__init__(message)
        self.proposed = list(proposed)


class ReadSubTaskEdits(dspy.Signature):
    """Read the user's reply on a numbered list of sub-tasks as edits to that list. Give only the edits the reply asks
    for: change a sub-task's means (a planning command written planning:NAME from the list given, or from_scratch when
    no single planning command fits), its inputs (name=value, or none to clear wrong drafted inputs), its desired
    outcome (field='outcome') or its goal (field='goal'); drop one; add one; merge several into one (when two sub-tasks
    are really one record or one job); or split one into parts. When the user says 'yes, but also ...' or 'not just X,
    I want ...' or says a drafted input/kind is wrong or unknown ('use your best judgment'), return the corresponding
    change/add/merge edit rather than an empty list. A reply that asks for no change at all gives an empty list. When
    the user says 'Step N should be NAME as the KIND, not ...' on a sub-task that has inputs name and kind, return
    change edits setting name=NAME and kind=KIND on step N rather than rewriting the goal to a different task. When the
    user asks to split one combined step into separate steps (one per person or item named) or to replace steps with an
    explicit list of steps, use op='split' with 'parts' containing the full goal sentence for each step, or op='change'/
    'add' with the full goal sentence, never short 2-word placeholders. Any text wrapped in single or double quotes in
    the user's reply (e.g. 'Henderson Ramirez and Lewis and Sons' or "Mcclain Miller and Henderson and Sons 1") is a
    single literal entity name: copy the entire quoted name verbatim into the input or goal and never split it on 'and'
    or commas or shorten it."""

    proposal: str = dspy.InputField(desc="the sub-tasks as shown to the user, numbered from 1")
    planning_commands: str = dspy.InputField(desc="the planning commands a means can name, one per line")
    reply: str = dspy.InputField(desc="the user's reply, in their own words")
    edits: list[ProposedEdit] = dspy.OutputField(desc="the edits the reply asks for, in order; empty when it asks for none")


def _resolve_proposed_item(proposal: Proposal, index: Optional[int]) -> int:
    count = len(proposal.items)
    if index is None:
        if count == 1:
            return 1
        raise EditRefused(f"Which of the {count} steps do you mean?")
    return index


_MODEL_LINE_SUFFIX = re.compile(
    r"^\s*(?:(?:\d+)\.\s+)?(?P<goal>.*?)\s*"
    r"\(\s*inputs:\s*(?P<inputs>[^;)]*?)\s*;\s*means:\s*(?P<means>[^)]*?)\s*\)"
    r"(?:\s*;\s*desired outcome:\s*(?P<outcome>.*?))?(?:\s*\[check:.*\])?\s*$",
    re.I | re.S,
)


def _strip_leading_step_number(goal: str) -> str:
    return re.sub(r"^\s*\d+\.\s+", "", (goal or "").strip()).strip()


def _clean_goal_value(text: str) -> tuple[str, list[str], Optional[str], Optional[str]]:
    """Strip a copied ``model_rendering()`` line down to the goal phrase only."""
    raw = (text or "").strip()
    match = _MODEL_LINE_SUFFIX.match(raw)
    if match:
        goal = _strip_leading_step_number(match.group("goal"))
        inputs_raw = match.group("inputs") or ""
        inputs_parts = [piece.strip() for piece in inputs_raw.split(",") if piece.strip()]
        means = (match.group("means") or "").strip() or None
        outcome = (match.group("outcome") or "").strip() or None
        return goal, inputs_parts, means, outcome
    return _strip_leading_step_number(raw), [], None, None


_FUNCTION_CALL_EDIT = re.compile(
    r"^(?:planning:)?(?P<cmd>[A-Za-z_/]\w*)\s*\((?P<args>.+)\)\s*$",
    re.DOTALL,
)
_FUNCTION_CALL_ARG = re.compile(
    r"(\w+)\s*=\s*(?:'([^']*)'|\"([^\"]*)\"|([^,\s)]+))",
)


def _parse_function_call_args(args: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for match in _FUNCTION_CALL_ARG.finditer(args):
        value = match.group(2) or match.group(3) or match.group(4) or ""
        pairs.append((match.group(1), value.strip()))
    return pairs


def _split_function_call_edit(edit: ProposedEdit) -> list[ProposedEdit]:
    field = (edit.field or "").strip().lower()
    if edit.op != "change" or field not in ("goal", "means", "inputs"):
        return [edit]
    value = (edit.value or "").strip()
    match = _FUNCTION_CALL_EDIT.match(value)
    if not match:
        return [edit]
    pairs = _parse_function_call_args(match.group("args"))
    if not pairs:
        return [edit]
    inputs_value = ", ".join(f"{key}={val}" for key, val in pairs)
    return [edit.model_copy(update={"field": "inputs", "value": inputs_value})]


def _clean_proposed_edit(edit: ProposedEdit) -> ProposedEdit:
    field = (edit.field or "").strip().lower()
    if edit.op == "change" and field == "goal" and edit.value.strip():
        return edit.model_copy(update={"value": _clean_goal_value(edit.value)[0]})
    if edit.op in ("add", "merge") and edit.value.strip():
        return edit.model_copy(update={"value": _clean_goal_value(edit.value)[0]})
    if edit.op == "split" and edit.parts:
        cleaned = [_clean_goal_value(part)[0] for part in edit.parts if (part or "").strip()]
        if cleaned:
            return edit.model_copy(update={"parts": cleaned})
    return edit


_STEP_FINE_CLAUSE = re.compile(
    r"\bsteps?\s+\d+(?:\s*(?:and|to|through|[-–—])\s*\d+)?\s+(?:look|are|is)\s+(?:fine|good|ok|okay|right)\b",
    re.I,
)
_CLAUSE_SPLIT = re.compile(r"(?:[.;!\n]|\s+[—–-]\s+)")
_READER_DROP_CUE = re.compile(r"\b(?:drop|remove|delete|skip|omit)\b", re.I)


def _reply_clauses(reply_text: str) -> list[str]:
    cleaned = _STEP_FINE_CLAUSE.sub("", reply_text or "")
    parts = [piece.strip() for piece in _CLAUSE_SPLIT.split(cleaned) if piece.strip()]
    return parts if parts else ([cleaned.strip()] if cleaned.strip() else [])


def _extract_step_numbers(segment: str, count: int) -> list[int]:
    from fastworkflow._workflows.consultation import _item_validation

    return _item_validation.extract_step_numbers(segment, count)


def _clause_edit_overlap(clause: str, edit: ProposedEdit) -> bool:
    words = {word for word in re.findall(r"[a-z0-9]+", clause.lower()) if len(word) > 2}
    blob = f"{edit.field} {edit.value}".lower()
    edit_words = {word for word in re.findall(r"[a-z0-9]+", blob) if len(word) > 2}
    return bool(words & edit_words)


def _explicit_drop_in_reply(reply_text: Optional[str], step: int, count: int) -> bool:
    if not reply_text:
        return False
    for clause in _reply_clauses(reply_text):
        if _READER_DROP_CUE.search(clause) and step in _extract_step_numbers(clause, count):
            return True
    return False


def _expand_edit_to_steps(edit: ProposedEdit, steps: list[int]) -> list[ProposedEdit]:
    if not steps:
        return [edit]
    return [edit.model_copy(update={"index": step, "indexes": []}) for step in steps]


def _infer_missing_indexes_from_reply(
    proposal: Proposal,
    proposed: list[ProposedEdit],
    reply_text: Optional[str],
) -> list[ProposedEdit]:
    if not reply_text:
        return proposed
    count = len(proposal.items)
    if count <= 1:
        return proposed
    if not any(edit.op in ("change", "drop", "split") and edit.index is None for edit in proposed):
        return proposed

    clauses = _reply_clauses(reply_text)
    drop_steps: set[int] = set()
    expanded: list[ProposedEdit] = []

    for edit in proposed:
        if edit.index is not None:
            expanded.append(edit)
            continue
        if edit.op == "drop":
            steps: list[int] = []
            for clause in clauses:
                if _READER_DROP_CUE.search(clause):
                    steps.extend(_extract_step_numbers(clause, count))
            if not steps and edit.indexes:
                steps = [idx for idx in edit.indexes if 1 <= idx <= count]
            steps = [step for step in dict.fromkeys(steps)]
            if steps:
                drop_steps.update(steps)
                expanded.extend(_expand_edit_to_steps(edit, steps))
            else:
                expanded.append(edit)
            continue
        expanded.append(edit)

    pending_changes = [edit for edit in expanded if edit.op == "change" and edit.index is None]
    resolved_changes: list[ProposedEdit] = []
    for edit in pending_changes:
        matched_steps: list[int] = []
        for clause in clauses:
            if _READER_DROP_CUE.search(clause):
                continue
            clause_steps = _extract_step_numbers(clause, count)
            if clause_steps and _clause_edit_overlap(clause, edit):
                matched_steps = clause_steps
                break
        if not matched_steps:
            for clause in clauses:
                if _READER_DROP_CUE.search(clause):
                    continue
                clause_steps = [
                    step for step in _extract_step_numbers(clause, count) if step not in drop_steps
                ]
                if clause_steps:
                    matched_steps = clause_steps
        matched_steps = [step for step in dict.fromkeys(matched_steps) if step not in drop_steps]
        if matched_steps:
            resolved_changes.extend(_expand_edit_to_steps(edit, matched_steps))
        else:
            resolved_changes.append(edit)

    result = [
        edit for edit in expanded
        if not (edit.op == "change" and edit.index is None)
    ] + resolved_changes
    for edit in expanded:
        if edit.op != "split" or edit.index is not None:
            continue
        nums = [
            step for step in dict.fromkeys(_extract_step_numbers(reply_text, count))
            if step not in drop_steps
        ]
        if len(nums) == 1:
            result.append(edit.model_copy(update={"index": nums[0], "indexes": []}))
        elif len(nums) > 1:
            result.extend(_expand_edit_to_steps(edit, nums))
        else:
            result.append(edit)
    return result


def _dedupe_proposed_edits(proposed: list[ProposedEdit]) -> list[ProposedEdit]:
    seen: set[tuple] = set()
    kept: list[ProposedEdit] = []
    for edit in proposed:
        if edit.op == "change":
            field = (edit.field or "").strip().lower()
            key = (edit.index, field, edit.value)
            if key in seen:
                continue
            seen.add(key)
        kept.append(edit)
    return kept


def _dedupe_engine_edits(
    edits: list,
    *,
    reply_text: Optional[str] = None,
    proposal: Optional[Proposal] = None,
) -> list:
    from collections import defaultdict

    count = len(proposal.items) if proposal is not None else 0
    drop_by_index: dict[int, Drop] = {}
    split_indexes: set[int] = set()
    merge_indexes: set[int] = set()
    changes: list[Change] = []
    others: list = []

    for edit in edits:
        if isinstance(edit, Drop):
            drop_by_index[edit.index] = edit
        elif isinstance(edit, Split):
            split_indexes.add(edit.index)
            others.append(edit)
        elif isinstance(edit, Merge):
            merge_indexes.update(edit.indexes)
            others.append(edit)
        elif isinstance(edit, Change):
            if _is_clearing_change(edit):
                others.append(edit)
            else:
                changes.append(edit)
        else:
            others.append(edit)

    whole_indexes = split_indexes | merge_indexes
    skip_drop: set[int] = set()
    skip_change: set[int] = set()
    for index in set(drop_by_index) & {change.index for change in changes}:
        if _explicit_drop_in_reply(reply_text, index, count):
            skip_change.add(index)
        else:
            skip_drop.add(index)
    skip_change |= whole_indexes

    surviving_changes: list[Change] = [
        change for change in changes
        if change.index not in skip_change
    ]
    surviving_drops = [
        drop for index, drop in drop_by_index.items()
        if index not in skip_drop
    ]

    by_index_field: dict[tuple[int, str], list[Change]] = defaultdict(list)
    for change in surviving_changes:
        by_index_field[(change.index, change.field)].append(change)

    merged_changes: list[Change] = []
    for (index, field), field_edits in by_index_field.items():
        if field == "inputs":
            merged_inputs: list[SubTaskInput] = []
            for field_edit in field_edits:
                merged_inputs = merge_subtask_input_edit(merged_inputs, field_edit.value)
            merged_changes.append(
                Change(
                    index=index,
                    field="inputs",
                    value=", ".join(item.render() for item in merged_inputs),
                )
            )
        else:
            merged_changes.append(field_edits[-1])

    return others + surviving_drops + merged_changes


def _normalize_reader_edits(
    proposal: Proposal,
    proposed: list[ProposedEdit],
    *,
    reply_text: Optional[str] = None,
) -> list[ProposedEdit]:
    count = len(proposal.items)
    normalized: list[ProposedEdit] = []
    for edit in proposed:
        if (
            edit.op == "merge"
            and isinstance(proposal, SubTaskProposal)
            and len(edit.indexes) == 2
            and reply_text
            and re.search(r"\b(?:swap|switch|reorder)\b", reply_text, re.I)
            and not re.search(r"\b(?:merge|combine|join|drop|remove|delete|skip|omit)\b", reply_text, re.I)
        ):
            idx_a, idx_b = edit.indexes[0], edit.indexes[1]
            if 1 <= idx_a <= count and 1 <= idx_b <= count and idx_a != idx_b:
                for src_idx, dst_idx in ((idx_a, idx_b), (idx_b, idx_a)):
                    src_item = proposal.items[src_idx - 1]
                    normalized.append(ProposedEdit(op="change", index=dst_idx, field="goal", value=src_item.goal))
                    normalized.append(
                        ProposedEdit(op="change", index=dst_idx, field="means", value=src_item.means.render())
                    )
                    normalized.append(
                        ProposedEdit(
                            op="change",
                            index=dst_idx,
                            field="inputs",
                            value=", ".join(inp.render() for inp in src_item.inputs) or "none",
                        )
                    )
                continue
        if edit.op == "merge" and (count < 2 or len(set(edit.indexes)) < 2):
            if edit.value.strip() and count == 1:
                normalized.append(
                    _clean_proposed_edit(
                        ProposedEdit(
                            op="change",
                            index=1,
                            field="goal",
                            value=_clean_goal_value(edit.value.strip())[0],
                        )
                    )
                )
            continue
        if edit.op in ("change", "drop", "split") and edit.index is None and len(edit.indexes) >= 2:
            for idx in edit.indexes:
                if 1 <= idx <= count:
                    normalized.append(
                        _clean_proposed_edit(edit.model_copy(update={"index": idx, "indexes": []}))
                    )
            continue
        if edit.op == "drop" and count == 1 and len(proposed) == 1:
            raise EditParseError("I could not read that as an edit.")
        for piece in _split_function_call_edit(_clean_proposed_edit(edit)):
            normalized.append(piece)
    if not normalized:
        raise EditParseError("I could not read that as an edit.")
    return normalized


def _token_overlap(left: str, right: str) -> int:
    left_tokens = set(re.findall(r"[a-z0-9]+", (left or "").lower()))
    right_tokens = set(re.findall(r"[a-z0-9]+", (right or "").lower()))
    return len(left_tokens & right_tokens)


def _is_truncated_match(partial: str, full: str) -> bool:
    if not partial or partial.lower() == full.lower():
        return False
    if partial.lower() in full.lower():
        return True
    return _token_overlap(partial, full) >= 2


def _best_quoted_match(value: str, quoted: list[str]) -> Optional[str]:
    for name in quoted:
        if value.lower() == name.lower():
            return name
        if _is_truncated_match(value, name):
            return name
    return None


def _repair_change_inputs_value(value: str, quoted: list[str], reply_text: str = "") -> str:
    parts = _split_input_assignments(value)
    if len(parts) <= 1:
        match = _best_quoted_match(_strip_outer_quotes(value), quoted)
        if match and "=" in value and not _said_literally(_strip_outer_quotes(value.partition("=")[2]), reply_text, quoted):
            name, _, _rest = value.partition("=")
            return f"{name.strip()}={match}"
        return value
    repaired: list[str] = []
    for part in parts:
        inp = SubTaskInput.from_text(part)
        if inp.name and inp.value:
            match = _best_quoted_match(inp.value, quoted)
            if match and not _said_literally(inp.value, reply_text, quoted):
                inp = inp.model_copy(update={"value": match})
            repaired.append(inp.render())
        else:
            repaired.append(part)
    return ", ".join(repaired)


def _repair_goal_value(value: str, quoted: list[str]) -> str:
    result = value
    for name in sorted(quoted, key=len, reverse=True):
        if name in result:
            continue
        for match in re.finditer(r"[A-Za-z][A-Za-z0-9\s&'\-.,]*", result):
            span = match.group(0).strip(" ,.")
            if span and _is_truncated_match(span, name):
                result = result.replace(match.group(0), name, 1)
                break
    return result


def _clause_containing_quote(reply_text: str, quoted_name: str) -> str:
    for clause in _reply_clauses(reply_text):
        if quoted_name in clause or quoted_name.lower() in clause.lower():
            return clause
    return reply_text


def _infer_kind_from_clause(clause: str) -> Optional[str]:
    lowered = clause.lower()
    if re.search(r"\b(?:unit|organization|organisation|department)\b", lowered):
        return "unit"
    if re.search(r"\bgroup\b", lowered):
        return "group"
    if re.search(r"\b(?:right|permission)\b", lowered):
        return "right"
    if re.search(r"\bapplication\b", lowered):
        return "application"
    if re.search(r"\bfeed\b", lowered):
        return "feed"
    if re.search(r"\b(?:person|identity)\b", lowered):
        return "person"
    if re.search(r"\b(?:login|account)\b", lowered):
        return "login"
    return None


def _quoted_already_in_edits(edits: list, quoted_name: str) -> bool:
    for edit in edits:
        if not isinstance(edit, Change) or edit.field != "inputs":
            continue
        for part in _split_input_assignments(edit.value):
            inp = SubTaskInput.from_text(part)
            if inp.value and (inp.value.lower() == quoted_name.lower()
                              or quoted_name.lower() in inp.value.lower()):
                return True
    return False


def _target_index_for_quoted(
    proposal: SubTaskProposal,
    reply_text: str,
    quoted_name: str,
    dropped_indexes: set[int],
) -> Optional[int]:
    clause = _clause_containing_quote(reply_text, quoted_name)
    steps = [
        step for step in _extract_step_numbers(clause, len(proposal.items))
        if step not in dropped_indexes
    ]
    if len(steps) == 1:
        return steps[0]
    surviving = [
        index for index in range(1, len(proposal.items) + 1) if index not in dropped_indexes
    ]
    if len(surviving) == 1:
        return surviving[0]
    for index, item in enumerate(proposal.items, 1):
        if index in dropped_indexes:
            continue
        for inp in item.inputs:
            if inp.name in _ENTITY_INPUT_KEYS and inp.value and inp.value.lower() in quoted_name.lower():
                return index
    return None


def _primary_entity_key(item: SubTask) -> Optional[str]:
    names = [inp.name for inp in item.inputs if inp.name in _ENTITY_INPUT_KEYS]
    if "name" in names:
        return "name"
    return names[0] if names else None


def _apply_quoted_names_from_reply(
    proposal: SubTaskProposal,
    edits: list,
    reply_text: str,
    quoted: list[str],
) -> list:
    repaired: list = []
    for edit in edits:
        if isinstance(edit, Change) and edit.field == "inputs":
            repaired.append(
                edit.model_copy(update={"value": _repair_change_inputs_value(edit.value, quoted, reply_text)})
            )
        elif isinstance(edit, Change) and edit.field == "goal":
            repaired.append(
                edit.model_copy(update={"value": _repair_goal_value(edit.value, quoted)})
            )
        else:
            repaired.append(edit)

    dropped_indexes = {edit.index for edit in repaired if isinstance(edit, Drop)}
    reader_edits = list(repaired)
    for quoted_name in _wanted_quoted_names(reply_text, quoted):
        if _quoted_already_in_edits(repaired, quoted_name):
            continue
        target_index = _target_index_for_quoted(proposal, reply_text, quoted_name, dropped_indexes)
        if target_index is None:
            continue
        target_item = proposal.items[target_index - 1]
        primary_key = _primary_entity_key(target_item)
        if not primary_key or _reader_touches_input(reader_edits, target_index, primary_key):
            continue
        clause = _clause_containing_quote(reply_text, quoted_name)
        kv = [f"{primary_key}={quoted_name}"]
        inferred_kind = _infer_kind_from_clause(clause)
        has_kind = any(inp.name == "kind" for inp in target_item.inputs)
        if inferred_kind and has_kind:
            kv.append(f"kind={inferred_kind}")
        repaired.append(Change(index=target_index, field="inputs", value=", ".join(kv)))
    return repaired


# ----------------------------------------------------------------------
# Deterministic guards on the reader's edits: the reader (a small model) misreads plain-English corrections in a few
# recurring ways. These act on what it returned, never on the prompt: a drop of a whole step that the reply aimed at an
# input becomes a clearing of that input; an edit for a step the reply never names, whose inputs another named step
# holds, goes to that step; a drop the reply does not ask for is not applied.
# ----------------------------------------------------------------------

#: Generic input names (they hold the entity a step is about): a bare mention of one in a reply does not select it.
_GENERIC_INPUT_NAMES = frozenset(_ENTITY_INPUT_KEYS | {"scope", "fact"})
_NAME_PREFIXES = ("include_", "with_", "is_", "has_", "show_")
_REMOVE_VERBS = r"(?:drop|remove|delete|skip|omit|cut|leave\s+out|take\s+out|get\s+rid\s+of|replace)"
_STEP_WORD = r"(?:steps?|items?|sub-?tasks?)"
_STEP_LIST = r"#?\d+(?:\s*(?:,|and|to|through|&|-|–)\s*#?\d+)*"
_REMOVE_STEP_OBJECT = re.compile(
    rf"\b{_REMOVE_VERBS}\s+(?:(?:the|these|those|this|that|both|all|extra|duplicate|stray|separate)\s+)*"
    rf"(?:{_STEP_WORD}\s+)?(?P<list>{_STEP_LIST})\b", re.I)
_REMOVE_STEP_PRONOUN = re.compile(rf"\b{_REMOVE_VERBS}\s+(?:it|them|those|these|that|this|both)\b", re.I)
_STEP_NOT_WANTED = re.compile(
    rf"\b{_STEP_WORD}\s+(?P<list>{_STEP_LIST})\s+(?:(?:should|must|can|could|needs?\s+to)\s+"
    r"(?:not\s+be\s+there|go|be\s+(?:removed|dropped|deleted))|"
    r"(?:is|are)\s+(?:a\s+|an\s+)?(?:duplicates?|stray|extra|unneeded|unnecessary|not\s+needed|not\s+wanted|not\s+there)|"
    r"(?:isn'?t|aren'?t)\s+needed)", re.I)
_KEEP_ONLY_STEP = re.compile(rf"\b(?:keep|retain)\s+(?:only|just)\s+(?:the\s+)?{_STEP_WORD}\s+#?(?P<n>\d+)\b", re.I)
_MERGE_REPLY = re.compile(
    r"\b(?:merge|combine|join|single\s+step|one\s+step|one\s+single|a\s+single|just\s+one|only\s+one|"
    r"one\s+(?:name|entity|organi[sz]ation|record)|not\s+two)\b", re.I)
_CLEAR_CUE_BODY = (
    r"drop|remove|delete|clear|omit|skip|leave\s+(?:it\s+|those\s+|them\s+)?out|take\s+out|get\s+rid\s+of|without|"
    r"no\s+longer|(?:did\s+not|didn'?t|never)\s+ask(?:ed)?\s+for|not\s+asked\s+for")
_CLEAR_CUE_STRONG = re.compile(rf"\b(?:{_CLEAR_CUE_BODY})\b", re.I)
_CLEAR_CUE_WEAK = re.compile(rf"\b(?:{_CLEAR_CUE_BODY}|not\s+(?:the|its|their)|not\s+just|not\s+only)\b", re.I)


def _norm_phrase(text: str) -> str:
    return re.sub(r"[\s_\-]+", " ", (text or "").lower()).strip()


def _list_numbers(text: str, count: int) -> list[int]:
    """The step numbers in a list text such as '2, 3 and 5' or '4 to 6' (1..count)."""
    found: list[int] = []
    for lo, hi in re.findall(r"(\d+)\s*(?:to|through|-|–)\s*(\d+)", text):
        found.extend(range(int(lo), int(hi) + 1))
    found.extend(int(each) for each in re.findall(r"\d+", text))
    return [number for number in dict.fromkeys(found) if 1 <= number <= count]


def _leading_step_numbers(clause: str, count: int) -> list[int]:
    match = re.search(rf"{_STEP_WORD}\s+(?P<list>{_STEP_LIST})", clause, re.I)
    return _list_numbers(match.group("list"), count) if match else []


def _input_mentioned(clause: str, inp: SubTaskInput) -> bool:
    """The clause names this input: its name as a phrase (include_risk: 'risk', whole_list: 'whole list'; generic
    entity names such as unit or right never count) or its value (4 or more characters, not true/false)."""
    text = _norm_phrase(clause)
    name = (inp.name or "").lower()
    if name and name not in _GENERIC_INPUT_NAMES:
        variants = {_norm_phrase(name)}
        for prefix in _NAME_PREFIXES:
            if name.startswith(prefix):
                variants.add(_norm_phrase(name[len(prefix):]))
        if any(re.search(rf"(?<![a-z0-9]){re.escape(variant)}(?![a-z0-9])", text) for variant in variants if variant):
            return True
    value = _norm_phrase(inp.value or "")
    if len(value) >= 4 and value not in ("true", "false", "none") and inp.from_subtask is None:
        if re.search(rf"(?<![a-z0-9]){re.escape(value)}", text):
            return True
    return False


def _explicit_removed_steps(reply_text: str, count: int, proposal: Optional["SubTaskProposal"] = None) -> set[int]:
    """Steps the reply explicitly asks to remove ('remove step 2', 'step 2 should not be there', 'keep only step 1').
    'drop it' counts only when the clause does not name an input of the step it follows ('I did not ask for risk, so
    drop it' is about the risk setting)."""
    removed: set[int] = set()
    previous: list[int] = []
    for clause in _reply_clauses(reply_text):
        for match in _REMOVE_STEP_OBJECT.finditer(clause):
            removed.update(_list_numbers(match.group("list"), count))
        for match in _STEP_NOT_WANTED.finditer(clause):
            removed.update(_list_numbers(match.group("list"), count))
        if _REMOVE_STEP_PRONOUN.search(clause):
            lead = _leading_step_numbers(clause, count) or previous
            if proposal is not None:
                lead = [step for step in lead
                        if not any(inp.name and _input_mentioned(clause, inp) for inp in proposal.items[step - 1].inputs)]
            removed.update(lead)
        previous = _leading_step_numbers(clause, count) or previous
    for match in _KEEP_ONLY_STEP.finditer(reply_text or ""):
        keep = int(match.group("n"))
        removed.update(step for step in range(1, count + 1) if step != keep)
    return removed


def _mentioned_steps(reply_text: str, count: int) -> set[int]:
    return set(_extract_step_numbers(_STEP_FINE_CLAUSE.sub("", reply_text or ""), count))


_OBJECT_END = re.compile(r"[,;:.!?]|\s+(?:and|but|so|then|because|since|which|instead|as)\b", re.I)
_PRONOUN_OBJECT = re.compile(r"^\s*(?:it|them|those|these|that|this|both|too|as\s+well)?\s*$", re.I)


def _removal_objects(clause: str, *, weak: bool) -> list[str]:
    """The text a removal cue in *clause* points at ('drop the whole-list setting' -> 'the whole-list setting'). A cue
    that points at a pronoun ('set tags to false, or remove it') points at what comes before it. A cue inside a quote
    is not one, and an object that names a step ('the separate Ann step') is a step, not an input."""
    masked = re.sub(r"(['\"])[^'\"]*\1", lambda m: " " * len(m.group(0)), clause)
    cue = _CLEAR_CUE_WEAK if weak else _CLEAR_CUE_STRONG
    objects: list[str] = []
    for match in cue.finditer(masked):
        rest = clause[match.end():]
        end = _OBJECT_END.search(masked[match.end():])
        obj = rest[:end.start()] if end else rest
        if _PRONOUN_OBJECT.match(obj):
            obj = clause[max(0, match.start() - 60):match.start()]
        if re.search(r"\b(?:steps?|items?)\b", obj, re.I):
            continue
        objects.append(obj)
    return objects


def _clear_requests(proposal: "SubTaskProposal", reply_text: str, *, weak: bool = False) -> list[tuple[int, str]]:
    """``(step, input name)`` pairs the reply asks to clear: a removal cue whose object names the input (its setting
    name or its value), in a clause that names the step (or the one step there is)."""
    count = len(proposal.items)
    found: list[tuple[int, str]] = []
    for clause in _reply_clauses(reply_text):
        objects = _removal_objects(clause, weak=weak)
        if not objects:
            continue
        steps = _extract_step_numbers(clause, count)

        def mentioned(inp: SubTaskInput) -> bool:
            return any(_input_mentioned(obj, inp) for obj in objects)

        for step in steps or ([1] if count == 1 else []):
            for inp in proposal.items[step - 1].inputs:
                if inp.name and mentioned(inp) and (step, inp.name) not in found:
                    found.append((step, inp.name))
        if not steps and count > 1:
            holders: dict[str, list[int]] = {}
            for index, item in enumerate(proposal.items, 1):
                for inp in item.inputs:
                    if inp.name and mentioned(inp):
                        holders.setdefault(inp.name, []).append(index)
            for name, where in holders.items():
                if len(where) == 1 and (where[0], name) not in found:
                    found.append((where[0], name))
    return found


def _clear_edit(step: int, name: str) -> ProposedEdit:
    return ProposedEdit(op="change", index=step, field="inputs", value=f"{name}=none")


def _edit_input_keys(edit: ProposedEdit) -> list[str]:
    field = (edit.field or "").strip().lower()
    if edit.op != "change":
        return []
    if field == "inputs":
        keys = [key for key in (_input_edit_part_key(part) for part in _input_edit_parts((edit.value or "").strip())) if key]
        return [key for key in keys if key not in _META_INPUT_KEYS]
    if field.startswith("inputs."):
        return [field[len("inputs."):]]
    if field and field not in ("goal", "outcome", "means", "check", "desired outcome", "desired_outcome"):
        return [field.replace(" ", "_")]
    return []


def _guard_drops_to_clears(proposal: "SubTaskProposal", proposed: list[ProposedEdit], reply_text: str) -> list[ProposedEdit]:
    """A drop of step N the reply does not ask for ('remove step N', 'step N should not be there'), where the reply
    names one of N's inputs (a setting or a value) with a removal cue, clears that input instead."""
    count = len(proposal.items)
    if any(edit.op == "drop" for edit in proposed) and not _MERGE_REPLY.search(reply_text):
        explicit = _explicit_removed_steps(reply_text, count, proposal)
        requests = _clear_requests(proposal, reply_text, weak=True)
        result: list[ProposedEdit] = []
        for edit in proposed:
            if edit.op == "drop" and edit.index is not None and 1 <= edit.index <= count and edit.index not in explicit:
                wanted = [name for step, name in requests if step == edit.index]
                if wanted:
                    result.extend(_clear_edit(edit.index, name) for name in wanted)
                    continue
            result.append(edit)
        return result
    return proposed


def _guard_unasked_drops(
    proposal: "SubTaskProposal", proposed: list[ProposedEdit], reply_text: str, refused: Optional[list[int]] = None,
) -> list[ProposedEdit]:
    """A drop the reply does not ask for goes to the one step it does ask to remove and the reader did not drop, or,
    when the reply never names that step, is not applied (its step numbers are added to *refused*)."""
    count = len(proposal.items)
    if count < 2 or not any(edit.op == "drop" for edit in proposed) or _MERGE_REPLY.search(reply_text):
        return proposed
    explicit = _explicit_removed_steps(reply_text, count, proposal)
    mentioned = _mentioned_steps(reply_text, count)
    uncovered = [step for step in sorted(explicit)
                 if not any(each.op == "drop" and each.index == step for each in proposed)]
    result: list[ProposedEdit] = []
    for edit in proposed:
        if edit.op == "drop" and edit.index is not None and 1 <= edit.index <= count and edit.index not in explicit:
            if len(uncovered) == 1 and edit.index not in mentioned:
                result.append(edit.model_copy(update={"index": uncovered.pop()}))
                continue
            if mentioned and edit.index not in mentioned:
                if refused is not None:
                    refused.append(edit.index)
                continue
        result.append(edit)
    return result


def _guard_retarget_changes(proposal: "SubTaskProposal", proposed: list[ProposedEdit], reply_text: str) -> list[ProposedEdit]:
    """An input change for a step the reply never names, whose inputs the step lacks, goes to the one named step that
    holds them."""
    count = len(proposal.items)
    mentioned = _mentioned_steps(reply_text, count)
    if count < 2 or not mentioned:
        return proposed
    result: list[ProposedEdit] = []
    for edit in proposed:
        if edit.op == "change" and edit.index is not None and 1 <= edit.index <= count and edit.index not in mentioned:
            keys = _edit_input_keys(edit)
            if keys and not set(keys) & {inp.name for inp in proposal.items[edit.index - 1].inputs}:
                holders = [step for step in sorted(mentioned)
                           if set(keys) <= {inp.name for inp in proposal.items[step - 1].inputs}]
                if len(holders) == 1:
                    edit = edit.model_copy(update={"index": holders[0]})
        result.append(edit)
    return result


def _guard_unreported_clears(proposal: "SubTaskProposal", proposed: list[ProposedEdit], reply_text: str) -> list[ProposedEdit]:
    """An input the reply asks to clear and no edit of the reader touches is cleared."""
    touched = {(each.index, key) for each in proposed for key in _edit_input_keys(each)}
    gone = {each.index for each in proposed if each.op in ("drop", "split")}
    gone.update(step for each in proposed if each.op == "merge" for step in each.indexes)
    result = list(proposed)
    for step, name in _clear_requests(proposal, reply_text):
        if (step, name) not in touched and step not in gone:
            result.append(_clear_edit(step, name))
            touched.add((step, name))
    return result


def _guard_reader_edits(
    proposal: Proposal, proposed: list[ProposedEdit], reply_text: Optional[str],
) -> list[ProposedEdit]:
    """The deterministic guards on the reader's edits, in order. Raises ``EditRefused`` (a question for the user) when
    every edit left was a drop the reply does not ask for."""
    if not reply_text or not isinstance(proposal, SubTaskProposal) or not proposed:
        return proposed
    if len(proposal.items) == 1:
        proposed = [edit.model_copy(update={"index": 1}) if edit.index is None and edit.op in ("change", "drop") else edit
                    for edit in proposed]
    refused: list[int] = []
    proposed = _guard_drops_to_clears(proposal, proposed, reply_text)
    proposed = _guard_retarget_changes(proposal, proposed, reply_text)
    proposed = _guard_unasked_drops(proposal, proposed, reply_text, refused)
    proposed = _guard_unreported_clears(proposal, proposed, reply_text)
    if not proposed and refused:
        raise EditRefused(
            f"I read that as removing step {refused[0]}, but you did not ask for that. "
            "Which step or setting do you mean?")
    return proposed


def _reader_touches_input(edits: list, index: int, key: str) -> bool:
    """Whether the reader's own edits already set or clear input *key* of step *index*."""
    for edit in edits:
        if isinstance(edit, Change) and edit.index == index and edit.field == "inputs":
            for part in _input_edit_parts((edit.value or "").strip()):
                if _input_edit_part_key(part) == key.lower():
                    return True
    return False


def _negated_before(text: str, start: int) -> bool:
    return bool(re.search(
        r"(?:\bnot|\bwithout|\binstead\s+of|\brather\s+than|\bno\s+longer|\bnor)\b[^'\"]{0,18}$", text[:start], re.I))


def _wanted_quoted_names(reply_text: str, quoted: list[str]) -> list[str]:
    """The quoted names the reply asks for: one it rules out ('not 'X'', 'without 'X'') is not wanted."""
    wanted: list[str] = []
    for name in quoted:
        positions = [m.start() for m in re.finditer(re.escape(name), reply_text)]
        if positions and all(_negated_before(reply_text, pos) for pos in positions):
            continue
        wanted.append(name)
    return wanted


def _said_literally(value: str, reply_text: str, quoted: list[str]) -> bool:
    """The user wrote *value* as such (outside quotes, and not as 'not just VALUE'): a quoted name is not its completion."""
    if not value or not reply_text:
        return False
    text = reply_text
    for name in quoted:
        text = text.replace(name, " " * len(name))
    for match in re.finditer(rf"(?<![A-Za-z0-9_]){re.escape(value)}(?![A-Za-z0-9_])", text, re.I):
        if not re.search(r"\b(?:not|without|instead\s+of)\s+(?:just\s+|only\s+)?$", text[:match.start()], re.I):
            return True
    return False


def _proposed_edits_to_engine(
    proposal: Proposal,
    proposed: list[ProposedEdit],
    *,
    reply_text: Optional[str] = None,
) -> list:
    proposed = _normalize_reader_edits(proposal, proposed, reply_text=reply_text)
    proposed = _infer_missing_indexes_from_reply(proposal, proposed, reply_text)
    proposed = _dedupe_proposed_edits(proposed)
    proposed = _guard_reader_edits(proposal, proposed, reply_text)
    count = len(proposal.items)
    if count > 1 and any(
        edit.op in ("change", "drop", "split") and edit.index is None for edit in proposed
    ):
        raise PendingItemSlot(f"Which of the {count} steps do you mean?", proposed)
    resolved: list = []
    for edit in proposed:
        if edit.op == "change":
            if not edit.field:
                raise EditParseError("An edit names no field to change.")
            resolved.append(_engine_edit(edit, _resolve_proposed_item(proposal, edit.index), proposal))
        elif edit.op == "drop":
            resolved.append(_engine_edit(edit, _resolve_proposed_item(proposal, edit.index), proposal))
        elif edit.op == "split":
            resolved.append(_engine_edit(edit, _resolve_proposed_item(proposal, edit.index), proposal))
        else:
            resolved.append(_engine_edit(edit, edit.index, proposal))
    resolved = _dedupe_engine_edits(resolved, reply_text=reply_text, proposal=proposal)
    if reply_text and isinstance(proposal, SubTaskProposal):
        quoted = extract_quoted_names(reply_text)
        if quoted:
            resolved = _apply_quoted_names_from_reply(proposal, resolved, reply_text, quoted)
            resolved = _dedupe_engine_edits(resolved, reply_text=reply_text, proposal=proposal)
    return resolved


def _engine_edit(edit: ProposedEdit, index: Optional[int] = None, proposal: Optional[Proposal] = None):
    """The review engine's typed edit for one read edit (the same edits the deterministic reader makes)."""
    number = edit.index if index is None else index
    if edit.op == "change":
        if not edit.field:
            raise EditParseError("An edit names no field to change.")
        field = edit.field.strip()
        field_key = field.lower().replace("_", " ")
        if field_key in ("desired outcome", "desired_outcome"):
            field = "outcome"
        elif field_key in ("title", "step title", "description", "summary", "prompt"):
            field = "goal"
        value = edit.value
        if isinstance(proposal, SubTaskProposal) and field not in ("goal", "outcome", "inputs", "means", "check"):
            if not value.strip().startswith(f"{field}="):
                value = f"{field}={edit.value}"
            field = "inputs"
        return Change(index=number, field=field, value=value)
    if edit.op == "drop":
        return Drop(index=number)
    if edit.op == "add":
        if not edit.value.strip():
            raise EditParseError("An added sub-task needs its goal.")
        after = 0 if edit.after is None else edit.after
        return Add(after=after, values={"goal": edit.value})
    if edit.op == "merge":
        return Merge(indexes=list(edit.indexes), values={"goal": edit.value} if edit.value.strip() else {})
    parts = [part for part in edit.parts if part.strip()]
    if not parts:
        raise EditParseError("A split needs at least one part.")
    return Split(index=number, parts=[{"goal": part} for part in parts])


def model_edit_reader(planner_lm=None, planning_names: tuple[str, ...] = ()):
    """The reader of free-text replies to a sub-task review, as the review engine calls it: ``(text, proposal) -> edits``.

    One model call (``ReadSubTaskEdits``, typed) on the planner's role. ``planner_lm`` is the planner's LM, or a callable
    giving it at reply time (the turn's stashed planner LM, read when the reply comes); None or a callable giving None
    takes ``LLM_PLANNER``. Any reply it cannot read as edits raises ``EditParseError``, which asks the user again and
    counts toward the review's reply cap. The edits are validated by the engine before they are applied.
    """

    def read(text: str, proposal: Proposal) -> list:
        from fastworkflow.utils import dspy_utils

        # A callable that is not an LM (dspy.BaseLM is callable too) is asked for the LM now.
        lm = planner_lm() if callable(planner_lm) and not isinstance(planner_lm, dspy.BaseLM) else planner_lm
        if lm is None:
            lm = dspy_utils.get_lm("LLM_PLANNER", "LITELLM_API_KEY_PLANNER")
        shown = proposal.model_rendering() if isinstance(proposal, SubTaskProposal) else proposal.rendering()
        try:
            with dspy.context(lm=lm):
                prediction = dspy.Predict(ReadSubTaskEdits)(
                    proposal=shown,
                    planning_commands="\n".join(f"planning:{name}" for name in planning_names) or "none",
                    reply=text)
            proposed = list(prediction.edits or [])
        except Exception as exc:  # noqa: BLE001 - an unreadable reply asks again; it never fails the turn
            logger.info("the reply could not be read as edits (%s)", type(exc).__name__)
            raise EditParseError("I could not read that as an edit.") from exc
        if not proposed:
            raise EditParseError("I could not read that as an edit.")
        return _proposed_edits_to_engine(proposal, proposed, reply_text=text)

    return read


# ----------------------------------------------------------------------
# The hand-off (fix-3rsq.plan-review.2.9): a sub-task's inputs and the text its run gets
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedInputs:
    """A sub-task's inputs as values: the named ones, the unnamed ones, and the 'from sub-task N' ones that are not yet
    given (sub-task N has not run)."""

    named: dict[str, str] = field(default_factory=dict)
    unnamed: tuple[str, ...] = ()
    missing: tuple[int, ...] = ()


def resolve_inputs(item: SubTask, answers: Mapping[int, str]) -> ResolvedInputs:
    """The values of *item*'s inputs. ``answers`` maps a sub-task's 1-based number to the output its run gave (its
    Output where it has one, else its answer text); a ``from sub-task N`` input takes N's, and is missing until N ran."""
    named: dict[str, str] = {}
    unnamed: list[str] = []
    missing: list[int] = []
    for each in item.inputs:
        # A drafted input is resolved as a given one (fix-48np.1): the review showed it, so the run uses it.
        if each.from_subtask is not None:
            if each.from_subtask not in answers:
                missing.append(each.from_subtask)
                continue
            value = answers[each.from_subtask]
        else:
            value = each.value or ""
        if each.name:
            named[each.name] = value
        else:
            unnamed.append(value)
    return ResolvedInputs(named=named, unnamed=tuple(unnamed), missing=tuple(missing))


def run_text(item: SubTask, resolved: ResolvedInputs) -> str:
    """The text a sub-task's run gets: its goal, its inputs and its desired outcome."""
    given = [f"{name}={value}" for name, value in resolved.named.items()] + list(resolved.unnamed)
    parts = []
    if given:
        parts.append("inputs: " + ", ".join(given))
    if item.outcome:
        parts.append(f"desired outcome: {item.outcome}")
    return f"{item.goal} ({'; '.join(parts)})" if parts else item.goal


# ----------------------------------------------------------------------
# The run of a reviewed sub-task (fix-48np.1): the inputs the review showed are the values that run
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ReviewedValues:
    """A bound sub-task's values at run time: the ones the review showed (``shown``), the ones drafted at run time
    because the review never drafted the sub-task for its means (``drafted_at_run``), and the required fields still
    without a value (``unbound``)."""

    shown: dict[str, str] = field(default_factory=dict)
    drafted_at_run: dict[str, str] = field(default_factory=dict)
    unbound: tuple[str, ...] = ()

    @property
    def values(self) -> dict[str, str]:
        return {**self.drafted_at_run, **self.shown}


def reviewed_values(workflow, item: SubTask, resolved: ResolvedInputs, required: list[str], *,
                    earlier: list[SubTask], answers: Mapping[int, str], planner_lm=None) -> ReviewedValues:
    """The values a bound sub-task runs with, when drafting is on (fix-48np.1).

    The inputs the review bound (*resolved*: drafted, edited by the user, or ``from sub-task N``) are presets: never
    extracted again. Only a sub-task the draft did not draft for its current means (``SubTask.drafted_means``: the
    draft failed, or the user added it) that leaves a *required* field unbound has its unbound fields drafted at run
    time, from the goal, by the draft's own function (``draft_inputs``: the shared extractor, its retry and the typed
    fill fallback), so the draft and the run follow the same rules. A sub-task the draft did draft is not drafted again.
    """
    shown = dict(resolved.named)
    shown.update(_produced_fields(workflow, item, answers))
    found: dict[str, str] = {}
    if item.drafted_means != item.means.name and any(name not in shown for name in required):
        drafted = draft_inputs(workflow, item, item.goal, earlier, planner_lm=planner_lm)
        unbound = [each for each in (drafted.inputs if drafted is not None else ()) if each.name not in shown]
        found = dict(resolve_inputs(SubTask(goal=item.goal, inputs=unbound), answers).named)
    values = {**found, **shown}
    return ReviewedValues(shown=shown, drafted_at_run=found,
                          unbound=tuple(name for name in required if name not in values))


def _produced_fields(workflow, item: SubTask, answers: Mapping[int, str]) -> dict[str, str]:
    """``{field: value}`` for the ``from sub-task N`` inputs of *item* whose field declares the producer field it takes
    (``from_command_output="find_book.book_uids"``) and whose sub-task N's output is JSON holding it: that field's value, the one
    element of a one-element list. Other ``from sub-task N`` inputs keep N's whole output (``resolve_inputs``)."""
    from fastworkflow import ModuleType, RoutingRegistry

    try:
        params_class = RoutingRegistry.get_definition(workflow.folderpath).get_command_class(
            item.means.name, ModuleType.COMMAND_PARAMETERS_CLASS)
    except Exception:  # noqa: BLE001 - no declared producers: the whole output stands
        return {}
    fields = getattr(params_class, "model_fields", {}) or {}
    out: dict[str, str] = {}
    for each in item.inputs:
        if each.from_subtask is None or not each.name or each.name not in fields \
                or each.from_subtask not in answers:
            continue
        extra = fields[each.name].json_schema_extra
        refs = (extra.get("available_from") or ()) if isinstance(extra, dict) else ()
        try:
            output = json.loads(answers[each.from_subtask])
        except (TypeError, ValueError):
            continue
        if not isinstance(output, dict):
            continue
        for ref in refs:
            key = str(ref).rpartition(".")[2]
            if key in output:
                value = output[key]
                if isinstance(value, list) and len(value) == 1:
                    value = value[0]
                out[each.name] = _input_text(value)
                break
    return out


def reviewed_arguments(workflow, command, item: SubTask, resolved: ResolvedInputs, request: str, *,
                       earlier: list[SubTask], answers: Mapping[int, str], earlier_outputs: tuple[str, ...] = (),
                       planner_lm=None):
    """``(ReviewedValues, run_listing.ArgumentCheck)`` for a sub-task bound to planning command *command* (fix-48np.1).

    The values (``reviewed_values``) go through the checks a typed planning call runs (``checked_arguments``:
    ``check_arguments``, the ``Input``'s model validators) against *request* with the turn's
    earlier outputs after it. A refusal is returned: the caller hands the sub-task to the agent with it (F2) and records
    the deviation.
    """
    from fastworkflow import planning_commands
    from fastworkflow.run_listing import ArgumentCheck

    reviewed = reviewed_values(workflow, item, resolved, [p.name for p in command.params if p.required],
                               earlier=earlier, answers=answers, planner_lm=planner_lm)
    notes = ([f"drafted at run time (the review bound no value): {', '.join(sorted(reviewed.drafted_at_run))}"]
             if reviewed.drafted_at_run else [])
    try:
        args, more = planning_commands.checked_arguments(
            command, reviewed.values, "\n\n".join([request, *earlier_outputs]), workflow)
    except planning_commands.PlanningCommandCallError as refused:
        return reviewed, ArgumentCheck(refusal=str(refused), notes=tuple(notes))
    return reviewed, ArgumentCheck(args=args, notes=tuple(notes + more))


def _same_value(shown: str, ran: Any) -> bool:
    """Whether a value that ran is the value the review showed: the same text, ignoring case, surrounding quotes and
    space (the checks may take the request's own spelling of it)."""
    def norm(text: str) -> str:
        return " ".join(str(text).strip().strip("\"'`").split()).casefold()

    return norm(shown) == norm(_input_text(ran) if not isinstance(ran, str) else ran)


def binding_record(number: int, item: SubTask, reviewed: Optional[ReviewedValues], ran: Optional[Mapping[str, Any]],
                   deviation: Optional[str] = None) -> dict:
    """One sub-task's shown == run record for the turn span's ``subtask_bindings`` (fix-48np.1).

    *ran* is the arguments the bound command ran with (None when it did not run with them). A deviation is recorded
    when the review's values were refused or left a required field unbound (the agent then handled the sub-task, F2),
    a value was drafted at run time (it was not shown), or a value that ran is not the one shown.
    """
    shown = dict(reviewed.shown) if reviewed is not None else {}
    reasons = [deviation] if deviation else []
    if reviewed is not None and reviewed.drafted_at_run:
        reasons.append(f"drafted at run time: {', '.join(sorted(reviewed.drafted_at_run))}")
    if ran is None and not reasons:
        reasons.append("the bound command did not run with the reviewed inputs")
    if ran is not None:
        differs = sorted(name for name, value in shown.items() if name not in ran or not _same_value(value, ran[name]))
        if differs:
            reasons.append(f"ran a value other than the one shown: {', '.join(differs)}")
    return {"subtask": number, "means": item.means.render(), "shown": shown,
            "ran": {name: _input_text(value) for name, value in ran.items()} if ran is not None else None,
            "match": not reasons, "deviation": "; ".join(reasons) or None}
