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
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from fastworkflow import turn_cache
from fastworkflow.review_engine import (
    KIND_PLAN, KIND_SUB_TASKS, PROPOSAL_SPECS, Add, Change, Drop, EditParseError, Merge, Proposal, Split,
    register_proposal)

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
            return cls(name=match.group("name"), value=match.group("value").strip())
        return cls(value=text)


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


@register_proposal
class SubTaskProposal(Proposal):
    kind = KIND_SUB_TASKS
    spec = "review_subtasks"
    title = "Sub-tasks"
    item_template = "{index}. {goal} (inputs: {inputs}; means: {means}); desired outcome: {outcome}{check}"
    min_items = 1
    items: list[SubTask] = Field(default_factory=list)

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


def draft_round1_proposal(workflow, subtasks: list[str]) -> SubTaskProposal:
    """The draft round-1 proposal for a split result, from real candidates (fix-3rsq.plan-review.2.3).

    Per sub-task, the candidates are: the heads' planning and execution top-k (``run_listing.head_candidates``), the
    names the gather reads (``run_listing.gather``: the names the sub-task writes, the heads' labels and, when
    ``JEV_API_KEY`` is set, Jev's planning-command distribution), and Jev's confident or unsure planning commands. Jev is
    the existing matcher (``workflow_agent.match_planning_command``): it runs no model without the key and then leaves the
    distribution empty. ``build_round1_proposal`` picks each sub-task's means (a planning command in the top-k first).
    Each sub-task with a command for its means gets its inputs drafted from its text (``draft_inputs``, fix-48np:
    the shared parameter extractor; shown, not run). A planning pick's refusal is shown on its line: the drafted
    values' refusal, or, with no draft, ``check_planned_arguments`` of no arguments.
    """
    from fastworkflow import planning_commands, run_listing
    from fastworkflow.workflow_agent import core_command_names

    folder = workflow.folderpath
    planning = set(planning_commands.resolve_field_sources(folder).planning_commands)
    # The framework's own commands (IntentDetection, ErrorCorrection: go_up, reset_context, ...) are not means for a job.
    core = core_command_names()
    candidates: dict[int, list[Candidate]] = {}
    for number, sub_task in enumerate(subtasks, 1):
        heads = run_listing.head_candidates(workflow, sub_task)
        _, _, probabilities = planning_match(workflow, sub_task)
        scores = {**heads.execution_scores, **heads.planning_scores}
        for name, probability in probabilities.items():
            scores[name] = max(scores.get(name, 0.0), probability)
        named = run_listing.gather(workflow, sub_task)        # the commands the sub-task writes by name
        names = (run_listing.gather(workflow, sub_task, probabilities or None) | heads.labels
                 | heads.planning_scores.keys() | heads.execution_scores.keys())
        # A candidate is offered when the sub-task names it, or its score reaches the unsure floor (run_listing's, the
        # band the listing already offers): a head's top-k entry below the floor is noise, not a means.
        candidates[number] = [Candidate(name, is_planning=name in planning, score=scores.get(name, 0.0))
                              for name in sorted(names)
                              if name not in core and (name in named
                                                       or scores.get(name, 0.0) >= run_listing.UNSURE_CANDIDATE_FLOOR)]
    proposal = build_round1_proposal(subtasks, candidates)
    items: list[SubTask] = []
    for item, sub_task in zip(proposal.items, subtasks):
        check = ""
        inputs: list[SubTaskInput] = []
        drafted = draft_inputs(workflow, item, sub_task, items) if item.means.kind != MEANS_FROM_SCRATCH else None
        if drafted is not None:
            inputs = list(drafted.inputs)
        if item.means.kind == MEANS_PLANNING:
            command = planning_commands.planning_command(folder, item.means.name)
            if drafted is not None:
                check = drafted.refusal or ""
            elif command is not None:
                refusal = run_listing.check_planned_arguments(command, {}, sub_task).refusal
                check = refusal or ""
        items.append(item.model_copy(update={"check": check, "inputs": inputs}))
    return SubTaskProposal(items=items)


#: ``FW_DRAFT_ARGUMENTS=on`` (or 1, true) fills a draft's inputs from the shared extractor. Off by default: the drafted
#: values are shown but the typed planning call still produces the values that run, and the two disagree on about 1
#: argument in 20 (fix-48np gate 1b), so the review could show a value other than the one that runs. Off leaves the
#: inputs empty, as before fix-48np.
DRAFT_ARGUMENTS_ENV_VAR = "FW_DRAFT_ARGUMENTS"


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
    """``{field: N}``: the fields of *means*'s ``Input`` whose declared producer (``from_``, resolved as
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


def draft_inputs(workflow, item: SubTask, sub_task: str, earlier: list[SubTask]) -> Optional[DraftedInputs]:
    """The inputs of *item*'s means drafted from *sub_task* (fix-48np, stage 1), or None when no draft was made.

    A field whose declared producer (``from_``) an earlier sub-task runs is ``from sub-task N``, not extracted. The
    other values are the shared extractor's (``parameter_extraction.extract_arguments``: the extractor execution
    commands use, with its checks, and for a planning command ``check_arguments``, the checks the typed planning call
    runs), each one the extractor gave and the checks kept that is not the field's default. Values that are refused,
    or a required value that is missing, give the refusal and no values. None when the switch is off
    (``FW_DRAFT_ARGUMENTS``, off by default) or the extraction fails: the draft is advisory and never fails the turn.
    """
    import fastworkflow
    from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import extract_arguments

    if str(fastworkflow.get_env_var(DRAFT_ARGUMENTS_ENV_VAR, default="off")).strip().lower() not in ("on", "1", "true"):
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
    if not result.valid and not covered:
        return DraftedInputs(inputs=tuple(inputs), refusal=result.refusal)
    fields = type(result.params).model_fields
    for name, info in fields.items():
        value = getattr(result.params, name, None)
        if name in from_earlier or not _given(info, value):
            continue
        inputs.append(SubTaskInput(name=name, value=_input_text(value), drafted=True))
    return DraftedInputs(inputs=tuple(inputs))


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


def planning_match(workflow, text: str) -> tuple[str, Any, dict[str, float]]:
    """``(template, command, distribution)`` of the planning-command match for *text*: the same call the turn's planner
    makes (``workflow_agent.planner_planning_command_for``), memoised for the turn so the fast-path check, the draft and
    the sub-task's run share one Jev call. The distribution is the match's options (empty without JEV_API_KEY)."""
    from fastworkflow import workflow_agent

    folder = workflow.folderpath

    def compute():
        distribution: dict[str, float] = {}
        template, command = workflow_agent.planner_planning_command_for(
            text, folder, scopes=workflow_agent.planning_scopes(workflow), distribution=distribution)
        return template, command, distribution

    template, command, distribution = turn_cache.memo(("planning_match", folder, text), compute)
    return template, command, dict(distribution)


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
    index: int = 0
    indexes: list[int] = Field(default_factory=list)
    after: int = 0
    field: str = ""
    value: str = ""
    parts: list[str] = Field(default_factory=list)


class ReadSubTaskEdits(dspy.Signature):
    """Read the user's reply on a numbered list of sub-tasks as edits to that list. Give only the edits the reply asks
    for: change a sub-task's means (a planning command, written planning:NAME from the list given), its inputs (name=value
    or a value), its desired outcome or its goal; drop one; add one; merge several into one; or split one into parts.
    A reply that asks for none of these gives an empty list."""

    proposal: str = dspy.InputField(desc="the sub-tasks as shown to the user, numbered from 1")
    planning_commands: str = dspy.InputField(desc="the planning commands a means can name, one per line")
    reply: str = dspy.InputField(desc="the user's reply, in their own words")
    edits: list[ProposedEdit] = dspy.OutputField(desc="the edits the reply asks for, in order; empty when it asks for none")


def _engine_edit(edit: ProposedEdit):
    """The review engine's typed edit for one read edit (the same edits the deterministic reader makes)."""
    if edit.op == "change":
        if not edit.field:
            raise EditParseError("An edit names no field to change.")
        return Change(index=edit.index, field=edit.field, value=edit.value)
    if edit.op == "drop":
        return Drop(index=edit.index)
    if edit.op == "add":
        if not edit.value.strip():
            raise EditParseError("An added sub-task needs its goal.")
        return Add(after=edit.after, values={"goal": edit.value})
    if edit.op == "merge":
        return Merge(indexes=list(edit.indexes), values={"goal": edit.value} if edit.value.strip() else {})
    parts = [part for part in edit.parts if part.strip()]
    if not parts:
        raise EditParseError("A split needs at least one part.")
    return Split(index=edit.index, parts=[{"goal": part} for part in parts])


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
        try:
            with dspy.context(lm=lm):
                prediction = dspy.Predict(ReadSubTaskEdits)(
                    proposal=proposal.rendering(),
                    planning_commands="\n".join(f"planning:{name}" for name in planning_names) or "none",
                    reply=text)
            proposed = list(prediction.edits or [])
        except Exception as exc:  # noqa: BLE001 - an unreadable reply asks again; it never fails the turn
            logger.info("the reply could not be read as edits (%s)", type(exc).__name__)
            raise EditParseError("I could not read that as an edit.") from exc
        if not proposed:
            raise EditParseError("I could not read that as an edit.")
        return [_engine_edit(edit) for edit in proposed]

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
        if each.drafted:
            # fix-48np stage 1: a drafted input is shown in the review, not yet run
            continue
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
