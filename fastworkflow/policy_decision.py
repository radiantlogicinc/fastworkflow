"""A deterministic policy decision point for the execution agent.

An agent's mistakes in a workflow are not random. Compare a run that succeeded
with one that failed on the same task (another model's run, or another trial of
the same model), and the differences fall into a handful of shapes that recur
across tasks: a write issued before every candidate record was examined, an id
in an argument that appears in no observation, a final answer claiming work no
command performed. Distillation names those shapes -- it writes them into
``Insights/`` as English, which is then pasted into the agent's prompt and
hoped over.

This module is the other option: state a shape as a predicate over the
conversation so far, and act on it when it matches. Most of those shapes need
nothing the ReAct loop does not already hold -- every command run and every
observation returned -- so "has every record the listing returned been read
before this write" is a set difference, not a judgement.

Four properties hold on every evaluation, and they are the reason to write this
as a table rather than as scattered ``if`` statements:

* **Versioned.** Every decision records the table version that produced it, so
  a result can be attributed to a rule set rather than to "the code at the time".
* **Deterministic, with no model call.** The same input yields the same
  decision. Where a fact only a model can read is needed (what the user asked
  for, whether a question repeats an earlier one), the model's reading is an
  INPUT to the rows (see ``fastworkflow.policy_intent``); the rows still decide.
* **Attributable.** A decision names the row that fired and the reason, so a
  behaviour change can be traced to one rule and that rule can be removed.
* **Ordered.** First match wins, so precedence is part of the table's meaning
  and is recorded with its version.

**Rewrite, do not reject.** A row returns replacement text, delivered to the
agent as an ordinary observation. A bare refusal costs a loop iteration and the
agent usually re-proposes the same call; a replacement tells it what to do
instead and it keeps going.

**Two modes.** ``OFF`` (a workflow without a table) is byte-for-byte the
unmodified behaviour. ``ENFORCE`` (a workflow with one) acts. What a table *would* have done on recorded runs is
answered offline by ``fastworkflow.policy_replay``.

The rules themselves are workflow content: a workflow ships a table module
(``_policy/table.py``) exposing ``TABLE`` (a ``DecisionTable``) and optionally
``FACTS`` (``ContractFacts``). Nothing in this module knows any workflow.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Sequence

TABLE_CONTRACT_VERSION = "1"


class PolicyError(ValueError):
    """A malformed table, or one that violates an invariant it declares."""


class PolicyMode(str, Enum):
    """Rollout state. ``OFF`` is the default and changes nothing."""

    OFF = "off"
    ENFORCE = "enforce"


class DecisionPhase(str, Enum):
    """Where in a turn a row may fire.

    The phases differ by what is knowable. At ``BEFORE_TASK`` nothing has run.
    At ``BEFORE_SIDE_EFFECT`` a command has been chosen and its parameters
    extracted, but it has not run, so it can still be stopped. At
    ``AFTER_OBSERVATION`` results are in hand and the agent has chosen a
    non-command step (a question to the user), which is what makes "you already
    have what you are about to ask for" decidable. At ``BEFORE_FINISH`` the
    answer exists, which is the only point where its claims can be checked
    against what actually ran.
    """

    BEFORE_TASK = "before-task"
    BEFORE_SIDE_EFFECT = "before-side-effect"
    AFTER_OBSERVATION = "after-observation"
    BEFORE_FINISH = "before-finish"


class PolicyOutcome(str, Enum):
    """What a row decided.

    ``REWRITE`` is the workhorse: the agent is handed replacement text and
    continues. ``ALLOW`` is an explicit exemption -- a row that matches in order
    to stop a later row from firing. ``ASK`` and ``ABSTAIN`` exist so a table can
    express "this needs the user" or "do not act on this" with a text to offer.
    """

    ALLOW = "allow"
    REWRITE = "rewrite"
    ASK = "ask"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class ContractFacts:
    """What the workflow declares about itself.

    Facts, not inferences: a command either changes state or it does not, and
    the workflow says which. Nothing here is a model output, so a row resting
    on these rests on something reproducible.

    ``id_patterns`` are the shapes of the identifiers this workflow's arguments
    carry. A row checking that an argument was actually read somewhere needs to
    know which arguments are identifiers in the first place. Left empty, the
    loader fills it from the workflow's ``_commands/_split.py``
    ``identifier_patterns`` when it declares some.

    ``enum_arguments`` maps ``command.argument`` to its accepted values, for the
    arguments whose domain is closed.

    ``failure_pattern`` is for workflows whose commands report errors as text
    while still returning success (``"Error: not found"``). A command's run is
    recorded as failed when its response says so (``CommandResponse.success``
    is False) or, when this is set, when the start of its response text
    matches it.
    """

    write_commands: frozenset[str] = frozenset()
    id_patterns: tuple[str, ...] = ()
    enum_arguments: Mapping[str, frozenset[str]] = field(default_factory=dict)
    #: The operations a user of this workflow can ask for, name -> meaning.
    #: The intent extractor may only answer in these terms, which keeps what
    #: it produces checkable against the commands that would satisfy it. Empty
    #: means the workflow declares no intent vocabulary and no extraction runs.
    intent_operations: Mapping[str, str] = field(default_factory=dict)
    #: Optional: given the phase input, describe what the agent could still
    #: look up for itself (e.g. "records R1, R2 were listed but not read").
    #: Passed to the question judge, which cannot know the workflow's data
    #: model. Returning "" means nothing is left to look up.
    lookup_hints: Optional[Callable[[Any], str]] = None
    #: Whether the question judge runs at all for this workflow.
    judge_questions: bool = False
    failure_pattern: Optional[str] = None

    def is_write(self, command_name: str) -> bool:
        return command_name in self.write_commands

    def looks_like_id(self, value: str) -> bool:
        return any(re.fullmatch(p, value) for p in self.id_patterns)

    def reports_failure(self, text: str) -> bool:
        """Whether a response's text says it failed (``failure_pattern``)."""
        if not self.failure_pattern:
            return False
        return bool(re.search(self.failure_pattern, (text or "").lstrip()[:160]))


@dataclass(frozen=True)
class CommandRecord:
    """One event in the conversation so far: a command, or something the user said.

    User messages are recorded in the same chronological list as commands, so a
    row can ask ordering questions -- "was the last thing the user said before
    this write a yes?" -- without a second timeline to reconcile. They use the
    name ``USER_MESSAGE`` for a message that opened a turn, and ``ask_user``
    (whose ``response`` is the user's answer) for a reply to the agent's
    question.
    """

    name: str
    args: Mapping[str, Any] = field(default_factory=dict)
    response: str = ""
    success: bool = True


#: Record name for a user message that opened a turn (not a command).
USER_MESSAGE = "@user"
#: Record name for a question the agent asked; its response is the answer.
ASK_USER = "ask_user"
#: Record names that carry something the user said, in ``response``.
USER_UTTERANCES = frozenset({USER_MESSAGE, ASK_USER})


@dataclass(frozen=True)
class RequestedOperation:
    """One thing the user has asked for, in the workflow's own vocabulary.

    Produced by the intent extractor, one call per user message, and
    accumulated over the conversation. The rules that read it are still
    deterministic: given the same intent and trajectory they decide the same
    way. The model's judgement is confined to reading the user's words.

    ``conditional`` marks a fallback the user named ("do X, and if that is not
    possible, do Y"): wanted only if the primary request cannot be met, so
    never required on its own.
    """

    op: str
    scope: str = "unspecified"  # "all" | "specific" | "unspecified"
    conditional: bool = False
    detail: str = ""


@dataclass(frozen=True)
class QuestionJudgement:
    """A model's reading of a question the agent is about to put to the user.

    Produced once per ``ask_user`` the agent attempts, before it is sent, and
    handed to the rules as an input -- the same division of labour as
    ``RequestedOperation``: the model reads, the rule decides.

    Word-overlap matching cannot tell a pointless repeat from a legitimate
    follow-up: asking again for an identifier after a lookup with the first
    answer failed is the same question in words, and legitimate. So the judge
    is given what has happened since each earlier question, not just the texts.
    """

    repeats_without_new_information: bool = False
    asks_for_what_agent_can_look_up: bool = False
    reason: str = ""


@dataclass(frozen=True)
class _PhaseInput:
    """What every phase sees. ``phase`` is the discriminant.

    The inputs are a tagged union rather than one wide struct of optionals so
    that a row cannot read a field that does not exist at its phase -- there is
    no ``answer`` at ``BEFORE_SIDE_EFFECT`` to read.
    """

    facts: ContractFacts = ContractFacts()
    raw_user_message: str = ""
    conversation: Sequence[str] = ()
    commands_run: Sequence[CommandRecord] = ()
    observations: Sequence[str] = ()
    intent: Sequence[RequestedOperation] = ()

    @property
    def phase(self) -> DecisionPhase:  # pragma: no cover - overridden
        raise NotImplementedError

    def requested(self, op: str, *, conditional: Optional[bool] = None) -> bool:
        """Whether the user has asked for ``op``.

        ``conditional=None`` counts fallbacks too; ``False`` counts only firm
        requests; ``True`` only fallbacks.
        """
        return any(
            r.op == op and (conditional is None or r.conditional == conditional)
            for r in self.intent
        )

    def last_user_utterance(self) -> Optional[CommandRecord]:
        """The most recent thing the user said, as its record, or None."""
        for record in reversed(tuple(self.commands_run)):
            if record.name in USER_UTTERANCES:
                return record
        return None

    def ran(self, command_name: str) -> tuple[CommandRecord, ...]:
        """Successful runs of one command, in order."""
        return tuple(
            c for c in self.commands_run if c.name == command_name and c.success
        )

    def observation_text(self) -> str:
        """Every observation so far, concatenated.

        The coarsest possible grounding check -- "does this string appear
        anywhere in what came back" -- and the one most rows need.
        """
        return "\n".join(str(o) for o in self.observations)


@dataclass(frozen=True)
class BeforeTaskInput(_PhaseInput):
    @property
    def phase(self) -> DecisionPhase:
        return DecisionPhase.BEFORE_TASK


@dataclass(frozen=True)
class BeforeSideEffectInput(_PhaseInput):
    """A command has been chosen, its parameters extracted, and it has not run."""

    command_name: str = ""
    command_args: Mapping[str, Any] = field(default_factory=dict)

    @property
    def phase(self) -> DecisionPhase:
        return DecisionPhase.BEFORE_SIDE_EFFECT

    def id_arguments(self) -> tuple[tuple[str, str], ...]:
        """(name, value) pairs of this command's identifier-shaped arguments.

        Flattens list-valued arguments: a list of ids carries several ids and
        each has to be grounded on its own.
        """
        found: list[tuple[str, str]] = []
        for key, value in (self.command_args or {}).items():
            values = value if isinstance(value, (list, tuple)) else [value]
            found.extend(
                (key, str(v))
                for v in values
                if isinstance(v, (str, int)) and self.facts.looks_like_id(str(v))
            )
        return tuple(found)


@dataclass(frozen=True)
class AfterObservationInput(_PhaseInput):
    """A non-command tool has been chosen, with earlier results in hand."""

    pending_tool: str = ""
    pending_args: Mapping[str, Any] = field(default_factory=dict)
    observation: str = ""
    #: Set when the pending tool is a question to the user and the workflow
    #: asked for questions to be judged; None if not judged or the judge failed.
    question_judgement: Optional[QuestionJudgement] = None

    @property
    def phase(self) -> DecisionPhase:
        return DecisionPhase.AFTER_OBSERVATION

    def pending_text(self) -> str:
        return " ".join(str(v) for v in (self.pending_args or {}).values())


@dataclass(frozen=True)
class BeforeFinishInput(_PhaseInput):
    """The answer exists and has not been delivered."""

    answer: str = ""

    @property
    def phase(self) -> DecisionPhase:
        return DecisionPhase.BEFORE_FINISH


@dataclass(frozen=True)
class PolicyDecision:
    """What fired, and enough to audit why."""

    outcome: PolicyOutcome
    reason: str
    source_policy: str
    table_version: str
    proposed_action: Optional[str] = None
    rewrite: Optional[str] = None

    @property
    def is_actionable(self) -> bool:
        """Whether the caller has something to do about it.

        An ``ALLOW`` is the table declining to interfere, which reads the same
        to the loop as no match at all.
        """
        return self.outcome is not PolicyOutcome.ALLOW and bool(self.rewrite)


@dataclass(frozen=True)
class PolicyRow:
    """One rule: a predicate over the conversation, and what to say when it holds.

    ``predicate`` must be a pure function of its input. It is called on every
    step at its phase, so it must also be cheap; these are set operations over
    a trajectory, not searches.

    ``rewrite`` may be a string or a callable taking the input, for rows whose
    replacement text names the specific thing that was wrong -- "records R1, R2
    have not been read" is actionable where "read every record" is not.
    """

    row_id: str
    phases: frozenset
    outcome: PolicyOutcome
    reason: str
    predicate: Callable[[Any], bool]
    rewrite: Optional[Any] = None
    source_insights: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.row_id.strip():
            raise PolicyError(
                "a policy row needs an id; it is what a decision cites as its "
                "source_policy"
            )
        if not self.phases:
            raise PolicyError(
                f"row {self.row_id!r} declares no phase, so it can never fire"
            )
        for phase in self.phases:
            if not isinstance(phase, DecisionPhase):
                raise PolicyError(
                    f"row {self.row_id!r} declares {phase!r}, "
                    "which is not a DecisionPhase"
                )
        if self.outcome is PolicyOutcome.REWRITE and self.rewrite is None:
            raise PolicyError(
                f"row {self.row_id!r} returns REWRITE with nothing to say. A "
                "rewrite with no replacement blanks the observation and the "
                "agent re-selects the same call, which is a loop, not a policy"
            )

    def rewrite_for(self, inputs: _PhaseInput) -> Optional[str]:
        if self.rewrite is None:
            return None
        return self.rewrite(inputs) if callable(self.rewrite) else str(self.rewrite)


class DecisionTable:
    """A versioned, ordered, deterministic table. No model call, ever."""

    def __init__(
        self,
        version: str,
        rows: Sequence[PolicyRow],
        *,
        contract_version: str = TABLE_CONTRACT_VERSION,
    ) -> None:
        if not version.strip():
            raise PolicyError(
                "a decision table must be versioned: a decision that cannot "
                "name the rule set that produced it cannot be reproduced"
            )
        seen: set[str] = set()
        for row in rows:
            if row.row_id in seen:
                raise PolicyError(
                    f"duplicate row id {row.row_id!r}: a decision cites "
                    "source_policy, so ids have to identify one row"
                )
            seen.add(row.row_id)
        self.version = version
        self.contract_version = contract_version
        self.rows = tuple(rows)

    def evaluate(self, inputs: _PhaseInput) -> Optional[PolicyDecision]:
        """First matching row, or None when no row claims this input.

        None means "this table says nothing here" and the caller keeps its
        existing behaviour. That is distinct from a row returning ``ALLOW``,
        which is the table deliberately declining.
        """
        phase = inputs.phase
        for row in self.rows:
            if phase not in row.phases:
                continue
            if not row.predicate(inputs):
                continue
            return PolicyDecision(
                outcome=row.outcome,
                reason=row.reason,
                source_policy=row.row_id,
                table_version=self.version,
                proposed_action=(
                    getattr(inputs, "command_name", None)
                    or getattr(inputs, "pending_tool", None)
                ),
                rewrite=row.rewrite_for(inputs),
            )
        return None


#: No rows, so no decision, so no behaviour change. ``OFF`` plus this is identity.
NO_OP_TABLE = DecisionTable(version="no-op/1", rows=())


class PolicyDecisionPoint:
    """The single call site: mode-gated, and it keeps what it enforced."""

    def __init__(
        self,
        table: DecisionTable = NO_OP_TABLE,
        mode: PolicyMode = PolicyMode.OFF,
    ) -> None:
        self.table = table
        self.mode = mode
        self.enforced_log: list[PolicyDecision] = []

    @property
    def enabled(self) -> bool:
        return self.mode is not PolicyMode.OFF

    def decide(self, inputs: _PhaseInput) -> Optional[PolicyDecision]:
        if not self.enabled:
            return None
        decision = self.table.evaluate(inputs)
        if decision is None or not decision.is_actionable:
            return None
        self.enforced_log.append(decision)
        return decision


@dataclass
class PolicyState:
    """What the policy remembers about one conversation.

    Scoped to the conversation, NOT to a turn or a run: a record read in the
    first turn is still read in the second, and an id the user gave earlier is
    still grounded later. Clearing it per turn makes every multi-turn task look
    as though nothing had been inspected.

    Plain data so it can travel with a suspended session across processes
    (``to_dict`` / ``from_dict``).
    """

    #: Commands run and things the user said, in order.
    commands_run: list[CommandRecord] = field(default_factory=list)
    #: Both sides' messages, oldest first: ("user" | "assistant", text).
    dialogue: list[tuple[str, str]] = field(default_factory=list)
    #: The intent reader's request list (policy_intent.TrackedRequest), as dicts.
    requests: list[dict] = field(default_factory=list)
    #: What the rules enforce, derived from ``requests``.
    intent: list[RequestedOperation] = field(default_factory=list)
    #: Blocks per (rule, tool, command), for the per-rule valve.
    block_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "commands_run": [asdict(c) for c in self.commands_run],
            "dialogue": [list(d) for d in self.dialogue],
            "requests": [dict(r) for r in self.requests],
            "intent": [asdict(r) for r in self.intent],
            "block_counts": dict(self.block_counts),
        }

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "PolicyState":
        data = data or {}
        return cls(
            commands_run=[CommandRecord(**c) for c in data.get("commands_run", ())],
            dialogue=[(str(a), str(b)) for a, b in data.get("dialogue", ())],
            requests=[dict(r) for r in data.get("requests", ())],
            intent=[RequestedOperation(**r) for r in data.get("intent", ())],
            block_counts={str(k): int(v) for k, v in data.get("block_counts", {}).items()},
        )
