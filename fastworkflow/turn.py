"""Turn-level result types for fastWorkflow (v2.21 forward-compatible subset).

A *logical turn* is one user interaction with a workflow: every command
execution, clarification exchange, and failure that occurs between the user's
message and the final answer. ``TurnResult`` captures that turn.

This module ships the turn-result types. See
``docs/turn_result_design_final.md``. As of v3.0, ``CommandOutput`` carries a
singular ``command_response`` (multiplicity lives on ``command_outputs``). It
also ships the two additive capture records architecture §12.2 puts on
``TurnResult``; the block above them explains why they are thin.

``CommandResponse`` and ``CommandOutput`` are forward references resolved at
the bottom of ``fastworkflow/__init__.py`` via ``TurnResult.model_rebuild``.
"""

from __future__ import annotations

import uuid
import warnings
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field

from fastworkflow import CommandOutput, CommandResponse

# Key marking an artifacts-dict envelope whose value was offloaded to a store
# and replaced in place by a scoped reference. [A10][A47]
FW_ARTIFACT_REF_KEY = "__fw_artifact_ref__"

# Scalar types allowed inside CommandResponse.artifacts (plus dict/list/tuple
# containers of the same). Anything else is unserializable for turn records.
_ALLOWED_SCALAR_TYPES = (str, int, float, bool)
_ALLOWED_CONTAINER_TYPES = (dict, list, tuple)


def merge_artifact_responses_into(
    target: "CommandResponse",
    artifact_responses: list["CommandResponse"],
) -> None:
    """Merge artifact dicts from turn tool responses into one user-facing response. [Topic 5]

    Each key from every ``artifact_responses`` entry is copied into ``target.artifacts``.
    When a key already exists on ``target``, the incoming key is suffixed with
    ``_<increment>`` (1, 2, ...) until unused.
    """
    for artifact_response in artifact_responses:
        for key, value in artifact_response.artifacts.items():
            target_key = key
            if target_key in target.artifacts:
                increment = 1
                while f"{key}_{increment}" in target.artifacts:
                    increment += 1
                target_key = f"{key}_{increment}"
            target.artifacts[target_key] = value


def collect_artifact_responses(
    command_outputs: list["CommandOutput"],
) -> list["CommandResponse"]:
    """Flatten every command response that carries artifacts, in turn order.

    Returns the subset of command responses (across every ``command_output``)
    whose ``artifacts`` dict is non-empty, projected to the flat
    ``CommandResponse`` shape so a single user-facing ``CommandOutput`` can
    surface every structured output without nested ``TurnResult`` serialization
    (no recursion). The returned responses are the original objects.

    The framework does not interpret artifact keys or values — it only preserves
    structured outputs that would otherwise be dropped. Which keys are meaningful
    (and how to render them) is entirely the consuming client's concern.
    """
    return [
        command_output.command_response
        for command_output in command_outputs
        if command_output.command_response.artifacts
    ]


class TurnStatus(str, Enum):
    """Terminal (or suspended) status of a logical turn."""

    COMPLETED = "completed"
    AWAITING_USER = "awaiting_user"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ABANDONED = "abandoned"


def mint_turn_key(now: Optional[datetime] = None, uuid_hex: Optional[str] = None) -> str:
    """Mint a new turn key: ``YYYYMMDDTHHMMSS.ffffffZ-<uuid4 hex, 12 chars>``.

    Colon-free and lexicographically sortable; the timestamp is the logical
    turn start in UTC.

    Args:
        now: Injectable timestamp (UTC) for deterministic tests.
            Defaults to ``datetime.now(timezone.utc)``.
        uuid_hex: Injectable uniqueness suffix for deterministic tests.
            Defaults to the first 12 hex chars of a fresh ``uuid4``.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    if uuid_hex is None:
        uuid_hex = uuid.uuid4().hex[:12]
    return f"{now.strftime('%Y%m%dT%H%M%S.%f')}Z-{uuid_hex}"


def _walk_artifact_value(value: Any) -> bool:
    """Return True if *value* is record-serializable (cheap recursive check)."""
    if value is None or isinstance(value, _ALLOWED_SCALAR_TYPES):
        return True
    if isinstance(value, dict):
        return all(
            _walk_artifact_value(k) and _walk_artifact_value(v)
            for k, v in value.items()
        )
    if isinstance(value, (list, tuple)):
        return all(_walk_artifact_value(item) for item in value)
    return False


def validate_artifacts_serializable(command_output: "CommandOutput") -> list[str]:
    """Cheap recursive type-walk over each command_response's artifacts dict.

    Allowed: ``None``/``str``/``int``/``float``/``bool``, and ``dict``/``list``/
    ``tuple`` compositions thereof. Returns a list of human-readable problem
    descriptions (empty list means everything is serializable). [X3a]
    """
    problems: list[str] = []
    command_name = getattr(command_output, "command_name", "") or ""
    command_response = getattr(command_output, "command_response", None)
    artifacts = getattr(command_response, "artifacts", None) if command_response else None
    if isinstance(artifacts, dict):
        problems.extend(
            f"artifacts[{key!r}] on command '{command_name}' is {type(value)}"
            for key, value in artifacts.items()
            if not _walk_artifact_value(value)
        )
    return problems


def warn_on_unserializable_artifacts(command_output: "CommandOutput") -> None:
    """Warn (never raise) if a command output carries unserializable artifacts.

    Runs for every turn. In v2.21 this only emits a ``warnings.warn``; from
    v3.0 the same problems are refused when the turn record is filed.
    """
    if problems := validate_artifacts_serializable(command_output):
        warnings.warn(
            "Unserializable command artifacts detected: "
            + "; ".join(problems)
            + ". These artifact values will be rejected at turn-record filing "
            "from fastWorkflow v3.0 onward; store only None/str/int/float/bool "
            "and dict/list/tuple compositions thereof.",
            stacklevel=2,
        )


class TurnOutput(BaseModel):
    """The public result of one logical turn, returned by ``process_turn()``.

    The agent's contribution to a turn is only the final-answer **text**
    (``answer``); it never reports success, artifacts, next actions, or
    recommendations itself. Those structured, per-command results live on each
    entry of ``command_outputs`` (each ``CommandOutput.command_response``).

    ``TurnOutput`` is the consumer-facing slice of the internal ``TurnResult``
    (which additionally carries observability/persistence fields). See
    ``docs/turn_result_design_final.md`` section 1a.

    - ``turn_key`` is exposed as a developer handle to look up the full turn
      record in the observability UI once integrated.
    - ``command_outputs`` preserves per-command provenance, including each
      command's own ``success``/``artifacts``/timing.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    turn_key: str
    status: TurnStatus
    failure_reason: Optional[str] = None
    answer: str = ""
    command_outputs: list["CommandOutput"] = []

    @computed_field
    @property
    def success(self) -> bool:
        """Whether every command in the turn reported success.

        ``success`` is purely ``all(command_outputs succeeded)`` — **orthogonal**
        to ``status`` and ``failure_reason``. The agent always phrases its final
        answer as if it succeeded (v2.20 hard-coded the synthesized answer to
        ``success=True``), so this is the framework's signal that some command
        returned a failure code, even when the agent masked it in prose or
        recovered from it. The offending command is visible in ``command_outputs``.

        The three turn-level signals are independent; a consumer combines them:
          - ``status``         — lifecycle outcome (completed/awaiting_user/failed/…)
          - ``failure_reason`` — elaboration of a *failure* status (e.g. max_iters)
          - ``success``        — did every command succeed

        (During ``AWAITING_USER`` the pending, unanswered ask_user entry counts
        as not-yet-successful, so ``success`` is ``False`` until the turn resumes.)
        """
        return all(
            command_output.success for command_output in self.command_outputs
        )

    @property
    def command_outputs_with_artifacts(self) -> list:
        """Command outputs carrying artifacts, in turn order.

        The subset of ``command_outputs`` where any command response has a
        non-empty ``artifacts`` dict — i.e. the outputs that carry structured
        data beyond plain text, in the order they occurred within the turn.

        The framework does not interpret the artifact keys; a consuming client
        decides which of these are worth rendering richly (a "gallery", a chart,
        a download, etc.).
        """
        return [
            command_output
            for command_output in self.command_outputs
            if command_output.command_response.artifacts
        ]


# ----------------------------------------------------------------------
# Additive turn-level capture (arch §12.2, EXP-003 Phase 0)
# ----------------------------------------------------------------------
#
# §12.2 gives ``TurnResult`` two additive lists, ``execution_records`` and
# ``routing_events``. What they hold is decided by three facts about this
# codebase rather than by transcribing §6.6's ``ExecutionRecord`` field list.
#
# **The span is the record.** §6.6's realization note (revision 0.6) says
# ``ExecutionRecord`` "is a contract, not a new table — its fields land as
# versioned attributes on the extended ``fw.command.execute``/
# ``fw.agent.tool_call`` spans and the turn record, with ``command_call_id`` as
# the join key", which is why ``execution_record.py`` is not a standalone
# module. Copying those fields onto the turn record as well would
# give each of them two homes that can disagree, which is the outcome that note
# exists to prevent.
#
# **The capture policy does not reach here.** ``observability_store.
# _apply_capture_policy`` walks exactly ``record["turn_output"]
# ["command_outputs"]`` — parameters, response text, artifacts. A new top-level
# list on ``TurnResult`` is dumped straight into ``record_json`` unpoliced, so
# under the evidence profile it would be the one place default-deny does not
# apply. Every field below is therefore a framework-minted opaque id, a closed
# vocabulary defined in this module, or a count: no free text, no entity
# content, nothing a user or a workflow author supplies. That is the rule
# §6.6.1 already imposes on uncertainty signals, for the same reason — a record
# no profile will filter has to be safe to retain under every profile.
#
# **Spans are best-effort; turn records are the evidence.**
# ``observability_store.WriterHealthDelta`` states the asymmetry: a dropped span
# leaves a run valid with incomplete detail, a dropped turn record invalidates
# it, and retention prunes spans. So these two lists carry the *skeleton*
# durably — which executions happened, in what order, under which parent, and
# how the turn's text resolved — and point at the best-effort tier for
# everything else. The skeleton is the part that is genuinely missing today: an
# internal CME hop dispatches a command that produces no ``CommandOutput`` at
# all, so that execution appears nowhere in the turn record.
#
# Nothing populates either list yet: the ``CommandDispatcher`` choke point that
# would do it does not exist in this tree. Both fields default to empty, so
# every existing constructor call and every already-serialized record keeps
# validating unchanged.

# Version of the two contracts below — their field sets and the vocabularies
# their enums admit. Deliberately ONE number for both, on the same terms as
# ``tracing.SPAN_CONTRACT_VERSION``: it is honest about saying "something
# changed" rather than which of the two, and a reader joining records across
# runs needs that much to know an absent field means "this engine could not emit
# it" rather than "it did not occur". Bump it when either model gains a field or
# either vocabulary gains a member.
TURN_CAPTURE_CONTRACT_VERSION = 1

# Which matching layer decided, named for what ``intent_detection.py`` actually
# writes to ``nlu_trace["matcher_layer"]`` — ``exact_prefix``,
# ``fuzzy_prematch``, ``embedding_cache``, ``classifier``,
# ``clarification_default`` — plus ``direct_action`` for
# ``process_action_turn``, which resolves an ``Action``'s command name with no
# NLU at all and therefore emits no ``fw.nlu.intent`` span to read a layer off.
#
# Architecture §10.3's tiers (effective-context alias, canonical definition id,
# unique simple name) are deliberately NOT here. They belong to the P0
# capability index and ``CommandDispatcher``, neither of which exists in this
# tree, and a vocabulary describing a resolution order no code performs would
# make a record claim more than the runtime can know. They join this list, as a
# ``TURN_CAPTURE_CONTRACT_VERSION`` bump, when the dispatcher lands.
RoutingTier = Literal[
    "direct_action",
    "exact_prefix",
    "fuzzy_prematch",
    "embedding_cache",
    "classifier",
    "clarification_default",
    # An emitter that did not say which layer decided. A real member rather
    # than a default, so "not recorded" never reads as a tier that ran.
    "unknown",
]

# What one attempt did. ``unresolved`` is not a failure: ``CommandNamePrediction
# .predict`` returns no command name when the utterance matches nothing in the
# context it ran against, and the CME wildcard command then walks up the parent
# chain — so unresolved attempts are ordinary steps of a successful turn.
# ``error`` is the ``_predict_impl`` exception path, where the prediction did
# not complete and there is no decision to characterise at all.
RoutingOutcome = Literal["resolved", "unresolved", "ambiguous", "error"]


class _CapturedRecord(BaseModel):
    """Shared posture for the two additive capture records.

    ``extra="forbid"`` and ``frozen=True`` for ``decision_signals._Strict``'s
    reasons: a typo'd field that parses is a field nobody notices is missing,
    and a captured record must not be editable after the fact by the code being
    measured. The cost is that a reader on an older contract meeting a newer
    record raises instead of reading it partially — which is the direction this
    repo chooses everywhere else, and ``contract_version`` is what lets such a
    reader say precisely what it met.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: int = TURN_CAPTURE_CONTRACT_VERSION


class ExecutionRecordRef(_CapturedRecord):
    """One command execution of this turn, and where its full record lives.

    A reference, not a copy: §6.6's ``ExecutionRecord`` fields are span
    attributes, and ``command_call_id`` is the join key that reaches them (arch
    §12.0 delta 1, the same id ``CommandOutput.command_call_id`` carries). What
    is here is what a span cannot be relied upon to still hold — this turn's own
    ordered ledger of what it executed.

    ``span_id`` is None for an execution that has no span of its own, which is a
    fact rather than a gap. ``CommandExecutor`` reaches internal CME/core
    commands through ``perform_action`` under an open ``tracing.call_scope``,
    and capture rides the emission sites that already exist rather than adding
    new ones, so those inner calls are recorded on the enclosing span's
    ``child_calls`` ledger instead. Such an execution has no ``CommandOutput``
    either — which is exactly why it needs a row here.

    Deliberately absent: ``status`` and ``failure``. §6.6 names
    ``ExecutionStatus`` and ``TypedFailure`` and the architecture document
    defines neither, while the outcome of an execution that produced a
    ``CommandOutput`` already sits on that output under the same
    ``command_call_id``. Inventing a status vocabulary to restate a value
    recorded twice already is how two homes start disagreeing.
    """

    command_call_id: str
    # Recorded as None rather than omitted at the outermost dispatch, for the
    # reason ``tracing`` stamps ``parent_call_id`` explicitly: an absent key and
    # a top-level call look identical to a reader, and "this dispatch had no
    # parent" is a fact worth stating.
    parent_call_id: Optional[str] = None
    command_ordinal: int = Field(ge=0)
    span_id: Optional[str] = None


class RoutingEvent(_CapturedRecord):
    """One attempt to resolve this turn's text to a command identity.

    Not a ``Ref``, because no fatter routing record exists elsewhere for it to
    point at: the tier and the outcome ARE the event. ``span_id`` reaches the
    ``fw.nlu.intent`` span that carries the utterance, the context name, the
    confidences and the candidate names — every one of which is either entity
    content or workflow-author text, all of which the span's own policy governs,
    and none of which this record may therefore repeat.

    One event per attempt, not per turn. The CME wildcard command walks up the
    parent-context chain calling ``CommandNamePrediction.predict`` once per
    context, and each of those attempts is a routing decision with its own tier
    and outcome; collapsing them into a single "how the turn routed" would erase
    exactly the near-miss structure a reliability analysis reads.
    """

    ordinal: int = Field(ge=0)
    tier: RoutingTier
    outcome: RoutingOutcome
    # How many candidates the attempt could not choose between, when the layer
    # that ran reports one. None means that layer produces no candidate set —
    # never a set of size zero.
    candidate_count: Optional[int] = Field(default=None, ge=0)
    span_id: Optional[str] = None
    # The dispatch this routing produced, when it produced one: the join from a
    # routing decision to the execution that carried it out. None for an attempt
    # that resolved nothing, and for one whose resolution was later discarded.
    command_call_id: Optional[str] = None


class TurnResult(BaseModel):
    """The complete internal capture of one logical turn.

    Composes the consumer-facing ``turn_output`` plus internal-only
    observability/persistence fields. ``process_turn()`` returns the
    ``turn_output``; the surrounding ``TurnResult`` is the framework's
    system-of-record for the turn (one logical turn = one key = one record,
    across any number of suspensions).

    ``execution_records`` and ``routing_events`` are architecture §12.2's
    additive fields, appended last and empty by default. They are the only part
    of this model a new API may return that the legacy projection
    (``TurnOutput``) does not carry.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    turn_output: "TurnOutput"
    channel_id: Optional[str] = None
    conversation_id: Optional[int] = None
    ordinal: Optional[int] = None
    user_message: str
    refined_user_message: Optional[str] = None
    # The conversation-history entry this turn appended, in the canonical 3-key
    # shape's own terms. Populated at terminal finalize only when the turn
    # actually grew the history; an awaiting_user emission leaves both None and
    # the terminal upsert fills them. This is what makes a turn record usable as
    # conversation memory rather than only as a trace
    # (observability_phase7_consolidation_design.md §2.1).
    conversation_summary: Optional[str] = None
    conversation_traces: Optional[str] = None
    entry_workflow_name: str = ""
    entry_context: str = ""
    continuation_of: Optional[str] = None
    trajectory_ref: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    suspended_ms: int = 0
    metadata: dict[str, Any] = {}
    # The experiment container's labels (`fix-bn1`,
    # docs/experiment_container_design.md `[XR4]`/`[XR17]`). Appended, never
    # reordered, all defaulted: an already-serialized record with none of them
    # still validates, and every existing keyword construction still works.
    #
    # NULL means "not part of an experiment", which is every ordinary chatbot
    # and CLI turn. They are bound onto the WorkflowExecutionContext before the
    # turn by whatever is running the experiment, exactly as channel_id and
    # conversation_id are, and copied here at construction.
    experiment_id: Optional[str] = None
    task_id: Optional[str] = None
    attempt: Optional[int] = None
    claim_epoch: Optional[int] = None
    server_incarnation: Optional[str] = None
    # Appended, never reordered: §12.2 keeps this model compatible, so an
    # already-serialized record with neither key still validates and every
    # existing keyword construction still works. See the "Additive turn-level
    # capture" block above for why these are thin correlation records rather
    # than copies of §6.6's field list.
    execution_records: tuple[ExecutionRecordRef, ...] = ()
    routing_events: tuple[RoutingEvent, ...] = ()
