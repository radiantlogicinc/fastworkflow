"""OTel-shaped trace emission at the agent↔workflow boundary.

Implements the flight-recorder half of the observability design
(docs/fastworkflow_observability_studio_design.md §3.1): a ``TraceSink``
protocol with a no-op default, the v1 span taxonomy, and safe emission
helpers that NEVER raise to the caller — a broken sink degrades to a log
line, not a failed turn.

Spans are OTel-*aligned* records, not wire-conformant OTel:
``trace_id`` is the logical turn_key, span ids are opaque strings, and the
translation to real OTel ids is an external script's contract.

Sink discovery is duck-typed off the host object (WorkflowExecutionContext,
or ChatSession delegating to its core) via ``trace_sink`` /
``current_turn_key`` / ``trace_span_stack`` — deliberately NOT the
transport-queue contract, so queue-less embedders still trace.

This module is stdlib-only by design: it is imported by core runtime
modules and must never pull torch/dspy/transformers.

For the decision-signal capture slice (arch §12.0 deltas 1/2/4) this module also
imports ``capture_policy`` and ``decision_signals``, which are architecture §22
leaf modules — standard library, Pydantic, and ``runtime_manifest`` only. The
invariant the paragraph above protects is unchanged: nothing on this import path
reaches torch, dspy or transformers. They are imported here rather than at each
emission site because ``command_executor`` and ``workflow_execution_context``
both stamp the same handles, and ``workflow_execution_context`` cannot import
``command_executor`` at module scope (it defers that import to ``__init__`` to
break a cycle).
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Protocol, runtime_checkable

from fastworkflow import runtime_manifest
from fastworkflow.observability import capture_policy, decision_signals, enrichment

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Host propagation for deep emission sites (v2 spans, D3 as amended)
# ----------------------------------------------------------------------
#
# The NLU internals (intent detection, parameter extraction) run several call
# frames below CommandExecutor.invoke_command and have no reference to the
# WEC/ChatSession that owns the trace sink. Rather than threading a host
# parameter through the whole pipeline, invoke_command binds the host into a
# ContextVar for the duration of the call; deep sites read it back with
# ``current_host()``. The pipeline is synchronous on the calling thread, so
# the binding is race-free per turn — and it honors [R28]: the sink is still
# reached via the WEC, never via a transport queue.

_current_host: contextvars.ContextVar = contextvars.ContextVar(
    "fw_trace_host", default=None
)


def current_host() -> Any:
    """The trace host bound by the nearest enclosing ``host_scope`` (or None)."""
    return _current_host.get()


@contextlib.contextmanager
def host_scope(host: Any) -> Iterator[None]:
    """Bind *host* as the current trace host for the enclosed call stack."""
    token = _current_host.set(host)
    try:
        yield
    finally:
        _current_host.reset(token)


# ----------------------------------------------------------------------
# Command call identity (arch §12.0 delta 1, §12.1 item 5)
# ----------------------------------------------------------------------
#
# A ``CommandOutput`` and the ``fw.command.execute`` span that produced it shared
# no key: the span carried a random uuid4 and the output carried nothing at all,
# so a reader holding a turn record could only guess which span belonged to which
# command outcome, from ordering and timestamps. ``command_call_id`` is that key,
# minted once per dispatch and stamped on both sides.
#
# The ContextVar is for the second half of the problem. An application command is
# dispatched through ``CommandExecutor.invoke_command``, which reaches the
# application only via an internal CME ``perform_action`` hop, and that hop may
# itself dispatch another core command — ``command_metadata_extraction/_commands/
# wildcard.py`` does exactly this for the ``IntentDetection/*`` and
# ``ErrorCorrection/*`` commands. Those inner calls have no span of their own,
# because capture rides the emission sites that already exist rather than adding
# new ones, so their correlation to the enclosing command is recorded as a
# child-call ledger stamped on the parent's span.

ATTR_COMMAND_CALL_ID = "command_call_id"
ATTR_PARENT_CALL_ID = "parent_call_id"
ATTR_CHILD_CALLS = "child_calls"
ATTR_CONTEXT_BEFORE = "context_before"
ATTR_CONTEXT_AFTER = "context_after"
ATTR_CONSEQUENCE = "consequence"

# The emitter's own attribute-contract version, stamped on every span by `_emit`
# (arch §12.0 delta 5). It is an ATTRIBUTE rather than a `Span` field because
# `spans` has fixed columns and `attributes` is the JSON bag that survives the
# store boundary — a span that describes its own contract has to carry the
# version where the store actually keeps it.
ATTR_SPAN_CONTRACT_VERSION = "span_contract_version"

_current_call_id: contextvars.ContextVar = contextvars.ContextVar(
    "fw_command_call_id", default=None
)

_child_call_ledger: contextvars.ContextVar = contextvars.ContextVar(
    "fw_child_call_ledger", default=None
)


def new_command_call_id() -> str:
    """Mint one command-call id."""
    return uuid.uuid4().hex


def current_call_id() -> Optional[str]:
    """The command call this frame runs under, or None at the outermost dispatch."""
    return _current_call_id.get()


@contextlib.contextmanager
def call_scope(call_id: str, *, command_name: Optional[str] = None) -> Iterator[list]:
    """Bind *call_id* as the current command call; yield the child-call ledger.

    The ledger is created by the OUTERMOST scope and shared by every nested one,
    so one dispatch produces one flat list in which each entry names its own
    parent. Yielding it rather than making the caller read it back through the
    ContextVar matters: the span closes *after* the scope has exited, and by then
    the ContextVar has been reset.

    A scope with no parent adds nothing to the ledger — it is not a child of
    anything, and its own id is already on the span.
    """
    parent = _current_call_id.get()
    ledger = _child_call_ledger.get()
    ledger_token = None
    if ledger is None:
        ledger = []
        ledger_token = _child_call_ledger.set(ledger)
    if parent is not None:
        ledger.append(
            {
                "call_id": call_id,
                "parent_call_id": parent,
                "command_name": command_name,
            }
        )
    call_token = _current_call_id.set(call_id)
    try:
        yield ledger
    finally:
        _current_call_id.reset(call_token)
        if ledger_token is not None:
            _child_call_ledger.reset(ledger_token)


# ----------------------------------------------------------------------
# Span-name taxonomy
# ----------------------------------------------------------------------

# The version of the span ATTRIBUTE contract — which keys each emitter sets and
# what they mean. Distinct from the DB's `user_version` (which versions tables and
# columns) and from `turns.record_version` (which versions the TurnResult
# envelope): `spans.attributes` is an unversioned JSON bag, so a replay or a
# calibration report comparing two runs has no way to notice that an emitter
# started spelling a key differently or changed a value's units.
#
# Recorded in run provenance by the evidence-grade run mode (arch §12.0 delta 6).
# Bump this whenever an emitter's attribute keys or their meanings change.
#
# This is deliberately ONE number for the whole taxonomy. Arch §12.0 delta 5 asks
# for a version *per emitter*, which is fix-ajv.8's work; a single number is what
# the evidence-grade run mode needs to detect drift between two runs, and it is
# honest about what it can and cannot localize — it says "something changed",
# not which emitter.
#
# v2 (EXP-003 capture slice): fw.command.execute and fw.agent.tool_call gained
# command_call_id, parent_call_id, child_calls, context_before, context_after and
# consequence. Bumped under the rule stated directly above — a run recorded
# before this and a run recorded after it are not attribute-comparable, and a
# calibration report joining them would silently read the older run's missing
# handles as "no context transition" rather than as "not captured".
#
# Amendment (fix-ajv.8): the per-emitter versions arch §12.0 delta 5 asks for now
# exist in `SPAN_CONTRACTS` below, and this number is KEPT rather than replaced.
# The two answer different questions and a reader needs both: a run-to-run
# comparison reads this one to learn that something moved, and the per-emitter map
# localizes what. Replacing it would force every comparison to diff a map to
# answer a yes/no question, and dropping it from provenance would break every
# existing reader of `ObservabilityProvenance.span_contract_version`.
#
# v3: every span now carries `span_contract_version` in its attributes;
# fw.agent.tool_call gained the §12.1.1 capture keys at its third emission site
# (workflow_agent.py, previously the only unmigrated one); and fw.nlu.intent's
# `classifier` attribute gained `topk_scores`.
#
# v4: routing. fw.nlu.intent's known-name refusal keys were declared here.
# The emitter had already been writing them. An intermediate draft of this
# same number also named an auto-navigation flag and four fw.command.execute
# keys for a composed step. Those keys did not remain in the contracts.
#
# v5: the aggregate moves so this number and SPAN_CONTRACTS describe the same
# taxonomy. fw.command.execute v3 has no composed-step attributes.
# fw.nlu.intent v3 keeps the known-name keys and does not carry
# auto_navigation_enabled. Auto-navigation was removed.
#
# v6: fw.search.route added -- the optional decision-model call that routes a
# search_memory request over a listing (observation_offloading/search_router.py).
# No existing span changed.
SPAN_CONTRACT_VERSION = 6

# v1 — emitted at the agent↔workflow boundary (decision D3).
SPAN_TURN = "fw.turn"
SPAN_PLANNER_PLAN = "fw.planner.plan"
SPAN_PLANNER_REPLAN = "fw.planner.replan"
SPAN_AGENT_TOOL_CALL = "fw.agent.tool_call"
SPAN_COMMAND_EXECUTE = "fw.command.execute"
SPAN_ASK_USER = "fw.ask_user"

V1_SPAN_NAMES = frozenset(
    {
        SPAN_TURN,
        SPAN_PLANNER_PLAN,
        SPAN_PLANNER_REPLAN,
        SPAN_AGENT_TOOL_CALL,
        SPAN_COMMAND_EXECUTE,
        SPAN_ASK_USER,
    }
)

# The agent loop's own structure. Without these two, a turn's shape has to be
# INFERRED by a reader — the planner and the ReAct calls are flat siblings of
# fw.turn, and "which tool call belongs to which reasoning step" is only
# recoverable from DSPy module names plus timestamps. They make the two levels
# a developer actually thinks in — the executor ran, and it took these steps —
# structural facts instead of a heuristic.
#
# fw.agent.execute is the executor as a phase, sibling to fw.planner.plan. It
# is NOT fw.command.execute: that one is a single command inside a tool call,
# this one is the whole loop.
SPAN_AGENT_EXECUTE = "fw.agent.execute"
SPAN_AGENT_STEP = "fw.agent.step"

AGENT_LOOP_SPAN_NAMES = frozenset({SPAN_AGENT_EXECUTE, SPAN_AGENT_STEP})

# Originally reserved-for-v2 so the schema needed no migration when the deeper
# emitters landed. They have (D3 amendment, 2026-08-26): fw.nlu.* emit inside
# the CME pipeline and fw.llm.call at the DSPy callback level; only fw.train.*
# is still reserved. The set name is historical and kept for stability.
SPAN_NLU_INTENT = "fw.nlu.intent"
SPAN_NLU_PARAM_EXTRACTION = "fw.nlu.param_extraction"
SPAN_LLM_CALL = "fw.llm.call"
SPAN_TRAIN_PREFIX = "fw.train."

RESERVED_V2_SPAN_NAMES = frozenset(
    {SPAN_NLU_INTENT, SPAN_NLU_PARAM_EXTRACTION, SPAN_LLM_CALL, SPAN_TRAIN_PREFIX}
)

# The decision-model call that may route one search_memory request over a
# listing. A model call of its own, so kind "llm"; not fw.llm.call, which is the
# DSPy callback's record and what cost and cut-at-limit readers key on. It
# carries the verdict and its cost, never the question or the observation --
# those are on the search's offload event, and copying them here would put them
# in a second store.
SPAN_SEARCH_ROUTE = "fw.search.route"


# ----------------------------------------------------------------------
# Per-emitter attribute contracts (arch §12.0 delta 5, FW-REQ-019 clause 3)
# ----------------------------------------------------------------------
#
# `SPAN_CONTRACT_VERSION` above says "something in the taxonomy changed". This
# table says what each emitter promises to write, and versions that promise
# separately, so a reader that notices the aggregate moved can find out which
# emitter moved and which of its keys it should stop expecting.
#
# `attributes` is not documentation. tests/test_span_contract_versioning.py
# recovers the key set every emission site actually writes — by AST, including
# keys added to an attribute bag several frames away — and fails when the two
# disagree. So adding a key to an emitter forces an edit here, which puts the
# version right under the cursor. That is as close to "you cannot add a key
# without deciding about the version" as a static check gets; the hand-written
# list it replaces would have gone stale the first time a bag gained a key
# somewhere other than the emission site.
#
# The version numbers all start at 1 on purpose. This counter begins here, and
# back-dating it would invent per-emitter history that was never recorded on any
# span — a span with no `span_contract_version` attribute at all is exactly the
# "recorded before per-emitter versioning" case, and that is already legible.
# Bump one when its emitter's keys, or their meanings, change.


@dataclass(frozen=True)
class SpanContract:
    """One emitter's attribute promise, and the version of that promise."""

    version: int
    attributes: frozenset[str]


SPAN_CONTRACTS: dict[str, SpanContract] = {
    SPAN_TURN: SpanContract(
        version=1,
        attributes=frozenset(
            {
                "turn_key",
                "channel_id",
                "conversation_id",
                "user_message",
                "status",
                "success",
                "failure_reason",
                "suspended_ms",
                "context_mutations",
            }
        ),
    ),
    SPAN_ASK_USER: SpanContract(
        version=1,
        attributes=frozenset(
            {"agent_query", "attempt", "user_response", "human_wait_ms"}
        ),
    ),
    # v3: the four auto-navigation keys (v2, ido-8ps.9) are gone with the
    # two-step dispatch that wrote them. Every execute step is now a step the
    # agent typed, so there is no composed-step shape to tell apart.
    SPAN_COMMAND_EXECUTE: SpanContract(
        version=3,
        attributes=frozenset(
            {
                "raw_command",
                "parameters",
                "response_text",
                "success",
                "error_type",
                ATTR_COMMAND_CALL_ID,
                ATTR_PARENT_CALL_ID,
                ATTR_CHILD_CALLS,
                ATTR_CONTEXT_BEFORE,
                ATTR_CONTEXT_AFTER,
                ATTR_CONSEQUENCE,
            }
        ),
    ),
    # Emitted from THREE sites — workflow_agent._execute_workflow_query and both
    # WorkflowExecutionContext dispatch paths — which is why one version for this
    # name would have been a lie until fix-ajv.3's capture reached the third.
    # It now has, so the three write the same keys and one version is honest; the
    # conformance test asserts that agreement rather than trusting this comment.
    # The alternative, versioning the sites apart, was rejected: they describe the
    # same thing (an agent asked the workflow to run one command) and a reader
    # joining them would have had to learn which of three dialects each span was
    # written in.
    SPAN_AGENT_TOOL_CALL: SpanContract(
        version=1,
        attributes=frozenset(
            {
                "raw_command",
                "response_text",
                "success",
                "error_type",
                ATTR_COMMAND_CALL_ID,
                ATTR_CONTEXT_BEFORE,
                ATTR_CONTEXT_AFTER,
                ATTR_CONSEQUENCE,
            }
        ),
    ),
    SPAN_AGENT_EXECUTE: SpanContract(
        version=1,
        attributes=frozenset(
            {
                "agent_input",
                "resumed",
                "model",
                "attempts",
                "final_answer",
                "suspended",
                "clarification",
                "exhausted",
                "error_type",
            }
        ),
    ),
    # v2: `roster_nudge` (ido-8ps.27) marks the one step whose observation is
    # not a tool result at all. A finish action taken while named items of the
    # request were never the subject of a command has its "Completed."
    # observation replaced by a bounded harness note and the loop continues, so
    # a reader counting tool results would otherwise count that note as one.
    SPAN_AGENT_STEP: SpanContract(
        version=2,
        attributes=frozenset(
            {
                "step_index",
                "thought",
                "tool_name",
                "tool_args",
                "observation",
                "clarification",
                "recovered",
                "tool_error",
                "error_type",
                "roster_nudge",
            }
        ),
    ),
    # One emitter, two names: build_query_with_next_steps opens .replan instead of
    # .plan when it was re-triggered mid-turn. Same keys, so same version, and
    # `replan_trigger` is on both — None on a first plan says "this was the first
    # plan", where an absent key would only say "older record".
    SPAN_PLANNER_PLAN: SpanContract(
        version=1,
        attributes=frozenset({"model", "replan_trigger", "plan"}),
    ),
    SPAN_PLANNER_REPLAN: SpanContract(
        version=1,
        attributes=frozenset({"model", "replan_trigger", "plan"}),
    ),
    # v2: the known-name refusal keys. The emitter wrote the three
    # `known_name_*` keys before they were declared here.
    # v3: `auto_navigation_enabled` is absent. Auto-navigation was removed, so
    # there is no flag and no composed step for a measured run to carry.
    SPAN_NLU_INTENT: SpanContract(
        version=3,
        attributes=frozenset(
            {
                "context",
                "stage",
                "utterance",
                "matcher_layer",
                "known_name_foreign_context",
                "known_name_owner_contexts",
                "known_name_foreign_context_hint",
                "escalation_outcome",
                "fuzzy_distance",
                "fuzzy_threshold",
                "fuzzy_prematch_tie",
                "candidate_count",
                "cache_similarity",
                "cache_similarity_threshold",
                "classifier",
                "classifier_signal_version",
                "candidates",
                "escalation_labels_discarded",
                "reducible",
                "decision_uncertainty",
                "command_name",
                "is_cme_command",
                "ambiguous",
                "resolved",
            }
        ),
    ),
    SPAN_NLU_PARAM_EXTRACTION: SpanContract(
        version=1,
        attributes=frozenset(
            {
                "command_name",
                "retry_round",
                "extraction_method",
                "missing_fields",
                "invalid_fields",
                "db_lookup",
                "validation_hook",
                "decision_uncertainty",
                "parameters_valid",
            }
        ),
    ),
    SPAN_LLM_CALL: SpanContract(
        version=1,
        attributes=frozenset(
            {
                "module",
                "capture_source",
                "model",
                "messages",
                "prompt",
                "call_kwargs",
                "module_chain",
                "module_input",
                "module_output",
                "module_exception",
                "reasoning",
                "output",
                "usage",
                "cost",
                "history_uuid",
                "response_model",
                "cache_hit",
                "provider_response",
                "usage_capture",
                "exception",
            }
        ),
    ),
    SPAN_SEARCH_ROUTE: SpanContract(
        version=1,
        attributes=frozenset(
            {
                "model",
                "choice",
                "p_all_rows",
                "for_report",
                "latency_ms",
                "input_tokens",
                "output_tokens",
                "error_type",
            }
        ),
    ),
}

# `fw.train.*` is a reserved PREFIX with no emitter — nothing in the package opens
# a span under it — so it has no contract to version. It gets one when something
# emits one, and the conformance test is what will notice.

# Written by the emission machinery rather than by an emitter, so it is not part
# of any emitter's declared set: it describes the record, not the thing recorded.
ENVELOPE_ATTRIBUTES = frozenset({ATTR_SPAN_CONTRACT_VERSION})


def span_contract_versions() -> dict[str, int]:
    """`span name -> attribute-contract version`, for run provenance (§12.4)."""
    return {name: contract.version for name, contract in SPAN_CONTRACTS.items()}


# Span kinds (spans.kind column): internal | llm | human_wait | tool.
KIND_INTERNAL = "internal"
KIND_LLM = "llm"
KIND_HUMAN_WAIT = "human_wait"
KIND_TOOL = "tool"

# Span statuses. "open" means started and not yet ended; a store treats a
# re-emission of the same span_id as an idempotent upsert ([R2][R6]).
STATUS_OPEN = "open"
STATUS_OK = "ok"
STATUS_ERROR = "error"
STATUS_CANCELLED = "cancelled"
STATUS_AWAITING_USER = "awaiting_user"

#: The cap on ONE span-attribute value written to the observability store, in
#: UTF-8 bytes. A constant since ``ido-pyw.1``: it bounds a database row, not a
#: model prompt, so it is not one of the context-window budgets in
#: ``fastworkflow.context_budget`` and does not scale with a model.
MAX_ATTR_BYTES = 16384
#: The pre-``ido-pyw.1`` name, kept as an alias for readers of the provenance
#: record.
_DEFAULT_MAX_ATTR_BYTES = MAX_ATTR_BYTES


def is_control_signal(exc: BaseException) -> bool:
    """True for the BaseExceptions that mean "pause", not "fail".

    `CommandCancelledError` and `AskUserSuspend` both unwind a dispatch without
    anything having gone wrong: the first suspends a trajectory for ask_user
    resume, the second is raised by ask_user when no user_message_queue is
    configured. Both subclass BaseException precisely so ordinary
    `except Exception` handlers do not swallow them.

    Imports are deferred: both defining modules import this one.
    """
    from fastworkflow.utils.react import AskUserSuspend
    from fastworkflow.workflow_execution_context import CommandCancelledError

    return isinstance(exc, (AskUserSuspend, CommandCancelledError))


def status_for_dispatch_exception(exc: BaseException) -> str:
    """The span status for an exception that ended a command dispatch.

    ONE mapping in ONE place, because five sites used to each carry their own
    isinstance check and they did not agree. The same
    AskUserSuspend closed as `error` in three of them, `awaiting_user` in a
    fourth, and escaped a fifth without closing its span at all — so an
    ordinary pause for input drew as a red ERROR node in the chatbot waterfall
    (index.html:1671), and what a reader saw depended on which layer happened
    to catch it.

    Control signals map to CANCELLED rather than AWAITING_USER deliberately.
    `awaiting_user` is a TURN-level state in this codebase: the store's
    non-terminal turn status, and the only thing the SPA tests it for
    (index.html:803/840/2342 all read `turn.status`). Nothing anywhere reads a
    SPAN status of awaiting_user. A span status describes that span's own
    outcome — this dispatch was cut short — while "we are waiting on a human"
    is recorded once, on the turn, where readers already look for it.
    """
    return STATUS_CANCELLED if is_control_signal(exc) else STATUS_ERROR


@dataclass
class Span:
    """One OTel-shaped span record (spans table shape, design §3.2)."""

    span_id: str
    trace_id: str  # = turn_key
    name: str
    kind: str = KIND_INTERNAL
    parent_span_id: Optional[str] = None
    channel_id: Optional[str] = None
    command_name: Optional[str] = None
    context: Optional[str] = None
    start_ns: int = 0
    end_ns: Optional[int] = None
    status: str = STATUS_OPEN
    attributes: dict[str, Any] = field(default_factory=dict)
    experiment_id: Optional[str] = None
    task_id: Optional[str] = None
    attempt: Optional[int] = None
    claim_epoch: Optional[int] = None
    server_incarnation: Optional[str] = None


# ----------------------------------------------------------------------
# Sink protocol
# ----------------------------------------------------------------------


@runtime_checkable
class TraceSink(Protocol):
    """Trace/record sink (design §3.1). Implementations must never raise to
    the caller; the emission helpers below additionally guard every call."""

    def emit_span(self, span: Span) -> None: ...

    def emit_turn_record(self, record: Any) -> None:
        """Receive the internal TurnResult at turn finalize (typed ``Any`` to
        keep this module import-light)."""
        ...

    def record_conversation_label(
        self,
        channel_id: str,
        conversation_id: int,
        topic: Optional[str],
        summary: Optional[str],
    ) -> None: ...  # [R15]


class NoOpTraceSink:
    """Default sink: tracing structurally present, nothing recorded."""

    def emit_span(self, span: Span) -> None:
        pass

    def emit_turn_record(self, record: Any) -> None:
        pass

    def record_conversation_label(
        self,
        channel_id: str,
        conversation_id: int,
        topic: Optional[str],
        summary: Optional[str],
    ) -> None:
        pass


# ----------------------------------------------------------------------
# Span identity
# ----------------------------------------------------------------------


def deterministic_span_id(turn_key: str, span_name: str, attempt: int = 0) -> str:
    """Deterministic span id for spans that must close in a different process
    than the one that opened them (fw.turn, fw.ask_user)."""
    digest = hashlib.sha256(f"{turn_key}|{span_name}|{attempt}".encode()).hexdigest()
    return digest[:32]


def root_span_id(turn_key: str) -> str:
    """The fw.turn root span id for a logical turn."""
    return deterministic_span_id(turn_key, SPAN_TURN, 0)


def cap_attr_value(value: Any) -> Any:
    """Cap one attribute value at ``MAX_ATTR_BYTES``.

    Truncation is lossy-and-counted: an over-limit string becomes an
    envelope carrying ``truncated: True``, the original byte length, and the
    sha256 of the original — never a silent prefix.
    """
    if not isinstance(value, str):
        return value
    limit = MAX_ATTR_BYTES
    raw = value.encode("utf-8")
    if len(raw) <= limit:
        return value
    return {
        "truncated": True,
        "original_length": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "value": raw[:limit].decode("utf-8", errors="ignore"),
    }


def _capped(attributes: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not attributes:
        return {}
    capped = {key: cap_attr_value(value) for key, value in attributes.items()}
    enrichment._apply(attributes, capped)
    return capped


# ----------------------------------------------------------------------
# Duck-typed host access (WEC directly, or ChatSession via _core)
# ----------------------------------------------------------------------


def _resolve(host: Any, attr: str) -> Any:
    value = getattr(host, attr, None)
    if value is not None:
        return value
    core = getattr(host, "_core", None)
    if core is not None:
        return getattr(core, attr, None)
    return None


def get_sink(host: Any) -> Optional[TraceSink]:
    sink = _resolve(host, "trace_sink")
    return sink if sink is not None and not isinstance(sink, NoOpTraceSink) else None


def get_turn_key(host: Any) -> Optional[str]:
    return _resolve(host, "current_turn_key")


def get_channel_id(host: Any) -> Optional[str]:
    return _resolve(host, "observability_channel_id")


def get_experiment_claim(host: Any) -> dict[str, Any]:
    return _resolve(host, "observability_experiment_claim") or {}


def _get_stack(host: Any) -> Optional[list]:
    return _resolve(host, "trace_span_stack")


# ----------------------------------------------------------------------
# Emission helpers — never raise to the caller
# ----------------------------------------------------------------------


def _emit(sink: TraceSink, span: Span) -> None:
    # Stamped here rather than at each emission site: this is the single funnel
    # every span passes through on its way to a sink — including the ones
    # reconstructed as a bare `Span` to close across a process boundary — so no
    # emitter can forget, and a span read back out of the store describes its own
    # attribute contract without a lookup table. An unregistered name (a future
    # fw.train.* span) is left unstamped rather than given a made-up version.
    contract = SPAN_CONTRACTS.get(span.name)
    if contract is not None:
        span.attributes[ATTR_SPAN_CONTRACT_VERSION] = contract.version
    try:
        sink.emit_span(span)
    except Exception as exc:  # a broken sink must never fail a turn
        logger.warning(f"TraceSink.emit_span failed (span {span.name}): {exc!r}")


def datetime_to_ns(value: Any) -> Optional[int]:
    """Epoch nanoseconds for a datetime, or None when absent/unparseable."""
    try:
        return int(value.timestamp() * 1_000_000_000) if value is not None else None
    except Exception:
        return None


def start_span(
    host: Any,
    name: str,
    *,
    kind: str = KIND_INTERNAL,
    attributes: Optional[dict[str, Any]] = None,
    span_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
    command_name: Optional[str] = None,
    context: Optional[str] = None,
    emit_open: bool = False,
    use_stack: bool = True,
) -> Optional[Span]:
    """Open a span under the host's current turn; returns None when there is
    no active sink or no open turn. Never raises.

    Short-lived spans are emitted once, at ``end_span``; pass
    ``emit_open=True`` for long-lived spans (fw.turn, fw.ask_user) whose open
    event must be visible before — and closable after — a suspension.
    ``use_stack=False`` keeps a span off the parenting stack (fw.turn and
    fw.ask_user: children reach the root via its deterministic id, which
    survives suspension where the in-memory stack does not).
    """
    try:
        sink = get_sink(host)
        if sink is None:
            return None
        turn_key = get_turn_key(host)
        if not turn_key:
            return None

        stack = _get_stack(host)
        if parent_span_id is None:
            if stack:
                parent_span_id = stack[-1].span_id
            elif name != SPAN_TURN:
                parent_span_id = root_span_id(turn_key)

        claim = get_experiment_claim(host)
        span = Span(
            span_id=span_id or uuid.uuid4().hex,
            trace_id=turn_key,
            name=name,
            kind=kind,
            parent_span_id=parent_span_id,
            channel_id=get_channel_id(host),
            command_name=command_name,
            context=context,
            start_ns=time.time_ns(),
            status=STATUS_OPEN,
            attributes=_capped(attributes),
            experiment_id=claim.get("experiment_id"),
            task_id=claim.get("task_id"),
            attempt=claim.get("attempt"),
            claim_epoch=claim.get("epoch"),
            server_incarnation=claim.get("server_incarnation"),
        )

        if use_stack and stack is not None:
            stack.append(span)

        if emit_open:
            _emit(sink, span)
        return span
    except Exception as exc:
        logger.warning(f"start_span({name}) failed: {exc!r}")
        return None


def end_span(
    host: Any,
    span: Optional[Span],
    *,
    status: str = STATUS_OK,
    attributes: Optional[dict[str, Any]] = None,
    command_name: Optional[str] = None,
    context: Optional[str] = None,
    close: bool = True,
) -> None:
    """Finish (or, with ``close=False``, update-in-place) a span and emit it.
    Never raises. A None span (start_span declined) is a silent no-op."""
    if span is None:
        return
    try:
        if close:
            span.end_ns = time.time_ns()
        span.status = status
        if attributes:
            span.attributes.update(_capped(attributes))
        if command_name is not None:
            span.command_name = command_name
        if context is not None:
            span.context = context

        stack = _get_stack(host)
        if stack is not None and span in stack:
            stack.remove(span)

        sink = get_sink(host)
        if sink is not None:
            _emit(sink, span)
    except Exception as exc:
        logger.warning(f"end_span({span.name}) failed: {exc!r}")


# ----------------------------------------------------------------------
# Capture projection (arch §12.0 deltas 2 and 4; §6.6.1, §6.7; FW-REQ-002)
# ----------------------------------------------------------------------
#
# Both helpers below are pure projections onto span attributes. Nothing in
# fastWorkflow reads what they return: that is EXP-003's no-control-flow-read
# exit criterion and architecture §17.3's stop condition, and it is asserted
# both structurally and behaviorally by tests/test_no_capture_control_flow.py.

# Names the thing that produced a handle, so a handle projected by a real
# workflow projector later is distinguishable from one projected by this
# fallback. §6.7's projector registry does not exist yet: `ContextDeclaration.
# handle_projector` is where a workflow names one, and nothing resolves it.
CONTEXT_PROJECTOR_ID = "fastworkflow.context_type"
CONTEXT_PROJECTOR_VERSION = "1"

# §6.7 has a host-injected `SecurityContext` create the handle. fastWorkflow has
# no such thing, so claiming a tenant or principal here would be inventing a
# scope no code enforces. FW-NFR-010 tenant scoping is the outstanding delta
# arch §12.4 already names.
UNSCOPED_SECURITY_SCOPE = "unscoped"


def context_handle(workflow: Any) -> Optional[dict[str, Any]]:
    """Project a workflow's active command context into a §6.7 handle.

    Returns the handle as a plain JSON-able dict, or None when there is no
    workflow to read (never raises: a capture failure must not fail a turn).

    **The handle is type-only, deliberately.** `project_context_handle` needs an
    `instance_key` to produce a concrete, HMAC-fingerprinted handle, and
    fastWorkflow has no framework-level notion of a context instance's identity.
    The current context is an arbitrary application object
    (`Workflow.current_command_context`) whose only framework-visible identity is
    its class name; `current_command_context_displayname` is a display string
    that a workflow may or may not derive from the instance, and on the test
    workflows it is just the class name again. `id()` is not an identity either —
    it is a memory address, reused after collection and meaningless across
    processes, so two records could agree on it while describing different
    objects.
    §6.7 provides for exactly this: `instance_key=None` yields a handle with
    `concrete=False`, which is documented feature-off legacy behavior and cannot
    contribute to G2A/G2B. Inventing an instance key would be worse than the
    honest degradation, because it would look concrete.

    `display_label` stays None: §6.7 admits it only under an explicit allowlist,
    and no allowlist mechanism exists.
    """
    try:
        if workflow is None:
            return None
        context_type = getattr(workflow, "current_command_context_name", None)
        if not context_type:
            return None
        handle = capture_policy.project_context_handle(
            context_type=context_type,
            instance_key=None,
            security_scope_ref=UNSCOPED_SECURITY_SCOPE,
            projector_id=CONTEXT_PROJECTOR_ID,
            projector_version=CONTEXT_PROJECTOR_VERSION,
        )
        return handle.model_dump(mode="json")
    except Exception as exc:
        logger.warning(f"context_handle projection failed: {exc!r}")
        return None


def consequence_assessment(
    workflow_folderpath: Optional[str], command_name: Optional[str]
) -> Optional[dict[str, Any]]:
    """Grade one executed command per §6.6.1, as a plain JSON-able dict.

    The effect contract comes from the workflow's runtime manifest, retained at
    startup by `runtime_manifest.register_runtime_metadata`. Nothing registered
    for this workflow, or nothing declared for this command, yields
    `effect_kind="unknown"` — never `read_only` (§7.3) — and `assess_consequence`
    floors unknown at high, which is §6.6.1's rule that an absent contract is a
    reason for more caution rather than less.

    Reversibility and blast radius have no declaration to read anywhere in the
    manifest schema today, so they stay `unknown` and carry their own floors.
    Recording a guess is the one failure mode that would produce a clean-looking
    row and a wrong one.
    """
    try:
        metadata = (
            runtime_manifest.get_runtime_metadata(workflow_folderpath)
            if workflow_folderpath
            else None
        )
        effect_kind = (
            metadata.effect_kind(command_name)
            if metadata is not None and command_name
            else "unknown"
        )
        return decision_signals.assess_consequence(
            effect_kind=effect_kind
        ).model_dump(mode="json")
    except Exception as exc:
        logger.warning(f"consequence assessment failed: {exc!r}")
        return None


def active_workflow(host: Any) -> Any:
    """The workflow whose command context a dispatch on *host* acts on, or None.

    Duck-typed and never raising, like the rest of this seam: it is called only to
    build capture attributes, and a host that cannot answer must degrade to an
    absent handle rather than fail the command. The fallback to ``app_workflow``
    matters for a bare WorkflowExecutionContext, whose ``get_active_workflow``
    reads a ContextVar stack that is empty outside a dispatch.
    """
    try:
        getter = getattr(host, "get_active_workflow", None)
        workflow = getter() if callable(getter) else None
    except Exception:
        workflow = None
    return workflow if workflow is not None else _resolve(host, "app_workflow")


def context_before(span: Optional[Span], workflow: Any) -> Optional[dict[str, Any]]:
    """The active context handle before a command runs, or None.

    Gated on a span having actually opened, matching the attribute-prep rule at
    every emission site: with observability off this must cost nothing.
    """
    if span is None:
        return None
    return context_handle(workflow)


def capture_attributes(
    span: Optional[Span],
    command_output: Any,
    handle_before: Optional[dict[str, Any]],
    workflow: Any,
    *,
    command_name: Optional[str] = None,
) -> dict[str, Any]:
    """Call-id, context-before/after and consequence for one executed command.

    §12.1.1 requires the dispatch paths to capture the same things, so the
    projection lives here once. All three ``fw.agent.tool_call`` emitters call it
    (``workflow_agent`` and both ``WorkflowExecutionContext`` paths), which is
    what makes a single contract version for that span name true rather than
    merely declared.

    ``command_output`` and ``workflow`` are read duck-typed, never imported: this
    module is on the import path of the core runtime and may not reach the models.

    The call id is read off the CommandOutput rather than minted here — the
    dispatcher that ran the command already stamped it, and a second id would join
    nothing. ``command_name`` overrides what the CommandOutput reports for the
    direct-action path, where it reports nothing.
    """
    if span is None:
        return {}
    return {
        ATTR_COMMAND_CALL_ID: getattr(command_output, "command_call_id", None),
        ATTR_CONTEXT_BEFORE: handle_before,
        ATTR_CONTEXT_AFTER: context_handle(workflow),
        ATTR_CONSEQUENCE: consequence_assessment(
            getattr(workflow, "folderpath", None),
            command_name or getattr(command_output, "command_name", None) or None,
        ),
    }


def emit_turn_record(host: Any, record: Any) -> Optional[bool]:
    """Hand the finalized internal TurnResult to the sink. Never raises.

    Returns the sink's "stored" ack (Phase 7 ruling I1), or None when there is
    no sink at all. The three values are distinct on purpose: True and False
    both mean a durable store exists and say whether the row reached it, while
    None means nothing is recording turns, so a caller gating on durability has
    nothing to wait for and must not defer forever.
    """
    try:
        sink = get_sink(host)
        if sink is None:
            return None
        return bool(sink.emit_turn_record(record))
    except Exception as exc:
        logger.warning(f"TraceSink.emit_turn_record failed: {exc!r}")
        return False
