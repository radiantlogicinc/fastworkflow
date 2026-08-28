"""OTel-shaped trace emission at the agent↔workflow boundary.

Implements the flight-recorder half of the observability design
(docs/fastworkflow_observability_studio_design.md §3.1): a ``TraceSink``
protocol with a no-op default, the v1 span taxonomy, and safe emission
helpers that NEVER raise to the caller — a broken sink degrades to a log
line, not a failed turn.

Spans are OTel-*aligned* records, not wire-conformant OTel (decision D4):
``trace_id`` is the logical turn_key, span ids are opaque strings, and the
translation to real OTel ids is an external script's contract ([R26]).

Sink discovery is duck-typed off the host object (WorkflowExecutionContext,
or ChatSession delegating to its core) via ``trace_sink`` /
``current_turn_key`` / ``trace_span_stack`` — deliberately NOT the
transport-queue contract, so queue-less embedders still trace ([R28]).

This module is stdlib-only by design: it is imported by core runtime
modules and must never pull torch/dspy/transformers.
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
# Span-name taxonomy
# ----------------------------------------------------------------------

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

_DEFAULT_MAX_ATTR_BYTES = 16384


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
    than the one that opened them (fw.turn, fw.ask_user) — [R6]."""
    digest = hashlib.sha256(f"{turn_key}|{span_name}|{attempt}".encode()).hexdigest()
    return digest[:32]


def root_span_id(turn_key: str) -> str:
    """The fw.turn root span id for a logical turn."""
    return deterministic_span_id(turn_key, SPAN_TURN, 0)


def _max_attr_bytes() -> int:
    try:
        return int(os.environ.get("FW_OBS_MAX_ATTR_BYTES", "") or _DEFAULT_MAX_ATTR_BYTES)
    except ValueError:
        return _DEFAULT_MAX_ATTR_BYTES


def cap_attr_value(value: Any) -> Any:
    """Cap one attribute value at FW_OBS_MAX_ATTR_BYTES.

    Truncation is lossy-and-counted ([R10]): an over-limit string becomes an
    envelope carrying ``truncated: True``, the original byte length, and the
    sha256 of the original — never a silent prefix.
    """
    if not isinstance(value, str):
        return value
    limit = _max_attr_bytes()
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
    return {key: cap_attr_value(value) for key, value in attributes.items()}


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


def _get_stack(host: Any) -> Optional[list]:
    return _resolve(host, "trace_span_stack")


# ----------------------------------------------------------------------
# Emission helpers — never raise to the caller
# ----------------------------------------------------------------------


def _emit(sink: TraceSink, span: Span) -> None:
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
    event must be visible before — and closable after — a suspension ([R6]).
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
