"""
Transport-free, synchronous workflow execution core.

Embedders (e.g. FastAPI) should use one WorkflowExecutionContext per session:
bind_app_workflow once, call process_turn per request in a worker thread or
asyncio task (ContextVar isolates active workflow per thread/task), and close()
on session end.

Topology B (no user_message_queue): ask_user is non-blocking — it suspends the
ReAct trajectory in memory and the turn returns an awaiting_user
CommandOutput; the next process_turn(answer) resumes it. A suspended turn
never hangs, so there is no timeout; embedders abandon an unanswered
clarification with cancel_pending() per their own session lifecycle.

ChatSession composes this core for CLI/REPL (queues, ChatWorker, keep_alive).
"""


from __future__ import annotations

import contextlib
import json
import os
import time
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from queue import Queue
from typing import Any, Optional

import dspy

import fastworkflow
import fastworkflow.turn
from fastworkflow import active_workflow, metrics, tracing
from fastworkflow.session_state_store import SCHEMA_VERSION, IncompatibleSessionState
from fastworkflow.state_serialization import validate_state
from fastworkflow.observability import store as observability_store
from fastworkflow.observability.execution_recorder import ExecutionRecorder, record_execution
from fastworkflow.turn import TurnResult, TurnStatus, mint_turn_key
from fastworkflow.turn_plan import TurnPlan
from fastworkflow.utils.logging import logger
from fastworkflow.utils import dspy_logger, dspy_utils
from fastworkflow.utils.react import NoSuspendedAgentStateError

#: Why the turn has the plan it has. The finish check records the cause of a
#: missing plan as ``no_plan_cause``.
TURN_PLAN_PLANNED = "planned"
#: The planner returned no plan at all.
TURN_PLAN_PLANNER_EMPTY = "planner_empty"
#: The planner returned a plan the check could not read steps out of.
TURN_PLAN_UNREADABLE = "plan_unreadable"
#: The turn was resumed from a session state written without its plan.
TURN_PLAN_LOST_ON_RESUME = "lost_on_resume"
#: No plan was made for the check: it was not active, or the turn was not planned.
TURN_PLAN_NOT_PLANNED = "not_planned"
TURN_PLAN_STATUSES = frozenset({
    TURN_PLAN_PLANNED, TURN_PLAN_PLANNER_EMPTY, TURN_PLAN_UNREADABLE,
    TURN_PLAN_LOST_ON_RESUME, TURN_PLAN_NOT_PLANNED,
})


def _agent_result_attributes(result: Any, attempts: int) -> dict[str, Any]:
    """Close-out attributes for fw.agent.execute — what the executor returned.

    Read defensively: a suspended run returns a Prediction with no
    ``final_answer``, and distillation passes their own result shapes through
    the same choke point.
    """
    return {
        "attempts": attempts,
        "final_answer": getattr(result, "final_answer", None),
        "suspended": bool(getattr(result, "suspended", False)),
        "clarification": getattr(result, "clarification", None),
        "exhausted": bool(getattr(result, "exhausted", False)),
    }


class CommandCancelledError(BaseException):
    """
    Raised when a command cannot continue (e.g. the nested intent-clarification
    ask_user is reached with no user_message_queue).

    Subclasses BaseException so fastWorkflowReAct's ``except Exception`` does not
    swallow it; _execute_message converts it to a failed CommandOutput.
    """


@dataclass(frozen=True)
class _RestoredAgentResult:
    """Stand-in for a restored agent result.

    The finalize path reads exactly one attribute off _turn_agent_result
    (``exhausted``) plus whether it is None at all, so a restore carries those
    two facts rather than a dspy Prediction it could not encode anyway.
    """

    exhausted: bool = False


def _parse_isoformat(value: Optional[str]) -> Optional[datetime]:
    """Parse a serialized timestamp, tolerating a missing or malformed one.

    A timestamp only feeds duration reporting, so a bad one degrades a metric.
    Raising here would fail a restore that is otherwise complete.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        logger.warning(f"Unparseable timestamp in restored turn state: {value!r}")
        return None


class WorkflowExecutionContext:
    """
    Owns NLU (cme_workflow), the bound app workflow, and message execution.

    No queues, threads, or session lifecycle — inject optional queues from
    ChatSession for trace/output/ask_user when running in CLI mode.
    """

    def __init__(
        self,
        run_as_agent: bool = False,
        session_key: Optional[str] = None,
        mirror_action_log_to_file: bool = False,
        generate_insights: bool = False,
        trace_sink: Optional[tracing.TraceSink] = None,
    ):
        """
        Args:
            session_key: Stable id (e.g. channel_id) for cme/app workflow persistence.
                         When omitted, cme uses an ephemeral uuid (CLI one-off sessions).
            mirror_action_log_to_file: DEPRECATED no-op. The cwd
                         action.jsonl debug mirror was retired; use the in-process
                         ``action_log`` property (live) or the observability DB
                         (post-mortem) instead. Kept one release for external
                         callers, then removed.
            generate_insights: If True, enable teacher/student distillation on each
                         agent turn (Topology A / CLI only).
            trace_sink: Observability sink for boundary spans and turn records
                         (observability design §3.1), reached via this context,
                         never the transport queues. When omitted, the context
                         opens the app workflow's own sink when a workflow is
                         bound (``bind_app_workflow``). A sink passed here --
                         including an explicit ``tracing.NoOpTraceSink()``,
                         which records no spans or turn records -- is always
                         kept. Observation offloading still writes its
                         evidence, subjects and events to the workflow's
                         observability database whatever the sink.
        """
        self._session_key = session_key
        self._run_as_agent = run_as_agent
        self._app_workflow: Optional[fastworkflow.Workflow] = None
        self._keep_alive = False
        # Deprecated no-op, retained one release for ctor compatibility [R25].
        self._mirror_action_log_to_file = mirror_action_log_to_file

        self._user_message_queue: Optional[Queue] = None
        self._command_output_queue: Optional[Queue] = None
        self._command_trace_queue: Optional[Queue] = None

        self._conversation_history: dspy.History = dspy.History(messages=[])
        self._action_log: list[dict[str, Any]] = []

        from fastworkflow.command_executor import CommandExecutor
        self._CommandExecutor = CommandExecutor

        self._workflow_tool_agent = None
        self._intent_clarification_agent = None
        self._context_change_listener = None

        self._awaiting_user = False
        self._suspended_user_message: Optional[str] = None
        self._pending_clarification_request: Optional[str] = None

        # Insights-distillation (teacher/student) state — CLI/Topology-A only.
        self._generate_insights = generate_insights
        self._distillation_insights_count = 0
        self._planning_insights: Optional[str] = None
        self._execution_insights: Optional[str] = None

        # The turn's initial plan, which the finish-time execution check
        # verifies, and why there is none when there is none: one of
        # TURN_PLAN_STATUSES. Written by the planner at the start of a turn and
        # carried across a cross-process ask_user resume by serialize_state.
        self._turn_plan: Optional[TurnPlan] = None
        self._turn_plan_status: str = TURN_PLAN_NOT_PLANNED

        # Observability (design §3.1): sink + identity + span bookkeeping.
        # The sink is a per-context attribute, not transport state [R28].
        self._trace_sink: tracing.TraceSink = trace_sink or tracing.NoOpTraceSink()
        #: Whether the caller chose the sink (here or via ``set_trace_sink``).
        #: A chosen sink is never replaced; only an automatic one follows the
        #: bound app workflow. Tracked explicitly because a caller may choose
        #: the no-op sink, which must stay chosen.
        self._trace_sink_supplied: bool = trace_sink is not None
        #: The workflow folder the automatic sink records into, when there is one.
        self._auto_sink_folder: Optional[str] = None
        self._metrics_sink: metrics.MetricsSink = metrics.NoOpMetricsSink()
        self._channel_id: Optional[str] = None
        self._conversation_id: Optional[int] = None
        self._embedder_owns_conversations: bool = False
        # The experiment container's labels, bound beside channel/conversation
        # identity and stamped onto every TurnResult this context produces
        # (`fix-bn1`, `[XR17]`). None on every ordinary turn.
        self._experiment_id: Optional[str] = None
        self._task_id: Optional[str] = None
        self._attempt: Optional[int] = None
        self._claim_epoch: Optional[int] = None
        self._server_incarnation: Optional[str] = None
        self._experiment_claim_store: Any = None
        self._trace_span_stack: list[tracing.Span] = []
        self._turn_root_span: Optional[tracing.Span] = None

        # Turn accumulator state (one logical turn = one key, across suspensions)
        self._turn_outputs: list = []
        # Bound here and not only in _begin_turn, because _build_turn_result is
        # reachable on a context that never began a turn in THIS process: resume
        # continues the same logical turn and deliberately skips _begin_turn
        # (see _serialize_turn_accumulator). The read at finalize guards on
        # `is not None`, so an attribute that does not exist made the guard
        # itself raise AttributeError — a missing binding wearing the costume of
        # a null check. fix-ajv.20.
        self._execution_recorder: Optional[ExecutionRecorder] = None
        self._turn_key: Optional[str] = None
        self._turn_started_at: Optional[datetime] = None
        self._turn_user_message: str = ""
        self._turn_refined_message: Optional[str] = None
        self._turn_suspended_ms: int = 0
        self._suspend_began_at: Optional[datetime] = None
        self._turn_entry_workflow_name: str = ""
        self._turn_entry_context: str = ""
        self._turn_agent_result: Any = None
        self._turn_history_baseline: int = 0
        # turn_key of the newest turn that both completed and contributed a
        # conversation-history entry — the row feedback attaches to (ruling
        # I3). Serialized with session state so a cross-process resume keys
        # feedback off a real turn instead of inferring one from SQL.
        self._last_completed_turn_key: Optional[str] = None
        # Ack from the last turn-record emission: True stored, False queued and
        # not yet durable, None no sink at all. Read by embedders that trim
        # conversation history (ruling I1/I2).
        self._last_turn_record_stored: Optional[bool] = None
        # Whether the last finalize contributed a conversation-history entry —
        # i.e. whether it grew the durable memory the label schedule counts.
        self._last_turn_added_memory: bool = False

        cme_id = (
            f"cme_{session_key}"
            if session_key
            else f"cme_{uuid.uuid4().hex}"
        )
        self._cme_workflow = fastworkflow.Workflow.create(
            fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
            workflow_id_str=cme_id,
            workflow_context={
                "NLU_Pipeline_Stage": fastworkflow.NLUPipelineStage.INTENT_DETECTION,
            },
        )

        self.clear_conversation_history()

    @property
    def session_key(self) -> Optional[str]:
        return self._session_key

    # ------------------------------------------------------------------
    # Observability: sink wiring + identity plumbing (design §3.1 [R1][R28])
    # ------------------------------------------------------------------

    @property
    def trace_sink(self) -> tracing.TraceSink:
        return self._trace_sink

    def set_trace_sink(self, sink: Optional[tracing.TraceSink]) -> None:
        """Wire an observability sink chosen by the caller; it is never replaced.

        ``None`` hands the choice back to the context: the sink becomes the
        automatic one again, opened for the bound app workflow now or at the
        next ``bind_app_workflow``. To record no spans or turn records, pass
        ``tracing.NoOpTraceSink()``; observation offloading still writes its
        evidence to the workflow's observability database.
        """
        if sink is None:
            self._trace_sink = tracing.NoOpTraceSink()
            self._trace_sink_supplied = False
            self._auto_sink_folder = None
            if self._app_workflow is not None:
                self._open_auto_sink(self._app_workflow)
            return
        self._trace_sink = sink
        self._trace_sink_supplied = True
        self._auto_sink_folder = None

    def _open_auto_sink(self, workflow: fastworkflow.Workflow) -> None:
        """Open the app workflow's observability sink for a context given none.

        Recording is always on, so a context an embedder builds without a sink
        records into the bound workflow's own DB, owner-only and pruned when
        the sink opens, exactly as fastWorkflow's entry points do. Rebinding to
        another workflow moves the automatic sink to that workflow's DB. The
        internal command-metadata workflow is never a recording target.
        ``get_observability_sink`` never raises: a store that cannot be opened
        leaves the context on the no-op sink.
        """
        folder = str(getattr(workflow, "folderpath", "") or "")
        if not folder:
            return
        resolved = os.path.realpath(folder)
        internal_root = os.path.dirname(os.path.realpath(
            fastworkflow.get_internal_workflow_path("command_metadata_extraction")
        ))
        if resolved == internal_root or resolved.startswith(internal_root + os.sep):
            return
        if self._auto_sink_folder == resolved and tracing.get_sink(self) is not None:
            return
        sink = observability_store.get_observability_sink(folder)
        self._trace_sink = sink or tracing.NoOpTraceSink()
        self._auto_sink_folder = resolved if sink is not None else None

    @property
    def metrics_sink(self) -> metrics.MetricsSink:
        return self._metrics_sink

    def set_metrics_sink(self, sink: Optional[metrics.MetricsSink]) -> None:
        """Wire a metrics sink (None restores the no-op default)."""
        self._metrics_sink = sink or metrics.NoOpMetricsSink()

    def bind_observability_identity(
        self,
        channel_id: Optional[str] = None,
        conversation_id: Optional[int] = None,
        embedder_owns_conversations: Optional[bool] = None,
        experiment_id: Optional[str] = None,
        task_id: Optional[str] = None,
        attempt: Optional[int] = None,
    ) -> None:
        """Bind channel/conversation identity BEFORE the turn.

        The embedder owns identity: FastAPI binds its channel_id, the CLI a
        synthetic ``cli:<session-start>`` channel. Stamped onto every
        span and TurnResult this context produces. A None argument leaves
        the corresponding binding unchanged (conversation ids rotate without
        re-binding the channel).

        ``embedder_owns_conversations=True`` (additive) disables the WEC's
        own conversation self-minting for this context. FastAPI passes it:
        its minting chokepoint carries the legacy-store floor and syncs it
        back, so a WEC self-mint on its degraded path would mint
        a floor-less id that can alias a legacy conversation and split the
        session across two ids once the chokepoint's own mint succeeds.
        """
        # Validate FIRST, mutate second: a rejected experiment triple must not
        # leave the context half-bound with a new channel_id and the old labels.
        if experiment_id is not None or task_id is not None or attempt is not None:
            self._bind_experiment_labels(experiment_id, task_id, attempt)
        if channel_id is not None:
            self._channel_id = channel_id
        if conversation_id is not None:
            self._conversation_id = conversation_id
        if embedder_owns_conversations is not None:
            self._embedder_owns_conversations = embedder_owns_conversations

    def bind_experiment_claim(self, claim: dict[str, Any], store: Any) -> None:
        """Bind only a store-issued claim, never caller-supplied raw labels."""
        store.validate_attempt_claim(claim)
        self._validate_experiment_labels(
            claim.get("experiment_id"), claim.get("task_id"), claim.get("attempt")
        )
        if not claim.get("epoch") or not claim.get("server_incarnation"):
            raise ValueError("an experiment claim needs epoch and server_incarnation")
        self._experiment_id = claim["experiment_id"]
        self._task_id = claim["task_id"]
        self._attempt = int(claim["attempt"])
        self._claim_epoch = int(claim["epoch"])
        self._server_incarnation = claim["server_incarnation"]
        self._experiment_claim_store = store
        self._channel_id = claim["channel_id"]
        self._conversation_id = int(claim["conversation_id"])

    @property
    def observability_experiment_claim(self) -> dict[str, Any]:
        if self._experiment_id is None:
            return {}
        return {
            "experiment_id": self._experiment_id,
            "task_id": self._task_id,
            "attempt": self._attempt,
            "epoch": self._claim_epoch,
            "server_incarnation": self._server_incarnation,
        }

    def assert_experiment_claim_current(self) -> None:
        """Fence turn admission and each registered command dispatch."""
        if self._experiment_id is None:
            return
        # The in-process ExperimentHarness predates registered external
        # channels and remains compatible. A registered binding always carries
        # both fields and can never degrade to this legacy/internal path.
        if self._claim_epoch is None and self._server_incarnation is None:
            return
        if self._claim_epoch is None or self._server_incarnation is None:
            raise RuntimeError("experiment claim fencing metadata is incomplete")
        if self._experiment_claim_store is not None:
            self._experiment_claim_store.validate_attempt_claim(
                self.observability_experiment_claim
            )

    def _bind_experiment_labels(
        self,
        experiment_id: Optional[str],
        task_id: Optional[str],
        attempt: Optional[int],
    ) -> None:
        """Validate and bind the experiment triple.

        All three or none. A turn labelled with an experiment but no task
        belongs to an experiment and to no task: it contributes to a numerator
        and to no denominator, and every GROUP BY in the scoring layer is
        silently wrong. Refusing here is the only cheap place to catch it.

        The `isinstance(attempt, int)` check is load-bearing and not decorative.
        SQLite's INTEGER is a type AFFINITY, not a constraint -- a string bound
        to it that cannot be losslessly converted is stored as TEXT -- so the
        column's declared type protects nothing on its own. This is what makes
        `attempt` safe to leave otherwise unpoliced.
        """
        resolved = self._validate_experiment_labels(
            experiment_id if experiment_id is not None else self._experiment_id,
            task_id if task_id is not None else self._task_id,
            attempt if attempt is not None else self._attempt,
        )
        self._experiment_id, self._task_id, self._attempt = resolved

    @staticmethod
    def _validate_experiment_labels(
        experiment_id: Optional[str],
        task_id: Optional[str],
        attempt: Optional[int],
    ) -> tuple[str, str, int]:
        """Check the triple and return it. Pure: assigns nothing."""
        if not experiment_id or not task_id:
            raise ValueError(
                "an experiment binding needs both experiment_id and task_id; "
                f"got experiment_id={experiment_id!r}, task_id={task_id!r}"
            )
        if isinstance(attempt, bool) or not isinstance(attempt, int):
            raise ValueError(f"attempt must be an int, got {type(attempt).__name__}")
        if attempt <= 0:
            raise ValueError(f"attempt must be positive, got {attempt}")
        return experiment_id, task_id, attempt

    def _ensure_observability_conversation(self) -> None:
        """Mint a conversation id when no embedder bound one.

        FastAPI and the CLI both bind one before the first turn. Code that
        embeds this context directly has no such layer, so its turns would be
        filed outside any conversation and grouped nowhere. Minting here keeps
        identity ownership with the embedder wherever one exists — a bound id
        is never replaced — while giving bare embedders the same grouping.

        Never fails a turn: an unmintable id (wedged or corrupt DB) leaves the
        turn conversation-less, exactly as before.

        Scope guard (ruling C2): an embedder that declared
        ``embedder_owns_conversations`` (FastAPI) mints through its own
        chokepoint, which carries the legacy-store floor and syncs it back;
        self-minting on its degraded path would mint a floor-less id that can
        alias a pre-existing legacy conversation and split the session across
        two ids once the chokepoint's own mint succeeds. For such embedders a
        failed mint leaves the turn conversation-less by design.
        """
        if self._conversation_id is not None or getattr(
            self, "_embedder_owns_conversations", False
        ):
            return
        store = getattr(tracing.get_sink(self), "store", None)
        if store is None:
            return
        try:
            # Mint against the channel the sink files this turn's row under.
            self._conversation_id = store.mint_conversation_id(
                self._channel_id or "",
                experiment_id=self._experiment_id,
                task_id=self._task_id,
                attempt=self._attempt,
            )
        except sqlite3.IntegrityError:
            # NOT swallowed. `idx_conv_experiment_attempt` is UNIQUE precisely so
            # that a second conversation under one (experiment, task, attempt)
            # is refused, and degrading that refusal to a conversation-less turn
            # would defeat the invariant silently: the attempt would keep running
            # and its turns would land outside any conversation, which is exactly
            # the unreconstructable state the index exists to prevent. An
            # ordinary turn cannot reach this arm -- the index is partial on
            # experiment_id IS NOT NULL.
            raise
        except Exception as exc:
            logger.warning(
                f"Could not mint a conversation id ({type(exc).__name__}: {exc}); "
                "this turn is recorded without a conversation"
            )

    @property
    def observability_channel_id(self) -> Optional[str]:
        return self._channel_id

    @property
    def observability_conversation_id(self) -> Optional[int]:
        return self._conversation_id

    @property
    def current_turn_key(self) -> Optional[str]:
        """The open logical turn's key, or None between turns."""
        return self._turn_key

    @property
    def trace_span_stack(self) -> list[tracing.Span]:
        """Open-span stack for parenting nested spans (single-turn: I5)."""
        return self._trace_span_stack

    def clear_action_log(self) -> None:
        """Clear in-memory action log for a new agent turn."""
        self._action_log.clear()

    def append_action_log(self, record: dict[str, Any]) -> None:
        """Append one agent/workflow interaction record (in-memory only)."""
        self._action_log.append(record)

    @property
    def action_log(self) -> list[dict[str, Any]]:
        return self._action_log

    # ------------------------------------------------------------------
    # Turn accumulator (v2.21: capture + TurnResult return type only)
    # ------------------------------------------------------------------

    def _begin_turn(self, user_message: str) -> None:
        """Atomic turn start: reset accumulator, mint key, stamp started_at.

        Never called while awaiting_user — a message during suspension is the
        resume answer and continues the same logical turn.
        """
        self.assert_experiment_claim_current()
        self._ensure_observability_conversation()
        self._turn_outputs = []
        self._turn_key = mint_turn_key()
        self._turn_started_at = datetime.now(timezone.utc)
        self._turn_user_message = user_message
        self._turn_refined_message = None
        self._turn_suspended_ms = 0
        self._suspend_began_at = None
        self._turn_agent_result = None
        self._turn_history_baseline = len(self.conversation_history.messages)

        self._turn_entry_workflow_name = ""
        self._turn_entry_context = ""
        with contextlib.suppress(Exception):
            if self._app_workflow is not None:
                self._turn_entry_workflow_name = (
                    self._app_workflow.folderpath.split("/")[-1]
                )
                self._turn_entry_context = (
                    self._app_workflow.current_command_context_name or ""
                )

        # Context-mutation baseline (D3 as amended): a shallow snapshot of the
        # app workflow's context, diffed at finalize so the root span records
        # what the turn's commands STORED into context — the "storing
        # information in context" feature is otherwise invisible in logs.
        # Sink-gated (zero cost with observability off); not serialized, so a
        # cross-process resume finalizes without a mutation record.
        self._turn_context_snapshot = None
        with contextlib.suppress(Exception):
            if self._app_workflow is not None and tracing.get_sink(self) is not None:
                self._turn_context_snapshot = dict(self._app_workflow.context)

        # Turn-scoped execution ledger (arch §12.1). Sink-gated like the other
        # capture projections: with observability off this must cost nothing.
        self._execution_recorder = (
            ExecutionRecorder() if tracing.get_sink(self) is not None else None
        )

        # Open the fw.turn root span (deterministic id [R6]; emitted at open so
        # a suspended turn is visible before — and closable after — a process
        # boundary). Off the stack: children parent to it via the deterministic
        # root id, which survives suspension where the stack does not.
        self._trace_span_stack.clear()
        self._turn_root_span = tracing.start_span(
            self,
            tracing.SPAN_TURN,
            span_id=tracing.root_span_id(self._turn_key),
            attributes={
                "turn_key": self._turn_key,
                "channel_id": self._channel_id,
                "conversation_id": self._conversation_id,
                "user_message": user_message,
            },
            context=self._turn_entry_context or None,
            use_stack=False,
            emit_open=True,
        )

    def append_turn_output(self, command_output: fastworkflow.CommandOutput) -> None:
        """Append one command execution to the current turn's accumulator."""
        self._turn_outputs.append(command_output)
        fastworkflow.turn.warn_on_unserializable_artifacts(command_output)

    def append_ask_user_entry(self, question: str) -> fastworkflow.CommandOutput:
        """Append an unanswered ask_user exchange entry and return it.

        Role inversion: command_parameters holds the agent's question; the
        response holds the user's answer ("" + success=False while unanswered).

        Also opens the fw.ask_user human-wait span (deterministic id per
        attempt; emitted at open so the wait is visible while the turn
        is suspended). Both topologies funnel through here: Topology A via
        _ask_user_tool, Topology B via _note_agent_suspension.
        """
        # `is_ask_user`, not the name: this count feeds a DETERMINISTIC span id,
        # so a failed command called `ask_user` would not just miscount, it
        # would make two real ask_user spans collide on one id. fix-ajv.17.
        attempt = sum(1 for output in self._turn_outputs if output.is_ask_user)
        entry = fastworkflow.CommandOutput(
            command_name="ask_user",
            ask_user_entry=True,
            command_parameters=question,
            command_response=
                fastworkflow.CommandResponse(response="", success=False),
            started_at=datetime.now(timezone.utc),
        )
        self.append_turn_output(entry)

        if self._turn_key:
            tracing.start_span(
                self,
                tracing.SPAN_ASK_USER,
                kind=tracing.KIND_HUMAN_WAIT,
                span_id=tracing.deterministic_span_id(
                    self._turn_key, tracing.SPAN_ASK_USER, attempt
                ),
                parent_span_id=tracing.root_span_id(self._turn_key),
                command_name="ask_user",
                attributes={"agent_query": question, "attempt": attempt},
                use_stack=False,
                emit_open=True,
            )
        return entry

    def complete_ask_user_entry(self, answer: str) -> None:
        """Fill the last unanswered ask_user entry with the user's answer.

        duration_ms is the user's think time. No-op when there is no
        unanswered ask_user entry.

        Closes the matching fw.ask_user span. The span is rebuilt from the
        entry rather than held in memory, so the close is an idempotent upsert
        that also works when the answer arrives in a different process than
        the question.
        """
        for index in range(len(self._turn_outputs) - 1, -1, -1):
            entry = self._turn_outputs[index]
            # `is_ask_user`, not the bare name: a failed command that happens
            # to be called `ask_user` also matches name+unsuccessful, and this
            # loop would overwrite its error with the user's answer and mark it
            # successful. fix-ajv.17.
            if entry.is_ask_user and entry.command_response.success is False:
                entry.command_response.response = answer
                entry.command_response.success = True
                if entry.started_at is not None:
                    entry.duration_ms = int(
                        (datetime.now(timezone.utc) - entry.started_at).total_seconds()
                        * 1000
                    )
                self._close_ask_user_span(index, entry, answer)
                return

    def _close_ask_user_span(
        self, entry_index: int, entry: fastworkflow.CommandOutput, answer: str
    ) -> None:
        """Emit the closed fw.ask_user span for a just-answered entry."""
        if not self._turn_key or tracing.get_sink(self) is None:
            return
        attempt = sum(
            1
            for output in self._turn_outputs[:entry_index]
            if output.is_ask_user
        )
        span = tracing.Span(
            span_id=tracing.deterministic_span_id(
                self._turn_key, tracing.SPAN_ASK_USER, attempt
            ),
            trace_id=self._turn_key,
            name=tracing.SPAN_ASK_USER,
            kind=tracing.KIND_HUMAN_WAIT,
            parent_span_id=tracing.root_span_id(self._turn_key),
            channel_id=self._channel_id,
            command_name="ask_user",
            start_ns=tracing.datetime_to_ns(entry.started_at) or 0,
            experiment_id=self._experiment_id,
            task_id=self._task_id,
            attempt=self._attempt,
            claim_epoch=self._claim_epoch,
            server_incarnation=self._server_incarnation,
        )
        tracing.end_span(
            self,
            span,
            attributes={
                "agent_query": entry.command_parameters,
                "attempt": attempt,
                "user_response": answer,
                "human_wait_ms": entry.duration_ms,
            },
        )

    def _note_agent_suspension(self, clarification: str) -> None:
        """Bookkeeping when the agent suspends on ask_user (Topology B).

        Appends the unanswered ask_user entry unless the last entry is already
        the same unanswered question (Topology-A's blocking path appends via
        workflow_agent), and stamps the suspension start for suspended_ms.
        """
        last = self._turn_outputs[-1] if self._turn_outputs else None
        already_appended = (
            last is not None
            and last.is_ask_user
            and last.command_response.success is False
            and last.command_parameters == clarification
        )
        if not already_appended:
            self.append_ask_user_entry(clarification)
        self._suspend_began_at = datetime.now(timezone.utc)

    def _note_agent_resume(self) -> None:
        """Fold the elapsed suspension into suspended_ms on resume entry."""
        if self._suspend_began_at is not None:
            self._turn_suspended_ms += int(
                (datetime.now(timezone.utc) - self._suspend_began_at).total_seconds()
                * 1000
            )
            self._suspend_began_at = None

    # ------------------------------------------------------------------
    # Queue injection (CLI driver only)
    # ------------------------------------------------------------------

    def set_transport_queues(
        self,
        user_message_queue: Optional[Queue] = None,
        command_output_queue: Optional[Queue] = None,
        command_trace_queue: Optional[Queue] = None,
        keep_alive: bool = False,
    ) -> None:
        """Wire ChatSession queues and keep_alive flag for REPL transport."""
        self._user_message_queue = user_message_queue
        self._command_output_queue = command_output_queue
        self._command_trace_queue = command_trace_queue
        self._keep_alive = keep_alive

    @property
    def user_message_queue(self) -> Optional[Queue]:
        return self._user_message_queue

    @property
    def command_output_queue(self) -> Optional[Queue]:
        return self._command_output_queue

    @property
    def command_trace_queue(self) -> Optional[Queue]:
        return self._command_trace_queue

    @property
    def keep_alive(self) -> bool:
        return self._keep_alive

    @keep_alive.setter
    def keep_alive(self, value: bool) -> None:
        self._keep_alive = value

    # ------------------------------------------------------------------
    # Core properties
    # ------------------------------------------------------------------

    @property
    def cme_workflow(self) -> fastworkflow.Workflow:
        return self._cme_workflow

    @property
    def run_as_agent(self) -> bool:
        return self._run_as_agent

    @property
    def app_workflow(self) -> Optional[fastworkflow.Workflow]:
        return self._app_workflow

    @property
    def workflow_tool_agent(self):
        return self._workflow_tool_agent

    @property
    def intent_clarification_agent(self):
        return self._intent_clarification_agent

    @property
    def conversation_history(self) -> dspy.History:
        return self._conversation_history

    @property
    def awaiting_user(self) -> bool:
        """True when the agent suspended on ask_user and awaits the next process_turn."""
        return self._awaiting_user

    @property
    def last_completed_turn_key(self) -> Optional[str]:
        """turn_key of the newest completed turn that produced a memory entry.

        Feedback keys off this rather than off "the newest row for the
        conversation": a max-ordinal query attaches feedback to whatever
        happened to be written last, which on a suspended or cancelled turn is
        not the turn the user was looking at (ruling I3/C4).
        """
        return self._last_completed_turn_key

    @property
    def last_turn_added_memory(self) -> bool:
        """Whether the last finalize grew the durable conversation memory.

        The label-refresh schedule counts usable turns, so it needs to know
        what THIS turn contributed; a cancelled turn or an abandoned suspension
        wrote a row but added nothing to summarize (ruling I10).
        """
        return self._last_turn_added_memory

    @property
    def last_turn_record_stored(self) -> Optional[bool]:
        """Whether the last turn record reached durable storage (ruling I1).

        None when no sink is installed — there is no durable record to wait
        for, so a caller gating a history trim on durability must treat it as
        "nothing to defer for" rather than as a failure.
        """
        return self._last_turn_record_stored

    def _serialize_turn_accumulator(self) -> Optional[dict[str, Any]]:
        """Project the logical-turn accumulator, or None when no turn is open.

        Resume continues the same logical turn rather than starting a new one
        (_begin_turn is deliberately skipped), so without this the resumed turn
        takes a fresh key and reports only its post-resume commands.

        _turn_agent_result is distilled to the one fact the finalize path reads
        (`exhausted`) instead of being serialized whole: it is a dspy Prediction
        whose other fields nothing downstream consults, and storing an opaque
        object would fail the strict encoder for no gain.
        """
        if self._turn_key is None:
            return None

        agent_result = None
        if self._turn_agent_result is not None:
            agent_result = {
                "exhausted": bool(
                    getattr(self._turn_agent_result, "exhausted", False)
                )
            }

        return {
            "key": self._turn_key,
            "outputs": [o.model_dump(mode="json") for o in self._turn_outputs],
            "started_at": (
                self._turn_started_at.isoformat() if self._turn_started_at else None
            ),
            "user_message": self._turn_user_message,
            "refined_message": self._turn_refined_message,
            "suspended_ms": self._turn_suspended_ms,
            "suspend_began_at": (
                self._suspend_began_at.isoformat() if self._suspend_began_at else None
            ),
            "entry_workflow_name": self._turn_entry_workflow_name,
            "entry_context": self._turn_entry_context,
            "agent_result": agent_result,
        }

    def _serialize_cme_continuation(self) -> Optional[dict[str, Any]]:
        """Project the in-flight CME command, or None when none is in flight.

        Restoring nlu_stage alone is not enough and is actively unsafe: at
        PARAMETER_EXTRACTION, wildcard.py reads context["command_name"]
        unconditionally, and parameter_extraction.py merges the user's answer
        into stored_parameters. Without these three keys a restored session
        either raises KeyError or silently discards every parameter collected
        so far and re-extracts from the answer text alone.
        """
        if self._cme_workflow is None:
            # close() treats this as reachable, and has_open_command() runs
            # after every turn rather than only suspensions, so a missing CME
            # workflow must read as "nothing in flight" instead of raising in
            # the post-turn persist path.
            return None

        ctx = self._cme_workflow.context
        stored = ctx.get("stored_parameters")

        # command_name is deliberately NOT part of this test. Unlike command and
        # stored_parameters, end_command_processing() leaves it behind, so a
        # session that merely ran a command once would look mid-extraction
        # forever and its state would be written on every completed turn.
        if stored is None and not self._is_extracting_parameters():
            return None

        command_name = ctx.get("command_name")
        command = ctx.get("command")

        stored_dump = None
        if stored is not None:
            # model_construct built this without validation (missing fields hold
            # NOT_FOUND sentinels), so dump without validating on the way out.
            stored_dump = stored.model_dump(mode="json")

        return {
            "command": command,
            "command_name": command_name,
            "stored_parameters": stored_dump,
        }

    def _is_extracting_parameters(self) -> bool:
        """True when the NLU pipeline is parked at parameter extraction.

        The stage survives serialization as a raw value, so compare on value
        rather than on enum identity.
        """
        if self._cme_workflow is None:
            return False
        stage = self._cme_workflow.context.get("NLU_Pipeline_Stage")
        target = fastworkflow.NLUPipelineStage.PARAMETER_EXTRACTION
        return stage == target or getattr(stage, "value", stage) == target.value

    def has_open_command(self) -> bool:
        """True when a CME command is mid-extraction.

        Such a session is not awaiting_user, so nothing else marks it as holding
        state that must survive eviction.
        """
        return self._serialize_cme_continuation() is not None

    def serialize_state(self, *, channel_id: str) -> dict[str, Any]:
        """
        Export durable Topology-B state for cross-process resume.

        Requires session_key and bound app_workflow when persisting.
        """
        react_blob = None
        if self._workflow_tool_agent is not None:
            react_blob = self._workflow_tool_agent.export_suspended()

        nlu_stage = self._cme_workflow.context.get("NLU_Pipeline_Stage")
        if hasattr(nlu_stage, "value"):
            nlu_stage = nlu_stage.value
        elif nlu_stage is not None:
            nlu_stage = str(nlu_stage)

        current_context_name = None
        if self._app_workflow and self._app_workflow.current_command_context is not None:
            current_context_name = self._app_workflow.current_command_context_name

        from fastworkflow.conversation_history_io import extract_turns_from_history

        payload = {
            "schema_version": SCHEMA_VERSION,
            "channel_id": channel_id,
            "session_key": self._session_key,
            "app_workflow_id_str": self._session_key or channel_id,
            "cme_workflow_id_str": (
                f"cme_{self._session_key}" if self._session_key else None
            ),
            "workflow_folderpath": (
                self._app_workflow.folderpath if self._app_workflow else None
            ),
            "awaiting_user": self._awaiting_user,
            "suspended_user_message": self._suspended_user_message,
            "pending_clarification_request": self._pending_clarification_request,
            "react": react_blob,
            "nlu_stage": nlu_stage,
            "turn": self._serialize_turn_accumulator(),
            "cme": self._serialize_cme_continuation(),
            "current_command_context_name": current_context_name,
            "action_log": list(self._action_log),
            "conversation_history_turns": extract_turns_from_history(
                self.conversation_history
            ),
            "last_completed_turn_key": self._last_completed_turn_key,
            # The experiment labels ride the suspension blob for the same
            # reason channel_id does: a turn suspended on ask_user and resumed
            # in another process must land in the same attempt, and the labels
            # are the only thing that says which one. Absent on every state
            # written before fix-bn1, which `apply_serialized_state` reads as
            # "not part of an experiment".
            "experiment_id": self._experiment_id,
            "task_id": self._task_id,
            "attempt": self._attempt,
            "claim_epoch": self._claim_epoch,
            "server_incarnation": self._server_incarnation,
            # Additive keys, read with .get: a blob written before them resumes
            # a suspended turn with no plan, recorded as lost on resume. The
            # plan is null whenever the finish check is off.
            "turn_plan": (
                self._turn_plan.model_dump(mode="json")
                if self._turn_plan is not None else None
            ),
            "turn_plan_status": self._turn_plan_status,
        }
        # No default=str round-trip. This is the first serializer, so coercing
        # here is what made every downstream strictness check vacuous: an
        # unsupported value would already be a string by the time anything
        # looked. Raising instead lets the caller keep the runtime live.
        validate_state(payload)
        return payload

    def apply_serialized_state(self, state: dict[str, Any]) -> None:
        """Restore fields from serialize_state() onto this context.

        Raises IncompatibleSessionState if the blob was written at a schema
        version this build does not read, having applied nothing.
        """
        found = state.get("schema_version", 0)
        if found != SCHEMA_VERSION:
            raise IncompatibleSessionState(found)
        # Validated up here with the version check, not applied halfway down:
        # this method's contract is "raises having applied nothing", and a
        # malformed experiment triple must not be the one exception that leaves
        # a half-restored context behind.
        if state.get("experiment_id") is not None:
            self._validate_experiment_labels(
                state.get("experiment_id"), state.get("task_id"), state.get("attempt")
            )
            if bool(state.get("claim_epoch")) != bool(state.get("server_incarnation")):
                raise IncompatibleSessionState(found)

        self._awaiting_user = bool(state.get("awaiting_user"))
        self._suspended_user_message = state.get("suspended_user_message")
        self._pending_clarification_request = state.get(
            "pending_clarification_request"
        )

        self._action_log = list(state.get("action_log") or [])
        # Absent from blobs written before ruling I3 landed; a missing key just
        # means feedback has no turn to attach to until the next turn completes.
        self._last_completed_turn_key = state.get("last_completed_turn_key")
        # Absent from blobs written before fix-bn1, which reads as "not part of
        # an experiment". Restored through the same validating chokepoint the
        # live binding uses, so a hand-edited blob cannot smuggle a partial
        # triple past `[XR17]`.
        if state.get("experiment_id") is not None:
            self._bind_experiment_labels(
                state.get("experiment_id"),
                state.get("task_id"),
                state.get("attempt"),
            )
            if state.get("claim_epoch") is not None:
                self._claim_epoch = int(state["claim_epoch"])
                self._server_incarnation = state["server_incarnation"]

        if turns := state.get("conversation_history_turns") or []:
            from fastworkflow.conversation_history_io import restore_history_from_turns

            self._conversation_history = restore_history_from_turns(turns)

        nlu_stage = state.get("nlu_stage")
        if nlu_stage is not None:
            try:
                self._cme_workflow.context["NLU_Pipeline_Stage"] = (
                    fastworkflow.NLUPipelineStage(nlu_stage)
                )
            except (ValueError, TypeError):
                self._cme_workflow.context["NLU_Pipeline_Stage"] = nlu_stage

        self._apply_cme_continuation(state.get("cme"))
        self._apply_turn_accumulator(state.get("turn"))

        react_blob = state.get("react")
        if react_blob and self._awaiting_user:
            self._ensure_agent_initialized()
            if self._workflow_tool_agent is not None:
                self._workflow_tool_agent.import_suspended(react_blob)
            # The suspended turn's plan, which only the finish check reads.
            # Advisory, so it never fails a restore: a plan that does not
            # validate is a plan lost on resume, like one in a blob written
            # without the key. Only for an ACTIVE check
            # (``workflow_agent.finish_check_active``): one switched off by
            # FW_EVAL_FINISH_REMINDERS=0 restores nothing.
            agent = self._workflow_tool_agent
            if (getattr(agent, "finish_checker", None) is not None
                    and bool(getattr(agent, "finish_reminders_enabled", True))):
                plan: Optional[TurnPlan] = None
                status = TURN_PLAN_LOST_ON_RESUME
                if "turn_plan" in state:
                    raw_plan = state.get("turn_plan")
                    raw_status = state.get("turn_plan_status")
                    try:
                        plan = TurnPlan.model_validate(raw_plan) if raw_plan is not None else None
                    except ValueError:
                        plan = None
                    else:
                        status = raw_status if raw_status in TURN_PLAN_STATUSES else (
                            TURN_PLAN_PLANNED if plan is not None else TURN_PLAN_NOT_PLANNED)
                    if plan is None and status == TURN_PLAN_PLANNED:
                        status = TURN_PLAN_LOST_ON_RESUME
                self._turn_plan = plan
                self._turn_plan_status = status

        saved_context_name = state.get("current_command_context_name")
        if (
            saved_context_name
            and self._app_workflow
            and self._app_workflow.current_command_context is not None
            and self._app_workflow.current_command_context_name != saved_context_name
        ):
            logger.debug(
                "Command context name after rehydrate (%s) differs from saved (%s); "
                "navigation depth may not match until workflow-specific restore is added",
                self._app_workflow.current_command_context_name,
                saved_context_name,
            )

    def _apply_turn_accumulator(self, turn: Optional[dict[str, Any]]) -> None:
        """Restore the logical turn so resume continues it instead of starting one."""
        if not turn:
            return

        # Memory-stamp baseline (ruling I5). apply_serialized_state restores the
        # conversation history BEFORE this, so the restored length is the right
        # baseline: a resumed turn stamps only an entry it appends after resume.
        # Reconstructing rather than serializing keeps a cross-process resume
        # from stamping the PREVIOUS turn's summary onto this write-once row.
        self._turn_history_baseline = len(self.conversation_history.messages)

        self._turn_key = turn.get("key")
        # The ledger is per-process and not serialized: the pre-suspension
        # process kept its own, and those records went durable with its spans.
        # What this rebuilds is the accumulator for the commands the RESUMED
        # turn is about to run, so their outcomes can still be joined to their
        # execution records. Sink-gated exactly as _begin_turn gates it, so
        # observability-off costs nothing here either. fix-ajv.20.
        self._execution_recorder = (
            ExecutionRecorder() if tracing.get_sink(self) is not None else None
        )
        self._turn_outputs = [
            fastworkflow.CommandOutput.model_validate(o)
            for o in (turn.get("outputs") or [])
        ]
        self._turn_started_at = _parse_isoformat(turn.get("started_at"))
        self._turn_user_message = turn.get("user_message") or ""
        self._turn_refined_message = turn.get("refined_message")
        self._turn_suspended_ms = int(turn.get("suspended_ms") or 0)
        self._suspend_began_at = _parse_isoformat(turn.get("suspend_began_at"))
        self._turn_entry_workflow_name = turn.get("entry_workflow_name") or ""
        self._turn_entry_context = turn.get("entry_context") or ""

        if agent_result := turn.get("agent_result"):
            self._turn_agent_result = _RestoredAgentResult(
                exhausted=bool(agent_result.get("exhausted"))
            )

    def _apply_cme_continuation(self, cme: Optional[dict[str, Any]]) -> None:
        """Restore the in-flight CME command so the next message continues it.

        stored_parameters is rebuilt through model_construct rather than
        model_validate: the saved instance was itself built that way and holds
        NOT_FOUND sentinels in the missing fields, which is precisely the state
        validation exists to reject.
        """
        if not cme:
            return

        context = self._cme_workflow.context
        command_name = cme.get("command_name")

        if cme.get("command") is not None:
            context["command"] = cme["command"]
        if command_name is not None:
            context["command_name"] = command_name

        stored = cme.get("stored_parameters")
        if stored is None or command_name is None:
            return

        params_class = self._command_parameters_class(command_name)
        if params_class is None:
            # Losing the partial parameters is bad, but resuming into a command
            # whose parameter class no longer exists is worse. Reset to intent
            # detection so the next message is routed rather than merged into a
            # command that cannot be completed.
            logger.warning(
                f"Cannot restore stored_parameters for '{command_name}': its "
                f"parameter class is gone. Resetting to intent detection."
            )
            context.pop("command", None)
            context.pop("command_name", None)
            context["NLU_Pipeline_Stage"] = fastworkflow.NLUPipelineStage.INTENT_DETECTION
            return

        context["stored_parameters"] = params_class.model_construct(**stored)

    def _command_parameters_class(self, command_name: str):
        """The Input class for a command name, or None if it cannot be resolved."""
        if self._app_workflow is None:
            return None
        try:
            routing = fastworkflow.RoutingRegistry.get_definition(
                self._app_workflow.folderpath
            )
            return routing.get_command_class(
                command_name, fastworkflow.ModuleType.COMMAND_PARAMETERS_CLASS
            )
        except Exception:
            return None

    def cancel_pending(self) -> bool:
        """
        Abort a pending ask_user clarification (Topology B).

        ask_user is non-blocking in Topology B (the clarification is returned as a
        CommandOutput), so a suspended trajectory never hangs — it simply waits in
        memory. Embedders call this to abandon it per their own session lifecycle
        (e.g. request timeout, user navigated away).

        Returns True if a pending clarification was cleared, False otherwise.
        """
        if not self._awaiting_user:
            return False
        self._reset_agent_suspension()
        self._suspend_began_at = None
        self._turn_suspended_ms = 0
        return True

    def clear_conversation_history(self) -> None:
        self._conversation_history = dspy.History(messages=[])
        # No history means no turn for feedback to attach to. Leaving the key
        # behind would let feedback given after a rotate land on a turn of the
        # conversation that was just archived (ruling I3).
        self._last_completed_turn_key = None

    def bind_last_completed_turn_key(self, turn_key: Optional[str]) -> None:
        """Point feedback at a turn this context did not run.

        Activating a stored conversation replaces the in-memory history, so the
        turn feedback belongs to is that conversation's newest usable turn, not
        whatever this process happened to run last. The embedder reads the key
        from the store (``get_last_completed_turn_key``) and installs it here.
        """
        self._last_completed_turn_key = turn_key

    def append_conversation_turn(
        self,
        conversation_summary: str,
        conversation_traces: Optional[str] = None,
    ) -> None:
        """Append one turn to conversation history in the canonical 2-key shape.

        Two keys, the shape `conversation_history_io` restores. The third was
        `feedback`, and fix-9eg.16 removed the agent-memory table that was its
        only source, so it could only ever be None here -- while
        `_refine_user_query` renders every key of every remembered turn into the
        refiner's prompt, which put a literal `feedback: None` line in front of
        the model on in-session turns and nowhere on restored ones (fix-24da).
        """

        self._conversation_history.messages.append(
            {
                "conversation summary": conversation_summary,
                "conversation_traces": conversation_traces,
            }
        )

    def trim_conversation_history(self, max_turns: int) -> int:
        """Drop all but the newest ``max_turns`` turns; return how many were dropped.

        Turns are request-sized, so an unbounded history grows with every request
        on a hot channel. Only the newest few are ever read (see
        _refine_user_query). Callers that persist history must record a turn
        durably BEFORE trimming it out of memory.
        """
        messages = self._conversation_history.messages
        excess = len(messages) - max_turns
        if excess <= 0:
            return 0
        del messages[:excess]
        return excess

    def summarize_and_record_turn(
        self, message: str, actions: list, result_text: str
    ) -> tuple[str, Optional[str]]:
        """Summarize a completed agent turn and append it to conversation history.

        When there are executed actions, run LLM summarization; otherwise fall back
        to the raw message. Appends the turn via append_conversation_turn and returns
        (summary, traces) so callers can reuse them (e.g. to set an artifact).
        """
        conversation_summary = message
        conversation_traces = None
        if actions:
            conversation_summary, conversation_traces = self._extract_conversation_summary(
                message, actions, result_text
            )
        self.append_conversation_turn(conversation_summary, conversation_traces)
        return conversation_summary, conversation_traces

    def bind_app_workflow(self, workflow: fastworkflow.Workflow) -> None:
        """Bind the app workflow for NLU (Path 1) and execution (Path 2).

        A context whose caller chose no sink opens the workflow's own sink here
        (see ``_open_auto_sink``).
        """
        self._app_workflow = workflow
        self._cme_workflow.context["app_workflow"] = workflow
        # getattr: a context built through ``__new__`` without ``__init__`` has
        # no flag, and is treated as having chosen its sink.
        if not getattr(self, "_trace_sink_supplied", True):
            self._open_auto_sink(workflow)

    def _on_app_context_change(self) -> None:
        """Context-change observer: refresh the ReAct agent's available_commands."""
        from fastworkflow.workflow_agent import _refresh_agent_available_commands
        _refresh_agent_available_commands(self)

    def close(self) -> bool:
        """
        Release the cme_workflow speedict session store.

        Call when an embedder session ends; does not close the app workflow
        (caller owns that lifecycle).
        """
        listener = getattr(self, "_context_change_listener", None)
        if listener is not None and self._app_workflow is not None:
            self._app_workflow.remove_context_change_listener(listener)
            self._context_change_listener = None

        self._reclaim_offloading_scope()

        if self._cme_workflow is None:
            return True
        try:
            return self._cme_workflow.close()
        except ValueError:
            # Child cme workflows should not occur; ignore if mis-invoked.
            logger.debug("WorkflowExecutionContext.close: cme_workflow is not a root session")
            return False

    def _reclaim_offloading_scope(self) -> None:
        """Release this session's offloading state.

        ``close`` is the one production signal that a session is over in this
        process: the fleet's session cache calls it when it retires or removes a
        channel, and an embedder calls it when its session ends. Until this, the
        offloading runtime's per-scope registries -- the archive memo, the
        context clauses, the hot observations, the raw in-flight copies of
        redacted evidence and the turn's diagnostic events -- only ever grew,
        for the lifetime of the process.

        Two turns are NOT reclaimed. A turn suspended on ask_user is still
        resumable: its scope is the one the resume writes and reads handles
        under, and an eviction is precisely the case where it is expected to
        come back. And nothing per-agent is touched at all -- the execute
        numbering ledger, the suspended trajectory and the truncated-execute
        count are the agent's own memory, they die with it, and a resume
        rebuilds them from the suspension payload.

        Evidence is redacted when it is written, and this process keeps the
        raw text of redacted observations only while their turn is live, so
        its own reads stay exact. Releasing the scope is what ends that: after
        it, a read of the turn's evidence returns the stored text. A suspended
        turn keeps its raw copies, by the same test as it keeps the rest.

        Best effort by construction: failing to reclaim memory must never turn
        a session close into an error the caller has to handle.
        """
        # getattr, not plain attribute access: ``close`` is reachable on a
        # context built through ``__new__`` without ``__init__`` -- the
        # context-change listener test constructs one exactly that way -- so
        # neither attribute is guaranteed to exist. A context with no agent has
        # no scope to reclaim, which is the same answer as an agent with no
        # scope. A context that cannot say whether it is awaiting the user is
        # assumed to be mid-turn, because declining to reclaim costs memory
        # while reclaiming early would take a live turn's raw evidence copies.
        agent = getattr(self, "_workflow_tool_agent", None)
        scope = getattr(agent, "continuation_scope", None)
        if scope is None:
            return
        try:
            if getattr(self, "_awaiting_user", True) or agent.export_suspended() is not None:
                return
            runtime = getattr(agent, "turn_runtime", None)
            if runtime is None:
                from fastworkflow.agent_runtime import build_turn_runtime

                runtime = build_turn_runtime(
                    scope, archive=getattr(agent, "observation_archive", None)
                )
                agent.turn_runtime = runtime
            runtime.finish_scope(scope)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "WorkflowExecutionContext.close: could not reclaim "
                f"offloading state ({type(exc).__name__}: {exc})"
            )

    # ------------------------------------------------------------------
    # Active workflow stack (contextvar)
    # ------------------------------------------------------------------

    def get_active_workflow(self) -> Optional[fastworkflow.Workflow]:
        return active_workflow.get_active_workflow()

    def push_active_workflow(self, workflow: fastworkflow.Workflow) -> None:
        active_workflow.push_active_workflow(workflow)

    def pop_active_workflow(self) -> Optional[fastworkflow.Workflow]:
        return active_workflow.pop_active_workflow()

    def clear_workflow_stack(self) -> None:
        active_workflow.clear_workflow_stack()

    # ------------------------------------------------------------------
    # Public execution API
    # ------------------------------------------------------------------

    def process_turn(self, message: str) -> "fastworkflow.TurnOutput":
        """
        Execute one user message synchronously and return the public TurnOutput.

        Shares dispatch with _execute_message(); additionally captures every
        command execution of the logical turn (including ask_user exchanges). The
        full internal TurnResult is built and projected onto the slim public
        TurnOutput (see docs/turn_result_design_final.md section 1a).
        """
        command_output = self._execute_message(message)
        turn_result = self._build_turn_result(command_output)
        return turn_result.turn_output

    @dspy_logger.observe_dspy_calls
    def _execute_message(self, message: str) -> fastworkflow.CommandOutput:
        """Shared message dispatch for _execute_message()/process_turn()."""
        self.assert_experiment_claim_current()
        if self._app_workflow is None:
            raise RuntimeError(
                "No app workflow bound; call bind_app_workflow() before executing a message"
            )

        if not self._awaiting_user:
            # A message during suspension is the resume answer — never a reset.
            self._begin_turn(message)

        self.push_active_workflow(self._app_workflow)
        try:
            self._prepare_message_routing(message)
            if self._should_run_agent_for_message(message):
                if self._awaiting_user:
                    return self._resume_agent_message(message)
                return self._process_agent_message(message)
            return self._process_message(message)
        except CommandCancelledError as exc:
            self._reset_agent_suspension()
            return self._command_cancelled_output(str(exc))
        finally:
            self.pop_active_workflow()
            if self._app_workflow:
                self._app_workflow.flush()

    def _build_turn_result(
        self, command_output: fastworkflow.CommandOutput
    ) -> TurnResult:
        """Assemble the TurnResult (and its public turn_output) for the message.

        The turn's ``answer`` is plain text — the agent's final answer (or the
        deterministic command's response text). Per-command structured results
        (success/artifacts) live on ``command_outputs``.
        """
        answer = command_output.command_response.response if command_output else ""

        failure_reason: Optional[str] = None
        if self._awaiting_user:
            status = TurnStatus.AWAITING_USER
            completed_at: Optional[datetime] = None
        else:
            status = TurnStatus.COMPLETED
            completed_at = datetime.now(timezone.utc)
            if self._turn_agent_result is not None:
                if getattr(self._turn_agent_result, "exhausted", False):
                    # The turn failed to complete (agent ran out of iterations).
                    # status carries the failure; failure_reason elaborates it.
                    # Orthogonal to TurnOutput.success (command success codes).
                    status = TurnStatus.FAILED
                    failure_reason = "max_iters_exhausted"
            elif self._turn_outputs:
                # Deterministic/assistant path: answer text is the last captured
                # output's first response text [A33]. A command-level failure is
                # surfaced by TurnOutput.success (all command_outputs succeeded),
                # not by status/failure_reason.
                answer = self._turn_outputs[-1].command_response.response

        turn_output = fastworkflow.TurnOutput(
            turn_key=self._turn_key or mint_turn_key(),
            status=status,
            failure_reason=failure_reason,
            answer=answer,
            command_outputs=list(self._turn_outputs),
        )

        # An awaiting_user emission leaves the memory columns NULL and the
        # terminal upsert fills them (§2.1). At suspension the newest history
        # entry is not yet the turn's contribution — the resumed half of the
        # exchange has not happened — so stamping here would record a partial
        # exchange as this turn's memory.
        conversation_summary, conversation_traces = (
            (None, None) if self._awaiting_user else self._turn_memory_entry()
        )

        turn_metadata: dict[str, Any] = {}
        if self._app_workflow is not None:
            turn_metadata["workflow_folderpath"] = self._app_workflow.folderpath

        turn_result = TurnResult(
            turn_output=turn_output,
            channel_id=self._channel_id,
            conversation_id=self._conversation_id,
            experiment_id=self._experiment_id,
            task_id=self._task_id,
            attempt=self._attempt,
            claim_epoch=self._claim_epoch,
            server_incarnation=self._server_incarnation,
            user_message=self._turn_user_message,
            refined_user_message=self._turn_refined_message,
            entry_workflow_name=self._turn_entry_workflow_name,
            entry_context=self._turn_entry_context,
            started_at=self._turn_started_at,
            completed_at=completed_at,
            suspended_ms=self._turn_suspended_ms,
            conversation_summary=conversation_summary,
            conversation_traces=conversation_traces,
            metadata=turn_metadata,
            execution_records=(
                self._execution_recorder.records()
                if self._execution_recorder is not None
                else ()
            ),
        )

        self._finalize_turn_trace(turn_result)
        return turn_result

    def _turn_memory_entry(self) -> tuple[Optional[str], Optional[str]]:
        """The conversation-history entry THIS turn appended, or (None, None).

        The turns table carries a row for every logical turn, but only some of
        those turns correspond to a conversation-history entry: a cancelled
        turn, an abandoned suspension, or an agent turn whose history never
        grew has nothing to contribute to memory. Stamping the newest entry
        unconditionally would attribute the previous turn's summary to this
        row, and the row is write-once — so the growth guard is what keeps the
        memory columns honest (§2.1, ruling I5).

        The history is read through the property because distillation replaces
        the object wholesale (and truncates it back to a pre-pass length, which
        is why the comparison is `>` rather than `!=`).
        """
        messages = self.conversation_history.messages
        if len(messages) <= self._turn_history_baseline:
            return None, None
        newest = messages[-1]
        if not isinstance(newest, dict):
            return None, None
        return (
            newest.get("conversation summary"),
            newest.get("conversation_traces"),
        )

    def _compute_context_mutations(self) -> Optional[dict]:
        """Shallow diff of the app workflow's context against the _begin_turn
        snapshot: {added, removed, changed} with repr-capped values, or None
        when nothing changed / no snapshot exists. Never raises — this feeds a
        span attribute and must not affect the turn."""
        snapshot = getattr(self, "_turn_context_snapshot", None)
        if snapshot is None or self._app_workflow is None:
            return None
        try:
            return self._context_mutations_diff(snapshot)
        except Exception:
            # App-authored context can hold anything (uncomparable keys,
            # exploding __eq__) — a diagnostic diff must never fail the turn.
            return None

    def _context_mutations_diff(self, snapshot: dict) -> Optional[dict]:
        current = dict(self._app_workflow.context)

        def brief(value: Any) -> str:
            try:
                return repr(value)[:200]
            except Exception:
                return f"<{type(value).__name__}>"

        mutations: dict = {}
        if added := {
            key: brief(value) for key, value in current.items() if key not in snapshot
        }:
            mutations["added"] = added
        # key=repr: context keys are app-authored and need not be mutually
        # comparable (a str key beside an int key would make plain sorted()
        # raise TypeError).
        if removed := sorted(
            (key for key in snapshot if key not in current), key=repr
        ):
            mutations["removed"] = removed
        changed = {}
        for key, old_value in snapshot.items():
            if key not in current:
                continue
            new_value = current[key]
            try:
                differs = new_value is not old_value and new_value != old_value
            except Exception:
                differs = True  # incomparable values: report, don't hide
            if differs:
                changed[key] = {"from": brief(old_value), "to": brief(new_value)}
        if changed:
            mutations["changed"] = changed
        return mutations or None

    def _finalize_turn_trace(self, turn_result: TurnResult) -> None:
        """Emit the fw.turn root span update/close, the turn record, and turn
        metrics at the finalize chokepoint. Never raises (tracing helpers and
        safe_* wrappers swallow sink failures).

        On AWAITING_USER the root span is updated in place (still open) and
        the record is emitted so the suspended turn is visible; the
        terminal finalize closes the same deterministic span id —
        including after a cross-process resume, where the in-memory span
        object is rebuilt from the restored accumulator.
        """
        turn_output = turn_result.turn_output
        status = turn_output.status

        root = self._turn_root_span
        if root is None and self._turn_key and tracing.get_sink(self) is not None:
            root = tracing.Span(
                span_id=tracing.root_span_id(self._turn_key),
                trace_id=self._turn_key,
                name=tracing.SPAN_TURN,
                channel_id=self._channel_id,
                context=self._turn_entry_context or None,
                start_ns=tracing.datetime_to_ns(self._turn_started_at) or 0,
                attributes={
                    "turn_key": self._turn_key,
                    "channel_id": self._channel_id,
                    "conversation_id": self._conversation_id,
                    "user_message": tracing.cap_attr_value(self._turn_user_message),
                },
                experiment_id=self._experiment_id,
                task_id=self._task_id,
                attempt=self._attempt,
                claim_epoch=self._claim_epoch,
                server_incarnation=self._server_incarnation,
            )
            self._turn_root_span = root

        awaiting = status == TurnStatus.AWAITING_USER
        tracing.end_span(
            self,
            root,
            status=status.value,
            close=not awaiting,
            attributes={
                "status": status.value,
                "success": turn_output.success,
                "failure_reason": turn_output.failure_reason,
                "suspended_ms": turn_result.suspended_ms,
                "context_mutations": self._compute_context_mutations(),
            },
        )
        if not awaiting:
            self._turn_root_span = None
            self._turn_context_snapshot = None

        self._last_turn_record_stored = tracing.emit_turn_record(self, turn_result)
        self._last_turn_added_memory = (
            not awaiting and turn_result.conversation_summary is not None
        )
        if self._last_turn_added_memory:
            # Only a turn that contributed a memory entry can carry feedback:
            # the memory window filters on exactly that, so keying feedback to
            # any other row would file it where no reader joins it (I3/I4).
            self._last_completed_turn_key = turn_result.turn_output.turn_key

        if turn_result.completed_at is not None:
            metrics.safe_increment(
                self._metrics_sink, "fw_turns_total", status=status.value
            )
            if turn_result.started_at is not None:
                metrics.safe_observe(
                    self._metrics_sink,
                    "fw_turn_duration_seconds",
                    (
                        turn_result.completed_at - turn_result.started_at
                    ).total_seconds(),
                    status=status.value,
                )

    def finalize_turn_for_observability(
        self, command_output: Optional[fastworkflow.CommandOutput]
    ) -> None:
        """Run the finalize chokepoint for a turn driven outside process_turn.

        The CLI chassis (ChatSession loop) dispatches via _execute_message /
        process_action directly — its transport is the queues — so nothing
        else builds the TurnResult for its turns. This emits the root-span
        close, turn record, and metrics; the TurnResult itself is discarded.
        No-op when no logical turn is open, or mid-suspension (Topology A
        blocks through ask_user, so a completed _execute_message is a
        completed turn).
        """
        if self._turn_key is None or self._awaiting_user:
            return
        if tracing.get_sink(self) is None and isinstance(
            self._metrics_sink, metrics.NoOpMetricsSink
        ):
            # No sink and no metrics sink (a store that could not be opened,
            # or a host built without either): nothing consumes the
            # TurnResult, so skip building it — this path runs after EVERY
            # turn and must cost ~nothing when there is nowhere to send it.
            return
        self._build_turn_result(command_output)

    def process_action(self, action: fastworkflow.Action) -> fastworkflow.CommandOutput:
        self.assert_experiment_claim_current()
        if self._app_workflow is None:
            raise RuntimeError(
                "No app workflow bound; call bind_app_workflow() before process_action()"
            )

        # Each direct action is its own logical turn [A30].
        self._begin_turn(action.command_name or "")

        self.push_active_workflow(self._app_workflow)
        try:
            return self._process_action(action)
        finally:
            self.pop_active_workflow()
            if self._app_workflow:
                self._app_workflow.flush()

    def process_action_turn(
        self, action: fastworkflow.Action
    ) -> "fastworkflow.TurnOutput":
        """
        Execute one direct action synchronously and return the public TurnOutput.

        Mirror of process_turn() for the direct-action path: same dispatch as
        process_action() (each direct action is its own logical turn),
        additionally building the full internal TurnResult and projecting it onto
        the slim public TurnOutput. This lets callers (e.g. the run_fastapi_mcp
        turn registry) store exactly one result type across both the message and
        action paths.
        """
        command_output = self.process_action(action)
        turn_result = self._build_turn_result(command_output)
        return turn_result.turn_output

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------

    def _prepare_message_routing(self, message: str) -> None:
        if (
            (
                "NLU_Pipeline_Stage" not in self._cme_workflow.context
                or self._cme_workflow.context["NLU_Pipeline_Stage"]
                == fastworkflow.NLUPipelineStage.INTENT_DETECTION
            )
            and message.startswith("/")
        ):
            self._cme_workflow.context["is_assistant_mode_command"] = True

    def _should_run_agent_for_message(self, message: str) -> bool:
        """Agent path unless assistant-mode '/' command flag is set."""
        return (
            self._run_as_agent
            and "is_assistant_mode_command" not in self._cme_workflow.context
        )

    def _command_cancelled_output(self, reason: str) -> fastworkflow.CommandOutput:
        # sourcery skip: class-extract-method
        command_response = fastworkflow.CommandResponse(
            response=f"Command cancelled: {reason}",
            success=False,
        )
        command_output = fastworkflow.CommandOutput(
            command_response=command_response
        )
        if self._app_workflow:
            command_output.workflow_name = self._app_workflow.folderpath.split("/")[-1]
        self._maybe_enqueue_output(command_output)
        self._maybe_enqueue_trace_sentinel()
        return command_output

    def _maybe_enqueue_output(self, command_output: fastworkflow.CommandOutput) -> None:
        if (
            (not command_output.success or self._keep_alive)
            and self._command_output_queue is not None
        ):
            self._command_output_queue.put(command_output)

    def _maybe_enqueue_trace_sentinel(self) -> None:
        if self._command_trace_queue is not None:
            self._command_trace_queue.put(None)

    # ------------------------------------------------------------------
    # Agent mode
    # ------------------------------------------------------------------

    def _initialize_agent_functionality(self) -> None:
        self._cme_workflow.context["run_as_agent"] = True
        if self._app_workflow:
            self._app_workflow.context["run_as_agent"] = True

        # Load workflow-specific insights for insights distillation (if present).
        # These enhance the agent + planner signatures; absent files -> None (no-op).
        from fastworkflow.utils.insights_loader import load_workflow_insights
        if self._app_workflow:
            self._planning_insights = load_workflow_insights(
                self._app_workflow.folderpath, "planning_agent"
            )
            self._execution_insights = load_workflow_insights(
                self._app_workflow.folderpath, "execution_agent"
            )

        from fastworkflow.workflow_agent import initialize_workflow_tool_agent
        self._workflow_tool_agent = initialize_workflow_tool_agent(
            self, execution_insights=self._execution_insights
        )

        # Re-scope the active ReAct agent's available_commands whenever the context changes,
        # driven by the workflow's context-change observer (the single switch chokepoint)
        # Registered once per WEC (agent init runs once). The listener reads the *active* agent
        # dynamically. No-ops when no agent is running. Bound method + remove on close()
        # (and WeakMethod storage on Workflow) avoid listener leaks.
        if self._app_workflow is not None:
            self._app_workflow.add_context_change_listener(self._on_app_context_change)
            self._context_change_listener = self._on_app_context_change

        from fastworkflow.intent_clarification_agent import initialize_intent_clarification_agent
        self._intent_clarification_agent = initialize_intent_clarification_agent(self)

    def _ensure_agent_initialized(self) -> None:
        if self._workflow_tool_agent is None:
            self._initialize_agent_functionality()

    def _reset_agent_suspension(self) -> None:
        """Clear Topology-B ask_user suspend state (abort, finalize, or cancel_pending)."""
        self._awaiting_user = False
        self._suspended_user_message = None
        self._pending_clarification_request = None
        if self._workflow_tool_agent is not None and hasattr(
            self._workflow_tool_agent, "clear_suspension"
        ):
            self._workflow_tool_agent.clear_suspension()

    def _agent_dspy_context(self):
        """Return (lm, adapter) for agent-mode dspy.context blocks."""
        lm = dspy_utils.get_lm("LLM_AGENT", "LITELLM_API_KEY_AGENT")
        from fastworkflow.utils.chat_adapter import CommandsSystemPreludeAdapter

        return lm, CommandsSystemPreludeAdapter()

    def _call_agent_with_retry(self, agent_call, lm=None, *, trace_input=None,
                               resumed=False):
        """Run agent_call under an agent dspy.context, retrying on AdapterParseError.

        lm: optional LM override (e.g. distillation's teacher/student model). When
        omitted, the default agent context (LLM_AGENT) is used.

        This is the one choke point both the fresh forward and the resume pass
        through, so it is where the executor phase is recorded: fw.agent.execute
        wraps the whole loop (retries included), and ``host_scope`` binds this
        context so ReAct's per-iteration fw.agent.step spans — several frames
        down, with no reference to the WEC — reach the same sink.
        """
        from dspy.utils.exceptions import AdapterParseError

        default_lm, agent_adapter = self._agent_dspy_context()
        if lm is None:
            lm = default_lm
        max_retries = 2
        span = tracing.start_span(
            self,
            tracing.SPAN_AGENT_EXECUTE,
            attributes={
                "agent_input": trace_input,
                "resumed": resumed,
                "model": getattr(lm, "model", None),
            },
        )
        attempts = 0
        try:
            with tracing.host_scope(self):
                for attempt in range(max_retries):
                    attempts = attempt + 1
                    try:
                        with dspy.context(lm=lm, adapter=agent_adapter):
                            result = agent_call()
                    except AdapterParseError:
                        if attempt == max_retries - 1:
                            raise
                        # A tool already ran: its side effects and its archived
                        # evidence rows exist under this turn's keys, which a
                        # retry would reuse. A resumed turn always qualifies,
                        # because resume writes the observation key first.
                        mirror = getattr(
                            self._workflow_tool_agent, "current_trajectory", None) or {}
                        if any(str(key).startswith("observation_") for key in mirror):
                            raise
                        continue
                    tracing.end_span(
                        self, span, attributes=_agent_result_attributes(result, attempts)
                    )
                    return result
        except BaseException as exc:
            # CommandCancelledError/AskUserSuspend are control signals, not
            # failures; either way the span must close rather than leak onto
            # the parenting stack for the rest of the turn.
            tracing.end_span(
                self,
                span,
                status=tracing.status_for_dispatch_exception(exc),
                attributes={"attempts": attempts, "error_type": type(exc).__name__},
            )
            raise
        # Every retry raised AdapterParseError but the last re-raise was
        # swallowed by the loop shape: close the span rather than leak it.
        tracing.end_span(self, span, status=tracing.STATUS_ERROR,
                         attributes={"attempts": attempts})

    def _run_agent(self, message: str):
        """Fresh agent turn setup and ReAct forward call."""
        self.clear_action_log()

        if self._app_workflow:
            self._app_workflow.context["raw_user_message"] = message

        refined_user_query = self._refine_user_query(message, self.conversation_history)
        self._turn_refined_message = refined_user_query

        from fastworkflow.workflow_agent import build_query_with_next_steps, _what_can_i_do

        # When there is prior conversation history, pass the agent trajectory and
        # inputs to the planner so it does not re-plan steps already completed in
        # earlier turns (uses TaskPlannerWithTrajectoryAndAgentInputsSignature).
        has_history = bool(self.conversation_history.messages)
        command_info_and_refined_message_with_todolist = build_query_with_next_steps(
            refined_user_query,
            self,
            with_agent_inputs_and_trajectory=has_history,
            planning_insights=self._planning_insights,
            planner_lm=getattr(self, "_current_planner_lm", None),
        )
        available_commands = _what_can_i_do(self)

        return self._call_agent_with_retry(
            lambda: self._workflow_tool_agent(
                user_query=command_info_and_refined_message_with_todolist,
                available_commands=available_commands,
            ),
            trace_input=command_info_and_refined_message_with_todolist,
        )

    def _call_agent_resume(self, observation: str):
        return self._call_agent_with_retry(
            lambda: self._workflow_tool_agent.resume(observation),
            trace_input=observation,
            resumed=True,
        )

    def _awaiting_user_output(self, clarification: str) -> fastworkflow.CommandOutput:
        command_response = fastworkflow.CommandResponse(response=clarification)
        command_response.artifacts["awaiting_user"] = True
        command_output = fastworkflow.CommandOutput(
            command_response=command_response
        )
        if self._app_workflow:
            command_output.workflow_name = self._app_workflow.folderpath.split("/")[-1]
        self._maybe_enqueue_output(command_output)
        self._maybe_enqueue_trace_sentinel()
        return command_output

    def _finalize_agent_output(
        self, original_message: str, agent_result
    ) -> fastworkflow.CommandOutput:
        result_text = (
            agent_result.final_answer
            if hasattr(agent_result, "final_answer")
            else str(agent_result)
        )

        command_response = fastworkflow.CommandResponse(response=result_text)

        conversation_summary, _ = self.summarize_and_record_turn(
            original_message, self._action_log, result_text
        )
        if self._action_log:
            command_response.artifacts["conversation_summary"] = conversation_summary

        # Topic 5: the synthesized agent answer carries only its own artifacts (e.g.
        # conversation_summary), so structured outputs from tool calls during the turn
        # would be dropped on the user-facing path. Merge every artifact-bearing turn
        # response into this single CommandResponse.artifacts dict; on key collision,
        # suffix the incoming key with "_<increment>" (1, 2, ...). The framework does
        # not interpret artifact keys — clients read whatever they need.
        if artifact_responses := fastworkflow.turn.collect_artifact_responses(
            self._turn_outputs
        ):
            fastworkflow.turn.merge_artifact_responses_into(
                command_response, artifact_responses
            )

        command_output = fastworkflow.CommandOutput(
            command_response=command_response
        )
        if self._app_workflow:
            command_output.workflow_name = self._app_workflow.folderpath.split("/")[-1]

        self._maybe_enqueue_output(command_output)
        self._maybe_enqueue_trace_sentinel()

        return command_output

    def _process_agent_message(self, message: str) -> fastworkflow.CommandOutput:
        self._ensure_agent_initialized()

        # Insights-distillation mode (CLI / Topology A only): run the teacher/student
        # comparison, which drives its own agent passes and returns the student's
        # CommandOutput. Guarded on user_message_queue so it can never run over a
        # Topology-B suspended trajectory.
        if self._generate_insights and self.user_message_queue is not None:
            from fastworkflow.distillation import distill_message
            result = distill_message(self, message)
            self._distillation_insights_count += result.insights_extracted
            self._maybe_enqueue_output(result.command_output)
            self._maybe_enqueue_trace_sentinel()
            return result.command_output

        agent_result = self._run_agent(message)
        self._turn_agent_result = agent_result
        if getattr(agent_result, "suspended", None) is True:
            self._awaiting_user = True
            self._suspended_user_message = message
            self._pending_clarification_request = agent_result.clarification
            self._note_agent_suspension(agent_result.clarification)
            return self._awaiting_user_output(agent_result.clarification)
        self._reset_agent_suspension()
        return self._finalize_agent_output(message, agent_result)

    def _resume_agent_message(self, user_answer: str) -> fastworkflow.CommandOutput:
        self._ensure_agent_initialized()
        # Catch awaiting_user/_suspended desync before any LLM work: resume()
        # consumes _suspended at entry, so a second request (or a restore that
        # lost the react blob) must not become a bare 500.
        agent = self._workflow_tool_agent
        if agent is None or agent.export_suspended() is None:
            raise NoSuspendedAgentStateError(
                "No suspended ReAct state to resume"
            )
        self._note_agent_resume()

        from fastworkflow.workflow_agent import _post_ask_user_response

        self._workflow_tool_agent.iteration_counter = -1
        observation = _post_ask_user_response(
            self._pending_clarification_request,
            user_answer,
            self,
        )
        agent_result = self._call_agent_resume(observation)
        self._turn_agent_result = agent_result
        if getattr(agent_result, "suspended", None) is True:
            self._pending_clarification_request = agent_result.clarification
            self._note_agent_suspension(agent_result.clarification)
            return self._awaiting_user_output(agent_result.clarification)

        original_message = self._suspended_user_message
        self._reset_agent_suspension()
        return self._finalize_agent_output(original_message, agent_result)

    # ------------------------------------------------------------------
    # Shared capture for the fw.agent.tool_call emission sites
    # ------------------------------------------------------------------
    #
    # _process_message and _process_action both open fw.agent.tool_call and both
    # owe §12.1.1's shared capture, so the projection lives here once rather than
    # being written twice and drifting. Everything below is additive recording:
    # no fastWorkflow control flow reads a context handle or a consequence class,
    # which arch §17.3 requires.
    #
    # "Here" is in fact `tracing`, because workflow_agent.py
    # opens the same span from a third site and owes the same record. These two
    # methods stay as the WEC-shaped entry points — they supply the app-workflow
    # fallback that the free functions cannot know about — but the projection
    # itself is written once, for all three sites.

    def _context_before(
        self, span, workflow: Optional[fastworkflow.Workflow] = None
    ) -> Optional[dict]:
        """The active context handle before a command runs, or None.

        Gated on a span having actually opened, matching the existing
        attribute-prep rule at this seam: with observability off this must cost
        nothing.
        """
        return tracing.context_before(span, workflow or self._app_workflow)

    def _capture_attributes(
        self,
        span,
        command_output: fastworkflow.CommandOutput,
        context_before: Optional[dict],
        workflow: Optional[fastworkflow.Workflow] = None,
        command_name: Optional[str] = None,
    ) -> dict:
        """Call-id, context-before/after and consequence for one command.

        The call id is read off the CommandOutput rather than minted here: the
        dispatcher that ran the command already stamped it, and minting a second
        one would produce two ids for one execution and join neither.

        ``command_name`` overrides what the CommandOutput reports, because on the
        direct-action path it reports nothing: ``CommandExecutor.perform_action``
        stamps ``workflow_name`` and ``context`` on its result but never
        ``command_name``, so a direct action's outcome carries "" and the
        consequence lookup would find no declaration for any command. The Action
        names what was dispatched, and that is the authoritative identity for
        that path. Passed in rather than fixed on the CommandOutput because
        writing it there would change a public shape, which this slice may not
        do — the empty ``command_name`` on direct-action outcomes is a separate
        defect.
        """
        return tracing.capture_attributes(
            span,
            command_output,
            context_before,
            workflow or self._app_workflow,
            command_name=command_name,
        )

    # ------------------------------------------------------------------
    # Deterministic / assistant mode
    # ------------------------------------------------------------------

    def _process_message(self, message: str) -> fastworkflow.CommandOutput:
        if self._command_trace_queue is not None:
            self._command_trace_queue.put(
                fastworkflow.CommandTraceEvent(
                    direction=fastworkflow.CommandTraceEventDirection.AGENT_TO_WORKFLOW,
                    raw_command=message,
                    command_name=None,
                    parameters=None,
                    response_text=None,
                    success=None,
                    timestamp_ms=int(time.time() * 1000),
                    turn_key=self._turn_key,
                )
            )

        # Span emission sits OUTSIDE the trace-queue guard: the sink is reached
        # via this context, not the transport-queue contract [R28].
        span = tracing.start_span(
            self,
            tracing.SPAN_AGENT_TOOL_CALL,
            kind=tracing.KIND_TOOL,
            attributes={"raw_command": message},
        )
        context_before = self._context_before(span)

        invoke_started_at = datetime.now(timezone.utc)
        try:
            command_output = self._CommandExecutor.invoke_command(self, message)
        except BaseException as exc:
            # One arm, not two: a separate `except CommandCancelledError` left
            # AskUserSuspend — the other control signal — falling through to the
            # BaseException arm below and closing as STATUS_ERROR. fix-ajv.19.
            tracing.end_span(
                self,
                span,
                status=tracing.status_for_dispatch_exception(exc),
                attributes={"error_type": type(exc).__name__},
            )
            raise
        command_output.started_at = invoke_started_at
        command_output.duration_ms = int(
            (datetime.now(timezone.utc) - invoke_started_at).total_seconds() * 1000
        )
        self.append_turn_output(command_output)

        response_text = command_output.command_response.response or ""

        params = command_output.command_parameters or {}
        if hasattr(params, "model_dump"):
            params_dict = params.model_dump()
        elif hasattr(params, "dict"):
            params_dict = params.dict()
        else:
            params_dict = params

        tracing.end_span(
            self,
            span,
            status=(
                tracing.STATUS_OK if command_output.success else tracing.STATUS_ERROR
            ),
            command_name=command_output.command_name or None,
            context=command_output.context or None,
            attributes={
                "response_text": response_text,
                "success": bool(command_output.success),
                **self._capture_attributes(span, command_output, context_before),
            },
        )

        if self._command_trace_queue is not None:
            self._command_trace_queue.put(
                fastworkflow.CommandTraceEvent(
                    direction=fastworkflow.CommandTraceEventDirection.WORKFLOW_TO_AGENT,
                    raw_command=None,
                    command_name=command_output.command_name or "",
                    parameters=params_dict,
                    response_text=response_text,
                    success=bool(command_output.success),
                    timestamp_ms=int(time.time() * 1000),
                    turn_key=self._turn_key,
                )
            )

        record = {
            "command": message,
            "command_name": command_output.command_name or "",
            "parameters": params_dict,
            "response": response_text,
        }

        # "conversation summary" has to identify the turn on its own: it is the
        # only field generate_topic_and_summary and get_conversation_summaries
        # read, and _refine_user_query feeds the last 5 of them to the LLM that
        # refines the next query. The user-visible message leads, matching the
        # agent path, so history reads the same whichever mode produced a turn;
        # on the '/'-prefixed path it is also '/<command_name> <args>', so it
        # names the command and keeps the arguments the resolved command_name
        # would drop. Each part is sliced before joining and newlines are
        # collapsed (the refine prompt is one "key: value" line per field), so
        # the field stays under ~400 chars and never materializes a large
        # command or response. parameters are deliberately absent -- they carry
        # the request payload, which stays in conversation_traces below.
        self.append_conversation_turn(
            " ".join(f"{message[:200]} -> {response_text[:200]}".split()),
            json.dumps(record),
        )

        self._maybe_enqueue_output(command_output)
        self._maybe_enqueue_trace_sentinel()

        return command_output

    def _process_action(self, action: fastworkflow.Action) -> fastworkflow.CommandOutput:
        workflow = self.get_active_workflow() or self._app_workflow

        params = action.parameters or {}
        if hasattr(params, "model_dump"):
            params_dict = params.model_dump()
        elif hasattr(params, "dict"):
            params_dict = params.dict()
        else:
            params_dict = params

        raw_command = f"{action.command_name} {json.dumps(params_dict)}"
        if self._command_trace_queue is not None:
            self._command_trace_queue.put(
                fastworkflow.CommandTraceEvent(
                    direction=fastworkflow.CommandTraceEventDirection.AGENT_TO_WORKFLOW,
                    raw_command=raw_command,
                    command_name=None,
                    parameters=None,
                    response_text=None,
                    success=None,
                    timestamp_ms=int(time.time() * 1000),
                    turn_key=self._turn_key,
                )
            )

        # Outside the trace-queue guard [R28]; see _process_message.
        span = tracing.start_span(
            self,
            tracing.SPAN_AGENT_TOOL_CALL,
            kind=tracing.KIND_TOOL,
            attributes={"raw_command": raw_command},
        )
        # The direct-action path's only span: CommandExecutor.perform_action
        # opens none of its own, so this is where §12.1.1's shared capture has
        # to land for this row of the matrix.
        context_before = self._context_before(span, workflow)

        action_started_at = datetime.now(timezone.utc)
        try:
            command_output = self._CommandExecutor.perform_action(workflow, action)
        except BaseException as exc:
            # One arm, not two: a separate `except CommandCancelledError` left
            # AskUserSuspend — the other control signal — falling through to the
            # BaseException arm below and closing as STATUS_ERROR. fix-ajv.19.
            tracing.end_span(
                self,
                span,
                status=tracing.status_for_dispatch_exception(exc),
                attributes={"error_type": type(exc).__name__},
            )
            raise
        command_output.started_at = action_started_at
        command_output.duration_ms = int(
            (datetime.now(timezone.utc) - action_started_at).total_seconds() * 1000
        )
        self.append_turn_output(command_output)

        response_text = command_output.command_response.response or ""

        tracing.end_span(
            self,
            span,
            status=(
                tracing.STATUS_OK if command_output.success else tracing.STATUS_ERROR
            ),
            command_name=command_output.command_name or None,
            context=command_output.context or None,
            attributes={
                "response_text": response_text,
                "success": bool(command_output.success),
                **self._capture_attributes(
                    span,
                    command_output,
                    context_before,
                    workflow,
                    command_name=action.command_name,
                ),
            },
        )
        record_execution(
            self._execution_recorder,
            command_call_id=command_output.command_call_id,
            parent_call_id=None,
            span_id=span.span_id if span is not None else None,
        )

        if self._command_trace_queue is not None:
            self._command_trace_queue.put(
                fastworkflow.CommandTraceEvent(
                    direction=fastworkflow.CommandTraceEventDirection.WORKFLOW_TO_AGENT,
                    raw_command=None,
                    command_name=command_output.command_name,
                    parameters=params_dict,
                    response_text=response_text,
                    success=bool(command_output.success),
                    timestamp_ms=int(time.time() * 1000),
                    turn_key=self._turn_key,
                )
            )

        record = {
            "command": "process_action",
            "command_name": action.command_name,
            "parameters": params_dict,
            "response": response_text,
        }

        # Same bound and one-line normalization as the deterministic path in
        # _process_message, for the same consumers. A direct action carries no
        # user text, so the command name is the only identity available; it is
        # sliced too, so the bound holds without relying on the caller sending a
        # short name. parameters stay out for the same reason as above.
        self.append_conversation_turn(
            " ".join(
                f"{action.command_name[:200]} -> {response_text[:200]}".split()
            ),
            json.dumps(record),
        )

        self._maybe_enqueue_output(command_output)
        self._maybe_enqueue_trace_sentinel()

        return command_output

    def _refine_user_query(
        self, user_query: str, conversation_history: dspy.History
    ) -> str:
        if not conversation_history.messages:
            return user_query
        messages = []
        for conv_dict in conversation_history.messages[-5:]:
            messages.extend(
                f"{k}: {v}"
                for k, v in conv_dict.items()
                if k != "conversation_traces"
            )
        messages.append(f"new_user_query: {user_query}")
        return "\n".join(messages)

    def _extract_conversation_summary(
        self,
        user_query: str,
        workflow_actions: list[dict[str, str]],
        final_agent_response: str,
    ) -> tuple[str, str]:
        conversation_traces = {
            "user_query": user_query,
            "agent_workflow_interactions": workflow_actions,
            "final_agent_response": final_agent_response,
        }

        class ConversationSummarySignature(dspy.Signature):
            """
            A summary of conversation
            Omit descriptions of action sequences
            Capture relevant facts and parameter values from user query, workflow actions and agent response
            """

            user_query: str = dspy.InputField()
            workflow_actions: list[dict[str, str]] = dspy.InputField()
            final_agent_response: str = dspy.InputField()
            conversation_summary: str = dspy.OutputField(
                desc="A multiline paragraph summary"
            )

        from fastworkflow.conversation_summary import bounded_summary_inputs

        inputs = bounded_summary_inputs(user_query, workflow_actions, final_agent_response)
        try:
            planner_lm = dspy_utils.get_lm("LLM_PLANNER", "LITELLM_API_KEY_PLANNER")
            with dspy.context(lm=planner_lm):
                prediction = dspy.ChainOfThought(ConversationSummarySignature)(**inputs)
            summary = prediction.conversation_summary
        except Exception as exc:
            # Conversation memory is ancillary: failure must not discard the
            # executor's completed answer. Preserve raw evidence in traces.
            logger.warning("Conversation summary failed; using bounded fallback (%s)",
                           type(exc).__name__)
            conversation_traces["summary_fallback_error_type"] = type(exc).__name__
            summary = ("Conversation summary unavailable. User request excerpt: "
                       + inputs["user_query"] + "\nAgent response excerpt: "
                       + inputs["final_agent_response"])
        return summary, json.dumps(conversation_traces)
