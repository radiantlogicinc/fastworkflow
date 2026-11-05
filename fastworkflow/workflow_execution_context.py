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
from typing import Any, Mapping, Optional

import litellm  # noqa: F401  (must precede dspy; see fastworkflow/utils/dspy_utils.py)
import dspy

import fastworkflow
import fastworkflow.turn
from fastworkflow import active_workflow, metrics, planning_commands, tracing
from fastworkflow import approval_binding, turn_cache
from fastworkflow.observation_offloading.state import current_execute_alias
from fastworkflow.session_state_store import SCHEMA_VERSION, IncompatibleSessionState
from fastworkflow.state_serialization import validate_state
from fastworkflow.plan_frame import PlanFrame, runs_outside_ranges, turn_max_steps
from fastworkflow.observability import store as observability_store
from fastworkflow.observability.execution_recorder import ExecutionRecorder, record_execution
from fastworkflow.turn import TurnResult, TurnStatus, mint_turn_key
from fastworkflow.utils.logging import logger
from fastworkflow.utils import dspy_logger, dspy_utils
from fastworkflow.utils.react import AskUserSuspend, NoSuspendedAgentStateError

#: A sub-task that hits the ReAct step cap is continued by a fresh executor at most this many times.
MAX_HANDOFFS = 2

#: The refusal for a planning command called once the turn-wide step budget is spent.
NESTED_BUDGET_SPENT_MESSAGE = (
    "The step budget for nested commands is spent; continue with the remaining steps directly."
)

#: The refusal for a direct action run while a question waits for the user's answer.
AWAITING_ANSWER_REFUSAL_MESSAGE = (
    "A question is waiting for the user's answer; reply to it before running an action."
)


def _planning_args(params: Any) -> dict:
    """The arguments of a planning call as a dict: an ``Input`` instance or a dict."""
    if params is None:
        return {}
    if hasattr(params, "model_dump"):
        return params.model_dump()
    return dict(params)


def _opened_part(trajectory: dict[str, Any]) -> str:
    """The context instances a run's observations entered, one line each; "" when none.

    Read from the observation headers fastworkflow prints, so the ids are exact
    where a summary of the run would drop them.
    """
    from fastworkflow.observation_offloading.labels import printed_instances

    opened: dict[tuple[str, str, str], None] = {}
    for key, observation in trajectory.items():
        if key.startswith("observation_") and isinstance(observation, str):
            opened.update(dict.fromkeys(printed_instances(observation)))
    if not opened:
        return ""
    lines = [f"- {name} {instance_id}" + (f" ({label})" if label else "")
             for name, instance_id, label in opened]
    return "Opened in this sub-task:\n" + "\n".join(lines)


def _join_parts(parts: list[str]) -> str:
    """One sub-task's answers, one per run in hand-off order; a lone answer is unlabelled.

    A run that repeats the previous run's answer word for word adds nothing, so it is dropped.
    """
    parts = [part for i, part in enumerate(parts) if i == 0 or part != parts[i - 1]]
    if len(parts) == 1:
        return parts[0]
    return "\n\n".join(f"Part {i} of {len(parts)}:\n{part}" for i, part in enumerate(parts, 1))


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


def _frame_property(field_name: str) -> property:
    """A property over one field of the top PlanFrame, ``self._frames[-1]``."""

    def fget(self):
        return getattr(self._top_frame(), field_name)

    def fset(self, value):
        setattr(self._top_frame(), field_name, value)

    return property(fget, fset)


#: The turn's answer when the sub-task review is declined or expires: nothing runs (fix-3rsq.plan-review.2.6).
SUBTASK_REVIEW_DECLINED = ("You declined the sub-task review, so nothing was run. "
                           "Send the request again to review it again.")
SUBTASK_REVIEW_EXPIRED = ("The sub-task review expired before you answered, so nothing was run, and your message "
                          "was not taken as a reply to it. Send the request again to start a new review.")

#: Owner of pending state saved before pending state was tagged with its conversation (fix-newconv-pending). Such
#: state is parked, never applied to a conversation automatically.
UNOWNED_PENDING = "unowned"
#: "No owner set": the live pending state belongs to the conversation the context is bound to.
_OWNER_UNSET = object()
#: The serialized keys of a parked pending snapshot: the suspended turn and what it waits on, without the history,
#: the approvals (cancelled, never parked) or the experiment labels (per context, not per conversation).
_PARKED_PENDING_KEYS = (
    "awaiting_user", "suspended_user_message", "pending_clarification_request", "skill_refusal_pending",
    "skill_refusal_answered", "subtasks", "subtask_origins", "subtask_index", "subtask_answers",
    "subtask_outputs", "subtask_exhausted", "subtask_review_open", "subtask_plan", "next_step",
    "skill_phase_run", "react", "frames", "frame_react", "turn_child_step_ranges", "turn_steps_used", "nlu_stage",
    "turn", "cme",
)


def _owner_key(owner) -> str:
    """The key a parked snapshot is filed under: the conversation id, "none" or the unowned marker."""
    return "none" if owner is None else str(owner)


def _owner_encoded(owner):
    """The owner as the serialized blob carries it: a conversation id, None, or the unowned marker."""
    return owner if owner is None or owner == UNOWNED_PENDING else int(owner)


def _owner_from_state(state: dict[str, Any]):
    """The conversation a serialized pending state belongs to. A blob with no conversation_id key is unowned."""
    if "conversation_id" not in state:
        return UNOWNED_PENDING
    value = state["conversation_id"]
    if value is None or value == UNOWNED_PENDING:
        return value
    return int(value)


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
                         evidence and events to the workflow's
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

        # One PlanFrame per planning-command call; the turn frame is at index 0. The per-call
        # planning attributes (_subtasks, _phase_run, ...) are properties over the top frame.
        self._frames: list[PlanFrame] = [PlanFrame.turn()]
        self._context_change_listener = None

        self._awaiting_user = False
        self._suspended_user_message: Optional[str] = None
        self._pending_clarification_request: Optional[str] = None
        # A refused planning command call put to the user (the planner's PlanningCommandInputError, workflow_agent):
        # the question the planner left for _call_agent_for_query, whether the suspension
        # waiting on the reply is one of these, and whether this turn already put one.
        self._planning_command_refusal: Optional[str] = None
        self._planning_command_refusal_pending = False
        self._planning_command_refusal_answered = False
        self._next_step = 0
        # Model steps taken by nested planning-command frames this turn (design 3.5).
        self._turn_steps_used = 0

        # Insights-distillation (teacher/student) state — CLI/Topology-A only.
        self._generate_insights = generate_insights
        self._distillation_insights_count = 0

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
        # fix-newconv-pending: the conversation the live pending state belongs to, when it is not the bound one (set
        # by apply_serialized_state from the saved blob), and the pending state parked per conversation.
        self._pending_owner_override = _OWNER_UNSET
        self._parked_pending: dict[str, list[dict[str, Any]]] = {}
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
        # Unattended turn: nobody is there to answer, so the agent has no
        # ask_user for the whole logical turn. Set when the turn begins.
        self._turn_unattended: bool = False
        # The sub-task review gate's state for the turn (fix-3rsq.plan-review.2.2/.2.6/.2.8): the marker the turn span
        # carries, and the plan the gate built (the accepted one after a review; the draft when no one could answer).
        self._turn_subtask_review: str = "not_applicable"
        self._turn_subtask_plan = None
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

        # Phase 9 step 4: the CME's heads serve every session, so an old CME is refused here, for
        # every runtime path (chat, MCP, FastAPI). Training does not construct this class.
        from fastworkflow.train.artifact_versioning import require_construction_record
        require_construction_record(
            fastworkflow.get_internal_workflow_path("command_metadata_extraction"))

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

    def _frame_stack(self) -> list[PlanFrame]:
        frames = getattr(self, "_frames", None)
        if not frames:
            # A context built with __new__ (no __init__) starts with the turn frame.
            frames = self._frames = [PlanFrame.turn()]
        return frames

    def _top_frame(self) -> PlanFrame:
        return self._frame_stack()[-1]

    def _turn_frame(self) -> PlanFrame:
        return self._frame_stack()[0]

    # Sub-task split of one turn (see _run_agent / _run_subtasks_from).
    _subtasks = _frame_property("subtasks")
    # Original sub-task number of each entry in _subtasks; a hand-off continuation shares its origin.
    _origins = _frame_property("origins")
    _subtask_index = _frame_property("subtask_index")
    _subtask_answers = _frame_property("subtask_answers")
    _subtask_outputs = _frame_property("subtask_outputs")
    # Set when a sub-task hit the step cap with no hand-off left: the turn is then exhausted.
    _subtask_exhausted = _frame_property("subtask_exhausted")
    # Finish check (fix-3xts): the first step each sub-task origin's outputs count from,
    # and the run's initial plan as the planner wrote it.
    _evidence_floors = _frame_property("evidence_floors")
    _initial_plan_text = _frame_property("initial_plan_text")
    # A phased planning command call in progress (planning_phases; see _run_planning_command_phases): JSON
    # state that rides the suspension blob, and the live (command, phases) for it.
    _phase_run = _frame_property("phase_run")
    _phase_live = _frame_property("phase_live")
    _phased_planning_command_call = _frame_property("phased_planning_command_call")
    # The ReAct agent of the call in progress: the top frame's agent (the turn frame's, for now).
    _workflow_tool_agent = _frame_property("agent")

    @property
    def _turn_subtask_plan(self):
        """The turn's sub-task plan (review_proposals.SubTaskProposal): the accepted proposal, or the draft when the
        turn was not reviewed. It is on the turn frame, so a suspended turn keeps it (fix-3rsq.plan-review.2.9)."""
        return self._turn_frame().subtask_plan

    @_turn_subtask_plan.setter
    def _turn_subtask_plan(self, plan) -> None:
        self._turn_frame().subtask_plan = plan

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
        if conversation_id is not None and conversation_id != self._conversation_id:
            # fix-newconv-pending: the live pending state settles before the conversation changes, and the new
            # conversation's parked pending state (if any) comes back.
            self._settle_pending_for_conversation(conversation_id)
            self._conversation_id = conversation_id
            self._restore_parked_pending(conversation_id)
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
    def current_frame_alias(self) -> str:
        """The alias of the planning frame on top, "" for the turn frame (design 3.5).
        Tracing reads it when a span opens, so the span records the frame it ran under."""
        return self._top_frame().alias

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

    def _begin_turn(self, user_message: str, unattended: bool = False) -> None:
        """Atomic turn start: reset accumulator, mint key, stamp started_at.

        Never called while awaiting_user — a message during suspension is the
        resume answer and continues the same logical turn.
        """
        self.assert_experiment_claim_current()
        self._ensure_observability_conversation()
        # Drops nested frames only. The turn frame's own state (its sub-tasks and its agent)
        # is left as it was: _clear_subtask_state clears the sub-task state when a run starts,
        # and a message that resumes a suspended turn still finds its sub-tasks.
        del self._frame_stack()[1:]
        self._turn_steps_used = 0
        self._turn_outputs = []
        self._turn_key = mint_turn_key()
        self._turn_started_at = datetime.now(timezone.utc)
        self._turn_user_message = user_message
        self._turn_unattended = bool(unattended)
        self._turn_subtask_review = "not_applicable"
        self._turn_subtask_plan = None
        turn_cache.begin_turn()
        self._planning_command_refusal_pending = False
        self._planning_command_refusal_answered = False
        self._turn_refined_message = None
        self._turn_suspended_ms = 0
        self._suspend_began_at = None
        self._turn_agent_result = None
        self._turn_history_baseline = len(self.conversation_history.messages)
        approval_binding.on_new_turn(self, self._app_workflow, unattended=self._turn_unattended)

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
                "unattended": self._turn_unattended,
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

    # ------------------------------------------------------------------
    # Pending state and conversations (fix-newconv-pending)
    # ------------------------------------------------------------------
    #
    # Live pending state belongs to the conversation that created it: a paused ask_user question, a planning
    # refusal question, an open review, a CME continuation (parked) and a pending approval (cancelled). When the
    # context moves to another conversation, the live state is settled before anything else runs: approvals are
    # cancelled (they would run side effects on a bare "yes"), the rest is parked under its owner and comes back
    # when that conversation is bound again. stop_asking's tool removal is scoped the same way.

    def _live_pending_owner(self):
        """The conversation the live pending state belongs to (see _OWNER_UNSET)."""
        if self._pending_owner_override is not _OWNER_UNSET:
            return self._pending_owner_override
        return self._conversation_id

    def _has_parkable_pending(self) -> bool:
        """Live pending state that is parked on a conversation change: a question, a refusal, a review or a CME run."""
        if self._awaiting_user or self._planning_command_refusal_pending or self.has_open_command():
            return True
        if getattr(self._turn_frame(), "subtask_review_open", False):
            return True
        return bool(approval_binding.active_reviews(self._app_workflow))

    def _has_live_pending(self) -> bool:
        """True when anything pending is live: a parkable question or review, or a pending approval."""
        return self._has_parkable_pending() or bool(approval_binding.active_approvals(self._app_workflow))

    def has_pending_state(self) -> bool:
        """Public form of _has_live_pending, for the embedder's New conversation."""
        return self._has_live_pending()

    def has_parked_pending(self) -> bool:
        """True when some conversation's pending state is parked (it must be saved with the rest of the state)."""
        return bool(self._parked_pending)

    def _cancel_live_approvals(self, reason: str) -> int:
        """Cancel the pending approvals (an approval runs its side effect on a bare reply, so it never outlives its
        conversation). A suspended approval also abandons the agent's suspended call."""
        if self._app_workflow is None:
            return 0
        awaiting = approval_binding.active_record(self._app_workflow)
        cancelled = approval_binding.cancel_conversation_approvals(self, self._app_workflow, reason=reason)
        if (cancelled and awaiting is not None and awaiting.suspended_agent
                and awaiting.kind == approval_binding.KIND_ACTION and self._awaiting_user):
            self._reset_agent_suspension()
        return cancelled

    def _clear_cme_continuation(self) -> None:
        """Drop the CME command in flight, as end_command_processing does (its command_name is kept there, not here)."""
        context = self._cme_workflow.context
        context.pop("command", None)
        context.pop("command_name", None)
        context.pop("stored_parameters", None)
        context["NLU_Pipeline_Stage"] = fastworkflow.NLUPipelineStage.INTENT_DETECTION

    def _clear_live_pending(self) -> None:
        """Clear the live pending state (the suspended turn, its review, its CME run) without touching history."""
        self._reset_agent_suspension()
        turn = self._turn_frame()
        turn.subtask_review_open = False
        turn.framework_entry = False
        turn.subtask_plan = None
        self._planning_command_refusal_answered = False
        self._suspend_began_at = None
        self._turn_suspended_ms = 0
        if self.has_open_command():
            self._clear_cme_continuation()

    def _pending_snapshot(self, owner) -> dict[str, Any]:
        """The live pending state as a parked snapshot, tagged with its owner."""
        payload = self.serialize_state(channel_id=str(self._channel_id or ""))
        snapshot: dict[str, Any] = {key: payload.get(key) for key in _PARKED_PENDING_KEYS}
        snapshot["conversation_id"] = owner
        snapshot["schema_version"] = SCHEMA_VERSION
        return snapshot

    def _park_and_clear_live_pending(self, owner) -> None:
        """Park the questions and reviews under their owner (never cancelled), then clear them from the live context.

        A snapshot that cannot be encoded is dropped with an error rather than applied to another conversation.
        """
        if self._has_parkable_pending():
            try:
                snapshot = self._pending_snapshot(owner)
            except Exception:  # noqa: BLE001 - losing one parked question must not route it to another conversation
                logger.error("pending state of conversation %s could not be parked; it is dropped", owner,
                             exc_info=True)
                snapshot = None
            if snapshot is not None:
                snapshot["reviews"] = approval_binding.park_reviews(self._app_workflow)
                self._parked_pending.setdefault(_owner_key(owner), []).append(snapshot)
                logger.info("parked pending state of conversation %s", owner)
        self._clear_live_pending()

    def _settle_pending_for_conversation(self, new_conversation_id) -> None:
        """Settle the live pending state for a context that is moving to *new_conversation_id*.

        Nothing happens when the state already belongs to that conversation. Otherwise approvals are cancelled,
        questions and reviews are parked under their owner, and stop_asking's tool removal ends with its conversation.
        """
        owner = self._live_pending_owner()
        # Unowned state is parked whatever the conversation is. Any other state is adopted, not parked, by a context
        # that has no conversation bound yet: a bare embedder's turn mints its id after this check, as it did before
        # pending state was tagged.
        if owner != UNOWNED_PENDING and (owner == new_conversation_id or new_conversation_id is None):
            self._pending_owner_override = _OWNER_UNSET
            return
        self._pending_owner_override = _OWNER_UNSET
        if self._app_workflow is not None:
            self._cancel_live_approvals(reason="conversation changed")
        if self._has_parkable_pending():
            self._park_and_clear_live_pending(owner)
        self._enable_all_tools()

    def park_pending_for_new_conversation(self) -> None:
        """New conversation (the button): settle the live pending state at once, not at the next message.

        Approvals are cancelled and questions and reviews are parked under the current conversation. The conversation
        id itself is changed by the embedder (reserve_conversation_id), which parks the state through bind.
        """
        owner = self._live_pending_owner()
        self._pending_owner_override = _OWNER_UNSET
        if self._app_workflow is not None:
            self._cancel_live_approvals(reason="new conversation")
        self._park_and_clear_live_pending(owner)
        self._enable_all_tools()

    def _restore_parked_pending(self, conversation_id) -> None:
        """Bring back the pending state parked for *conversation_id*, the oldest one first.

        Unowned state is never restored here. A snapshot that cannot be applied stays parked.
        """
        key = _owner_key(conversation_id)
        items = self._parked_pending.get(key)
        if not items:
            return
        snapshot = items.pop(0)
        if not items:
            del self._parked_pending[key]
        if snapshot.get("schema_version") != SCHEMA_VERSION:
            self._parked_pending.setdefault(key, []).insert(0, snapshot)
            logger.error("parked pending state of conversation %s is from another schema; it stays parked", key)
            return
        approval_binding.unpark_reviews(self._app_workflow, snapshot.get("reviews") or [])
        state = {
            **snapshot,
            "approval": None,
            "conversation_history_turns": [],
            "action_log": list(self._action_log),
            "last_completed_turn_key": self._last_completed_turn_key,
        }
        self.apply_serialized_state(state)
        self._pending_owner_override = _OWNER_UNSET
        logger.info("restored parked pending state of conversation %s", key)

    def serialize_state(self, *, channel_id: str) -> dict[str, Any]:
        """
        Export durable Topology-B state for cross-process resume.

        Requires session_key and bound app_workflow when persisting.
        """
        turn = self._turn_frame()
        react_blob = None
        if turn.agent is not None:
            react_blob = turn.agent.export_suspended()

        nlu_stage = self._cme_workflow.context.get("NLU_Pipeline_Stage")
        if hasattr(nlu_stage, "value"):
            nlu_stage = nlu_stage.value
        elif nlu_stage is not None:
            nlu_stage = str(nlu_stage)

        current_context_name = None
        if self._app_workflow and self._app_workflow.current_command_context is not None:
            current_context_name = self._app_workflow.current_command_context_name

        from fastworkflow.conversation_history_io import extract_turns_from_history

        nested = self._frame_stack()[1:]
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
            # Additive (ADR-EXP-025 section 7): the session's approval records and turn clock.
            "approval": self._approval_state_payload(),
            "suspended_user_message": self._suspended_user_message,
            "pending_clarification_request": self._pending_clarification_request,
            "skill_refusal_pending": self._planning_command_refusal_pending,
            "skill_refusal_answered": self._planning_command_refusal_answered,
            "subtasks": list(turn.subtasks),
            "subtask_origins": list(turn.origins),
            "subtask_index": turn.subtask_index,
            "subtask_answers": list(turn.subtask_answers),
            "subtask_outputs": list(turn.subtask_outputs),
            "subtask_exhausted": turn.subtask_exhausted,
            # Additive (fix-3rsq.plan-review.2.7/.2.9): the open sub-task review and the turn's sub-task plan, so an
            # accept after a restart resumes the review and runs the accepted goals.
            "subtask_review_open": bool(turn.subtask_review_open),
            "subtask_plan": None if turn.subtask_plan is None else turn.subtask_plan.model_dump(mode="json"),
            "next_step": self._next_step,
            "skill_phase_run": turn.phase_run,
            "react": react_blob,
            "frames": [frame.to_state() for frame in nested],
            "frame_react": [
                None if frame.agent is None else frame.agent.export_suspended()
                for frame in nested
            ],
            "turn_child_step_ranges": [list(step_range) for step_range in turn.child_step_ranges],
            "turn_steps_used": self._turn_steps_used,
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
            # Additive (fix-newconv-pending): the conversation the live pending state belongs to, and the pending
            # state parked for the other conversations. Absent from older blobs (read as unowned, see apply).
            "conversation_id": _owner_encoded(self._live_pending_owner()),
            "parked_pending": {key: list(items) for key, items in self._parked_pending.items()},
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
        # Additive (fix-newconv-pending): what was parked, kept whole until its conversation is bound again.
        if "parked_pending" in state:
            self._parked_pending = {
                str(key): [dict(item) for item in items or []]
                for key, items in (state.get("parked_pending") or {}).items()
            }
        # Absent from blobs written before approval binding: no records.
        if state.get("approval") and self._app_workflow is not None:
            approval_binding.store_for(self._app_workflow).load_state(state["approval"])
        self._suspended_user_message = state.get("suspended_user_message")
        self._pending_clarification_request = state.get(
            "pending_clarification_request"
        )
        # Absent from blobs written before a refused planning command call could suspend a turn.
        self._planning_command_refusal_pending = bool(state.get("skill_refusal_pending"))
        self._planning_command_refusal_answered = bool(state.get("skill_refusal_answered"))
        # Absent from blobs written before sub-task splitting; the defaults mean "not split".
        turn = self._turn_frame()
        turn.subtasks = list(state.get("subtasks") or [])
        # Absent before hand-off continuations: each entry is then its own origin.
        turn.origins = list(state.get("subtask_origins") or range(len(turn.subtasks)))
        turn.subtask_index = int(state.get("subtask_index") or 0)
        turn.subtask_answers = list(state.get("subtask_answers") or [])
        turn.subtask_outputs = list(state.get("subtask_outputs") or [])
        turn.subtask_exhausted = bool(state.get("subtask_exhausted"))
        # Absent from blobs written before the sub-task review gate: no review open, no plan.
        turn.subtask_review_open = bool(state.get("subtask_review_open"))
        # The review's frame flag rides with its open flag: the review is the framework call the turn resumes into.
        turn.framework_entry = turn.subtask_review_open
        plan = state.get("subtask_plan")
        if plan:
            from fastworkflow.review_proposals import SubTaskProposal

            turn.subtask_plan = SubTaskProposal.model_validate(plan)
        else:
            turn.subtask_plan = None
        if turn.subtask_review_open:
            # The restored review keeps its free-text reader (set by the gate of the turn that opened it).
            self._give_review_reader()
        self._next_step = int(state.get("next_step") or 0)
        # Absent before planning command phases: no phased planning command call in progress.
        turn.phase_run = state.get("skill_phase_run") or None
        turn.phase_live = None
        # Absent from blobs written before nested frames: no child ranges, zero steps used.
        turn.child_step_ranges = [
            list(step_range) for step_range in state.get("turn_child_step_ranges") or []
        ]
        self._turn_steps_used = int(state.get("turn_steps_used") or 0)

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

        # The live pending state belongs to the conversation the blob names (unowned when it names none). The binding
        # that follows settles it against the conversation the context is bound to (fix-newconv-pending).
        self._pending_owner_override = (
            _owner_from_state(state) if self._has_live_pending() else _OWNER_UNSET
        )

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

        # Restore the turn agent with the turn frame on top: _workflow_tool_agent reads the top frame.
        del self._frame_stack()[1:]
        react_blob = state.get("react")
        if react_blob and self._awaiting_user:
            self._ensure_agent_initialized()
            if self._workflow_tool_agent is not None:
                self._workflow_tool_agent.import_suspended(react_blob)

        # Absent from blobs written before nested frames: no nested frames.
        nested = [PlanFrame.from_state(frame_state) for frame_state in state.get("frames") or []]
        frame_react = state.get("frame_react") or [None] * len(nested)
        for frame, blob in zip(nested, frame_react):
            if blob and self._awaiting_user:
                self._attach_nested_agent(frame)
                frame.agent.import_suspended(blob)
        self._frames.extend(nested)

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
        cancelled_approvals = self.cancel_pending_approvals()
        if not self._awaiting_user:
            return cancelled_approvals > 0
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
        """Context-change observer: refresh the ReAct agent's available_commands.

        A plain goal bound to its command keeps its narrowed listing through a context switch (fix-3rsq.plan-review.2.9):
        the listing is the bound command's, re-read from the workflow now. Only the turn's own agent (the turn frame).
        """
        from fastworkflow.workflow_agent import _refresh_agent_available_commands

        bound = getattr(self, "_narrow_to", None)
        if bound is not None and self._app_workflow is not None and len(self._frame_stack()) == 1:
            from fastworkflow import run_listing

            inputs = getattr(self._workflow_tool_agent, "inputs", None)
            if isinstance(inputs, dict) and "available_commands" in inputs:
                inputs["available_commands"] = run_listing.bound_command_listing(self._app_workflow, bound)
                return
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

        if self._cme_workflow is None:
            return True
        try:
            return self._cme_workflow.close()
        except ValueError:
            # Child cme workflows should not occur; ignore if mis-invoked.
            logger.debug("WorkflowExecutionContext.close: cme_workflow is not a root session")
            return False

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

    def process_turn(
        self, message: str, *, unattended: bool = False,
        approval_surface: "approval_binding.Surface | None" = None,
        approval_token: Optional[str] = None,
    ) -> "fastworkflow.TurnOutput":
        """
        Execute one user message synchronously and return the public TurnOutput.

        ``unattended=True`` runs the turn with no one to answer: the agent's
        ask_user tool is removed for the whole logical turn. It applies to the
        turn this message begins; a message answering a suspended turn
        continues that turn as it began.

        Shares dispatch with _execute_message(); additionally captures every
        command execution of the logical turn (including ask_user exchanges). The
        full internal TurnResult is built and projected onto the slim public
        TurnOutput (see docs/turn_result_design_final.md section 1a).
        """
        self._note_approval_inbound(approval_surface, approval_token, unattended)
        try:
            command_output = self._execute_message(message, unattended=unattended)
        except Exception as exc:
            approval_binding.clear_inbound(self._app_workflow)
            # A turn that raised is still a turn: record it as failed, with its
            # spans closed under it, before the error propagates. Without this
            # the turn had no record at all and read as lost.
            first_line = (str(exc).strip().splitlines() or [""])[0]
            try:
                self._build_turn_result(
                    None, failure_reason=f"{type(exc).__name__}: {first_line}"[:500])
            except Exception:  # noqa: BLE001 - never mask the turn's own error
                logger.warning("could not record the failed turn", exc_info=True)
            raise
        approval_binding.clear_inbound(self._app_workflow)
        turn_result = self._build_turn_result(command_output)
        return turn_result.turn_output

    @dspy_logger.observe_dspy_calls
    def _execute_message(
        self, message: str, unattended: bool = False
    ) -> fastworkflow.CommandOutput:
        """Shared message dispatch for _execute_message()/process_turn()."""
        self.assert_experiment_claim_current()
        if self._app_workflow is None:
            raise RuntimeError(
                "No app workflow bound; call bind_app_workflow() before executing a message"
            )

        # fix-newconv-pending: a message never answers pending state that belongs to another conversation. The live
        # state is parked (or its approvals cancelled) first, so this message begins a new turn.
        self._settle_pending_for_conversation(self._conversation_id)

        if not self._awaiting_user:
            # A message during suspension is the resume answer — never a reset.
            self._begin_turn(message, unattended=unattended)

        self.push_active_workflow(self._app_workflow)
        try:
            awaiting = approval_binding.active_record(self._app_workflow)
            if awaiting is None or not awaiting.suspended_agent:
                managed = approval_binding.management_reply(self, self._app_workflow, message)
                if managed is not None:
                    return self._approval_text_output(managed)
            if awaiting is not None and (not awaiting.suspended_agent
                                         or message.strip() == approval_binding.CANCEL_COMMAND):
                return self._resume_approval_direct(message)
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
        self, command_output: Optional[fastworkflow.CommandOutput],
        failure_reason: Optional[str] = None,
    ) -> TurnResult:
        """Assemble the TurnResult (and its public turn_output) for the message.

        The turn's ``answer`` is plain text — the agent's final answer (or the
        deterministic command's response text). Per-command structured results
        (success/artifacts) live on ``command_outputs``.

        ``failure_reason`` set means the turn raised instead of producing an
        output: it is recorded as FAILED with that reason.
        """
        answer = command_output.command_response.response if command_output else ""

        if failure_reason is not None:
            status = TurnStatus.FAILED
            completed_at: Optional[datetime] = datetime.now(timezone.utc)
        elif self._awaiting_user:
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

        # The approval token is never in an output: it reaches the host through
        # approval_binding.take_delivery, once, on the live response (ADR section 6.4).
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
                    # The root span belongs to the turn frame, whose alias is "" (design 3.5).
                    # frame_alias is an envelope attribute: tracing.end_span stamps it from the
                    # frame on top, which is the turn frame here, so no emitter writes it.
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
                "subtask_review": self._turn_subtask_review,
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
        if self._awaiting_user:
            # The suspended turn is still waiting for its answer: it is neither ended nor
            # begun over, and its frames stay as they are.
            return fastworkflow.CommandOutput(
                command_response=fastworkflow.CommandResponse(
                    response=AWAITING_ANSWER_REFUSAL_MESSAGE, success=False),
                command_name=action.command_name or "",
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
        self, action: fastworkflow.Action, *,
        approval_surface: "approval_binding.Surface | None" = None,
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
        self._note_approval_inbound(approval_surface, None, False)
        try:
            command_output = self.process_action(action)
        finally:
            approval_binding.clear_inbound(self._app_workflow)
        if self._awaiting_user:
            # The refusal: the suspended turn's record was emitted when it suspended and
            # stands; building the result again would emit it once more with the refusal.
            return fastworkflow.TurnOutput(
                turn_key=self._turn_key or mint_turn_key(),
                status=TurnStatus.AWAITING_USER,
                answer=command_output.command_response.response,
                command_outputs=list(self._turn_outputs),
            )
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

    def _build_react_agent(self):
        """A workflow-tool ReAct agent for one run; its steps number from this context's next step."""
        from fastworkflow.workflow_agent import initialize_workflow_tool_agent
        agent = initialize_workflow_tool_agent(self)
        agent.step_floor = lambda: self._next_step
        return agent

    def _initialize_agent_functionality(self) -> None:
        self._cme_workflow.context["run_as_agent"] = True
        if self._app_workflow:
            self._app_workflow.context["run_as_agent"] = True

        self._workflow_tool_agent = self._build_react_agent()

        # Re-scope the active ReAct agent's available_commands whenever the context changes,
        # driven by the workflow's context-change observer (the single switch chokepoint)
        # Registered once per WEC (agent init runs once). The listener reads the *active* agent
        # dynamically. No-ops when no agent is running. Bound method + remove on close()
        # (and WeakMethod storage on Workflow) avoid listener leaks.
        if self._app_workflow is not None:
            self._app_workflow.add_context_change_listener(self._on_app_context_change)
            self._context_change_listener = self._on_app_context_change

    def _ensure_agent_initialized(self) -> None:
        if self._workflow_tool_agent is None:
            self._initialize_agent_functionality()

    def _reset_agent_suspension(self) -> None:
        """Clear Topology-B ask_user suspend state (abort, finalize, or cancel_pending)."""
        self._awaiting_user = False
        self._suspended_user_message = None
        self._pending_clarification_request = None
        self._planning_command_refusal_pending = False
        # A suspended nested frame (run_planning_command) is dropped with the suspension;
        # what is cleared below belongs to the turn frame, the only frame left.
        del self._frame_stack()[1:]
        self._clear_subtask_state()
        turn_agent = self._turn_frame().agent
        if turn_agent is not None and hasattr(turn_agent, "clear_suspension"):
            turn_agent.clear_suspension()

    def _clear_subtask_state(self) -> None:
        turn = self._turn_frame()
        turn.subtasks = []
        turn.origins = []
        turn.subtask_index = 0
        turn.subtask_answers = []
        turn.subtask_outputs = []
        turn.subtask_exhausted = False
        self._next_step = 0
        turn.evidence_floors = {}
        turn.phase_run = None
        turn.phase_live = None
        # Step numbers restart at 0 with _next_step, so ranges from an earlier run no longer apply.
        turn.child_step_ranges = []
        self._enable_all_tools()

    def _enable_all_tools(self) -> None:
        """Lift any tool restriction a reply left on the agent (see _apply_reply_intent).

        An unattended turn keeps ask_user off: its baseline has no one to ask.
        """
        if self._workflow_tool_agent is not None:
            self._workflow_tool_agent.disabled_tools = (
                frozenset({"ask_user"}) if self._turn_unattended else frozenset()
            )

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
                        trajectory = getattr(
                            self._workflow_tool_agent, "trajectory", None) or {}
                        if any(str(key).startswith("observation_") for key in trajectory):
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
        self._clear_subtask_state()

        # A workflow with planning commands answers a plain "what can you do?" in code, with no model call.
        from fastworkflow.capability_answer import capability_answer

        if self._app_workflow and (answer := capability_answer(
                message, self._app_workflow, self._cme_workflow.folderpath)) is not None:
            return dspy.Prediction(final_answer=answer, exhausted=False)

        from fastworkflow.workflow_agent import split_into_subtasks

        subtasks = split_into_subtasks(
            refined_user_query,
            planner_lm=getattr(self, "_current_planner_lm", None),
            workflow_folderpath=self._app_workflow.folderpath if self._app_workflow else None,
        )
        if self._app_workflow is not None:
            # The review gate (fix-3rsq.plan-review.2.6): None when the sub-tasks run now, or the suspended review.
            asked = self._subtask_gate(subtasks)
            if asked is not None:
                return asked
        return self._run_turn_subtasks(subtasks, refined_user_query, message)

    def _run_turn_subtasks(self, subtasks: list[str], refined_user_query: str, message: str):
        """Run the turn's sub-tasks: several, one after another; or one, with the step-cap hand-off. A turn with a
        sub-task plan (accepted, or the draft) runs through _run_subtasks_from, which binds each goal to its means."""
        plan = self._turn_subtask_plan
        planned = plan is not None and len(plan.items) == len(subtasks)
        if len(subtasks) > 1 or planned:
            self._subtasks = subtasks
            self._origins = list(range(len(subtasks)))
            return self._run_subtasks_from(0)

        # An unsplit request is one sub-task, so a run that hits the step cap is handed off the same way.
        self._subtasks = [refined_user_query]
        self._origins = [0]
        self._evidence_floors = {0: 0}
        result = self._call_agent_for_query(refined_user_query, message, evidence_floor=0)
        if getattr(result, "exhausted", None) is not True:
            return result
        self._record_subtask_answer(result)
        self._hand_off(0)
        return self._run_subtasks_from(1)

    # ------------------------------------------------------------------
    # The sub-task review gate (fix-3rsq.plan-review.2.2, .2.3, .2.6, .2.8)
    # ------------------------------------------------------------------

    def _subtask_no_answerer(self) -> bool:
        """True when nobody can answer a review this turn: an unattended turn (ask_user is off for it), or a host
        surface (REST, MCP). Owner decision 2. An agent-to-agent turn reaches the framework as an unattended turn."""
        store = approval_binding.store_for(self._app_workflow)
        host = store is not None and store.surface.kind == "host"
        return bool(self._turn_unattended) or host

    def _subtask_gate(self, subtasks: list[str]):
        """Decide how the turn's sub-tasks run, after split and merge, before any of them runs.

        Fast path (one sub-task, confident head or exact name): no review, today's run. A turn with no one to answer:
        the draft runs as proposed, recorded as not reviewed. Otherwise the review_subtasks framework command opens with
        the draft and suspends the turn; returns the suspended prediction (its replies come through
        _resume_agent_message). Returns None when the sub-tasks run now.
        """
        from fastworkflow import review_proposals

        workflow = self._app_workflow
        if review_proposals.fast_path_for(workflow, subtasks) is not None:
            # plan-review.6 hook: review_proposals.planning_intent_confirmation is where the one-line confirmation of a
            # fast-path planning command goes. Round 1 asks nothing here.
            self._turn_subtask_review = "fast_path"
            return None
        draft = review_proposals.draft_round1_proposal(workflow, subtasks)
        self._turn_subtask_plan = draft
        if self._subtask_no_answerer():
            # Owner decision 2: no one can answer, so the draft is recorded as not reviewed and the turn carries on.
            # The draft runs as proposed: its goals and bound means drive the run (_run_subtasks_from).
            self._turn_subtask_review = "not_reviewed"
            logger.info("sub-task review not opened (no one to answer): %d sub-task(s) run as drafted, not reviewed",
                        len(draft.items))
            return None
        self._turn_subtask_review = "open"
        turn = self._turn_frame()
        turn.subtask_review_open = True
        self._give_review_reader()
        try:
            with review_proposals.draft_scope(draft):
                asked = self.run_framework_planning_command(
                    "review_subtasks", {"subtasks": review_proposals.DRAFT_ITEMS}, "review the sub-tasks",
                    within_turn=True)
        except BaseException:
            turn.subtask_review_open = False
            raise
        if getattr(asked, "suspended", None) is not True:
            turn.subtask_review_open = False
            raise AssertionError("review_subtasks must suspend the turn on its review")
        return asked

    def _give_review_reader(self) -> None:
        """The free-text reader of this turn's sub-task review (fix-3rsq.plan-review.2.7): one model call on the
        planner's role, with the planner's LM when the turn has one. Set on the session's review store."""
        from fastworkflow import planning_commands, review_engine, review_proposals

        store = review_engine.store_for(self._app_workflow)
        if store is None:
            return
        names = tuple(sorted(planning_commands.resolve_field_sources(self._app_workflow.folderpath).planning_commands))
        # The planner's LM is read when a reply comes, not when the review opens: the turn's planner LM is the one in place then.
        store.edit_reader = review_proposals.model_edit_reader(
            planner_lm=lambda: getattr(self, "_current_planner_lm", None), planning_names=names)

    def _accept_subtask_review(self, agent_result):
        """The gate's review was accepted: its frame finishes with the accepted proposal, which is stored for the
        hand-off (fix-3rsq.plan-review.2.9 reads ``_turn_subtask_plan``), and the turn runs the accepted goals as today.
        Its inputs and means are not used until 2.9."""
        from fastworkflow import review_proposals

        turn = self._turn_frame()
        turn.subtask_review_open = False
        turn.framework_entry = False
        self._finish_framework_call(agent_result)
        accepted = review_proposals.accepted_subtask_proposal(self._app_workflow)
        if accepted is None:
            raise RuntimeError("the accepted sub-task review is not on the session")
        self._turn_subtask_plan = accepted
        self._turn_subtask_review = "reviewed"
        subtasks = [item.goal for item in accepted.items]
        return self._run_turn_subtasks(subtasks, subtasks[0] if len(subtasks) == 1 else self._turn_refined_message,
                                       self._turn_user_message)

    def _end_subtask_review_unrun(self, state: str):
        """The gate's review was declined or expired: its frame is dropped, nothing runs, and the turn says so."""
        from fastworkflow import review_engine

        turn = self._turn_frame()
        turn.subtask_review_open = False
        turn.framework_entry = False
        self._pop_frame(self._top_frame())
        declined = state == review_engine.DECLINED
        self._turn_subtask_review = "declined" if declined else "expired"
        self._turn_subtask_plan = None
        text = SUBTASK_REVIEW_DECLINED if declined else SUBTASK_REVIEW_EXPIRED
        return self._settle_agent_result(self._turn_user_message, dspy.Prediction(final_answer=text, exhausted=False))

    def _call_agent_for_query(self, query: str, planner_user_query: str, *,
                              first_step: int | None = None,
                              template_text: str | None = None,
                              evidence_floor: int | None = None,
                              bound_command: str | None = None):
        """Plan one query with the planner, then run the ReAct agent on it.

        ``template_text`` picks the planning command template; it defaults to the query itself.
        ``evidence_floor`` is the first step index whose archived outputs count as
        this run's for the finish check (a hand-off continuation counts the runs
        of its sub-task before it); it defaults to ``first_step``.
        """
        from fastworkflow import review_proposals
        from fastworkflow.workflow_agent import build_query_with_next_steps, sub_task_commands_text

        # One planning command match gives both the template and its parsed form; a hand-off continuation
        # (query differs from its original text) plans from the template only, since the planning command
        # call would plan the whole job again.
        # The match's option distribution (empty without JEV_API_KEY) feeds the listings' gathering (run_listing).
        # The match is memoised for the turn (review_proposals.planning_match): the gate's draft asked it already.
        planning_distribution: dict[str, float] = {}
        plan_template, planning_command = "", None
        if self._app_workflow is not None:
            plan_template, planning_command, planning_distribution = review_proposals.planning_match(
                self._app_workflow, template_text or query)
        if template_text is not None and template_text != query:
            planning_command = None
        if bound_command is not None:
            # The bound means of a plain goal: the plan names its command, so the finish check holds the goal to it.
            plan_template = f"This sub-task is done with the command {bound_command}."
            planning_command = None
        # Initial turn planning uses every prior turn summary, not the previous
        # turn's full agent trajectory (mid-turn replans still use trajectory).
        command_info_and_refined_message_with_todolist = build_query_with_next_steps(
            query,
            self,
            with_agent_inputs_and_trajectory=False,
            planner_lm=getattr(self, "_current_planner_lm", None),
            planner_user_query=planner_user_query,
            plan_template=plan_template or None,
            planning_command=planning_command,
            planning_probabilities=planning_distribution or None,
        )
        refusal, self._planning_command_refusal = self._planning_command_refusal, None
        if refusal is not None:
            # A refused planning command call the user is asked about: the turn suspends on the question as
            # an ask_user does (no plan, no run); the reply re-runs it (_rerun_after_planning_command_refusal).
            self._planning_command_refusal_pending = True
            return dspy.Prediction(suspended=True, clarification=refusal, exhausted=False)
        phased, self._phased_planning_command_call = getattr(self, "_phased_planning_command_call", None), None
        if phased is not None:
            # A planning command call: its phases run as sub-plans, with no hand-off.
            return self._start_planning_command_phases(query, *phased, first_step=first_step or 0)
        if bound_command is not None:
            from fastworkflow import run_listing

            available_commands = run_listing.bound_command_listing(self._app_workflow, bound_command)
        else:
            available_commands = sub_task_commands_text(self, query, planning_probabilities=planning_distribution or None)
        step_kwargs = {} if first_step is None else {"first_step": first_step}
        self._attach_finish_check(
            evidence_floor if evidence_floor is not None else (first_step or 0))

        return self._call_agent_with_retry(
            lambda: self._workflow_tool_agent(
                user_query=command_info_and_refined_message_with_todolist,
                available_commands=available_commands,
                **step_kwargs,
            ),
            trace_input=command_info_and_refined_message_with_todolist,
        )

    def _attach_finish_check(self, floor: int, *, values: bool = False,
                             item_binding: Optional[tuple[Any, list, list]] = None) -> None:
        """Give the agent this run's finish check, or none when the run has no plan.

        The plan is the one ``build_query_with_next_steps`` just made for this
        run. Set before every run, so a check never outlives the run it was made for.
        *values*: a step's concrete argument values must match (a planning command phase's run).
        *item_binding* ``(item, other items, the phase's planned command names)``: the
        run is one item of a fan-out phase,
        bound to its record; the check also flags a run on another item's record and
        an item never opened (``planning_phases.item_record_issues``).
        """
        agent = self._workflow_tool_agent
        if agent is None:
            return
        agent.finish_reviewer = None
        from fastworkflow.observation_offloading import finish_check

        plan_text = str(getattr(self, "_initial_plan_text", "") or "")
        if not plan_text.strip() or self._app_workflow is None:
            return
        folderpath = self._app_workflow.folderpath

        def review(trajectory, iters_left: int):
            from fastworkflow.workflow_agent import core_command_names

            # The archive keeps the command text as sent; the action log says
            # which command each text resolved to (an alias, a misspelling).
            note = finish_check.review_finish(
                plan_text=plan_text, floor=floor, iters_left=iters_left,
                commands=self._workflow_command_names(),
                core_commands=core_command_names(),
                resolved=finish_check.resolved_command_names(
                    getattr(self, "_action_log", None) or ()),
                values=values)
            if item_binding is None or iters_left < finish_check.MIN_ITERS_LEFT:
                return note
            from fastworkflow import planning_phases

            try:
                runs = self._archived_command_runs(from_step=int(floor or 0))
                item, others, planned = item_binding
                binding = planning_phases.item_record_note(item, planning_phases.item_record_issues(
                    runs, item, others, self._current_record()[0], planned))
            except Exception as error:  # noqa: BLE001 - the check never fails a turn
                logger.warning("item record check skipped: %s: %s", type(error).__name__, error)
                return note
            return "\n".join(part for part in (note, binding) if part) or None

        agent.finish_reviewer = review

    # ------------------------------------------------------------------
    # Phased planning command calls (fastworkflow.planning_phases)
    # ------------------------------------------------------------------

    def _start_planning_command_phases(self, query: str, command, call, *, first_step: int):
        """Run a phased planning command call's phases, in order; the sub-task's result.

        ``call`` is the planning command call (``planning_commands.PlanningCommandCallPlan``) whose ``phases``
        run. See ``fastworkflow.planning_phases`` for what the runner does and decides.
        Extra command steps after the planning command's own that would land in an answer-only last
        phase run in a phase of their own just before it (``planning_phases.extra_steps_phase``).
        The extra steps before the planning command's own run first, in a phase of their own
        (``planning_phases.steps_before_phase``).
        """
        from fastworkflow import planning_phases
        from fastworkflow.observation_offloading.facts import fact_sources

        names = self._phase_command_names()
        names["commands"] = names["commands"] + [
            source.command for phase in call.phases if phase.output is not None
            for source in fact_sources(phase.output)]
        extra, steps_after = planning_phases.extra_steps_phase(call.phases, call.steps_after, **names)
        phases = planning_phases.with_phase_before(call.phases, extra)
        before = planning_phases.steps_before_phase(phases, call.steps_before)
        phases = planning_phases.with_phase_before(phases, before, 0)
        self._phase_live = (command, phases)
        self._phase_run = {
            "skill": command.name,
            "args": json.loads(json.dumps(call.args, default=str)),
            "query": query,
            "phases": [phase.name for phase in phases],
            "steps_before": list(call.steps_before),
            "steps_after": steps_after,
            # The phase the extra command steps run in (None when there is none), its steps
            # (to rebuild it after a restore) and its answer (added to the job's answer).
            "extra_phase": extra.name if extra is not None else None,
            "extra_steps": extra.steps_fn() if extra is not None else [],
            "extra_answer": "",
            # The phase the extra steps before the planning command's own run in (None when there
            # are none; its steps are steps_before) and its answer (added to the job's).
            "before_phase": before.name if before is not None else None,
            "before_answer": "",
            "index": 0,
            "outputs": {},
            "answers": [],
            "phase_answers": [],
            # A fan-out phase in progress: its items (capped), how many there were, and
            # the item now running. None outside a fan-out phase.
            "items": None,
            "items_total": 0,
            "item_index": 0,
            # A fan-out phase's `until` condition was met: its remaining items do not run.
            "until_met": False,
            "notes": [],
            "call_first_step": first_step,
            "first_step": first_step,
            "plan": "",
            "uses_result": bool(getattr(command, "uses_result", False)),
            # A flat call: one phase named after the planning command (call_phases).
            "flat": len(call.phases) == 1 and call.phases[0].name == command.name,
        }
        return self._run_planning_command_phases(first_step)

    def _live_planning_command_phases(self):
        """``(command, phases)`` of the call in progress; rebuilt from the planning command after a restore."""
        if self._phase_live is not None:
            return self._phase_live
        from fastworkflow.planning_commands import call_phases
        from fastworkflow.planning_phases import PhaseError
        from fastworkflow.workflow_agent import load_planning_command_map

        state = self._phase_run
        folderpath = self._app_workflow.folderpath
        command = load_planning_command_map(folderpath).get(state["skill"])
        if command is None:
            # A planning command's call (run_planning_command) is not a planning command file.
            command = planning_commands.planning_command(folderpath, state["skill"])
        if command is None:
            raise PhaseError(f"planning command {state['skill']!r} is no longer defined")
        # A flat call (no plan(), or no phases for these arguments) is one phase.
        phases = call_phases(command, state["args"])
        if state.get("extra_phase"):
            from fastworkflow.planning_phases import Phase, with_phase_before

            moved = list(state.get("extra_steps") or [])
            phases = with_phase_before(phases, Phase(state["extra_phase"], lambda: list(moved)))
        if state.get("before_phase"):
            from fastworkflow.planning_phases import Phase, with_phase_before

            first = list(state.get("steps_before") or [])
            phases = with_phase_before(phases, Phase(state["before_phase"], lambda: list(first)), 0)
        if [phase.name for phase in phases] != state["phases"]:
            raise PhaseError(f"planning command {state['skill']!r} now plans other phases")
        self._phase_live = (command, phases)
        return self._phase_live

    def _phase_outputs(self, phases, index: int) -> dict[str, Any]:
        """The typed outputs of the phases before *index* that filled one (a fan-out
        phase's: the list of its per-item outputs)."""
        state = self._phase_run
        outputs: dict[str, Any] = {}
        for p in phases[:index]:
            if p.output is None or p.name not in state["outputs"]:
                continue
            value = state["outputs"][p.name]
            outputs[p.name] = ([p.output.model_validate(v) for v in value] if p.fans_out
                               else p.output.model_validate(value))
        return outputs

    def _run_planning_command_phases(self, first_step: Optional[int] = None, *,
                          resume_observation: Optional[str] = None,
                          resume_completes_step: bool = False):
        """Run the phases from the current one; a suspension, or the sub-task's result.

        With ``resume_observation`` the current phase's executor run, suspended on
        ask_user (or on a nested command, ``resume_completes_step``), is resumed first. The
        result never reports ``exhausted``: a phased
        planning command is not handed off. A fan-out phase runs one sub-plan per item, until its
        items are done or its ``until`` condition is met. A phase with a callable
        ``when`` is decided when it is reached (``planning_phases.runs_now``); a skipped one
        runs nothing and keeps the answers of the phase before it.
        """
        from fastworkflow import planning_phases

        state = self._phase_run
        try:
            command, phases = self._live_planning_command_phases()
        except Exception as error:  # noqa: BLE001 - the call stops, saying why
            return self._planning_command_phases_failed(state["phases"][state["index"]], f"{error}")
        phase_names = frozenset(state["phases"])
        while state["index"] < len(phases):
            if resume_observation is None and self._nested_budget_spent():
                # A nested call's later phases do not start once the turn-wide budget is
                # spent (run_planning_command); the call ends here, as exhausted. A
                # resumed phase is not stopped here: it is already running.
                state["exhausted"] = True
                break
            phase = phases[state["index"]]
            starting = resume_observation is None and (
                not phase.fans_out or state.get("items") is None)
            if starting:
                # What an optional phase that fails gives back (_phase_failure).
                state["phase_start"] = {"answers": len(state["answers"]),
                                        "phase_answers": list(state.get("phase_answers") or []),
                                        "notes": len(state.get("notes") or [])}
            if starting and phase.has_condition:
                try:
                    runs, why = planning_phases.runs_now(
                        phase, self._phase_outputs(phases, state["index"]),
                        phase_names=phase_names), None
                except planning_phases.PhaseError as error:
                    runs, why = False, str(error)
                if not runs:
                    # Skipped: counts as not run (no output, later parameters keep their
                    # defaults). The phase's span says so; no executor runs.
                    # Phase 3 step 1a: phase-named parameters bind None when skipped, not
                    # those defaults (``planning_phases.bound_arguments`` with ``phase_names``).
                    outcome, value = self._run_planning_command_phase(command, phases, first_step,
                                                           skip=True, skip_error=why)
                    if outcome == "failed":
                        failed = self._phase_failure(phase, value)
                        if failed is not None:
                            return failed
                        continue
                    state["index"] += 1
                    continue
            if starting:
                state["phase_answers"] = []
            if phase.fans_out and state.get("items") is None:
                try:
                    items, total = planning_phases.fan_out_items(
                        phase, self._phase_outputs(phases, state["index"]))
                except planning_phases.PhaseError as error:
                    failed = self._phase_failure(phase, str(error))
                    if failed is not None:
                        return failed
                    continue
                state.update(items=json.loads(json.dumps(items, default=str)),
                             items_total=total, item_index=0, until_met=False)
                state["outputs"][phase.name] = []
                if total > len(items) and phase.until is None:
                    note = planning_phases.cap_note(phase.name, len(items), total, phase.noun)
                    # Two fan-outs over the same items with one noun say it once.
                    if note not in state["notes"]:
                        state["notes"].append(note)
            if not phase.fans_out or (state["item_index"] < len(state["items"])
                                      and not state.get("until_met")):
                outcome, value = self._run_planning_command_phase(
                    command, phases,
                    state["first_step"] if resume_observation is not None else first_step,
                    resume_observation=resume_observation,
                    resume_completes_step=resume_completes_step)
                resume_observation = None
                resume_completes_step = False
                if outcome == "suspended":
                    return value
                if outcome == "failed":
                    first_step = self._next_step
                    failed = self._phase_failure(phase, value)
                    if failed is not None:
                        return failed
                    continue
                first_step = self._next_step
                if phase.fans_out:
                    state["item_index"] += 1
                    continue
            if phase.fans_out and phase.until is not None:
                # Said at the end: stopped on its condition, or capped.
                note = planning_phases.fan_out_note(phase.name, state["item_index"],
                                                 state["items_total"], bool(state.get("until_met")),
                                                 phase.noun)
                if note and note not in state["notes"]:
                    state["notes"].append(note)
            state.update(items=None, items_total=0, item_index=0, until_met=False)
            state["index"] += 1
        uses_result = state.get("uses_result")
        command_output = None
        response = None
        if uses_result and command.plan_cls is not None:
            from fastworkflow.planning_commands import _phase_names_for_command, input_instance

            typed = planning_phases.typed_phase_outputs(state["outputs"], phases)
            input_obj = input_instance(command, state["args"])
            # __result__ may name any phase of the command: one this call left out is None (3.16).
            response, _fallback = planning_phases.run_call_result(
                command.plan_cls, input_obj, typed,
                phase_names=phase_names | frozenset(_phase_names_for_command(command)),
                output_model=command.output_model,
                command_name=command.name,
                request=state["query"],
            )
        if response is not None:
            answer = response.response or ""
            notes = list(state.get("notes") or [])
            if notes:
                answer = "\n\n".join(part for part in (answer, " ".join(notes)) if part)
            command_output = fastworkflow.CommandOutput(
                command_response=response,
                command_name=command.name,
                command_parameters=input_obj,
            )
            if self._app_workflow:
                command_output.workflow_name = self._app_workflow.folderpath.split("/")[-1]
            fastworkflow.turn.warn_on_missing_declared_output(
                command_output,
                output_model=command.output_model,
            )
            if len(self._frame_stack()) == 1:
                # The turn's own call appends its output here. A nested call's output is
                # returned instead: run_planning_command hands it to the step that ran it,
                # which appends it (workflow_agent._execute_workflow_query).
                self.append_turn_output(command_output)
        else:
            # No __result__, or it returned None (no code-written answer, 3.16): the answer is the
            # one the phases give, as with no __result__. A deferred answer is written now.
            settled: dict = {}
            answers = [self._settled_phase_answer(a, phases, settled) for a in state["answers"]]
            phase_answers = [self._settled_phase_answer(a, phases, settled)
                             for a in state.get("phase_answers") or []]
            # The last phase writes the answer, from the facts its steps carry (a fan-out
            # last phase: its items' answers, in item order).
            last = [a for a in phase_answers if a]
            answer = "\n\n".join(last) if last else "\n\n".join(a for a in answers if a)
            extra = self._settled_phase_answer(state.get("extra_answer"), phases, settled)
            if extra and last:
                # The extra command steps' own report (their phase is not the last).
                answer = f"{answer}\n\n{extra}"
            before = self._settled_phase_answer(state.get("before_answer"), phases, settled)
            if before and last:
                # The report of the extra steps before the planning command's own (their phase is first).
                answer = f"{before}\n\n{answer}"
            notes = list(state.get("notes") or [])
            if notes:
                answer = "\n\n".join(part for part in (answer, " ".join(notes)) if part)
        self._phase_run = self._phase_live = None
        # exhausted stays False: a phased planning command is not handed off. phase_exhausted says a
        # phase ran out of steps, and command_output is the call's own output when its
        # __result__ built one (uses_result), which a nested call returns to its step.
        return dspy.Prediction(final_answer=answer, exhausted=False,
                               phase_exhausted=bool(state.get("exhausted")),
                               command_output=command_output)

    def _defer_phase_answer(self, index: int, query: str, plan: str) -> dict:
        """Hold an answer-only phase's model call until the answer is settled (3.16): it runs
        only when ``__result__`` returned None. Returns the marker the phase records."""
        deferred = self._phase_run.setdefault("deferred", {})
        key = str(len(deferred))
        deferred[key] = {"index": index, "query": query, "plan": plan}
        return {"deferred": key}

    def _settled_phase_answer(self, value, phases, settled: dict) -> str:
        """A recorded phase answer as text. A deferred one (a ``{"deferred": key}`` marker) is
        written now, once, by the model from the facts its phase had (``answer_only_answer``)."""
        if not (isinstance(value, dict) and "deferred" in value):
            return value or ""
        key = value["deferred"]
        if key not in settled:
            import dspy
            from fastworkflow import planning_phases

            deferred = self._phase_run["deferred"][key]
            phase = phases[deferred["index"]]
            # The facts of the phase itself and of the phases before it (a judgement phase's own
            # output included), given to the model with the request.
            outputs = self._phase_outputs(phases, deferred["index"] + 1)
            facts = "\n\n".join(planning_phases.found_facts(outputs))
            query = f"{deferred['query']}\n\n{facts}" if facts else deferred["query"]
            span = tracing.start_span(self, tracing.SPAN_PLANNING_COMMAND_ANSWER, attributes={
                "skill": self._phase_run["skill"], "phase": phase.name,
                "phase_index": deferred["index"], "phase_count": len(phases)})
            try:
                result = self._call_agent_with_retry(
                    lambda: dspy.Prediction(
                        final_answer=planning_phases.answer_only_answer(
                            query, deferred["plan"], outputs),
                        suspended=False, exhausted=False),
                    trace_input=query,
                )
            except BaseException as exc:
                tracing.end_span(self, span, status=tracing.status_for_dispatch_exception(exc),
                                 attributes={"error_type": type(exc).__name__})
                raise
            settled[key] = planning_phases.clean_answer(getattr(result, "final_answer", "") or "")
            tracing.end_span(self, span)
        return settled[key]

    def _record_phase_answer(self, phase, answer) -> None:
        """Record a phase's answer (text, or a deferred marker) for the call's answer."""
        state = self._phase_run
        if phase.name == state.get("extra_phase"):
            state["extra_answer"] = answer
        if phase.name == state.get("before_phase"):
            state["before_answer"] = answer
        state["answers"].append(answer)
        state.setdefault("phase_answers", []).append(answer)

    def _phase_failure(self, phase, error: str):
        """Phase *phase* failed: the sub-task's result (``_planning_command_phases_failed``), or None
        when the phase is ``optional``. An optional phase then counts as not run: its
        output, the answers its runs wrote and the notes it added are dropped, and the
        runner goes on with the next phase."""
        if not getattr(phase, "optional", False):
            return self._planning_command_phases_failed(phase.name, error)
        state = self._phase_run
        start = state.get("phase_start") or {}
        del state["answers"][start.get("answers", len(state["answers"])):]
        state["phase_answers"] = list(start.get("phase_answers") or [])
        del state["notes"][start.get("notes", len(state["notes"])):]
        state["outputs"].pop(phase.name, None)
        state.update(items=None, items_total=0, item_index=0, until_met=False)
        state["index"] += 1
        logger.info("planning command %s: optional phase %s failed and is left out: %s",
                    state["skill"], phase.name, error)
        return None

    def _planning_command_phases_failed(self, name: str, error: str):
        """The sub-task's result when phase ``name`` failed: the facts found, and what failed.

        Only facts filled in code are shown (``planning_phases.found_facts``), never text a
        phase's run wrote: the failed phase's own answer may report what it never read."""
        from fastworkflow.planning_phases import failure_text, found_facts

        state = self._phase_run
        names = state["phases"]
        not_run = names[names.index(name) + 1:] if name in names else []
        logger.info("planning command %s: phase %s failed: %s", state["skill"], name, error)
        self._phase_run = self._phase_live = None
        return dspy.Prediction(
            final_answer=failure_text(found_facts(state.get("outputs") or {}), name, error,
                                      not_run, notes=state.get("notes") or ()),
            exhausted=False, phase_failed=name)

    # ------------------------------------------------------------------
    # Planning commands as nested frames (design 3.5)
    # ------------------------------------------------------------------

    def _nested_budget_spent(self) -> bool:
        """True inside a nested frame once the turn-wide step budget is spent.

        Only nested frames stop on it: the turn frame is bounded by its own caps (design 3.5).
        """
        return len(self._frame_stack()) > 1 and self._turn_steps_used >= turn_max_steps()

    def run_framework_planning_command(self, qualified_name: str, params: Any, step_text: str, *,
                                       unattended: bool = False, within_turn: bool = False):
        """The framework's own entry point for a framework planning command (fix-3rsq.plan-review.1).

        Runs *qualified_name* (review_subtasks, review_plan) as one turn, through the same nested frame
        and executor a planning step uses, without the agent's explicit-name resolution (which refuses
        framework commands). Returns the command's output, or the awaiting output when the review asks the
        user (its replies come as ordinary messages on this session).

        ``within_turn``: the call runs inside a turn already begun (the sub-task gate): that turn's state, its
        unattended flag and its approval surface are kept, and a suspension returns the suspended prediction for the
        caller to settle, not the awaiting output (fix-3rsq.plan-review.2.8). Without it the call begins a turn of its
        own, with *unattended* (attended by default, as the review always was).
        """
        from fastworkflow.utils.react import AskUserSuspend

        from fastworkflow.workflow_agent import framework_call_scope

        self._ensure_agent_initialized()
        if not within_turn:
            self._begin_turn(step_text, unattended=unattended)
            self._note_approval_inbound(None, None, False)
        # The turn frame has no agent step of its own: when the call finishes, its output is the turn's answer.
        self._turn_frame().framework_entry = True
        self.push_active_workflow(self._app_workflow)
        try:
            with framework_call_scope():
                output = self.run_planning_command(qualified_name, params, step_text)
        except AskUserSuspend as asked:
            self._awaiting_user = True
            if within_turn:
                return dspy.Prediction(suspended=True, clarification=asked.clarification_request)
            return self._awaiting_user_output(asked.clarification_request)
        finally:
            self.pop_active_workflow()
        self.append_turn_output(output)
        return output

    def run_planning_command(self, qualified_name: str, params: Any,
                             step_text: str) -> fastworkflow.CommandOutput:
        """Run planning command *qualified_name* as a nested frame, for the step *step_text*.

        *params* are the arguments the executor extracted and checked: an ``Input``
        instance or a dict. Returns the command's output, which the step that called it
        holds as its run. A call that asks the user raises AskUserSuspend and keeps its
        frame on the stack for the resume (sibling task; design 3.5, Ask_user inside a child).
        """
        frames = self._frame_stack()
        if any(frame.command_name == qualified_name for frame in frames):
            return self._nested_refusal(
                qualified_name,
                f"{qualified_name} is already running; call its steps directly.")
        limit = turn_max_steps()
        if self._turn_steps_used >= limit:
            return self._nested_refusal(qualified_name, NESTED_BUDGET_SPENT_MESSAGE)
        folderpath = self._app_workflow.folderpath if self._app_workflow is not None else ""
        command = planning_commands.planning_command(folderpath, qualified_name) if folderpath else None
        if command is None:
            return self._nested_refusal(
                qualified_name, f"{qualified_name} is a planning command that could not be loaded.")
        args = _planning_args(params)
        try:
            phases = planning_commands.call_phases(command, args)
        except planning_commands.PlanningCommandCallError as error:
            return self._nested_refusal(qualified_name, f"{qualified_name} could not run: {error}")

        # The step in flight is the parent's execute step k; the child's steps number after
        # it and after every step already run (step numbers stay unique across frames).
        parent = self._top_frame()
        alias = current_execute_alias(parent.agent) if parent.agent is not None else None
        invoking_step = int(alias[1:]) if alias else None
        first_step = self._next_step
        if invoking_step is not None:
            first_step = max(first_step, invoking_step + 1)
        stem = qualified_name.split("/")[-1]
        count = parent.child_counts.get(stem, 0) + 1
        parent.child_counts[stem] = count
        ran_in_clause, ran_in_context_name = self._current_record()
        run_state = getattr(parent.agent, "run_state", None)
        execute_call = run_state.get("execute_call", {}) if isinstance(run_state, dict) else {}
        frame = PlanFrame(
            kind="planning",
            command_name=qualified_name,
            depth=parent.depth + 1,
            alias=f"{parent.alias}/{stem}#{count}",
            subtasks=[step_text],
            origins=[0],
            subtask_index=0,
            invoking_step=invoking_step,
            first_step=first_step,
            ran_in_clause=ran_in_clause,
            ran_in_context_name=ran_in_context_name,
            call_args=json.loads(json.dumps(args, default=str)),
            command_text=execute_call.get("command", ""),
            folded_note=execute_call.get("folded_note", ""),
        )
        frames.append(frame)
        self._attach_nested_agent(frame)
        # The call's plan is its phases, which carry their own steps: nothing is planned
        # with a model here (the phases of call_phases).
        call = planning_commands.PlanningCommandCallPlan(plan="", args=args, notes=[], prediction=None,
                                            phases=phases)
        try:
            result = self._start_planning_command_phases(step_text, command, call, first_step=first_step)
        except AskUserSuspend:
            raise  # the frame stays for the resume
        except BaseException:
            self._pop_frame(frame)
            raise
        if getattr(result, "suspended", None) is True:
            # The frame stays on the stack; the parent's step is suspended with it.
            raise AskUserSuspend(str(getattr(result, "clarification", "") or ""))

        command_parameters = params if hasattr(params, "model_dump") else (
            planning_commands.input_instance(command, args))
        return self._close_planning_call(frame, result, command_parameters)

    def _attach_nested_agent(self, frame: PlanFrame) -> None:
        """Give a nested frame its agent: each of its model steps counts against the turn-wide
        budget, and ask_user is off for an unattended turn. Also set on a restored frame, which
        is built the same way before its suspended state is imported."""
        frame.agent = self._build_react_agent()
        frame.agent.disabled_tools = (
            frozenset({"ask_user"}) if getattr(self, "_turn_unattended", False) else frozenset())
        limit = turn_max_steps()

        def count_step() -> bool:
            # Each model step of a nested frame counts against the turn-wide budget.
            self._turn_steps_used += 1
            return self._turn_steps_used < limit

        frame.agent.on_iteration = count_step

    def _close_planning_call(self, frame: PlanFrame, result,
                             command_parameters) -> fastworkflow.CommandOutput:
        """Pop *frame*, a nested call that finished with *result*, and build the output its
        parent's step holds: the call's step range is recorded on the parent."""
        limit = turn_max_steps()
        ran = self._next_step > frame.first_step
        self._pop_frame(frame)
        if ran:
            self._top_frame().child_step_ranges.append([frame.first_step, self._next_step - 1])

        output = getattr(result, "command_output", None)
        if output is None:
            output = fastworkflow.CommandOutput(
                command_response=fastworkflow.CommandResponse(
                    response=str(getattr(result, "final_answer", "") or "")),
            )
        if getattr(result, "phase_exhausted", False) and self._turn_steps_used >= limit:
            # The turn-wide budget stopped this call's steps (design 3.5): its response says so.
            # A call that finished, or a phase that ran out of its own cap, keeps its answer.
            output = fastworkflow.CommandOutput(
                command_response=fastworkflow.CommandResponse(
                    response=NESTED_BUDGET_SPENT_MESSAGE, success=False),
            )
        if getattr(result, "phase_failed", None) or getattr(result, "phase_exhausted", False):
            # Out of steps or failed: the step's observation says so, and the run is not a success.
            output.command_response.success = False
        output.command_name = frame.command_name
        output.command_parameters = command_parameters
        # The context the call ended on: the one its parent's later steps start from.
        active = self.get_active_workflow()
        if active is not None:
            output.workflow_name = active.folderpath.split("/")[-1]
            output.context = active.current_command_context_displayname
        return output

    def _finish_framework_call(self, result) -> fastworkflow.CommandOutput:
        """The framework's own call (run_framework_planning_command) finished after a suspension: its planning
        command's typed output is the turn's answer, and its frame is closed."""
        frame = self._top_frame()
        command = planning_commands.planning_command(self._app_workflow.folderpath, frame.command_name)
        output = self._close_planning_call(frame, result, planning_commands.input_instance(command, frame.call_args))
        self._pop_frame(frame)
        self._awaiting_user = False
        self.append_turn_output(output)
        return output

    def _finish_nested_frame(self, result) -> str:
        """Close the nested call on top, which finished after a suspension, and return the
        observation of the step it was called from.

        That step is still in flight, so the call's run is held under it, and its turn output
        and action-log record are written, as a call that finishes at once writes them
        (_execute_workflow_query). The observation reads the run as the step's own would have,
        with its handle line. The step's alias, command and folded note are the frame's: a
        restored frame has them, where the step's agent may not have its trajectory yet.
        """
        # inline import avoids circular import: workflow_agent imports this module at load (CommandCancelledError)
        from fastworkflow.workflow_agent import (
            _append_action_record, _append_turn_output, _hold_command_run, action_record,
            command_run_response, run_observation)

        frame = self._top_frame()
        command = planning_commands.planning_command(
            self._app_workflow.folderpath, frame.command_name)
        output = self._close_planning_call(
            frame, result, planning_commands.input_instance(command, frame.call_args))
        _append_turn_output(self, output)
        alias = f"O{frame.invoking_step}" if frame.invoking_step is not None else None
        # The child's last step left the workflow where it ended: the run is held on that record.
        now_clause, now_name = self._current_record()
        if alias is not None:
            _hold_command_run(self, output, now_clause, now_name, alias=alias)
        response = command_run_response(output)
        _append_action_record(self, action_record(frame.command_text, output, response))
        run_state = getattr(self._workflow_tool_agent, "run_state", None)
        if isinstance(run_state, dict):
            # A folded call that suspended never reached the step that reads this.
            run_state.pop("folded_command", None)
        return run_observation(
            alias, frame.ran_in_clause, response, folded_note=frame.folded_note,
            context_changed=now_clause != frame.ran_in_clause, now_clause=now_clause)

    def _nested_refusal(self, qualified_name: str, message: str) -> fastworkflow.CommandOutput:
        """A planning-command call that did not run: a failed output and no frame."""
        return fastworkflow.CommandOutput(
            command_response=fastworkflow.CommandResponse(response=message, success=False),
            command_name=qualified_name,
        )

    def _pop_frame(self, frame: PlanFrame) -> None:
        """Remove *frame* from the stack when it is on top, which releases its agent and phases."""
        frames = self._frame_stack()
        if frames and frames[-1] is frame:
            frames.pop()

    def _archived_command_runs(self, *, from_step: Optional[int] = None,
                               to_step: Optional[int] = None) -> list[dict[str, Any]]:
        """The command runs archived for this turn, in run order (``offload_command_runs``);
        only steps ``from_step``..``to_step`` (inclusive) when given, filtered in SQL. Runs
        inside the top frame's child step ranges are left out (design 3.5)."""
        from fastworkflow.observation_offloading.state import (
            archive_for_path, observability_db_path, scope_for_host)

        runs = archive_for_path(observability_db_path(self)).command_runs(
            scope_for_host(self), from_step=from_step, to_step=to_step)
        return runs_outside_ranges(runs, self._top_frame().child_step_ranges)

    def _phase_command_names(self) -> dict[str, list[str]]:
        """``commands`` / ``core_commands`` a phase's steps are read with, as the finish check
        reads them. Without the routing definition only the runs' and the facts' commands
        are known (``planning_phases.fill_output`` adds those)."""
        from fastworkflow.workflow_agent import core_command_names

        return {"commands": self._workflow_command_names(),
                "core_commands": list(core_command_names())}

    def _workflow_command_names(self) -> list[str]:
        """The app workflow's command names a step may name: the routed commands and the
        planning commands (which routing does not hold, and which a step may call, 3.5).
        Empty when the names cannot be read: the steps are then read with what is known."""
        if self._app_workflow is None:
            return []
        folderpath = self._app_workflow.folderpath
        names: list[str] = []
        try:
            routing = fastworkflow.RoutingRegistry.get_definition(folderpath)
            names += list(routing.routing_definition_map)
            # Framework execution commands (review_proposal) are not agent-visible, but a plan step may
            # call them by exact name: the framework's own review steps.
            names += [name for name, meta in routing.command_directory.map_command_2_metadata.items()
                      if meta.framework and not meta.is_planning]
        except Exception:  # noqa: BLE001 - read the steps with what is known
            pass
        names += list(self._planning_commands())
        return names

    def _planning_commands(self) -> Mapping[str, Any]:
        """The app workflow's loaded planning commands (qualified name -> ``PlanningCommandCommand``),
        or {} when they cannot be read: a step then reads as before, with the routed commands."""
        if self._app_workflow is None:
            return {}
        try:
            return planning_commands.resolve_field_sources(self._app_workflow.folderpath).planning_commands
        except Exception:  # noqa: BLE001 - read the steps with what is known
            return {}

    def _current_record(self) -> tuple[str, str]:
        """``(clause, context name)`` of the record the workflow is on now (``""`` at the root)."""
        from fastworkflow.context_identity import context_clause_for, context_identity

        active = self.get_active_workflow() or self._app_workflow
        if active is None:
            return "", ""
        return context_clause_for(active), context_identity(active)[0]

    def _run_planning_command_phase(self, command, phases, first_step: int, *,
                         resume_observation: Optional[str] = None,
                         resume_completes_step: bool = False,
                         skip: bool = False, skip_error: Optional[str] = None):
        """Run the current phase (of a fan-out phase: its current item) as one sub-plan
        and fill its output.

        Returns ``("done", None)``, ``("suspended", result)`` or ``("failed", message)``.
        With *skip* the phase's callable ``when`` was false (or, with *skip_error*, could
        not be decided): only its span is written (``skipped``), nothing runs, and the
        result is ``("done", None)`` (``("failed", skip_error)``).
        One ``fw.skill.phase`` span per run (a resumed run opens a new one). An
        answer-only phase (``planning_phases.is_answer_only``) is not an executor run: one
        model call with no tools writes its answer (``planning_phases.write_answer``).
        """
        from fastworkflow import planning_phases
        from fastworkflow.planning_commands import number
        from fastworkflow.observation_offloading.facts import FactsError, fact_sources
        from fastworkflow.workflow_agent import _what_can_i_do

        state = self._phase_run
        index = state["index"]
        phase = phases[index]
        last = index == len(phases) - 1
        uses_result = state.get("uses_result")
        phase_names = frozenset(state["phases"])
        # A call that runs one phase (call_phases: one phase, whatever its output): its
        # run gets the executor's own step cap, not PHASE_MAX_STEPS.
        # A call that runs one phase has the agent's own cap; a call of several gives each phase the phase cap.
        budget = (getattr(self._workflow_tool_agent, "max_iters", None)
                  if len(phases) == 1 else None) or planning_phases.phase_step_budget()
        resumed = resume_observation is not None
        attributes = {
            "skill": state["skill"],
            "phase": phase.name,
            "phase_index": index,
            "phase_count": len(phases),
            "step_budget": budget,
            "first_step": first_step,
            "resumed": resumed,
        }
        item = None
        others: list[Any] = []
        if phase.fans_out and not skip:
            items = state["items"]
            k = state["item_index"]
            item, others = items[k], items[:k] + items[k + 1:]
            attributes.update({"item": item, "item_index": k, "item_count": len(items),
                               "items_total": state["items_total"]})
            last_item = k == len(items) - 1
        else:
            last_item = True
        span = tracing.start_span(self, tracing.SPAN_PLANNING_COMMAND_PHASE, attributes=attributes)
        if skip:
            if skip_error is not None:
                tracing.end_span(self, span, status=tracing.STATUS_ERROR,
                                 attributes={"skipped": False, "facts_error": skip_error})
                return "failed", skip_error
            tracing.end_span(self, span, attributes={"skipped": True})
            return "done", None
        names = self._phase_command_names()
        if phase.output is not None:
            names["commands"] = names["commands"] + [
                source.command for source in fact_sources(phase.output)]
        if not resumed:
            outputs = self._phase_outputs(phases, index)
            try:
                steps = (planning_phases.phase_steps(phase, outputs, item, phase_names=phase_names)
                         if phase.fans_out
                         else planning_phases.phase_steps(phase, outputs, phase_names=phase_names))
            except planning_phases.PhaseError as error:
                tracing.end_span(self, span, status=tracing.STATUS_ERROR,
                                 attributes={"steps": "", "facts_error": str(error)})
                return "failed", str(error)
            # Code may run the phase's own steps only: never the planner's steps_before
            # (their phase has none of its own) or steps_after (after them).
            own_steps = 0 if phase.name == state.get("before_phase") else len(steps)
            steps = steps + (state["steps_after"] if last and last_item else [])
            state["first_step"] = first_step
            state["plan"] = number(steps)
        plan_text = state["plan"]
        answer_only = not resumed and planning_phases.is_answer_only(plan_text, **names)
        if answer_only and last and planning_phases.answer_by_code(plan_text):
            # 3.16 (b): a last phase of 'say that' lines the load check could not read (its template
            # did not build). Refused as the load check refuses it, as this phase's failure.
            message = planning_phases.say_that_last_message(command.path, phase.name)
            tracing.end_span(self, span, status=tracing.STATUS_ERROR,
                             attributes={"steps": plan_text, "answer_only": True,
                                         "facts_error": message})
            return "failed", message
        planned = sorted(planning_phases.planned_steps(plan_text, **names))
        item_frame = (state["item_index"], len(state["items"]), item) if phase.fans_out else None
        try:
            if answer_only:
                # Steps that name no command: one model call, with no tools, writes the
                # answer from the facts the steps carry (none when every step is verbatim
                # or a guard: code writes it, planning_phases.answer_by_code). Nothing can run a command.
                # A long list the steps do not tell the answer to name is cut short in the
                # prompt (planning_phases.capped_steps); a list to name is given whole.
                earlier_outputs = self._phase_outputs(phases, index)
                prompt_steps = planning_phases.capped_steps(plan_text, earlier_outputs)
                user_query = planning_phases.phase_query(
                    state["query"], prompt_steps, index=index, count=len(phases),
                    name=phase.name, answer_only=True, item=item_frame)
                if not last and planning_phases.answer_by_code(prompt_steps):
                    # 3.16 (a): a phase before the last made only of 'say that' lines writes no
                    # answer; its text reaches the answer only through __result__.
                    tracing.end_span(self, span, attributes={"skipped": True, "answer_only": True})
                    return "done", None
                if uses_result and not planning_phases.answer_by_code(prompt_steps):
                    # __result__ may return None, and then the model writes this answer: it is
                    # asked then, not now (WorkflowExecutionContext._settled_phase_answer).
                    marker = self._defer_phase_answer(index, user_query, prompt_steps)
                    tracing.end_span(self, span, attributes={"skipped": True, "answer_only": True,
                                                             "uses_result": True})
                    self._record_phase_answer(phase, marker)
                    return "done", None
                result = self._call_agent_with_retry(
                    lambda: dspy.Prediction(
                        final_answer=planning_phases.answer_only_answer(
                            user_query, prompt_steps, earlier_outputs),
                        suspended=False, exhausted=False),
                    trace_input=user_query,
                )
            else:
                # The finish check reads this phase's steps against what this phase ran.
                self._initial_plan_text = plan_text
                self._attach_finish_check(
                    state["first_step"], values=True,
                    item_binding=(item, others, planned) if phase.fans_out and phase.bind
                    else None)
                if resumed:
                    result = self._call_agent_resume(
                        resume_observation, completes_step=resume_completes_step)
                    # The code that ran the phase's leading steps before the suspension is
                    # still the run's: its count and where it stopped are the phase's.
                    code = getattr(self._workflow_tool_agent, "last_code", None)
                else:
                    # Code runs the leading mechanical steps first (planning_phases.CodeSteps).
                    code = self._phase_code_steps(
                        plan_text, own_steps, names,
                        answers=last or phase.name == state.get("extra_phase"))
                    # A phase other than the last sees only its own steps, never the request.
                    # A phased command with __result__ gets its answer from __result__ (or, when that
                    # returns None, from the deferred model call): its last phase's agent only runs steps.
                    result_driven = bool(uses_result and last and not state.get("flat"))
                    user_query = planning_phases.phase_query(
                        state["query"], plan_text, index=index, count=len(phases),
                        name=phase.name, item=item_frame,
                        code_steps=code is not None and code.runs_any,
                        result_driven=result_driven)
                    if uses_result and last and last_item:
                        notes = list(state.get("notes") or [])
                        if notes:
                            user_query = f"{user_query}\n\n{' '.join(notes)}"
                    if code is not None and last:
                        # Code may run every step: the agent then answers from their outputs.
                        code.done_inputs = {"user_query": planning_phases.phase_query(
                            state["query"], plan_text, index=index, count=len(phases),
                            name=phase.name, item=item_frame, code_ran_all=True,
                            result_driven=result_driven)}
                    available_commands = _what_can_i_do(self)

                    def run_phase():
                        self._workflow_tool_agent.code_steps = code
                        return self._workflow_tool_agent(
                            user_query=user_query,
                            available_commands=available_commands,
                            first_step=first_step,
                            max_iters=budget,
                        )

                    result = self._call_agent_with_retry(run_phase, trace_input=user_query)
        except BaseException as exc:
            tracing.end_span(self, span, status=tracing.status_for_dispatch_exception(exc),
                             attributes={"steps": plan_text, "answer_only": answer_only,
                                         "error_type": type(exc).__name__})
            raise
        by_code = {}
        if not answer_only and (not resumed or code is not None):
            by_code = {"steps_by_code": code.count if code is not None else 0,
                       "code_stop": (code.stop if code is not None else None) or "off"}
        if answer_only:
            steps_used = 0
            # Whether code wrote the answer from the steps, with no model call (fix-hia1.30).
            by_code = {"answer_by_code": planning_phases.answer_by_code(prompt_steps)}
        else:
            trajectory = getattr(self._workflow_tool_agent, "trajectory", None) or {}
            indices = [int(key.rsplit("_", 1)[1]) for key in trajectory
                       if key.rsplit("_", 1)[-1].isdigit()]
            steps_used = len({key for key in trajectory if key.startswith("thought_")
                              and int(key.rsplit("_", 1)[1]) >= state["first_step"]})
            if getattr(result, "suspended", None) is True:
                tracing.end_span(self, span, status=tracing.STATUS_CANCELLED,
                                 attributes={"steps": plan_text, "steps_used": steps_used,
                                             "answer_only": False, "suspended": True,
                                             **by_code})
                return "suspended", result
            if indices:
                # Never backwards: a nested frame this phase ran (run_planning_command) may be past it.
                self._next_step = max(self._next_step, max(indices) + 1)
        if uses_result and last and not state.get("flat") and not answer_only:
            # 3.16: the agent wrote no answer; the answer is __result__'s, or the deferred model call's.
            answer_query = planning_phases.phase_query(
                state["query"], plan_text, index=index, count=len(phases), name=phase.name,
                answer_only=True, item=item_frame)
            answer = self._defer_phase_answer(index, answer_query, plan_text)
        else:
            answer = planning_phases.clean_answer(getattr(result, "final_answer", "") or "")
        self._record_phase_answer(phase, answer)
        exhausted = getattr(result, "exhausted", None) is True
        if exhausted:
            # The call (run_planning_command) reports it as the frame's failure.
            state["exhausted"] = True
        closing = {"steps": plan_text, "steps_used": steps_used, "exhausted": exhausted,
                   "suspended": False, "answer_only": answer_only, **by_code}
        record = None
        facts = None
        issues = None
        need_runs = phase.output is not None or (phase.fans_out and phase.bind)
        try:
            ended_on, record_context = self._current_record() if need_runs else (None, None)
            record = ended_on
            runs = (self._archived_command_runs(from_step=state["first_step"])
                    if need_runs else [])
            earlier = (self._archived_command_runs(to_step=state["first_step"] - 1)
                       if phase.output is not None else [])
            if phase.fans_out and phase.bind:
                # The item is bound to the record that names it, wherever the run ended.
                issues = planning_phases.item_record_issues(runs, item, others, ended_on, planned)
                record = planning_phases.item_record(runs, item, ended_on, planned=planned)
                if record != ended_on:
                    record_context = None
            elif phase.output is not None and not phase.fans_out:
                # The phase did its steps on one record, then moved off it: fill on
                # that record (fix-hia1.10), not on the clause the run ended on.
                rebound = planning_phases.planned_record(
                    runs, plan_text, ended_on=ended_on, **names)
                if rebound is not None:
                    record, record_context = rebound, None
            if phase.output is not None:
                if phase.fans_out and phase.bind and record is None:
                    raise FactsError(f"item {item} was never opened: no command ran on a "
                                     "record that names it")
                # Only runs that do one of the phase's own steps count; an opening step
                # the workflow was already on is done by the run that entered the record.
                filled = planning_phases.fill_output(
                    phase.output, runs, record=record, record_context=record_context,
                    plan_text=plan_text, earlier_runs=earlier, ended_on=ended_on, **names)
                facts = filled.model_dump(mode="json")
        except Exception as error:  # noqa: BLE001 - the phase fails, saying why
            from pydantic import ValidationError

            message = (str(error) if isinstance(error, (FactsError, ValidationError)) else
                       f"its command runs could not be read ({type(error).__name__}: {error})")
            if phase.fans_out:
                message = f"item {item}: {message}"
            tracing.end_span(self, span, status=tracing.STATUS_ERROR, attributes={
                **closing, "record": record, "record_issues": issues, "facts_error": message})
            return "failed", message
        if phase.output is not None:
            if phase.fans_out:
                state["outputs"][phase.name].append(facts)
            else:
                state["outputs"][phase.name] = facts
        if phase.fans_out and phase.until is not None:
            # The stop condition reads the items' outputs so far (typed) and earlier outputs.
            from fastworkflow.typed_phase_facts import validate_phase_output_value

            so_far = ([validate_phase_output_value(phase.output, v)
                       for v in state["outputs"][phase.name]]
                      if phase.output is not None else [])
            try:
                state["until_met"] = planning_phases.stops_after(
                    phase, so_far, self._phase_outputs(phases, index),
                    phase_names=phase_names)
            except planning_phases.PhaseError as error:
                message = f"item {item}: {error}"
                tracing.end_span(self, span, status=tracing.STATUS_ERROR, attributes={
                    **closing, "record": record, "record_issues": issues, "facts": facts,
                    "facts_error": message})
                return "failed", message
            closing["stop_condition_met"] = state["until_met"]
        tracing.end_span(self, span, attributes={
            **closing, "record": record, "record_issues": issues, "facts": facts})
        return "done", None

    def _phase_code_steps(self, plan_text: str, own_steps: int, names: dict, *,
                          answers: bool):
        """The steps code runs at the start of this phase run (``planning_phases.CodeSteps``),
        or None: the run has no steps of its own (planner-written steps only), or nothing
        could be read (never a reason to fail the phase; the agent then runs every step)."""
        from fastworkflow import planning_phases

        if not own_steps or self._app_workflow is None:
            return None
        try:
            state = getattr(self, "_phase_run", None) or {}
            call_first = state.get("call_first_step", state.get("first_step", 0))
            return planning_phases.CodeSteps(
                plan_text, workflow=lambda: self.get_active_workflow() or self._app_workflow,
                runs=lambda index: self._archived_command_runs(from_step=index,
                                                                to_step=index),
                call_runs=lambda: self._archived_command_runs(from_step=call_first),
                own_steps=own_steps, answers=answers, planning=self._planning_commands(),
                **names)
        except Exception as error:  # noqa: BLE001 - the agent runs the steps instead
            logger.warning("code-run steps skipped: %s: %s", type(error).__name__, error)
            return None

    def _run_subtasks_from(self, start: int):
        """Run sub-tasks start.. one after another; the combined answer, or a suspension.

        Each sub-task's ReAct steps continue numbering from the previous one so
        archive aliases stay unique across the turn. A run that hits the step cap
        is continued by a fresh executor, inserted as the next entry (see _hand_off).
        """
        k = start
        # A while loop: a hand-off inserts an entry after the current one, so the end moves.
        while k < len(self._subtasks):
            self._subtask_index = k
            origin = self._origins[k]
            # A hand-off continuation keeps the stop/abort decision of the run before it.
            if k == 0 or self._origins[k - 1] != origin:
                self._enable_all_tools()
            query = self._subtasks[k]
            floors = getattr(self, "_evidence_floors", None)
            if not isinstance(floors, dict):
                floors = self._evidence_floors = {}
            floors.setdefault(origin, self._next_step)
            first = k == 0 or self._origins[k - 1] != origin
            outputs_before = len(self._turn_outputs)
            result = self._run_entry(k, origin, query, first=first, floor=floors[origin])
            if getattr(result, "suspended", None) is True:
                return result
            self._record_subtask_answer(result, self._typed_output_since(outputs_before))
            if getattr(result, "exhausted", None) is True:
                self._hand_off(k)
            k += 1

        return dspy.Prediction(final_answer=self._combined_answer(),
                               exhausted=self._subtask_exhausted)

    def _bound_item(self, origin: int):
        """The reviewed sub-task (SubTask: goal, inputs, outcome, means) that origin *origin* runs, or None when the turn
        has no plan for it (a turn that was not split under a plan, or a fast-path turn)."""
        plan = self._turn_subtask_plan
        if plan is None or not 0 <= origin < len(plan.items):
            return None
        return plan.items[origin]

    def _answers_by_number(self) -> dict[int, str]:
        """Each sub-task that has run, by its 1-based number, with the answer its run gave (the last of its runs).

        The hand-off's 'from sub-task N' inputs take these: the run's typed Output (JSON) when it produced one, else the
        text its run returned (the command's answer for a planning command, the agent's final answer otherwise)."""
        answers: dict[int, str] = {}
        typed = list(self._subtask_outputs)
        typed += [""] * (len(self._subtask_answers) - len(typed))
        for origin, answer, output in zip(self._origins, self._subtask_answers, typed):
            answers[origin + 1] = output or answer
        return answers

    def _typed_output_since(self, start: int) -> str:
        """The typed Output of the last successful command run since turn output *start*, as JSON; "" when none."""
        for command_output in reversed(self._turn_outputs[start:]):
            response = command_output.command_response
            if response.success and response.output is not None:
                dump = getattr(response.output, "model_dump_json", None)
                return dump() if callable(dump) else json.dumps(response.output, default=str)
        return ""

    def _run_entry(self, k: int, origin: int, query: str, *, first: bool, floor: int):
        """Run entry *k* of the turn: the bound means of its goal when the turn has a plan, else the agent as today.

        A planning means runs its phases with its inputs (through the phase runner), its first entry only. A plain
        means and a from-scratch goal run the agent with the goal's text (its inputs and desired outcome in it); a plain
        means narrows the executor's listing to its command, and the plan of the run names that command, so the
        finish check holds the goal to it. Nothing here marks a goal complete: the finish check does (arch 19.0 inv. 3).
        """
        from fastworkflow import review_proposals

        item = self._bound_item(origin)
        if item is None:
            return self._call_agent_for_query(query, query, first_step=self._next_step,
                                              template_text=self._original_text(origin), evidence_floor=floor)
        resolved = review_proposals.resolve_inputs(item, self._answers_by_number())
        if resolved.missing:
            numbers = ", ".join(str(number) for number in resolved.missing)
            return dspy.Prediction(final_answer=f"This sub-task needs the output of sub-task {numbers}, which has not run.",
                                   exhausted=False)
        if first:
            query = review_proposals.run_text(item, resolved)
        else:
            query = self._continued_text(query, item, resolved)
        if first and item.means.kind == review_proposals.MEANS_PLANNING:
            return self._run_planning_means(item, query, origin, floor=floor)
        bound = item.means.name if item.means.kind == review_proposals.MEANS_PLAIN else None
        self._narrow_to = bound
        try:
            return self._call_agent_for_query(
                query, query, first_step=self._next_step, template_text=self._original_text(origin),
                evidence_floor=floor, bound_command=bound)
        finally:
            self._narrow_to = None

    @staticmethod
    def _continued_text(text: str, item, resolved) -> str:
        """A hand-off continuation's text with the sub-task's inputs and outcome in its first line."""
        from fastworkflow import review_proposals

        head, separator, tail = text.partition("\n")
        full = review_proposals.run_text(item, resolved)
        suffix = full[len(item.goal):]
        return f"{head}{suffix}{separator}{tail}" if suffix else text

    def _run_planning_means(self, item, query: str, origin: int, *, floor: int):
        """A sub-task bound to a planning command: its phases run with its inputs (the phase runner, as a planned call's
        phases do; no model plans them). Its arguments are checked as a planned command's are (check_planned_arguments);
        a refusal, or no command, falls back to the agent with the refusal as a note and the bound command's listing."""
        from fastworkflow import planning_commands, review_proposals, run_listing

        folder = self._app_workflow.folderpath
        name = item.means.name
        command = planning_commands.planning_command(folder, name)
        refusal = None
        if command is None:
            refusal = f"{name} could not be loaded"
        else:
            resolved = review_proposals.resolve_inputs(item, self._answers_by_number())
            checked = run_listing.check_planned_arguments(command, dict(resolved.named), query,
                                                          earlier_outputs=list(self._subtask_answers))
            refusal = checked.refusal
            if refusal is None:
                try:
                    phases = planning_commands.call_phases(command, checked.args)
                except planning_commands.PlanningCommandCallError as error:
                    refusal = str(error)
        if refusal is not None:
            return self._call_agent_for_query(
                f"{query}\n\nThe planning command {name} cannot run as planned: {refusal}.",
                query, first_step=self._next_step, template_text=self._original_text(origin),
                evidence_floor=floor, bound_command=name)
        call = planning_commands.PlanningCommandCallPlan(
            plan="", args=checked.args, notes=list(checked.notes), prediction=None, phases=phases)
        return self._start_planning_command_phases(query, command, call, first_step=self._next_step)

    def _original_text(self, origin: int) -> str:
        """The sub-task text an origin was split out as (its first entry, before any hand-off)."""
        return self._subtasks[self._origins.index(origin)]

    def _hand_off(self, k: int) -> None:
        """Queue a fresh-executor continuation of sub-task k, which hit the step cap.

        The progress check reads the agent's current trajectory, so it runs before
        the next run replaces that trajectory. At most MAX_HANDOFFS continuations per sub-task.
        """
        turn = self._turn_frame()
        origin = turn.origins[k]
        if turn.origins.count(origin) > MAX_HANDOFFS:
            # No continuation left: the capped answer stands as partial and the turn is exhausted.
            turn.subtask_exhausted = True
            return
        from fastworkflow.workflow_agent import run_progress_check

        done = run_progress_check(self, "step_limit", done_only=True).strip()
        opened = _opened_part(getattr(turn.agent, "trajectory", None) or {})
        logger.info("sub-task %d hit the step limit; handing off to a fresh executor (hand-off %d)",
                    origin + 1, turn.origins.count(origin))
        turn.subtasks.insert(k + 1, (
            f"{self._original_text(origin)}\n\n"
            "Already done in this sub-task (do not repeat it; observation ids such as O12 can be looked up):\n"
            f"{done}"
            + (f"\n\n{opened}" if opened else "")))
        turn.origins.insert(k + 1, origin)
        # This capped run's answer is kept: _combined_answer joins it with the continuation's.

    def _combined_answer(self) -> str:
        """Answers grouped under their original sub-task; a single-origin turn is its answers with no heading.

        A sub-task that was handed off has one answer per run, kept in order and labelled as parts.
        """
        answers: dict[int, list[str]] = {}
        for origin, answer in zip(self._origins, self._subtask_answers):
            if answer:
                answers.setdefault(origin, []).append(answer)
        joined = {origin: _join_parts(parts) for origin, parts in answers.items()}
        if len(joined) == 1:
            return next(iter(joined.values()))
        return "\n\n".join(
            f"### {origin + 1}. {self._original_text(origin)}\n{text}"
            for origin, text in joined.items()
        )

    def _record_subtask_answer(self, result, typed: str = "") -> None:
        outputs = self._subtask_outputs
        # Padded first: a blob written before the typed outputs were kept has answers and no outputs.
        outputs.extend([""] * (len(self._subtask_answers) - len(outputs)))
        self._subtask_answers.append(result.final_answer)
        outputs.append(typed)
        trajectory = getattr(self._workflow_tool_agent, "trajectory", None) or {}
        steps = [int(key.rsplit("_", 1)[1]) for key in trajectory
                 if key.rsplit("_", 1)[-1].isdigit()]
        if steps:
            # Never backwards: a nested frame's steps (run_planning_command) may be past it.
            self._next_step = max(self._next_step, max(steps) + 1)

    def _call_agent_resume(self, observation: str, *, completes_step: bool = False):
        def resume():
            # A plain answer is resumed as it always was; the flag is passed only for a nested run.
            if completes_step:
                return self._workflow_tool_agent.resume(observation, completes_step=True)
            return self._workflow_tool_agent.resume(observation)

        return self._call_agent_with_retry(resume, trace_input=observation, resumed=True)

    def _resume_top_frame(self, observation: str, *, completes_step: bool = False):
        """Resume the top frame's suspended run with *observation*: its phase run, or its agent.

        *completes_step*: the observation is the run of a nested command the step waited on,
        not a user's answer (``fastWorkflowReAct.resume``)."""
        if self._phase_run is not None:
            # A phase of a phased planning command call was suspended: resume it, then run the rest.
            return self._run_planning_command_phases(resume_observation=observation,
                                          resume_completes_step=completes_step)
        return self._call_agent_resume(observation, completes_step=completes_step)

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
        if "[[" in result_text:
            # A DSPy field marker ("[[ ## completed ## ]]") a reply carried into its text.
            from fastworkflow.planning_phases import clean_answer
            result_text = clean_answer(result_text)

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
        return self._settle_agent_result(message, agent_result)

    def _settle_agent_result(self, message: str, agent_result) -> fastworkflow.CommandOutput:
        """The turn's output for one agent run: its suspension (the turn awaits the user), or its answer."""
        self._turn_agent_result = agent_result
        if getattr(agent_result, "suspended", None) is True:
            self._awaiting_user = True
            self._suspended_user_message = message
            self._pending_clarification_request = agent_result.clarification
            self._note_agent_suspension(agent_result.clarification)
            return self._awaiting_user_output(agent_result.clarification)
        self._reset_agent_suspension()
        return self._finalize_agent_output(message, agent_result)

    def _rerun_after_planning_command_refusal(self, reply: str) -> fastworkflow.CommandOutput:
        """The reply to a refused planning command call's question: the turn runs once more from its request,
        with the reply after it, so the planner's typed call sees the value the user gave (its
        guards read the request). The question's exchange is completed here. Nothing ran before
        the question (``_planning_command_refusal_may_ask``), so nothing is run twice."""
        message = f"{self._suspended_user_message}\n{reply}"
        self._note_agent_resume()
        self.complete_ask_user_entry(reply)
        self._planning_command_refusal_pending = False
        self._planning_command_refusal_answered = True
        self._reset_agent_suspension()
        return self._settle_agent_result(message, self._run_agent(message))

    def _approval_state_payload(self) -> Optional[dict]:
        store = approval_binding.store_for(self._app_workflow)
        if store is None or not store.records and not store.turn_seq:
            return None
        return store.to_state()

    def _note_approval_inbound(self, surface, token: Optional[str], unattended: bool) -> None:
        """Record the surface, token and turn kind the approval gate reads for this message."""
        store = approval_binding.store_for(self._app_workflow)
        if store is None:
            return
        store.surface = surface or approval_binding.CHANNEL_SURFACE
        store.inbound_token = token
        store.channel = str(self._channel_id or "")
        store.task = str(self._session_key or self._channel_id or "")

    def _resume_approval_direct(self, message: str) -> fastworkflow.CommandOutput:
        """The reply to an approval raised outside the agent's loop (an MCP tool call), or a
        /cancel_pending for a suspended one: the record resolves here, and its outcome is this
        turn's output."""
        awaiting = approval_binding.active_record(self._app_workflow)
        was_suspended = awaiting is not None and awaiting.suspended_agent
        resolution = approval_binding.resolve_reply(self, self._app_workflow, message)
        if not resolution.done:
            self._awaiting_user = True
            return self._awaiting_user_output(resolution.text)
        self._awaiting_user = False
        if was_suspended:
            self._reset_agent_suspension()
        output = resolution.output
        if output is None:
            output = fastworkflow.CommandOutput(
                command_response=fastworkflow.CommandResponse(response=resolution.text))
        self.append_turn_output(output)
        return output

    def _approval_text_output(self, text: str) -> fastworkflow.CommandOutput:
        """A plain text output for the user's approval management messages (never agent-visible)."""
        output = fastworkflow.CommandOutput(command_response=fastworkflow.CommandResponse(response=text))
        if self._app_workflow is not None:
            output.workflow_name = self._app_workflow.folderpath.split("/")[-1]
        self.append_turn_output(output)
        return output

    def cancel_pending_approvals(self) -> int:
        """Invalidate every awaiting approval on this session (session end, eviction, /cancel_pending).

        A suspended approval also abandons the agent's suspended tool call, so a later message is
        never taken as its answer. Returns how many records were cancelled.
        """
        if self._app_workflow is None:
            return 0
        awaiting = approval_binding.active_record(self._app_workflow)
        cancelled = approval_binding.cancel_all(self, self._app_workflow, reason="cancelled")
        if awaiting is not None and awaiting.suspended_agent and self._awaiting_user:
            self._reset_agent_suspension()
        return cancelled

    def _hold_resumed_approval_run(self, output: fastworkflow.CommandOutput) -> None:
        """Hold the run an approval reply executed under the step it suspended (fix-3rsq.approval-finish-check).

        The approved call runs through approval_binding._execute_stored (perform_action), not the
        agent's execute path, so nothing holds its run for the step's archive row. Without that row
        the finish check reads the step as not run and sends the agent back to run it again. The
        step is the suspended agent's in-flight execute step: the nested frame's agent when the
        suspension is inside a planning command, else the turn's agent. Never raises.
        """
        from fastworkflow.workflow_agent import _held_context, _hold_command_run

        try:
            agent = self._top_frame().agent or self._workflow_tool_agent
            alias = current_execute_alias(agent)
            if alias is None:
                return
            clause, name = _held_context(self, output, *self._current_record())
            _hold_command_run(self, output, clause, name, alias=alias)
        except Exception:  # noqa: BLE001 - holding a record must not fail the approved command
            logger.debug("could not hold the approved run for its archive row", exc_info=True)

    def _resume_agent_message(self, user_answer: str) -> fastworkflow.CommandOutput:
        self._ensure_agent_initialized()
        if self._planning_command_refusal_pending:
            return self._rerun_after_planning_command_refusal(user_answer)
        # Catch awaiting_user/_suspended desync before any LLM work: resume()
        # consumes _suspended at entry, so a second request (or a restore that
        # lost the react blob) must not become a bare 500.
        agent = self._workflow_tool_agent
        if agent is None or agent.export_suspended() is None:
            raise NoSuspendedAgentStateError(
                "No suspended ReAct state to resume"
            )
        awaiting = approval_binding.active_record(self._app_workflow)
        approval_resumed = awaiting is not None and awaiting.suspended_agent
        if approval_resumed:
            # The suspended tool call is an approval (ADR-EXP-025): the reply resolves it, and
            # the resolution is the observation. Nothing else reads the reply.
            resolution = approval_binding.resolve_reply(self, self._app_workflow, user_answer)
            if not resolution.done:
                return self._awaiting_user_output(resolution.text)
            self._note_agent_resume()
            self.complete_ask_user_entry(user_answer)
            if resolution.output is None and getattr(self._turn_frame(), "subtask_review_open", False):
                # Declined or expired: the gate's review ends, and nothing runs (fix-3rsq.plan-review.2.6).
                return self._end_subtask_review_unrun(awaiting.state)
            if resolution.output is not None:
                self.append_turn_output(resolution.output)
                self._hold_resumed_approval_run(resolution.output)
            self._workflow_tool_agent.iteration_counter = -1
            observation = resolution.text
        else:
            self._note_agent_resume()

            from fastworkflow.workflow_agent import _post_ask_user_response

            self._workflow_tool_agent.iteration_counter = -1
            observation = _post_ask_user_response(
                self._pending_clarification_request,
                user_answer,
                self,
            )
        # An approval's observation is the approved command's own run: the step completes as one
        # that returned normally, so its completion hook archives the run held above.
        agent_result = self._resume_top_frame(observation, completes_step=approval_resumed)
        # The suspension may be nested: each call that finishes hands its run to the step it
        # was called from, which resumes next, until the turn frame is on top again.
        while len(self._frame_stack()) > 1 and getattr(agent_result, "suspended", None) is not True:
            if getattr(self._turn_frame(), "framework_entry", False) and len(self._frame_stack()) == 2:
                if getattr(self._turn_frame(), "subtask_review_open", False):
                    # The gate's review was accepted: the turn runs its sub-tasks (fix-3rsq.plan-review.2.6).
                    return self._settle_agent_result(self._turn_user_message, self._accept_subtask_review(agent_result))
                return self._finish_framework_call(agent_result)
            observation = self._finish_nested_frame(agent_result)
            agent_result = self._resume_top_frame(observation, completes_step=True)
        if self._subtasks and getattr(agent_result, "suspended", None) is not True:
            self._record_subtask_answer(agent_result)
            if getattr(agent_result, "exhausted", None) is True:
                self._hand_off(self._subtask_index)
            agent_result = self._run_subtasks_from(self._subtask_index + 1)
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
    # no fastWorkflow control flow reads a recorded context type or a
    # consequence class, which arch §17.3 requires.
    #
    # "Here" is in fact `tracing`, because workflow_agent.py
    # opens the same span from a third site and owes the same record. These two
    # methods stay as the WEC-shaped entry points — they supply the app-workflow
    # fallback that the free functions cannot know about — but the projection
    # itself is written once, for all three sites.

    def _context_before(
        self, span, workflow: Optional[fastworkflow.Workflow] = None
    ) -> Optional[str]:
        """The active context type before a command runs, or None.

        Gated on a span having actually opened, matching the existing
        attribute-prep rule at this seam: with observability off this must cost
        nothing.
        """
        return tracing.context_before(span, workflow or self._app_workflow)

    def _capture_attributes(
        self,
        span,
        command_output: fastworkflow.CommandOutput,
        context_before: Optional[str],
        workflow: Optional[fastworkflow.Workflow] = None,
        command_name: Optional[str] = None,
    ) -> dict:
        """Call-id, context type before/after and consequence for one command.

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
