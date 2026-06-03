"""
Transport-free, synchronous workflow execution core.

Embedders (e.g. FastAPI) should use one WorkflowExecutionContext per session:
bind_app_workflow once, call process_message per request in a worker thread or
asyncio task (ContextVar isolates active workflow per thread/task), set
ask_user_timeout when not using ChatSession queues, and close() on session end.

ChatSession composes this core for CLI/REPL (queues, ChatWorker, keep_alive).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from queue import Queue
from typing import Optional

import dspy

import fastworkflow
from fastworkflow import active_workflow
from fastworkflow.utils.logging import logger
from fastworkflow.utils import dspy_utils


class CommandCancelledError(BaseException):
    """
    Raised when a command cannot continue (e.g. ask_user timed out).

    Subclasses BaseException so fastWorkflowReAct's ``except Exception`` does not
    swallow it; process_message converts it to a failed CommandOutput.
    """


class WorkflowExecutionContext:
    """
    Owns NLU (cme_workflow), the bound app workflow, and message execution.

    No queues, threads, or session lifecycle — inject optional queues from
    ChatSession for trace/output/ask_user when running in CLI mode.
    """

    def __init__(
        self,
        run_as_agent: bool = False,
        ask_user_timeout: Optional[float] = None,
    ):
        self._run_as_agent = run_as_agent
        self._ask_user_timeout = ask_user_timeout
        self._app_workflow: Optional[fastworkflow.Workflow] = None
        self._keep_alive = False

        self._user_message_queue: Optional[Queue] = None
        self._command_output_queue: Optional[Queue] = None
        self._command_trace_queue: Optional[Queue] = None

        self._conversation_history: dspy.History = dspy.History(messages=[])

        from fastworkflow.command_executor import CommandExecutor
        self._CommandExecutor = CommandExecutor

        self._workflow_tool_agent = None
        self._intent_clarification_agent = None

        self._awaiting_user = False
        self._suspended_user_message: Optional[str] = None
        self._pending_clarification_request: Optional[str] = None

        self._cme_workflow = fastworkflow.Workflow.create(
            fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
            workflow_id_str=f"cme_{uuid.uuid4().hex}",
            workflow_context={
                "NLU_Pipeline_Stage": fastworkflow.NLUPipelineStage.INTENT_DETECTION,
            },
        )

        self.clear_conversation_history()

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
    def ask_user_timeout(self) -> Optional[float]:
        return self._ask_user_timeout

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
        """True when the agent suspended on ask_user and awaits the next process_message."""
        return self._awaiting_user

    def cancel_pending(self) -> bool:
        """
        Abort a pending ask_user clarification (Topology B).

        Returns True if a pending clarification was cleared, False otherwise.
        """
        if not self._awaiting_user:
            return False
        self._reset_agent_suspension()
        return True

    def clear_conversation_history(self) -> None:
        self._conversation_history = dspy.History(messages=[])

    def bind_app_workflow(self, workflow: fastworkflow.Workflow) -> None:
        """Bind the app workflow for NLU (Path 1) and execution (Path 2)."""
        self._app_workflow = workflow
        self._cme_workflow.context["app_workflow"] = workflow

    def close(self) -> bool:
        """
        Release the cme_workflow speedict session store.

        Call when an embedder session ends; does not close the app workflow
        (caller owns that lifecycle).
        """
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

    def process_message(self, message: str) -> fastworkflow.CommandOutput:
        """
        Execute one user message synchronously.

        Pushes app_workflow onto the contextvar stack for the duration of the
        call so CommandExecutor and agent tools resolve the correct workflow.
        """
        if self._app_workflow is None:
            raise RuntimeError(
                "No app workflow bound; call bind_app_workflow() before process_message()"
            )

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

    def process_action(self, action: fastworkflow.Action) -> fastworkflow.CommandOutput:
        if self._app_workflow is None:
            raise RuntimeError(
                "No app workflow bound; call bind_app_workflow() before process_action()"
            )

        self.push_active_workflow(self._app_workflow)
        try:
            return self._process_action(action)
        finally:
            self.pop_active_workflow()
            if self._app_workflow:
                self._app_workflow.flush()

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
        command_response = fastworkflow.CommandResponse(
            response=f"Command cancelled: {reason}",
            success=False,
        )
        command_output = fastworkflow.CommandOutput(
            command_responses=[command_response]
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

        from fastworkflow.workflow_agent import initialize_workflow_tool_agent
        self._workflow_tool_agent = initialize_workflow_tool_agent(self)

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

    def _call_agent_with_retry(self, agent_call):
        from dspy.utils.exceptions import AdapterParseError

        lm, agent_adapter = self._agent_dspy_context()
        max_retries = 2
        for attempt in range(max_retries):
            try:
                with dspy.context(lm=lm, adapter=agent_adapter):
                    return agent_call()
            except AdapterParseError:
                if attempt == max_retries - 1:
                    raise

    def _run_agent(self, message: str):
        """Fresh agent turn setup and ReAct forward call."""
        if os.path.exists("action.jsonl"):
            os.remove("action.jsonl")

        if self._app_workflow:
            self._app_workflow.context["raw_user_message"] = message

        refined_user_query = self._refine_user_query(message, self.conversation_history)

        from fastworkflow.workflow_agent import build_query_with_next_steps, _what_can_i_do

        command_info_and_refined_message_with_todolist = build_query_with_next_steps(
            refined_user_query,
            self,
        )
        available_commands = _what_can_i_do(self)

        return self._call_agent_with_retry(
            lambda: self._workflow_tool_agent(
                user_query=command_info_and_refined_message_with_todolist,
                available_commands=available_commands,
            )
        )

    def _call_agent_resume(self, observation: str):
        return self._call_agent_with_retry(
            lambda: self._workflow_tool_agent.resume(observation)
        )

    def _awaiting_user_output(self, clarification: str) -> fastworkflow.CommandOutput:
        command_response = fastworkflow.CommandResponse(response=clarification)
        command_response.artifacts["awaiting_user"] = True
        command_output = fastworkflow.CommandOutput(
            command_responses=[command_response]
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

        conversation_traces = None
        conversation_summary = original_message
        if os.path.exists("action.jsonl"):
            with open("action.jsonl", "r", encoding="utf-8") as f:
                actions = [json.loads(line) for line in f if line.strip()]
            conversation_summary, conversation_traces = self._extract_conversation_summary(
                original_message, actions, result_text
            )
            command_response.artifacts["conversation_summary"] = conversation_summary

        self.conversation_history.messages.append(
            {
                "conversation summary": conversation_summary,
                "conversation_traces": conversation_traces,
                "feedback": None,
            }
        )

        command_output = fastworkflow.CommandOutput(
            command_responses=[command_response]
        )
        if self._app_workflow:
            command_output.workflow_name = self._app_workflow.folderpath.split("/")[-1]

        self._maybe_enqueue_output(command_output)
        self._maybe_enqueue_trace_sentinel()

        return command_output

    def _process_agent_message(self, message: str) -> fastworkflow.CommandOutput:
        self._ensure_agent_initialized()
        agent_result = self._run_agent(message)
        if getattr(agent_result, "suspended", None) is True:
            self._awaiting_user = True
            self._suspended_user_message = message
            self._pending_clarification_request = agent_result.clarification
            return self._awaiting_user_output(agent_result.clarification)
        self._reset_agent_suspension()
        return self._finalize_agent_output(message, agent_result)

    def _resume_agent_message(self, user_answer: str) -> fastworkflow.CommandOutput:
        self._ensure_agent_initialized()

        from fastworkflow.workflow_agent import _post_ask_user_response

        self._workflow_tool_agent.iteration_counter = -1
        observation = _post_ask_user_response(
            self._pending_clarification_request,
            user_answer,
            self,
        )
        agent_result = self._call_agent_resume(observation)
        if getattr(agent_result, "suspended", None) is True:
            self._pending_clarification_request = agent_result.clarification
            return self._awaiting_user_output(agent_result.clarification)

        original_message = self._suspended_user_message
        self._reset_agent_suspension()
        return self._finalize_agent_output(original_message, agent_result)

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
                )
            )

        command_output = self._CommandExecutor.invoke_command(self, message)

        response_text = ""
        if command_output.command_responses:
            response_text = command_output.command_responses[0].response or ""

        params = command_output.command_parameters or {}
        if hasattr(params, "model_dump"):
            params_dict = params.model_dump()
        elif hasattr(params, "dict"):
            params_dict = params.dict()
        else:
            params_dict = params

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
                )
            )

        record = {
            "command": message,
            "command_name": command_output.command_name or "",
            "parameters": params_dict,
            "response": response_text,
        }

        self.conversation_history.messages.append(
            {
                "conversation summary": "assistant_mode_command",
                "conversation_traces": json.dumps(record),
                "feedback": None,
            }
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
                )
            )

        command_output = self._CommandExecutor.perform_action(workflow, action)

        response_text = ""
        if command_output.command_responses:
            response_text = command_output.command_responses[0].response or ""

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
                )
            )

        record = {
            "command": "process_action",
            "command_name": action.command_name,
            "parameters": params_dict,
            "response": response_text,
        }

        self.conversation_history.messages.append(
            {
                "conversation summary": "process_action command",
                "conversation_traces": json.dumps(record),
                "feedback": None,
            }
        )

        self._maybe_enqueue_output(command_output)
        self._maybe_enqueue_trace_sentinel()

        return command_output

    def _refine_user_query(
        self, user_query: str, conversation_history: dspy.History
    ) -> str:
        if conversation_history.messages:
            messages = []
            for conv_dict in conversation_history.messages[-5:]:
                messages.extend([f"{k}: {v}" for k, v in conv_dict.items()])
            messages.append(f"new_user_query: {user_query}")
            return "\n".join(messages)

        return user_query

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

        planner_lm = dspy_utils.get_lm("LLM_PLANNER", "LITELLM_API_KEY_PLANNER")
        with dspy.context(lm=planner_lm):
            cs_func = dspy.ChainOfThought(ConversationSummarySignature)
            prediction = cs_func(
                user_query=user_query,
                workflow_actions=workflow_actions,
                final_agent_response=final_agent_response,
            )
            return prediction.conversation_summary, json.dumps(conversation_traces)
