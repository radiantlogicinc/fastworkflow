from enum import Enum
from queue import Empty, Queue
from threading import Thread, Lock
from typing import ClassVar, Optional
from collections import deque
import json
import contextlib
import uuid
from pathlib import Path
import os
from datetime import datetime

import dspy

import fastworkflow
from fastworkflow.utils.logging import logger
from fastworkflow.utils import dspy_utils
from fastworkflow.model_pipeline_training import CommandRouter
from fastworkflow.utils.startup_progress import StartupProgress


class SessionStatus(Enum):
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"

class ChatWorker(Thread):
    def __init__(self, chat_session: "ChatSession"):
        super().__init__()
        self.chat_session = chat_session
        self.daemon = True
        
    def run(self):
        """Process messages for the root workflow"""
        try:
            self.chat_session._status = SessionStatus.RUNNING
            workflow = self.chat_session.get_active_workflow()
            logger.debug(f"Started root workflow {workflow.id}")
            
            # Run the workflow loop
            self.chat_session._run_workflow_loop()
            
        finally:
            self.chat_session._status = SessionStatus.STOPPED
            # Ensure workflow is popped if thread terminates unexpectedly
            if self.chat_session.get_active_workflow() is not None:
                self.chat_session.pop_active_workflow()

class ChatSession:
    def get_active_workflow(self) -> Optional[fastworkflow.Workflow]:
        """Get the currently active workflow (top of stack)"""
        with self._workflow_stack_lock:
            return self._workflow_stack[-1] if self._workflow_stack else None
    
    def push_active_workflow(self, workflow: fastworkflow.Workflow) -> None:
        """Push a workflow onto this session's stack"""
        with self._workflow_stack_lock:
            self._workflow_stack.append(workflow)
            logger.debug(f"Workflow stack: {[w.id for w in self._workflow_stack]}")
    
    def pop_active_workflow(self) -> Optional[fastworkflow.Workflow]:
        """Pop a workflow from this session's stack"""
        with self._workflow_stack_lock:
            if not self._workflow_stack:
                return None
            workflow = self._workflow_stack.pop()
            logger.debug(f"Workflow stack after pop: {[w.id for w in self._workflow_stack]}")
            return workflow

    def clear_workflow_stack(self) -> None:
        """Clear the entire workflow stack for this session"""
        with self._workflow_stack_lock:
            self._workflow_stack.clear()
            logger.debug("Workflow stack cleared")

    def stop_workflow(self) -> None:
        """
        Stop the current workflow and clear the workflow stack.
        This method is called when starting a new root workflow to ensure
        the previous workflow is properly stopped and resources are cleaned up.
        """
        # Set status to stopping to signal the workflow loop to exit
        self._status = SessionStatus.STOPPING
        
        # Wait for the chat worker thread to finish if it exists
        if self._chat_worker and self._chat_worker.is_alive():
            self._chat_worker.join(timeout=5.0)  # Wait up to 5 seconds
            if self._chat_worker.is_alive():
                logger.warning("Chat worker thread did not terminate within timeout")
        
        # Clear the workflow stack
        self.clear_workflow_stack()
        
        # Reset status to stopped
        self._status = SessionStatus.STOPPED
        
        # Clear current workflow reference
        self._current_workflow = None
        
        logger.debug("Workflow stopped and workflow stack cleared")

    def __init__(self, run_as_agent: bool = False):
        """
        Initialize a chat session.
        
        Args:
            run_as_agent: If True, use agent mode (DSPy-based tool selection).
                         If False (default), use traditional command execution.
        
        A chat session can run multiple workflows that share the same message queues.
        Use start_workflow() to start a specific workflow within this session.
        """
        # Create instance-level workflow stack (supports nested workflows within this session)
        self._workflow_stack: deque[fastworkflow.Workflow] = deque()
        self._workflow_stack_lock = Lock()
        
        # Create queues for user messages and command outputs
        self._user_message_queue = Queue()
        self._command_output_queue = Queue()
        self._command_trace_queue = Queue()
        self._status = SessionStatus.STOPPED
        self._chat_worker = None

        self._conversation_history: dspy.History = dspy.History(messages=[])
        
        # Import here to avoid circular imports
        from fastworkflow.command_executor import CommandExecutor
        self._CommandExecutor = CommandExecutor
        
        # Initialize workflow-related attributes that will be set in start_workflow
        self._current_workflow = None
        
        # Initialize agent-related attributes
        self._run_as_agent = run_as_agent
        self._workflow_tool_agent = None            
        
        # Create the command metadata extraction workflow with a unique ID
        self._cme_workflow = fastworkflow.Workflow.create(
            fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
            workflow_id_str=f"cme_{uuid.uuid4().hex}",
            workflow_context={
                "NLU_Pipeline_Stage": fastworkflow.NLUPipelineStage.INTENT_DETECTION,
            }
        )

        # this intializes the conversation traces file name also
        # which is necessary when starting a brand new chat session
        self.clear_conversation_history()

    def start_workflow(self,
        workflow_folderpath: str, 
        workflow_id_str: Optional[str] = None, 
        parent_workflow_id: Optional[int] = None, 
        workflow_context: dict = None, 
        startup_command: str = "", 
        startup_action: Optional[fastworkflow.Action] = None, 
        keep_alive: bool = False,
        project_folderpath: Optional[str] = None
        ) -> Optional[fastworkflow.CommandOutput]:
        """
        Create and start a workflow within this chat session.
        
        Args:
            workflow_folderpath: The folder containing the fastworkflow Workflow
            workflow_id_str: Arbitrary key used to persist the workflow state
            parent_workflow_id: Persist this workflow under a parent workflow
            workflow_context: The starting context for the workflow.
            startup_command: Optional command to execute on startup
            startup_action: Optional action to execute on startup
            keep_alive: Whether to keep the chat session alive after workflow completion
            
        Returns:
            CommandOutput for non-keep_alive workflows, None otherwise
        """
        if startup_command and startup_action:
            raise ValueError("Cannot provide both startup_command and startup_action")

        # Create the workflow
        workflow = fastworkflow.Workflow.create(
            workflow_folderpath,
            workflow_id_str=workflow_id_str,
            parent_workflow_id=parent_workflow_id,
            workflow_context=workflow_context,
            project_folderpath=project_folderpath
        )
        
        self._current_workflow = workflow
        self._status = SessionStatus.STOPPED
        self._startup_command = startup_command

        if startup_action and startup_action.workflow_id is None:
            startup_action.workflow_id = workflow.id
        self._startup_action = startup_action
        self._keep_alive = False if parent_workflow_id else keep_alive

        # Check if we need to stop the current workflow
        # Stop if this is a new root workflow (no parent, keep_alive=True)
        current_workflow = self.get_active_workflow()
        if (current_workflow and 
            parent_workflow_id is None and 
            self._keep_alive):
            logger.info(f"Stopping current workflow {current_workflow.id} to start new root workflow {workflow.id}")
            self.stop_workflow()

        # ------------------------------------------------------------
        # Eager warm-up of CommandRouter / ModelPipeline
        # ------------------------------------------------------------
        # Loading transformer checkpoints and moving them to device is
        # expensive (~1 s).  We do it here *once* for every model artifact
        # directory so that the first user message does not pay the cost.
        try:
            command_info_root = Path(workflow.folderpath) / "___command_info"
            if command_info_root.is_dir():
                subdirs = [d for d in command_info_root.iterdir() if d.is_dir()]

                # Tell the progress bar how many extra steps we are going to
                # perform (one per directory plus one for the wildcard "*").
                StartupProgress.add_total(len(subdirs) + 1)

                for subdir in subdirs:
                    # Instantiating CommandRouter triggers ModelPipeline
                    # construction and caches it process-wide.
                    with contextlib.suppress(Exception):
                        CommandRouter(str(subdir))
                    StartupProgress.advance(f"Warm-up {subdir.name}")

                # Also warm-up the global-context artefacts, which live in a
                # pseudo-folder named '*' in some workflows.
                with contextlib.suppress(Exception):
                    CommandRouter(str(command_info_root / '*'))
                StartupProgress.advance("Warm-up global")
        except Exception as warm_err:  # pragma: no cover – warm-up must never fail
            logger.debug(f"Model warm-up skipped due to error: {warm_err}")

        # Update the command metadata extraction workflow's context with the app workflow
        self._cme_workflow.context["app_workflow"] = workflow

        # Start the workflow
        if self._status != SessionStatus.STOPPED:
            raise RuntimeError("Workflow already started")
        
        self._status = SessionStatus.STARTING
        
        # Push this workflow as active
        self.push_active_workflow(workflow)
        
        # Initialize workflow tool agent if in agent mode
        # This must happen after pushing the workflow to the stack
        # so that get_active_workflow() returns the correct workflow
        if self._run_as_agent:
            self._initialize_agent_functionality()
        
        command_output = None
        if self._keep_alive:
            # Root workflow gets a worker thread
            self._chat_worker = ChatWorker(self)
            self._chat_worker.start()
        else:
            # Child workflows run their loop in the current thread
            self._status = SessionStatus.RUNNING
            command_output = self._run_workflow_loop()

        return command_output

    def _initialize_agent_functionality(self):
        """
        Initialize the workflow tool agent for agent mode.
        This agent handles individual tool selection and execution.
        """
        self._cme_workflow.context["run_as_agent"] = True
        self._current_workflow.context["run_as_agent"] = True

        # Initialize the workflow tool agent
        from fastworkflow.workflow_agent import initialize_workflow_tool_agent       
        self._workflow_tool_agent = initialize_workflow_tool_agent(self)

    @property
    def workflow_tool_agent(self):
        """Get the workflow tool agent for agent mode."""
        return self._workflow_tool_agent
    @property
    def cme_workflow(self) -> fastworkflow.Workflow:
        """Get the command metadata extraction workflow."""
        return self._cme_workflow
    
    @property
    def run_as_agent(self) -> bool:
        """Check if running in agent mode."""
        return self._run_as_agent

    @property
    def user_message_queue(self) -> Queue:
        return self._user_message_queue

    @property
    def command_output_queue(self) -> Queue:
        return self._command_output_queue

    @property
    def command_trace_queue(self) -> Queue:
        return self._command_trace_queue

    @property
    def workflow_is_complete(self) -> bool:
        workflow = self.get_active_workflow()
        return workflow.is_complete if workflow else True
    
    @workflow_is_complete.setter
    def workflow_is_complete(self, value: bool) -> None:
        if workflow := self.get_active_workflow():
            workflow.is_complete = value
    
    @property
    def conversation_history(self) -> dspy.History:
        """Return the conversation history."""
        return self._conversation_history

    # def clear_conversation_history(self, trace_filename_suffix: Optional[str] = None) -> None:
    def clear_conversation_history(self) -> None:
        """
        Clear the conversation history.
        This resets the conversation history to an empty state.
        """
        self._conversation_history = dspy.History(messages=[])
        # Filename for conversation traces
        # if trace_filename_suffix:
        #     self._conversation_traces_file_name: str = (
        #         f"conversation_traces_{trace_filename_suffix}"
        #     )
        # else:
        #     self._conversation_traces_file_name: str = (
        #         f"conversation_traces_{datetime.now().strftime('%m_%d_%Y:%H_%M_%S')}.jsonl"
        #     )

    def _run_workflow_loop(self) -> Optional[fastworkflow.CommandOutput]:
        """
        Run the workflow message processing loop.
        For child workflows (keep_alive=False):
        - Returns final CommandOutput when workflow completes
        - All outputs (success or failure) are sent to queue during processing
        """
        last_output = None
        workflow = self.get_active_workflow()

        try:
            # Handle startup command/action
            if self._startup_command:
                if self._run_as_agent and not self._startup_command.startswith('/'):
                    # In agent mode, use workflow tool agent for processing
                    last_output = self._process_agent_message(self._startup_command)
                else:
                    last_output = self._process_message(self._startup_command)
            elif self._startup_action:
                last_output = self._process_action(self._startup_action)

            while (
                not self.workflow_is_complete or self._keep_alive
            ) and self._status != SessionStatus.STOPPING:
                try:
                    message = self.user_message_queue.get()

                    if ((
                            "NLU_Pipeline_Stage" not in self._cme_workflow.context or
                            self._cme_workflow.context["NLU_Pipeline_Stage"] == fastworkflow.NLUPipelineStage.INTENT_DETECTION) and
                        message.startswith('/')
                    ):
                        self._cme_workflow.context["is_assistant_mode_command"] = True
                    
                    # Route based on mode and message type
                    if self._run_as_agent and "is_assistant_mode_command" not in self._cme_workflow.context:
                        # In agent mode, use workflow tool agent for processing
                        last_output = self._process_agent_message(message)
                    # elif self._is_mcp_tool_call(message):
                    #     last_output = self._process_mcp_tool_call(message)
                    else:
                        last_output = self._process_message(message)
                        
                except Empty:
                    continue

            # Return final output for child workflows, regardless of success/failure
            if not self._keep_alive:
                return last_output

        finally:
            self._status = SessionStatus.STOPPED
            self.pop_active_workflow()
            logger.debug(f"Workflow {workflow.id if workflow else 'unknown'} completed")

        return None
    
    # def _is_mcp_tool_call(self, message: str) -> bool:
    #     """Detect if message is an MCP tool call JSON"""
    #     try:
    #         data = json.loads(message)
    #         return data.get("type") == "mcp_tool_call"
    #     except (json.JSONDecodeError, AttributeError):
    #         return False
    
    # def _process_mcp_tool_call(self, message: str) -> fastworkflow.CommandOutput:
    #     # sourcery skip: class-extract-method, extract-method
    #     """Process an MCP tool call message"""
    #     workflow = self.get_active_workflow()
        
    #     try:
    #         # Parse JSON message
    #         data = json.loads(message)
    #         tool_call_data = data["tool_call"]
            
    #         # Create MCPToolCall object
    #         tool_call = fastworkflow.MCPToolCall(
    #             name=tool_call_data["name"],
    #             arguments=tool_call_data["arguments"]
    #         )
            
    #         # Execute via command executor
    #         mcp_result = self._CommandExecutor.perform_mcp_tool_call(
    #             workflow, 
    #             tool_call, 
    #             command_context=workflow.current_command_context_name
    #         )
            
    #         # Convert MCPToolResult back to CommandOutput for consistency
    #         command_output = self._convert_mcp_result_to_command_output(mcp_result)
            
    #         # Put in output queue if needed
    #         if (not command_output.success or self._keep_alive) and self.command_output_queue:
    #             self.command_output_queue.put(command_output)

    #         # Flush on successful or failed tool call – state may have changed.
    #         if workflow := self.get_active_workflow():
    #             workflow.flush()
            
    #         return command_output
            
    #     except Exception as e:
    #         logger.error(f"Error processing MCP tool call: {e}. Tool call content: {message}")
    #         return self._process_message(message)  # process as a message
    
    # def _convert_mcp_result_to_command_output(self, mcp_result: fastworkflow.MCPToolResult) -> fastworkflow.CommandOutput:
    #     """Convert MCPToolResult to CommandOutput for compatibility"""
    #     command_response = fastworkflow.CommandResponse(
    #         response=mcp_result.content[0].text if mcp_result.content else "No response",
    #         success=not mcp_result.isError
    #     )
        
    #     command_output = fastworkflow.CommandOutput(command_responses=[command_response])
    #     command_output._mcp_source = mcp_result  # Mark for special formatting
    #     return command_output
    
    def _process_agent_message(self, message: str) -> fastworkflow.CommandOutput:
        # sourcery skip: class-extract-method
        """Process a message in agent mode using workflow tool agent"""
        # The agent processes the user's message and may make multiple tool calls
        # to the workflow internally (directly via CommandExecutor)

        # Ensure any prior action log is removed before a fresh agent run
        if os.path.exists("action.jsonl"):
            os.remove("action.jsonl")

        refined_user_query = self._refine_user_query(message, self.conversation_history)

        from fastworkflow.workflow_agent import build_query_with_next_steps
        command_info_and_refined_message_with_todolist = build_query_with_next_steps(
            refined_user_query, 
            self
        )

        # Get available commands for current context and pass to agent.
        # The CommandsSystemPreludeAdapter will inject these commands into the system 
        # message, keeping them out of the trajectory to avoid token bloat while still 
        # providing context-specific command info.
        from fastworkflow.workflow_agent import _what_can_i_do
        available_commands = _what_can_i_do(self)

        lm = dspy_utils.get_lm("LLM_AGENT", "LITELLM_API_KEY_AGENT")
        from dspy.utils.exceptions import AdapterParseError
        from fastworkflow.utils.chat_adapter import CommandsSystemPreludeAdapter
        
        # Use CommandsSystemPreludeAdapter specifically for workflow agent calls
        agent_adapter = CommandsSystemPreludeAdapter()
        
        # Retry logic for AdapterParseError
        max_retries = 2
        for attempt in range(max_retries):
            try:
                with dspy.context(lm=lm, adapter=agent_adapter):
                    agent_result = self._workflow_tool_agent(
                        user_query=command_info_and_refined_message_with_todolist,
                        available_commands=available_commands
                    )
                break  # Success, exit retry loop
            except AdapterParseError as _:
                if attempt == max_retries - 1:  # Last attempt
                    raise  # Re-raise the exception if all retries failed
                # Continue to next attempt

            # dspy.inspect_history(n=1)

        # Extract the final result from the agent
        result_text = (
            agent_result.final_answer
            if hasattr(agent_result, 'final_answer')
            else str(agent_result)
        )

        # Create CommandOutput with the agent's response
        command_response = fastworkflow.CommandResponse(response=result_text)

        conversation_traces = None
        conversation_summary = message
        # Attach actions captured during agent execution as artifacts if available
        if os.path.exists("action.jsonl"):
            with open("action.jsonl", "r", encoding="utf-8") as f:
                actions = [json.loads(line) for line in f if line.strip()]
            conversation_summary, conversation_traces = self._extract_conversation_summary(message, actions, result_text)
            command_response.artifacts["conversation_summary"] = conversation_summary

        self.conversation_history.messages.append(
            {
                "conversation summary": conversation_summary,
                "conversation_traces": conversation_traces,
                "feedback": None  # Initialize feedback slot for this turn
            }
        )

        command_output = fastworkflow.CommandOutput(
            command_responses=[command_response]
        )
        command_output.workflow_name = self._current_workflow.folderpath.split('/')[-1]

        # Put output in queue (following same pattern as _process_message)
        if (not command_output.success or self._keep_alive) and \
                    self.command_output_queue:
            self.command_output_queue.put(command_output)

        # Persist workflow state changes
        if workflow := self.get_active_workflow():
            workflow.flush()

        return command_output

    def _process_message(self, message: str) -> fastworkflow.CommandOutput:
        """Process a single message"""
        # Use our specialized profiling method
        # command_output = self.profile_invoke_command(message)
        
        command_output = self._CommandExecutor.invoke_command(self, message)

        # Record assistant mode trace to action.jsonl (similar to agent mode in workflow_agent.py)
        # This ensures assistant commands are captured even when interspersed with agent commands
        response_text = ""
        if command_output.command_responses:
            response_text = command_output.command_responses[0].response or ""
        
        # Convert parameters to dict if it's a Pydantic model or other complex object
        params = command_output.command_parameters or {}
        if hasattr(params, 'model_dump'):
            params = params.model_dump()
        elif hasattr(params, 'dict'):
            params = params.dict()
        
        record = {
            "command": message,
            "command_name": command_output.command_name or "",
            "parameters": params,
            "response": response_text
        }

        self.conversation_history.messages.append(
            {
                "conversation summary": "assistant_mode_command",
                "conversation_traces": json.dumps(record),
                "feedback": None  # Initialize feedback slot for this turn
            }
        )

        if (not command_output.success or self._keep_alive) and \
            self.command_output_queue:
            self.command_output_queue.put(command_output)

        # Persist workflow state changes lazily accumulated during message processing.
        if workflow := self.get_active_workflow():
            workflow.flush()

        return command_output

    def _process_action(self, action: fastworkflow.Action) -> fastworkflow.CommandOutput:
        """Process a startup action"""
        workflow = self.get_active_workflow()
        command_output = self._CommandExecutor.perform_action(workflow, action)

        # Record action trace to action.jsonl
        response_text = ""
        if command_output.command_responses:
            response_text = command_output.command_responses[0].response or ""
        
        # Convert parameters to dict if it's a Pydantic model or other complex object
        params = action.parameters or {}
        if hasattr(params, 'model_dump'):
            params = params.model_dump()
        elif hasattr(params, 'dict'):
            params = params.dict()
        
        record = {
            "command": "process_action",
            "command_name": action.command_name,
            "parameters": params,
            "response": response_text
        }

        self.conversation_history.messages.append(
            {
                "conversation summary": "process_action command",
                "conversation_traces": json.dumps(record),
                "feedback": None  # Initialize feedback slot for this turn
            }
        )

        if (not command_output.success or self._keep_alive) and \
            self.command_output_queue:
            self.command_output_queue.put(command_output)

        # Flush any pending workflow updates triggered by this startup action.
        if workflow:
            workflow.flush()

        return command_output

    def _refine_user_query(self, user_query: str, conversation_history: dspy.History) -> str:
        """
        Refine user query using conversation history. 
        Return the refined user query
        """
        if conversation_history.messages:
            messages = []
            for conv_dict in conversation_history.messages[-5:]:
                messages.extend([
                    f'{k}: {v}' for k, v in conv_dict.items()
                ])
            messages.append(f'new_user_query: {user_query}')
            return '\n'.join(messages)

        return user_query    

    def _extract_conversation_summary(self, 
        user_query: str, workflow_actions: list[dict[str, str]], final_agent_response: str) -> str:
        """
        Summarizes conversation based on original user query, workflow actions and agentt response.
        Returns the conversation summary and the log entry
        """
        # Lets log everything to a file called action_log.jsonl, if it exists
        conversation_traces = {
            "user_query": user_query,
            "agent_workflow_interactions": workflow_actions,
            "final_agent_response": final_agent_response
        }
        # with open(self._conversation_traces_file_name, "a", encoding="utf-8") as f:
        #     f.write(json.dumps(log_entry) + "\n")

        class ConversationSummarySignature(dspy.Signature):
            """
            A summary of conversation
            Omit descriptions of action sequences 
            Capture relevant facts and parameter values from user query, workflow actions and agent response
            """
            user_query: str = dspy.InputField()
            workflow_actions: list[dict[str, str]] = dspy.InputField()
            final_agent_response: str = dspy.InputField()
            conversation_summary: str = dspy.OutputField(desc="A multiline paragraph summary")

        planner_lm = dspy_utils.get_lm("LLM_PLANNER", "LITELLM_API_KEY_PLANNER")
        with dspy.context(lm=planner_lm):
            cs_func = dspy.ChainOfThought(ConversationSummarySignature)
            prediction = cs_func(
                user_query=user_query, 
                workflow_actions=workflow_actions, 
                final_agent_response=final_agent_response)
            return prediction.conversation_summary, json.dumps(conversation_traces)

    
    def profile_invoke_command(self, message: str):
        """
        Profile the invoke_command method with detailed focus on performance issues.
        
        Args:
            message: The message to process
            output_file: Name of the profile output file
            
        Returns:
            The result of the invoke_command call
        """
        from datetime import datetime
        
        # Generate a unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"invoke_command_{timestamp}.prof"        

        import cProfile
        import pstats
        import io
        import time
        
        # Create a Profile object
        profiler = cProfile.Profile()
        
        # Enable profiling
        profiler.enable()
        
        # Execute invoke_command and time it
        start_time = time.time()
        result = self._CommandExecutor.invoke_command(self, message)
        elapsed = time.time() - start_time
        
        # Disable profiling
        profiler.disable()
        
        # Save profile results to file
        profiler.dump_stats(output_file)
        print(f"\nProfile data saved to {os.path.abspath(output_file)}")
        print(f"invoke_command execution took {elapsed:.4f} seconds")
        
        # Create summary report
        report_file = f"{os.path.splitext(output_file)[0]}_report.txt"
        with open(report_file, "w") as f:
            # Overall summary by cumulative time
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats('cumulative').print_stats(30)
            f.write(f"=== CUMULATIVE TIME SUMMARY (TOP 30) === Execution time: {elapsed:.4f}s\n")
            f.write(s.getvalue())
            f.write("\n\n")
            
            # Internal time summary
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats('time').print_stats(30)
            f.write("=== INTERNAL TIME SUMMARY (TOP 30) ===\n")
            f.write(s.getvalue())
            f.write("\n\n")
            
            # Most called functions
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats('calls').print_stats(30)
            f.write("=== MOST CALLED FUNCTIONS (TOP 30) ===\n")
            f.write(s.getvalue())
            
            # Focus areas for issues 3-7
            focus_areas = [
                ('lock_contention', ['lock', 'acquire', 'release'], 'time'),
                ('model_operations', ['torch', 'nn', 'model'], 'cumulative'),
                ('command_extraction', ['wildcard.py', 'extract', 'predict'], 'cumulative'),
                ('file_io', ['_get_sessiondb_folderpath', '_load', '_save'], 'cumulative'),
                ('frequent_operations', ['startswith', 'isinstance', 'get'], 'calls')
            ]
            
            for name, patterns, sort_by in focus_areas:
                f.write(f"\n\n=== {name.upper()} ===\n")
                for pattern in patterns:
                    s = io.StringIO()
                    ps = pstats.Stats(profiler, stream=s)
                    ps.sort_stats(sort_by).print_stats(pattern, 10)
                    f.write(f"\nPattern: '{pattern}'\n")
                    f.write(s.getvalue())
        
        print(f"Detailed report saved to {os.path.abspath(report_file)}")
        
        return result
