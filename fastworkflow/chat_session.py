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

import dspy

import fastworkflow
from fastworkflow.utils.logging import logger
from fastworkflow.utils import dspy_utils
from fastworkflow.model_pipeline_training import CommandRouter
from fastworkflow.utils.startup_progress import StartupProgress
from fastworkflow.command_metadata_api import CommandMetadataAPI


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
            workflow = ChatSession.get_active_workflow()
            logger.debug(f"Started root workflow {workflow.id}")
            
            # Run the workflow loop
            self.chat_session._run_workflow_loop()
            
        finally:
            self.chat_session._status = SessionStatus.STOPPED
            # Ensure workflow is popped if thread terminates unexpectedly
            if ChatSession.get_active_workflow() is not None:
                ChatSession.pop_active_workflow()

class ChatSession:
    _workflow_stack_lock = Lock()
    _workflow_stack: ClassVar[deque[fastworkflow.Workflow]] = deque()  # Stack of workflow objects
    
    @classmethod
    def get_active_workflow(cls) -> Optional[fastworkflow.Workflow]:
        """Get the currently active workflow (top of stack)"""
        with cls._workflow_stack_lock:
            return cls._workflow_stack[-1] if cls._workflow_stack else None
    
    @classmethod
    def push_active_workflow(cls, workflow: fastworkflow.Workflow) -> None:
        with cls._workflow_stack_lock:
            cls._workflow_stack.append(workflow)
            logger.debug(f"Workflow stack: {[w.id for w in cls._workflow_stack]}")
    
    @classmethod
    def pop_active_workflow(cls) -> Optional[fastworkflow.Workflow]:
        with cls._workflow_stack_lock:
            if not cls._workflow_stack:
                return None
            workflow = cls._workflow_stack.pop()
            logger.debug(f"Workflow stack after pop: {[w.id for w in cls._workflow_stack]}")
            return workflow

    @classmethod
    def clear_workflow_stack(cls) -> None:
        """Clear the entire workflow stack"""
        with cls._workflow_stack_lock:
            cls._workflow_stack.clear()
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
        ChatSession.clear_workflow_stack()
        
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
        # Create queues for user messages and command outputs
        self._user_message_queue = Queue()
        self._command_output_queue = Queue()
        self._command_trace_queue = Queue()
        self._status = SessionStatus.STOPPED
        self._chat_worker = None

        self._conversation_history = dspy.History(messages=[])
        
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
        current_workflow = ChatSession.get_active_workflow()
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
        ChatSession.push_active_workflow(workflow)
        
        # Initialize workflow tool agent if in agent mode
        # This must happen after pushing the workflow to the stack
        # so that get_active_workflow() returns the correct workflow
        if self._run_as_agent:
            self._initialize_workflow_tool_agent()
        
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

    def _initialize_workflow_tool_agent(self):
        """
        Initialize the workflow tool agent for agent mode.
        This agent handles individual tool selection and execution.
        """
        if not self._workflow_tool_agent:
            # Initialize the workflow tool agent
            from fastworkflow.mcp_server import FastWorkflowMCPServer
            from fastworkflow.workflow_agent import initialize_workflow_tool_agent
            
            mcp_server = FastWorkflowMCPServer(self)
            self._workflow_tool_agent = initialize_workflow_tool_agent(mcp_server)
    
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
        workflow = ChatSession.get_active_workflow()
        return workflow.is_complete if workflow else True
    
    @workflow_is_complete.setter
    def workflow_is_complete(self, value: bool) -> None:
        if workflow := ChatSession.get_active_workflow():
            workflow.is_complete = value
    
    @property
    def conversation_history(self) -> dspy.History:
        """Return the conversation history."""
        return self._conversation_history
       

    def _run_workflow_loop(self) -> Optional[fastworkflow.CommandOutput]:
        """
        Run the workflow message processing loop.
        For child workflows (keep_alive=False):
        - Returns final CommandOutput when workflow completes
        - All outputs (success or failure) are sent to queue during processing
        """
        last_output = None
        workflow = ChatSession.get_active_workflow()

        try:
            # Handle startup command/action
            if self._startup_command:
                if self._run_as_agent:
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
                    
                    # Route based on mode and message type
                    if self._run_as_agent:
                        # In agent mode, use workflow tool agent for processing
                        last_output = self._process_agent_message(message)
                    elif self._is_mcp_tool_call(message):
                        last_output = self._process_mcp_tool_call(message)
                    else:
                        last_output = self._process_message(message)
                        
                except Empty:
                    continue

            # Return final output for child workflows, regardless of success/failure
            if not self._keep_alive:
                return last_output

        finally:
            self._status = SessionStatus.STOPPED
            ChatSession.pop_active_workflow()
            logger.debug(f"Workflow {workflow.id if workflow else 'unknown'} completed")

        return None
    
    def _is_mcp_tool_call(self, message: str) -> bool:
        """Detect if message is an MCP tool call JSON"""
        try:
            data = json.loads(message)
            return data.get("type") == "mcp_tool_call"
        except (json.JSONDecodeError, AttributeError):
            return False
    
    def _process_mcp_tool_call(self, message: str) -> fastworkflow.CommandOutput:
        # sourcery skip: class-extract-method, extract-method
        """Process an MCP tool call message"""
        workflow = ChatSession.get_active_workflow()
        
        try:
            # Parse JSON message
            data = json.loads(message)
            tool_call_data = data["tool_call"]
            
            # Create MCPToolCall object
            tool_call = fastworkflow.MCPToolCall(
                name=tool_call_data["name"],
                arguments=tool_call_data["arguments"]
            )
            
            # Execute via command executor
            mcp_result = self._CommandExecutor.perform_mcp_tool_call(
                workflow, 
                tool_call, 
                command_context=workflow.current_command_context_name
            )
            
            # Convert MCPToolResult back to CommandOutput for consistency
            command_output = self._convert_mcp_result_to_command_output(mcp_result)
            
            # Put in output queue if needed
            if (not command_output.success or self._keep_alive) and self.command_output_queue:
                self.command_output_queue.put(command_output)

            # Flush on successful or failed tool call – state may have changed.
            if workflow := ChatSession.get_active_workflow():
                workflow.flush()
            
            return command_output
            
        except Exception as e:
            logger.error(f"Error processing MCP tool call: {e}. Tool call content: {message}")
            return self._process_message(message)  # process as a message
    
    def _convert_mcp_result_to_command_output(self, mcp_result: fastworkflow.MCPToolResult) -> fastworkflow.CommandOutput:
        """Convert MCPToolResult to CommandOutput for compatibility"""
        command_response = fastworkflow.CommandResponse(
            response=mcp_result.content[0].text if mcp_result.content else "No response",
            success=not mcp_result.isError
        )
        
        command_output = fastworkflow.CommandOutput(command_responses=[command_response])
        command_output._mcp_source = mcp_result  # Mark for special formatting
        return command_output
    
    def _process_agent_message(self, message: str) -> fastworkflow.CommandOutput:
        """Process a message in agent mode using workflow tool agent"""
        # The agent processes the user's message and may make multiple tool calls
        # to the workflow internally (directly via CommandExecutor)

        # Ensure any prior action log is removed before a fresh agent run
        if os.path.exists("action.json"):
            os.remove("action.json")

        refined_message = f'messsage\n{self._think_and_plan(message, self.conversation_history)}'

        lm = dspy_utils.get_lm("LLM_AGENT", "LITELLM_API_KEY_AGENT")
        from dspy.utils.exceptions import AdapterParseError
        # Retry logic for AdapterParseError
        max_retries = 2
        for attempt in range(max_retries):
            try:
                with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
                    agent_result = self._workflow_tool_agent(
                        user_query=refined_message,
                        conversation_history=self.conversation_history
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

        user_instructions_summary = message
        # Attach actions captured during agent execution as artifacts if available
        if os.path.exists("action.json"):
            with open("action.json", "r", encoding="utf-8") as f:
                actions = [json.loads(line) for line in f if line.strip()]
            user_instructions_summary = self._extract_user_instructions(message, actions)
            command_response.artifacts["user_instructions_summary"] = user_instructions_summary

        self.conversation_history.messages.append(
            {"user_instructions": user_instructions_summary, 
             "agent_response": result_text}
        )

        command_output = fastworkflow.CommandOutput(
            command_responses=[command_response]
        )

        # Put output in queue (following same pattern as _process_message)
        if (not command_output.success or self._keep_alive) and \
                self.command_output_queue:
            self.command_output_queue.put(command_output)

        # Persist workflow state changes
        if workflow := ChatSession.get_active_workflow():
            workflow.flush()

        return command_output
    
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

    def _process_message(self, message: str) -> fastworkflow.CommandOutput:
        """Process a single message"""
        # Use our specialized profiling method
        # command_output = self.profile_invoke_command(message)
        
        command_output = self._CommandExecutor.invoke_command(self, message)
        if (not command_output.success or self._keep_alive) and \
            self.command_output_queue:
            self.command_output_queue.put(command_output)

        # Persist workflow state changes lazily accumulated during message processing.
        if workflow := ChatSession.get_active_workflow():
            workflow.flush()

        return command_output

    def _process_action(self, action: fastworkflow.Action) -> fastworkflow.CommandOutput:
        """Process a startup action"""
        workflow = ChatSession.get_active_workflow()
        command_output = self._CommandExecutor.perform_action(workflow, action)
        if (not command_output.success or self._keep_alive) and \
            self.command_output_queue:
            self.command_output_queue.put(command_output)

        # Flush any pending workflow updates triggered by this startup action.
        if workflow:
            workflow.flush()

        return command_output

    def _think_and_plan(self, user_query: str, conversation_history: dspy.History) -> str:
        """
        Returns a refined plan by breaking down a user_query into simpler tasks.
        """
        class TaskPlannerSignature(dspy.Signature):
            """
            Break down a user_query into simpler tasks based only on available commands and conversation_history.
            If user_query is simple, return a single todo that is the user_query as-is
            """
            user_query: str = dspy.InputField()
            conversation_history: dspy.History = dspy.InputField()
            available_commands: list[str] = dspy.InputField()
            todo_list: list[str] = dspy.OutputField(desc="task descriptions as short sentences")

        current_workflow = ChatSession.get_active_workflow()
        available_commands = CommandMetadataAPI.get_command_display_text(
            subject_workflow_path=current_workflow.folderpath,
            cme_workflow_path=fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
            active_context_name=current_workflow.current_command_context_name,
        )

        planner_lm = dspy_utils.get_lm("LLM_PLANNER", "LITELLM_API_KEY_PLANNER")
        with dspy.context(lm=planner_lm):
            task_planner_func = dspy.ChainOfThought(TaskPlannerSignature)
            prediction = task_planner_func(
                user_query=user_query,
                conversation_history=conversation_history, 
                available_commands=available_commands)

            if not prediction.todo_list or (len(prediction.todo_list) == 1 and prediction.todo_list[0] == user_query):
                return user_query

            steps_list = '\n'.join([f'{i + 1}. {task}' for i, task in enumerate(prediction.todo_list)])
            return f"{user_query}\nNext steps:\n{steps_list}"


    def _extract_user_instructions(self, 
        user_query: str, workflow_actions: list[dict[str, str]]) -> str:
        """
        Summarizes user instructions based on original user query and subsequent user feedback in workflow actions.
        """
        class UserInstructionCompilerSignature(dspy.Signature):
            """
            Concise summary of user instructions based on their commands to the workflow. 
            Include parameter values passed in commands in the summary.
            """
            commands_list: list[str] = dspy.InputField()
            user_instructions_summary: str = dspy.OutputField(desc="A single paragraph summary")

        commands_list: list[str] = [user_query]
        commands_list.extend([wf_action['command'] for wf_action in workflow_actions if 'command' in wf_action])

        planner_lm = dspy_utils.get_lm("LLM_PLANNER", "LITELLM_API_KEY_PLANNER")
        with dspy.context(lm=planner_lm):
            uic_func = dspy.ChainOfThought(UserInstructionCompilerSignature)
            prediction = uic_func(commands_list=commands_list)
            return prediction.user_instructions_summary
