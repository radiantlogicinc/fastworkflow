from enum import Enum
from queue import Empty, Queue
from threading import Thread, Lock
from typing import ClassVar, Optional
from collections import deque
import json
import contextlib

import fastworkflow
from fastworkflow.utils.logging import logger
from pathlib import Path
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
            logger.debug(f"Started root workflow {self.chat_session.app_workflow.id}")
            
            # Run the workflow loop
            self.chat_session._run_workflow_loop()
            
        finally:
            self.chat_session._status = SessionStatus.STOPPED
            # Ensure workflow is popped if thread terminates unexpectedly
            if ChatSession.get_active_workflow_id() == self.chat_session.app_workflow.id:
                ChatSession.pop_active_session()

class ChatSession:
    _workflow_stack_lock = Lock()
    _workflow_stack: ClassVar[deque[int]] = deque()  # Stack of workflow IDs
    _map_workflow_id_2_chat_session: ClassVar[dict[int, "ChatSession"]] = {}
    
    @classmethod
    def get_active_workflow_id(cls) -> Optional[int]:
        """Get the currently active workflow ID (top of stack)"""
        with cls._workflow_stack_lock:
            return cls._workflow_stack[-1] if cls._workflow_stack else None
    
    @classmethod
    def push_active_workflow(cls, workflow_id: int) -> None:
        with cls._workflow_stack_lock:
            if workflow_id not in cls._map_workflow_id_2_chat_session:
                raise ValueError(f"Workflow {workflow_id} does not exist")
            cls._workflow_stack.append(workflow_id)
            logger.debug(f"Workflow stack: {list(cls._workflow_stack)}")
    
    @classmethod
    def pop_active_session(cls) -> Optional[int]:
        with cls._workflow_stack_lock:
            if not cls._workflow_stack:
                return None
            workflow_id = cls._workflow_stack.pop()
            logger.debug(f"Workflow stack after pop: {list(cls._workflow_stack)}")
            return workflow_id

    def __init__(self,
                 workflow_folderpath: str, 
                 workflow_id_str: Optional[str] = None, 
                 parent_workflow_id: Optional[int] = None, 
                 workflow_context: dict = None, 
                 startup_command: str = "", 
                 startup_action: Optional[fastworkflow.Action] = None, 
                 keep_alive: bool = False, 
                 user_message_queue: Optional[Queue] = None, 
                 command_output_queue: Optional[Queue] = None):
        """
        Initialize a workflow chat workflow.
        
        Args:
            workflow_folderpath: The folder containing the fastworkflow Workflow
            workflow_id_str: Arbitrary key used to persist the workflow state
            parent_workflow_id: Persist this workflow under a parent workflow
            workflow_context: The starting context for the workflow.
            startup_command: Optional command to execute on startup
            startup_action: Optional action to execute on startup
            keep_alive: Whether to keep the chat session alive after workflow completion
            user_message_queue: If this is a keep_alive child workflow, pass this queue 
            command_output_queue: If this is a keep_alive child workflow, pass this queue
        """
        if startup_command and startup_action:
            raise ValueError("Cannot provide both startup_command and startup_action")

        if keep_alive:
            if user_message_queue is not None or command_output_queue is not None:
                raise ValueError("user_message_queue and command_output_queue are created automatically when keep_alive is True")
            user_message_queue = Queue()
            command_output_queue = Queue()
        elif user_message_queue is None and command_output_queue is not None:
            raise ValueError("when keep_alive is False - Provide both user_message_queue and command_output_queue OR provide neither")
        elif user_message_queue is not None and command_output_queue is None:
            raise ValueError("when keep_alive is False - Provide both user_message_queue and command_output_queue OR provide neither")

        self._user_message_queue=user_message_queue
        self._command_output_queue=command_output_queue

        self._app_workflow = fastworkflow.Workflow.create(
            workflow_folderpath,
            workflow_id_str=workflow_id_str,
            parent_workflow_id=parent_workflow_id,
            # user_message_queue=user_message_queue,
            # command_output_queue=command_output_queue,
            workflow_context=workflow_context
        )

        self._status = SessionStatus.STOPPED
        self._chat_worker: Optional[ChatWorker] = None
        self._startup_command = startup_command

        if startup_action and startup_action.workflow_id is None:
            startup_action.workflow_id = self._app_workflow.id
        self._startup_action = startup_action
        self._keep_alive = keep_alive

        # Register chat session
        ChatSession._map_workflow_id_2_chat_session[self._app_workflow.id] = self

        # ------------------------------------------------------------
        # Eager warm-up of CommandRouter / ModelPipeline
        # ------------------------------------------------------------
        # Loading transformer checkpoints and moving them to device is
        # expensive (~1 s).  We do it here *once* for every model artifact
        # directory so that the first user message does not pay the cost.
        try:
            command_info_root = Path(self._app_workflow.folderpath) / "___command_info"
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

        # create the command metadata extraction workflow
        self._cme_workflow = fastworkflow.Workflow.create(
            fastworkflow.get_internal_workflow_path("command_metadata_extraction"),
            parent_workflow_id = self._app_workflow.id,
            workflow_context = {
                "NLU_Pipeline_Stage": fastworkflow.NLUPipelineStage.INTENT_DETECTION,
                "app_workflow": self._app_workflow
            }
        )

        from fastworkflow.command_executor import CommandExecutor
        self._CommandExecutor = CommandExecutor
    
    def start(self) -> Optional[fastworkflow.CommandOutput]:
        """Start the chat session"""
        if self._status != SessionStatus.STOPPED:
            raise RuntimeError("Workflow already started")
        
        self._status = SessionStatus.STARTING
        
        # Push this workflow as active
        ChatSession.push_active_workflow(self._app_workflow.id)
        
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

    @property
    def app_workflow(self) -> fastworkflow.Workflow:
        return self._app_workflow

    @property
    def cme_workflow(self) -> fastworkflow.Workflow:
        return self._cme_workflow

    @property
    def user_message_queue(self) -> Queue:
        return self._user_message_queue

    @property
    def command_output_queue(self) -> Queue:
        return self._command_output_queue

    @property
    def workflow_is_complete(self) -> bool:
        return self._app_workflow.is_complete
    
    @workflow_is_complete.setter
    def workflow_is_complete(self, value: bool) -> None:
        self._app_workflow.is_complete = value
       

    def _run_workflow_loop(self) -> Optional[fastworkflow.CommandOutput]:
        """
        Run the workflow message processing loop.
        For child workflows (keep_alive=False):
        - Returns final CommandOutput when workflow completes
        - All outputs (success or failure) are sent to queue during processing
        """
        last_output = None

        try:
            # Handle startup command/action
            if self._startup_command:
                last_output = self._process_message(self._startup_command)
            elif self._startup_action:
                last_output = self._process_action(self._startup_action)

            while (
                not self.workflow_is_complete or self._keep_alive
            ) and self._status != SessionStatus.STOPPING:
                try:
                    message = self.user_message_queue.get()
                    
                    # NEW: Route based on message type
                    if self._is_mcp_tool_call(message):
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
            ChatSession.pop_active_session()
            logger.debug(f"Workflow {self._app_workflow.id} completed")

        return None
    
    def _is_mcp_tool_call(self, message: str) -> bool:
        """Detect if message is an MCP tool call JSON"""
        try:
            data = json.loads(message)
            return data.get("type") == "mcp_tool_call"
        except (json.JSONDecodeError, AttributeError):
            return False
    
    def _process_mcp_tool_call(self, message: str) -> fastworkflow.CommandOutput:
        """Process an MCP tool call message"""
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
                self._app_workflow, 
                tool_call, 
                context=self._app_workflow.current_command_context_name
            )
            
            # Convert MCPToolResult back to CommandOutput for consistency
            command_output = self._convert_mcp_result_to_command_output(mcp_result)
            
            # Put in output queue if needed
            if (not command_output.success or self._keep_alive) and self.command_output_queue:
                self.command_output_queue.put(command_output)

            # Flush on successful or failed tool call – state may have changed.
            self._app_workflow.flush()
            
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
    
    def profile_invoke_command(self, message: str):
        """
        Profile the invoke_command method with detailed focus on performance issues.
        
        Args:
            message: The message to process
            output_file: Name of the profile output file
            
        Returns:
            The result of the invoke_command call
        """
        import os
        from datetime import datetime
        
        # Generate a unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"invoke_command_{timestamp}.prof"        

        import cProfile
        import pstats
        import io
        import os
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
        self._app_workflow.flush()

        return command_output

    def _process_action(self, action: fastworkflow.Action) -> fastworkflow.CommandOutput:
        """Process a startup action"""
        command_output = self._CommandExecutor.perform_action(self._app_workflow, action)
        if (not command_output.success or self._keep_alive) and \
            self.command_output_queue:
            self.command_output_queue.put(command_output)

        # Flush any pending workflow updates triggered by this startup action.
        self._app_workflow.flush()

        return command_output
