from enum import Enum
from queue import Empty, Queue
from threading import Thread, Lock
from typing import ClassVar, Optional
from collections import deque

import fastworkflow
from fastworkflow.utils.logging import logger
from fastworkflow.command_interfaces import CommandExecutorInterface, CommandRouterInterface


class SessionStatus(Enum):
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"

class WorkflowWorker(Thread):
    def __init__(self, workflow_session: "WorkflowSession"):
        super().__init__()
        self.workflow_session = workflow_session
        self.daemon = True
        
    def run(self):
        """Process messages for the root workflow"""
        try:
            self.workflow_session._status = SessionStatus.RUNNING
            logger.debug(f"Started root workflow {self.workflow_session.session.id}")
            
            # Run the workflow loop
            self.workflow_session._run_workflow_loop()
            
        finally:
            self.workflow_session._status = SessionStatus.STOPPED
            # Ensure session is popped if thread terminates unexpectedly
            if WorkflowSession.get_active_session_id() == self.workflow_session.session.id:
                WorkflowSession.pop_active_session()

class WorkflowSession:
    _session_stack_lock = Lock()
    _session_stack: ClassVar[deque[int]] = deque()  # Stack of session IDs
    _map_session_id_2_workflow_session: ClassVar[dict[int, "WorkflowSession"]] = {}
    
    @classmethod
    def get_active_session_id(cls) -> Optional[int]:
        """Get the currently active session ID (top of stack)"""
        with cls._session_stack_lock:
            return cls._session_stack[-1] if cls._session_stack else None
    
    @classmethod
    def push_active_session(cls, session_id: int) -> None:
        with cls._session_stack_lock:
            if session_id not in cls._map_session_id_2_workflow_session:
                raise ValueError(f"Session {session_id} does not exist")
            cls._session_stack.append(session_id)
            logger.debug(f"Session stack: {list(cls._session_stack)}")
    
    @classmethod
    def pop_active_session(cls) -> Optional[int]:
        with cls._session_stack_lock:
            if not cls._session_stack:
                return None
            session_id = cls._session_stack.pop()
            logger.debug(f"Session stack after pop: {list(cls._session_stack)}")
            return session_id

    def __init__(self,
                 command_router: CommandRouterInterface,
                 command_executor: CommandExecutorInterface,
                 workflow_folderpath: str,
                 session_id_str: Optional[str] = None,
                 parent_session_id: Optional[int] = None,
                 context: dict = {},
                 startup_command: str = "",
                 startup_action: Optional[fastworkflow.Action] = None,
                 keep_alive: bool = False,
                 user_message_queue: Optional[Queue] = None,
                 command_output_queue: Optional[Queue] = None):
        """
        Initialize a workflow session.
        
        Args:
            session: The underlying fastworkflow Session
            startup_command: Optional command to execute on startup
            startup_action: Optional action to execute on startup
            keep_alive: Whether to keep the session alive after workflow completion
        """
        if startup_command and startup_action:
            raise ValueError("Cannot provide both startup_command and startup_action")

        if keep_alive:
            if not (user_message_queue is None and command_output_queue is None):
                raise ValueError("user_message_queue and command_output_queue are created automatically when keep_alive is True")
            user_message_queue = Queue()
            command_output_queue = Queue()
        else:
            if user_message_queue is None or command_output_queue is None:
                raise ValueError("user_message_queue and command_output_queue must be provided when keep_alive is False")

        self._session = fastworkflow.Session.create(
            workflow_folderpath,
            session_id_str=session_id_str,
            parent_session_id=parent_session_id,
            user_message_queue=user_message_queue,
            command_output_queue=command_output_queue,
            context=context
        )

        self._status = SessionStatus.STOPPED
        self._worker: Optional[WorkflowWorker] = None
        self._startup_command = startup_command

        if startup_action and startup_action.session_id is None:
            startup_action.session_id = self._session.id
        self._startup_action = startup_action
        self._keep_alive = keep_alive

        self._command_router = command_router
        self._command_executor = command_executor
        
        # Register session
        WorkflowSession._map_session_id_2_workflow_session[self._session.id] = self
    
    def start(self) -> Optional[fastworkflow.CommandOutput]:
        """Start the workflow session"""
        if self._status != SessionStatus.STOPPED:
            raise RuntimeError("Session already started")
        
        self._status = SessionStatus.STARTING
        
        # Push this session as active
        WorkflowSession.push_active_session(self._session.id)
        
        command_output = None
        if self._keep_alive:
            # Root workflow gets a worker thread
            self._worker = WorkflowWorker(self)
            self._worker.start()
        else:
            # Child workflows run their loop in the current thread
            self._status = SessionStatus.RUNNING
            command_output = self._run_workflow_loop()

        return command_output

    @property
    def session(self) -> fastworkflow.Session:
        return self._session

    @property
    def user_message_queue(self) -> Queue:
        return self._session.user_message_queue

    @property
    def command_output_queue(self) -> Queue:
        return self._session.command_output_queue

    @property
    def workflow_is_complete(self) -> bool:
        return self._session.workflow_snapshot.workflow.is_complete
    
    @workflow_is_complete.setter
    def workflow_is_complete(self, value: bool) -> None:
        self._session.workflow_snapshot.workflow.is_complete = value

    @property
    def command_router(self) -> CommandRouterInterface:
        return self._command_router

    @property
    def command_executor(self) -> CommandExecutorInterface:
        return self._command_executor

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
            
            while not self.workflow_is_complete or self._keep_alive:
                if self._status == SessionStatus.STOPPING:
                    break
                    
                try:
                    message = self.user_message_queue.get()
                    last_output = self._process_message(message)
                except Empty:
                    continue
            
            # Return final output for child workflows, regardless of success/failure
            if not self._keep_alive:
                return last_output
                
        finally:
            self._status = SessionStatus.STOPPED
            WorkflowSession.pop_active_session()
            logger.debug(f"Workflow {self._session.id} completed")
            
        return None
    
    def _process_message(self, message: str) -> fastworkflow.CommandOutput:
        """Process a single message"""
        command_output = self._command_router.route_command(self, message)
        if not command_output.success or self._keep_alive:
            self._session.command_output_queue.put(command_output)
        return command_output
    
    def _process_action(self, action: fastworkflow.Action) -> fastworkflow.CommandOutput:
        """Process a startup action"""
        command_output = self._command_executor.perform_action(self._session, action)
        if not command_output.success or self._keep_alive:
            self._session.command_output_queue.put(command_output)
        return command_output
