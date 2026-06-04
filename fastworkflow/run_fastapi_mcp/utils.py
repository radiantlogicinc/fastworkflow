import asyncio
import os
import queue
import time
from collections import OrderedDict
from dataclasses import dataclass
from queue import Queue
from typing import Any, Callable, Optional

from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jwt.exceptions import PyJWTError as JWTError
from pydantic import BaseModel, field_validator

import fastworkflow
from fastworkflow.session_state_store import SessionStateStore, get_session_state_store
from fastworkflow.workflow_execution_context import WorkflowExecutionContext
from fastworkflow.utils.logging import logger

from .conversation_store import ConversationStore, restore_history_from_turns
from .jwt_manager import verify_token


# ============================================================================
# Data Models (aligned with FastWorkflow canonical types)
# ============================================================================

class InitializationRequest(BaseModel):
    """Request to initialize a FastWorkflow session for a channel"""
    channel_id: str
    user_id: Optional[str] = None  # Required if startup_command or startup_action provided
    stream_format: Optional[str] = None  # "ndjson" | "sse" (default ndjson)
    startup_command: Optional[str] = None  # Mutually exclusive with startup_action
    startup_action: Optional[dict[str, Any]] = None  # Mutually exclusive with startup_command


class TokenResponse(BaseModel):
    """JWT token pair returned from initialization or token refresh"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # Access token expiration in seconds


class InitializeResponse(BaseModel):
    """Response from initialization including tokens and optional startup output"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # Access token expiration in seconds
    startup_output: Optional[fastworkflow.CommandOutput] = None  # Present if startup was executed


class SessionData(BaseModel):
    """Validated session data extracted from JWT token"""
    channel_id: str
    user_id: Optional[str] = None  # From JWT uid claim
    token_type: str  # "access" or "refresh"
    issued_at: int  # Unix timestamp
    expires_at: int  # Unix timestamp
    jti: str  # JWT ID (unique token identifier)
    http_bearer_token: Optional[str] = None  # The actual JWT token string for workflow context access


class InvokeRequest(BaseModel):
    """
    Request to invoke agent or assistant.
    Requires channel_id to be passed in the Authorization header (via JWT token).
    """
    user_query: str
    timeout_seconds: int = 60


class PerformActionRequest(BaseModel):
    """
    Request to perform a specific action.
    Requires channel_id to be passed in the Authorization header (via JWT token).
    """
    action: dict[str, Any]  # Will be converted to fastworkflow.Action
    timeout_seconds: int = 60


class PostFeedbackRequest(BaseModel):
    """
    Request to post feedback on the latest turn.
    Requires channel_id to be passed in the Authorization header (via JWT token).
    
    Note: binary_or_numeric_score accepts numeric values (float).
    Boolean values (True/False) are automatically converted to 1.0/0.0.
    """
    binary_or_numeric_score: Optional[float] = None
    nl_feedback: Optional[str] = None

    @field_validator('nl_feedback')
    @classmethod
    def validate_feedback_presence(cls, v, info):
        """Ensure at least one feedback field is provided"""
        if v is None and info.data.get('binary_or_numeric_score') is None:
            raise ValueError("At least one of binary_or_numeric_score or nl_feedback must be provided")
        return v


class ActivateConversationRequest(BaseModel):
    """
    Request to activate a conversation by ID.
    Requires channel_id to be passed in the Authorization header (via JWT token).
    """
    conversation_id: int


class DumpConversationsRequest(BaseModel):
    """Admin request to dump all conversations"""
    output_folder: str


class GenerateMCPTokenRequest(BaseModel):
    """Request to generate a long-lived MCP token"""
    channel_id: str
    user_id: Optional[str] = None
    expires_days: int = 365


class CancelPendingRequest(BaseModel):
    """Optional body for /cancel_pending (channel from JWT)."""
    pass


# class CommandOutputWithTraces(BaseModel):
#     """CommandOutput extended with optional traces for HTTP responses"""
#     command_responses: list[dict[str, Any]]
#     workflow_name: str = ""
#     context: str = ""
#     command_name: str = ""
#     command_parameters: str = ""
#     success: bool = True
#     traces: Optional[list[dict[str, Any]]] = None


# ============================================================================
# Helper Functions
# ============================================================================

# Create HTTPBearer security scheme instance
# This integrates with FastAPI's OpenAPI/Swagger UI to provide the "Authorize" button
http_bearer = HTTPBearer(
    scheme_name="BearerAuth",
    description="JWT Bearer token obtained from /initialize or /refresh_token endpoint",
    auto_error=True
)

def get_session_from_jwt(
    credentials: HTTPAuthorizationCredentials = Depends(http_bearer)
) -> SessionData:
    """
    FastAPI dependency to extract and validate session data from JWT Bearer token.
    
    This dependency integrates with FastAPI's security system and Swagger UI:
    - Shows the "Authorize" button in Swagger UI
    - Automatically handles "Bearer " prefix (no need to type it manually)
    - Validates token format and presence
    
    Args:
        credentials: HTTPAuthorizationCredentials from the Authorization header.
                    FastAPI automatically extracts and validates the Bearer token format.
        
    Returns:
        SessionData: Validated session data extracted from the JWT token
        
    Raises:
        HTTPException: If the Authorization header is missing, malformed, or contains an invalid/expired token
        
    Example:
        Use as a dependency in FastAPI endpoints:
        ```python
        @app.post("/endpoint")
        async def endpoint(session: SessionData = Depends(get_session_from_jwt)):
            # Use session.channel_id, session.token_type, etc.
            pass
        ```
        
    HTTP Request Example:
        ```bash
        curl -X POST "http://localhost:8000/endpoint" \\
             -H "Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9..." \\
             -H "Content-Type: application/json" \\
             -d '{"data": "value"}'
        ```
    
    Swagger UI Usage:
        1. Click the "Authorize" button (lock icon)
        2. Enter ONLY your JWT token (without "Bearer " prefix)
        3. Swagger UI automatically adds the "Bearer " prefix
    """
    # Extract token from credentials (already validated by HTTPBearer)
    token = credentials.credentials

    # Verify and decode token
    try:
        payload = verify_token(token, expected_type="access")

        # Extract session data from payload, including the token for workflow context
        return SessionData(
            channel_id=payload["sub"],
            user_id=payload.get("uid"),  # Optional user_id from uid claim
            token_type=payload["type"],
            issued_at=payload["iat"],
            expires_at=payload["exp"],
            jti=payload["jti"],
            http_bearer_token=token  # Store the actual token for workflow access
        )

    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token missing required claim: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


def _merge_workflow_context(
    context: Optional[dict],
    http_bearer_token: Optional[str],
) -> Optional[dict]:
    if not http_bearer_token:
        return context
    merged = dict(context) if context else {}
    merged["http_bearer_token"] = http_bearer_token
    return merged


def _update_http_bearer_token(runtime: "ChannelRuntime", token: str) -> None:
    workflow = runtime.execution_context.get_active_workflow()
    if workflow and workflow.context is not None:
        workflow.context["http_bearer_token"] = token


def _run_startup_sync(
    ctx: WorkflowExecutionContext,
    startup_command: Optional[str],
    startup_action: Optional[fastworkflow.Action],
) -> Optional[fastworkflow.CommandOutput]:
    if startup_action:
        if startup_action.workflow_id is None and ctx.app_workflow:
            startup_action.workflow_id = ctx.app_workflow.id
        return ctx.process_action(startup_action)
    if startup_command:
        assistant_command = f"/{startup_command.lstrip('/')}"
        return ctx.process_message(assistant_command)
    return None


async def ensure_user_runtime_exists(
    channel_id: str,
    session_manager: "ChannelSessionManager",
    workflow_path: str,
    context: Optional[dict] = None,
    startup_command: Optional[str] = None,
    startup_action: Optional["fastworkflow.Action"] = None,
    stream_format: str = "ndjson",
    http_bearer_token: Optional[str] = None,
    *,
    run_startup: bool = True,
) -> None:
    """
    Ensure a Topology-B runtime exists for channel_id (WorkflowExecutionContext, no worker thread).
    """
    existing_runtime = await session_manager.get_session(channel_id)
    if existing_runtime:
        logger.debug(f"Session for channel_id {channel_id} already exists, skipping creation")
        if http_bearer_token:
            _update_http_bearer_token(existing_runtime, http_bearer_token)
        return

    context = _merge_workflow_context(context, http_bearer_token)
    logger.info(f"Creating new Topology-B session for channel_id: {channel_id}")

    conv_base_folder = get_channelconversations_dir()
    conversation_store = ConversationStore(channel_id, conv_base_folder)

    ctx = WorkflowExecutionContext(run_as_agent=True, session_key=channel_id)
    trace_queue: Queue = Queue()
    ctx.set_transport_queues(command_trace_queue=trace_queue)

    app_workflow = fastworkflow.Workflow.create(
        workflow_path,
        workflow_id_str=channel_id,
        workflow_context=context,
    )
    ctx.bind_app_workflow(app_workflow)

    conv_id_to_restore = None
    if conv_id_to_restore := conversation_store.get_last_conversation_id():
        conversation = conversation_store.get_conversation(conv_id_to_restore)
        if not conversation:
            conv_id_to_restore = conv_id_to_restore - 1
            conversation = conversation_store.get_conversation(conv_id_to_restore)
        if conversation:
            ctx._conversation_history = restore_history_from_turns(conversation["turns"])
            logger.info(f"Restored conversation {conv_id_to_restore} for user {channel_id}")
        else:
            conv_id_to_restore = None

    loop = asyncio.get_running_loop()
    startup_ran = False
    if run_startup and (startup_command or startup_action):
        await loop.run_in_executor(
            None,
            lambda: _run_startup_sync(ctx, startup_command, startup_action),
        )
        startup_ran = True

    pending = session_manager.session_state_store.load(channel_id)
    if pending:
        ctx.apply_serialized_state(pending)
        logger.info(f"Restored pending suspended session for channel_id {channel_id}")

    await session_manager.create_session(
        channel_id=channel_id,
        execution_context=ctx,
        conversation_store=conversation_store,
        active_conversation_id=conv_id_to_restore,
        stream_format=stream_format,
        workflow_path=workflow_path,
        startup_ran=startup_ran,
    )
    logger.info(f"Successfully created session for channel_id: {channel_id}")


def get_channel_session_state_dir() -> str:
    """SPEEDDICT_FOLDERNAME/channel_session_state for suspended Topology-B blobs."""
    speedict_foldername = fastworkflow.get_env_var("SPEEDDICT_FOLDERNAME")
    session_state_dir = os.path.join(speedict_foldername, "channel_session_state")
    os.makedirs(session_state_dir, exist_ok=True)
    return session_state_dir


def get_channelconversations_dir() -> str:
    """
    Return SPEEDDICT_FOLDERNAME/channel_conversations, creating the directory if missing.
    fastworkflow is injected to avoid circular imports and to access get_env_var.
    """
    speedict_foldername = fastworkflow.get_env_var("SPEEDDICT_FOLDERNAME")
    user_conversations_dir = os.path.join(speedict_foldername, "channel_conversations")
    os.makedirs(user_conversations_dir, exist_ok=True)
    return user_conversations_dir


def _is_awaiting_user_output(output: fastworkflow.CommandOutput) -> bool:
    if not output.command_responses:
        return False
    return bool(output.command_responses[0].artifacts.get("awaiting_user"))


def persist_pending_after_turn(
    session_manager: "ChannelSessionManager",
    runtime: "ChannelRuntime",
    output: fastworkflow.CommandOutput,
) -> None:
    """Save or clear durable suspended state after a Topology-B turn."""
    ctx = runtime.execution_context
    if ctx.awaiting_user or _is_awaiting_user_output(output):
        session_manager.session_state_store.save(
            runtime.channel_id,
            ctx.serialize_state(channel_id=runtime.channel_id),
        )
    else:
        session_manager.session_state_store.clear(runtime.channel_id)


async def run_process_message(
    runtime: "ChannelRuntime",
    message: str,
    timeout_seconds: int,
    session_manager: "ChannelSessionManager",
) -> fastworkflow.CommandOutput:
    """Run process_message in a thread pool with timeout (Topology B)."""
    loop = asyncio.get_running_loop()
    ctx = runtime.execution_context

    def _run() -> fastworkflow.CommandOutput:
        return ctx.process_message(message)

    try:
        output = await asyncio.wait_for(
            loop.run_in_executor(None, _run),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        logger.error(
            f"Command execution timed out after {timeout_seconds}s "
            f"for channel_id: {runtime.channel_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Command execution timed out after {timeout_seconds} seconds",
        ) from exc

    persist_pending_after_turn(session_manager, runtime, output)
    return output


async def run_process_action(
    runtime: "ChannelRuntime",
    action: fastworkflow.Action,
    timeout_seconds: int,
    session_manager: "ChannelSessionManager",
) -> fastworkflow.CommandOutput:
    loop = asyncio.get_running_loop()
    ctx = runtime.execution_context

    def _run() -> fastworkflow.CommandOutput:
        return ctx.process_action(action)

    try:
        output = await asyncio.wait_for(
            loop.run_in_executor(None, _run),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Action execution timed out after {timeout_seconds} seconds",
        ) from exc

    persist_pending_after_turn(session_manager, runtime, output)
    return output


def _format_trace_event(evt: Any, user_id: Optional[str]) -> dict[str, Any]:
    trace = {
        "direction": evt.direction.value if hasattr(evt.direction, "value") else str(evt.direction),
        "raw_command": evt.raw_command,
        "command_name": evt.command_name,
        "parameters": evt.parameters,
        "response_text": evt.response_text,
        "success": evt.success,
        "timestamp_ms": evt.timestamp_ms,
    }
    if user_id is not None:
        trace["user_id"] = user_id
    return trace


async def _emit_trace_callback(
    on_trace: Callable[[dict[str, Any]], Any],
    trace_dict: dict[str, Any],
    _user_id: Optional[str],
) -> None:
    result = on_trace(trace_dict)
    if asyncio.iscoroutine(result):
        await result


async def run_process_message_with_trace_stream(
    runtime: "ChannelRuntime",
    message: str,
    timeout_seconds: int,
    session_manager: "ChannelSessionManager",
    on_trace: Callable[[dict[str, Any]], Any],
    user_id: Optional[str] = None,
) -> fastworkflow.CommandOutput:
    """
    Run process_message in an executor while draining command_trace_queue concurrently.
    """
    loop = asyncio.get_running_loop()
    ctx = runtime.execution_context
    trace_queue = ctx.command_trace_queue
    if trace_queue is None:
        return await run_process_message(
            runtime, message, timeout_seconds, session_manager
        )

    exec_future = loop.run_in_executor(
        None, lambda: ctx.process_message(message)
    )
    start = time.time()

    while not exec_future.done() and time.time() - start < timeout_seconds:
        while True:
            try:
                evt = trace_queue.get_nowait()
            except queue.Empty:
                break
            if evt is None:
                continue
            await _emit_trace_callback(
                on_trace, _format_trace_event(evt, user_id), user_id
            )
        await asyncio.sleep(0.05)

    if not exec_future.done():
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Command execution timed out after {timeout_seconds} seconds",
        )

    output = await exec_future

    while True:
        try:
            evt = trace_queue.get_nowait()
        except queue.Empty:
            break
        if evt is None:
            continue
        await _emit_trace_callback(
            on_trace, _format_trace_event(evt, user_id), user_id
        )
    persist_pending_after_turn(session_manager, runtime, output)
    return output


def collect_trace_events(runtime: "ChannelRuntime", user_id: Optional[str] = None) -> list[dict[str, Any]]:
    """
    Drain and collect all trace events from the queue.
    
    Args:
        runtime: ChannelRuntime containing the trace queue
        user_id: Optional user_id to include in traces
        
    Returns:
        List of trace event dictionaries with optional user_id
    """
    traces = []
    
    trace_queue = runtime.execution_context.command_trace_queue
    if trace_queue is None:
        return traces

    while True:
        try:
            evt = trace_queue.get_nowait()
            if evt is None:
                break
            traces.append(_format_trace_event(evt, user_id))
        except queue.Empty:
            break

    return traces


async def collect_trace_events_async(
    trace_queue: queue.Queue,
    user_id: Optional[str] = None
) -> list[dict[str, Any]]:
    """
    Async version: Drain and collect all trace events from a trace queue.
    
    Args:
        trace_queue: The trace queue to drain
        user_id: Optional user_id to include in traces
        
    Returns:
        List of trace event dictionaries with optional user_id
    """
    traces = []
    
    while True:
        try:
            evt = trace_queue.get_nowait()
            if evt is None:
                break
            trace = {
                "direction": evt.direction.value if hasattr(evt.direction, 'value') else str(evt.direction),
                "raw_command": evt.raw_command,
                "command_name": evt.command_name,
                "parameters": evt.parameters,
                "response_text": evt.response_text,
                "success": evt.success,
                "timestamp_ms": evt.timestamp_ms
            }
            if user_id is not None:
                trace["user_id"] = user_id
            traces.append(trace)
        except queue.Empty:
            break
    
    return traces


# ============================================================================
# Session Management
# ============================================================================

@dataclass
class ChannelRuntime:
    """Per-channel Topology-B runtime (live WEC + metadata)."""

    channel_id: str
    active_conversation_id: int
    execution_context: WorkflowExecutionContext
    lock: asyncio.Lock
    conversation_store: ConversationStore
    stream_format: str = "ndjson"
    workflow_path: str = ""
    startup_ran: bool = False

    @property
    def chat_session(self) -> WorkflowExecutionContext:
        """Backward-compatible alias for endpoints that referenced chat_session."""
        return self.execution_context


class ChannelSessionManager:
    """
    Process-local cache of live WorkflowExecutionContext instances.

    Suspended (awaiting_user) state is persisted via SessionStateStore so any
    worker can cold-rehydrate after eviction or restart.
    """

    def __init__(
        self,
        session_state_store: Optional[SessionStateStore] = None,
        max_live_sessions: int = 2000,
    ):
        self._sessions: OrderedDict[str, ChannelRuntime] = OrderedDict()
        self._lock = asyncio.Lock()
        self._max_live_sessions = max_live_sessions
        self.session_state_store = session_state_store or get_session_state_store(
            base_folder=get_channel_session_state_dir()
        )

    def _touch(self, channel_id: str) -> None:
        if channel_id in self._sessions:
            self._sessions.move_to_end(channel_id)

    async def _evict_oldest_if_needed(self) -> None:
        while len(self._sessions) > self._max_live_sessions:
            channel_id, runtime = self._sessions.popitem(last=False)
            if runtime.execution_context.awaiting_user:
                self.session_state_store.save(
                    channel_id,
                    runtime.execution_context.serialize_state(channel_id=channel_id),
                )
            runtime.execution_context.close()
            logger.debug(f"Evicted live session cache for channel_id {channel_id}")

    async def get_session(self, channel_id: str) -> Optional[ChannelRuntime]:
        async with self._lock:
            runtime = self._sessions.get(channel_id)
            if runtime:
                self._touch(channel_id)
            return runtime

    async def create_session(
        self,
        channel_id: str,
        execution_context: WorkflowExecutionContext,
        conversation_store: ConversationStore,
        active_conversation_id: Optional[int] = None,
        stream_format: str = "ndjson",
        workflow_path: str = "",
        startup_ran: bool = False,
    ) -> ChannelRuntime:
        async with self._lock:
            runtime = ChannelRuntime(
                channel_id=channel_id,
                active_conversation_id=active_conversation_id or 0,
                execution_context=execution_context,
                lock=asyncio.Lock(),
                conversation_store=conversation_store,
                stream_format=stream_format,
                workflow_path=workflow_path,
                startup_ran=startup_ran,
            )
            self._sessions[channel_id] = runtime
            self._touch(channel_id)
            await self._evict_oldest_if_needed()
            return runtime

    async def remove_session(self, channel_id: str) -> None:
        async with self._lock:
            runtime = self._sessions.pop(channel_id, None)
            if runtime:
                runtime.execution_context.close()

    async def evict_live_session(self, channel_id: str) -> None:
        """Drop from process cache without clearing durable pending state."""
        await self.remove_session(channel_id)


# ============================================================================
# Helper Functions
# ============================================================================

def save_conversation_incremental(runtime: ChannelRuntime, extract_turns_func, logger) -> None:
    """
    Save conversation turns incrementally after each turn (without generating topic/summary).
    This provides crash protection - all turns except the last will be preserved.
    """
    # Extract turns from conversation history
    if turns := extract_turns_func(runtime.execution_context.conversation_history):
        # Initialize conversation ID for first conversation if needed
        if runtime.active_conversation_id == 0:
            # This is the first conversation for this session
            # Reserve ID 1 and use it
            runtime.active_conversation_id = runtime.conversation_store.reserve_next_conversation_id()
            logger.debug(f"Initialized first conversation with ID {runtime.active_conversation_id} for user {runtime.channel_id}")
        
        # Save turns using the active conversation ID
        runtime.conversation_store.save_conversation_turns(
            runtime.active_conversation_id, turns
        )
        logger.debug(f"Incrementally saved {len(turns)} turn(s) to conversation {runtime.active_conversation_id}")


