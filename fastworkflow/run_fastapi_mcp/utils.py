import asyncio
import os
import queue
import time
from dataclasses import dataclass
from typing import Any, Optional

from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError
from pydantic import BaseModel, field_validator

import fastworkflow
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


async def ensure_user_runtime_exists(
    channel_id: str,
    session_manager: 'ChannelSessionManager',
    workflow_path: str,
    context: Optional[dict] = None,
    startup_command: Optional[str] = None,
    startup_action: Optional['fastworkflow.Action'] = None,
    stream_format: str = "ndjson",
    http_bearer_token: Optional[str] = None
) -> None:
    """
    Ensure a user runtime exists in the session manager. If not, create it.
    
    This function encapsulates the session creation logic from the initialize endpoint,
    allowing it to be reused across different parts of the application without duplicating code.
    
    Args:
        channel_id: The user identifier
        session_manager: The ChannelSessionManager instance
        workflow_path: Path to the workflow directory (validated at server startup)
        context: Optional workflow context dictionary
        startup_command: Optional startup command
        startup_action: Optional startup action
        stream_format: Stream format preference ("ndjson" or "sse", default "ndjson")
        http_bearer_token: Optional JWT token to update in workflow context
        
    Raises:
        HTTPException: If session creation fails
    """
    # Check if user already has an active session
    existing_runtime = await session_manager.get_session(channel_id)
    if existing_runtime:
        logger.debug(f"Session for channel_id {channel_id} already exists, skipping creation")
        
        # Update the workflow's context with the current token if provided
        if http_bearer_token and existing_runtime.chat_session:
            active_workflow = existing_runtime.chat_session.get_active_workflow()
            if active_workflow and active_workflow.context:
                # Update the workflow's context with the current token
                # Note: We mutate the dictionary in-place (no setter call), which means:
                # 1. The change is immediate and visible to workflow code
                # 2. The workflow is NOT marked dirty (won't persist to disk)
                # 3. This is intentional for JWT tokens - we don't want to persist sensitive tokens
                active_workflow.context['http_bearer_token'] = http_bearer_token
                logger.debug(f"Updated http_bearer_token in workflow context for channel_id {channel_id}")
        
        return
    
    # Prepare workflow context, ensuring http_bearer_token is available
    if http_bearer_token:
        if context:
            # Add or replace http_bearer_token in the context
            context['http_bearer_token'] = http_bearer_token
        else:
            # Initialize context with http_bearer_token
            context = {'http_bearer_token': http_bearer_token}
    
    logger.info(f"Creating new session for channel_id: {channel_id}")
    
    # Resolve conversation store base folder from SPEEDDICT_FOLDERNAME/channel_conversations
    conv_base_folder = get_channelconversations_dir()

    # Create conversation store for this user
    conversation_store = ConversationStore(channel_id, conv_base_folder)

    # Create ChatSession in agent mode (forced)
    chat_session = fastworkflow.ChatSession(run_as_agent=True)

    # Restore last conversation if it exists; else start new
    conv_id_to_restore = None
    if conv_id_to_restore := conversation_store.get_last_conversation_id():
        conversation = conversation_store.get_conversation(conv_id_to_restore)
        if not conversation:
            # this means a new conversation was started but not saved
            conv_id_to_restore = conv_id_to_restore-1
            conversation = conversation_store.get_conversation(conv_id_to_restore)
        
        if conversation:
            # Restore the conversation history from saved turns
            restored_history = restore_history_from_turns(conversation["turns"])
            chat_session._conversation_history = restored_history
            logger.info(f"Restored conversation {conv_id_to_restore} for user {channel_id}")
        else:
            logger.info(f"No conversations available for user {channel_id}, starting new")
            conv_id_to_restore = None

    # Start the workflow
    chat_session.start_workflow(
        workflow_folderpath=workflow_path,
        workflow_context=context,
        startup_command=startup_command,
        startup_action=startup_action,
        keep_alive=True
    )

    # Create and store user runtime
    await session_manager.create_session(
        channel_id=channel_id,
        chat_session=chat_session,
        conversation_store=conversation_store,
        active_conversation_id=conv_id_to_restore,
        stream_format=stream_format
    )

    logger.info(f"Successfully created session for channel_id: {channel_id}")
    
    # Wait for workflow to be ready (background thread sets status to RUNNING)
    import asyncio
    import time
    max_wait = 5  # seconds
    wait_start = time.time()
    from fastworkflow.chat_session import SessionStatus
    while chat_session._status != SessionStatus.RUNNING and time.time() - wait_start < max_wait:
        await asyncio.sleep(0.1)
    
    if chat_session._status != SessionStatus.RUNNING:
        logger.warning(f"Workflow not fully started after {max_wait}s, status={chat_session._status}")


def get_channelconversations_dir() -> str:
    """
    Return SPEEDDICT_FOLDERNAME/channel_conversations, creating the directory if missing.
    fastworkflow is injected to avoid circular imports and to access get_env_var.
    """
    speedict_foldername = fastworkflow.get_env_var("SPEEDDICT_FOLDERNAME")
    user_conversations_dir = os.path.join(speedict_foldername, "channel_conversations")
    os.makedirs(user_conversations_dir, exist_ok=True)
    return user_conversations_dir


async def wait_for_command_output(
    runtime: 'ChannelRuntime',
    timeout_seconds: int
) -> 'fastworkflow.CommandOutput':
    """Wait for command output from the queue with timeout"""
    start_time = time.time()

    while time.time() - start_time < timeout_seconds:
        try:
            return runtime.chat_session.command_output_queue.get(timeout=0.5)
        except queue.Empty:
            await asyncio.sleep(0.1)
            continue

    raise HTTPException(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        detail=f"Command execution timed out after {timeout_seconds} seconds"
    )


def collect_trace_events(runtime: 'ChannelRuntime', user_id: Optional[str] = None) -> list[dict[str, Any]]:
    """
    Drain and collect all trace events from the queue.
    
    Args:
        runtime: ChannelRuntime containing the trace queue
        user_id: Optional user_id to include in traces
        
    Returns:
        List of trace event dictionaries with optional user_id
    """
    traces = []
    
    while True:
        try:
            evt = runtime.chat_session.command_trace_queue.get_nowait()
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
    """Per-channel runtime state"""
    channel_id: str
    active_conversation_id: int
    chat_session: 'fastworkflow.ChatSession'
    lock: asyncio.Lock
    conversation_store: 'ConversationStore'
    stream_format: str = "ndjson"  # "ndjson" | "sse"


class ChannelSessionManager:
    """Process-wide manager for channel sessions"""
    
    def __init__(self):
        self._sessions: dict[str, ChannelRuntime] = {}
        self._lock = asyncio.Lock()
    
    async def get_session(self, channel_id: str) -> Optional[ChannelRuntime]:
        """Get a session by channel_id"""
        async with self._lock:
            return self._sessions.get(channel_id)
    
    async def create_session(
        self,
        channel_id: str,
        chat_session: 'fastworkflow.ChatSession',
        conversation_store: 'ConversationStore',
        active_conversation_id: Optional[int] = None,
        stream_format: str = "ndjson"
    ) -> ChannelRuntime:
        """Create or update a session"""
        async with self._lock:            
            runtime = ChannelRuntime(
                channel_id=channel_id,
                active_conversation_id=active_conversation_id or 0,
                chat_session=chat_session,
                lock=asyncio.Lock(),
                conversation_store=conversation_store,
                stream_format=stream_format
            )
            self._sessions[channel_id] = runtime
            return runtime
    
    async def remove_session(self, channel_id: str) -> None:
        """Remove a session"""
        async with self._lock:
            if channel_id in self._sessions:
                del self._sessions[channel_id]


# ============================================================================
# Helper Functions
# ============================================================================

def save_conversation_incremental(runtime: ChannelRuntime, extract_turns_func, logger) -> None:
    """
    Save conversation turns incrementally after each turn (without generating topic/summary).
    This provides crash protection - all turns except the last will be preserved.
    """
    # Extract turns from conversation history
    if turns := extract_turns_func(runtime.chat_session.conversation_history):
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


