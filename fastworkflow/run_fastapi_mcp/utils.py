import asyncio
import os
import queue
import time
import weakref
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from queue import Queue
from typing import Annotated, Any, Callable, Optional

from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jwt.exceptions import PyJWTError as JWTError
from pydantic import BaseModel, Field, field_validator

import fastworkflow
from fastworkflow.session_state_store import (
    IncompatibleSessionState,
    PendingReapOutcome,
    PendingRetentionPolicy,
    SessionStateStore,
    get_session_state_store,
)
from fastworkflow.state_serialization import StateEncodingError
from fastworkflow.workflow_execution_context import WorkflowExecutionContext
from fastworkflow.utils.logging import logger

from fastworkflow.checkpoint_store import (
    ChannelCheckpointStore,
    CheckpointIdentity,
    CheckpointStoreError,
    QuarantineReason,
    RetentionPolicy,
)
from . import checkpoint
from .conversation_store import ConversationStore, restore_history_from_turns
from .jwt_manager import verify_token


# Live sessions kept in the process cache. Lowered from 2000 now that eviction
# writes a checkpoint instead of discarding state.
DEFAULT_MAX_LIVE_SESSIONS = 50

# How often the server reclaims abandoned checkpoints. A private constant, not a
# knob: the store deliberately has no timer of its own, so somebody has to drive
# it, but an operator does not need a second dial to turn.
CHECKPOINT_REAP_INTERVAL_SECONDS = 300.0


def resolve_max_live_sessions() -> tuple[int, str]:
    """Resolve MAX_LIVE_SESSIONS with the process environment taking precedence.

    OS first, deliberately. ``get_env_var`` returns a supplied default *before*
    consulting ``os.environ``, so passing a default here would make the container
    variable unreachable — and an operator control whose documented default
    outranks the container override is not an operator control.

    Returns the value and where it came from, because an operator has to be able
    to see whether their override actually took effect.
    """
    raw = os.environ.get("MAX_LIVE_SESSIONS")
    source = "process environment"
    if raw is None or raw == "":
        raw = fastworkflow.get_env_var("MAX_LIVE_SESSIONS", default=None)
        source = "workflow env file"
    if raw is None or raw == "":
        return DEFAULT_MAX_LIVE_SESSIONS, "default"

    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"MAX_LIVE_SESSIONS={raw!r} (from {source}) is not an integer"
        ) from exc
    if value <= 0:
        raise ValueError(
            f"MAX_LIVE_SESSIONS={value} (from {source}) must be greater than zero"
        )
    return value, source


# Newest conversation turns kept in memory per channel. Turns are request-sized
# (~450 KB in the workload this bound was sized against, so ~9 MB per channel),
# only the newest few are ever read, and the durable record keeps the rest.
MAX_CONVERSATION_TURNS_IN_MEMORY = 20


# ============================================================================
# Data Models (aligned with FastWorkflow canonical types)
# ============================================================================

# A channel id as a client supplies it. Empty is the one value the storage layer
# refuses: both durable stores key records by `encode_path_component(channel_id)`,
# which raises on "" and is total for every other string. Rejected here so a
# malformed request is answered as one, rather than surfacing as a 500 carrying a
# storage-layer message the caller cannot act on (fix-3xf). The ValueError stays
# as the backstop; this is what makes it unreachable from a well-formed request.
#
# Deliberately no length cap. The encoder's ceiling is 200 bytes on the *encoded*
# name, which percent-escaping can triple, so any raw-length number derived from
# it would reject ordinary ids to prevent nothing: an oversized id is a supported
# path (prefix plus sha256, with the raw id re-checked on read), not an error.
#
# Applies to values a client sends. `SessionData.channel_id` comes out of a JWT
# this server minted, so closing both minting endpoints closes that route too.
ChannelId = Annotated[str, Field(min_length=1)]


class InitializationRequest(BaseModel):
    """Request to initialize a FastWorkflow session for a channel"""
    channel_id: ChannelId
    user_id: Optional[str] = None  # Required if startup_command or startup_action provided
    stream_format: Optional[str] = None  # "ndjson" | "sse" (default ndjson)
    startup_command: Optional[str] = None  # Mutually exclusive with startup_action
    startup_action: Optional[dict[str, Any]] = None  # Mutually exclusive with startup_command
    # How long the request blocks for the startup turn before deferring (202).
    # Same shape/default as InvokeRequest/PerformActionRequest.timeout_seconds.
    timeout_seconds: int = 60


class TokenResponse(BaseModel):
    """JWT token pair returned from initialization or token refresh"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # Access token expiration in seconds


class InitializeResponse(BaseModel):
    """Response from initialization including tokens and optional startup output.

    Startup runs as a turn (wait-or-defer). If it finishes within the wait
    window, ``startup_output`` is present (200). Otherwise it is still running
    and the caller polls via ``startup_turn_key`` (202). The "already exists"
    branch returns the SAME startup execution's three-state status, never a
    silently-empty result (§3.3).

    ``startup_output`` is the startup turn's ``TurnOutput`` — the same public
    projection every other turn endpoint returns. Each command's own
    ``command_response``/``artifacts`` are still reachable under
    ``startup_output.command_outputs``.
    """
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # Access token expiration in seconds
    startup_output: Optional[fastworkflow.TurnOutput] = None  # Present if startup finished in-window
    startup_turn_key: Optional[str] = None  # Handle to poll the startup turn
    startup_exec_state: Optional[str] = None  # queued | running | done | lost
    startup_error: Optional[str] = None  # Present if the startup turn failed


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
    channel_id: ChannelId
    user_id: Optional[str] = None
    expires_days: int = 365


class CancelPendingRequest(BaseModel):
    """Optional body for /cancel_pending (channel from JWT)."""
    pass


# class CommandOutputWithTraces(BaseModel):
#     """CommandOutput extended with optional traces for HTTP responses"""
#     command_response: dict[str, Any]
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
    """Launch context only. The credential is installed per accepted turn.

    It used to be merged in here and refreshed on every lookup, which put a
    request-scoped secret into shared, checkpointable workflow state — readable
    by whichever turn happened to be running, and durable once state is
    persisted. See turns.installed_credential.
    """
    return context


def _run_startup_sync(
    ctx: WorkflowExecutionContext,
    startup_command: Optional[str],
    startup_action: Optional[fastworkflow.Action],
) -> Optional[fastworkflow.TurnOutput]:
    """Run startup as its own logical turn. Mirrors __main__._build_startup_work_fn."""
    if startup_action:
        if startup_action.workflow_id is None and ctx.app_workflow:
            startup_action.workflow_id = ctx.app_workflow.id
        return ctx.process_action_turn(startup_action)
    if startup_command:
        assistant_command = f"/{startup_command.lstrip('/')}"
        return ctx.process_turn(assistant_command)
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
        return

    if session_manager.admission_closed:
        # Creating one now would produce a session the drain has already scanned
        # past, so it would be neither drained nor closed.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server is shutting down; not creating new sessions",
        )

    # Single-flight session creation (§3.5): serialize per-channel so two
    # concurrent cold requests don't both build a ctx (and double-run startup).
    # Bind the lock to a local: the manager's mapping is weak-valued.
    creation_lock = session_manager.get_creation_lock(channel_id)
    async with creation_lock:
        # Re-check under the creation lock — another request may have created
        # the session while we waited.
        existing_runtime = await session_manager.get_session(channel_id)
        if existing_runtime:
            logger.debug(
                f"Session for channel_id {channel_id} created concurrently, skipping creation"
            )
            return

        # Hold the lease across creation so the overflow sweep inside
        # create_session cannot choose the runtime being created (invariant 20).
        async with session_manager.initialization_lease(channel_id):
            await _create_user_runtime(
                channel_id=channel_id,
                session_manager=session_manager,
                workflow_path=workflow_path,
                context=context,
                startup_command=startup_command,
                startup_action=startup_action,
                stream_format=stream_format,
                http_bearer_token=http_bearer_token,
                run_startup=run_startup,
            )


@dataclass
class _RestoredCheckpoint:
    """What a checkpoint contributed to a freshly built runtime."""

    session_incarnation: str
    runtime_fields: dict = field(default_factory=dict)
    startup: dict = field(default_factory=dict)
    applied: bool = False


def _restore_from_checkpoint(
    session_manager: "ChannelSessionManager",
    channel_id: str,
    workflow_path: str,
    app_workflow: fastworkflow.Workflow,
    launch_context: Optional[dict],
) -> _RestoredCheckpoint:
    """Apply this channel's checkpoint, or start from launch configuration.

    A record that cannot be applied is quarantined and the channel starts clean.
    Partially applying one would leave a session that looks restored and is not.
    """
    store = session_manager.checkpoint_store
    fresh_incarnation = checkpoint.new_session_incarnation()

    try:
        # Adoption, not continuation: a cold session has no way to know which
        # incarnation the stored record names until it reads it. Every other
        # identity field is still validated.
        record = store.load_for_adoption(
            deployment_id=checkpoint.deployment_id(),
            workflow_fingerprint=checkpoint.workflow_fingerprint(workflow_path),
            channel_id=channel_id,
        )
    except Exception as exc:
        logger.warning(
            f"Checkpoint for channel_id {channel_id} could not be read "
            f"({type(exc).__name__}); starting from launch configuration"
        )
        record = None

    if record is None:
        return _RestoredCheckpoint(session_incarnation=fresh_incarnation)

    try:
        runtime_fields = checkpoint.restore(
            app_workflow,
            record,
            current_launch=dict(launch_context or {}),
            channel_id=channel_id,
        )
    except Exception as exc:
        logger.error(
            f"Checkpoint for channel_id {channel_id} could not be applied "
            f"({type(exc).__name__}: {exc}); quarantining and starting from "
            "launch configuration"
        )
        store.quarantine(record.identity, QuarantineReason.UNREADABLE_RECORD)
        return _RestoredCheckpoint(session_incarnation=fresh_incarnation)

    logger.info(f"Restored checkpoint for channel_id {channel_id}")
    return _RestoredCheckpoint(
        session_incarnation=record.identity.session_incarnation,
        runtime_fields=runtime_fields,
        startup=dict(record.startup or {}),
        applied=True,
    )


async def _create_user_runtime(
    channel_id: str,
    session_manager: "ChannelSessionManager",
    workflow_path: str,
    context: Optional[dict],
    startup_command: Optional[str],
    startup_action: Optional["fastworkflow.Action"],
    stream_format: str,
    http_bearer_token: Optional[str],
    run_startup: bool,
) -> None:
    """Build and register a fresh Topology-B runtime (caller holds creation lock)."""
    context = _merge_workflow_context(context, http_bearer_token)
    logger.info(f"Creating new Topology-B session for channel_id: {channel_id}")

    conv_base_folder = get_channelconversations_dir(workflow_path)
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

    restored = _restore_from_checkpoint(
        session_manager, channel_id, workflow_path, app_workflow, context
    )

    conv_id_to_restore = None
    if conv_id_to_restore := conversation_store.get_last_conversation_id():
        conversation = conversation_store.get_conversation(conv_id_to_restore)
        if not conversation:
            conv_id_to_restore = conv_id_to_restore - 1
            conversation = conversation_store.get_conversation(conv_id_to_restore)
        if conversation:
            # Restore the same window the running session keeps, not the whole
            # conversation: the rest stays in the durable record.
            ctx._conversation_history = restore_history_from_turns(
                conversation["turns"][-MAX_CONVERSATION_TURNS_IN_MEMORY:]
            )
            logger.info(f"Restored conversation {conv_id_to_restore} for user {channel_id}")
        else:
            conv_id_to_restore = None

    loop = asyncio.get_running_loop()
    startup_state = restored.startup.get("state") or checkpoint.STARTUP_NOT_ATTEMPTED
    startup_epoch = int(restored.startup.get("epoch") or 0)
    startup_key = restored.startup.get("idempotency_key")
    startup_ran = bool(restored.runtime_fields.get("startup_ran"))

    # Whether startup already ran is read from the durable record, never from
    # in-process turn retention, so it cannot depend on a retention window.
    already_succeeded = startup_state == checkpoint.STARTUP_SUCCEEDED
    if run_startup and (startup_command or startup_action) and not already_succeeded:
        await loop.run_in_executor(
            None,
            lambda: _run_startup_sync(ctx, startup_command, startup_action),
        )
        startup_ran = True
        startup_state = checkpoint.STARTUP_SUCCEEDED
    elif already_succeeded:
        logger.info(
            f"Skipping startup for channel_id {channel_id}: the durable record "
            f"says it already succeeded at epoch {startup_epoch}"
        )

    if pending := session_manager.session_state_store.load(channel_id):
        try:
            ctx.apply_serialized_state(pending)
        except IncompatibleSessionState as exc:
            # Drop it rather than retrying forever: nothing about a later read
            # makes an unreadable version readable, and leaving it in place
            # would re-raise on every cold create for this channel.
            session_manager.session_state_store.clear(channel_id)
            logger.error(
                f"Discarded unreadable pending state for channel_id {channel_id}: "
                f"{exc}. The suspended turn is lost; the session starts clean."
            )
        else:
            logger.info(
                f"Restored pending suspended session for channel_id {channel_id}"
            )

    # Anything restored from a conversation is by definition already durable, so
    # the next incremental save must not append it again.
    durable_turn_count = (
        len(ctx.conversation_history.messages) if conv_id_to_restore else 0
    )

    # A restored record is authoritative for the fields the framework knows how
    # to restore; the request's stream_format only applies to a fresh session.
    await session_manager.create_session(
        channel_id=channel_id,
        execution_context=ctx,
        conversation_store=conversation_store,
        active_conversation_id=(
            restored.runtime_fields.get("active_conversation_id")
            if restored.applied
            else conv_id_to_restore
        ),
        stream_format=(
            restored.runtime_fields.get("stream_format") or stream_format
            if restored.applied
            else stream_format
        ),
        workflow_path=workflow_path,
        startup_ran=startup_ran,
        durable_turn_count=durable_turn_count,
        session_incarnation=restored.session_incarnation,
        startup_state=startup_state,
        startup_idempotency_key=startup_key,
        startup_epoch=startup_epoch,
    )
    logger.info(f"Successfully created session for channel_id: {channel_id}")


def get_channel_session_state_dir(workflow_path: str) -> str:
    """Workflow-namespaced folder for suspended Topology-B blobs (created)."""
    from fastworkflow import state_paths
    return state_paths.session_state_dir(workflow_path)


def get_channelconversations_dir(workflow_path: str) -> str:
    """Workflow-namespaced folder for per-channel conversation DBs (created)."""
    from fastworkflow import state_paths
    return state_paths.conversations_dir(workflow_path)


def _is_awaiting_user_output(output: Optional[fastworkflow.TurnOutput]) -> bool:
    """Read suspension off the turn's own status, not out of a command's artifacts.

    The artifacts key this used to inspect is a per-command detail that happens
    to correlate with suspension; ``TurnStatus`` is the turn-level statement of
    it, which is what a caller deciding whether to persist a suspended turn is
    actually asking about.
    """
    return (
        output is not None
        and output.status == fastworkflow.TurnStatus.AWAITING_USER
    )


def persist_pending_after_turn(
    session_manager: "ChannelSessionManager",
    runtime: "ChannelRuntime",
    output: Optional[fastworkflow.TurnOutput],
) -> None:
    """Save or clear durable suspended state after a Topology-B turn."""
    ctx = runtime.execution_context
    # has_open_command(): a mid-parameter-extraction session is not
    # awaiting_user but still holds partially extracted parameters (see the
    # twin in turns.persist_pending_after_turn).
    if ctx.awaiting_user or _is_awaiting_user_output(output) or ctx.has_open_command():
        try:
            state = ctx.serialize_state(channel_id=runtime.channel_id)
        except StateEncodingError as exc:
            logger.warning(
                f"Suspended state for channel_id {runtime.channel_id} is not "
                f"losslessly encodable, so it was not persisted: {exc}"
            )
            return
        session_manager.session_state_store.save(runtime.channel_id, state)
    else:
        session_manager.session_state_store.clear(runtime.channel_id)


async def run_process_turn(
    runtime: "ChannelRuntime",
    message: str,
    timeout_seconds: int,
    session_manager: "ChannelSessionManager",
) -> fastworkflow.TurnOutput:
    """Run one logical turn in a thread pool with timeout (Topology B)."""
    loop = asyncio.get_running_loop()
    ctx = runtime.execution_context

    def _run() -> fastworkflow.TurnOutput:
        # process_turn() is the public execution API: it returns the TurnOutput
        # the whole transport edge speaks (and captures the full turn).
        return ctx.process_turn(message)

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
) -> fastworkflow.TurnOutput:
    loop = asyncio.get_running_loop()
    ctx = runtime.execution_context

    def _run() -> fastworkflow.TurnOutput:
        return ctx.process_action_turn(action)

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


async def _maybe_await(result: Any) -> None:
    """Accept either a sync or an async callback."""
    if asyncio.iscoroutine(result):
        await result


async def _emit_trace_callback(
    on_trace: Callable[[dict[str, Any]], Any],
    trace_dict: dict[str, Any],
    _user_id: Optional[str],
) -> None:
    await _maybe_await(on_trace(trace_dict))


async def run_process_message_with_trace_stream(
    runtime: "ChannelRuntime",
    message: str,
    timeout_seconds: int,
    session_manager: "ChannelSessionManager",
    on_trace: Callable[[dict[str, Any]], Any],
    user_id: Optional[str] = None,
    on_timeout: Optional[Callable[[str], Any]] = None,
) -> fastworkflow.TurnOutput:
    """
    Run one logical turn in an executor while draining command_trace_queue concurrently.

    The deadline governs *delivery*, not ownership. When it passes, ``on_timeout``
    is invoked once so the caller can tell the client, but the executor future is
    still awaited to completion: abandoning it would leave a thread mutating the
    workflow context after the caller released ``runtime.lock``, free for eviction
    or shutdown to snapshot and close that context underneath it.
    """
    loop = asyncio.get_running_loop()
    ctx = runtime.execution_context
    trace_queue = ctx.command_trace_queue
    if trace_queue is None:
        return await run_process_turn(
            runtime, message, timeout_seconds, session_manager
        )

    exec_future = loop.run_in_executor(
        None, lambda: ctx.process_turn(message)
    )
    start = time.time()
    timed_out = False

    while not exec_future.done():
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
        if not timed_out and time.time() - start >= timeout_seconds:
            timed_out = True
            logger.warning(
                f"Streaming turn for channel_id {runtime.channel_id} passed its "
                f"{timeout_seconds}s delivery deadline; still owning the executor "
                "until it exits"
            )
            if on_timeout is not None:
                await _maybe_await(
                    on_timeout(
                        f"Command execution timed out after {timeout_seconds} seconds"
                    )
                )
        await asyncio.sleep(0.05)

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
    # Startup authority is durable (invariant 18): whether startup already ran
    # is read from the checkpoint, never inferred from in-process turn
    # retention, so it cannot depend on a wall-clock retention window.
    startup_state: str = checkpoint.STARTUP_NOT_ATTEMPTED
    startup_idempotency_key: Optional[str] = None
    startup_epoch: int = 0
    # Binds this live session to the checkpoint it may write. A record carrying
    # a different incarnation is a channel-id reuse, and is quarantined rather
    # than applied.
    session_incarnation: str = field(default_factory=checkpoint.new_session_incarnation)
    # High-water mark: how many of the leading in-memory conversation turns are
    # already durably recorded. Everything after it is what the next incremental
    # save appends. Trimming the in-memory window lowers it by the number of
    # turns dropped, so it stays an index into the live message list.
    durable_turn_count: int = 0
    # turn_key of the startup turn (if any), so the /initialize "already exists"
    # branch can return its three-state status (§3.3).
    startup_turn_key: Optional[str] = None

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
        max_live_sessions: int = DEFAULT_MAX_LIVE_SESSIONS,
    ):
        self._sessions: OrderedDict[str, ChannelRuntime] = OrderedDict()
        self._lock = asyncio.Lock()
        self._max_live_sessions = max_live_sessions
        self._max_live_sessions_source = "constructor"
        # A FastAPI process serves exactly one workflow; this is set at server
        # startup (before any store access) so the lazily-built stores below can
        # namespace their paths under that workflow. See set_workflow_path.
        self.workflow_path: str = ""
        # Built lazily on first access so the state-root read happens after
        # fastworkflow.init() loads the env file, not at module-import time.
        self._session_state_store = session_state_store
        # Per-channel creation guard for single-flight session creation: two
        # concurrent cold requests for the same channel must not both build a
        # ctx (wasted work / double startup). Keyed by channel_id; dict access
        # is atomic in the single event loop.
        #
        # Weak values, because a channel_id seen once would otherwise keep its
        # lock forever. The holder and every queued waiter keep a strong
        # reference for as long as the lock matters, so the entry only
        # disappears once nobody is using it. Never prune it by hand: a waiter
        # can be awake but not yet resumed, and dropping the shared lock in that
        # window lets the next caller create a second one and lose single-flight.
        self._creation_locks: "weakref.WeakValueDictionary[str, asyncio.Lock]" = (
            weakref.WeakValueDictionary()
        )
        # Optional predicate (channel_id -> bool) wired by the server to the
        # turn registry's active-execution pointer. Eviction must never close a
        # live turn's ctx, so a busy channel is skipped. See §3.6 of the design.
        self.is_channel_busy: Optional[Callable[[str], bool]] = None
        # Per-channel eviction leases (refcounts). A lease is taken under the
        # manager lock by the same lookup that hands out the runtime, which is
        # what makes lookup and admission atomic. A boolean sampled afterwards
        # cannot: get_session() releases the manager lock before the caller has
        # registered a turn or taken runtime.lock, and in that window both halves
        # of the busy predicate are false while the runtime is very much in use.
        # Keyed by channel_id rather than by runtime, so a channel can be leased
        # while it is still being created.
        self._leases: dict[str, int] = {}
        self._checkpoint_store: Optional[ChannelCheckpointStore] = None
        self._admission_closed = False
        # The configured initial context, kept so restore can tell an operator's
        # change from an application's (three-way merge, design 11.9).
        self._launch_context: dict = {}

    @asynccontextmanager
    async def leased_session(self, channel_id: str):
        """Look up a runtime and hold an eviction lease for the whole block.

        This is the safe way to obtain a runtime you are going to *use*. The
        lease outlives the manager lock and is released only when the block
        exits, by which time either the turn registry or ``runtime.lock`` owns
        the runtime — so there is no interval where a live runtime looks idle.

        Yields None when the channel has no live session, so callers keep their
        existing not-found handling.
        """
        async with self._lock:
            runtime = self._sessions.get(channel_id)
            if runtime is not None:
                self._touch(channel_id)
                self._acquire_lease(channel_id)
        try:
            yield runtime
        finally:
            if runtime is not None:
                self._release_lease(channel_id)

    @asynccontextmanager
    async def initialization_lease(self, channel_id: str):
        """Hold a lease across session creation, before the session exists.

        A runtime under construction is the *only* apparently safe eviction
        candidate exactly when every older session is pinned — it has no registry
        pointer, its lock is free, and its workflow has no context object yet. So
        the manager can evict the runtime it is in the middle of creating, inside
        create_session's own overflow sweep. Leasing the channel_id before
        creation starts closes that window.
        """
        async with self._lock:
            self._acquire_lease(channel_id)
        try:
            yield
        finally:
            self._release_lease(channel_id)

    def _acquire_lease(self, channel_id: str) -> None:
        """Caller holds ``self._lock``."""
        self._leases[channel_id] = self._leases.get(channel_id, 0) + 1

    def _release_lease(self, channel_id: str) -> None:
        remaining = self._leases.get(channel_id, 0) - 1
        if remaining > 0:
            self._leases[channel_id] = remaining
        else:
            self._leases.pop(channel_id, None)

    def close_admission(self) -> None:
        """Stop creating sessions. Closing the registry alone is not enough.

        A channel created concurrently with shutdown is in neither the registry
        nor ``_sessions`` when the drain scans, so it would be neither drained
        nor closed.
        """
        self._admission_closed = True

    @property
    def admission_closed(self) -> bool:
        return self._admission_closed

    def busy_channel_ids(self) -> list[str]:
        """Channels that are leased or have work in flight.

        The shutdown drain needs this rather than ``runtime.lock.locked()``: a
        QUEUED execution whose task has not yet taken the lock reports not-busy,
        and so does a request that has been handed a runtime but not yet
        registered its turn.

        Leases are scanned separately from ``_sessions`` because a channel being
        created holds one before it has a session to be found under.
        """
        candidates = set(self._sessions) | set(self._leases)
        return sorted(
            channel_id
            for channel_id in candidates
            if self._leases.get(channel_id) or self._has_work_in_flight(channel_id)
        )

    def _has_work_in_flight(self, channel_id: str) -> bool:
        """Union predicate: a live registry execution OR a held runtime lock.

        The registry pointer alone is not enough, because /invoke_agent_stream
        runs an entire turn without ever creating a TurnExecution and guards
        itself with the lock instead. Invariant 2 still forbids the lock as the
        409 idempotency source; eviction safety is a different question, and the
        lock answers it correctly where the pointer does not.
        """
        if self.is_channel_busy and self.is_channel_busy(channel_id):
            return True
        runtime = self._sessions.get(channel_id)
        return runtime is not None and runtime.lock.locked()

    def get_creation_lock(self, channel_id: str) -> asyncio.Lock:
        """Return the per-channel creation lock, creating it on first use.

        Callers must hold on to the returned lock for as long as they need it;
        the mapping only keeps a weak reference.
        """
        lock = self._creation_locks.get(channel_id)
        if lock is None:
            lock = asyncio.Lock()
            self._creation_locks[channel_id] = lock
        return lock

    @property
    def max_live_sessions(self) -> int:
        return self._max_live_sessions

    @property
    def max_live_sessions_source(self) -> str:
        return self._max_live_sessions_source

    def configure_max_live_sessions(self) -> tuple[int, str]:
        """Apply the resolver. Must run after fastworkflow.init() loads env files."""
        value, source = resolve_max_live_sessions()
        self._max_live_sessions = value
        self._max_live_sessions_source = source
        return value, source

    def reap_checkpoints(self, policy: Optional[RetentionPolicy] = None) -> Any:
        """Reclaim abandoned checkpoints, protecting every channel we hold live.

        The store has no timer of its own — a background sweeper inside it would
        be a second writer per channel — so the server drives retention and names
        what it must not touch.
        """
        return self.checkpoint_store.reap(
            policy or RetentionPolicy(),
            protected_channel_ids=set(self._sessions) | set(self._leases),
        )

    def reap_pending_state(
        self, policy: Optional[PendingRetentionPolicy] = None
    ) -> PendingReapOutcome:
        """Reclaim abandoned suspended sessions, protecting every channel we hold.

        A suspended session waits on a user who may never return, and since
        v2.28.0 removed the pin it can be evicted — which means the blob is then
        the only copy of that conversation. Protecting live and leased channels
        is what keeps this a reaper of abandoned state rather than of state
        someone is still using.
        """
        return self.session_state_store.reap(
            policy or PendingRetentionPolicy(),
            protected_channel_ids=set(self._sessions) | set(self._leases),
        )

    @property
    def session_state_store(self) -> SessionStateStore:
        if self._session_state_store is None:
            self._session_state_store = get_session_state_store(
                base_folder=get_channel_session_state_dir(self.workflow_path)
            )
        return self._session_state_store

    @property
    def checkpoint_store(self) -> ChannelCheckpointStore:
        # Built lazily for the same reason as the state store: the state-root
        # read must happen after fastworkflow.init().
        if self._checkpoint_store is None:
            self._checkpoint_store = ChannelCheckpointStore(
                checkpoint.get_checkpoint_dir(self.workflow_path),
                min_readable_protocol_version=checkpoint.fleet_protocol_floor(),
            )
        return self._checkpoint_store

    def set_launch_context(self, launch_context: Optional[dict]) -> None:
        """The configured initial context, needed for three-way reconciliation."""
        self._launch_context = dict(launch_context or {})

    def checkpoint_for_shutdown(self, skip_channel_ids: list[str]) -> int:
        """Publish checkpoints for quiescent runtimes before the process exits.

        Retirement writes on eviction; without this, a clean restart loses
        everything a live channel accumulated since it was last evicted — the
        same loss the deadline rule guards against, reached by writing nothing
        rather than by writing something stale. Busy channels are skipped for
        exactly that reason: their snapshot would predate work still running.
        """
        skip = set(skip_channel_ids)
        written = 0
        for channel_id, runtime in list(self._sessions.items()):
            if channel_id in skip:
                continue
            eligibility = checkpoint.assess(runtime, self.session_state_store)
            if not eligibility.evictable:
                continue
            try:
                checkpoint.publish(
                    self.checkpoint_store, runtime, eligibility, self._launch_context
                )
                written += 1
            except Exception as exc:
                logger.warning(
                    f"Could not checkpoint channel_id {channel_id} at shutdown "
                    f"({type(exc).__name__}: {exc})"
                )
        return written

    def commit_startup_state(self, runtime: ChannelRuntime) -> None:
        """Persist the startup outcome now, not at retirement.

        Invariant 25: a fact whose loss changes restart behaviour is committed
        before it becomes observable, and its durability must not depend on
        whether the application happened to mutate its context.
        """
        eligibility = checkpoint.assess(runtime, self.session_state_store)
        if not eligibility.evictable:
            # Nothing durable to attach it to. The session is pinned anyway, so
            # it will not be evicted and the fact stays in memory where it is
            # still correct for this process.
            return
        checkpoint.publish(
            self.checkpoint_store, runtime, eligibility, self._launch_context
        )

    def _touch(self, channel_id: str) -> None:
        if channel_id in self._sessions:
            self._sessions.move_to_end(channel_id)

    async def trim_live_sessions(self) -> None:
        """Bring the cache back to target after a turn frees a channel up.

        Creation is not the only moment the cache can be over target: a channel
        that was skipped because it was busy becomes retirable the moment its
        turn ends, and nothing else would revisit it.
        """
        async with self._lock:
            await self._evict_oldest_if_needed()

    async def _evict_oldest_if_needed(self) -> None:
        while len(self._sessions) > self._max_live_sessions:
            # Never close a live turn's ctx mid-mutation (§3.6) — eviction would
            # race the executor thread — and never evict a session whose state
            # cannot be written back. OrderedDict iterates oldest-first.
            if not await self._retire_one_candidate():
                break

    async def _retire_one_candidate(self) -> bool:
        """Retire the oldest retirable session. False when none can be.

        Retirement is publish-then-pop. Popping first and writing "best effort"
        afterwards is how a failed write becomes lost application state.
        """
        skipped_pinned = 0
        for channel_id, runtime in list(self._sessions.items()):
            if self._leases.get(channel_id) or self._has_work_in_flight(channel_id):
                continue

            eligibility = checkpoint.assess(runtime, self.session_state_store)
            if not eligibility.evictable:
                skipped_pinned += 1
                checkpoint.warn_pinned_once(
                    channel_id, runtime.workflow_path, eligibility.reason
                )
                continue

            try:
                generation = checkpoint.publish(
                    self.checkpoint_store, runtime, eligibility, self._launch_context
                )
            except (StateEncodingError, CheckpointStoreError, OSError) as exc:
                # Leave it live and try the next candidate. The cache staying
                # over target is a visible, metered condition; a dropped runtime
                # whose state never landed is not.
                logger.warning(
                    f"Not evicting channel_id {channel_id}: checkpoint write "
                    f"failed ({type(exc).__name__}: {exc})"
                )
                continue

            self._sessions.pop(channel_id, None)
            runtime.execution_context.close()
            logger.debug(
                f"Retired channel_id {channel_id} at generation {generation}"
            )
            return True

        logger.warning(
            "Session cache over capacity but no candidate could be retired "
            f"({skipped_pinned} pinned); staying over target"
        )
        return False

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
        durable_turn_count: int = 0,
        session_incarnation: Optional[str] = None,
        startup_state: str = checkpoint.STARTUP_NOT_ATTEMPTED,
        startup_idempotency_key: Optional[str] = None,
        startup_epoch: int = 0,
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
                durable_turn_count=durable_turn_count,
                session_incarnation=(
                    session_incarnation or checkpoint.new_session_incarnation()
                ),
                startup_state=startup_state,
                startup_idempotency_key=startup_idempotency_key,
                startup_epoch=startup_epoch,
            )
            self._sessions[channel_id] = runtime
            self._touch(channel_id)
            await self._evict_oldest_if_needed()
            return runtime

    async def remove_session(self, channel_id: str) -> None:
        async with self._lock:
            if runtime := self._sessions.pop(channel_id, None):
                runtime.execution_context.close()

    async def evict_live_session(self, channel_id: str) -> None:
        """Drop from process cache without clearing durable pending state."""
        await self.remove_session(channel_id)


# ============================================================================
# Helper Functions
# ============================================================================

def save_conversation_incremental(runtime: ChannelRuntime, extract_turns_func, logger) -> int:
    """
    Save conversation turns incrementally after each turn (without generating topic/summary).
    This provides crash protection - all turns except the last will be preserved.

    Only turns above the runtime's high-water mark are appended, then the
    in-memory history is windowed. The order matters: a turn is durably recorded
    before it can be dropped from memory, so trimming never shortens the durable
    record. Returns the number of turns appended.
    """
    turns = extract_turns_func(runtime.execution_context.conversation_history)
    already_saved = runtime.durable_turn_count
    if already_saved > len(turns):
        # The mark outran the history, so it was established against some other
        # history. Re-append everything: a duplicated turn is recoverable, a
        # turn dropped from memory that was never recorded is not.
        logger.warning(
            f"Conversation high-water mark ({already_saved}) exceeds in-memory "
            f"turns ({len(turns)}) for channel_id {runtime.channel_id}; "
            "re-appending from the start rather than risking an unrecorded turn"
        )
        already_saved = 0

    if new_turns := turns[already_saved:]:
        # Initialize conversation ID for first conversation if needed
        if runtime.active_conversation_id == 0:
            # This is the first conversation for this session
            # Reserve ID 1 and use it
            runtime.active_conversation_id = runtime.conversation_store.reserve_next_conversation_id()
            logger.debug(f"Initialized first conversation with ID {runtime.active_conversation_id} for user {runtime.channel_id}")

        runtime.conversation_store.append_conversation_turns(
            runtime.active_conversation_id, new_turns
        )
        runtime.durable_turn_count = len(turns)
        logger.debug(f"Incrementally saved {len(new_turns)} turn(s) to conversation {runtime.active_conversation_id}")

    trim_conversation_window(runtime, logger)
    return len(new_turns)


def trim_conversation_window(runtime: ChannelRuntime, logger) -> int:
    """Window the in-memory conversation history, keeping the high-water mark aligned.

    Only safe to call once the turns being dropped are durably recorded.
    """
    trimmed = runtime.execution_context.trim_conversation_history(
        MAX_CONVERSATION_TURNS_IN_MEMORY
    )
    if trimmed:
        runtime.durable_turn_count = max(0, runtime.durable_turn_count - trimmed)
        logger.debug(
            f"Trimmed {trimmed} in-memory conversation turn(s) for channel_id "
            f"{runtime.channel_id} (window={MAX_CONVERSATION_TURNS_IN_MEMORY})"
        )
    return trimmed


def save_last_turn_feedback(runtime: ChannelRuntime, extract_turns_func, logger) -> None:
    """Persist an edit to the newest conversation turn (feedback).

    The append path cannot express a change to a turn that is already recorded,
    so rewrite that one turn in place. If the turn is not durable yet, the normal
    incremental save carries the edit with it.
    """
    if save_conversation_incremental(runtime, extract_turns_func, logger):
        return

    turns = extract_turns_func(runtime.execution_context.conversation_history)
    if not turns or runtime.active_conversation_id == 0:
        return

    # Rewriting by position is only meaningful if this conversation is the one
    # the mark was established against. A restored suspended session can leave
    # the two disagreeing, and writing blind would overwrite an unrelated turn.
    durable_turns = runtime.conversation_store.count_conversation_turns(
        runtime.active_conversation_id
    )
    if not runtime.durable_turn_count or durable_turns < runtime.durable_turn_count:
        logger.warning(
            f"Skipping feedback write for channel_id {runtime.channel_id}: "
            f"conversation {runtime.active_conversation_id} holds {durable_turns} "
            f"turn(s), which does not match the {runtime.durable_turn_count} "
            "recorded in memory"
        )
        return

    runtime.conversation_store.update_last_conversation_turn(
        runtime.active_conversation_id, turns[-1]
    )
    logger.debug(
        f"Updated latest durable turn of conversation "
        f"{runtime.active_conversation_id} for channel_id {runtime.channel_id}"
    )


