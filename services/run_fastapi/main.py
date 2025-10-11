"""
FastAPI application for FastWorkflow
Exposes FastWorkflow workflows as HTTP endpoints with synchronous and streaming execution

Implementation Status:
- ✅ All endpoints implemented per spec
- ✅ Session management and concurrency control  
- ✅ Rdict-backed conversation persistence
- ✅ Agent trace collection and inclusion in responses
- ✅ SSE streaming for real-time trace events (/invoke_agent_stream)
- ✅ Error handling with proper HTTP status codes
- ✅ Conversation history extraction and restoration
- ✅ Session resume with conversation_id support
- ✅ Direct action execution (bypasses parameter extraction)
- ✅ Graceful shutdown with configurable timeout
- ✅ Complete conversation dump (all users, active or not)

See docs/fastworkflow_fastapi_spec.md for complete specification.
"""

import asyncio
import json
import os
import queue
import time
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional

import dspy
import uvicorn
from dotenv import dotenv_values
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, field_validator

import fastworkflow
from services.conversation_store import (
    ConversationStore,
    ConversationSummary,
    generate_topic_and_summary,
    extract_turns_from_history,
    restore_history_from_turns
)
from fastworkflow.utils.logging import logger

# ============================================================================
# Data Models (aligned with FastWorkflow canonical types)
# ============================================================================

# Import canonical types from fastworkflow
Action = fastworkflow.Action
CommandResponse = fastworkflow.CommandResponse
CommandOutput = fastworkflow.CommandOutput


class InitializationRequest(BaseModel):
    """Request to initialize a FastWorkflow session for a user"""
    user_id: str
    workflow_path: str
    env_file_path: Optional[str] = None
    passwords_file_path: Optional[str] = None
    context: Optional[dict[str, Any]] = None
    startup_command: Optional[str] = None
    startup_action: Optional[dict[str, Any]] = None  # Will be converted to Action
    show_agent_traces: bool = True
    conversation_id: Optional[int] = None  # If provided, restore this specific conversation

    @field_validator('startup_command', 'startup_action')
    @classmethod
    def validate_startup_mutual_exclusion(cls, v, info):
        """Ensure startup_command and startup_action are mutually exclusive"""
        if info.field_name == 'startup_action' and v is not None and info.data.get('startup_command'):
            raise ValueError("Cannot provide both startup_command and startup_action")
        return v


class InitializationResponse(BaseModel):
    """Response from initialization"""
    user_id: str


class InvokeRequest(BaseModel):
    """Request to invoke agent or assistant"""
    user_id: str
    user_query: str
    timeout_seconds: int = 60


class PerformActionRequest(BaseModel):
    """Request to perform a specific action"""
    user_id: str
    action: dict[str, Any]  # Will be converted to Action
    timeout_seconds: int = 60


class NewConversationRequest(BaseModel):
    """Request to start a new conversation"""
    user_id: str


class PostFeedbackRequest(BaseModel):
    """Request to post feedback on the latest turn"""
    user_id: str
    binary_or_numeric_score: Optional[bool | float] = None
    nl_feedback: Optional[str] = None

    @field_validator('nl_feedback')
    @classmethod
    def validate_feedback_presence(cls, v, info):
        """Ensure at least one feedback field is provided"""
        if v is None and info.data.get('binary_or_numeric_score') is None:
            raise ValueError("At least one of binary_or_numeric_score or nl_feedback must be provided")
        return v


class ActivateConversationRequest(BaseModel):
    """Request to activate a conversation by ID"""
    user_id: str
    conversation_id: int


class DumpConversationsRequest(BaseModel):
    """Admin request to dump all conversations"""
    output_folder: str


class CommandOutputWithTraces(BaseModel):
    """CommandOutput extended with optional traces for HTTP responses"""
    command_responses: list[dict[str, Any]]
    workflow_name: str = ""
    context: str = ""
    command_name: str = ""
    command_parameters: str = ""
    success: bool = True
    traces: Optional[list[dict[str, Any]]] = None


# ============================================================================
# Session Management
# ============================================================================

@dataclass
class UserRuntime:
    """Per-user runtime state"""
    user_id: str
    active_conversation_id: int
    chat_session: fastworkflow.ChatSession
    lock: asyncio.Lock
    show_agent_traces: bool
    conversation_store: 'ConversationStore'


class UserSessionManager:
    """Process-wide manager for user sessions"""
    
    def __init__(self):
        self._sessions: dict[str, UserRuntime] = {}
        self._lock = asyncio.Lock()
    
    async def get_session(self, user_id: str) -> Optional[UserRuntime]:
        """Get a user's session"""
        async with self._lock:
            return self._sessions.get(user_id)
    
    async def create_session(
        self,
        user_id: str,
        chat_session: fastworkflow.ChatSession,
        show_agent_traces: bool,
        conversation_store: 'ConversationStore',
        active_conversation_id: Optional[int] = None
    ) -> UserRuntime:
        """Create or update a user session"""
        async with self._lock:            
            runtime = UserRuntime(
                user_id=user_id,
                active_conversation_id=active_conversation_id or 0,
                chat_session=chat_session,
                lock=asyncio.Lock(),
                show_agent_traces=show_agent_traces,
                conversation_store=conversation_store
            )
            self._sessions[user_id] = runtime
            return runtime
    
    async def remove_session(self, user_id: str) -> None:
        """Remove a user session"""
        async with self._lock:
            if user_id in self._sessions:
                del self._sessions[user_id]


# Global session manager
session_manager = UserSessionManager()


# ============================================================================
# Helper Functions
# ============================================================================

def load_env_from_files(env_file_path: Optional[str], passwords_file_path: Optional[str]) -> dict[str, str]:
    """Load environment variables from specified files only"""
    env_vars = {}

    if env_file_path:
        if not os.path.isfile(env_file_path):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"env_file_path does not exist: {env_file_path}"
            )
        env_vars |= dotenv_values(env_file_path)

    if passwords_file_path:
        if not os.path.isfile(passwords_file_path):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"passwords_file_path does not exist: {passwords_file_path}"
            )
        env_vars.update(dotenv_values(passwords_file_path))

    return env_vars


async def wait_for_command_output(
    runtime: UserRuntime,
    timeout_seconds: int
) -> CommandOutput:
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


def collect_trace_events(runtime: UserRuntime) -> list[dict[str, Any]]:
    """Drain and collect all trace events from the queue"""
    traces = []
    
    while True:
        try:
            evt = runtime.chat_session.command_trace_queue.get_nowait()
            traces.append({
                "direction": evt.direction.value if hasattr(evt.direction, 'value') else str(evt.direction),
                "raw_command": evt.raw_command,
                "command_name": evt.command_name,
                "parameters": evt.parameters,
                "response_text": evt.response_text,
                "success": evt.success,
                "timestamp_ms": evt.timestamp_ms
            })
        except queue.Empty:
            break
    
    return traces


def save_conversation_incremental(runtime: UserRuntime) -> None:
    """
    Save conversation turns incrementally after each turn (without generating topic/summary).
    This provides crash protection - all turns except the last will be preserved.
    """
    try:
        # Extract turns from conversation history
        if turns := extract_turns_from_history(runtime.chat_session.conversation_history):
            # Initialize conversation ID for first conversation if needed
            if runtime.active_conversation_id == 0:
                # This is the first conversation for this user
                # Reserve ID 1 and use it
                runtime.active_conversation_id = runtime.conversation_store.reserve_next_conversation_id()
                logger.debug(f"Initialized first conversation with ID {runtime.active_conversation_id}")
            
            # Save turns using the active conversation ID
            runtime.conversation_store.save_conversation_turns(
                runtime.active_conversation_id, turns
            )
            logger.debug(f"Incrementally saved {len(turns)} turn(s) to conversation {runtime.active_conversation_id}")
    except Exception as e:
        # Log but don't fail the request if incremental save fails
        logger.warning(f"Failed to incrementally save conversation: {e}")


# ============================================================================
# FastAPI App Setup
# ============================================================================

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Startup and shutdown hooks"""
    logger.info("FastWorkflow FastAPI service starting...")
    yield
    logger.info("FastWorkflow FastAPI service shutting down...")
    
    # Graceful shutdown: wait for in-flight turns to complete
    max_wait = int(os.getenv("FASTWORKFLOW_SHUTDOWN_MAX_WAIT_SECONDS", "30"))
    logger.info(f"Waiting up to {max_wait}s for active turns to complete...")
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        # Check if any users have active turns (locks held)
        active_turns = []
        for user_id in list(session_manager._sessions.keys()):
            runtime = await session_manager.get_session(user_id)
            if runtime and runtime.lock.locked():
                active_turns.append(user_id)
        
        if not active_turns:
            logger.info("All turns completed, shutting down gracefully")
            break
        
        logger.debug(f"Waiting for {len(active_turns)} active turns: {active_turns}")
        await asyncio.sleep(0.5)
    else:
        logger.warning(f"Shutdown timeout reached with {len(active_turns)} turns still active")
    
    # Finalize conversations with topic/summary before shutdown (turns already saved incrementally)
    logger.info("Finalizing conversations with topic and summary...")
    for user_id in list(session_manager._sessions.keys()):
        runtime = await session_manager.get_session(user_id)
        if runtime:
            # Check if there are turns in the conversation history (turns already saved, just need topic/summary)
            if turns := extract_turns_from_history(runtime.chat_session.conversation_history):
                try:
                    # Generate topic and summary (turns already persisted incrementally)
                    topic, summary = generate_topic_and_summary(turns)
                    
                    # Update topic/summary for the conversation
                    if runtime.active_conversation_id > 0:
                        runtime.conversation_store.update_conversation_topic_summary(
                            runtime.active_conversation_id, topic, summary
                        )
                        logger.info(f"Finalized conversation {runtime.active_conversation_id} for user {user_id} during shutdown")
                    else:
                        # Edge case: shouldn't happen with incremental saves, but handle it
                        logger.warning(f"Conversation history exists but no active_conversation_id for user {user_id} during shutdown")
                        conv_id = runtime.conversation_store.save_conversation(topic, summary, turns)
                        logger.info(f"Created conversation {conv_id} for user {user_id} during shutdown")
                except Exception as e:
                    logger.error(f"Failed to finalize conversation for user {user_id} during shutdown: {e}")
    
    # Stop all chat sessions
    for user_id in list(session_manager._sessions.keys()):
        runtime = await session_manager.get_session(user_id)
        if runtime:
            runtime.chat_session.stop_workflow()
    
    logger.info("FastWorkflow FastAPI service shutdown complete")


app = FastAPI(
    title="FastWorkflow API",
    description="HTTP interface for FastWorkflow workflows",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with health check and docs link"""
    return """
    <html>
        <head>
            <title>FastWorkflow API</title>
        </head>
        <body>
            <h1>FastWorkflow API is running!</h1>
            <p>For API documentation, visit <a href="/docs">/docs</a></p>
        </body>
    </html>
    """


@app.post(
    "/initialize",
    response_model=InitializationResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Session successfully initialized"},
        400: {"description": "Both startup_command and startup_action provided"},
        422: {"description": "Invalid paths or missing env vars"},
        500: {"description": "Internal error during initialization"}
    }
)
async def initialize(request: InitializationRequest) -> InitializationResponse:
    """
    Initialize a FastWorkflow session for a user.
    Creates or resumes a ChatSession and starts the workflow.
    """
    try:
        logger.info(f"Initializing session for user_id: {request.user_id}")

        # Check if user already has an active session
        existing_runtime = await session_manager.get_session(request.user_id)
        if existing_runtime:
            logger.info(f"User {request.user_id} already has an active session, skipping initialization")
            return InitializationResponse(user_id=request.user_id)

        # Validate workflow path
        if not os.path.isdir(request.workflow_path):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"workflow_path is not a valid directory: {request.workflow_path}"
            )

        commands_dir = os.path.join(request.workflow_path, "_commands")
        if not os.path.isdir(commands_dir):
            logger.warning(f"No _commands directory found at {request.workflow_path}")

        # Load environment variables from files only
        env_vars = load_env_from_files(request.env_file_path, request.passwords_file_path)

        # Initialize fastworkflow with loaded env
        fastworkflow.init(env_vars=env_vars)

        # Get conversation store base folder from env
        conv_base_folder = env_vars.get(
            "USER_CONVERSATIONS_SPEEDDICT_FOLDERNAME",
            os.path.join(os.getcwd(), "___user_conversations")
        )

        # Create conversation store for this user
        conversation_store = ConversationStore(request.user_id, conv_base_folder)

        # Create ChatSession in agent mode (forced)
        chat_session = fastworkflow.ChatSession(run_as_agent=True)

        # Determine which conversation to restore (if any)
        # Spec: "If conversation_id provided and exists, restore its history; else restore last; else start new"
        conv_id_to_restore = None
        if request.conversation_id is not None:
            # Specific conversation requested
            conv_id_to_restore = request.conversation_id
        else:
            # No specific conversation, try to restore the last one
            conv_id_to_restore = conversation_store.get_last_conversation_id()

        # Attempt to restore the conversation
        if conv_id_to_restore:
            conversation = conversation_store.get_conversation(conv_id_to_restore)            
            if not conversation:
                # this means a new conversation was started but not saved
                conv_id_to_restore = conv_id_to_restore-1
                conversation = conversation_store.get_conversation(conv_id_to_restore)
            
            if conversation:
                # Restore the conversation history from saved turns
                restored_history = restore_history_from_turns(conversation["turns"])
                chat_session._conversation_history = restored_history
                logger.info(f"Restored conversation {conv_id_to_restore} for user {request.user_id}")
            else:
                logger.info(f"No conversations available for user {request.user_id}, starting new")

        # Prepare startup action if provided
        startup_action = None
        if request.startup_action and request.startup_action.get("command_name"):
            startup_action = Action(**request.startup_action)

        # Start the workflow
        chat_session.start_workflow(
            workflow_folderpath=request.workflow_path,
            workflow_context=request.context,
            startup_command=request.startup_command,
            startup_action=startup_action,
            keep_alive=True
        )

        # Create and store user runtime
        await session_manager.create_session(
            user_id=request.user_id,
            chat_session=chat_session,
            show_agent_traces=request.show_agent_traces,
            conversation_store=conversation_store,
            active_conversation_id=conv_id_to_restore
        )

        logger.info(f"Successfully initialized session for user_id: {request.user_id}")
        return InitializationResponse(user_id=request.user_id)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initializing session for user {request.user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error in initialize() for user_id: {request.user_id}",
        ) from e


@app.post(
    "/invoke_agent",
    response_model=None,  # Use custom response to include traces
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Agent query processed successfully"},
        404: {"description": "User session not found"},
        409: {"description": "Concurrent turn already in progress"},
        504: {"description": "Command execution timed out"}
    }
)
async def invoke_agent(request: InvokeRequest) -> JSONResponse:
    """
    Submit a natural language query to the agent.
    Leading '/' characters are stripped for compatibility.
    """
    try:
        runtime = await session_manager.get_session(request.user_id)
        if not runtime:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User session not found: {request.user_id}"
            )

        # Serialize turns per user
        if runtime.lock.locked():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"A turn is already in progress for user: {request.user_id}"
            )

        async with runtime.lock:
            # Strip leading slashes from user query
            user_query = request.user_query.lstrip('/')

            # Enqueue the user message
            runtime.chat_session.user_message_queue.put(user_query)

            # Wait for command output
            command_output = await wait_for_command_output(runtime, request.timeout_seconds)

            # Incrementally save conversation turns (without generating topic/summary)
            save_conversation_incremental(runtime)

            traces = collect_trace_events(runtime) if runtime.show_agent_traces else None
            # Build response with traces
            response_data = command_output.model_dump()
            if traces is not None:
                response_data["traces"] = traces

            return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in invoke_agent for user {request.user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error in invoke_agent() for user_id: {request.user_id}",
        ) from e


@app.post(
    "/invoke_agent_stream",
    responses={
        200: {
            "description": "SSE stream with trace events and final command output",
            "content": {"text/event-stream": {}}
        },
        404: {"description": "User session not found"},
        409: {"description": "Concurrent turn already in progress"},
        504: {"description": "Command execution timed out"}
    }
)
async def invoke_agent_stream(request: InvokeRequest):
    """
    Submit a natural language query to the agent and stream trace events via SSE.
    Leading '/' characters are stripped for compatibility.
    
    SSE Events:
    - event: trace, data: <trace_json> - emitted for each trace event (if show_agent_traces=true)
    - event: command_output, data: <CommandOutput_json> - final result
    - event: error, data: <error_json> - error occurred during execution
    """
    
    async def event_generator():
        """Generate SSE events for traces and final output"""
        try:
            runtime = await session_manager.get_session(request.user_id)
            if not runtime:
                error_data = json.dumps({"detail": f"User session not found: {request.user_id}"})
                yield f"event: error\ndata: {error_data}\n\n"
                return
            
            # Serialize turns per user
            if runtime.lock.locked():
                error_data = json.dumps({"detail": f"A turn is already in progress for user: {request.user_id}"})
                yield f"event: error\ndata: {error_data}\n\n"
                return
            
            async with runtime.lock:
                # Strip leading slashes from user query
                user_query = request.user_query.lstrip('/')
                
                # Enqueue the user message
                runtime.chat_session.user_message_queue.put(user_query)
                
                # Helper function to format trace event (DRY)
                def format_trace_event(evt) -> str:
                    trace_data = {
                        "direction": evt.direction.value if hasattr(evt.direction, 'value') else str(evt.direction),
                        "raw_command": evt.raw_command,
                        "command_name": evt.command_name,
                        "parameters": evt.parameters,
                        "response_text": evt.response_text,
                        "success": evt.success,
                        "timestamp_ms": evt.timestamp_ms
                    }
                    return f"event: trace\ndata: {json.dumps(trace_data)}\n\n"
                
                # Wait for command output, streaming traces as they arrive
                # Pattern matches CLI implementation in run/__main__.py (lines 229-247)
                start_time = time.time()
                command_output = None
                
                while time.time() - start_time < request.timeout_seconds:
                    # Drain all available trace events (like CLI does)
                    if runtime.show_agent_traces:
                        while True:
                            try:
                                evt = runtime.chat_session.command_trace_queue.get_nowait()
                                yield format_trace_event(evt)
                            except queue.Empty:
                                break
                    
                    # Check for command output
                    try:
                        command_output = runtime.chat_session.command_output_queue.get_nowait()
                        break
                    except queue.Empty:
                        await asyncio.sleep(0.1)
                        continue
                
                # Drain any remaining trace events after command completes
                if runtime.show_agent_traces:
                    while True:
                        try:
                            evt = runtime.chat_session.command_trace_queue.get_nowait()
                            yield format_trace_event(evt)
                        except queue.Empty:
                            break
                
                # Check if we timed out
                if command_output is None:
                    error_data = json.dumps({"detail": f"Command execution timed out after {request.timeout_seconds} seconds"})
                    yield f"event: error\ndata: {error_data}\n\n"
                    return
                
                # Incrementally save conversation turns (without generating topic/summary)
                save_conversation_incremental(runtime)
                
                # Emit final command output
                output_data = command_output.model_dump()
                yield f"event: command_output\ndata: {json.dumps(output_data)}\n\n"
        
        except Exception as e:
            logger.error(f"Error in invoke_agent_stream for user {request.user_id}: {e}")
            traceback.print_exc()
            error_data = json.dumps({"detail": f"Internal error in invoke_agent_stream() for user_id: {request.user_id}"})
            yield f"event: error\ndata: {error_data}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable buffering in nginx
        }
    )


@app.post(
    "/invoke_assistant",
    response_model=None,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Assistant query processed successfully"},
        404: {"description": "User session not found"},
        409: {"description": "Concurrent turn already in progress"},
        504: {"description": "Command execution timed out"}
    }
)
async def invoke_assistant(request: InvokeRequest) -> JSONResponse:
    """
    Submit a query for deterministic/assistant execution (no planning).
    The query is processed as-is without agent mode.
    """
    try:
        runtime = await session_manager.get_session(request.user_id)
        if not runtime:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User session not found: {request.user_id}"
            )

        if runtime.lock.locked():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"A turn is already in progress for user: {request.user_id}"
            )

        async with runtime.lock:
            # Check if already in assistant mode (handling error state corrections)
            if "is_assistant_mode_command" in runtime.chat_session.cme_workflow.context:
                # Already in assistant mode - pass message as-is (no '/' prefix)
                # User is providing corrections for ambiguity/misunderstanding/parameter errors
                assistant_query = request.user_query
            else:
                # Starting new assistant command - prepend '/' to enter assistant mode
                assistant_query = f"/{request.user_query.lstrip('/')}"

            # Enqueue the message
            runtime.chat_session.user_message_queue.put(assistant_query)

            # Wait for output
            command_output = await wait_for_command_output(runtime, request.timeout_seconds)

            # Incrementally save conversation turns (without generating topic/summary)
            save_conversation_incremental(runtime)

            traces = collect_trace_events(runtime) if runtime.show_agent_traces else None
            response_data = command_output.model_dump()
            if traces is not None:
                response_data["traces"] = traces

            return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in invoke_assistant for user {request.user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error in invoke_assistant() for user_id: {request.user_id}",
        ) from e


@app.post(
    "/perform_action",
    response_model=None,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Action performed successfully"},
        404: {"description": "User session not found"},
        409: {"description": "Concurrent turn already in progress"},
        422: {"description": "Invalid action format"},
        504: {"description": "Action execution timed out"}
    }
)
async def perform_action(request: PerformActionRequest) -> JSONResponse:
    """
    Execute a specific workflow action directly (bypasses parameter extraction).
    """
    try:
        runtime = await session_manager.get_session(request.user_id)
        if not runtime:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User session not found: {request.user_id}"
            )

        if runtime.lock.locked():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"A turn is already in progress for user: {request.user_id}"
            )

        async with runtime.lock:
            # Convert dict to Action
            try:
                action = Action(**request.action)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid action format: {e}",
                ) from e
            
            # Directly call _process_action to bypass parameter extraction
            # This executes synchronously in the current thread (not via queue)
            command_output = runtime.chat_session._process_action(action)

            traces = collect_trace_events(runtime) if runtime.show_agent_traces else None
            response_data = command_output.model_dump()
            if traces is not None:
                response_data["traces"] = traces

            return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in perform_action for user {request.user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error in perform_action() for user_id: {request.user_id}",
        ) from e


@app.post(
    "/new_conversation",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "New conversation started successfully"},
        404: {"description": "User session not found"},
        500: {"description": "Failed to generate topic/summary or persist conversation"}
    }
)
async def new_conversation(request: NewConversationRequest) -> dict[str, str]:
    """
    Persist the current conversation and start a new one.
    Generates topic and summary synchronously; on failure, does not rotate.
    """
    try:
        runtime = await session_manager.get_session(request.user_id)
        if not runtime:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User session not found: {request.user_id}"
            )

        # Extract turns from chat_session conversation history
        if turns := extract_turns_from_history(runtime.chat_session.conversation_history):
            # Generate topic and summary synchronously (turns already saved incrementally)
            topic, summary = generate_topic_and_summary(turns)

            # Update topic/summary for the conversation (turns already persisted)
            if runtime.active_conversation_id > 0:
                conv_id = runtime.active_conversation_id
                runtime.conversation_store.update_conversation_topic_summary(
                    conv_id, topic, summary
                )
                logger.info(f"Finalized conversation {conv_id} with topic and summary for user {request.user_id}")
            else:
                # Edge case: conversation history exists but no active ID (shouldn't happen with incremental saves)
                logger.warning(f"Conversation history exists but no active_conversation_id for user {request.user_id}")
                conv_id = runtime.conversation_store.save_conversation(topic, summary, turns)
                logger.info(f"Created conversation {conv_id} for user {request.user_id}")

            # Reserve next conversation ID for the next conversation
            next_id = runtime.conversation_store.reserve_next_conversation_id()
            runtime.active_conversation_id = next_id
            runtime.chat_session.clear_conversation_history()

            logger.info(f"Ready for new conversation {runtime.active_conversation_id} for user {request.user_id}")
            return {"status": "ok"}
        else:
            # No turns to save, just clear history and start fresh
            runtime.chat_session.clear_conversation_history()
            logger.info(f"No turns to save for user {request.user_id}, cleared history")
            return {"status": "ok", "message": "No turns to save"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in new_conversation for user {request.user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error in new_conversation() for user_id: {request.user_id}",
        ) from e


@app.get(
    "/conversations",
    response_model=list[ConversationSummary],
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Conversations retrieved successfully"},
        404: {"description": "User session not found"}
    }
)
async def list_conversations(user_id: str) -> list[ConversationSummary]:
    """
    List conversations for a user, ordered by updated_at desc.
    Returns up to FASTWORKFLOW_CONVERSATIONS_LIST_LIMIT entries.
    """
    try:
        runtime = await session_manager.get_session(user_id)
        if not runtime:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User session not found: {user_id}"
            )

        # Get limit from env (default 50)
        limit = int(os.getenv("FASTWORKFLOW_CONVERSATIONS_LIST_LIMIT", "50"))

        return runtime.conversation_store.list_conversations(limit)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in list_conversations for user {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error in list_conversations() for user_id: {user_id}",
        ) from e


@app.post(
    "/post_feedback",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Feedback posted successfully"},
        404: {"description": "User session not found"},
        422: {"description": "No feedback provided or no turns to give feedback on"}
    }
)
async def post_feedback(request: PostFeedbackRequest) -> dict[str, str]:
    """
    Post feedback on the latest turn of the active (in-memory) conversation.
    Feedback is attached to the turn in conversation_history and will be persisted
    when the conversation ends (on /new_conversation or shutdown).
    At least one of binary_or_numeric_score or nl_feedback must be provided.
    """
    try:
        runtime = await session_manager.get_session(request.user_id)
        if not runtime:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User session not found: {request.user_id}"
            )

        # Check if there are any in-memory turns to give feedback on
        if not runtime.chat_session.conversation_history.messages:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"No turns available to give feedback on for user: {request.user_id}"
            )

        # Update feedback on the last turn in the in-memory conversation history
        last_turn = runtime.chat_session.conversation_history.messages[-1]
        last_turn["feedback"] = {
            "binary_or_numeric_score": request.binary_or_numeric_score,
            "nl_feedback": request.nl_feedback,
            "timestamp": int(time.time() * 1000)
        }

        # Incrementally save the updated turns with feedback
        save_conversation_incremental(runtime)

        logger.info(f"Added feedback to latest turn for user {request.user_id}")
        return {"status": "ok"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in post_feedback for user {request.user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error in post_feedback() for user_id: {request.user_id}",
        ) from e


@app.post(
    "/activate_conversation",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Conversation activated successfully"},
        404: {"description": "User or conversation not found"}
    }
)
async def activate_conversation(request: ActivateConversationRequest) -> dict[str, str]:
    """
    Activate a conversation by its conversation_id.
    """
    try:
        runtime = await session_manager.get_session(request.user_id)
        if not runtime:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User session not found: {request.user_id}"
            )

        # Get conversation by ID
        conv = runtime.conversation_store.get_conversation(request.conversation_id)
        if not conv:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation not found with ID: {request.conversation_id}"
            )

        runtime.active_conversation_id = request.conversation_id
        
        # Restore conversation history to chat_session
        restored_history = restore_history_from_turns(conv["turns"])
        runtime.chat_session._conversation_history = restored_history
        logger.info(f"Activated conversation {request.conversation_id} for user {request.user_id}")
        
        return {"status": "ok"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in activate_conversation for user {request.user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error in activate_conversation() for user_id: {request.user_id}",
        ) from e


@app.post(
    "/admin/dump_all_conversations",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Conversations dumped successfully"},
        500: {"description": "Failed to dump conversations"}
    }
)
async def dump_all_conversations(request: DumpConversationsRequest) -> dict[str, str]:
    """
    Admin endpoint: dump all conversations from all users to a JSONL file.
    Scans all .rdb files in the base folder, not just active sessions.
    """
    try:
        os.makedirs(request.output_folder, exist_ok=True)
        timestamp = int(time.time())
        output_file = os.path.join(request.output_folder, f"all_conversations_{timestamp}.jsonl")
        
        # Get base folder from environment
        # Default to current directory if not specified
        base_folder = os.getenv(
            "USER_CONVERSATIONS_SPEEDDICT_FOLDERNAME",
            os.path.join(os.getcwd(), "___user_conversations")
        )
        
        all_conversations = []
        
        # Scan the base folder for all .rdb files (all users, active or not)
        if os.path.isdir(base_folder):
            for filename in os.listdir(base_folder):
                if filename.endswith('.rdb'):
                    # Extract user_id from filename (format: <user_id>.rdb)
                    user_id = filename[:-4]  # Remove .rdb extension
                    
                    # Create temporary ConversationStore for this user
                    store = ConversationStore(user_id, base_folder)
                    user_convs = store.get_all_conversations_for_dump()
                    all_conversations.extend(user_convs)
        
        # Write to JSONL
        with open(output_file, 'w') as f:
            for conv in all_conversations:
                f.write(json.dumps(conv) + '\n')
        
        logger.info(f"Dumped {len(all_conversations)} conversations from {len(all_conversations)} users to {output_file}")
        return {"file_path": output_file}

    except Exception as e:
        logger.error(f"Error in dump_all_conversations: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to dump conversations",
        ) from e


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
