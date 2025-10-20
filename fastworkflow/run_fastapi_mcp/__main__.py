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
- ✅ Graceful shutdown (30s)
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
import argparse

import uvicorn
from jose import JWTError
from dotenv import dotenv_values

import fastworkflow
from fastworkflow.utils.logging import logger

from fastapi import FastAPI, HTTPException, status, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from .mcp_specific import setup_mcp
from .utils import (
    get_userconversations_dir,
    UserSessionManager,
    save_conversation_incremental,
    InitializationRequest,
    TokenResponse,
    SessionData,
    InvokeRequest,
    PerformActionRequest,
    PostFeedbackRequest,
    ActivateConversationRequest,
    DumpConversationsRequest,
    GenerateMCPTokenRequest,
    wait_for_command_output,
    collect_trace_events,
    get_session_from_jwt,
    ensure_user_runtime_exists
)
from .jwt_manager import (
    create_access_token,
    create_refresh_token,
    verify_token,
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES
)

from .conversation_store import (
    ConversationStore,
    ConversationSummary,
    generate_topic_and_summary,
    extract_turns_from_history,
    restore_history_from_turns
)

 
# ============================================================================
# Session Management
# ============================================================================

# Global session manager
session_manager = UserSessionManager()


# ============================================================================
# Dependencies
# ============================================================================

async def get_session_and_ensure_runtime(
    session: SessionData = Depends(get_session_from_jwt)
) -> SessionData:
    """
    FastAPI dependency that validates JWT token AND ensures user runtime exists.
    
    This dependency combines JWT token validation with automatic session creation.
    If the user's runtime doesn't exist in the session manager, it will be created
    automatically using the same logic as the /initialize endpoint (but without
    generating new JWT tokens).
    
    This is particularly useful for MCP endpoints where the client already has
    a valid JWT token but the server may have restarted or the session expired.
    
    Args:
        session: SessionData extracted and validated from JWT Bearer token
        
    Returns:
        SessionData: The same session data after ensuring runtime exists
        
    Raises:
        HTTPException: If token is invalid or session creation fails
        
    Example:
        Use as a dependency in FastAPI endpoints:
        ```python
        @app.post("/endpoint")
        async def endpoint(session: SessionData = Depends(get_session_and_ensure_runtime)):
            # session.user_id can now safely be used with session_manager
            runtime = await session_manager.get_session(session.user_id)
            # runtime is guaranteed to exist
        ```
    """
    # Ensure the user runtime exists (creates if missing)
    await ensure_user_runtime_exists(
        user_id=session.user_id,
        session_manager=session_manager,
        workflow_path=ARGS.workflow_path,
        context=json.loads(ARGS.context) if ARGS.context else None,
        startup_command=ARGS.startup_command,
        startup_action=fastworkflow.Action(**json.loads(ARGS.startup_action)) if ARGS.startup_action else None
    )
    
    return session


# ============================================================================
# FastAPI App Setup
# ============================================================================

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Startup and shutdown hooks"""
    logger.info("FastWorkflow FastAPI service starting...")
    logger.info(f"Startup with CLI params: workflow_path={ARGS.workflow_path}, env_file_path={ARGS.env_file_path}, passwords_file_path={ARGS.passwords_file_path}")

    def initialize_fastworkflow_on_startup() -> None:
        env_vars: dict[str, str] = {}
        if ARGS.env_file_path:
            env_vars |= dotenv_values(ARGS.env_file_path)
        if ARGS.passwords_file_path:
            env_vars.update(dotenv_values(ARGS.passwords_file_path))
        fastworkflow.init(env_vars=env_vars)

    async def _active_turn_user_ids() -> list[str]:
        active: list[str] = []
        for user_id in list(session_manager._sessions.keys()):
            rt = await session_manager.get_session(user_id)
            if rt and rt.lock.locked():
                active.append(user_id)
        return active

    async def wait_for_active_turns_to_complete(max_wait_seconds: int) -> None:
        logger.info(f"Waiting up to {max_wait_seconds}s for active turns to complete...")
        start_time = time.time()
        while time.time() - start_time < max_wait_seconds:
            active_turns = await _active_turn_user_ids()
            if not active_turns:
                logger.info("All turns completed, shutting down gracefully")
                return
            logger.debug(f"Waiting for {len(active_turns)} active turns: {active_turns}")
            await asyncio.sleep(0.5)
        remaining = await _active_turn_user_ids()
        logger.warning(f"Shutdown timeout reached with {len(remaining)} turns still active")

    async def finalize_conversations_on_shutdown() -> None:
        logger.info("Finalizing conversations with topic and summary...")
        for user_id in list(session_manager._sessions.keys()):
            runtime = await session_manager.get_session(user_id)
            if not runtime:
                continue
            if turns := extract_turns_from_history(runtime.chat_session.conversation_history):
                try:
                    topic, summary = generate_topic_and_summary(turns)
                    if runtime.active_conversation_id > 0:
                        runtime.conversation_store.update_conversation_topic_summary(
                            runtime.active_conversation_id, topic, summary
                        )
                        logger.info(f"Finalized conversation {runtime.active_conversation_id} for user {user_id} during shutdown")
                    else:
                        logger.warning(f"Conversation history exists but no active_conversation_id for user {user_id} during shutdown")
                        conv_id = runtime.conversation_store.save_conversation(topic, summary, turns)
                        logger.info(f"Created conversation {conv_id} for user {user_id} during shutdown")
                except Exception as e:
                    logger.error(f"Failed to finalize conversation for user {user_id} during shutdown: {e}")

    async def stop_all_chat_sessions() -> None:
        for user_id in list(session_manager._sessions.keys()):
            runtime = await session_manager.get_session(user_id)
            if runtime:
                runtime.chat_session.stop_workflow()

    try:
        initialize_fastworkflow_on_startup()
        yield
    finally:
        logger.info("FastWorkflow FastAPI service shutting down...")
        await wait_for_active_turns_to_complete(max_wait_seconds=30)
        await finalize_conversations_on_shutdown()
        await stop_all_chat_sessions()
        logger.info("FastWorkflow FastAPI service shutdown complete")


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow_path", required=True)
    parser.add_argument("--env_file_path", required=False)
    parser.add_argument("--passwords_file_path", required=False)
    parser.add_argument("--context", required=False)  # JSON string
    parser.add_argument("--startup_command", required=False)
    parser.add_argument("--startup_action", required=False)  # JSON string
    parser.add_argument("--project_folderpath", required=False)
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on (default: 8000)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to (default: 0.0.0.0)")
    return parser.parse_args()

ARGS = load_args()

app = FastAPI(
    title="FastWorkflow API",
    description="HTTP interface for FastWorkflow workflows with JWT authentication",
    version="1.0.0",
    lifespan=lifespan,
    swagger_ui_parameters={
        "persistAuthorization": True  # Remember Bearer token in Swagger UI
    }
)

# Configure OpenAPI security scheme for JWT Bearer tokens
# This enables the "Authorize" button in Swagger UI
# Note: The security scheme is automatically generated by HTTPBearer in utils.py,
# but we customize it here to improve the description and ensure proper integration
from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Enhance the auto-generated Bearer token security scheme with better documentation
    # The HTTPBearer dependency in utils.py creates the base scheme, we just improve it
    if "components" in openapi_schema and "securitySchemes" in openapi_schema["components"]:
        if "BearerAuth" in openapi_schema["components"]["securitySchemes"]:
            # Update the description to be more helpful
            openapi_schema["components"]["securitySchemes"]["BearerAuth"]["description"] = (
                "JWT access token from /initialize or /refresh_token endpoint. "
                "Enter ONLY the token (Swagger UI automatically adds 'Bearer ' prefix)"
            )
    
    # Apply security globally to all endpoints except public ones
    for path, path_item in openapi_schema["paths"].items():
        # Skip endpoints that don't require authentication
        if path in ["/initialize", "/refresh_token", "/", "/admin/dump_all_conversations", "/admin/generate_mcp_token"]:
            continue
        for method in path_item:
            if method in ["get", "post", "put", "delete", "patch"]:
                # Ensure security is applied (should already be set by HTTPBearer dependency)
                if "security" not in path_item[method]:
                    path_item[method]["security"] = [{"BearerAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

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

@app.get("/", response_class=HTMLResponse, operation_id="root")
async def root():
    """Root endpoint with health check and docs link"""
    return """
    <html>
        <head>
            <title>FastWorkflow API</title>
        </head>
        <body>
            <h1>FastWorkflow API is running!</h1>
            <p>For API testing, go to <a href="/docs">Swagger UI</a></p>
            <p>For API documentation, go to <a href="/redoc">ReDoc</a></p>
        </body>
    </html>
    """


@app.post(
    "/initialize",
    operation_id="rest_initialize",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Session successfully initialized, JWT tokens returned"},
        400: {"description": "Both startup_command and startup_action provided"},
        422: {"description": "Invalid paths or missing env vars"},
        500: {"description": "Internal error during initialization"}
    }
)
async def initialize(request: InitializationRequest) -> TokenResponse:
    """
    Initialize a FastWorkflow session for a user.
    Creates or resumes a ChatSession and starts the workflow.
    """
    try:
        user_id = request.user_id
        logger.info(f"Initializing session for user_id: {user_id}")

        # Check if user already has an active session
        existing_runtime = await session_manager.get_session(user_id)
        if existing_runtime:
            logger.info(f"Session for user_id {user_id} already exists, generating new tokens")
            
            # Generate new JWT tokens for existing session
            access_token = create_access_token(user_id)
            refresh_token = create_refresh_token(user_id)
            
            return TokenResponse(
                access_token=access_token,
                refresh_token=refresh_token,
                token_type="bearer",
                expires_in=JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60  # Convert to seconds
            )

        # Prepare startup action if provided via CLI args
        startup_action = None
        if ARGS.startup_action:
            startup_action = fastworkflow.Action(**json.loads(ARGS.startup_action))

        # Use the modular helper function to create the session
        # This ensures consistent session creation logic across the application
        await ensure_user_runtime_exists(
            user_id=user_id,
            session_manager=session_manager,
            workflow_path=ARGS.workflow_path,
            context=json.loads(ARGS.context) if ARGS.context else None,
            startup_command=ARGS.startup_command,
            startup_action=startup_action,
            stream_format=(request.stream_format if request.stream_format in ("ndjson", "sse") else "ndjson")
        )
        
        # Generate JWT tokens
        access_token = create_access_token(user_id)
        refresh_token = create_refresh_token(user_id)
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60  # Convert to seconds
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initializing session for user_id: {request.user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error in initialize() for user_id: {request.user_id}",
        ) from e


@app.post(
    "/refresh_token",
    operation_id="refresh_token",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "New access token issued successfully"},
        401: {"description": "Invalid or expired refresh token"},
        404: {"description": "Session not found (session may have been cleaned up)"}
    }
)
async def refresh_token(
    authorization: str = Header(..., description="Refresh token in Bearer format")
) -> TokenResponse:
    """
    Refresh an access token using a valid refresh token.
    Returns a new access token and a new refresh token.
    
    Requires the refresh token to be passed in the Authorization header (Bearer token format).
    """
    try:
        # Validate Bearer token format
        if not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Authorization header format. Expected: Bearer <refresh_token>",
                headers={"WWW-Authenticate": "Bearer"}
            )

        # Extract token
        refresh_token_str = authorization[7:]  # Remove "Bearer " prefix

        # Verify refresh token
        try:
            payload = verify_token(refresh_token_str, expected_type="refresh")
        except JWTError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid or expired refresh token: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"},
            ) from e

        # Extract user_id from payload
        user_id = payload["sub"]

        # Verify session still exists
        runtime = await session_manager.get_session(user_id)
        if not runtime:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User session not found: {user_id} (may have been cleaned up)"
            )

        # Generate new tokens
        new_access_token = create_access_token(user_id)
        new_refresh_token = create_refresh_token(user_id)

        logger.info(f"Refreshed tokens for user_id: {user_id}")

        return TokenResponse(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            token_type="bearer",
            expires_in=JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60  # Convert to seconds
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing token: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error in refresh_token()",
        ) from e


@app.post(
    "/invoke_agent",
    operation_id="rest_invoke_agent",
    response_model=None,  # Use custom response to include traces
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Agent query processed successfully"},
        401: {"description": "Invalid or expired JWT token"},
        404: {"description": "Session not found"},
        409: {"description": "Concurrent turn already in progress"},
        504: {"description": "Command execution timed out"}
    }
)
async def invoke_agent(
    request: InvokeRequest,
    session: SessionData = Depends(get_session_and_ensure_runtime)
) -> JSONResponse:
    """
    Submit a natural language query to the agent.
    Leading '/' characters are stripped for compatibility.
    
    Requires a valid JWT access token in the Authorization header (Bearer token format).
    """
    user_id = session.user_id
    try:
        runtime = await session_manager.get_session(user_id)
        if not runtime:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User session not found: {user_id}"
            )

        # Serialize turns per user
        if runtime.lock.locked():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"A turn is already in progress for user: {user_id}"
            )

        async with runtime.lock:
            # Strip leading slashes from user query
            user_query = request.user_query.lstrip('/')

            # Enqueue the user message
            runtime.chat_session.user_message_queue.put(user_query)

            # Wait for command output
            command_output = await wait_for_command_output(runtime, request.timeout_seconds)

            # Incrementally save conversation turns (without generating topic/summary)
            save_conversation_incremental(runtime, extract_turns_from_history, logger)

            traces = collect_trace_events(runtime)
            # Build response with traces
            response_data = command_output.model_dump()
            if traces:
                response_data["traces"] = traces

            return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in invoke_agent for user {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error in invoke_agent() for user_id: {user_id}",
        ) from e


@app.post(
    "/invoke_agent_stream",
    operation_id="invoke_agent",
    responses={
        200: {
            "description": "Stream with trace events and final command output",
            "content": {
                "application/x-ndjson": {},
                "text/event-stream": {}
            }
        },
        401: {"description": "Invalid or expired JWT token"},
        404: {"description": "Session not found"},
        409: {"description": "Concurrent turn already in progress"},
        504: {"description": "Command execution timed out"}
    }
)
async def invoke_agent_stream(
    request: InvokeRequest,
    session: SessionData = Depends(get_session_and_ensure_runtime)
):
    """
    Submit a natural language query to the agent and stream responses.
    
    Streams via NDJSON or SSE based on the session's stream_format preference.
    - NDJSON: {"type":"trace","data":<trace_json>} for each trace, {"type":"output","data":<CommandOutput_json>} for final result
    - SSE: event: trace/output with data payloads
    
    Requires a valid JWT access token in the Authorization header (Bearer token format).
    Exposed as 'invoke_agent' tool for MCP clients (who don't need JWT auth).
    """
    user_id = session.user_id
    
    # Get runtime and validate session exists
    runtime = await session_manager.get_session(user_id)
    if not runtime:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": f"User session not found: {user_id}"}
        )

    async def ndjson_stream():
        try:
            if runtime.lock.locked():
                yield {"type": "error", "data": {"detail": f"A turn is already in progress for user: {user_id}"}}
                return
            
            async with runtime.lock:
                runtime.chat_session.user_message_queue.put(request.user_query.lstrip("/"))
                start_time = time.time()
                command_output = None
                
                while time.time() - start_time < request.timeout_seconds:
                    while True:
                        try:
                            evt = runtime.chat_session.command_trace_queue.get_nowait()
                            trace_json = {
                                "direction": evt.direction.value if hasattr(evt.direction, "value") else str(evt.direction),
                                "raw_command": evt.raw_command,
                                "command_name": evt.command_name,
                                "parameters": evt.parameters,
                                "response_text": evt.response_text,
                                "success": evt.success,
                                "timestamp_ms": evt.timestamp_ms,
                            }
                            yield {"type": "trace", "data": trace_json}
                        except queue.Empty:
                            break
                    
                    try:
                        command_output = runtime.chat_session.command_output_queue.get_nowait()
                        break
                    except queue.Empty:
                        await asyncio.sleep(0.1)
                        continue
                
                # Drain remaining traces
                while True:
                    try:
                        evt = runtime.chat_session.command_trace_queue.get_nowait()
                        trace_json = {
                            "direction": evt.direction.value if hasattr(evt.direction, "value") else str(evt.direction),
                            "raw_command": evt.raw_command,
                            "command_name": evt.command_name,
                            "parameters": evt.parameters,
                            "response_text": evt.response_text,
                            "success": evt.success,
                            "timestamp_ms": evt.timestamp_ms,
                        }
                        yield {"type": "trace", "data": trace_json}
                    except queue.Empty:
                        break
                
                if command_output is None:
                    yield {"type": "error", "data": {"detail": f"Command execution timed out after {request.timeout_seconds} seconds"}}
                    return
                
                save_conversation_incremental(runtime, extract_turns_from_history, logger)
                yield {"type": "output", "data": command_output.model_dump()}
        
        except Exception as e:
            logger.error(f"Error in invoke_agent_stream for user {user_id}: {e}")
            traceback.print_exc()
            yield {"type": "error", "data": {"detail": f"Internal error in invoke_agent_stream() for user_id: {user_id}"}}

    async def sse_stream():
        try:
            if runtime.lock.locked():
                yield "event: error\n" + f"data: {json.dumps({'detail': f'A turn is already in progress for user: {user_id}'})}\n\n"
                return
            
            async with runtime.lock:
                runtime.chat_session.user_message_queue.put(request.user_query.lstrip("/"))
                
                def fmt(evt):
                    trace_data = {
                        "direction": evt.direction.value if hasattr(evt.direction, "value") else str(evt.direction),
                        "raw_command": evt.raw_command,
                        "command_name": evt.command_name,
                        "parameters": evt.parameters,
                        "response_text": evt.response_text,
                        "success": evt.success,
                        "timestamp_ms": evt.timestamp_ms,
                    }
                    return f"event: trace\ndata: {json.dumps(trace_data)}\n\n"
                
                start_time = time.time()
                command_output = None
                
                while time.time() - start_time < request.timeout_seconds:
                    while True:
                        try:
                            evt = runtime.chat_session.command_trace_queue.get_nowait()
                            yield fmt(evt)
                        except queue.Empty:
                            break
                    
                    try:
                        command_output = runtime.chat_session.command_output_queue.get_nowait()
                        break
                    except queue.Empty:
                        await asyncio.sleep(0.1)
                        continue
                
                # Drain remaining traces
                while True:
                    try:
                        evt = runtime.chat_session.command_trace_queue.get_nowait()
                        yield fmt(evt)
                    except queue.Empty:
                        break
                
                if command_output is None:
                    yield "event: error\n" + f"data: {json.dumps({'detail': f'Command execution timed out after {request.timeout_seconds} seconds'})}\n\n"
                    return
                
                save_conversation_incremental(runtime, extract_turns_from_history, logger)
                yield "event: output\n" + f"data: {json.dumps(command_output.model_dump())}\n\n"
        
        except Exception as e:
            logger.error(f"Error in invoke_agent_stream SSE for user {user_id}: {e}")
            traceback.print_exc()
            yield "event: error\n" + f"data: {json.dumps({'detail': f'Internal error in invoke_agent_stream() for user_id: {user_id}'})}\n\n"

    # Route to appropriate stream format
    if runtime.stream_format == "sse":
        return StreamingResponse(
            sse_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    
    # Default to NDJSON with JSON serialization wrapper
    async def ndjson_body():
        async for part in ndjson_stream():
            yield json.dumps(part) + "\n"
    
    return StreamingResponse(ndjson_body(), media_type="application/x-ndjson")


@app.post(
    "/invoke_assistant",
    operation_id="invoke_assistant",
    response_model=None,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Assistant query processed successfully"},
        401: {"description": "Invalid or expired JWT token"},
        404: {"description": "Session not found"},
        409: {"description": "Concurrent turn already in progress"},
        504: {"description": "Command execution timed out"}
    }
)
async def invoke_assistant(
    request: InvokeRequest,
    session: SessionData = Depends(get_session_and_ensure_runtime)
) -> JSONResponse:
    """
    Submit a query for deterministic/assistant execution (no planning).
    The query is processed as-is without agent mode.
    
    Requires a valid JWT access token in the Authorization header (Bearer token format).
    """
    user_id = session.user_id
    try:
        runtime = await session_manager.get_session(user_id)
        if not runtime:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User session not found: {user_id}"
            )

        if runtime.lock.locked():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"A turn is already in progress for user: {user_id}"
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
            save_conversation_incremental(runtime, extract_turns_from_history, logger)

            traces = collect_trace_events(runtime)
            response_data = command_output.model_dump()
            if traces:
                response_data["traces"] = traces

            return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in invoke_assistant for session {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error in invoke_assistant() for user_id: {user_id}",
        ) from e


@app.post(
    "/perform_action",
    operation_id="perform_action",
    response_model=None,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Action performed successfully"},
        401: {"description": "Invalid or expired JWT token"},
        404: {"description": "Session not found"},
        409: {"description": "Concurrent turn already in progress"},
        422: {"description": "Invalid action format"},
        504: {"description": "Action execution timed out"}
    }
)
async def perform_action(
    request: PerformActionRequest,
    session: SessionData = Depends(get_session_and_ensure_runtime)
) -> JSONResponse:
    """
    Execute a specific workflow action directly (bypasses parameter extraction).
    
    Requires a valid JWT access token in the Authorization header (Bearer token format).
    """
    user_id = session.user_id
    try:
        runtime = await session_manager.get_session(user_id)
        if not runtime:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User session not found: {user_id}"
            )

        if runtime.lock.locked():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"A turn is already in progress for user: {user_id}"
            )

        async with runtime.lock:
            # Convert dict to fastworkflow.Action
            try:
                action = fastworkflow.Action(**request.action)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid action format: {e}",
                ) from e
            
            # Directly call _process_action to bypass parameter extraction
            # This executes synchronously in the current thread (not via queue)
            command_output = runtime.chat_session._process_action(action)

            traces = collect_trace_events(runtime)
            response_data = command_output.model_dump()
            if traces:
                response_data["traces"] = traces

            return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in perform_action for session {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error in perform_action() for user_id: {user_id}",
        ) from e


@app.post(
    "/new_conversation",
    operation_id="new_conversation",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "New conversation started successfully"},
        401: {"description": "Invalid or expired JWT token"},
        404: {"description": "Session not found"},
        500: {"description": "Failed to generate topic/summary or persist conversation"}
    }
)
async def new_conversation(
    session: SessionData = Depends(get_session_and_ensure_runtime)
) -> dict[str, str]:
    """
    Persist the current conversation and start a new one.
    Generates topic and summary synchronously; on failure, does not rotate.
    
    Requires a valid JWT access token in the Authorization header (Bearer token format).
    """
    user_id = session.user_id
    try:
        runtime = await session_manager.get_session(user_id)
        if not runtime:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User session not found: {user_id}"
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
                logger.info(f"Finalized conversation {conv_id} with topic and summary for session {user_id}")
            else:
                # Edge case: conversation history exists but no active ID (shouldn't happen with incremental saves)
                logger.warning(f"Conversation history exists but no active_conversation_id for session {user_id}")
                conv_id = runtime.conversation_store.save_conversation(topic, summary, turns)
                logger.info(f"Created conversation {conv_id} for session {user_id}")

            # Reserve next conversation ID for the next conversation
            next_id = runtime.conversation_store.reserve_next_conversation_id()
            runtime.active_conversation_id = next_id
            runtime.chat_session.clear_conversation_history()

            logger.info(f"Ready for new conversation {runtime.active_conversation_id} for session {user_id}")
            return {"status": "ok"}
        else:
            # No turns to save, just clear history and start fresh
            runtime.chat_session.clear_conversation_history()
            logger.info(f"No turns to save for session {user_id}, cleared history")
            return {"status": "ok", "message": "No turns to save"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in new_conversation for session {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error in new_conversation() for user_id: {user_id}",
        ) from e


@app.get(
    "/conversations",
    operation_id="get_all_conversations",
    response_model=list[ConversationSummary],
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Conversations retrieved successfully"},
        401: {"description": "Invalid or expired JWT token"},
        404: {"description": "Session not found"}
    }
)
async def list_conversations(
    limit: int = 20,
    session: SessionData = Depends(get_session_and_ensure_runtime)
) -> list[ConversationSummary]:
    """
    List conversations for a session, ordered by updated_at desc.
    Returns up to `limit` entries.
    
    Requires a valid JWT access token in the Authorization header (Bearer token format).
    """
    user_id = session.user_id
    try:
        runtime = await session_manager.get_session(user_id)
        if not runtime:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User session not found: {user_id}"
            )
        return runtime.conversation_store.list_conversations(limit)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in list_conversations for session {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error in list_conversations() for user_id: {user_id}",
        ) from e


@app.post(
    "/post_feedback",
    operation_id="post_feedback",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Feedback posted successfully"},
        401: {"description": "Invalid or expired JWT token"},
        404: {"description": "Session not found"},
        422: {"description": "No feedback provided or no turns to give feedback on"}
    }
)
async def post_feedback(
    request: PostFeedbackRequest,
    session: SessionData = Depends(get_session_and_ensure_runtime)
) -> dict[str, str]:
    """
    Post feedback on the latest turn of the active (in-memory) conversation.
    Feedback is attached to the turn in conversation_history and will be persisted
    when the conversation ends (on /new_conversation or shutdown).
    At least one of binary_or_numeric_score or nl_feedback must be provided.
    
    Requires a valid JWT access token in the Authorization header (Bearer token format).
    """
    user_id = session.user_id
    try:
        runtime = await session_manager.get_session(user_id)
        if not runtime:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User session not found: {user_id}"
            )

        # Check if there are any in-memory turns to give feedback on
        if not runtime.chat_session.conversation_history.messages:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"No turns available to give feedback on for user: {user_id}"
            )

        # Update feedback on the last turn in the in-memory conversation history
        last_turn = runtime.chat_session.conversation_history.messages[-1]
        last_turn["feedback"] = {
            "binary_or_numeric_score": request.binary_or_numeric_score,
            "nl_feedback": request.nl_feedback,
            "timestamp": int(time.time() * 1000)
        }

        # Incrementally save the updated turns with feedback
        save_conversation_incremental(runtime, extract_turns_from_history, logger)

        logger.info(f"Added feedback to latest turn for session {user_id}")
        return {"status": "ok"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in post_feedback for session {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error in post_feedback() for user_id: {user_id}",
        ) from e


@app.post(
    "/activate_conversation",
    operation_id="activate_conversation",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Conversation activated successfully"},
        401: {"description": "Invalid or expired JWT token"},
        404: {"description": "Session or conversation not found"}
    }
)
async def activate_conversation(
    request: ActivateConversationRequest,
    session: SessionData = Depends(get_session_and_ensure_runtime)
) -> dict[str, str]:
    """
    Activate a conversation by its conversation_id.
    
    Requires a valid JWT access token in the Authorization header (Bearer token format).
    """
    user_id = session.user_id
    try:
        runtime = await session_manager.get_session(user_id)
        if not runtime:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User session not found: {user_id}"
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
        logger.info(f"Activated conversation {request.conversation_id} for session {user_id}")
        
        return {"status": "ok"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in activate_conversation for session {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error in activate_conversation() for user_id: {user_id}",
        ) from e


@app.post(
    "/admin/dump_all_conversations",
    operation_id="dump_all_conversations",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Conversations dumped successfully"},
        500: {"description": "Failed to dump conversations"}
    }
)
async def dump_all_conversations(request: DumpConversationsRequest) -> dict[str, str]:
    """
    Admin endpoint: dump all conversations from all sessions to a JSONL file.
    Scans all .rdb files in the base folder, not just active sessions.
    """
    try:
        os.makedirs(request.output_folder, exist_ok=True)
        timestamp = int(time.time())
        output_file = os.path.join(request.output_folder, f"all_conversations_{timestamp}.jsonl")
        
        # Resolve base folder using SPEEDDICT_FOLDERNAME/user_conversations
        base_folder = get_userconversations_dir()
        
        all_conversations = []
        session_count = 0
        
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
                    session_count += 1
        
        # Write to JSONL
        with open(output_file, 'w') as f:
            for conv in all_conversations:
                f.write(json.dumps(conv) + '\n')
        
        logger.info(f"Dumped {len(all_conversations)} conversations from {session_count} users to {output_file}")
        return {"file_path": output_file}

    except Exception as e:
        logger.error(f"Error in dump_all_conversations: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to dump conversations",
        ) from e


@app.post(
    "/admin/generate_mcp_token",
    operation_id="generate_mcp_token",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "MCP token generated successfully"},
        500: {"description": "Failed to generate token"}
    }
)
async def generate_mcp_token(request: GenerateMCPTokenRequest) -> TokenResponse:
    """
    Admin endpoint: Generate a long-lived access token for MCP client configuration.
    
    These tokens are meant to be configured in MCP client settings (e.g., Claude Desktop)
    and have extended expiration times (default 365 days) since they can't be easily refreshed.
    
    Args:
        user_id: Identifier for the MCP user/client
        expires_days: Token expiration in days (default: 365 days / 1 year)
        
    Returns:
        TokenResponse with long-lived access_token (no refresh_token needed for MCP)
        
    Note: This endpoint should be restricted to administrators only in production.
    """
    try:
        # Generate long-lived access token
        access_token = create_access_token(request.user_id, expires_days=request.expires_days)
        
        logger.info(f"Generated MCP token for user_id: {request.user_id}, expires in {request.expires_days} days")
        
        return TokenResponse(
            access_token=access_token,
            refresh_token="",  # Not needed for MCP (long-lived token)
            token_type="bearer",
            expires_in=request.expires_days * 24 * 60 * 60  # Convert to seconds
        )
    
    except Exception as e:
        logger.error(f"Error generating MCP token: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate MCP token",
        ) from e


# =========================================================================
# MCP Mount (tools over Streamable HTTP and optional SSE per session)
# IMPORTANT: Must be called AFTER all endpoints are defined so fastapi-mcp
# can discover and convert them to MCP tools automatically
# =========================================================================

setup_mcp(
    app=app,
    session_manager=session_manager,
)

# ============================================================================
# Main
# ============================================================================

def main():
    """Entry point for the FastAPI MCP server."""
    host = ARGS.host if hasattr(ARGS, 'host') else "0.0.0.0"
    port = ARGS.port if hasattr(ARGS, 'port') else 8000
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
