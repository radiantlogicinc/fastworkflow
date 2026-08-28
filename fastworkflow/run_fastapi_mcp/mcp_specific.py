# pyright: reportUnusedFunction=false

from fastapi_mcp import FastApiMCP

def setup_mcp(
    app,
    session_manager,
):
    """Mount MCP to automatically convert FastAPI endpoints to MCP tools.

    FastAPI endpoints are automatically exposed as MCP tools, except those in the exclude list.
    
    Key exposed tools:
    - invoke_agent: Streaming agent invocation with NDJSON/SSE support (from /invoke_agent_stream)
    - invoke_assistant: Assistant mode (deterministic execution)
    - new_conversation, get_all_conversations, post_feedback, activate_conversation
    
    MCP Client Setup:
    - MCP clients use pre-configured long-lived access tokens (generated via /admin/generate_mcp_token)
    - Tokens are added to the MCP client configuration, not obtained via tool calls
    - No need for initialize or refresh_token tools in MCP context
    
    Note: Prompt registration (format-command, clarify-params) is commented out
    as fastapi-mcp 0.4.0 does not support custom prompts.

    MCP isError vs. turn outcome (deliberately NOT mapped — see below).
    """
    # TODO(fix-qtq.6): map MCP isError = not TurnOutput.success once the library
    # allows it without lying about the transport. It cannot be done today, and
    # the reason is worth recording so nobody re-derives it:
    #
    # 1. There is no hook. fastapi-mcp 0.4.0 registers its own @call_tool
    #    handler inside FastApiMCP.setup_server() and answers from the private
    #    _execute_api_tool(), which returns list[TextContent]. The MCP SDK marks
    #    a result isError=True only when the handler RAISES, and _execute_api_tool
    #    raises for exactly one reason: an HTTP status of 400-599.
    # 2. So the only available lever is the HTTP status code — and a failed turn
    #    is NOT a failed call. Turning `success == False` into a 4xx/5xx would
    #    destroy the separation this migration exists to establish: transport
    #    status answers "did the call work", TurnOutput answers "did the turn
    #    work" (turn_result_design_final.md section 14).
    # 3. The streaming `invoke_agent` tool could not read `success` anyway: its
    #    body is an NDJSON/SSE stream, so response.json() fails and the tool
    #    result is the raw stream text. There is no top-level success field to
    #    map without parsing the stream and locating its terminal output event.
    #
    # Until then, MCP clients read the outcome from the tool result body, which
    # carries the full TurnOutput projection (status, failure_reason, success).

    # =========================================================================
    # Mount MCP (FastApiMCP will scan and find all FastAPI endpoints)
    # =========================================================================
    
    # Exclude endpoints that should not be exposed as MCP tools:
    # - root: HTML homepage endpoint
    # - dump_all_conversations: Admin-only endpoint for dumping all user conversations
    # - generate_mcp_token: Admin-only endpoint for generating long-lived MCP tokens
    # - rest_initialize: Regular initialization (MCP clients use pre-configured tokens, don't need to initialize)
    # - perform_action: Low-level action execution (use invoke_agent/invoke_assistant instead)
    # - rest_invoke_agent: Non-streaming version (use "invoke_agent" streaming endpoint instead)
    # - refresh_token: JWT token refresh (not needed for MCP since MCP uses long-lived tokens)
    # - get_turn / get_turn_trace: polling/replay surface for HTTP clients. The
    #   85g design left MCP exposure of GET /turns as an explicit decision for
    #   when it was built; resolved conservatively — MCP clients use the
    #   streaming invoke_agent tool with a longer wait window and do not poll.
    #
    # Exposed MCP tools:
    # - invoke_agent (operation_id) → /invoke_agent_stream endpoint (streaming with NDJSON/SSE support)
    # - invoke_assistant: Assistant mode (deterministic execution)
    # - new_conversation, get_all_conversations, post_feedback, activate_conversation
    #
    # Note: MCP clients are configured with long-lived access tokens generated via /admin/generate_mcp_token
    mcp = FastApiMCP(
        app, 
        exclude_operations=[
            "root", 
            "dump_all_conversations",
            "generate_mcp_token",
            "rest_initialize",
            "perform_action", 
            "rest_invoke_agent",
            "refresh_token",
            "get_turn",
            "get_turn_trace"
        ]
    )
    mcp.mount_http()

    # Note: Prompt registration is not supported in fastapi-mcp 0.4.0
    # The library automatically converts FastAPI endpoints to MCP tools,
    # but does not provide a way to register custom prompts.
    # Prompts may be added in a future version or via manual MCP server implementation.
    
    # TODO: Re-enable when fastapi-mcp supports prompts or implement custom prompt handler
    # # Prompts
    # mcp.add_prompt(
    #     name="format-command",
    #     description="Given command metadata and a user intent, format a single executable command with XML-tagged parameters.",
    #     arguments=[{"name": "intent", "required": True}, {"name": "metadata", "required": True}],
    #     handler=lambda intent, metadata: [
    #         {
    #             "role": "user",
    #             "content": {
    #                 "type": "text",
    #                 "text": (
    #                     f"Intent: {intent}\n\nMetadata:\n{metadata}\n\n"
    #                     "Format a single command: command_name <param>value</param> ..."
    #                 ),
    #             },
    #         }
    #     ],
    # )
    #
    # mcp.add_prompt(
    #     name="clarify-params",
    #     description="Compose a concise clarification question for missing parameters using the provided metadata.",
    #     arguments=[{"name": "error_message", "required": True}, {"name": "metadata", "required": True}],
    #     handler=lambda error_message, metadata: [
    #         {
    #             "role": "user",
    #             "content": {
    #                 "type": "text",
    #                 "text": (
    #                     f"{error_message}\n\nMetadata:\n{metadata}\n\n"
    #                     "Ask one short question to request the missing parameters."
    #                 ),
    #             },
    #         }
    #     ],
    # )


