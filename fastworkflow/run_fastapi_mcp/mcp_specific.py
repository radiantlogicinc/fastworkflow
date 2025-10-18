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
    """

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
            "refresh_token"
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


