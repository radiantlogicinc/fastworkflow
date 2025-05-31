# Agent Development Guide

## Agent System Overview
The FastWorkflow agent system uses a delegated architecture with DSPy ReAct agents that integrate with MCP-compliant tool calling via queue-based communication and specialized MCP Tool Agents.

## Key Files for Agent Development

### Agent Core
- **[fastworkflow/run_agent/__main__.py](mdc:fastworkflow/run_agent/__main__.py)** - CLI entry point for running agents with colorama output formatting
- **[fastworkflow/run_agent/agent_module.py](mdc:fastworkflow/run_agent/agent_module.py)** - Agent initialization, tool creation, and delegation logic

### Agent Communication Architecture
- **Main Agent**: Uses `DialogueWithWorkflow` DSPy signature with `WorkflowAssistant` and `AskUser` tools
- **MCP Tool Agent**: Specialized agent using `ExecuteMCPTool` signature that exposes individual MCP tools
- **Delegation Logic**: `_execute_workflow_command_tool_with_delegation()` delegates tool requests to the MCP Tool Agent
- Communication via `user_message_queue.put()` and `command_output_queue.get()` with workflow session

## Working with Agent Tools

### The WorkflowAssistant Tool (Delegated Architecture)
```python
_execute_workflow_command_tool_with_delegation(tool_request: str, *, mcp_tool_agent, workflow_session_obj)
```

**Behavior:**
- Main agent delegates all tool requests to specialized MCP Tool Agent
- MCP Tool Agent exposes individual workflow commands as separate DSPy tools
- Each individual tool expects specific query format for its command
- Returns formatted response for main agent consumption

### Individual MCP Tools
- Created dynamically via `_create_individual_query_tool()` and `_create_individual_mcp_tool()`
- Each workflow command becomes a separate DSPy tool with full documentation
- Tools accept either query strings or MCP JSON payloads depending on configuration
- Direct integration with workflow session queues

### Helper Functions
- `_build_simplified_tool_documentation()` - Creates main agent tool documentation (names + descriptions only)
- `initialize_mcp_tool_agent()` - Sets up the specialized MCP Tool Agent with individual tools
- `_format_workflow_output_for_agent()` - Formats responses for agent consumption with MCP result detection
- `_ask_user_tool()` - CLI-based user interaction for clarification and approval

## Agent Initialization Pattern

```python
# Initialize MCP server and get available tools
mcp_server = FastWorkflowMCPServer(workflow_session)
available_tools = mcp_server.list_tools()

# Initialize specialized MCP Tool Agent
mcp_tool_agent = initialize_mcp_tool_agent(mcp_server, max_iters=5)

# Main agent gets simplified tool documentation
workflow_assistant_tool.__doc__ = _build_simplified_tool_documentation(available_tools)

# Main agent delegates to MCP Tool Agent
main_agent = dspy.ReAct(DialogueWithWorkflow, tools=[workflow_assistant_tool, ask_user_tool])
```

## Development Best Practices

### When Adding New Agent Capabilities
1. Extend MCP Tool Agent by modifying `initialize_mcp_tool_agent()`
2. Add new individual tool creation functions for complex query processing
3. Update main agent documentation via `_build_simplified_tool_documentation()`
4. Test delegation flow between main agent and MCP Tool Agent
5. Verify tool discovery and dynamic tool creation

### DSPy Configuration and Caching
- Use `clear_dspy_cache()` and `configure_dspy_cache()` for cache management
- `clear_cache=True` in agent initialization for fresh LLM calls
- Colorama integration for colored console output during agent execution
- LM configuration via `dspy.LM()` with model name and API key

### Testing Agent Functionality
- Use `keep_alive=True` when creating `WorkflowSession` for agents
- Test both main agent delegation and individual MCP tool execution
- Verify MCP server initialization and tool discovery
- Test error handling and graceful degradation in delegation chain
- Integration tests in `tests/test_mcp_server_integration.py`

### Debugging Agent Issues
- Check delegation logs: "Agent -> Workflow Assistant>" and "Workflow Assistant -> Agent>"
- Monitor MCP Tool Agent execution: "Workflow Assistant -> Workflow>" and "Workflow -> Workflow Assistant>"
- Verify MCP server and tool agent initialization don't fail
- Test individual tool creation and documentation generation
- Ensure `workflow_session.start()` is called before using queues

### Exception Handling Policy
**IMPORTANT**: Do not add try/catch blocks in agent code. Let exceptions bubble up naturally so developers can see and fix underlying issues. This applies to:
- JSON parsing errors in MCP tool calls
- Agent delegation failures between main and MCP tool agents
- DSPy LM configuration errors
- Individual tool creation errors
- User input interruptions in CLI 