# Delegated Agent Architecture Guide

## Overview
The FastWorkflow system uses a delegated agent architecture where the main agent delegates tool requests to a specialized MCP Tool Agent. This creates a clear separation between high-level conversation management and specific tool execution.

## Architecture Components

### Main Agent (DialogueWithWorkflow)
- Located in [fastworkflow/run_agent/agent_module.py](mdc:fastworkflow/run_agent/agent_module.py)
- Uses `DialogueWithWorkflow` DSPy signature for conversational interaction
- Exposes two tools: `WorkflowAssistant` and `AskUser`
- `WorkflowAssistant` delegates all tool requests to the specialized MCP Tool Agent
- Manages high-level conversation flow and user interaction

### MCP Tool Agent (ExecuteMCPTool)
- Specialized DSPy ReAct agent that exposes individual workflow commands as separate tools
- Uses `ExecuteMCPTool` DSPy signature for tool selection and execution
- Each workflow command becomes a dedicated DSPy tool with full documentation
- Handles tool mapping, parameter extraction, and execution
- Initialized by `initialize_mcp_tool_agent()` function

### MCP Server Integration
- [fastworkflow/mcp_server_example.py](mdc:fastworkflow/mcp_server_example.py) provides MCP-compliant tool execution
- `FastWorkflowMCPServer` class wraps the CommandExecutor with JSON-RPC 2.0 support
- Generates tool definitions with JSON schemas from Pydantic command parameter models
- Provides `list_tools()` and `call_tool()` methods for MCP compliance

## Key Functions and Flow

### Tool Creation and Discovery
- `initialize_mcp_tool_agent()`: Creates the specialized MCP Tool Agent with individual tools
- `_create_individual_query_tool()`: Creates DSPy tool functions for query string input
- `_create_individual_mcp_tool()`: Creates DSPy tool functions for MCP JSON input
- `_build_simplified_tool_documentation()`: Generates simplified docs for main agent (names + descriptions only)

### Delegation Logic
- `_execute_workflow_command_tool_with_delegation()`: Main delegation function in main agent
- Main agent passes natural language requests to MCP Tool Agent
- MCP Tool Agent selects appropriate individual tool and formats parameters
- Individual tools execute via workflow session queues (`_execute_workflow_query_tool()` or `_execute_workflow_mcp_tool()`)

### Communication Flow
```
User Query → Main Agent → WorkflowAssistant → MCP Tool Agent → Individual Tool → Workflow Session
```

## Agent Initialization Pattern
```python
# Initialize MCP server and discover tools
mcp_server = FastWorkflowMCPServer(workflow_session)
available_tools = mcp_server.list_tools()

# Create specialized MCP Tool Agent
mcp_tool_agent = initialize_mcp_tool_agent(mcp_server, max_iters=5)

# Configure main agent with delegation
workflow_assistant_tool = functools.partial(
    _execute_workflow_command_tool_with_delegation,
    mcp_tool_agent=mcp_tool_agent,
    workflow_session_obj=workflow_session
)
workflow_assistant_tool.__doc__ = _build_simplified_tool_documentation(available_tools)

# Create main agent with WorkflowAssistant and AskUser tools
main_agent = dspy.ReAct(DialogueWithWorkflow, tools=[workflow_assistant_tool, ask_user_tool])
```

## Tool Documentation Levels

### Main Agent Documentation (Simplified)
- Shows only tool names and descriptions
- Guides main agent on when to delegate to WorkflowAssistant
- Example: "get_user_details: Returns user's first and last name, address, email, payment methods, and order id's"

### Individual Tool Documentation (Detailed)
- Full parameter schemas with types, descriptions, and examples
- Tool-specific formatting requirements (query strings vs MCP JSON)
- Example usage patterns for each workflow command

## Exception Handling Policy
**CRITICAL**: Do not add try/catch blocks in this architecture. Let exceptions bubble up naturally for debugging:
- Agent delegation failures between main and MCP tool agents
- Individual tool creation or execution errors
- DSPy LM configuration errors
- MCP server initialization failures
- JSON parsing errors in tool selection

## Workflow Examples
The [examples/retail_workflow](mdc:examples/retail_workflow) demonstrates the architecture with commands like:
- `get_user_details` - User information lookup
- `get_order_details` - Order status retrieval
- `cancel_pending_order` - Order cancellation with parameters
- `list_all_product_types` - Product catalog browsing

## Testing and Debugging
- Integration tests in [tests/test_mcp_server_integration.py](mdc:tests/test_mcp_server_integration.py) cover MCP server functionality
- Monitor delegation flow via colorama-colored logs:
  - "Agent -> Workflow Assistant>" (main to MCP tool agent)
  - "Workflow Assistant -> Agent>" (MCP tool agent to main)
  - "Workflow Assistant -> Workflow>" (tool to workflow session)
  - "Workflow -> Workflow Assistant>" (workflow session to tool)
- Test both agent layers independently and together
- Verify tool discovery, creation, and execution chains

## Design Philosophy
- **Clear separation of concerns**: Main agent for conversation, MCP Tool Agent for execution
- **Dynamic tool creation**: Individual tools created from workflow command definitions
- **Natural exception flow**: No error masking, let issues surface for debugging
- **Structured delegation**: Consistent patterns for tool request routing
- **MCP compliance**: Full Model Context Protocol support with JSON-RPC 2.0 