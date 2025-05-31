# MCP Integration Architecture

## Overview
FastWorkflow implements comprehensive Model Context Protocol (MCP) compliance with intelligent message routing, delegated agent integration, and full JSON-RPC 2.0 support for external MCP clients.

## Core Components

### MCP Server Implementation
- **[fastworkflow/mcp_server_example.py](mdc:fastworkflow/mcp_server_example.py)** - Complete MCP server with JSON-RPC 2.0 protocol support
- **[fastworkflow/__init__.py](mdc:fastworkflow/__init__.py)** - MCP data classes (`MCPToolCall`, `MCPContent`, `MCPToolResult`)
- **[fastworkflow/command_executor.py](mdc:fastworkflow/command_executor.py)** - MCP-compliant command execution methods

### Agent Integration Architecture
- **[fastworkflow/run_agent/agent_module.py](mdc:fastworkflow/run_agent/agent_module.py)** - Delegated agent architecture with MCP Tool Agent
- **[fastworkflow/workflow_session.py](mdc:fastworkflow/workflow_session.py)** - Intelligent message routing for MCP vs regular commands

## Message Flow Architecture

```
Agent Query → Main Agent → MCP Tool Agent → Individual Tools → WorkflowSession
                                                                      ↓
                                                            Message Routing Logic
                                                                      ↓
                                                      _process_mcp_tool_call() or _process_message()
```

## Key Implementation Details

### Intelligent Message Routing in WorkflowSession
- `_is_mcp_tool_call()` detects JSON MCP tool call format via `"type": "mcp_tool_call"` marker
- `_process_mcp_tool_call()` handles structured MCP tool calls with validation
- `_process_message()` handles regular natural language commands
- `_convert_mcp_result_to_command_output()` maintains compatibility between MCP and regular responses

### Agent Layer Delegation
- Main agent creates MCP Tool Agent via `initialize_mcp_tool_agent()`
- Individual tools created dynamically: `_create_individual_query_tool()` and `_create_individual_mcp_tool()`
- Tool documentation enforced through DSPy tool docstrings with examples
- No client-side validation - all processing delegated to specialized agents and workflow session

### Tool Discovery & Documentation
- Agent initialization creates `FastWorkflowMCPServer` and calls `list_tools()`
- Automatic schema generation from Pydantic command parameter models
- `_build_simplified_tool_documentation()` for main agent (names + descriptions)
- Individual tools get complete schemas with parameter types, descriptions, and examples
- Graceful fallback if MCP server initialization fails

### MCP Data Format Support
Two supported input formats in workflow session:
1. **MCP Tool Call JSON**: `{"type": "mcp_tool_call", "tool_call": {"name": "tool_name", "arguments": {...}}}`
2. **Regular Commands**: Plain text natural language queries

### Processing Flow
- WorkflowSession routes based on JSON detection in `_run_workflow_loop()`
- MCP tool calls → `_process_mcp_tool_call()` → `CommandExecutor.perform_mcp_tool_call()`
- Regular commands → `_process_message()` → `CommandRouter.route_command()`
- Results marked with `_mcp_source` for special formatting in agent responses

### Output Formatting
- `_format_workflow_output_for_agent()` detects MCP results via `_mcp_source` marker
- `_format_mcp_result_for_agent()` handles MCP-specific result formatting
- Maintains backward compatibility for command response structures
- Error handling preserves MCP `isError` flag and error content

## JSON-RPC 2.0 Protocol Support

### External MCP Client Integration
- `handle_json_rpc_request()` processes standard MCP protocol requests
- Supports `tools/list` and `tools/call` methods
- Proper error handling with JSON-RPC error codes
- Request/response ID tracking for protocol compliance

### Tool Definition Generation
- JSON Schema generation from Pydantic model fields
- MCP annotations with `readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`
- Standard FastWorkflow parameters (`command`, optional `workitem_path`)
- Tool titles generated from command names

## Testing
- **[tests/test_mcp_server_integration.py](mdc:tests/test_mcp_server_integration.py)** - 23+ comprehensive integration tests
- Tests cover tool discovery, execution, JSON-RPC protocol, and data conversions
- Uses retail workflow data for realistic testing scenarios
- No mocks - full end-to-end testing of MCP compliance

## Design Philosophy
- **Intelligent routing**: Automatic detection and routing of MCP vs regular messages
- **Agent delegation**: Main agent delegates to specialized MCP Tool Agent for execution
- **Protocol compliance**: Full MCP and JSON-RPC 2.0 support for external clients
- **Graceful fallback**: MCP tool calls can fall back to regular message processing on errors
- **Natural exceptions**: No error masking - let issues bubble up for debugging
- **Dynamic tool creation**: Workflow commands automatically become MCP tools

## Error Handling and Resilience
- MCP tool call parsing errors fall back to regular message processing
- Individual tool failures return MCP error results with `isError: true`
- Agent delegation failures bubble up naturally for debugging
- JSON-RPC errors use standard error codes and proper response format 