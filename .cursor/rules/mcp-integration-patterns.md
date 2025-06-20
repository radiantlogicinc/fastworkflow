# MCP Integration Patterns Guide

## Overview
FastWorkflow integrates with the Model Context Protocol (MCP) to expose workflow commands as structured tools with automatic JSON schema generation from Pydantic models and delegated agent execution.

## Core MCP Components

### FastWorkflowMCPServer
Located in [fastworkflow/mcp_server_example.py](mdc:fastworkflow/mcp_server_example.py)
- Wraps `CommandExecutor` to provide MCP-compliant interfaces
- Generates tool definitions automatically from Pydantic command parameter models
- Handles JSON-RPC 2.0 requests (`tools/list`, `tools/call`)
- Supports external MCP client integration

### Automatic Tool Definition Generation
```python
def list_tools(self) -> List[Dict[str, Any]]:
    # Discovers command names from active workitem path
    # Extracts Pydantic command parameter classes automatically
    # Converts model fields to JSON Schema properties
    # Adds standard FastWorkflow parameters (command, workitem_path)
```

### JSON Schema Mapping from Pydantic Models
- Pydantic `Field` → JSON Schema `properties`
- `description` → JSON Schema `description`
- `pattern` → JSON Schema `pattern`  
- `examples` → JSON Schema `examples`
- `is_required()` → JSON Schema `required` array
- Field types automatically converted to JSON Schema types

## Tool Call Format Support

### MCP Tool Call JSON Format
```json
{
  "type": "mcp_tool_call",
  "tool_call": {
    "name": "command_name",
    "arguments": {
      "parameter1": "value1",
      "parameter2": "value2"
    }
  }
}
```

### Regular Command Format  
Plain text natural language queries are also supported and processed through the regular command routing system.

## Dynamic Schema Generation Process
1. `list_tools()` gets available commands for current workitem via command routing
2. For each command, extracts the `CommandParameters` Pydantic class
3. Iterates through `model_fields` to build JSON schema properties
4. Adds field descriptions, validation patterns, and examples from Pydantic Field definitions
5. Determines required vs optional fields using `field_info.is_required()`
6. Adds standard FastWorkflow parameters (`command` is required, `workitem_path` optional)
7. Creates MCP tool definition with complete `inputSchema`

## MCP Tool Annotations
```python
"annotations": {
    "title": command_name.replace("_", " ").title(),
    "readOnlyHint": False,      # Assume tools can modify state
    "destructiveHint": False,   # Conservative default
    "idempotentHint": False,
    "openWorldHint": True       # FastWorkflow can interact with external systems
}
```

## Tool Execution Flow

### MCP-Compliant Execution
1. `call_tool()` receives tool name and arguments
2. Creates `MCPToolCall` object from parameters
3. Calls `CommandExecutor.perform_mcp_tool_call()`
4. Converts to FastWorkflow `Action` via `action_from_mcp_tool_call()`
5. Executes via existing `perform_action()` method
6. Returns `MCPToolResult` with success/error status

### Command Executor Integration
- `perform_mcp_tool_call()` method provides MCP-compliant interface
- `action_from_mcp_tool_call()` static method converts MCP calls to FastWorkflow Actions
- Preserves workitem path context and parameter mapping
- Error handling returns MCP-formatted error results

## Result Format
```python
class MCPToolResult:
    isError: bool
    content: List[MCPContent]  # Contains text results and artifacts

class MCPContent:
    type: str = "text"  # Content type (text, image, etc.)
    text: Optional[str] = None  # Text content
```

## Integration with Delegated Agent Architecture

### Agent-Level Integration
- Main agent receives simplified tool documentation (names + descriptions only)
- MCP Tool Agent exposes individual commands as separate DSPy tools
- Individual tools can accept query strings or MCP JSON payloads
- Agent delegation handles tool selection and parameter formatting

### Tool Creation Patterns
- `_create_individual_query_tool()`: Creates tools that accept query strings  
- `_create_individual_mcp_tool()`: Creates tools that accept MCP JSON payloads
- Dynamic tool docstring generation with examples and parameter schemas
- Direct integration with workflow session message queues

### Result Processing
- `_format_mcp_result_for_agent()` formats MCP results for agent consumption
- Results tagged with `_mcp_source` for special formatting detection
- Maintains compatibility between MCP and regular command responses

## Example Retail Workflow Tools
The [examples/retail_workflow](mdc:examples/retail_workflow) automatically exposes these MCP tools:
- `get_user_details` - User information lookup with user_id parameter
- `get_order_details` - Order status retrieval with order_id parameter
- `cancel_pending_order` - Order cancellation with order_id and reason parameters
- `list_all_product_types` - Product catalog browsing (no parameters)
- `modify_pending_order_address` - Address updates with order_id and address parameters

## External MCP Client Support

### JSON-RPC 2.0 Protocol
- `handle_json_rpc_request()` processes standard MCP protocol requests
- Supports `tools/list` for tool discovery
- Supports `tools/call` for tool execution
- Proper error handling with standard JSON-RPC error codes
- Request/response ID tracking for client compatibility

### Server Creation Helper
```python
def create_mcp_server_for_workflow(workflow_path: str) -> FastWorkflowMCPServer:
    # Creates complete MCP server for any workflow
    # Handles FastWorkflow initialization
    # Returns ready-to-use MCP server instance
```

## Error Handling and Resilience
- MCP tool call parsing errors fall back to regular command processing
- Invalid parameters result in MCP error responses with `isError: true`
- JSON-RPC errors use standard error codes (-32603 for internal errors)
- Agent delegation failures bubble up naturally for debugging
- Schema generation gracefully handles missing or malformed Pydantic models

## Development Patterns

### Adding New Workflow Commands
1. Create command in appropriate `_commands/` directory with Pydantic parameter class
2. MCP tool automatically available via `list_tools()` with generated schema
3. Test via MCP server, agent integration, or external MCP clients
4. No additional MCP-specific configuration required

### Testing MCP Integration
- Use [tests/test_mcp_server_integration.py](mdc:tests/test_mcp_server_integration.py) patterns
- Test tool discovery, schema generation, and execution
- Test both direct MCP server calls and agent delegation
- Use real workflow data for realistic scenarios 