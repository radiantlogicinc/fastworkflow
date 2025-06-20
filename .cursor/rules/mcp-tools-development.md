# MCP Tools Development Guide

## Adding New MCP Tools

### Workflow Commands Automatically Become MCP Tools
All workflow commands in `_commands/` directories automatically become MCP tools via:
- **[fastworkflow/mcp_server_example.py](mdc:fastworkflow/mcp_server_example.py)** - `list_tools()` method discovers commands automatically
- **[fastworkflow/command_executor.py](mdc:fastworkflow/command_executor.py)** - `perform_mcp_tool_call()` method executes commands
- **[fastworkflow/run_agent/agent_module.py](mdc:fastworkflow/run_agent/agent_module.py)** - Creates individual DSPy tools for delegated execution

### Automatic MCP Tool Schema Generation
```python
# Schema generated automatically from Pydantic command parameter classes
command_parameters_class = command_routing_definition.get_command_class(
    active_workitem_path, 
    command_name, 
    CommandModuleType.COMMAND_PARAMETERS_CLASS
)
```

**Schema includes:**
- JSON Schema automatically generated from Pydantic model fields
- Required vs optional parameters via `field_info.is_required()`
- Field descriptions, validation patterns, and examples from Pydantic Field annotations
- Standard `command` parameter (required) and optional `workitem_path` parameter

## MCP Data Classes and Conversion

### Core MCP Types in [fastworkflow/__init__.py](mdc:fastworkflow/__init__.py)
```python
class MCPToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any] = {}

class MCPContent(BaseModel):  
    type: str = "text"
    text: Optional[str] = None

class MCPToolResult(BaseModel):
    content: List[MCPContent]
    isError: bool = False
```

### Converting Between Formats
- `CommandOutput.to_mcp_result()` - Converts workflow results to MCP format
- `CommandExecutor.action_from_mcp_tool_call()` - Converts MCP calls to FastWorkflow Actions
- `WorkflowSession._convert_mcp_result_to_command_output()` - Maintains queue compatibility with `_mcp_source` marker

## MCP Server Implementation

### Key Methods in FastWorkflowMCPServer
```python
def list_tools() -> List[Dict[str, Any]]     # Automatic tool discovery from workflow commands
def call_tool(name: str, arguments: Dict)    # MCP-compliant tool execution  
def handle_json_rpc_request(request: Dict)   # JSON-RPC 2.0 protocol support
```

### Automatic Tool Definition Format
```json
{
    "name": "command_name",
    "description": "Generated from Pydantic class docstring and field descriptions", 
    "inputSchema": {...},  // JSON Schema from Pydantic model
    "annotations": {
        "title": "Command Name",
        "readOnlyHint": false,
        "destructiveHint": false,
        "idempotentHint": false,
        "openWorldHint": true
    }
}
```

## Agent Integration Patterns

### Delegated Agent Architecture for MCP Tools
- **Main Agent**: Gets simplified tool documentation (names + descriptions only)
- **MCP Tool Agent**: Exposes individual workflow commands as separate DSPy tools
- **Individual Tools**: Created via `_create_individual_query_tool()` and `_create_individual_mcp_tool()`

### Dynamic Tool Creation
```python
def _create_individual_query_tool(tool_def: Dict, workflow_session_obj):
    # Creates DSPy tool function for query string input
    # Generates tool docstring with examples and parameter schemas
    # Returns callable tool function for DSPy ReAct agent

def _create_individual_mcp_tool(tool_def: Dict, workflow_session_obj):
    # Creates DSPy tool function for MCP JSON payload input
    # Handles complete MCP JSON string formatting
    # Provides examples of properly formatted MCP JSON calls
```

### Tool Documentation Generation
- `_build_simplified_tool_documentation()` for main agent (names + descriptions)
- Individual tools get complete parameter schemas with types, examples, and usage patterns
- Automatic docstring generation with MCP JSON examples for individual tools

## Development Patterns

### Adding New Workflow Commands
1. Create command in appropriate `_commands/` directory following standard structure:
   ```
   _commands/
     <command_name>/
       parameter_extraction/signatures.py  # CommandParameters Pydantic class
       response_generation/
       utterances/
   ```
2. Define Pydantic `CommandParameters` class with Field annotations for automatic schema generation
3. MCP tool automatically available via `list_tools()` - no additional configuration needed
4. Tool appears in both MCP Tool Agent and external MCP client tool lists

### Example Command Parameters Pattern
```python
class CommandParameters(BaseModel):
    """Tool description that becomes MCP tool description"""
    user_id: Annotated[
        str,
        Field(
            default="NOT_FOUND",
            description="The user ID to get details for",
            pattern=r"^([a-z]+_[a-z]+_\d+|NOT_FOUND)$",
            examples=["sara_doe_496"]
        )
    ]
```

### Testing New MCP Tools
- **[tests/test_mcp_server_integration.py](mdc:tests/test_mcp_server_integration.py)** - Add test methods for new commands
- Test automatic tool discovery via `mcp_server.list_tools()`
- Test direct tool execution via `mcp_server.call_tool()`
- Test agent delegation through individual tool creation
- Use real workflow data from `examples/retail_workflow/` for realistic testing

### MCP Protocol Compliance Testing
- Test JSON-RPC 2.0 protocol via `handle_json_rpc_request()`
- Verify tool definitions include proper JSON schemas
- Test error handling with invalid parameters and nonexistent tools
- Validate MCP result format with `content` array and `isError` flag

## Common Use Cases

### External MCP Clients
```python
# Create MCP server for any workflow
server = create_mcp_server_for_workflow("path/to/workflow")

# Discover available tools
tools = server.list_tools()

# Execute tool with parameters
result = server.call_tool("get_user_details", {"user_id": "sara_doe_496", "command": "Get user details"})
```

### Agent Integration
```python  
# Main agent delegates to MCP Tool Agent
main_agent("Get details for user sara_doe_496")

# MCP Tool Agent selects and executes appropriate individual tool
# Individual tool formats and executes the specific command
# Results flow back through delegation chain to main agent
```

### Direct Workflow Integration
```python
# MCP tool calls are automatically routed in WorkflowSession
# JSON MCP format: {"type": "mcp_tool_call", "tool_call": {...}}
# Regular commands: Plain text natural language
# Both formats supported seamlessly
```

## Error Handling and Debugging

### Exception Handling Policy
**CRITICAL**: Do not add try/catch blocks in MCP tools development. Let exceptions bubble up naturally:
- Schema generation errors from malformed Pydantic models
- Tool execution errors from invalid parameters
- Agent delegation failures between main and MCP tool agents
- JSON parsing errors in MCP tool call processing

### Debugging Tools Integration
- Monitor tool creation: Check `initialize_mcp_tool_agent()` logs
- Verify schema generation: Inspect `list_tools()` output for correct JSON schemas
- Test delegation flow: Watch colorama-colored logs for agent communication
- Validate MCP compliance: Use integration tests for protocol verification 