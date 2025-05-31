# FastWorkflow MCP Integration Tests

This directory contains integration tests for the FastWorkflow MCP (Model Context Protocol) server functionality.

## Test Structure

- `test_mcp_server_integration.py` - Main integration tests using the retail workflow
- `conftest.py` - Pytest configuration and shared fixtures
- `README.md` - This file

## Prerequisites

Make sure you have the required dependencies installed:

```bash
pip install pytest pydantic
```

## Running the Tests

### Run all MCP integration tests
```bash
# From the project root
pytest tests/test_mcp_server_integration.py -v

# Or from the tests directory
cd tests
pytest test_mcp_server_integration.py -v
```

### Run specific test classes
```bash
# Test just the MCP server functionality
pytest tests/test_mcp_server_integration.py::TestFastWorkflowMCPServer -v

# Test retail workflow commands
pytest tests/test_mcp_server_integration.py::TestRetailWorkflowCommands -v

# Test data conversions
pytest tests/test_mcp_server_integration.py::TestMCPDataConversion -v
```

### Run with markers
```bash
# Run only integration tests
pytest -m integration -v

# Run only slow tests
pytest -m slow -v

# Skip slow tests
pytest -m "not slow" -v
```

## Test Coverage

The integration tests cover:

1. **MCP Server Initialization**
   - Server creation and setup
   - Workflow session integration

2. **Tool Discovery**
   - `list_tools()` functionality
   - MCP-compliant tool schema generation
   - Validation of tool definitions

3. **Tool Execution**
   - `call_tool()` with various retail workflow commands
   - Parameter handling and validation
   - Error handling for invalid inputs

4. **JSON-RPC Protocol**
   - `tools/list` requests
   - `tools/call` requests  
   - Error responses
   - Malformed request handling

5. **Retail Workflow Commands**
   - All major retail workflow operations
   - Real parameter validation
   - Actual command execution

6. **Data Format Conversions**
   - `CommandOutput` to `MCPToolResult`
   - `MCPToolCall` to `Action`
   - Error state handling

## Test Data

The tests use the actual retail workflow example data located in:
- `examples/retail_workflow/retail_data/`
- `examples/retail_workflow/_base_commands/`

This provides realistic test scenarios without requiring mock data.

## Adding New Tests

When adding new MCP functionality:

1. Add test methods to the appropriate test class
2. Use the `mcp_server` fixture for server instance
3. Test both success and error cases
4. Verify MCP compliance of inputs/outputs
5. Use real retail workflow commands when possible

## Debugging Tests

For debugging failed tests:

```bash
# Run with detailed output
pytest tests/test_mcp_server_integration.py -v -s

# Run a specific test with debugging
pytest tests/test_mcp_server_integration.py::TestFastWorkflowMCPServer::test_list_tools_returns_valid_schema -v -s --pdb

# Show local variables on failure
pytest tests/test_mcp_server_integration.py -v -l
```

## Integration Test Philosophy

These are integration tests, not unit tests:

- ✅ **DO**: Use real workflow data and commands
- ✅ **DO**: Test end-to-end functionality  
- ✅ **DO**: Verify actual MCP protocol compliance
- ❌ **DON'T**: Mock FastWorkflow components
- ❌ **DON'T**: Use fake data when real data is available
- ❌ **DON'T**: Test individual functions in isolation 