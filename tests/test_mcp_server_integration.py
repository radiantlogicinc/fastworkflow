"""
Integration tests for FastWorkflow MCP Server functionality.

These tests use the sample_workflow example to test MCP server integration
without mocks, providing end-to-end testing of the MCP compliance features.
"""

import pytest
import json
import os
from typing import Dict, Any
import uuid

import fastworkflow
from fastworkflow.mcp_server import FastWorkflowMCPServer, create_mcp_server_for_workflow
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.command_routing_definition import CommandRoutingDefinition


@pytest.fixture(scope="module")
def sample_workflow_path():
    """Get the path to the sample workflow example."""
    return os.path.join(os.path.dirname(__file__), "..", "examples", "retail_workflow")


@pytest.fixture(scope="module")
def initialized_fastworkflow():
    """Initialize FastWorkflow with minimal configuration."""
    # Initialize with empty env vars for testing
    fastworkflow.init({})
    return True


@pytest.fixture
def workflow_session(sample_workflow_path, initialized_fastworkflow):
    """Create a FastWorkflow session for the sample workflow."""
    # Build command routing definition once so that command metadata is ready
    CommandRoutingDefinition.build(sample_workflow_path)
    return fastworkflow.WorkflowSession(
        CommandExecutor(), sample_workflow_path, session_id_str=str(uuid.uuid4())
    )


@pytest.fixture
def mcp_server(workflow_session):
    """Create an MCP server instance."""
    return FastWorkflowMCPServer(workflow_session)


class TestFastWorkflowMCPServer:
    """Test the FastWorkflowMCPServer class."""

    def test_server_initialization(self, mcp_server):
        """Test that MCP server initializes correctly."""
        assert mcp_server is not None
        assert mcp_server.workflow_session is not None
        assert mcp_server.command_executor is not None

    def test_list_tools_returns_valid_schema(self, mcp_server):
        """Test that list_tools returns valid MCP tool definitions."""
        tools = mcp_server.list_tools()
        
        assert isinstance(tools, list)
        assert len(tools) > 0
        
        # Check that sample workflow commands are present
        tool_names = [tool["name"] for tool in tools]
        expected_commands = [
            "list_all_product_types",
            "find_user_id_by_email",
            "get_order_details",
            "get_product_details"
        ]
        
        for command in expected_commands:
            assert command in tool_names, f"Command {command} not found in tools"

    def test_tool_schema_structure(self, mcp_server):
        """Test that each tool has the correct MCP schema structure."""
        tools = mcp_server.list_tools()
        
        for tool in tools:
            # Check required MCP tool fields
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert "annotations" in tool
            
            # Check inputSchema structure
            schema = tool["inputSchema"]
            assert schema["type"] == "object"
            assert "properties" in schema
            assert "required" in schema
            
            # Check that command is always present
            assert "command" in schema["properties"]
            
            # Check annotations
            annotations = tool["annotations"]
            assert "title" in annotations
            assert "readOnlyHint" in annotations
            assert "destructiveHint" in annotations
            assert "idempotentHint" in annotations
            assert "openWorldHint" in annotations

    def test_call_tool_with_simple_command(self, mcp_server):
        """Test calling a tool with no required parameters."""
        result = mcp_server.call_tool(
            name="list_all_product_types",
            arguments={}
        )
        
        assert isinstance(result, fastworkflow.MCPToolResult)
        assert result.isError is False
        assert "product" in result.content[0].text.lower()

    def test_call_tool_with_parameters(self, mcp_server):
        """Test calling a tool that requires parameters (get_path_and_id)."""
        result = mcp_server.call_tool(
            name="find_user_id_by_email",
            arguments={
                "command": "find user by email",
                "email": "john.doe@example.com"
            }
        )

        assert isinstance(result, fastworkflow.MCPToolResult)
        assert result.isError is False
        assert len(result.content) > 0
        assert "user id" in result.content[0].text.lower()

    def test_call_tool_with_invalid_parameters(self, mcp_server):
        """Test calling a tool with missing required parameters to induce validation failure."""
        result = mcp_server.call_tool(
            name="find_user_id_by_email",
            arguments={
                "command": "find user by email"
                # missing email parameter expected to default but still acceptable.
            }
        )

        assert isinstance(result, fastworkflow.MCPToolResult)
        # May not be an error, but should not crash

    def test_call_nonexistent_tool(self, mcp_server):
        """Test calling a tool that doesn't exist."""
        result = mcp_server.call_tool(
            name="nonexistent_tool",
            arguments={"command": "test"}
        )
        
        assert isinstance(result, fastworkflow.MCPToolResult)
        assert result.isError is True
        assert "nonexistent_tool" in result.content[0].text

    def test_handle_json_rpc_tools_list(self, mcp_server):
        # sourcery skip: class-extract-method
        """Test JSON-RPC tools/list request."""
        request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "id": "test-123"
        }
        
        response = mcp_server.handle_json_rpc_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "test-123"
        assert "result" in response
        assert "tools" in response["result"]
        assert len(response["result"]["tools"]) > 0

    def test_handle_json_rpc_tools_call(self, mcp_server):
        """Test JSON-RPC tools/call request."""
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "list_all_product_types",
                "arguments": {}
            },
            "id": "test-456"
        }
        
        response = mcp_server.handle_json_rpc_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "test-456"
        assert "result" in response
        assert "content" in response["result"]
        assert response["result"]["isError"] is False

    def test_handle_json_rpc_invalid_method(self, mcp_server):
        """Test JSON-RPC request with invalid method."""
        request = {
            "jsonrpc": "2.0",
            "method": "invalid/method",
            "id": "test-789"
        }
        
        response = mcp_server.handle_json_rpc_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "test-789"
        assert "error" in response
        assert response["error"]["code"] == -32603  # Internal error

    def test_handle_json_rpc_malformed_request(self, mcp_server):
        """Test JSON-RPC with malformed request."""
        request = {
            "jsonrpc": "2.0",
            # Missing method
            "id": "test-bad"
        }
        
        response = mcp_server.handle_json_rpc_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == "test-bad"
        assert "error" in response


class TestRetailWorkflowCommands:
    """Test specific retail workflow commands through MCP interface."""

    def test_find_user_by_email(self, mcp_server):
        """Test finding user by email."""
        result = mcp_server.call_tool(
            name="find_user_id_by_email",
            arguments={
                "command": "Find user by email",
                "email": "sara.doe@example.com"
            }
        )
        
        assert isinstance(result, fastworkflow.MCPToolResult)
        assert len(result.content) > 0
        assert result.content[0].type == "text"
        assert result.isError is False
        assert "user not found" in result.content[0].text

    def test_find_user_by_name_zip(self, mcp_server):
        """Test finding user by name and zip."""
        result = mcp_server.call_tool(
            name="find_user_id_by_name_zip",
            arguments={
                "command": "Find user by name and zip",
                "first_name": "Sara",
                "last_name": "Doe", 
                "zip": "12345"
            }
        )
        
        assert isinstance(result, fastworkflow.MCPToolResult)
        assert len(result.content) > 0
        assert result.content[0].type == "text"
        assert result.isError is False
        assert "user id" in result.content[0].text.lower()

    def test_get_order_details(self, mcp_server):
        """Test getting order details."""
        result = mcp_server.call_tool(
            name="get_order_details",
            arguments={
                "command": "Get order details",
                "order_id": "#W0000001"
            }
        )

        assert isinstance(result, fastworkflow.MCPToolResult)
        assert len(result.content) > 0
        assert result.content[0].type == "text"
        assert result.isError is False
        assert "Order details:" in result.content[0].text

    def test_get_product_details(self, mcp_server):
        """Test getting product details."""
        result = mcp_server.call_tool(
            name="get_product_details",
            arguments={
                "command": "Get product details",
                "product_id": "6086499569"
            }
        )
        
        assert isinstance(result, fastworkflow.MCPToolResult)
        assert len(result.content) > 0
        assert result.content[0].type == "text"
        assert result.isError is False
        assert "product not found" in result.content[0].text

    def test_cancel_pending_order(self, mcp_server):
        """Test cancelling a pending order."""
        result = mcp_server.call_tool(
            name="cancel_pending_order",
            arguments={
                "command": "Cancel my order",
                "order_id": "#W0000001",
                "reason": "no longer needed"
            }
        )
        
        assert isinstance(result, fastworkflow.MCPToolResult)
        assert len(result.content) > 0
        assert result.content[0].type == "text"
        assert result.isError is False
        assert "current status is:" in result.content[0].text

    def test_transfer_to_human_agents(self, mcp_server):
        """Test transferring to human agents."""
        result = mcp_server.call_tool(
            name="transfer_to_human_agents",
            arguments={
                "command": "I need to speak to a human",
                "summary": "Customer needs complex assistance"
            }
        )

        assert isinstance(result, fastworkflow.MCPToolResult)
        assert len(result.content) > 0
        assert result.content[0].type == "text"
        assert result.isError is False
        assert "transfer status:" in result.content[0].text


class TestMCPServerCreation:
    """Test the create_mcp_server_for_workflow function."""

    def test_create_server_for_workflow(self, sample_workflow_path, initialized_fastworkflow):
        """Test creating MCP server for a workflow path."""
        server = create_mcp_server_for_workflow(sample_workflow_path)
        
        assert isinstance(server, FastWorkflowMCPServer)
        assert server.workflow_session is not None
        
        # Test that it can list tools
        tools = server.list_tools()
        assert len(tools) > 0


class TestMCPDataConversion:
    """Test MCP data format conversions."""

    def test_command_output_to_mcp_result(self):
        """Test converting CommandOutput to MCPToolResult."""
        command_response = fastworkflow.CommandResponse(
            response="Test response",
            success=True
        )
        command_output = fastworkflow.CommandOutput(
            command_responses=[command_response]
        )
        
        mcp_result = command_output.to_mcp_result()
        
        assert isinstance(mcp_result, fastworkflow.MCPToolResult)
        assert mcp_result.isError is False
        assert len(mcp_result.content) == 1
        assert mcp_result.content[0].type == "text"
        assert mcp_result.content[0].text == "Test response"

    def test_command_output_to_mcp_result_with_error(self):
        """Test converting failed CommandOutput to MCPToolResult."""
        command_response = fastworkflow.CommandResponse(
            response="Error occurred",
            success=False
        )
        command_output = fastworkflow.CommandOutput(
            command_responses=[command_response]
        )
        
        mcp_result = command_output.to_mcp_result()
        
        assert isinstance(mcp_result, fastworkflow.MCPToolResult)
        assert mcp_result.isError is True
        assert mcp_result.content[0].text == "Error occurred"


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 