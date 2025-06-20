"""
Example MCP Server for FastWorkflow integration.

This demonstrates how to wrap FastWorkflow's CommandExecutor 
to provide MCP-compliant tool execution.
"""

from typing import Dict, Any, List
import fastworkflow
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.command_routing_definition import CommandRoutingDefinition, ModuleType
from uuid import uuid4


class FastWorkflowMCPServer:
    """
    MCP Server wrapper for FastWorkflow CommandExecutor.
    
    This class provides MCP-compliant interfaces while leveraging
    the existing FastWorkflow command execution infrastructure.
    """
    
    def __init__(self, workflow_session: fastworkflow.WorkflowSession):
        """
        Initialize the MCP server.
        
        Args:
            workflow_session: Active FastWorkflow session
        """
        self.workflow_session = workflow_session
        self.command_executor = CommandExecutor()
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """
        Return MCP-compliant tool definitions.
        
        Returns:
            List of tool definitions in MCP format
        """
        NOT_FOUND = fastworkflow.get_env_var('NOT_FOUND')

        # Get available commands from workflow
        workflow_folderpath = self.workflow_session.session.workflow_snapshot.workflow_folderpath
        routing = CommandRoutingDefinition.build(workflow_folderpath)

        # Get active context name from session, not workflow_snapshot
        active_ctx_name = self.workflow_session.session.current_command_context_name
        if active_ctx_name not in routing.contexts:
            active_ctx = '*'
        else:
            active_ctx = active_ctx_name
        command_names = routing.get_command_names(active_ctx)

        tools = []
        for command_name in command_names:
            # Get command parameters class to build schema
            command_parameters_class = routing.get_command_class(
                command_name,
                ModuleType.COMMAND_PARAMETERS_CLASS
            )

            # Build JSON schema from Pydantic model
            input_schema = {
                "type": "object",
                "properties": {},
                "required": []
            }

            description = f"Executes {command_name}"

            if command_parameters_class:
                field_descriptions = []
                model_fields = command_parameters_class.model_fields
                for field_name, field_info in model_fields.items():
                    # Convert Pydantic field to JSON schema property
                    prop = {"type": "string"}  # Default type
                    if hasattr(field_info, 'description') and field_info.description:
                        prop["description"] = field_info.description
                        field_descriptions.append(field_info.description)
                    else:
                        field_descriptions.append(field_name)

                    if hasattr(field_info, 'default') and field_info.default:
                        prop["default"] = field_info.default
                        # Add to required if default is NOT_FOUND
                        if field_info.default == NOT_FOUND:
                            input_schema["required"].append(field_name)
                    elif field_info.is_required():
                        input_schema["required"].append(field_name)

                    input_schema["properties"][field_name] = prop

                if field_descriptions:
                    description = f"Input: {', '.join(field_descriptions)}"
                if command_parameters_class.__doc__:
                    description = f"{description}. {command_parameters_class.__doc__.strip()}"

            # Add standard FastWorkflow parameters
            input_schema["properties"]["command"] = {
                "type": "string",
                "description": "Natural language command or query"
            }

            input_schema["properties"]["workitem_path"] = {
                "type": "string",
                "description": "Command context (optional)",
                "default": active_ctx
            }

            tool_def = {
                "name": command_name,
                "description": description,
                "inputSchema": input_schema,
                "annotations": {
                    "title": command_name.replace("_", " ").title(),
                    "readOnlyHint": False,  # Assume tools can modify state
                    "destructiveHint": False,  # Conservative default
                    "idempotentHint": False,
                    "openWorldHint": True  # FastWorkflow can interact with external systems
                }
            }
            tools.append(tool_def)

        return tools
    
    def call_tool(self, name: str, arguments: Dict[str, Any]) -> fastworkflow.MCPToolResult:
        """
        Execute a tool call in MCP format.
        
        Args:
            name: Tool name (command name)
            arguments: Tool arguments
            
        Returns:
            MCPToolResult: Result in MCP format
        """
        # Create MCP tool call
        tool_call = fastworkflow.MCPToolCall(
            name=name,
            arguments=arguments
        )
        
        # Execute using MCP-compliant method
        return self.command_executor.perform_mcp_tool_call(
            self.workflow_session.session,
            tool_call,
            command_context=self._resolve_workitem_path_for_call()
        )
    
    def handle_json_rpc_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle JSON-RPC 2.0 request in MCP format.
        
        Args:
            request: JSON-RPC request
            
        Returns:
            JSON-RPC response
        """
        try:
            method = request.get("method")
            params = request.get("params", {})
            request_id = request.get("id")
            
            if method == "tools/list":
                result = {"tools": self.list_tools()}
            elif method == "tools/call":
                tool_result = self.call_tool(
                    params.get("name"),
                    params.get("arguments", {})
                )
                result = tool_result.model_dump()
            else:
                raise ValueError(f"Unknown method: {method}")
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
            
        except Exception as e:
            return {
                "jsonrpc": "2.0", 
                "id": request.get("id"),
                "error": {
                    "code": -32603,  # Internal error
                    "message": str(e)
                }
            }

    # ---------------------------------------------------------------------
    def _resolve_workitem_path_for_call(self) -> str:
        """Return a workitem path that has registered commands.

        Falls back to the first available path if the active workitem has none.
        """
        return '*'


# Example usage
def create_mcp_server_for_workflow(workflow_path: str) -> FastWorkflowMCPServer:
    """
    Create an MCP server for a FastWorkflow workflow.
    
    Args:
        workflow_path: Path to workflow definition
        
    Returns:
        FastWorkflowMCPServer: Configured MCP server
    """
    # Initialize FastWorkflow (would need actual env vars in practice)
    fastworkflow.init({})
    
    # Recreate routing definition to guarantee it reflects the latest object model
    CommandRoutingDefinition.build(workflow_path)
    
    # Create workflow session
    workflow_session = fastworkflow.WorkflowSession(
        CommandExecutor(),
        workflow_path,
        session_id_str=str(uuid4())
    )
    
    return FastWorkflowMCPServer(workflow_session) 