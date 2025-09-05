"""
Example MCP Server for FastWorkflow integration.

This demonstrates how to wrap FastWorkflow's CommandExecutor 
to provide MCP-compliant tool execution.
"""

from typing import Dict, Any, List
import fastworkflow
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.command_directory import CommandDirectory
from fastworkflow.command_routing import RoutingDefinition, RoutingRegistry, ModuleType
from uuid import uuid4
from fastworkflow.command_metadata_api import CommandMetadataAPI


class FastWorkflowMCPServer:
    """
    MCP Server wrapper for FastWorkflow CommandExecutor.
    
    This class provides MCP-compliant interfaces while leveraging
    the existing FastWorkflow command execution infrastructure.
    """
    
    def __init__(self, chat_session: fastworkflow.ChatSession):
        """
        Initialize the MCP server.
        
        Args:
            chat_session: Active FastWorkflow chat session
        """
        self.chat_session = chat_session
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """
        Return MCP-compliant tool definitions.
        
        Returns:
            List of tool definitions in MCP format
        """
        NOT_FOUND = fastworkflow.get_env_var('NOT_FOUND')

        # Get available commands from workflow
        workflow = fastworkflow.ChatSession.get_active_workflow()
        workflow_folderpath = workflow.folderpath
        # Use cached routing definition instead of rebuilding every time
        routing = RoutingRegistry.get_definition(workflow_folderpath)

        # Get active context name from chat session
        active_ctx_name = workflow.current_command_context_name
        active_ctx = active_ctx_name if active_ctx_name in routing.contexts else '*'
        command_names = routing.get_command_names(active_ctx)

        # Centralized command metadata (docstrings, inputs, plain_utterances)
        cme_path = fastworkflow.get_internal_workflow_path("command_metadata_extraction")
        enhanced_meta = CommandMetadataAPI.get_enhanced_command_info(
            subject_workflow_path=workflow_folderpath,
            cme_workflow_path=cme_path,
            active_context_name=active_ctx_name,
        )
        meta_by_fq = {m.get("qualified_name"): m for m in enhanced_meta.get("commands", [])}

        # Centralized parameters for building schemas
        params_by_cmd = CommandMetadataAPI.get_params_for_all_commands(workflow_folderpath)

        tools = []
        for command_name in command_names:
            # Centralized metadata for this command (if any)
            meta_for_cmd = meta_by_fq.get(command_name)
            # Build JSON schema from centralized API params
            input_schema = {
                "type": "object",
                "properties": {},
                "required": []
            }

            # Build input schema and required fields
            field_descriptions = []
            for param in params_by_cmd.get(command_name, {}).get("inputs", []):
                field_name = param.get("name") or "param"
                prop = {"type": "string"}
                if desc := param.get("description"):
                    prop["description"] = desc
                    field_descriptions.append(desc)
                else:
                    field_descriptions.append(field_name)
                if (default := param.get("default")) is not None:
                    if default == NOT_FOUND:
                        input_schema["required"].append(field_name)
                input_schema["properties"][field_name] = prop

            # Use centralized display generator for a single command for rich description
            description = CommandMetadataAPI.get_command_display_text_for_command(
                subject_workflow_path=workflow_folderpath,
                cme_workflow_path=cme_path,
                active_context_name=active_ctx_name,
                qualified_command_name=command_name,
                for_agents=True,
                omit_command_name=True,
            )

            # Add standard FastWorkflow parameters
            # input_schema["properties"]["command"] = {
            #     "type": "string",
            #     "description": "Natural language command or query"
            # }

            # input_schema["properties"]["workitem_path"] = {
            #     "type": "string",
            #     "description": "Command context (optional)",
            #     "default": active_ctx
            # }

            tool_def = {
                "name": command_name.split("/")[-1],
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
            # Attach plain_utterances from centralized metadata if present
            if meta_for_cmd and meta_for_cmd.get("plain_utterances"):
                tool_def["annotations"]["plain_utterances"] = meta_for_cmd["plain_utterances"]
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
        
        workflow = fastworkflow.ChatSession.get_active_workflow()
        # Execute using MCP-compliant method
        return CommandExecutor.perform_mcp_tool_call(
            workflow,
            tool_call,
            command_context=self._resolve_context_for_call()
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
    def _resolve_context_for_call(self) -> str:
        """Return the context that has registered commands.

        Falls back to the first available path if the active context is none.
        """
        workflow = fastworkflow.ChatSession.get_active_workflow()
        return workflow.current_command_context_name


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
    
    # Ensure routing definition exists (build once if missing)
    try:
        RoutingRegistry.get_definition(workflow_path)
    except Exception:
        RoutingDefinition.build(workflow_path)
    
    # Create workflow chat session and start the workflow so an active workflow exists
    chat_session = fastworkflow.ChatSession()
    # Start in keep-alive mode so creation does not block on the workflow loop
    chat_session.start_workflow(
        workflow_path,
        workflow_id_str=str(uuid4()),
        keep_alive=True,
    )
    
    return FastWorkflowMCPServer(chat_session)