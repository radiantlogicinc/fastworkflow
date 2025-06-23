from enum import Enum
import os
from typing import Any, Optional, Union

from pydantic import BaseModel
import mmh3


class Action(BaseModel):
    command_name: str
    command: str = ""   # only use is to display autocomplete item in the UI
    parameters: dict[str, Optional[Union[str, bool, int, float, BaseModel]]] = {}
    session_id: Optional[int] = None    # when creating a new action, this is set by the workflow session

# MCP-compliant classes
class MCPToolCall(BaseModel):
    """MCP-compliant tool call request format"""
    name: str
    arguments: dict[str, Any] = {}

class MCPContent(BaseModel):
    """MCP content block"""
    type: str  # "text", "image", etc.
    text: Optional[str] = None
    # Add other content type fields as needed

class MCPToolResult(BaseModel):
    """MCP-compliant tool result format"""
    content: list[MCPContent]
    isError: bool = False

class Recommendation(BaseModel):
    summary: str
    suggested_actions: list[Action] = []

class CommandResponse(BaseModel):
    response: str
    success: bool = True
    artifacts: dict[str, Any] = {}
    next_actions: list[Action] = []
    recommendations: list[Recommendation] = []

class CommandOutput(BaseModel):
    command_responses: list[CommandResponse]

    @property
    def success(self) -> bool:
        return all(response.success for response in self.command_responses)

    @property
    def command_aborted(self) -> bool:
        return any(response.artifacts.get("command_name", None) == "abort" for response in self.command_responses)

    @property
    def command_handled(self) -> bool:
        return any(response.artifacts.get("command_handled", False) == True for response in self.command_responses)
    
    @property
    def not_what_i_meant(self) -> bool:
        return any(response.artifacts.get("command_name", None) == "misunderstood_intent" for response in self.command_responses)

    def to_mcp_result(self) -> MCPToolResult:
        """Convert CommandOutput to MCP-compliant format"""
        content = []
        content.extend(
            MCPContent(type="text", text=response.response)
            for response in self.command_responses
        )
        return MCPToolResult(
            content=content,
            isError=not self.success
        )

class ModuleType(Enum):
    """Specifies which part of a command's implementation to load."""
    INPUT_FOR_PARAM_EXTRACTION_CLASS = 0
    COMMAND_PARAMETERS_CLASS = 1
    RESPONSE_GENERATION_INFERENCE = 2
    CONTEXT_CLASS = 3

_env_vars: dict = {}
CommandContextModel = None
CommandRoutingRegistry = None
UtteranceRegistry = None
ModelPipelineRegistry=None


def init(env_vars: dict):
    global _env_vars, CommandContextModel, CommandRoutingRegistry, UtteranceRegistry, ModelPipelineRegistry, Session, WorkflowSession
    _env_vars = env_vars

    # init before importing other modules so env vars are available
    from .command_context_model import CommandContextModel as CommandContextModelClass
    from .command_routing_definition import CommandRoutingRegistry as CommandRoutingRegistryClass
    from .utterance_definition import UtteranceRegistry as UtteranceRegistryClass
    from .model_pipeline_training import ModelPipeline

    # from .session import Session as SessionClass
    # from .workflow_session import WorkflowSession as WorkflowSessionClass

    # Assign to global variables
    CommandContextModel = CommandContextModelClass
    CommandRoutingRegistry = CommandRoutingRegistryClass
    UtteranceRegistry = UtteranceRegistryClass
    ModelPipelineRegistry = ModelPipeline
    # Session = SessionClass
    # WorkflowSession = WorkflowSessionClass

def get_env_var(var_name: str, var_type: type = str, default: Optional[Union[str, int, float, bool]] = None) -> Union[str, int, float, bool]:
    """get the environment variable"""
    global _env_vars

    value = _env_vars.get(var_name)
    if value is None:
        if default is not None:
            return default
        value = os.getenv(var_name)

    if value is None:
        from fastworkflow.utils.logging import logger
        logger.warning(f"Environment variable '{var_name}' does not exist and no default value is provided.")

    try:
        if value is None:
            return None
        if var_type is int:
            return int(value)
        elif var_type is float:
            return float(value)
        elif var_type is bool:
            if value.lower() in ('true', '1'):
                return True
            elif value.lower() in ('false', '0'):
                return False
            else:
                raise ValueError(f"Cannot convert '{value}' to {var_type.__name__}.")
        return str(value)  # Default case for str
    except ValueError as e:
        raise ValueError(f"Cannot convert '{value}' to {var_type.__name__}.") from e

def get_fastworkflow_package_path() -> str:
    """Get the fastworkflow package directory.
    
    This works both in development (when working in the fastworkflow repo)
    and when fastworkflow is pip installed.
    
    Returns:
        str: Path to the fastworkflow package directory
    """
    return os.path.dirname(os.path.abspath(__file__))

def get_internal_workflow_path(workflow_name: str) -> str:
    """Get the path to an internal fastworkflow workflow.
    
    Args:
        workflow_name: Name of the workflow in the _workflows directory
        
    Returns:
        str: Full path to the internal workflow
    """
    return os.path.join(get_fastworkflow_package_path(), "_workflows", workflow_name)

def get_session_id(session_id_str: str) -> int:
    return int(mmh3.hash(session_id_str))

from .session import Session as Session
from .workflow_session import WorkflowSession as WorkflowSession
