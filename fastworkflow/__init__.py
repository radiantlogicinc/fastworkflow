import contextlib
from dataclasses import dataclass
from enum import Enum
import os
import time
from typing import Any, Optional, Union

from pydantic import BaseModel
import mmh3


class NLUPipelineStage(Enum):
    """Specifies the stages of the NLU Pipeline processing."""
    INTENT_DETECTION = 0
    INTENT_AMBIGUITY_CLARIFICATION = 1
    INTENT_MISUNDERSTANDING_CLARIFICATION = 2
    PARAMETER_EXTRACTION = 3

class Action(BaseModel):
    command_name: str
    command: str = ""   # only use is to display autocomplete item in the UI
    parameters: dict[str, Optional[Union[str, bool, int, float, BaseModel]]] = {}
    workflow_id: Optional[int] = None    # when creating a new action, this is set by the workflow

class Recommendation(BaseModel):
    summary: str
    suggested_actions: list[Action] = []

class CommandResponse(BaseModel):
    response: str
    success: bool = True
    artifacts: dict[str, Any] = {}
    next_actions: list[Action] = []
    recommendations: list[Recommendation] = []

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

class CommandTraceEventDirection(str, Enum):
    AGENT_TO_WORKFLOW = "agent_to_workflow"
    WORKFLOW_TO_AGENT = "workflow_to_agent"

@dataclass
class CommandTraceEvent:
    direction: CommandTraceEventDirection
    raw_command: str | None               # for AGENT_TO_WORKFLOW
    command_name: str | None              # for WORKFLOW_TO_AGENT
    parameters: dict | str | None
    response_text: str | None
    success: bool | None
    timestamp_ms: int

class CommandOutput(BaseModel):
    command_responses: list[CommandResponse]
    workflow_name: str = ""
    context: str = ""
    command_name: str = ""
    command_parameters: str = ""

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

_chat_session: Optional["ChatSession"] = None
class ChatSessionDescriptor:
    """Descriptor for accessing the global chat session.""" 
    def __get__(self, obj, objtype = None):
        global _chat_session
        return _chat_session
    
    def __set__(self, obj, value):
        global _chat_session
        if _chat_session:
            raise RuntimeError("Cannot set chat session. It is already set.")
        _chat_session = value
# Create the descriptor instance
chat_session = ChatSessionDescriptor()


_env_vars: dict = {}
CommandContextModel = None
RoutingDefinition = None
RoutingRegistry = None
ModelPipelineRegistry=None


def init(env_vars: dict):
    global _env_vars, CommandContextModel, RoutingDefinition, RoutingRegistry, ModelPipelineRegistry
    _env_vars = env_vars

    # init before importing other modules so env vars are available
    from .command_context_model import CommandContextModel as CommandContextModelClass
    from .command_routing import RoutingDefinition as RoutingDefinitionClass
    from .command_routing import RoutingRegistry as RoutingRegistryClass
    from .model_pipeline_training import ModelPipeline

    # Assign to global variables
    CommandContextModel = CommandContextModelClass
    RoutingDefinition = RoutingDefinitionClass
    RoutingRegistry = RoutingRegistryClass
    ModelPipelineRegistry = ModelPipeline

    # Ensure DSPy logging is properly configured after all imports
    # This needs to happen after DSPy is imported by other modules
    import logging
    logging.getLogger("dspy").setLevel(logging.ERROR)
    logging.getLogger("dspy.adapters.json_adapter").setLevel(logging.ERROR)

    # ------------------------------------------------------------
    # Eager imports for heavy libraries that otherwise trigger lock
    # contention during the first wildcard command.  Importing them
    # once here (at server start-up) shifts the cost out of the
    # request path and takes advantage of Python's module cache.
    # ------------------------------------------------------------
    with contextlib.suppress(Exception):
        import datasets  # noqa: F401 – pre-load Hugging Face datasets

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

def get_workflow_id(workflow_id_str: str) -> int:
    return int(mmh3.hash(workflow_id_str))

from .workflow import Workflow as Workflow
from .chat_session import ChatSession as ChatSession
