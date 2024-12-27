import os
from typing import Any, Optional, Union
from enum import Enum

from pydantic import BaseModel
import murmurhash


class Action(BaseModel):
    workitem_type: str
    command_name: str
    command: str = ""   # only use is to display autocomplete item in the UI
    parameters: dict[str, Optional[Union[str, bool, int, float, BaseModel]]] = {}
    session_id: Optional[int] = None    # when creating a new action, this is set by the workflow session

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

class CommandSource(str, Enum):
    BASE_COMMANDS = ("_base_commands",)
    COMMANDS = "_commands"


_env_vars: dict = {}
WorkflowRegistry = None
CommandRoutingRegistry = None
UtteranceRegistry = None
RouteLayerRegistry = None

def init(env_vars: dict):
    global _env_vars, Session, WorkflowRegistry, CommandRoutingRegistry, UtteranceRegistry, RouteLayerRegistry, WorkflowSession
    _env_vars = env_vars

    # init before importing other modules so env vars are available
    from .workflow_definition import WorkflowRegistry as WorkflowRegistryClass
    from .command_routing_definition import CommandRoutingRegistry as CommandRoutingRegistryClass
    from .utterance_definition import UtteranceRegistry as UtteranceRegistryClass
    from .semantic_router_definition import RouteLayerRegistry as RouteLayerRegistryClass

    # Assign to global variables
    WorkflowRegistry = WorkflowRegistryClass
    CommandRoutingRegistry = CommandRoutingRegistryClass
    UtteranceRegistry = UtteranceRegistryClass
    RouteLayerRegistry = RouteLayerRegistryClass

def get_env_var(var_name: str, var_type: type = str, default: Optional[Union[str, int, float, bool]] = None) -> Union[str, int, float, bool]:
    """get the environment variable"""
    global _env_vars

    value = _env_vars.get(var_name)
    if value is None:
        if default is None:           
            value = os.getenv(var_name)
            if value is None:
                raise ValueError(f"Environment variable '{var_name}' does not exist and no default value is provided.")
        else:
            return default
    
    try:
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
    except ValueError:
        raise ValueError(f"Cannot convert '{value}' to {var_type.__name__}.")

def get_session_id(session_id_str: str) -> int:
    return int(murmurhash.hash(session_id_str))

from .session import Session
from .workflow_session import WorkflowSession
