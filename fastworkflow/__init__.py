from typing import Any, Optional, Union
from enum import Enum

from pydantic import BaseModel


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

class CommandSource(str, Enum):
    BASE_COMMANDS = ("_base_commands",)
    COMMANDS = "_commands"

def init(env_vars: dict):
    global _env_vars
    _env_vars = env_vars

def get_env_var(var_name: str, var_type: type = str, default: Optional[Union[str, int, float, bool]] = None) -> Union[str, int, float, bool]:
    """get the environment variable"""
    global _env_vars

    value = _env_vars.get(var_name)
    if value is None:
        if default is None:
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

_env_vars: dict = {}


from .workflow_definition import WorkflowRegistry
from .command_routing_definition import CommandRoutingRegistry
from .utterance_definition import UtteranceRegistry
from .semantic_router_definition import RouteLayerRegistry
from .session import Session
from .workflow_session import WorkflowSession
