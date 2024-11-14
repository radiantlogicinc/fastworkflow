import os
import shutil
import sys
from functools import wraps
from typing import Optional, Union

from speedict import Rdict

from semantic_router.encoders import HuggingFaceEncoder
from semantic_router import RouteLayer

from fastworkflow.command_routing_definition import CommandRoutingDefinition
from fastworkflow.utils.logging import logger
from fastworkflow.utterance_definition import UtteranceDefinition
from fastworkflow.workflow import Workflow, Workitem
from fastworkflow.workflow_definition import WorkflowDefinition


# implements the enablecache decorator
def enablecache(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Create a cache key based on the function arguments
        key = str(args) + str(kwargs)

        # Get the cache database
        cache_db_path = self.get_cachedb_folderpath(func.__name__)
        cache_db = Rdict(cache_db_path)

        if key not in cache_db:
            # If the result is not in the cache, call the function and store the result
            result = func(self, *args, **kwargs)
            cache_db[key] = result
        else:
            result = cache_db[key]

        cache_db.close()
        return result

    return wrapper


SPEEDDICT_FOLDERNAME = "___workflow_contexts"


class Session:
    """Session class"""

    def __init__(self, 
                 session_id: int, 
                 workflow_folderpath: str, 
                 env_vars: dict = {}, 
                 context: dict = {}, 
                 caller_session: Optional["Session"] = None,
                 for_training_semantic_router: bool = False):
        """initialize the Session class"""
        if not os.path.exists(workflow_folderpath):
            raise ValueError(f"The folder path {workflow_folderpath} does not exist")

        if not os.path.isdir(workflow_folderpath):
            raise ValueError(f"{workflow_folderpath} must be a directory")

        # THIS IS IMPORTANT: it allows relative import of modules in the code inside workflow_folderpath
        sys.path.insert(0, workflow_folderpath)

        speedict_folderpath = os.path.join(workflow_folderpath, SPEEDDICT_FOLDERNAME)
        os.makedirs(speedict_folderpath, exist_ok=True)

        root_workitem_type = os.path.basename(workflow_folderpath.rstrip("/"))

        self._session_id = session_id
        self._workflow_folderpath = workflow_folderpath
        self._env_vars = env_vars
        self._caller_session = caller_session

        self._root_workitem_type = root_workitem_type
        self._workflow_definition = WorkflowDefinition.create(workflow_folderpath)
        self._command_routing_definition = CommandRoutingDefinition.create(
            workflow_folderpath
        )
        self._utterance_definition = UtteranceDefinition.create(workflow_folderpath)

        self._map_workitem_type_2_route_layer = None
        if not for_training_semantic_router:
            route_layers_folderpath = os.path.join(self._workflow_folderpath, "___route_layers")
            if not os.path.exists(route_layers_folderpath):
                raise ValueError(f"Train the semantic router first. Before running the workflow.")
            
            # importing here to avoid circular import
            from fastworkflow.semantic_router_definition import SemanticRouterDefinition

            encoder = HuggingFaceEncoder()
            semantic_router = SemanticRouterDefinition(encoder, self.workflow_folderpath)
            map_workitem_type_2_route_layer: dict[str, RouteLayer] = {}
            for workitem_type in self._workflow_definition.types:
                route_layer = semantic_router.get_route_layer(workitem_type)
                map_workitem_type_2_route_layer[workitem_type] = route_layer
            self._map_workitem_type_2_route_layer = map_workitem_type_2_route_layer

        self._workflow = Workflow(
            workflow_definition=self._workflow_definition,
            type=root_workitem_type,
            parent_workflow=None,
        )

        context_dict = self.get_context()
        if context:
            # if there are duplicates, the values already in context_dict
            # will override the values in provided context
            context.update(context_dict)
            context_dict = context
        self.set_context(context_dict)

    @property
    def id(self) -> int:
        """get the session id"""
        return self._session_id

    @property
    def env_vars(self) -> dict:
        """get the environment variables"""
        return self._env_vars

    @property
    def workflow_folderpath(self) -> str:
        """get the workflow folderpath"""
        return self._workflow_folderpath

    @property
    def caller_session(self) -> Optional["Session"]:
        """get the caller session"""
        return self._caller_session

    @property
    def root_workitem_type(self) -> str:
        """get the root workitem type"""
        return self._root_workitem_type

    @property
    def workflow_definition(self) -> WorkflowDefinition:
        """get the workflow definition"""
        return self._workflow_definition.model_copy()

    @property
    def command_routing_definition(self) -> CommandRoutingDefinition:
        """get the command routing definition"""
        return self._command_routing_definition.model_copy()

    @property
    def utterance_definition(self) -> UtteranceDefinition:
        """get the utterance definition"""
        return self._utterance_definition.model_copy()

    @property
    def workflow(self) -> Workflow:
        """get the workflow"""
        return self._workflow

    @property
    def parameter_extraction_info(self) -> Optional[dict]:
        """get the parameter extraction information"""
        return self._parameter_extraction_info

    @parameter_extraction_info.setter
    def parameter_extraction_info(self, value: Optional[dict]) -> None:
        """set the parameter extraction information"""
        self._parameter_extraction_info = value

    def get_env_var(self, var_name: str, var_type: type = str, default: Optional[Union[str, int, float, bool]] = None) -> Union[str, int, float, bool]:
        """get the environment variable"""
        value = self._env_vars.get(var_name)
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

    def get_active_workitem(self) -> Optional[Union[Workitem, Workflow]]:
        """get the active workitem"""
        context = self.get_context()
        active_workitem_path = context["active_workitem_path"]
        active_workitem_id = context["active_workitem_id"]

        return self.workflow.find_workitem(active_workitem_path, active_workitem_id)

    def set_active_workitem(self, workitem: Union[Workitem, Workflow]) -> None:
        """set the active workitem"""
        context = self.get_context()
        context["active_workitem_path"] = workitem.path
        context["active_workitem_id"] = workitem.id
        self.set_context(context)

    def get_context(self) -> dict:
        """get the context"""
        contextdb_folderpath = self.get_contextdb_folderpath(self.id)
        keyvalue_db = Rdict(contextdb_folderpath)

        context = keyvalue_db["context"] if "context" in keyvalue_db else {}
        if not context:
            context = {
                "active_workitem_path": "/",
                "active_workitem_id": None
            }
            keyvalue_db["context"] = context

        keyvalue_db.close()
        return context

    def set_context(self, context: dict) -> None:
        """set the context"""
        contextdb_folderpath = self.get_contextdb_folderpath(self.id)
        keyvalue_db = Rdict(contextdb_folderpath)
        keyvalue_db["context"] = context
        keyvalue_db.close()

    def reset_session(self) -> None:
        """reset the session"""
        self.set_context({})

    def close(self) -> bool:
        """close the session"""
        contextdb_folderpath = self.get_contextdb_folderpath(self.id)
        try:
            shutil.rmtree(contextdb_folderpath, ignore_errors=True)
        except OSError as e:
            logger.error(f"Error closing session: {e}")
            return False

        sys.path.remove(self._workflow_folderpath)
        return True

    def get_contextdb_folderpath(self, session_id: int) -> str:
        """get the db folder path"""
        session_id_str = str(session_id).replace("-", "_")
        return os.path.join(
            self._workflow_folderpath, SPEEDDICT_FOLDERNAME, session_id_str
        )

    def get_cachedb_folderpath(self, function_name: str) -> str:
        """Get the cache database folder path for a specific function"""
        return os.path.join(
            self._workflow_folderpath,
            SPEEDDICT_FOLDERNAME,
            f"/function_cache/{function_name}",
        )

    def get_route_layer(self, workitem_type: str) -> RouteLayer:
        """get the route layer for a given workitem type"""
        return self._map_workitem_type_2_route_layer[workitem_type]
