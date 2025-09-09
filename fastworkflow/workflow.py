import os
import shutil
import sys
from functools import wraps
from typing import Optional

from speedict import Rdict

import fastworkflow
from fastworkflow.utils.logging import logger


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

class Workflow:
    """Workflow class"""
    @classmethod
    def create(
        cls, 
        workflow_folderpath: str, 
        workflow_id_str: Optional[str] = None, 
        parent_workflow_id: Optional[int] = None, 
        workflow_context: dict = None,
        project_folderpath: Optional[str] = None
    ) -> "Workflow":
        if workflow_id_str is not None and parent_workflow_id is not None:
            raise ValueError("workflow_id_str and parent_workflow_id cannot both be provided")

        if not os.path.exists(workflow_folderpath):
            raise ValueError(f"The folder path {workflow_folderpath} does not exist")

        if not os.path.isdir(workflow_folderpath):
            raise ValueError(f"{workflow_folderpath} must be a directory")

        if workflow_id_str:
            workflow_id = fastworkflow.get_workflow_id(workflow_id_str)
        else:
            workflow_id = cls.generate_child_workflow_id(workflow_folderpath, parent_workflow_id)

        python_project_path = project_folderpath or workflow_folderpath
        if python_project_path not in sys.path:
            # THIS IS IMPORTANT: it allows relative import of modules in the code inside workflow_folderpath
            sys.path.insert(0, python_project_path)

        if workflow := cls.get_workflow(workflow_id):
            if workflow_context is not None:
                workflow.context = workflow_context
            return workflow

        # Resolve the workflow path to ensure consistent cache keys
        from pathlib import Path
        resolved_workflow_path = str(Path(workflow_folderpath).resolve())
        
        workflow_snapshot = {
            "workflow_id": workflow_id,
            "workflow_folderpath": resolved_workflow_path,
            "workflow_context": workflow_context or {},
            "parent_workflow_id": parent_workflow_id,
            "is_complete": False
        }
        workflow = Workflow(cls.__create_key, workflow_snapshot)

        session_db_folderpath = Workflow._get_sessiondb_folderpath(
            workflow_id=workflow_id,
            parent_workflow_id=parent_workflow_id,
            workflow_folderpath=workflow_folderpath
        )

        workflowid_2_sessiondata_mapdir = Workflow._get_workflow_id_2_sessiondata_mapdir()
        map_workflowid_2_session_db = Rdict(workflowid_2_sessiondata_mapdir)       

        map_workflowid_2_session_db[workflow.id] = {
            "sessiondb_folderpath": session_db_folderpath,
            "children": []
        }
        if workflow.parent_id:
            parent_session_data = map_workflowid_2_session_db[workflow.parent_id]
            sibling_list = parent_session_data["children"]
            if workflow.id not in sibling_list:
                sibling_list.append(workflow.id)
                parent_session_data["children"] = sibling_list
                map_workflowid_2_session_db[workflow.parent_id] = parent_session_data

        map_workflowid_2_session_db.close()
        return workflow

    @classmethod
    def get_workflow(cls, workflow_id: int) -> Optional["Workflow"]:
        """load the workflow"""
        workflowid_2_sessiondata_mapdir = Workflow._get_workflow_id_2_sessiondata_mapdir()
        map_workflowid_2_session_db = Rdict(workflowid_2_sessiondata_mapdir)
        sessiondata_dict = map_workflowid_2_session_db.get(workflow_id, None)
        map_workflowid_2_session_db.close()

        if not sessiondata_dict:
            return None

        # Gracefully handle stale session entries where the underlying
        # session database folder has been deleted (e.g. between test
        # runs).  Treat such cases as "workflow not found" so that the
        # caller can create a fresh workflow instead of raising.
        sessiondb_folderpath = sessiondata_dict["sessiondb_folderpath"]
        if not os.path.exists(sessiondb_folderpath):
            # Stale entry – pretend it does not exist
            return None

        workflow_snapshot = Workflow._load(sessiondb_folderpath)

        return Workflow(
            cls.__create_key,
            workflow_snapshot,
        )

    @classmethod
    def generate_child_workflow_id(cls, workflow_folderpath: str, parent_workflow_id: Optional[int] = None) -> int:
        """generate a child workflow id"""
        workflow_type = os.path.basename(workflow_folderpath).rstrip("/")
        workflow_id_str = f"{parent_workflow_id}{workflow_type}" \
                         if parent_workflow_id \
                         else f"{workflow_type}"

        return fastworkflow.get_workflow_id(workflow_id_str)

    # enforce workflow creation exclusively using Workflow.create
    # https://stackoverflow.com/questions/8212053/private-constructor-in-python
    __create_key = object()
   
    def __init__(self, create_key, workflow_snapshot: dict[str, str|int|bool]):
        """initialize the Workflow class"""
        if create_key is not Workflow.__create_key:
            raise ValueError("Workflow objects must be created using Workflow.create")

        from pathlib import Path
        self._id = workflow_snapshot["workflow_id"]
        # Always resolve the workflow path to ensure consistent cache keys
        self._folderpath = str(Path(workflow_snapshot["workflow_folderpath"]).resolve())
        self._parent_id = workflow_snapshot.get("parent_workflow_id")
        self._is_complete = workflow_snapshot.get("is_complete", False)
        self._context = workflow_snapshot.get("workflow_context", {}) 

        self._root_command_context = None
        self._current_command_context = None
        self._command_context_for_response_generation = None

        # ------------------------------------------------------------------
        # Persistence control
        # ------------------------------------------------------------------
        # ``_dirty`` tracks whether state has changed since the last flush.
        # We persist the freshly-constructed workflow immediately so that it
        # exists on disk, then mark it clean.
        self._dirty: bool = False
        self._save()
        self._dirty = False

    @property
    def current_command_context(self) -> object:
        return self._current_command_context

    @property
    def current_command_context_name(self) -> str:
        return Workflow.get_command_context_name(self._current_command_context)

    @property
    def current_command_context_displayname(self) -> str:
        crd = fastworkflow.CommandContextModel.load(self._folderpath)
        context_class = crd.get_context_class(
                Workflow.get_command_context_name(self._current_command_context),
                fastworkflow.ModuleType.CONTEXT_CLASS
        )
        if context_class and hasattr(context_class, 'get_displayname'):
            return context_class.get_displayname(self._current_command_context)

        return Workflow.get_command_context_name(self._current_command_context, for_display=True)


    @property
    def is_current_command_context_root(self) -> bool:
        return self._current_command_context == self._root_command_context

    @current_command_context.setter
    def current_command_context(self, value: Optional[object]) -> None:
        self._current_command_context = value

    @property
    def root_command_context(self) -> object:
        return self._root_command_context

    @root_command_context.setter
    def root_command_context(self, value: Optional[object]) -> None:
        if self._root_command_context:
            raise ValueError("Root command context can only be set once per Workflow")

        self._root_command_context = value
        self._current_command_context = value

    def get_parent(self, command_context_object: Optional[object] = None) -> Optional[object]:
        if command_context_object == self._root_command_context or command_context_object is None:
            return self._root_command_context

        crd = fastworkflow.CommandContextModel.load(
            self._folderpath)
        context_class = crd.get_context_class(
                Workflow.get_command_context_name(command_context_object),
                fastworkflow.ModuleType.CONTEXT_CLASS
        )
        if context_class:
            command_context_object = context_class.get_parent(command_context_object)
        else:
            command_context_object = None

        return command_context_object

    @property
    def command_context_for_response_generation(self) -> object:
        return self._command_context_for_response_generation

    @command_context_for_response_generation.setter
    def command_context_for_response_generation(self, value: Optional[object]) -> None:
        self._command_context_for_response_generation = value

    @property
    def is_command_context_for_response_generation_root(self) -> bool:
        return self._command_context_for_response_generation == self._root_command_context

    @staticmethod
    def get_command_context_name(command_context_object: Optional[object], for_display = False) -> str:
        if command_context_object:
            return command_context_object.__class__.__name__
        return 'global' if for_display else '*'

    @property
    def id(self) -> int:
        """get the workflow id"""
        return self._id
    
    @property
    def parent_id(self) -> Optional[int]:
        """get the parent workflow id"""
        return self._parent_id

    @property
    def folderpath(self) -> str:
        return self._folderpath

    @folderpath.setter
    def folderpath(self, value: str) -> None:
        self._folderpath = value
        self._mark_dirty()

    @property
    def context(self) -> dict:
        return self._context
    
    @context.setter
    def context(self, value: dict) -> None:
        self._context = value
        self._mark_dirty()

    @property
    def is_complete(self) -> bool:
        return self._is_complete
    
    @is_complete.setter
    def is_complete(self, value: bool) -> None:
        self._is_complete = value
        self._mark_dirty()
    
    def end_command_processing(self) -> None:
        """Process the end of a command"""
        mark_dirty = False
        # important to clear the current command from the workflow context
        if "command" in self._context:
            del self._context["command"]
            mark_dirty = True

        # important to clear parameter extraction error state (if any)
        if "stored_parameters" in self._context:
            del self._context["stored_parameters"]
            mark_dirty = True

        self._context["NLU_Pipeline_Stage"] = fastworkflow.NLUPipelineStage.INTENT_DETECTION

        if mark_dirty:
            self._mark_dirty()

    def close(self) -> bool:
        """close the session"""
        if self.parent_id:
            raise ValueError("close should only be called on the root session")

        workflowid_2_sessiondata_mapdir = Workflow._get_workflow_id_2_sessiondata_mapdir()
        map_workflowid_2_session_db = Rdict(workflowid_2_sessiondata_mapdir)

        # collect all descendants
        descendant_list = []
        to_process = map_workflowid_2_session_db[self.id]["children"][:]  # Create a shallow copy
        while to_process:
            current_id = to_process.pop()
            child_session_data = Workflow._get_session_data(
                current_id,
                map_workflowid_2_session_db=map_workflowid_2_session_db
            )
            descendant_list.append(current_id)
            if child_session_data[3] in sys.path:
                sys.path.remove(child_session_data[3])  # remove the workflow folderpath from sys.path
            # Add any children to the processing queue
            to_process.extend(child_session_data[2])

        # process all descendants
        for descendant_workflow_id in descendant_list:
            del map_workflowid_2_session_db[descendant_workflow_id]

        sys.path.remove(self._folderpath)
        # remove ourselves from the workflowid_2_sessiondata_map
        del map_workflowid_2_session_db[self.id]

        map_workflowid_2_session_db.close()

        try:
            sessiondb_folderpath = Workflow._get_sessiondb_folderpath(
                workflow_id=self._id,
                parent_workflow_id=self._parent_id,
                workflow_folderpath=self._folderpath
            )
            shutil.rmtree(sessiondb_folderpath, ignore_errors=True)
        except OSError as e:
            logger.error(f"Error closing session: {e}", stack_info=True)
            return False

        return True

    @classmethod
    def _get_sessiondb_folderpath(
        cls, 
        workflow_id: int, 
        parent_workflow_id: Optional[int] = None,
        workflow_folderpath: Optional[str] = None
    ) -> str:
        """get the db folder path"""
        if parent_workflow_id is None and workflow_folderpath is None:
            raise ValueError("parent_workflow_id or workflow_folderpath must be provided")

        if parent_workflow_id:
            parent_session_folder = ""

            workflowid_2_sessiondata_mapdir = Workflow._get_workflow_id_2_sessiondata_mapdir()
            map_workflowid_2_session_db = Rdict(workflowid_2_sessiondata_mapdir)       
            
            while parent_workflow_id:
                parent_session_data = Workflow._get_session_data(
                    parent_workflow_id,
                    map_workflowid_2_session_db=map_workflowid_2_session_db
                )
                parent_workflow_id = parent_session_data[0]
                parent_session_folder = os.path.join(parent_session_data[1], parent_session_folder)
            
            map_workflowid_2_session_db.close()
        else:
            speedict_foldername = fastworkflow.get_env_var("SPEEDDICT_FOLDERNAME")
            parent_session_folder = os.path.join(
                workflow_folderpath, 
                speedict_foldername
            )
    
        workflow_id_str = str(workflow_id).replace("-", "_")
        return os.path.join(parent_session_folder, workflow_id_str)

    @classmethod
    def _get_session_data(cls, workflow_id: int, map_workflowid_2_session_db: Rdict) -> tuple[int, str, list, str]:
        """get the parent id, the session db folder path, the children list, and the workflow folderpath"""
        sessiondata_dict = map_workflowid_2_session_db.get(workflow_id, None)

        if not sessiondata_dict:
            raise ValueError(f"Workflow {workflow_id} not found")

        sessiondb_folderpath = sessiondata_dict["sessiondb_folderpath"]
        if not os.path.exists(sessiondb_folderpath):
            raise ValueError(f"Workflow database folder path {sessiondb_folderpath} does not exist")

        children_list = sessiondata_dict["children"]
        if children_list is None:
            raise ValueError(f"Workflow {workflow_id} must have a children list even if it is empty")

        workflow_snapshot = Workflow._load(sessiondb_folderpath)
        return (
            workflow_snapshot["parent_workflow_id"],
            sessiondb_folderpath,
            children_list,
            workflow_snapshot["workflow_folderpath"]
        )

    @classmethod
    def _get_workflow_id_2_sessiondata_mapdir(cls) -> str:
        """get the workflowid_2_sessiondata_map folder path"""
        speedict_foldername = fastworkflow.get_env_var("SPEEDDICT_FOLDERNAME")
        workflowid_2_sessiondata_mapdir = os.path.join(
            speedict_foldername,
            "workflowid_2_sessiondata_map"
        )
        os.makedirs(workflowid_2_sessiondata_mapdir, exist_ok=True)
        return workflowid_2_sessiondata_mapdir

    def get_cachedb_folderpath(self, function_name: str) -> str:
        """Get the cache database folder path for a specific function"""
        speedict_foldername = fastworkflow.get_env_var("SPEEDDICT_FOLDERNAME")
        return os.path.join(
            self._folderpath,
            speedict_foldername,
            f"function_cache/{function_name}",
        )

    @classmethod
    def _load(cls, sessiondb_folderpath: str) -> dict[str, str|int|bool]:
        """load the workflow snapshot (as plain dict to avoid pickle)"""
        keyvalue_db = Rdict(sessiondb_folderpath)
        workflow_snapshot = {
            
            "workflow_id": keyvalue_db["workflow_id"],
            "workflow_folderpath": keyvalue_db["workflow_folderpath"],
            "parent_workflow_id": keyvalue_db["parent_workflow_id"],
            "is_complete": keyvalue_db["is_complete"],
            "workflow_context": keyvalue_db["workflow_context"]
        }
        keyvalue_db.close()

        return workflow_snapshot

    def _save(self) -> None:
        """save the workflow snapshot (as plain dict to avoid pickle)"""
        sessiondb_folderpath = Workflow._get_sessiondb_folderpath(
            workflow_id=self._id,
            parent_workflow_id=self._parent_id,
            workflow_folderpath=self._folderpath
        )
        os.makedirs(sessiondb_folderpath, exist_ok=True)     

        keyvalue_db = Rdict(sessiondb_folderpath)
        for k, v in self._to_dict().items():
            keyvalue_db[k] = v
        keyvalue_db.close()

    def _to_dict(self) -> dict[str, str|int|bool]:
        """Return a JSON-serialisable representation of the workflow."""
        return {
            "workflow_id": self._id,
            "workflow_folderpath": self._folderpath,
            "parent_workflow_id": self._parent_id,
            "is_complete": self._is_complete,
            "workflow_context": self._context
        }

    # ------------------------------------------------------------------
    # Deferred-save helpers
    # ------------------------------------------------------------------

    def _mark_dirty(self) -> None:
        """Flag that the workflow state has changed and needs persistence."""
        self._dirty = True

    def flush(self) -> None:
        """Write pending state changes to disk if necessary."""
        if self._dirty:
            self._save()
            self._dirty = False
