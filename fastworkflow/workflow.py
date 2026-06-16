import os
import sys
import threading
from functools import wraps
from typing import Optional

from speedict import Rdict

import fastworkflow
from fastworkflow.utils.logging import logger


# ----------------------------------------------------------------------
# Process-global, in-memory session-state store.
#
# This replaces the two per-session speedict (RocksDB) backends that used to
# live on the hot path: the per-workflow snapshot DB and the global
# ``workflowid_2_sessiondata_map``.  ``Workflow.create`` previously paid ~6
# RocksDB open/close cycles per new session (snapshot + map, x2 for the cme and
# app workflows) even though the payload is trivial and all heavy definitions
# are already cached per folderpath (RoutingRegistry / command_directory).
#
# Session state is mutable but cheap and inherently per-process, so it is held
# in memory here.  Durable cross-process state for the FastAPI service is owned
# separately by SessionStateStore + ConversationStore; the CLI no longer
# resumes workflow context across process restarts (accepted trade-off).
#
# speedict is still used elsewhere (the enablecache decorator below,
# ConversationStore, and the NLU clarification cache) and is intentionally
# left in place there.
# ----------------------------------------------------------------------
_STATE_LOCK = threading.RLock()
# workflow_id -> snapshot dict (the former per-session snapshot DB)
_WORKFLOW_SNAPSHOTS: dict[int, dict] = {}
# workflow_id -> {"children": list[int]} (the former workflowid_2_sessiondata map)
_WORKFLOW_SESSIONDATA: dict[int, dict] = {}


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

        with _STATE_LOCK:
            _WORKFLOW_SESSIONDATA[workflow.id] = {"children": []}
            if workflow.parent_id:
                parent_session_data = _WORKFLOW_SESSIONDATA.get(workflow.parent_id)
                if parent_session_data is not None:
                    sibling_list = parent_session_data["children"]
                    if workflow.id not in sibling_list:
                        sibling_list.append(workflow.id)

        return workflow

    @classmethod
    def get_workflow(cls, workflow_id: int) -> Optional["Workflow"]:
        """load the workflow from the in-memory session-state store"""
        with _STATE_LOCK:
            snapshot = _WORKFLOW_SNAPSHOTS.get(workflow_id)
            if snapshot is None:
                return None
            # Shallow-copy the snapshot wrapper so the reconstructed Workflow
            # does not mutate the stored dict structure as it re-saves.
            workflow_snapshot = dict(snapshot)

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
        # important to clear the current command from the workflow context
        if "command" in self._context:
            del self._context["command"]

        # important to clear parameter extraction error state (if any)
        if "stored_parameters" in self._context:
            del self._context["stored_parameters"]

        self._context["NLU_Pipeline_Stage"] = fastworkflow.NLUPipelineStage.INTENT_DETECTION

        self._mark_dirty()

    def close(self) -> bool:
        """close the session and drop its (and its descendants') in-memory state"""
        if self.parent_id:
            raise ValueError("close should only be called on the root session")

        with _STATE_LOCK:
            root_session_data = _WORKFLOW_SESSIONDATA.get(self.id)
            children = list(root_session_data["children"]) if root_session_data else []

            # collect all descendants
            descendant_list = []
            to_process = children
            while to_process:
                current_id = to_process.pop()
                child_session_data = Workflow._get_session_data(current_id)
                descendant_list.append(current_id)
                if child_session_data[3] in sys.path:
                    sys.path.remove(child_session_data[3])  # remove the workflow folderpath from sys.path
                # Add any children to the processing queue
                to_process.extend(child_session_data[2])

            # process all descendants
            for descendant_workflow_id in descendant_list:
                _WORKFLOW_SESSIONDATA.pop(descendant_workflow_id, None)
                _WORKFLOW_SNAPSHOTS.pop(descendant_workflow_id, None)

            if self._folderpath in sys.path:
                sys.path.remove(self._folderpath)
            # remove ourselves from the in-memory session-state store
            _WORKFLOW_SESSIONDATA.pop(self.id, None)
            _WORKFLOW_SNAPSHOTS.pop(self.id, None)

        return True

    @classmethod
    def _get_session_data(cls, workflow_id: int) -> tuple[int, Optional[str], list, str]:
        """get the parent id, (legacy, now None) session db folder path, the children list, and the workflow folderpath"""
        with _STATE_LOCK:
            sessiondata_dict = _WORKFLOW_SESSIONDATA.get(workflow_id)
            snapshot = _WORKFLOW_SNAPSHOTS.get(workflow_id)

        if not sessiondata_dict:
            raise ValueError(f"Workflow {workflow_id} not found")

        children_list = sessiondata_dict["children"]
        if children_list is None:
            raise ValueError(f"Workflow {workflow_id} must have a children list even if it is empty")

        if snapshot is None:
            raise ValueError(f"Workflow {workflow_id} snapshot not found")

        return (
            snapshot["parent_workflow_id"],
            None,
            children_list,
            snapshot["workflow_folderpath"]
        )

    def get_cachedb_folderpath(self, function_name: str) -> str:
        """Get the cache database folder path for a specific function"""
        speedict_foldername = fastworkflow.get_env_var("SPEEDDICT_FOLDERNAME")
        return os.path.join(
            self._folderpath,
            speedict_foldername,
            f"function_cache/{function_name}",
        )

    @classmethod
    def _load(cls, workflow_id: int) -> dict[str, str|int|bool]:
        """load the workflow snapshot from the in-memory session-state store"""
        with _STATE_LOCK:
            snapshot = _WORKFLOW_SNAPSHOTS.get(workflow_id)
            if snapshot is None:
                raise ValueError(
                    f"Workflow {workflow_id} not found in session-state store"
                )
            return dict(snapshot)

    def _save(self) -> None:
        """save the workflow snapshot into the in-memory session-state store"""
        with _STATE_LOCK:
            _WORKFLOW_SNAPSHOTS[self._id] = self._to_dict()

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
