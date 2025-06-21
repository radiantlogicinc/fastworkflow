import os
import shutil
import sys
from functools import wraps
from typing import Any, Optional
from pydantic import BaseModel

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

class WorkflowSnapshot:
    def __init__(self, session_id: int, 
                 workflow_folderpath: str, workflow_context: dict = None, 
                 parent_session_id: Optional[int] = None):
        self._session_id = session_id
        self._workflow_folderpath = workflow_folderpath
        self._workflow_context = {} if workflow_context is None else workflow_context
        self._parent_session_id = parent_session_id
        self._is_complete = False
        self._save()

    @property
    def session_id(self) -> int:
        return self._session_id

    @property
    def workflow_folderpath(self) -> str:
        return self._workflow_folderpath

    @workflow_folderpath.setter
    def workflow_folderpath(self, value: str) -> None:
        self._workflow_folderpath = value
        self._save()

    @property
    def workflow_context(self) -> dict:
        return self._workflow_context
    
    @workflow_context.setter
    def workflow_context(self, value: dict) -> None:
        self._workflow_context = value
        self._save()

    @property
    def parent_session_id(self) -> Optional[int]:
        return self._parent_session_id

    @property
    def is_complete(self) -> bool:
        return self._is_complete
    
    @is_complete.setter
    def is_complete(self, value: bool) -> None:
        self._is_complete = value
        self._save()

    def _save(self) -> None:
        """save the workflow snapshot"""
        sessiondb_folderpath = Session._get_sessiondb_folderpath(
            session_id=self._session_id,
            parent_session_id=self._parent_session_id,
            workflow_folderpath=self._workflow_folderpath
        )
        os.makedirs(sessiondb_folderpath, exist_ok=True)     

        keyvalue_db = Rdict(sessiondb_folderpath)
        keyvalue_db["workflow_snapshot"] = self
        keyvalue_db.close()

class Session:
    """Session class"""
    @classmethod
    def create(cls, workflow_folderpath: str, session_id_str: Optional[str] = None, parent_session_id: Optional[int] = None, context: dict = None, for_training_semantic_router: bool = False) -> "Session":
        if context is None:
            context = {}
        if session_id_str is None and parent_session_id is None:
            raise ValueError("Either session_id_str or parent_session_id must be provided")

        if session_id_str is not None and parent_session_id is not None:
            raise ValueError("session_id_str and parent_session_id cannot both be provided")

        if session_id_str:
            session_id = fastworkflow.get_session_id(session_id_str)
        else:
            session_id = cls._generate_child_session_id(parent_session_id, workflow_folderpath)

        if session := cls.get_session(session_id, context):
            return session

        if not os.path.exists(workflow_folderpath):
            raise ValueError(f"The folder path {workflow_folderpath} does not exist")

        if not os.path.isdir(workflow_folderpath):
            raise ValueError(f"{workflow_folderpath} must be a directory")

        workflow_snapshot = WorkflowSnapshot(
            session_id=session_id,
            workflow_folderpath=workflow_folderpath,
            workflow_context=context,
            parent_session_id=parent_session_id,
        )
        session = Session(cls.__create_key, workflow_snapshot)

        session_db_folderpath = Session._get_sessiondb_folderpath(
            session_id=session_id,
            parent_session_id=parent_session_id,
            workflow_folderpath=workflow_folderpath
        )

        sessionid_2_sessiondata_mapdir = Session._get_session_id_2_sessiondata_mapdir()
        map_sessionid_2_session_db = Rdict(sessionid_2_sessiondata_mapdir)       

        map_sessionid_2_session_db[session.id] = {
            "sessiondb_folderpath": session_db_folderpath,
            "children": []
        }
        if session.parent_id:
            parent_session_data = map_sessionid_2_session_db[session.parent_id]
            sibling_list = parent_session_data["children"]
            if session.id not in sibling_list:
                sibling_list.append(session.id)
                parent_session_data["children"] = sibling_list
                map_sessionid_2_session_db[session.parent_id] = parent_session_data

        map_sessionid_2_session_db.close()
        return session

    @classmethod
    def get_session(cls, 
             session_id: int, 
             context: Optional[dict] = None) -> Optional["Session"]:
        """load the session"""
        sessionid_2_sessiondata_mapdir = Session._get_session_id_2_sessiondata_mapdir()
        map_sessionid_2_session_db = Rdict(sessionid_2_sessiondata_mapdir)
        sessiondata_dict = map_sessionid_2_session_db.get(session_id, None)
        map_sessionid_2_session_db.close()

        if not sessiondata_dict:
            return None

        sessiondb_folderpath = sessiondata_dict["sessiondb_folderpath"]
        if not os.path.exists(sessiondb_folderpath):
            raise ValueError(f"Session database folder path {sessiondb_folderpath} does not exist")

        keyvalue_db = Rdict(sessiondb_folderpath)
        workflow_snapshot: WorkflowSnapshot = keyvalue_db["workflow_snapshot"]
        keyvalue_db.close()

        if context:
            workflow_snapshot.workflow_context = context

        return Session(
            cls.__create_key,
            workflow_snapshot,
        )

    @classmethod
    def _generate_child_session_id(cls, parent_session_id: int, workflow_folderpath: str) -> int:
        """generate a child session id"""
        workflow_type = os.path.basename(workflow_folderpath).rstrip("/")
        session_id_str = f"{parent_session_id}{workflow_type}"
        return fastworkflow.get_session_id(session_id_str)

    # enforce session creation exclusively using Session.create_session
    # https://stackoverflow.com/questions/8212053/private-constructor-in-python
    __create_key = object()
   
    def __init__(self, create_key, workflow_snapshot: WorkflowSnapshot):
        """initialize the Session class"""
        if create_key is not Session.__create_key:
            raise ValueError("Session objects must be created using Session.create")

        workflow_folderpath = workflow_snapshot.workflow_folderpath
        if workflow_folderpath not in sys.path:
            # THIS IS IMPORTANT: it allows relative import of modules in the code inside workflow_folderpath
            sys.path.insert(0, workflow_folderpath)

        self._workflow_snapshot = workflow_snapshot
        self._root_command_context = None
        self._current_command_context = None
        self._command_context_for_response_generation = None

    @property
    def current_command_context(self) -> object:
        return self._current_command_context

    @property
    def current_command_context_name(self) -> str:
        return Session.get_command_context_name(self._current_command_context)

    @property
    def is_current_context_root(self) -> bool:
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
            raise ValueError("Root command context can only be set once per Session")

        self._root_command_context = value
        self._current_command_context = value

    def get_parent(self, command_context_object: Optional[object] = None) -> Optional[object]:
        if command_context_object == self._root_command_context or command_context_object is None:
            return self._root_command_context

        crd = fastworkflow.CommandContextModel.load(
            self._workflow_snapshot.workflow_folderpath)
        context_class = crd.get_context_class(
                Session.get_command_context_name(command_context_object),
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

    @current_command_context.setter
    def command_context_for_response_generation(self, value: Optional[object]) -> None:
        self._command_context_for_response_generation = value

    @property
    def is_command_context_for_response_generation_root(self) -> bool:
        return self._command_context_for_response_generation == self._root_command_context

    @staticmethod
    def get_command_context_name(command_context_object: Optional[object]) -> str:
        if command_context_object:
            return command_context_object.__class__.__name__
        return '*'

    @property
    def id(self) -> int:
        """get the session id"""
        return self._workflow_snapshot._session_id
    
    @property
    def parent_id(self) -> Optional[int]:
        """get the parent session id"""
        return self._workflow_snapshot._parent_session_id

    @property
    def workflow_snapshot(self) -> WorkflowSnapshot:
        """get the workflow snapshot"""
        return self._workflow_snapshot

    def close(self) -> bool:
        """close the session"""
        if self.parent_id:
            raise ValueError("close should only be called on the root session")

        sessionid_2_sessiondata_mapdir = Session._get_session_id_2_sessiondata_mapdir()
        map_sessionid_2_session_db = Rdict(sessionid_2_sessiondata_mapdir)

        # collect all descendants
        descendant_list = []
        to_process = map_sessionid_2_session_db[self.id]["children"][:]  # Create a shallow copy
        while to_process:
            current_id = to_process.pop()
            child_session_data = Session._get_session_data(
                current_id,
                map_sessionid_2_session_db=map_sessionid_2_session_db
            )
            descendant_list.append(current_id)
            if child_session_data[3] in sys.path:
                sys.path.remove(child_session_data[3])  # remove the workflow folderpath from sys.path
            # Add any children to the processing queue
            to_process.extend(child_session_data[2])

        # process all descendants
        for descendant_session_id in descendant_list:
            del map_sessionid_2_session_db[descendant_session_id]

        sys.path.remove(self.workflow_snapshot.workflow_folderpath)
        # remove ourselves from the sessionid_2_sessiondata_map
        del map_sessionid_2_session_db[self.id]

        map_sessionid_2_session_db.close()

        try:
            sessiondb_folderpath = Session._get_sessiondb_folderpath(
                session_id=self.id,
                parent_session_id=self.parent_id,
                workflow_folderpath=self.workflow_snapshot.workflow_folderpath
            )
            shutil.rmtree(sessiondb_folderpath, ignore_errors=True)
        except OSError as e:
            logger.error(f"Error closing session: {e}")
            return False

        return True

    @classmethod
    def _get_sessiondb_folderpath(
        cls, 
        session_id: int, 
        parent_session_id: Optional[int] = None,
        workflow_folderpath: Optional[str] = None
    ) -> str:
        """get the db folder path"""
        if parent_session_id is None and workflow_folderpath is None:
            raise ValueError("parent_session_id or workflow_folderpath must be provided")

        if parent_session_id:
            parent_session_folder = ""

            sessionid_2_sessiondata_mapdir = Session._get_session_id_2_sessiondata_mapdir()
            map_sessionid_2_session_db = Rdict(sessionid_2_sessiondata_mapdir)       
            
            while parent_session_id:
                parent_session_data = Session._get_session_data(
                    parent_session_id,
                    map_sessionid_2_session_db=map_sessionid_2_session_db
                )
                parent_session_id = parent_session_data[0]
                parent_session_folder = os.path.join(parent_session_data[1], parent_session_folder)
            
            map_sessionid_2_session_db.close()
        else:
            speedict_foldername = fastworkflow.get_env_var("SPEEDDICT_FOLDERNAME")
            parent_session_folder = os.path.join(
                workflow_folderpath, 
                speedict_foldername
            )
    
        session_id_str = str(session_id).replace("-", "_")
        return os.path.join(parent_session_folder, session_id_str)

    @classmethod
    def _get_session_data(cls, session_id: int, map_sessionid_2_session_db: Rdict) -> tuple[int, str, list, str]:
        """get the parent id, the session db folder path, the children list, and the workflow folderpath"""
        sessiondata_dict = map_sessionid_2_session_db.get(session_id, None)

        if not sessiondata_dict:
            raise ValueError(f"Session {session_id} not found")

        sessiondb_folderpath = sessiondata_dict["sessiondb_folderpath"]
        if not os.path.exists(sessiondb_folderpath):
            raise ValueError(f"Session database folder path {sessiondb_folderpath} does not exist")

        children_list = sessiondata_dict["children"]
        if children_list is None:
            raise ValueError(f"Session {session_id} must have a children list even if it is empty")

        keyvalue_db = Rdict(sessiondb_folderpath)
        workflow_snapshot: WorkflowSnapshot = keyvalue_db["workflow_snapshot"]
        keyvalue_db.close()

        return (workflow_snapshot.parent_session_id, sessiondb_folderpath, children_list, workflow_snapshot.workflow_folderpath)

    @classmethod
    def _get_session_id_2_sessiondata_mapdir(cls) -> str:
        """get the sessionid_2_sessiondata_map folder path"""
        speedict_foldername = fastworkflow.get_env_var("SPEEDDICT_FOLDERNAME")
        sessionid_2_sessiondata_mapdir = os.path.join(
            speedict_foldername,
            "sessionid_2_sessiondata_map"
        )
        os.makedirs(sessionid_2_sessiondata_mapdir, exist_ok=True)
        return sessionid_2_sessiondata_mapdir

    def get_cachedb_folderpath(self, function_name: str) -> str:
        """Get the cache database folder path for a specific function"""
        speedict_foldername = fastworkflow.get_env_var("SPEEDDICT_FOLDERNAME")
        return os.path.join(
            self.workflow_snapshot.workflow_folderpath,
            speedict_foldername,
            f"/function_cache/{function_name}",
        )
