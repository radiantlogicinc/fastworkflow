import os
from queue import Queue
import shutil
import sys
from functools import wraps
from typing import Optional, Union

from pydantic import BaseModel
from speedict import Rdict
import murmurhash

import fastworkflow
from fastworkflow.utils.logging import logger
from fastworkflow.workflow import Workflow, Workitem


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


class WorkflowSnapshot(BaseModel):
    workflow: Workflow
    context: dict
    active_workitem: Optional[Union[Workitem, Workflow]] = None

    def __init__(self, 
                 workflow: Workflow, 
                 active_workitem_path: str, 
                 active_workitem_id: Optional[Union[int, str]] = None,
                 context: dict = {}):
        super().__init__(
            workflow=workflow,
            context=context,
            active_workitem=workflow.find_workitem(
                active_workitem_path,
                active_workitem_id
            )
        )

class Session:
    """Session class"""
    @classmethod
    def create(
        cls,
        workflow_folderpath: str,
        session_id_str: Optional[str] = None, 
        parent_session_id: Optional[int] = None, 
        keep_alive: bool = False,
        user_message_queue: Optional[Queue] = None,
        command_output_queue: Optional[Queue] = None,
        context: dict = {},
        for_training_semantic_router: bool = False
    ) -> "Session":
        if session_id_str is None and parent_session_id is None:
            raise ValueError("Either session_id_str or parent_session_id must be provided")

        if session_id_str is not None and parent_session_id is not None:
            raise ValueError("session_id_str and parent_session_id cannot both be provided")

        if parent_session_id is not None and keep_alive:
            raise ValueError("keep_alive must be False if parent_session_id is provided")

        if session_id_str:
            session_id = cls._generate_session_id_from_str(session_id_str)
        else:
            session_id = cls._generate_child_session_id(parent_session_id, workflow_folderpath)

        if session := cls.get_session(
            session_id, 
            keep_alive, 
            user_message_queue, 
            command_output_queue,
            context
        ):
            return session

        if not os.path.exists(workflow_folderpath):
            raise ValueError(f"The folder path {workflow_folderpath} does not exist")

        if not os.path.isdir(workflow_folderpath):
            raise ValueError(f"{workflow_folderpath} must be a directory")

        workflow_snapshot = WorkflowSnapshot(
            workflow=Workflow(
                workflow_folderpath=workflow_folderpath,
                type=os.path.basename(workflow_folderpath).rstrip("/"),
                parent_workflow=None,
            ),
            active_workitem_path="/",
            active_workitem_id=None,
            context=context,
        )
        session = Session(cls.__create_key, 
                          workflow_snapshot,
                          session_id, 
                          parent_session_id,
                          keep_alive, 
                          user_message_queue, 
                          command_output_queue)
        session.save()  # save the workflow snapshot

        return session

    @classmethod
    def get_session(cls, 
             session_id: int, 
             keep_alive: bool = False,
             user_message_queue: Optional[Queue] = None,
             command_output_queue: Optional[Queue] = None,
             context: dict = {}) -> Optional["Session"]:
        """load the session"""
        map_sessionid_2_sessiondb_folderpath_dir = Session._get_session_id_2_sessiondb_folderpath()
        map_sessionid_2_sessiondb_folderpath_db = Rdict(map_sessionid_2_sessiondb_folderpath_dir)
        sessiondb_folderpath = map_sessionid_2_sessiondb_folderpath_db.get(session_id, None)
        map_sessionid_2_sessiondb_folderpath_db.close()
        if not sessiondb_folderpath:
            return None

        if not os.path.exists(sessiondb_folderpath):
            raise ValueError(f"Session database folder path {sessiondb_folderpath} does not exist")

        keyvalue_db = Rdict(sessiondb_folderpath)
        workflow_snapshot: WorkflowSnapshot = keyvalue_db["workflow_snapshot"]
        parent_session_id = keyvalue_db.get("parent_session_id", None)
        keyvalue_db.close()

        # set the new context, this means we have to save the session again
        workflow_snapshot.context = context

        session = Session(cls.__create_key, 
                          workflow_snapshot,
                          session_id, 
                          parent_session_id,
                          keep_alive, 
                          user_message_queue, 
                          command_output_queue)
        
        session.save()
        return session

    @classmethod
    def _generate_session_id_from_str(cls, session_id_str: str) -> int:
        return int(murmurhash.hash(session_id_str))

    @classmethod
    def _generate_child_session_id(cls, parent_session_id: int, workflow_folderpath: str) -> int:
        """generate a child session id"""
        workflow_type = os.path.basename(workflow_folderpath).rstrip("/")
        session_id_str = f"{parent_session_id}{workflow_type}"
        return cls._generate_session_id_from_str(session_id_str)

    # enforce session creation exclusively using Session.create_session
    # https://stackoverflow.com/questions/8212053/private-constructor-in-python
    __create_key = object()
   
    def __init__(self,
                 create_key, 
                 workflow_snapshot: WorkflowSnapshot,
                 session_id: int, 
                 parent_session_id: Optional[int] = None,
                 keep_alive: bool = False,
                 user_message_queue: Optional[Queue] = None,
                 command_output_queue: Optional[Queue] = None):
        """initialize the Session class"""
        if create_key is Session.__create_key:
            pass
        else:
            raise ValueError("Session objects must be created using Session.create")

        workflow_folderpath = workflow_snapshot.workflow.workflow_folderpath
        if workflow_folderpath not in sys.path:
            # THIS IS IMPORTANT: it allows relative import of modules in the code inside workflow_folderpath
            sys.path.insert(0, workflow_folderpath)

        self._workflow_snapshot = workflow_snapshot
        self._session_id = session_id
        self._parent_session_id = parent_session_id
        self._keep_alive = keep_alive
        self._user_message_queue = user_message_queue
        self._command_output_queue = command_output_queue

    @property
    def id(self) -> int:
        """get the session id"""
        return self._session_id

    @property
    def workflow_snapshot(self) -> WorkflowSnapshot:
        """get the workflow snapshot"""
        return self._workflow_snapshot

    @property
    def keep_alive(self) -> bool:
        """get the keep alive flag"""
        return self._keep_alive

    @property
    def user_message_queue(self) -> Queue:
        """get the user message queue"""
        return self._user_message_queue

    @property
    def command_output_queue(self) -> Queue:
        """get the command output queue"""
        return self._command_output_queue

    def close(self) -> bool:
        """close the session"""
        sessiondb_folderpath = self._get_sessiondb_folderpath()
        try:
            shutil.rmtree(sessiondb_folderpath, ignore_errors=True)
        except OSError as e:
            logger.error(f"Error closing session: {e}")
            return False

        sys.path.remove(self.workflow_snapshot.workflow.workflow_folderpath)

        map_sessionid_2_sessiondb_folderpath_dir = Session._get_session_id_2_sessiondb_folderpath()
        map_sessionid_2_sessiondb_folderpath_db = Rdict(map_sessionid_2_sessiondb_folderpath_dir)
        del map_sessionid_2_sessiondb_folderpath_db[self._session_id]
        map_sessionid_2_sessiondb_folderpath_db.close()

        return True

    def save(self) -> None:
        """save the session"""
        sessiondb_folderpath = self._get_sessiondb_folderpath()
        os.makedirs(sessiondb_folderpath, exist_ok=True)       

        keyvalue_db = Rdict(sessiondb_folderpath)
        keyvalue_db["workflow_snapshot"] = self._workflow_snapshot
        keyvalue_db["parent_session_id"] = self._parent_session_id
        keyvalue_db.close()

    def _get_sessiondb_folderpath(self) -> str:
        """get the db folder path"""
        if self._parent_session_id:
            parent_session = Session.get_session(self._parent_session_id)
            if not parent_session:
                raise ValueError(f"Parent session {self._parent_session_id} not found")
            parent_session_folder = parent_session._get_sessiondb_folderpath()
        else:
            speedict_foldername = fastworkflow.get_env_var("SPEEDDICT_FOLDERNAME")
            parent_session_folder = os.path.join(
                self.workflow_snapshot.workflow.workflow_folderpath, 
                speedict_foldername
            )
    
        session_id_str = str(self._session_id).replace("-", "_")
        session_db_folderpath = os.path.join(parent_session_folder, session_id_str)

        map_sessionid_2_sessiondb_folderpath_dir = Session._get_session_id_2_sessiondb_folderpath()
        map_sessionid_2_sessiondb_folderpath_db = Rdict(map_sessionid_2_sessiondb_folderpath_dir)
        map_sessionid_2_sessiondb_folderpath_db[self._session_id] = session_db_folderpath
        map_sessionid_2_sessiondb_folderpath_db.close()

        return session_db_folderpath

    @classmethod
    def _get_session_id_2_sessiondb_folderpath(cls) -> str:
        """get the session db folder path"""
        speedict_foldername = fastworkflow.get_env_var("SPEEDDICT_FOLDERNAME")
        map_sessionid_2_sessiondb_folderpath_dir = os.path.join(
            speedict_foldername,
            "map_sessionid_2_sessiondb_folderpath"
        )
        os.makedirs(map_sessionid_2_sessiondb_folderpath_dir, exist_ok=True)
        return map_sessionid_2_sessiondb_folderpath_dir

    def get_cachedb_folderpath(self, function_name: str) -> str:
        """Get the cache database folder path for a specific function"""
        speedict_foldername = fastworkflow.get_env_var("SPEEDDICT_FOLDERNAME")
        return os.path.join(
            self.workflow_snapshot.workflow.workflow_folderpath,
            speedict_foldername,
            f"/function_cache/{function_name}",
        )
