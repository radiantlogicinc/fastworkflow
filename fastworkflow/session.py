import os
from queue import Queue
import shutil
import sys
from functools import wraps
from typing import Any, Optional, Union

from pydantic import BaseModel
from speedict import Rdict

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

SPEEDDICT_FOLDERNAME = "___workflow_contexts"


class Session:
    """Session class"""
    @classmethod
    def get_session(cls, 
                    session_id: int,
                    keep_alive: bool = False,
                    user_message_queue: Optional[Queue] = None,
                    command_output_queue: Optional[Queue] = None) -> "Session":
        if session_id not in cls._map_session_id_2_session:
            session = cls._load(session_id, keep_alive, user_message_queue, command_output_queue)
            if not session:
                return None

            workflow_folderpath = session.workflow_snapshot.workflow.workflow_folderpath
            if workflow_folderpath not in sys.path:
                # THIS IS IMPORTANT: it allows relative import of modules in the code inside workflow_folderpath
                sys.path.insert(0, workflow_folderpath)

            cls._map_session_id_2_session[session_id] = session

        return cls._map_session_id_2_session[session_id]

    @classmethod
    def create(
        cls,
        session_id: int, 
        workflow_folderpath: str,
        keep_alive: bool = False,
        user_message_queue: Optional[Queue] = None,
        command_output_queue: Optional[Queue] = None,
        context: dict = {},
        for_training_semantic_router: bool = False
    ) -> "Session":
        if session := cls.get_session(session_id, keep_alive, user_message_queue, command_output_queue):
            return session

        if not os.path.exists(workflow_folderpath):
            raise ValueError(f"The folder path {workflow_folderpath} does not exist")

        if not os.path.isdir(workflow_folderpath):
            raise ValueError(f"{workflow_folderpath} must be a directory")

        # THIS IS IMPORTANT: it allows relative import of modules in the code inside workflow_folderpath
        sys.path.insert(0, workflow_folderpath)

        speedict_folderpath = os.path.join(workflow_folderpath, SPEEDDICT_FOLDERNAME)
        os.makedirs(speedict_folderpath, exist_ok=True)       

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
                          session_id, 
                          workflow_snapshot, 
                          keep_alive, 
                          user_message_queue, 
                          command_output_queue)
        session.save()  # save the workflow snapshot

        Session._map_session_id_2_session[session_id] = session
        return session

    @classmethod
    def _load(cls, 
             session_id: int, 
             keep_alive: bool = False,
             user_message_queue: Optional[Queue] = None,
             command_output_queue: Optional[Queue] = None) -> Optional["Session"]:
        """load the session"""
        if session_id in cls._map_session_id_2_session:
            return cls._map_session_id_2_session[session_id]

        sessiondb_folderpath = cls._get_sessiondb_folderpath(session_id)
        if not os.path.exists(sessiondb_folderpath):
            return None

        keyvalue_db = Rdict(sessiondb_folderpath)
        workflow_snapshot: WorkflowSnapshot = keyvalue_db.get["workflow_snapshot"]
        keyvalue_db.close()

        session = Session(cls.__create_key, 
                          session_id, 
                          workflow_snapshot, 
                          keep_alive, 
                          user_message_queue, 
                          command_output_queue)
        Session._map_session_id_2_session[session_id] = session
        return session

    _map_session_id_2_session: dict[int, "Session"] = {}

    # enforce session creation exclusively using Session.create_session
    # https://stackoverflow.com/questions/8212053/private-constructor-in-python
    __create_key = object()
   
    def __init__(self,
                 create_key, 
                 session_id: int, 
                 workflow_snapshot: WorkflowSnapshot,
                 keep_alive: bool = False,
                 user_message_queue: Optional[Queue] = None,
                 command_output_queue: Optional[Queue] = None):
        """initialize the Session class"""
        if create_key is Session.__create_key:
            pass
        else:
            raise ValueError("Session objects must be created using Session.create")

        self._session_id = session_id
        self._workflow_snapshot = workflow_snapshot
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
        sessiondb_folderpath = self._get_sessiondb_folderpath(self.id)
        try:
            shutil.rmtree(sessiondb_folderpath, ignore_errors=True)
        except OSError as e:
            logger.error(f"Error closing session: {e}")
            return False

        sys.path.remove(self.workflow_snapshot.workflow.workflow_folderpath)

        if self.id in Session._map_session_id_2_session:
            del Session._map_session_id_2_session[self.id]

        return True

    def save(self) -> None:
        """save the session"""
        sessiondb_folderpath = self._get_sessiondb_folderpath(self.id)
        keyvalue_db = Rdict(sessiondb_folderpath)
        keyvalue_db["workflow_snapshot"] = self._workflow_snapshot
        keyvalue_db.close()

    @classmethod
    def _get_sessiondb_folderpath(cls, session_id: int) -> str:
        """get the db folder path"""
        session_id_str = str(session_id).replace("-", "_")
        return os.path.join(
            SPEEDDICT_FOLDERNAME, session_id_str
        )

    def get_cachedb_folderpath(self, function_name: str) -> str:
        """Get the cache database folder path for a specific function"""
        return os.path.join(
            self.workflow_snapshot.workflow.workflow_folderpath,
            SPEEDDICT_FOLDERNAME,
            f"/function_cache/{function_name}",
        )
