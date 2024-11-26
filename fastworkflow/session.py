import os
from queue import Queue
import shutil
import sys
from functools import wraps
from typing import Optional, Union

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


class WorkflowSnapshot:
    def __init__(self,
                 session_id: int,
                 workflow: Workflow, 
                 active_workitem: Union[Workitem, Workflow],
                 context: dict = {},
                 parent_session_id: Optional[int] = None):
        self._session_id = session_id
        self._workflow = workflow
        self._active_workitem = active_workitem
        self._context = context
        self._parent_session_id = parent_session_id
        self._save()

    @property
    def session_id(self) -> int:
        return self._session_id

    @property
    def workflow(self) -> Workflow:
        return self._workflow

    @workflow.setter
    def workflow(self, value: Workflow) -> None:
        self._workflow = value
        self._save()

    @property
    def active_workitem(self) -> Union[Workitem, Workflow]:
        return self._active_workitem
    
    @active_workitem.setter
    def active_workitem(self, value: Union[Workitem, Workflow]) -> None:
        if value.path == self._active_workitem.path and value.id == self._active_workitem.id:
            return
        self._active_workitem = value
        self._save()

    @property
    def context(self) -> dict:
        return self._context
    
    @context.setter
    def context(self, value: dict) -> None:
        if value == self._context:
            return
        self._context = value
        self._save()

    @property
    def parent_session_id(self) -> Optional[int]:
        return self._parent_session_id

    def _save(self) -> None:
        """save the workflow snapshot"""
        sessiondb_folderpath = Session._get_sessiondb_folderpath(
            session_id=self._session_id,
            parent_session_id=self._parent_session_id,
            workflow_folderpath=self._workflow.workflow_folderpath
        )
        os.makedirs(sessiondb_folderpath, exist_ok=True)     

        keyvalue_db = Rdict(sessiondb_folderpath)
        keyvalue_db["workflow_snapshot"] = self
        keyvalue_db.close()


class Session:
    """Session class"""
    @classmethod
    def create(
        cls,
        workflow_folderpath: str,
        session_id_str: Optional[str] = None, 
        parent_session_id: Optional[int] = None, 
        user_message_queue: Optional[Queue] = None,
        command_output_queue: Optional[Queue] = None,
        context: dict = {},
        for_training_semantic_router: bool = False
    ) -> "Session":
        if session_id_str is None and parent_session_id is None:
            raise ValueError("Either session_id_str or parent_session_id must be provided")

        if session_id_str is not None and parent_session_id is not None:
            raise ValueError("session_id_str and parent_session_id cannot both be provided")

        if session_id_str:
            session_id = fastworkflow.get_session_id(session_id_str)
        else:
            session_id = cls._generate_child_session_id(parent_session_id, workflow_folderpath)

        if session := cls.get_session(
            session_id, 
            user_message_queue, 
            command_output_queue,
            context
        ):
            return session

        if not os.path.exists(workflow_folderpath):
            raise ValueError(f"The folder path {workflow_folderpath} does not exist")

        if not os.path.isdir(workflow_folderpath):
            raise ValueError(f"{workflow_folderpath} must be a directory")

        workflow=Workflow(
            workflow_folderpath=workflow_folderpath,
            type=os.path.basename(workflow_folderpath).rstrip("/"),
            parent_workflow=None,
        )
        workflow_snapshot = WorkflowSnapshot(
            session_id=session_id,
            workflow=workflow,
            active_workitem=workflow.find_workitem("/", None),
            context=context,
            parent_session_id=parent_session_id,
        )
        session = Session(cls.__create_key, 
                          workflow_snapshot,
                          user_message_queue, 
                          command_output_queue)

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
             user_message_queue: Optional[Queue] = None,
             command_output_queue: Optional[Queue] = None,
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
            workflow_snapshot.context = context

        session = Session(cls.__create_key, 
                          workflow_snapshot,
                          user_message_queue, 
                          command_output_queue)

        return session

    @classmethod
    def _generate_child_session_id(cls, parent_session_id: int, workflow_folderpath: str) -> int:
        """generate a child session id"""
        workflow_type = os.path.basename(workflow_folderpath).rstrip("/")
        session_id_str = f"{parent_session_id}{workflow_type}"
        return fastworkflow.get_session_id(session_id_str)

    # enforce session creation exclusively using Session.create_session
    # https://stackoverflow.com/questions/8212053/private-constructor-in-python
    __create_key = object()
   
    def __init__(self,
                 create_key, 
                 workflow_snapshot: WorkflowSnapshot,
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
        self._user_message_queue = user_message_queue
        self._command_output_queue = command_output_queue

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

        sys.path.remove(self.workflow_snapshot.workflow.workflow_folderpath)
        # remove ourselves from the sessionid_2_sessiondata_map
        del map_sessionid_2_session_db[self.id]

        map_sessionid_2_session_db.close()

        try:
            sessiondb_folderpath = Session._get_sessiondb_folderpath(
                session_id=self.id,
                parent_session_id=self.parent_id,
                workflow_folderpath=self.workflow_snapshot.workflow.workflow_folderpath
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
    def _get_session_data(cls, session_id: int, map_sessionid_2_session_db: Rdict) -> (int, str, list, str):
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

        return (workflow_snapshot.parent_session_id, sessiondb_folderpath, children_list, workflow_snapshot.workflow.workflow_folderpath)

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
            self.workflow_snapshot.workflow.workflow_folderpath,
            speedict_foldername,
            f"/function_cache/{function_name}",
        )
