import os
from typing import List, Optional, Union

from pydantic import BaseModel

import fastworkflow
from fastworkflow.workflow_definition import NodeType, SizeMetaData


class Workitem(BaseModel):
    def __init__(
        self,
        path: str,
        node_type: NodeType,
        parent_workflow: Optional["Workflow"],
        id: Optional[Union[str, int]] = None
    ):
        super().__init__()

        if not path:
            raise ValueError("path cannot be empty")
        self._path = path
        self._node_type = node_type
        self._parent_workflow = parent_workflow
        self._id = id

    @property
    def node_type(self) -> str:
        return self._node_type

    @property
    def is_complete(self) -> bool:
        return self._is_complete

    @is_complete.setter
    def is_complete(self, value: bool = True):
        if value:
            self._is_complete = True
            if self._parent_workflow:
                self._parent_workflow.has_started = True
        elif self._parent_workflow:
            self._parent_workflow._recalculate_started_state()

    @property
    def parent_workflow(self) -> "Workflow":
        return self._parent_workflow

    @property
    def path(self) -> str:
        return self._path

    @property
    def id(self) -> Optional[Union[str, int]]:
        return self._id

    def next_workitem(
        self, skip_completed: bool = True, current_item_path: str = None
    ) -> Union["Workitem", "Workflow", None]:
        if self.node_type == NodeType.Workflow:
            # if the current item is a child of this workitem (self),
            # then we need to loop until we find the current item
            found_current_item = (
                not current_item_path or self.path not in current_item_path
            )

            # search children
            for item in self._workitems:
                if not found_current_item:
                    if item.path == current_item_path and item.id == self.id:
                        found_current_item = True
                    continue

                if not skip_completed or not item.is_complete:
                    return item

            # delegate search to grandparent workflow
            if self._parent_workflow:
                return self._parent_workflow.next_workitem(skip_completed, self.path)
        elif self._parent_workflow:
            current_item_path = self.path
            # search siblings
            found_current_item = False
            for item in self._parent_workflow._workitems:
                if item.path == current_item_path and item.id == self.id:
                    found_current_item = True
                    continue

                if not found_current_item:
                    continue

                if skip_completed and item.is_complete:
                    continue

                return item

            # delegate search to grandparent workflow
            if self._parent_workflow._parent_workflow:
                return self._parent_workflow._parent_workflow.next_workitem(
                    skip_completed, self._parent_workflow.path
                )

        return None

    def print(self, indent=0):
        print(" " * (indent + 2) + f"Workitem(path={self._path}, id={self._id})")

    _node_type: str
    _is_complete: bool = False
    _parent_workflow: Optional["Workflow"] = None
    _path: str
    _id: Optional[str] = None


class Workflow(Workitem):
    # don't use this directly, use the registry to create workflows
    def __init__(
        self,
        workflow_folderpath: str,
        path: str,
        parent_workflow: Optional["Workflow"],
        id: Optional[str] = None
    ):
        super().__init__(path, NodeType.Workflow, parent_workflow, id)

        self._workflow_folderpath = workflow_folderpath
        self._path = path

        workflow_definition = fastworkflow.WorkflowRegistry.get_definition(self._workflow_folderpath)
        if path in workflow_definition.paths_2_allowable_child_paths_2_sizemetadata:
            allowable_child_paths_2_sizemetadata = (
                workflow_definition.paths_2_allowable_child_paths_2_sizemetadata[path]
            )
        else:
            allowable_child_paths_2_sizemetadata = {}

        # for each child type, check if the min and max size are same
        # if they are, add min size child workitems to the list
        for child_path, size_metadata in allowable_child_paths_2_sizemetadata.items():
            if size_metadata.min == size_metadata.max:
                type_metadata = workflow_definition.paths_2_typemetadata[child_path]
                if type_metadata.node_type == NodeType.Workitem:
                    self._workitems.extend(
                        [
                            Workitem(
                                path=child_path,
                                node_type=NodeType.Workitem,
                                parent_workflow=self,
                            )
                        ]
                        * size_metadata.min
                    )
                elif type_metadata.node_type == NodeType.Workflow:
                    self._workitems.extend(
                        [
                            Workflow(
                                workflow_folderpath=workflow_folderpath,
                                path=child_path,
                                parent_workflow=self,
                            )
                        ]
                        * size_metadata.min
                    )
                else:
                    raise ValueError(
                        f"Invalid workitem type '{type_metadata.node_type}'"
                    )

    @property
    def workflow_folderpath(self) -> str:
        return self._workflow_folderpath

    @property
    def has_started(self) -> bool:
        return self._has_started

    @has_started.setter
    def has_started(self, value: bool = True):
        self._has_started = value
        if value:
            # if all workitems are complete, mark this workflow as complete
            if all(item.is_complete for item in self._workitems):
                self.is_complete = True
            elif self._parent_workflow:
                self._parent_workflow.has_started = True
        elif self._parent_workflow:
            self._parent_workflow._recalculate_started_state()

    def _recalculate_started_state(self):
        # if all workitems under parent's workflow are incomplete and all workflows under parent workflow are not started, mark this workflow as incomplete
        any_workitem_is_complete = any(item.is_complete for item in self._workitems if item.node_type == NodeType.Workitem)
        any_workflow_has_started = any(item.has_started for item in self._workitems if item.node_type == NodeType.Workflow)
        self.has_started = any_workitem_is_complete or any_workflow_has_started

    def add_workitem(self, item_path: str, item_id: Optional[Union[str, int]] = None):
        workflow_definition = fastworkflow.WorkflowRegistry.get_definition(self._workflow_folderpath)
        allowable_child_paths_2_sizemetadata = (
            workflow_definition.paths_2_allowable_child_paths_2_sizemetadata[self._path]
        )
        # check if item_path is valid
        if item_path not in allowable_child_paths_2_sizemetadata:
            raise ValueError(
                f"Item type '{item_path}' is not allowed. Allowed types: {allowable_child_paths_2_sizemetadata}"
            )

        # check that the number of child workitems of this type is within the allowable range
        size_metadata = allowable_child_paths_2_sizemetadata[item_path]
        if (
            size_metadata.max
            and len([i for i in self._workitems if i._path == item_path])
            >= size_metadata.max
        ):
            raise ValueError(
                f"Maximum number of child workitems of type '{item_path}' reached"
            )

        # check that the id is not already used
        if item_id is not None and any(i.id == item_id for i in self._workitems):
            raise ValueError(f"Workitem with ID '{item_id}' already exists in this workflow.")

        # is this a Workitem or a Workflow?
        type_metadata = workflow_definition.paths_2_typemetadata[item_path]
        if type_metadata.node_type == NodeType.Workitem:
            item = Workitem(
                path=item_path,
                node_type=NodeType.Workitem,
                parent_workflow=self,
                id=item_id,
            )
        elif type_metadata.node_type == NodeType.Workflow:
            item = Workflow(
                workflow_folderpath=self._workflow_folderpath,
                path=item_path,
                parent_workflow=self,
                id=item_id,
            )
        else:
            raise ValueError(f"Invalid workitem type '{type_metadata.node_type}'")

        self._workitems.append(item)
        return item

    # implement a function that finds a workitem, given the path and an optional id
    # the path uses the same format as a file system path, e.g. "Anomalous/Leavers"
    # the path could be absolute or relative to the current workflow
    def find_workitem(
        self, path: str, item_id: Optional[Union[str, int]] = None, relative_to_root: bool = False
    ) -> Union[Workitem, "Workflow", None]:
        """Find a workitem by path and optional ID.
        
        Args:
            path: Path to the workitem. Can be:
                - Absolute path starting with '/' (e.g. '/accessreview/anomalous')
                - Relative path (e.g. 'anomalous/leavers' or './anomalous/leavers')
            item_id: Optional ID to match specific workitem
            relative_to_root: If True, treats path as relative to root workflow
        """
        if not path:
            raise ValueError("path cannot be empty")

        # Validate and normalize path
        parts = [p for p in path.split("/") if p]  # Remove empty parts
        if any("." in part for part in parts):  # Catches both '.' and '..'
            raise ValueError("Path cannot contain '.' or '..' components")

        # Handle absolute paths or relative_to_root flag
        workitem = self
        if path.startswith("/") or relative_to_root:
            while workitem._parent_workflow:
                workitem = workitem._parent_workflow

            # Skip root name if it matches first path part to avoid double-processing
            if parts and os.path.basename(workitem._path.rstrip('/')) == parts[0]:
                parts = parts[1:]

        # Handle empty paths or paths pointing to root
        return workitem._find_workitem_recursive(parts, item_id) if parts else workitem

    def _find_workitem_recursive(
        self, parts: List[str], item_id: Optional[Union[str, int]] = None
    ) -> Union[Workitem, "Workflow", None]:
        """Recursively search for a workitem in the workflow tree.
        
        Args:
            parts: Normalized path parts to search
            item_id: Optional ID to match specific workitem
        """
        current_name = os.path.basename(self._path.rstrip('/'))

        # Check if current workflow matches first path part
        if current_name == parts[0]:
            if len(parts) == 1:  # This is the target
                return self if not item_id or self._id == item_id else None
            return self._find_workitem_recursive(parts[1:], item_id)

        # Search through child items
        for item in self._workitems:
            item_name = os.path.basename(item._path.rstrip('/'))

            if item_name == parts[0]:
                if len(parts) == 1:
                    if not item_id or str(item._id) == str(item_id):
                        return item
                    else:
                        continue

                if isinstance(item, Workflow):
                    if result := item._find_workitem_recursive(parts[1:], item_id):
                        return result

            elif isinstance(item, Workflow):
                if result := item._find_workitem_recursive(parts, item_id):
                    return result

        return None

    def print_tree(self, indent=0):
        print(" " * indent + f"Workflow(path={self._path})")
        for item in self._workitems:
            if isinstance(item, Workflow):
                item.print_tree(indent + 2)
            else:
                item.print(indent)

    class Config:
        arbitrary_types_allowed = True

    _has_started: bool = False
    _workitems: List[Union[Workitem, "Workflow"]] = []
