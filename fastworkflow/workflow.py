import os
from typing import List, Optional, Union

from pydantic import BaseModel

import fastworkflow
from fastworkflow.workflow_definition import NodeType, SizeMetaData, WorkflowDefinition


class Workitem(BaseModel):
    def __init__(
        self,
        type: str,
        node_type: NodeType,
        parent_workflow: Optional["Workflow"],
        id: Optional[str] = None
    ):
        super().__init__()

        if not type:
            raise ValueError("type cannot be empty")
        self._type = type
        self._node_type = node_type
        self._parent_workflow = parent_workflow
        self._id = id

        if parent_workflow:
            parent_path = parent_workflow.path
        else:
            parent_path = ""
        self._path = f"{parent_path}/{type}"

    @property
    def type(self) -> str:
        return self._type

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
        else:
            if self._parent_workflow:
                self._parent_workflow._recalculate_started_state()

    @property
    def parent_workflow(self) -> "Workflow":
        return self._parent_workflow

    @property
    def path(self) -> str:
        return self._path

    @property
    def id(self) -> Optional[str]:
        return self._id

    def next_workitem(
        self, skip_completed: bool = True, current_item_path: str = None
    ) -> Union["Workitem", "Workflow", None]:
        if self.node_type == NodeType.Workflow:
            # if the current item is a child of this workitem (self),
            # then we need to loop until we find the current item
            found_current_item = (
                False if current_item_path and self.path in current_item_path else True
            )

            # search children
            for item in self._workitems:
                if not found_current_item:
                    if item.path == current_item_path:
                        found_current_item = True
                    continue

                if skip_completed and item.is_complete:
                    continue

                return item

            # delegate search to grandparent workflow
            if self._parent_workflow:
                return self._parent_workflow.next_workitem(skip_completed, self.path)
        else:
            if self._parent_workflow:
                current_item_path = self.path
                # search siblings
                found_current_item = False
                for item in self._parent_workflow._workitems:
                    if item.path == current_item_path:
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
        print(" " * (indent + 2) + f"Workitem(type={self._type}, id={self._id})")

    _type: str
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
        type: str,
        parent_workflow: Optional["Workflow"],
        id: Optional[str] = None
    ):
        super().__init__(type, NodeType.Workflow, parent_workflow, id)

        self._workflow_folderpath = workflow_folderpath
        self._type = type

        workflow_definition = fastworkflow.WorkflowRegistry.get_definition(self._workflow_folderpath)
        self._allowable_child_types = workflow_definition.allowable_child_types[
            type
        ]

        # for each child type, check if the min and max size are same
        # if they are, add min size child workitems to the list
        for child_type, size_metadata in self._allowable_child_types.items():
            if size_metadata.min == size_metadata.max:
                type_metadata = workflow_definition.types[child_type]
                if type_metadata.node_type == NodeType.Workitem:
                    self._workitems.extend(
                        [
                            Workitem(
                                type=child_type,
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
                                type=child_type,
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
    def type(self) -> str:
        return self._type

    @property
    def has_started(self) -> bool:
        return self._has_started

    @has_started.setter
    def has_started(self, value: bool = True):
        self._has_started = value
        if value:
            # if all workitems are complete, mark this workflow as complete
            if all([item.is_complete for item in self._workitems]):
                self.is_complete = True
            else:
                if self._parent_workflow:
                    self._parent_workflow.has_started = True
        else:
            self._parent_workflow._recalculate_started_state()

    def _recalculate_started_state(self):
        # if all workitems under parent's workflow are incomplete and all workflows under parent workflow are not started, mark this workflow as incomplete
        any_workitem_is_complete = any(item.is_complete for item in self._workitems if item.node_type == NodeType.Workitem)
        any_workflow_has_started = any(item.has_started for item in self._workitems if item.node_type == NodeType.Workflow)
        self.has_started = any_workitem_is_complete or any_workflow_has_started

    def add_workitem(self, item_type: str, item_id: Optional[str] = None):
        # check if item_type is valid
        if item_type not in self._allowable_child_types:
            raise ValueError(
                f"Item type '{item_type}' is not allowed. Allowed types: {self._allowable_child_types}"
            )

        # check that the number of child workitems of this type is within the allowable range
        size_metadata = self._allowable_child_types[item_type]
        if (
            size_metadata.max
            and len([i for i in self._workitems if i._type == item_type])
            >= size_metadata.max
        ):
            raise ValueError(
                f"Maximum number of child workitems of type '{item_type}' reached"
            )

        # is this a Workitem or a Workflow?
        workflow_definition = fastworkflow.WorkflowRegistry.get_definition(self._workflow_folderpath)
        type_metadata = workflow_definition.types[item_type]
        if type_metadata.node_type == NodeType.Workitem:
            item = Workitem(
                type=item_type,
                node_type=NodeType.Workitem,
                parent_workflow=self,
                id=item_id,
            )
        elif type_metadata.node_type == NodeType.Workflow:
            item = Workflow(
                workflow_folderpath=self._workflow_folderpath,
                type=item_type,
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
        self, path: str, item_id: str = None, relative_to_root: bool = False
    ) -> Union[Workitem, "Workflow", None]:
        if not path:
            raise ValueError("path cannot be empty")

        workitem = self

        # split the path into parts
        parts = path.split("/")

        # Handle '//' at the beginning of the path
        if path.startswith("//"):
            if relative_to_root:
                # find the root workflow by recursively going up the parent chain
                while workitem._parent_workflow:
                    workitem = workitem._parent_workflow
            return workitem._find_workitem_recursive(parts[2:], item_id)

        # if path starts with a '/', it is an absolute path
        # if path is absolute or empty, parts[0] is empty
        if not parts[0] or relative_to_root:
            # find the root workflow by recursively going up the parent chain
            while workitem._parent_workflow:
                workitem = workitem._parent_workflow

            parts = parts[1:]
            if (
                workitem.type == parts[0]
            ):  # since we are getting stuff relative to the root workflow
                parts = parts[1:]
        elif parts[0] == ".":  # relative path
            parts = parts[1:]

        if not parts or not parts[0]:
            return workitem

        # raise error if any part contains a '/', '.' or '..', or is empty - as this is not supported
        if any(
            [part == "" or "/" in part or part == "." or part == ".." for part in parts]
        ):
            raise ValueError("Invalid path")

        for part in parts:
            found = False
            for item in workitem._workitems:
                if item._type == part:
                    if isinstance(item, Workflow):
                        workitem = item
                        found = True
                        break
                    else:
                        if item_id:
                            if item._id == item_id:
                                return item
                        else:
                            return item

            if not found:
                return None

        return workitem

    def _find_workitem_recursive(
        self, parts: List[str], item_id: str = None
    ) -> Union[Workitem, "Workflow", None]:
        # Check if the current workflow matches the first part
        if parts and self._type == parts[0]:
            if len(parts) == 1:
                return self if not item_id or self._id == item_id else None
            return self._find_workitem_recursive(parts[1:], item_id)

        for item in self._workitems:
            if item._type == parts[0]:
                if len(parts) == 1:
                    if item_id:
                        if item._id == item_id:
                            return item
                        else:
                            continue
                    return item
                if isinstance(item, Workflow):
                    result = item._find_workitem_recursive(parts[1:], item_id)
                    if result:
                        return result
            elif isinstance(item, Workflow):
                result = item._find_workitem_recursive(parts, item_id)
                if result:
                    return result

        return None

    def print_tree(self, indent=0):
        print(" " * indent + f"Workflow(type={self._type})")
        for item in self._workitems:
            if isinstance(item, Workflow):
                item.print_tree(indent + 2)
            else:
                item.print(indent)

    class Config:
        arbitrary_types_allowed = True

    _has_started: bool = False
    _allowable_child_types: dict[str, SizeMetaData] = {}
    _workitems: List[Union[Workitem, "Workflow"]] = []
