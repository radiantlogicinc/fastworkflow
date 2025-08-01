"""WorkItem class for representing individual work items.

This module defines the WorkItem class, which represents a single work item
with attributes like id, type, and status.
"""

from typing import Dict, Any, Type, Union, Optional, List
import re
import json


class WorkItem:
    """Represents a single work item."""

    class ChildSchema:
        def __init__(self,
                     child_workitem_type: str,
                     min_cardinality: int = 0,
                     max_cardinality: Optional[int] = None) -> None:
            """Initialize a new ChildSchema."""
            if (not child_workitem_type or
                    ' ' in child_workitem_type):
                raise ValueError(
                    "child_workitem_type cannot be empty or contain spaces"
                )

            # Check for negative cardinality values
            if min_cardinality < 0:
                raise ValueError("min_cardinality cannot be negative")

            if max_cardinality is not None and max_cardinality < 0:
                raise ValueError("max_cardinality cannot be negative")

            # Check cardinality constraints before assigning
            if (max_cardinality is not None and
                    min_cardinality > max_cardinality):
                raise ValueError(
                    "min_cardinality cannot be greater than max_cardinality"
                )

            self.child_workitem_type = child_workitem_type
            self.min_cardinality = min_cardinality
            self.max_cardinality = max_cardinality

        def to_dict(self) -> Dict[str, Any]:
            """Serialize ChildSchema to a dictionary."""
            return {
                "child_workitem_type": self.child_workitem_type,
                "min_cardinality": self.min_cardinality,
                "max_cardinality": self.max_cardinality,
            }

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> "WorkItem.ChildSchema":
            """Create a ChildSchema instance from a dictionary."""
            return cls(
                child_workitem_type=data["child_workitem_type"],
                min_cardinality=data.get("min_cardinality", 0),
                max_cardinality=data.get("max_cardinality"),
            )

    class WorkflowSchema:
        def __init__(
            self,
            workflow_types: Optional[List[str]] = None,
            child_schema_dict: Optional[Dict[str, List["WorkItem.ChildSchema"]]] = None
        ) -> None:
            self.workflow_types = workflow_types

            # Validate workflow_types and child_schema_dict
            if workflow_types and child_schema_dict:
                # Check that all child_schema_dict keys are in workflow_types
                for key in child_schema_dict:
                    if key not in workflow_types:
                        raise ValueError(f"Child schema key '{key}' not found in workflow_types")
                
                # Check that all ChildSchema.child_workitem_type values are in workflow_types
                for schema_list in child_schema_dict.values():
                    if schema_list is not None:
                        for schema in schema_list:
                            if schema.child_workitem_type not in workflow_types:
                                raise ValueError(
                                    f"Child workitem type '{schema.child_workitem_type}' not found in workflow_types"
                                )
                
                # Add entries for workflow types missing in child_schema_dict
                for workflow_type in workflow_types:
                    if workflow_type not in child_schema_dict:
                        child_schema_dict[workflow_type] = None

            self.child_schema_dict = child_schema_dict

        def to_dict(self) -> Dict[str, Any]:
            """Serialize WorkflowSchema to a dictionary."""
            child_schema_serialized: Optional[Dict[str, Any]] = None
            if self.child_schema_dict is not None:
                child_schema_serialized = {
                    parent_type: (
                        None if schemas is None else [s.to_dict() for s in schemas]
                    )
                    for parent_type, schemas in self.child_schema_dict.items()
                }
            return {
                "workflow_types": self.workflow_types,
                "child_schema_dict": child_schema_serialized,
            }

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> "WorkItem.WorkflowSchema":
            """Create a WorkflowSchema instance from a dictionary."""
            workflow_types = data.get("workflow_types")
            child_schema_dict_data = data.get("child_schema_dict")
            child_schema_dict: Optional[Dict[str, List[WorkItem.ChildSchema]]] = None
            if child_schema_dict_data is not None:
                child_schema_dict = {
                    parent_type: (
                        None
                        if schemas_data is None
                        else [
                            WorkItem.ChildSchema.from_dict(schema)
                            for schema in schemas_data
                        ]
                    )
                    for parent_type, schemas_data in child_schema_dict_data.items()
                }
            return cls(workflow_types=workflow_types, child_schema_dict=child_schema_dict)

        def create_workitem(self, workitem_type: str) -> "WorkItem":
            """Instantiate a top-level :class:`WorkItem` that follows this schema.

            Args:
                workitem_type: The type name of the work-item to create.

            Returns
            -------
            WorkItem
                A new *root* work-item configured with *workflow_schema=self*.

            Raises
            ------
            ValueError
                If *workitem_type* is not declared in *workflow_types* (when
                they are defined).
            """
            if self.workflow_types is not None and workitem_type not in self.workflow_types:
                raise ValueError(
                    f"workitem_type '{workitem_type}' not found in workflow_types"
                )
            # Create the WorkItem and attach this schema so that child rules apply
            return WorkItem(workitem_type=workitem_type, workflow_schema=self)

        def to_json_file(self, path: str) -> None:
            """Write the WorkflowSchema to a JSON file at *path*."""
            with open(path, "w", encoding="utf-8") as fp:
                json.dump(self.to_dict(), fp, indent=2)

        @classmethod
        def from_json_file(cls, path: str) -> "WorkItem.WorkflowSchema":
            """Load a WorkflowSchema from a JSON file located at *path*."""
            with open(path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            return cls.from_dict(data)

    def __init__(self,
                 workitem_type: str = '',
                 parent: "WorkItem" = None,
                 data_dict: Optional[dict] = None,
                 workflow_schema: Optional["WorkItem.WorkflowSchema"] = None) -> None:
        """Initialize a new WorkItem.

        Args:
            workitem_type (str): Type of the work item.
            parent (WorkItem, optional): Parent work item.
            data_dict (dict, optional): Initial data dictionary.
            workflow_schema (WorkflowSchema, optional): Workflow schema to enforce.

        Raises:
            ValueError: If workitem_type contains spaces.
        """
        # Fix the bug: check workitem_type instead of undefined 'type'
        if ' ' in workitem_type:
            raise ValueError("workitem_type cannot contain spaces")
        if not workitem_type:
            workitem_type = self.__class__.__name__

        if data_dict is None:
            data_dict = {}

        self._workitem_type: str = workitem_type
        self._is_complete: bool = False
        self._parent: "WorkItem" = parent
        self._workflow_schema = workflow_schema

        self._data_dict = data_dict

        self._children: List["WorkItem"] = []
        self._child_pos: Dict[WorkItem, int] = {}

        if children_schema := self._get_children_schema():
            for schema in children_schema:
                for _ in range(schema.min_cardinality):
                    # Create a bare child WorkItem of the required type
                    child = WorkItem(
                        workitem_type=schema.child_workitem_type,
                        parent=self,
                        data_dict={},
                        workflow_schema=workflow_schema,
                    )
                    # Directly append without re-checking to avoid double count
                    self._child_pos[child] = len(self._children)
                    self._children.append(child)

    @property
    def workitem_type(self) -> str:
        """Get the workitem_type of the work item.

        Returns:
            str: The workitem_type of the work item.
        """
        return self._workitem_type

    @property
    def is_complete(self) -> bool:
        """Get the status of the work item.

        Returns:
            bool: True if the work item is COMPLETE.
        """
        return self._is_complete

    @is_complete.setter
    def is_complete(self, value: bool) -> None:
        """Set the status of the work item.

        Args:
            value (bool): True if the work item is COMPLETE.
        """
        self._is_complete = value

    @property
    def parent(self) -> "WorkItem":
        """Get the parent of the work item.

        Returns:
            WorkItem: the parent workitem.
        """
        return self._parent

    def _typed_children(self, workitem_type: Optional[str] = None) -> List["WorkItem"]:
        """Get children filtered by workitem_type, or all children if None."""
        return (
            [c for c in self._children if c.workitem_type == workitem_type]
            if workitem_type is not None
            else self._children
        )

    def _get_children_schema(self) -> Optional[List["WorkItem.ChildSchema"]]:
        """Get the children schema for this workitem type from the workflow schema."""
        if (self._workflow_schema and 
                self._workflow_schema.child_schema_dict):
            return self._workflow_schema.child_schema_dict.get(self._workitem_type)
        return None

    def __getitem__(self, key: str) -> Any:
        """Get a value from the data dictionary.

        Args:
            key (str): The key to retrieve.

        Returns:
            Any: The value associated with the key.

        Raises:
            KeyError: If the key doesn't exist in the data dictionary.
        """
        return self._data_dict[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set a value in the data dictionary.

        Args:
            key (str): The key to set.
            value (Any): The value to associate with the key.
        """
        self._data_dict[key] = value

    def clear_data_dict(self) -> None:
        """Clear all data from the data dictionary."""
        self._data_dict.clear()

    def add_child(self, child: "WorkItem") -> None:
        """Add a child WorkItem while enforcing the children schema.

        Raises:
            ValueError: If the child's type is not allowed or adding it would
                violate *max_cardinality*.
        """
        if children_schema := self._get_children_schema():
            schema = next(
                (s for s in children_schema
                 if s.child_workitem_type == child.workitem_type),
                None,
            )
            if schema is None:
                raise ValueError(
                    (
                        "Child type "
                        f"'{child.workitem_type}' not allowed for parent type "
                        f"'{self._workitem_type}'."
                    )
                )
            if (
                schema.max_cardinality is not None and
                self.get_child_count(child.workitem_type)
                >= schema.max_cardinality
            ): 
                raise ValueError(
                    "Cannot add more than "
                    f"{schema.max_cardinality} children of type "
                    f"'{child.workitem_type}'."
                )
        # Accept and register the child
        child._parent = self  # ensure parent pointer is correct
        self._child_pos[child] = len(self._children)
        self._children.append(child)

    def remove_child(self, child: "WorkItem") -> None:
        """Remove a child while respecting *min_cardinality* constraints."""
        if children_schema := self._get_children_schema():
            if schema := next(
                (
                    s
                    for s in children_schema
                    if s.child_workitem_type == child.workitem_type
                ),
                None,
            ):
                current = self.get_child_count(child.workitem_type)
                if current - 1 < schema.min_cardinality:
                    raise ValueError(
                        "Cannot remove child; would violate min_cardinality="
                        f"{schema.min_cardinality} for type "
                        f"'{child.workitem_type}'."
                    )
        idx = self._child_pos.pop(child)
        self._children.pop(idx)
        # rebuild positions of elements after idx
        for i in range(idx, len(self._children)):
            self._child_pos[self._children[i]] = i

    def index_of(self, child: "WorkItem") -> int:
        return self._child_pos[child]          # O(1)

    def get_child(self, index: int,
                  workitem_type: Optional[str] = None) -> Optional["WorkItem"]:
        """Return a child WorkItem by position, optionally filtered by type.

        If *workitem_type* is provided, the children list is filtered to
        include only those whose ``workitem_type`` attribute matches the
        supplied value **before** applying *index*.

        Args:
            index (int): Zero-based position within the (possibly filtered)
                list.
            workitem_type (str | None): Restrict search to children of this
                type.

        Returns:
            WorkItem: The requested child.

        Raises:
            IndexError: If *index* is negative or out of range for the
                relevant child collection.
        """
        if index < 0:
            raise IndexError("child index cannot be negative")

        if workitem_type is None:
            if index >= len(self._children):
                raise IndexError("child index out of range")
            return self._children[index]

        # type-filtered access
        filtered_children = [
            c for c in self._children if c.workitem_type == workitem_type
        ]
        if not filtered_children:
            return None
        if index >= len(filtered_children):
            raise IndexError(
                f"child index out of range for workitem_type '{workitem_type}'"
            )
        return filtered_children[index]

    def get_child_count(self, workitem_type: Optional[str] = None) -> int:
        """Return the number of children, optionally filtered by type.

        Args:
            workitem_type (str | None): If provided, count only children
                whose ``workitem_type`` matches this value.

        Returns:
            int: Number of matching child WorkItems.
        """
        if workitem_type is None:
            return len(self._children)
        return sum(
            c.workitem_type == workitem_type
            for c in self._children
        )

    def remove_all_children(self) -> None:
        """Remove all children while respecting *min_cardinality* rules."""
        if children_schema := self._get_children_schema():
            if violating := [
                s.child_workitem_type
                for s in children_schema
                if s.min_cardinality > 0
            ]:
                raise ValueError(
                    (
                        "Cannot remove all children; min_cardinality > 0 for "
                        "types: " + ", ".join(violating)
                    )
                )
        self._children.clear()
        self._child_pos.clear()

    def _get_child_relative_path(self, child_workitem: "WorkItem") -> Optional[str]:
        """Get the path component for a child workitem.

        Args:
            child_workitem: The child workitem to get the path for.

        Returns:
            str: Path component in format 'workitem_type[index]'. None if not found
        """
        children = self._typed_children(child_workitem.workitem_type)
        if child_workitem not in children:
            return None

        idx = children.index(child_workitem)
        return f"{child_workitem.workitem_type}[{idx}]"

    def get_absolute_path(self) -> str:
        """Get the absolute path to this workitem.

        Returns:
            str: Absolute path starting with '/'.
        """
        if self._parent is None:
            return "/"
        component = self._parent._get_child_relative_path(self)
        parent_path = self._parent.get_absolute_path()
        return parent_path.rstrip("/") + "/" + component

    def _get_child_workitem(self, relative_path: str) -> Optional["WorkItem"]:
        """Get a child workitem by relative path.

        Args:
            path: Path string in format 'workitem_type[index]' or 'workitem_type'.

        Returns:
            WorkItem | None: The child workitem, or None if not found.
        """
        # Remove leading '/' and split by '/'
        relative_path = relative_path.lstrip("/")
        if not relative_path:
            return self

        segments = relative_path.split("/")
        current = self

        for segment in segments:
            # Parse segment: 'workitem_type[index]' or 'workitem_type'
            match = re.match(r'(\w+)(?:\[(\d+)])?$', segment)
            if not match:
                return None

            workitem_type = match[1]
            index_str = match[2]
            index = int(index_str) if index_str else 0

            try:
                current = current.get_child(index, workitem_type)
                if current is None:
                    return None
            except IndexError:
                return None

        return current

    def _get_next_child(self, child: "WorkItem", 
                      workitem_type: Optional[str] = None) -> Optional["WorkItem"]:
        """Get the next child after the specified child.

        Args:
            child: The reference child workitem.
            workitem_type: Optional filter for child type.

        Returns:
            WorkItem | None: The next child, or None if not found.
        """
        children = self._typed_children(workitem_type)
        if child not in children:
            return None
        
        pos = children.index(child)
        next_idx = pos + 1
        return children[next_idx] if next_idx < len(children) else None

    def get_next_workitem(self, workitem_type: Optional[str] = None) -> Optional["WorkItem"]:
        """Get the first child workitem, or the next sibling if no children.

        Args:
            workitem_type: Optional filter for workitem type.

        Returns:
            WorkItem | None: The next workitem, or None if not found.
        """
        if children := self._typed_children(workitem_type):
            return children[0]

        # Look for next sibling, climbing up the tree if necessary
        current = self
        while current._parent:
            next_sibling = current._parent._get_next_child(current, workitem_type)
            if next_sibling:
                return next_sibling
            current = current._parent
        return None

    def get_workitem(self, absolute_path: str) -> Optional["WorkItem"]:
        """Locate a workitem by absolute path from the current node.

        The method climbs to the root WorkItem (where ``parent is None``) and
        then uses _get_child_workitem() to traverse the remainder of
        *absolute_path*.

        Args:
            absolute_path: Path beginning with ``/``. If ``/`` or empty, the
                root workitem is returned.

        Returns:
            WorkItem | None: The target workitem or ``None`` when not found.
        """
        # Normalize path
        absolute_path = absolute_path.strip()
        if not absolute_path:
            return None

        # Climb to root
        root: WorkItem = self
        while root._parent is not None:
            root = root._parent

        if absolute_path == "/":
            return root

        # Delegate to root instance with relative path component
        return root._get_child_workitem(absolute_path.lstrip("/"))

    def _to_dict(self) -> Dict[str, Any]:
        """Convert the WorkItem to a dictionary for serialization.

        Returns:
            dict: Dictionary representation of the WorkItem.
        """
        return {
            'workitem_type': self.workitem_type,
            'is_complete': self.is_complete,
            'data_dict': self._data_dict.copy()
        }

    @classmethod
    def _from_dict(cls: Type['WorkItem'], data: Dict[str, Any]) -> 'WorkItem':
        """Create a WorkItem from a dictionary.

        Args:
            data (dict): Dictionary containing WorkItem attributes.

        Returns:
            WorkItem: A new WorkItem instance.
        """
        workitem = cls(
            workitem_type=data.get('workitem_type', ''),
            data_dict=data.get('data_dict', {})
        )
        workitem.is_complete = data.get('is_complete', False)
        return workitem