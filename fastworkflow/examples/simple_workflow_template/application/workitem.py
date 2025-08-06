"""WorkItem class for representing individual work items.

This module defines the WorkItem class, which represents a single work item
with attributes like id, type, and status.
"""

from typing import Dict, Any, Type, Optional, List
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
            workflow_types: Optional[Dict[str, str]] = None,
            child_schema_dict: Optional[Dict[str, Dict[str, Dict[str, int]]]] = None,
            parents_dict: Optional[Dict[str, Optional[Dict[str, List[str]]]]] = None
        ) -> None:
            self.workflow_types = workflow_types
            self.parents_dict = parents_dict or {}

            # Validate workflow_types and child_schema_dict
            if workflow_types and child_schema_dict:
                workflow_type_keys = list(workflow_types.keys())
                # Check that all child_schema_dict keys are in workflow_types
                for key in child_schema_dict:
                    if key not in workflow_type_keys:
                        raise ValueError(f"Child schema key '{key}' not found in workflow_types")
                
                # Check that all child workitem types are in workflow_types
                for child_schemas in child_schema_dict.values():
                    if child_schemas is not None:
                        for child_type in child_schemas.keys():
                            if child_type not in workflow_type_keys:
                                raise ValueError(
                                    f"Child workitem type '{child_type}' not found in workflow_types"
                                )
                
                # Add entries for workflow types missing in child_schema_dict
                for workflow_type in workflow_type_keys:
                    if workflow_type not in child_schema_dict:
                        child_schema_dict[workflow_type] = None

            self.child_schema_dict = child_schema_dict

            # Validate parents_dict
            if workflow_types and self.parents_dict:
                self._validate_parents_dict(workflow_types)

        def _validate_parents_dict(self, workflow_types: Dict[str, str]) -> None:
            """Validate the parents_dict against workflow_types.
            
            Args:
                workflow_types: Dictionary of workflow type names to descriptions.
                
            Raises:
                ValueError: If parents_dict contains invalid types or parent references.
            """
            workflow_type_keys = list(workflow_types.keys())
            
            # Check that all parents_dict keys are in workflow_types
            for key in self.parents_dict:
                if key not in workflow_type_keys:
                    raise ValueError(f"Parents dict key '{key}' not found in workflow_types")
            
            # Check that all parent types referenced are in workflow_types
            for workitem_type, parent_config in self.parents_dict.items():
                if parent_config is not None:
                    allowed_parents = parent_config.get("parent", [])
                    for parent_type in allowed_parents:
                        if parent_type not in workflow_type_keys:
                            raise ValueError(
                                f"Parent type '{parent_type}' for '{workitem_type}' not found in workflow_types"
                            )

        def _validate_parent_for_workitem_type(self, workitem_type: str, parent: Optional["WorkItem"]) -> None:
            """Validate that the parent is allowed for the given workitem type.
            
            Args:
                workitem_type: The type of workitem being created.
                parent: The parent WorkItem, or None for root-level items.
                
            Raises:
                ValueError: If parent validation fails according to parents_dict.
            """
            allowed_parents = self.parents_dict.get(workitem_type, None)
            
            if allowed_parents is None:
                # Root-only type - no parent allowed
                if parent is not None:
                    raise ValueError(f"{workitem_type} is root-level only; parent given.")
            else:
                # Parent required - check if provided and valid
                allowed_parent_types = allowed_parents.get("parent", [])
                if parent is None:
                    raise ValueError(f"{workitem_type} requires a parent of type {allowed_parent_types}.")
                if parent.type not in allowed_parent_types:
                    raise ValueError(f"Invalid parent type '{parent.type}' for {workitem_type}; allowed: {allowed_parent_types}.")

        def to_dict(self) -> Dict[str, Any]:
            """Serialize WorkflowSchema to a dictionary."""
            child_schema_serialized: Optional[Dict[str, Any]] = None
            if self.child_schema_dict is not None:
                child_schema_serialized = {
                    parent_type: (
                        None if schemas is None else schemas
                    )
                    for parent_type, schemas in self.child_schema_dict.items()
                }
            return {
                "workflow_types": self.workflow_types,
                "child_schema_dict": child_schema_serialized,
                "parents_dict": self.parents_dict,
            }

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> "WorkItem.WorkflowSchema":
            """Create a WorkflowSchema instance from a dictionary."""
            workflow_types = data.get("workflow_types")
            child_schema_dict_data = data.get("child_schema_dict")
            child_schema_dict: Optional[Dict[str, Dict[str, Dict[str, int]]]] = None
            if child_schema_dict_data is not None:
                child_schema_dict = {
                    parent_type: (
                        None
                        if schemas_data is None
                        else schemas_data
                    )
                    for parent_type, schemas_data in child_schema_dict_data.items()
                }
            parents_dict = data.get("parents_dict")
            return cls(
                workflow_types=workflow_types, 
                child_schema_dict=child_schema_dict,
                parents_dict=parents_dict
            )

        def create_workitem(self, 
                            workitem_type: str, 
                            parent: Optional["WorkItem"] = None,
                            id: Optional[str] = None,
                            data_dict: Optional[dict] = None
                            ) -> "WorkItem":
            """Instantiate a :class:`WorkItem` that follows this schema.

            Args:
                workitem_type: The type name of the work-item to create.
                parent: Optional parent WorkItem. If provided, must comply with
                    the parents_dict constraints.

            Returns
            -------
            WorkItem
                A new work-item configured with *workflow_schema=self*.

            Raises
            ------
            ValueError
                If *workitem_type* is not declared in *workflow_types* (when
                they are defined).
                If parent validation fails according to parents_dict.
                If parent belongs to a different WorkflowSchema.
            """
            if self.workflow_types is not None and workitem_type not in self.workflow_types:
                raise ValueError(
                    f"workitem_type '{workitem_type}' not found in workflow_types"
                )
            
            # Validate parent according to parents_dict
            if self.parents_dict:
                self._validate_parent_for_workitem_type(workitem_type, parent)
            
            # Ensure parent belongs to the same schema
            if parent and parent._workflow_schema is not self:
                raise ValueError("Parent belongs to a different WorkflowSchema.")
            
            # Create the WorkItem and attach this schema so that child rules apply
            item = WorkItem(
                workitem_type=workitem_type, 
                parent=parent, 
                workflow_schema=self,
                id=id,
                data_dict=data_dict)
            
            # Add to parent if provided
            if parent:
                parent.add_child(item)
            
            return item

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

        def __getitem__(self, key: str) -> Dict:
            """Get workflow type information by key.
            
            Args:
                key: The workflow type name to look up.
                
            Returns:
                A dictionary containing:
                - description: The description of the workflow type
                - child_schema: A dictionary mapping child workitem types to their cardinality information
                
            Raises:
                KeyError: If the workflow type is not found in the schema.
            """
            if self.workflow_types is None or key not in self.workflow_types:
                raise KeyError(f"Workflow type '{key}' not found in schema")
            
            description = self.workflow_types[key]
            child_schema = {}
            
            if self.child_schema_dict is not None and key in self.child_schema_dict:
                schemas = self.child_schema_dict[key]
                if schemas is not None:
                    child_schema = schemas.copy()
            
            return {
                "description": description,
                "child_schema": child_schema
            }

    def __init__(self,
                 workitem_type: str = '',
                 parent: "WorkItem" = None,
                 data_dict: Optional[dict] = None,
                 workflow_schema: Optional["WorkItem.WorkflowSchema"] = None,
                 id: Optional[str] = None) -> None:
        """Initialize a new WorkItem.

        Args:
            workitem_type (str): Type of the work item.
            parent (WorkItem, optional): Parent work item.
            data_dict (dict, optional): Initial data dictionary.
            workflow_schema (WorkflowSchema, optional): Workflow schema to enforce.
            id (str, optional): Unique identifier for the work item among its siblings.

        Raises:
            ValueError: If workitem_type contains spaces.
            ValueError: If id is not unique among siblings of the same type.
        """
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
        self._id: Optional[str] = id
        
        # Check id uniqueness among siblings of the same type
        if parent is not None and id is not None:
            siblings = parent._typed_children(workitem_type)
            for sibling in siblings:
                if hasattr(sibling, '_id') and sibling._id == id:
                    raise ValueError(f"Work item id '{id}' is not unique among siblings of type '{workitem_type}'")

        self._data_dict = data_dict

        self._children: List["WorkItem"] = []
        self._child_pos: Dict[WorkItem, int] = {}

        if children_schema := self._get_children_schema():
            for child_type, schema in children_schema.items():
                min_cardinality = schema.get("min_cardinality", 0)
                for _ in range(min_cardinality):
                    # Create a bare child WorkItem of the required type
                    child = WorkItem(
                        workitem_type=child_type,
                        parent=self,
                        data_dict={},
                        workflow_schema=workflow_schema,
                    )
                    # Directly append without re-checking to avoid double count
                    self._child_pos[child] = len(self._children)
                    self._children.append(child)

    @property
    def type(self) -> str:
        """Get the workitem_type of the work item.

        Returns:
            str: The workitem_type of the work item.
        """
        return self._workitem_type

    @property
    def schema(self) -> Optional[Dict[str, Any]]:
        """Get the schema for this work item type from the workflow schema.

        Returns:
            Dict[str, Any] | None: A dictionary containing the schema information for this work item type,
            including description and child schema, or None if not found.
        """
        if self._workflow_schema:
            try:
                return self._workflow_schema[self._workitem_type]
            except KeyError:
                return None
        return None

    @property
    def is_complete(self) -> bool:
        """Get the status of the work item.

        Returns:
            bool: True if the work item is COMPLETE.
        """
        # If this work item is marked as incomplete, return False
        return self._is_complete

    @is_complete.setter
    def is_complete(self, value: bool):
        """Set the status of the work item as well as ancestor workitem's statuses"""
        if not self._children:
            # Leaf item - can be set directly
            self._is_complete = value
            # Always propagate leaf changes to parent
            if self.parent:
                self.parent.is_complete = value
        else:
            # Non-leaf item - calculate based on children
            self._recalculate_completion_status()

    def _recalculate_completion_status(self):
        """Recalculate completion status based on children and propagate to parent."""
        if not self._children:
            # Leaf items don't need recalculation
            return
        
        old_complete = self._is_complete
        self._is_complete = all(child.is_complete for child in self._children)
        
        # Propagate if completion status changed
        if self._is_complete != old_complete and self.parent:
            self.parent.is_complete = self._is_complete

    @property
    def parent(self) -> "WorkItem":
        """Get the parent of the work item.

        Returns:
            WorkItem: the parent workitem.
        """
        return self._parent

    def _typed_children(self, workitem_type: Optional[str] = None, is_complete: Optional[bool] = None) -> List["WorkItem"]:
        """Get children filtered by workitem_type and completion status, or all children if None."""
        filtered_children = self._children
        
        if workitem_type is not None:
            filtered_children = [c for c in filtered_children if c.type.lower() == workitem_type.lower()]
        
        if is_complete is not None:
            filtered_children = [c for c in filtered_children if c.is_complete == is_complete]
        
        return filtered_children

    def _get_children_schema(self) -> Optional[Dict[str, Dict[str, int]]]:
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
            ValueError: If the child's id is not unique among siblings of the same type.
        """
        if children_schema := self._get_children_schema():
            if child.type not in children_schema:
                raise ValueError(
                    (
                        "Child type "
                        f"'{child.type}' not allowed for parent type "
                        f"'{self._workitem_type}'."
                    )
                )
            schema = children_schema[child.type]
            max_cardinality = schema.get("max_cardinality")
            if (
                max_cardinality is not None and
                self.get_child_count(child.type)
                >= max_cardinality
            ): 
                raise ValueError(
                    "Cannot add more than "
                    f"{max_cardinality} children of type "
                    f"'{child.type}'."
                )
        # Check for duplicate ID
        if hasattr(child, '_id') and child._id is not None:
            siblings = self._typed_children(child.type)
            for sibling in siblings:
                if hasattr(sibling, '_id') and sibling._id == child._id:
                    raise ValueError(f"Work item id '{child._id}' is not unique among siblings of type '{child.type}'")
        
        # Accept and register the child
        child._parent = self  # ensure parent pointer is correct
        self._child_pos[child] = len(self._children)
        self._children.append(child)
        self._recalculate_completion_status()

    def remove_child(self, child: "WorkItem") -> None:
        """Remove a child while respecting *min_cardinality* constraints."""
        if children_schema := self._get_children_schema():
            if child.type in children_schema:
                schema = children_schema[child.type]
                current = self.get_child_count(child.type)
                min_cardinality = schema.get("min_cardinality", 0)
                if current - 1 < min_cardinality:
                    raise ValueError(
                        "Cannot remove child; would violate min_cardinality="
                        f"{min_cardinality} for type "
                        f"'{child.type}'."
                    )
        idx = self._child_pos.pop(child)
        self._children.pop(idx)
        # rebuild positions of elements after idx
        for i in range(idx, len(self._children)):
            self._child_pos[self._children[i]] = i
        self._recalculate_completion_status()

    def index_of(self, child: "WorkItem") -> int:
        return self._child_pos[child]          # O(1)

    def get_child(self, index: int,
                  workitem_type: Optional[str] = None, is_complete: Optional[bool] = None) -> Optional["WorkItem"]:
        """Return a child WorkItem by position, optionally filtered by type and completion status.

        If *workitem_type* or *is_complete* is provided, the children list is filtered to
        include only those that match the criteria **before** applying *index*.

        Args:
            index (int): Zero-based position within the (possibly filtered)
                list.
            workitem_type (str | None): Restrict search to children of this
                type.
            is_complete (bool | None): Restrict search to children with this
                completion status.

        Returns:
            WorkItem: The requested child.

        Raises:
            IndexError: If *index* is negative or out of range for the
                relevant child collection.
        """
        if index < 0:
            raise IndexError("child index cannot be negative")

        if workitem_type is None and is_complete is None:
            if index >= len(self._children):
                raise IndexError("child index out of range")
            return self._children[index]

        # filtered access
        filtered_children = self._typed_children(workitem_type, is_complete)
        if not filtered_children:
            return None
        if index >= len(filtered_children):
            raise IndexError(
                f"child index out of range for filters: workitem_type='{workitem_type}', is_complete={is_complete}"
            )
        return filtered_children[index]

    def get_child_count(self, workitem_type: Optional[str] = None, is_complete: Optional[bool] = None) -> int:
        """Return the number of children, optionally filtered by type and completion status.

        Args:
            workitem_type (str | None): If provided, count only children
                whose ``workitem_type`` matches this value.
            is_complete (bool | None): If provided, count only children
                whose completion status matches this value.

        Returns:
            int: Number of matching child WorkItems.
        """
        return len(self._typed_children(workitem_type, is_complete))

    def remove_all_children(self) -> None:
        """Remove all children while respecting *min_cardinality* rules."""
        if children_schema := self._get_children_schema():
            if violating := [
                child_type
                for child_type, schema in children_schema.items()
                if schema and schema.get("min_cardinality", 0) > 0
            ]:
                raise ValueError(
                    (
                        "Cannot remove all children; min_cardinality > 0 for "
                        "types: " + ", ".join(violating)
                    )
                )
        self._children.clear()
        self._child_pos.clear()
        self._recalculate_completion_status()

    @property
    def id(self) -> Optional[str]:
        """Get the id of the work item.

        Returns:
            Optional[str]: The id of the work item, or None if not set.
        """
        return self._id

    def _get_child_relative_path(self, child_workitem: "WorkItem") -> Optional[str]:
        """Get the path component for a child workitem.

        Args:
            child_workitem: The child workitem to get the path for.

        Returns:
            str: Path component in format 'workitem_type[id=<id>]' if id is set,
                 or 'workitem_type[index=<idx>]' otherwise. None if not found.
        """
        children = self._typed_children(child_workitem.type)
        if child_workitem not in children:
            return None

        if child_workitem._id is not None:
            return f"{child_workitem.type}[id={child_workitem._id}]"
            
        idx = children.index(child_workitem)
        return f"{child_workitem.type}[index={idx}]"

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
            path: Path string in format 'workitem_type[index=<idx>]', 'workitem_type[id=<id>]', or 'workitem_type'.

        Returns:
            WorkItem | None: The child workitem, or None if not found.
        """
        # Remove leading and trailing '/' and replace '//' by '/'
        relative_path = relative_path.strip("/").replace('//', '/')
        if not relative_path:
            return self

        segments = relative_path.split("/")
        current = self

        for segment in segments:
            # Parse segment: 'workitem_type[index=<idx>]', 'workitem_type[id=<id>]', or 'workitem_type'
            match_index = re.match(r'(\w+)\[index=(\d+)]$', segment)
            match_id = re.match(r'(\w+)\[id=([^\]]+)]$', segment)
            match_simple = re.match(r'(\w+)$', segment)

            if match_index:
                # Format: workitem_type[index=<idx>]
                workitem_type = match_index[1]
                index = int(match_index[2])

                try:
                    current = current.get_child(index, workitem_type)
                    if current is None:
                        return None
                except IndexError:
                    return None

            elif match_id:
                # Format: workitem_type[id=<id>]
                workitem_type = match_id[1]
                workitem_id = match_id[2]

                # Find child with matching id
                found = False
                children = current._typed_children(workitem_type)
                for child in children:
                    if child._id == workitem_id:
                        current = child
                        found = True
                        break

                if not found:
                    return None

            elif match_simple:
                # Format: workitem_type (use index 0)
                workitem_type = match_simple[1]

                try:
                    current = current.get_child(0, workitem_type)
                    if current is None:
                        return None
                except IndexError:
                    return None

            else:
                return None

        return current

    def find_child_workitems(self, relative_matching_path: str) -> List["WorkItem"]:
        """Find *immediate* child workitems that satisfy a type + data-field query.

        The *relative_matching_path* must be of the form::

            "WorkItemType[field=value]"

        where:
            * **WorkItemType** – child work-item type to filter by.
            * **field**          – any key that might exist inside the child's ``_data_dict``.
            * **value**          – target value (string comparison).

        Behaviour:
            1.  Exact match on ``field`` → return those children.
            2.  If no exact match, perform a fuzzy search on the list of
                existing values for that ``field`` using
                :py:meth:`fastworkflow.utils.signatures.DatabaseValidator.fuzzy_match`.
            3.  Return children whose field value equals the fuzzy result(s).

        Notes
        -----
        • Search scope is *only* the direct children of *self* – no recursion.
        • Returns an empty list when nothing matches.

        Raises
        ------
        ValueError
            If *relative_matching_path* does **not** comply with the required
            ``Type[field=value]`` syntax.
        """
        # Lazy import to avoid circulars / heavy deps at module load time.
        from fastworkflow.utils.signatures import DatabaseValidator  # noqa: WPS433 – local import is intentional

        # Accept leading '/' for parity with other helpers
        query = relative_matching_path.lstrip("/")
        pattern = re.compile(r"^(\w+)\[(\w+)=([^\]]*)]$")
        match = pattern.match(query)
        if not match:
            raise ValueError("relative_matching_path must be of the form 'Type[field=value]'")

        workitem_type, field, target_value = match.groups()

        # 1. Filter by type first (O(N) over immediate children)
        candidates = self._typed_children(workitem_type)
        if not candidates:
            return []

        if exact_matches := [
            c
            for c in candidates
            if str(c._data_dict.get(field, "")) == target_value
        ]:
            return exact_matches

        # 3. Fuzzy matching on existing values for this field among candidates
        key_values = [str(c._data_dict.get(field, "")) for c in candidates]
        is_unique_match, single_match, multiple_matches = DatabaseValidator.fuzzy_match(
            target_value, key_values
        )

        matched_values: List[str] = []
        if is_unique_match and single_match is not None:
            matched_values = [single_match]
        elif multiple_matches:
            matched_values = multiple_matches
        else:
            return []  # No fuzzy matches

        return [
            c for c in candidates if str(c._data_dict.get(field, "")) in matched_values
        ]

    def _get_next_child(self, child: "WorkItem", 
                      workitem_type: Optional[str] = None, is_complete: Optional[bool] = None) -> Optional["WorkItem"]:
        """Get the next child after the specified child.

        Args:
            child: The reference child workitem.
            workitem_type: Optional filter for child type.
            is_complete: Optional filter for completion status.

        Returns:
            WorkItem | None: The next child, or None if not found.
        """
        children = self._typed_children(workitem_type, is_complete)
        if child not in children:
            return None
        
        pos = children.index(child)
        next_idx = pos + 1
        return children[next_idx] if next_idx < len(children) else None

    def _get_previous_child(self, child: "WorkItem", 
                          workitem_type: Optional[str] = None, is_complete: Optional[bool] = None) -> Optional["WorkItem"]:
        """Get the previous child before the specified child.

        Args:
            child: The reference child workitem.
            workitem_type: Optional filter for child type.
            is_complete: Optional filter for completion status.

        Returns:
            WorkItem | None: The previous child, or None if not found.
        """
        children = self._typed_children(workitem_type, is_complete)
        if child not in children:
            return None
        
        pos = children.index(child)
        prev_idx = pos - 1
        return children[prev_idx] if prev_idx >= 0 else None

    def get_next_workitem(self, workitem_type: Optional[str] = None, is_complete: Optional[bool] = None) -> Optional["WorkItem"]:
        """Get the first child workitem, or the next sibling if no children.

        Args:
            workitem_type: Optional filter for workitem type.
            is_complete: Optional filter for completion status.

        Returns:
            WorkItem | None: The next workitem, or None if not found.
        """
        if children := self._typed_children(workitem_type, is_complete):
            return children[0]

        # Look for next sibling, climbing up the tree if necessary
        current = self
        while current._parent:
            if next_sibling := current._parent._get_next_child(
                current, workitem_type, is_complete
            ):
                return next_sibling
            current = current._parent
        return None

    def get_previous_workitem(self, workitem_type: Optional[str] = None, is_complete: Optional[bool] = None) -> Optional["WorkItem"]:
        """Get the previous sibling workitem, or the parent if no previous sibling.

        Args:
            workitem_type: Optional filter for workitem type.
            is_complete: Optional filter for completion status.

        Returns:
            WorkItem | None: The previous workitem, or None if not found.
        """
        # Look for previous sibling, climbing up the tree if necessary
        current = self
        while current._parent:
            if prev_sibling := current._parent._get_previous_child(
                current, workitem_type, is_complete
            ):
                return prev_sibling
            # If no previous sibling found, return the parent
            return current._parent
        return None

    def get_first_workitem(self, workitem_type: Optional[str] = None, is_complete: Optional[bool] = None) -> "WorkItem":
        """Get the first sibling workitem, or the current workitem if no siblings.

        Args:
            workitem_type: Optional filter for workitem type.
            is_complete: Optional filter for completion status.

        Returns:
            WorkItem: The first sibling workitem, or the current workitem if no siblings.
        """
        if self._parent:
            if siblings := self._parent._typed_children(workitem_type, is_complete):
                return siblings[0]
        return self

    def get_last_workitem(self, workitem_type: Optional[str] = None, is_complete: Optional[bool] = None) -> "WorkItem":
        """Get the last sibling workitem, or the current workitem if no siblings.

        Args:
            workitem_type: Optional filter for workitem type.
            is_complete: Optional filter for completion status.

        Returns:
            WorkItem: The last sibling workitem, or the current workitem if no siblings.
        """
        if self._parent:
            if siblings := self._parent._typed_children(workitem_type, is_complete):
                return siblings[-1]
        return self

    def get_status(self) -> Dict[str, Any]:
        """Get the completion status of this work item and its children.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'is_complete': bool - completion status of this work item
                - For each child work item type: 'type_name': str - format 'completed_count/total_count'
        """
        status = {
            'is_complete': self.is_complete
        }
        
        # Group children by type and count completion status
        children_by_type = {}
        for child in self._children:
            child_type = child.type
            if child_type not in children_by_type:
                children_by_type[child_type] = {'total': 0, 'completed': 0}
            children_by_type[child_type]['total'] += 1
            if child.is_complete:
                children_by_type[child_type]['completed'] += 1
        
        # Add status for each child type
        for child_type, counts in children_by_type.items():
            status[child_type] = f"{counts['completed']}/{counts['total']} completed"
        
        return status

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
        result = {
            'workitem_type': self.type,
            'is_complete': self.is_complete,
            'data_dict': self._data_dict.copy(),
            'children': [child._to_dict() for child in self._children]
        }
        
        # Include id if it exists
        if self._id is not None:
            result['id'] = self._id
            
        return result

    @classmethod
    def _from_dict(cls: Type['WorkItem'], data: Dict[str, Any], 
                   workflow_schema: Optional["WorkItem.WorkflowSchema"] = None) -> 'WorkItem':
        """Create a WorkItem from a dictionary.

        Args:
            data (dict): Dictionary containing WorkItem attributes.
            workflow_schema (WorkflowSchema, optional): Workflow schema to enforce.

        Returns:
            WorkItem: A new WorkItem instance.
        """
        workitem = cls(
            workitem_type=data['workitem_type'],
            data_dict=data.get('data_dict', {}),
            workflow_schema=workflow_schema,
            id=data.get('id')
        )
        
        # Create all children first before setting is_complete
        # to avoid incorrect propagation during deserialization
        children_data = data.get('children', [])
        
        # Store the is_complete value for later
        is_complete_value = data.get('is_complete', False)
        
        # Recursively create children
        for child_data in children_data:
            child = cls._from_dict(child_data, workflow_schema)
            workitem.add_child(child)
        
        # Now set is_complete after all children are added
        # If this is a leaf node, set directly; otherwise, set via the property
        if is_complete_value:
            workitem.is_complete = True
            
        return workitem
