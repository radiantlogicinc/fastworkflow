from typing import List, Optional, Union, Dict, Any, Set
from .todo_item import TodoItem

# If needed, define a base class for the composite pattern (optional, as TodoItem is already a base)
# For now, we use TodoItem as the base, and TodoList will inherit from it.

class TodoList(TodoItem):
    """A TodoList that can contain both TodoItems and other TodoLists.

    This class implements the Composite design pattern, allowing for
    hierarchical organization of todo items.

    Attributes:
        id (int): Unique identifier for this TodoList
        description (str): Description of this TodoList
        assign_to (str): Person assigned to this TodoList
        status (str): Status of this TodoList (COMPLETE or INCOMPLETE)
        children (List[Union[TodoItem, 'TodoList']]): List of child items
        parent: Optional reference to containing TodoList or TodoListManager
    """
    def __init__(self, id: int, description: str, assign_to: str = "", status: str = TodoItem.INCOMPLETE) -> None:
        super().__init__(id, description, assign_to, status)
        self.children: List[Union[TodoItem, 'TodoList']] = []

    def _generate_child_id(self) -> int:
        """Generate a unique child ID within this TodoList."""
        return max(child.id for child in self.children) + 1 if self.children else 1

    def add_child_todoitem(self, description: str, assign_to: str = "", status: str = TodoItem.INCOMPLETE) -> TodoItem:
        """Add a new TodoItem as a child to this TodoList.
        Args:
            description (str): Description of the todo item.
            assign_to (str): Person assigned to the todo item.
            status (str): Status of the todo item.
        Returns:
            TodoItem: The newly created child TodoItem.
        Raises:
            ValueError: If a child with the generated id already exists.
        """
        new_id = self._generate_child_id()
        if any(child.id == new_id for child in self.children):
            raise ValueError(f"A child with id {new_id} already exists.")
        child = TodoItem(new_id, description, assign_to, status)
        child.parent = self
        self.children.append(child)
        return child

    def add_child_todolist(self, description: str, assign_to: str, status: str = TodoItem.INCOMPLETE) -> 'TodoList':
        """Add a new TodoList as a child to this TodoList.
        Args:
            description (str): Description of the todo list.
            assign_to (str): Person assigned to the todo list.
            status (str): Status of the todo list.
        Returns:
            TodoList: The newly created child TodoList.
        Raises:
            ValueError: If a child with the generated id already exists or if trying to add self as a child.
        """
        new_id = self._generate_child_id()
        if any(child.id == new_id for child in self.children):
            raise ValueError(f"A child with id {new_id} already exists.")
        child = TodoList(new_id, description, assign_to, status)
        if child is self:
            raise ValueError("Cannot add a TodoList as a child of itself")
        child.parent = self
        self.children.append(child)
        return child

    def remove_child_by_id(self, child_id: int) -> bool:
        """Remove a child by its ID.
        Args:
            child_id: The ID of the child to remove
        Returns:
            bool: True if the child was removed, False if not found
        """
        for child in self.children:
            if child.id == child_id:
                child.parent = None
                self.children.remove(child)
                return True
        return False

    def get_child_by_id(self, child_id: int) -> Optional[Union[TodoItem, 'TodoList']]:
        """Get a child by its ID.
        Args:
            child_id: The ID of the child to find
        Returns:
            The child if found, None otherwise
        """
        return next((child for child in self.children if child.id == child_id), None)

    def get_all_children(self) -> List[Union[TodoItem, 'TodoList']]:
        """Get all children of this TodoList.
        Returns:
            List of all children
        """
        return self.children

    def update_status(self) -> None:
        """Update the status of this TodoList based on its children.
        If all children are complete, this TodoList is complete. Otherwise, it's incomplete.
        """
        if not self.children:
            return
        if all(child.status == TodoItem.COMPLETE for child in self.children):
            self.status = TodoItem.COMPLETE
        else:
            self.status = TodoItem.INCOMPLETE

    def mark_completed(self) -> None:
        """Mark this TodoList and all children as complete."""
        self.status = TodoItem.COMPLETE
        for child in self.children:
            if isinstance(child, TodoList):
                child.mark_completed()
            else:
                child.status = TodoItem.COMPLETE

    def mark_pending(self) -> None:
        """Mark this TodoList and all children as incomplete."""
        self.status = TodoItem.INCOMPLETE
        for child in self.children:
            if isinstance(child, TodoList):
                child.mark_pending()
            else:
                child.status = TodoItem.INCOMPLETE 
                
    def _to_dict(self, visited: Set[int] = None) -> Dict[str, Any]:
        """Convert this TodoList to a dictionary for serialization.
        
        Args:
            visited: Set of object IDs that have already been processed to prevent circular references
            
        Returns:
            Dict[str, Any]: Dictionary representation of this TodoList including all children
        
        Note:
            This method handles the entire hierarchy by recursively serializing all children.
            It also handles circular references by tracking visited objects.
        """
        # Initialize visited set if not provided
        if visited is None:
            visited = set()
            
        # Check if this object has already been visited (to handle circular references)
        if self.id in visited:
            # Return a reference instead of the full object to break the circular reference
            return {
                'id': self.id,
                'type': 'TodoListRef',  # Mark as a reference
                'reference': True
            }
            
        # Add this object to the visited set
        visited.add(self.id)
        
        # Get the base attributes from parent class
        result = {
            'id': self.id,
            'description': self.description,
            'assign_to': self.assign_to,
            'status': self.status,
            'type': 'TodoList',  # Type discriminator to distinguish from TodoItem
            'children': []
        }
        
        # Recursively serialize all children
        for child in self.children:
            if isinstance(child, TodoList):
                # Pass the visited set to detect circular references
                result['children'].append(child._to_dict(visited))
            else:
                # For TodoItem, call its _to_dict method and add type discriminator
                child_dict = child._to_dict()
                child_dict['type'] = 'TodoItem'
                result['children'].append(child_dict)
                
        return result
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any], id_to_obj_map: Dict[int, 'TodoList'] = None) -> 'TodoList':
        """Create a TodoList instance from a dictionary representation.
        
        Args:
            data (Dict[str, Any]): Dictionary containing TodoList data
            id_to_obj_map: Dictionary mapping object IDs to already created TodoList instances
                           (used to handle circular references)
            
        Returns:
            TodoList: A new TodoList instance with all its children
            
        Raises:
            ValueError: If the data dictionary is malformed or missing required fields
            TypeError: If the data types are incorrect
        """
        try:
            # Initialize the map if not provided
            if id_to_obj_map is None:
                id_to_obj_map = {}

            # Check if this is a reference to an already created object
            if data.get('type') == 'TodoListRef' and data.get('reference') is True:
                # Look up the object in the map
                obj_id = data.get('id')
                if obj_id in id_to_obj_map:
                    return id_to_obj_map[obj_id]
                else:
                    raise ValueError(f"Reference to non-existent TodoList with ID: {obj_id}")

            # Validate required fields
            required_fields = ['id', 'description', 'assign_to', 'status']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")

            # Create the TodoList instance
            todo_list = cls(
                id=data['id'],
                description=data['description'],
                assign_to=data['assign_to'],
                status=data['status']
            )

            # Add to map before processing children to handle circular references
            id_to_obj_map[todo_list.id] = todo_list

            # Process children if present
            for child_data in data.get('children', []):
                # Check the type discriminator
                child_type = child_data.get('type', 'TodoItem')

                if child_type in ['TodoList', 'TodoListRef']:
                    # Recursively create TodoList, passing the map
                    child = cls._from_dict(child_data, id_to_obj_map)
                else:
                    # Create TodoItem
                    child = TodoItem._from_dict(child_data)
                todo_list.add_child(child)
            return todo_list

        except (KeyError, TypeError, ValueError) as e:
            # Enhance the error message with context
            raise ValueError(f"Error deserializing TodoList: {str(e)}") from e 