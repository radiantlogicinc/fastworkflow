"""TodoItem class for representing individual todo items in the todo list manager.

This module defines the TodoItem class, which represents a single todo item with
attributes like id, description, assign_to, and status.
"""

from typing import Dict, Any, Type, Optional, Union

class TodoItem:
    """Represents a single todo item in the todo list.

    Attributes:
        id (int): Unique identifier for the todo item.
        description (str): Description of the todo item.
        assign_to (str): Person assigned to the todo item.
        status (str): Current status of the todo item ('INCOMPLETE' or 'COMPLETE').
        parent: Optional reference to containing TodoList or TodoListManager.
    """

    INCOMPLETE: str = 'INCOMPLETE'
    COMPLETE: str = 'COMPLETE'

    def __init__(self, id: int, description: str, assign_to: str = "", status: str = INCOMPLETE) -> None:
        """Initialize a new TodoItem.

        Args:
            id (int): Unique identifier for the todo item.
            description (str): Description of the todo item.
            assign_to (str): Person assigned to the todo item.
            status (str, optional): Current status of the todo item. Defaults to INCOMPLETE.

        Raises:
            TypeError: If id is not an integer.
            ValueError: If description or assign_to is empty, or if status is invalid.
        """
        if not isinstance(id, int):
            raise TypeError("id must be an integer")
        self.id: int = id
        self.description = description  # Will use setter for validation
        self.assign_to = assign_to      # Will use setter for validation
        self.status = status            # Will use setter for validation
        self.parent = None

    @property
    def description(self) -> str:
        """Get the description of the todo item.

        Returns:
            str: The description of the todo item.
        """
        return self._description

    @description.setter
    def description(self, value: str) -> None:
        """Set the description of the todo item.

        Args:
            value (str): The new description.

        Raises:
            ValueError: If the description is empty.
            TypeError: If the description is not a string.
        """
        if not isinstance(value, str):
            raise TypeError("Description must be a string")
        if not value.strip():
            raise ValueError("Description cannot be empty")
        self._description = value

    @property
    def assign_to(self) -> str:
        """Get the person assigned to the todo item.

        Returns:
            str: The person assigned to the todo item.
        """
        return self._assign_to

    @assign_to.setter
    def assign_to(self, value: str) -> None:
        """Set the person assigned to the todo item.

        Args:
            value (str): The new assignee.

        Raises:
            TypeError: If the assignee is not a string.
        """
        if not isinstance(value, str):
            raise TypeError("Assignee must be a string")
        self._assign_to = value

    @property
    def status(self) -> str:
        """Get the status of the todo item.

        Returns:
            str: The status of the todo item (COMPLETE or INCOMPLETE).
        """
        return self._status

    @status.setter
    def status(self, value: str) -> None:
        """Set the status of the todo item.

        Args:
            value (str): The new status (must be either COMPLETE or INCOMPLETE).

        Raises:
            ValueError: If the status is not one of the allowed values.
        """
        allowed_statuses = [self.COMPLETE, self.INCOMPLETE]
        if value not in allowed_statuses:
            raise ValueError(f"Status must be one of {allowed_statuses}")
        self._status = value

    def _to_dict(self) -> Dict[str, Any]:
        """Convert the TodoItem to a dictionary for serialization.

        Returns:
            dict: Dictionary representation of the TodoItem.
        """
        return {
            'id': self.id,
            'description': self.description,
            'assign_to': self.assign_to,
            'status': self.status
        }

    @classmethod
    def _from_dict(cls: Type['TodoItem'], data: Dict[str, Any]) -> 'TodoItem':
        """Create a TodoItem from a dictionary.

        Args:
            data (dict): Dictionary containing TodoItem attributes.

        Returns:
            TodoItem: A new TodoItem instance.
        """
        return cls(
            id=data['id'],
            description=data['description'],
            assign_to=data['assign_to'],
            status=data['status']
        ) 