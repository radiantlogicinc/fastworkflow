"""TodoListManager class for managing a collection of todo items with JSON persistence.

This module defines the TodoListManager class, which provides CRUD operations for
todo items and handles persistence to a JSON file.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from .todo_item import TodoItem
from .todo_list import TodoList


def _read_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Read and parse a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        list: List of dictionaries parsed from the JSON file.
        If the file doesn't exist, returns an empty list.

    Raises:
        IOError: If there's an error reading the file.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    try:
        if not os.path.exists(file_path):
            return []
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Error parsing JSON from {file_path}: {str(e)}", e.doc, e.pos
        ) from e
    except IOError as e:
        raise IOError(f"Error reading file {file_path}: {str(e)}") from e


def _write_json_file(file_path: str, data: List[Dict[str, Any]]) -> None:
    """Write data to a JSON file atomically.

    Args:
        file_path (str): Path to the JSON file.
        data (list): List of dictionaries to write to the file.

    Raises:
        IOError: If there's an error writing to the file.
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    temp_file = f"{file_path}.tmp"
    try:
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
        if os.name == 'nt' and os.path.exists(file_path):
            os.remove(file_path)
        os.rename(temp_file, file_path)
    except Exception as e:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise IOError(f"Error writing to file {file_path}: {str(e)}") from e


class TodoListManager:
    """Manages a collection of TodoLists with persistence to a JSON file.

    Attributes:
        file_path (str): Path to the JSON file where todo lists are stored.
        lists (dict): Dictionary of TodoList objects indexed by their IDs.
    """

    def __init__(self, file_path: str = "@todo_list.json") -> None:
        """Initialize a new TodoListManager.

        Args:
            file_path (str, optional): Path to the JSON file. Defaults to "@todo_list.json".
        """
        self.file_path: str = file_path
        self.lists: Dict[int, TodoList] = {}
        self._load_lists()

    def _load_lists(self) -> None:
        """Load todo lists from the JSON file.

        Handles and logs errors for individual list parsing and file I/O issues.
        """
        try:
            lists_data = _read_json_file(self.file_path)
            self.lists = {}
            for list_data in lists_data:
                try:
                    todo_list = TodoList._from_dict(list_data)
                    todo_list.parent = self
                    self._set_parent_refs(todo_list)
                    self.lists[todo_list.id] = todo_list
                except (KeyError, ValueError, TypeError) as e:
                    print(f"Warning: Skipping invalid todo list: {str(e)}")
        except FileNotFoundError:
            self.lists = {}
        except (IOError, json.JSONDecodeError) as e:
            self.lists = {}
            print(f"Warning: Error loading todo lists: {str(e)}")

    def _set_parent_refs(self, todo_list: TodoList) -> None:
        """Recursively set parent references for a todo list and its children.
        
        Args:
            todo_list: The TodoList to process
        """
        for child in todo_list.children:
            child.parent = todo_list
            if isinstance(child, TodoList):
                self._set_parent_refs(child)

    def save_lists(self) -> None:
        """Save todo lists to the JSON file."""
        lists_data = [todo_list._to_dict() for todo_list in self.lists.values()]
        _write_json_file(self.file_path, lists_data)

    def _get_next_id(self) -> int:
        """Get the next available ID for a new todo list.

        Returns:
            int: Next available ID.
        """
        return max(self.lists.keys()) + 1 if self.lists else 1

    def create_todo_list(self, description: str) -> TodoList:
        """Create a new todo list.

        Args:
            description (str): Description of the todo list.

        Returns:
            TodoList: The newly created TodoList.

        Raises:
            ValueError: If description is empty.

        Example:
            >>> manager = TodoListManager()
            >>> list = manager.create_todo_list("Groceries")
            >>> list.id
            1
            >>> list.name
            'Groceries'
        """
        if not description:
            raise ValueError("description cannot be empty")
        list_id = self._get_next_id()
        todo_list = TodoList(id=list_id, description=description)
        todo_list.parent = self
        self.lists[list_id] = todo_list
        self.save_lists()
        return todo_list

    def get_todo_list(self, id: int) -> Optional[TodoList]:
        """Get a todo list by ID.

        Args:
            id (int): ID of the todo list to get.

        Returns:
            TodoList or None: The TodoList with the given ID, or None if not found.
        """
        return self.lists.get(id)

    def update_todo_list(self, id: int, **fields: Any) -> bool:
        """Update a todo list.

        Args:
            id (int): ID of the todo list to update.
            **fields: Fields to update (name).

        Returns:
            bool: True if the list was updated, False if not found.

        Raises:
            ValueError: If an invalid name is provided.
        """
        todo_list = self.get_todo_list(id)
        if not todo_list:
            return False
        # Update fields
        if 'name' in fields:
            todo_list.name = fields['name']
        self.save_lists()
        return True

    def delete_todo_list(self, id: int) -> bool:
        """Delete a todo list.

        Args:
            id (int): ID of the todo list to delete.

        Returns:
            bool: True if the list was deleted, False if not found.
        """
        if id not in self.lists:
            return False
        del self.lists[id]
        self.save_lists()
        return True

    def list_todo_lists(self) -> List[TodoList]:
        """List all todo lists.

        Returns:
            list: List of TodoLists.
        """
        return list(self.lists.values())
