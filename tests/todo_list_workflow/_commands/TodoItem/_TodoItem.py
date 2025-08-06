from typing import Optional
from ...application.todo_item import TodoItem
from ...application.todo_list import TodoList

class Context:
    @classmethod
    def get_parent(cls, command_context_object: TodoItem) -> Optional[TodoList]:
        return command_context_object.parent 