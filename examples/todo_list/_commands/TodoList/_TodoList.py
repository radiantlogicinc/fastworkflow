from typing import Optional
from ...application.todo_list import TodoList
from ...application.todo_manager import TodoListManager

class Context:
    @classmethod
    def get_parent(cls, command_context_object: TodoList) -> Optional[TodoListManager]:
        return command_context_object.parent 