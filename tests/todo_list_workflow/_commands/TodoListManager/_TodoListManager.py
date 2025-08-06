from typing import Optional
from ...application.todo_manager import TodoListManager

class Context:
    @classmethod
    def get_parent(cls, command_context_object: TodoListManager) -> None:
        return None 