from typing import Optional
from ...application.workitem import WorkItem

class Context:
    @classmethod
    def get_parent(cls, command_context_object: WorkItem) -> Optional[WorkItem]:
        return command_context_object.parent or command_context_object

    @classmethod
    def get_displayname(cls, command_context_object: WorkItem) -> str:
        return f'{command_context_object.__class__.__name__}: {command_context_object.get_absolute_path()}'