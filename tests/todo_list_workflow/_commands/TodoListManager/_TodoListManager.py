from typing import Any, Optional, Union

import fastworkflow

from ...application.todo_item import TodoItem
from ...application.todo_list import TodoList
from ...application.todo_manager import TodoListManager

class Context:
    state_version = 1

    @classmethod
    def get_parent(cls, command_context_object: TodoListManager) -> None:
        return None

    @classmethod
    def get_state(cls, command_context_object: TodoListManager) -> dict[str, Any]:
        """Snapshot the manager so a session holding it can be evicted.

        Without this hook the session is pinned: a TodoListManager is an
        ordinary Python object that no generic projection can reach. Return
        fastworkflow.EPHEMERAL instead to declare deliberately that this
        context is not worth persisting.
        """
        return command_context_object.to_state_dict()

    @classmethod
    def from_state(cls, state: dict[str, Any], workflow: fastworkflow.Workflow,
                   *, state_version: int) -> TodoListManager:
        """Rebuild the manager from a snapshot written at *state_version*.

        Only version 1 has ever shipped, so there is nothing to migrate yet.
        Once a second version exists, translate the older shape here rather
        than refusing it — this hook is the only code that understands it.
        """
        if state_version != cls.state_version:
            raise fastworkflow.UnsupportedStateVersion(state_version)

        # Recomputed the way startup computes it, so a snapshot stays portable
        # across deployments instead of carrying one machine's absolute path.
        filepath = (
            f'{workflow.folderpath}/'
            'application/'
            'todo_list.json'
        )
        return TodoListManager.from_state_dict(state, filepath)

    @classmethod
    def get_locator(cls, command_context_object: Union[TodoList, TodoItem]) -> str:
        """Identify a list or item within the manager, as a path of ids.

        The current context is usually a node *inside* the snapshotted manager,
        and a second snapshot of that node would restore as a second object
        rather than as part of the hierarchy. Ids are unique only among
        siblings, so the whole ancestor chain is needed to make the path
        unambiguous.
        """
        ids = []
        node = command_context_object
        while isinstance(node, TodoItem):
            ids.append(str(node.id))
            node = node.parent
        return '/'.join(reversed(ids))

    @classmethod
    def find_by_locator(cls, anchor: TodoListManager,
                        locator: str) -> Optional[Union[TodoList, TodoItem]]:
        ids = [int(segment) for segment in locator.split('/')]

        node = anchor.get_todo_list(ids[0])
        for child_id in ids[1:]:
            if not isinstance(node, TodoList):
                return None
            node = node.get_child_by_id(child_id)

        return node
