from typing import Any, Optional

import fastworkflow

from ...application.workitem import WorkItem

class Context:
    state_version = 1

    @classmethod
    def get_parent(cls, command_context_object: WorkItem) -> Optional[WorkItem]:
        return command_context_object.parent or command_context_object

    @classmethod
    def get_displayname(cls, command_context_object: WorkItem) -> str:
        return f'{command_context_object.__class__.__name__}: {command_context_object.get_absolute_path()}'

    @classmethod
    def get_state(cls, command_context_object: WorkItem) -> dict[str, Any]:
        """Snapshot the work item tree so a session holding it can be evicted.

        Without this hook the session is pinned: a WorkItem is an ordinary
        Python object that no generic projection can reach. Return
        fastworkflow.EPHEMERAL instead to declare deliberately that this
        context is not worth persisting.
        """
        return command_context_object.to_state_dict()

    @classmethod
    def from_state(cls, state: dict[str, Any], workflow: fastworkflow.Workflow,
                   *, state_version: int) -> WorkItem:
        """Rebuild the tree from a snapshot written at *state_version*.

        Only version 1 has ever shipped, so there is nothing to migrate yet.
        Once a second version exists, translate the older shape here rather
        than refusing it — this hook is the only code that understands it.
        """
        if state_version != cls.state_version:
            raise fastworkflow.UnsupportedStateVersion(state_version)

        # The schema is static workflow configuration, so it is reloaded rather
        # than snapshotted, and one instance is shared by the whole tree.
        workflow_schema = WorkItem.WorkflowSchema.from_json_file(
            f'{workflow.folderpath}/simple_workflow_template.json'
        )
        return WorkItem.from_state_dict(state, workflow_schema)

    @classmethod
    def get_locator(cls, command_context_object: WorkItem) -> str:
        """Identify a work item within its tree.

        The current context is usually a node *inside* the snapshotted root, and
        a second snapshot of that node would restore as a second object rather
        than as part of the tree. A locator points into the restored tree
        instead, so identity survives.
        """
        return command_context_object.get_absolute_path()

    @classmethod
    def find_by_locator(cls, anchor: WorkItem, locator: str) -> Optional[WorkItem]:
        return anchor.get_workitem(locator)
