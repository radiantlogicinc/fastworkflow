from typing import Any

import fastworkflow

from ...application.user import User


class Context:
    state_version = 1

    @classmethod
    def get_parent(cls, command_context_object: User) -> None:
        return None

    @classmethod
    def get_state(cls, command_context_object: User) -> dict[str, Any]:
        """Snapshot the user so a session holding it can be evicted.

        Without this hook the session is pinned from its first turn: a User is
        an ordinary Python object that no generic projection can reach. The
        whole of this context is one field, so there is nothing to delegate to
        the application class. Return fastworkflow.EPHEMERAL instead to declare
        deliberately that this context is not worth persisting.
        """
        return {'name': command_context_object.name}

    @classmethod
    def from_state(cls, state: dict[str, Any], workflow: fastworkflow.Workflow,
                   *, state_version: int) -> User:
        """Rebuild the user from a snapshot written at *state_version*.

        Only version 1 has ever shipped, so there is nothing to migrate yet.
        Once a second version exists, translate the older shape here rather
        than refusing it — this hook is the only code that understands it.
        """
        if state_version != cls.state_version:
            raise fastworkflow.UnsupportedStateVersion(state_version)

        return User(state['name'])

    # No get_locator/find_by_locator: this workflow never points the current
    # context at anything but the root user, so every slot is the anchor itself.
