from typing import Any, Optional

import fastworkflow

from ...application.chatroom import ChatRoom
from ...application.user import User


class Context:
    state_version = 1

    @classmethod
    def get_parent(cls, command_context_object: ChatRoom) -> None:
        return None

    @classmethod
    def get_state(cls, command_context_object: ChatRoom) -> dict[str, Any]:
        """Snapshot the chat room so a session holding it can be evicted.

        Without this hook the session is pinned from the moment
        set_root_context runs: a ChatRoom is an ordinary Python object that no
        generic projection can reach. Return fastworkflow.EPHEMERAL instead to
        declare deliberately that this context is not worth persisting.
        """
        return command_context_object.to_state_dict()

    @classmethod
    def from_state(cls, state: dict[str, Any], workflow: fastworkflow.Workflow,
                   *, state_version: int) -> ChatRoom:
        """Rebuild the room from a snapshot written at *state_version*.

        Only version 1 has ever shipped, so there is nothing to migrate yet.
        Once a second version exists, translate the older shape here rather
        than refusing it — this hook is the only code that understands it.
        """
        if state_version != cls.state_version:
            raise fastworkflow.UnsupportedStateVersion(state_version)

        return ChatRoom.from_state_dict(state)

    @classmethod
    def get_locator(cls, command_context_object: User) -> str:
        """Identify a user within the room by the name the room addresses them by.

        set_current_user points the current context at a user *inside* the
        snapshotted room, and a second snapshot of that user would restore as a
        second object rather than as a member of the room. add_user rejects a
        duplicate name, so the name is the room's natural key; if one ever gets
        through, the framework's identity check pins the session rather than
        restoring the wrong user.
        """
        return command_context_object.name

    @classmethod
    def find_by_locator(cls, anchor: ChatRoom, locator: str) -> Optional[User]:
        return next((user for user in anchor.users if user.name == locator), None)
