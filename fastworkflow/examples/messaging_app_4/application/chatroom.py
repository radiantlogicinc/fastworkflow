from typing import Any

from .user import User, PremiumUser


class ChatRoom:
    def __init__(self):
        self.current_user = None
        self.users = []

    def add_user(self, user: User):
        self.users.append(user)

    def list_users(self) -> list[str]:
        return [user.name for user in self.users]

    @property
    def current_user(self) -> User:
        return self._current_user

    @current_user.setter
    def current_user(self, value: User):
        self._current_user = value

    def broadcast(self, message) -> None:
        sender_name = self._current_user.name if self._current_user else 'Anonymous'
        msg_priority = 'PRIORITY' if isinstance(self._current_user, PremiumUser) else ''

        if self.users:
            for user in self.users:
                if user.name == sender_name:
                    continue
                print(f"user {sender_name} is broadcasting {msg_priority} '{message}' to {user.name}")
        else:
            print("No users found in this chat room. Add some users first")

    def to_state_dict(self) -> dict[str, Any]:
        """Snapshot this room and its users as JSON-native data.

        The current user is recorded as a position in ``users`` rather than as a
        second snapshot of the user, so that on restore ``current_user`` is the
        same object as the corresponding entry of ``users`` instead of an equal
        copy of it.
        """
        current_index = next(
            (index for index, user in enumerate(self.users) if user is self._current_user),
            None
        )

        return {
            "users": [user.to_state_dict() for user in self.users],
            "current_user_index": current_index,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "ChatRoom":
        """Rebuild a room from :meth:`to_state_dict`.

        Users are created against this room so that every ``user.chatroom``
        points back at it, which is the link the snapshot deliberately drops.
        """
        chatroom = cls()

        for user_state in state.get("users", []):
            chatroom.add_user(User.from_state_dict(user_state, chatroom))

        current_index = state.get("current_user_index")
        if current_index is not None:
            chatroom.current_user = chatroom.users[current_index]

        return chatroom
