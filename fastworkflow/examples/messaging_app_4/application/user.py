from __future__ import annotations

# Avoid a runtime circular import: only import ChatRoom when running type checks
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover – only needed for static type checkers
    from .chatroom import ChatRoom

class User:
    """Simple user class representing the current messaging user."""

    def __init__(self, chatroom: "ChatRoom", name: str):
        self.chatroom = chatroom
        self.name = name

    def send_message(self, to: str, message: str):
        """Send a message to the target user (prints to stdout)."""
        print(f"{self.name} sends '{message}' to {to}")

    def to_state_dict(self) -> dict[str, Any]:
        """Snapshot this user as JSON-native data.

        ``chatroom`` is a back-reference that makes the live graph cyclic, so it
        is omitted and supplied again by the room on restore. The subclass is
        recorded because it decides which commands the user context exposes.
        """
        return {
            "name": self.name,
            "type": type(self).__name__,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any], chatroom: "ChatRoom") -> "User":
        """Rebuild a user from :meth:`to_state_dict`, inside *chatroom*."""
        user_class = PremiumUser if state.get("type") == PremiumUser.__name__ else User
        return user_class(chatroom, state["name"])

class PremiumUser(User):
    def send_priority_message(self, to, message):
        print(f"{self.name} sends PRIORITY message '{message}' to {to}")