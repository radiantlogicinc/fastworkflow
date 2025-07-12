from __future__ import annotations

# Avoid a runtime circular import: only import ChatRoom when running type checks
from typing import TYPE_CHECKING

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

class PremiumUser(User):
    def send_priority_message(self, to, message):
        print(f"{self.name} sends PRIORITY message '{message}' to {to}")