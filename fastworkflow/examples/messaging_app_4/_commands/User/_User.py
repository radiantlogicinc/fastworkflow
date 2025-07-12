from ...application.chatroom import ChatRoom
from ...application.user import User

class Context:
    @classmethod
    def get_parent(cls, command_context_object: User) -> ChatRoom:
        return command_context_object.chatroom