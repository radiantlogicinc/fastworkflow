from ...application.chatroom import ChatRoom
from ...application.user import PremiumUser

class Context:
    @classmethod
    def get_parent(cls, command_context_object: PremiumUser) -> ChatRoom:
        return command_context_object.chatroom 