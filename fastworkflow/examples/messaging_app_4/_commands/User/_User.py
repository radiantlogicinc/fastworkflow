from ...application.chatroom import ChatRoom
from ...application.user import User

class Context:
    @classmethod
    def get_parent(cls, record: User) -> ChatRoom:
        return record.chatroom