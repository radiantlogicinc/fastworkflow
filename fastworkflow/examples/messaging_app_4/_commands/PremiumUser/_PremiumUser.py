from ...application.chatroom import ChatRoom
from ...application.user import PremiumUser

class Context:
    @classmethod
    def get_parent(cls, record: PremiumUser) -> ChatRoom:
        return record.chatroom 