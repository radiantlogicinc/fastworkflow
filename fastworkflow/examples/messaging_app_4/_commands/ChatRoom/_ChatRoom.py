from ...application.chatroom import ChatRoom


class Context:
    @classmethod
    def get_parent(cls, command_context_object: ChatRoom) -> None:
        return None