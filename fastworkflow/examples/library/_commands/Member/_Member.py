from ...application.library import Library, Member


class Context:
    @classmethod
    def get_parent(cls, command_context_object: Member) -> Library:
        return command_context_object.library

    @classmethod
    def get_displayname(cls, command_context_object: Member) -> str:
        return f"Member: {command_context_object.name} ({command_context_object.uid})"
