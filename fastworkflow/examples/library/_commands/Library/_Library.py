from ...application.library import Library


class Context:
    @classmethod
    def get_parent(cls, command_context_object: Library) -> None:
        return None

    @classmethod
    def get_displayname(cls, command_context_object: Library) -> str:
        return "Library"
