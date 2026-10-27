from ...application.library import Book, Library


class Context:
    @classmethod
    def get_parent(cls, command_context_object: Book) -> Library:
        return command_context_object.library

    @classmethod
    def get_displayname(cls, command_context_object: Book) -> str:
        return f"Book: {command_context_object.title} ({command_context_object.uid})"
