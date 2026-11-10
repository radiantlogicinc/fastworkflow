from ...application.library import Library


class Context:
    @classmethod
    def get_parent(cls, record: Library) -> None:
        return None

    @classmethod
    def get_displayname(cls, record: Library) -> str:
        return "Library"
