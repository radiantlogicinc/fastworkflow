from ...application.library import Book, Library


class Context:
    @classmethod
    def get_parent(cls, record: Book) -> Library:
        return record.library

    @classmethod
    def get_displayname(cls, record: Book) -> str:
        return f"Book: {record.title} ({record.uid})"
