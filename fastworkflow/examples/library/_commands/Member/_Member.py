from ...application.library import Library, Member


class Context:
    @classmethod
    def get_parent(cls, record: Member) -> Library:
        return record.library

    @classmethod
    def get_displayname(cls, record: Member) -> str:
        return f"Member: {record.name} ({record.uid})"
