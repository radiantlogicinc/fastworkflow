"""overdue_summary: the library's branches, then how many loans are overdue.

Two phases and no inputs: branches lists the branches; overdue counts the overdue loans
across the library, and its agent writes the answer.
"""
from fastworkflow.planning import run


class Signature:
    """List the library's branches, then say how many loans are overdue across the library."""

    plain_utterances = [
        "give me the library overview",
        "how is the library doing",
        "a status report on the library: branches and overdue loans",
        "summarize the branches and how many loans are late",
        "overall picture of the library, please",
        "run the library summary",
        "branches and overdue count for the whole library",
        "how is everything at the library, including late loans",
    ]


class PlanGenerator:
    def branches(self):
        return [run("list_branches")]

    def overdue(self):
        return [run("count_overdue"),
                "say how many loans are overdue across the library"]
