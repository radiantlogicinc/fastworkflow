"""overdue_summary: the library's branches, then how many loans are overdue.

Two phases and no inputs: branches lists the branches; overdue counts the overdue loans
across the library, and its agent writes the answer.
"""
from fastworkflow.planning import run


class Signature:
    """List the library's branches, then say how many loans are overdue across the library."""

    plain_utterances = ["give me the library overview", "how is the library doing"]


class PlanGenerator:
    def branches(self):
        return [run("list_branches")]

    def overdue(self):
        return [run("count_overdue"),
                "say how many loans are overdue across the library"]
