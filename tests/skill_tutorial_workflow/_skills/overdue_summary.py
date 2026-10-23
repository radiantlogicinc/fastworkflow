"""overdue_summary: the library's branches, then how many loans are overdue.

Two phases and no inputs: branches lists the branches; overdue counts the overdue loans
across the library and answers.
"""


class Signature:
    """List the library's branches, then say how many loans are overdue across the library."""

    plain_utterances = ["give me the library overview", "how many loans are overdue overall"]


class PlanGenerator:
    def branches(self):
        return ["list_branches"]

    def overdue(self):
        return ["count_overdue",
                "say how many loans are overdue across the library"]
