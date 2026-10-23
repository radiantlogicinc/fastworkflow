"""overdue_summary: the library's branches and how many loans are overdue.

Fixed steps: the skill takes no inputs. It lists the branches, counts the overdue loans
across the library and says both.
"""


class Signature:
    """Summarize the library: its branches and how many loans are overdue."""

    plain_utterances = ["give me the library overview", "how many loans are overdue overall"]


class PlanGenerator:
    def summary(self):
        return ["list_branches",
                "count_overdue",
                "say how many branches the library has and how many loans are overdue"]
