"""overdue_summary: the library's branches, then how many loans are overdue.

Two phases and one optional input: branches lists the branches and hands them on; overdue
counts the overdue loans in the branch the request names (in every branch when it names
none), and its agent writes the answer, which names the branches.
"""
from pydantic import BaseModel, Field, ValidationInfo, field_validator

from fastworkflow.planning import run

from ...application.library import HOURS
from .list_branches import Signature as ListBranches


class Signature:
    """Name the branches, then say how many loans are overdue, in one branch or in all."""

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

    class Input(BaseModel):
        branch: str = Field(default="", description="the branch asked about, if named (such as Central "
                                                    "or Riverside); empty for all",
                            json_schema_extra={"copied_from_command_text": {"ignore": ["the", "branch"]}})

        @field_validator("branch")
        @classmethod
        def a_branch_of_the_library(cls, value: str, info: ValidationInfo) -> str:
            if not value.strip() or not (info.context or {}).get("command_text"):
                return value      # no branch, or a placeholder of the template
            if value.strip().casefold() in EVERY_BRANCH:
                return ""
            for name in HOURS:
                if value.strip().casefold() == name.casefold():
                    return name
            raise ValueError(f"branch is {value!r}, which is not a branch of the library; "
                             f"the branches are {', '.join(HOURS)}, or leave it empty for every branch")


class PlanGenerator:
    def branches(self) -> ListBranches.Output:
        return [run("list_branches")]

    def overdue(self, input, branches):
        where = f"at {input.branch}" if input.branch else "across the library"
        return [run("count_overdue", branch=input.branch or None),
                f"name the branches ({', '.join(branches.names)}) and the number of overdue loans {where}"]


#: Words that name every branch, not one: the whole library.
EVERY_BRANCH = {"all", "all branches", "every branch", "each branch", "any branch", "the library",
                "the whole library", "library", "overall"}
