"""overdue_summary: the library's branches, then how many loans are overdue.

Two phases and one optional input: branches lists the branches and hands them on; overdue
counts the overdue loans in the branch the request names (in every branch when it names
none) and answers, naming the branches.
"""
from pydantic import BaseModel, Field, ValidationInfo, field_validator

from fastworkflow.planning import run

from ..application import library
from .list_branches import Signature as ListBranches


class Signature:
    """Name the branches, then say how many loans are overdue, in one branch or in all."""

    class Input(BaseModel):
        branch: str = Field(default="", description="the branch asked about, if named (such as Central "
                                                    "or Riverside); empty for all",
                            json_schema_extra={"in_request": {"ignore": ["the", "branch"]}})

        @field_validator("branch")
        @classmethod
        def a_branch_of_the_library(cls, value: str, info: ValidationInfo) -> str:
            if not value.strip() or not (info.context or {}).get("request"):
                return value      # no branch, or a placeholder of the template
            if value.strip().casefold() in EVERY_BRANCH:
                return ""
            for name in library.BRANCHES:
                if value.strip().casefold() == name.casefold():
                    return name
            raise ValueError(f"branch is {value!r}, which is not a branch of the library; the branches "
                             f"are {', '.join(library.BRANCHES)}, or leave it empty for every branch")

    plain_utterances = ["give me the library overview", "how many loans are overdue overall"]


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
