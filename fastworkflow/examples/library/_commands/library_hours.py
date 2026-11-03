"""library_hours: when each branch is open. A global command: it is found from every context."""
import fastworkflow
from pydantic import BaseModel, Field

from ..application.library import HOURS


class Signature:
    """Say when the library's branches are open, or one branch's hours."""

    class Input(BaseModel):
        branch: str = Field(default="", description="one branch's name; empty for every branch")

    class Output(BaseModel):
        branches: list[str] = Field(description="each branch named, in row order")
        hours: list[str] = Field(description="each branch's opening hours, in row order")

    plain_utterances = [
        "when are you open",
        "what are Riverside's opening hours",
        "what time do you close on Saturdays",
        "hours for Central",
        "is the library open on Tuesday",
        "what time do you open tomorrow",
        "opening times at the Riverside branch",
        "when does Central shut",
    ]


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str,
                 command_parameters: Signature.Input) -> Signature.Output:
        wanted = command_parameters.branch.casefold()
        rows = [(name, hours) for name, hours in HOURS.items()
                if not wanted or name.casefold() == wanted]
        if not rows:
            raise fastworkflow.Refused(
                f"no branch is named {command_parameters.branch}; the branches are "
                f"{', '.join(HOURS)}")
        return Signature.Output(branches=[n for n, _ in rows], hours=[h for _, h in rows])
