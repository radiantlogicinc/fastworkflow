"""note_member: add the request's note to the record of the member it names.

One phase, so the call is one plan, run by one executor run. It finds the member, opens
the one named, and adds the note there.
"""
import re

from pydantic import BaseModel, Field, model_validator

from fastworkflow.planning import pick, run


class Signature:
    """Add a note to a member's record."""

    plain_utterances = [
        "note on Raj Patel's record that he prefers email reminders",
        "put a note on Ann Lee: card renewed",
        "put on Mia Chen's record that she wants large print",
        "add to Raj Patel's record: moved address",
        "leave a note for Ann Lee saying her card is due for renewal",
        "write a note about Mia Chen: prefers pickup at Riverside",
        "record for Raj Patel that he asked for a call back",
        "attach to Ann Lee's record a note about the damaged card",
    ]

    class Input(BaseModel):
        member: str = Field(description="the member's name",
                            json_schema_extra={"in_request": True})
        text: str = Field(description="what the note says, without the request around it "
                                      "(to note that someone called: 'someone called')",
                          json_schema_extra={"in_request": True})

        @model_validator(mode="after")
        def note_without_the_request(self):
            # A note on Mia Chen's record does not name Mia Chen: a text that does holds the
            # request ("Note on Mia Chen's record that ..."), not the note.
            if re.search(rf"\b{re.escape(self.member)}\b", self.text, re.IGNORECASE):
                raise ValueError(f"text is {self.text!r}, which holds the member's name: that is the "
                                 "request around the note; give only the words the note says")
            return self


class PlanGenerator:
    def note(self, input):
        return [run("find_member", query=input.member),
                pick("open_member", named=input.member),
                run("add_note", text=input.text),
                "say whose record the note was added to"]
