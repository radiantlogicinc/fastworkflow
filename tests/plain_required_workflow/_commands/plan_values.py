"""plan_values: a planning command whose inputs are plain required fields (no default)."""

from typing import List

from pydantic import BaseModel, Field

from fastworkflow.planning import do_not_run


class Signature:
    """Plan a recording of four values."""

    plain_utterances = ["plan a recording of the values"]

    class Input(BaseModel):
        name: str = Field(description="a name", json_schema_extra={"in_request": True})
        count: int = Field(description="a count")
        ratio: float = Field(description="a ratio")
        ids: list[str] = Field(description="a list of ids")


class PlanGenerator:
    def check(self, input):
        return [do_not_run("record_values")]

    def __result__(self, input):
        return f"Planned {input.name}."
