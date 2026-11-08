"""ProposalReview/edit: free-form reply read as structured edits."""

from __future__ import annotations

import re
from typing import Any, Optional, Sequence

import fastworkflow
from fastworkflow.review_proposals import (
    PendingItemSlot,
    ProposedEdit,
    _proposed_edits_to_engine,
)
from fastworkflow.review_engine import EditParseError, EditRefused
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from fastworkflow.workflow import Workflow
from pydantic import BaseModel, ConfigDict, Field

from fastworkflow._workflows.consultation import _item_validation

class Signature:
    """Interpret the reply as edits to the plan."""

    class SlotInput(BaseModel):
        model_config = ConfigDict(extra="forbid")
        item: str
        edits: list[dict[str, Any]] = Field(default_factory=list)

    class Output(BaseModel):
        model_config = ConfigDict(extra="forbid")
        reply: str = Field(description="the user's raw reply text")
        edits: list[Any] = Field(default_factory=list, exclude=True)
        pending_proposed: list[dict[str, Any]] = Field(default_factory=list)
        refusal: Optional[str] = None

    plain_utterances = [
        "change the title to Dune",
        "use IT Support for step 2",
        "Lewis and Sons is a company",
        "the name should be Ann Lee",
        "in step 1 make the branch Central",
        "set the format to paperback",
        "step 2 should be for Bob instead",
        "replace the query with Emma",
        "update step 1 to use phone number",
        "drop 2 and set the author to Lee",
        "make step 2 use the other branch",
        "step 3 should check large print instead",
        "yes, but also include the reasons behind it",
        "no, not just one kind — include all of them",
        "not quite, I asked for the factors driving it as well",
        "I don't know which kind it is, use your best judgment",
        "that is one name for a single record, not two separate items",
        "the plan misses the why, also show the reasons",
        "please redo this to cover every kind, not just one",
    ]

    @staticmethod
    def generate_utterances(workflow: Workflow, command_name: str) -> list[str]:
        return [
            command_name.split("/")[-1].lower().replace("_", " ")
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)

    @staticmethod
    def validate_extracted_parameters(workflow, command, cmd_parameters) -> None:
        if not isinstance(cmd_parameters, Signature.SlotInput):
            return None
        proposal = workflow.context.get("consultation_proposal")
        if proposal is None:
            return None
        item_text = str(cmd_parameters.item)
        cmd_parameters.item = _item_validation.resolve_slot_item(
            proposal, item_text, field_name="item"
        )
        return None


def _expand_slot_edits(
    item: str,
    edits: list[dict[str, Any]],
    proposal: object,
) -> list[ProposedEdit]:
    raw_edits: list[ProposedEdit] = []
    if item == "all" and proposal is not None:
        count = len(proposal.items)
        for edit in edits:
            payload = dict(edit)
            if payload.get("index") is None:
                for idx in range(1, count + 1):
                    expanded = dict(payload)
                    expanded["index"] = idx
                    raw_edits.append(ProposedEdit.model_validate(expanded))
            else:
                raw_edits.append(ProposedEdit.model_validate(payload))
        return raw_edits
    if "," in item:
        indices = [int(part.strip()) for part in item.split(",") if part.strip()]
        for edit in edits:
            payload = dict(edit)
            if payload.get("index") is None:
                for idx in indices:
                    expanded = dict(payload)
                    expanded["index"] = idx
                    raw_edits.append(ProposedEdit.model_validate(expanded))
            else:
                raw_edits.append(ProposedEdit.model_validate(payload))
        return raw_edits
    resolved_index = int(item)
    for edit in edits:
        payload = dict(edit)
        if payload.get("index") is None:
            payload["index"] = resolved_index
        raw_edits.append(ProposedEdit.model_validate(payload))
    return raw_edits


def _coerce_to_engine_edits(proposal: object, edits: list, *, reply_text: str | None = None) -> list:
    if not edits:
        return []
    if isinstance(edits[0], ProposedEdit):
        return _proposed_edits_to_engine(proposal, edits, reply_text=reply_text)
    return list(edits)


def _proposal_with_rendering(proposal: object, rendering: str) -> object:
    class _Shim:
        __slots__ = ("_inner", "_rendering")

        def __init__(self, inner: object, text: str) -> None:
            self._inner = inner
            self._rendering = text

        def model_rendering(self) -> str:
            return self._rendering

        def __getattr__(self, name: str) -> object:
            return getattr(self._inner, name)

    return _Shim(proposal, rendering)


def _read_edits(workflow: Workflow, reply: str) -> list:
    scripted = workflow.context.get("consultation_edit_reader")
    proposal = workflow.context.get("consultation_proposal")
    if callable(scripted):
        if proposal is None:
            return []
        return _coerce_to_engine_edits(proposal, list(scripted(reply, proposal)), reply_text=reply)

    rendering = workflow.context.get("consultation_model_rendering")
    planning_names = workflow.context.get("consultation_planning_names")
    if not isinstance(rendering, str) or not isinstance(planning_names, Sequence):
        return []

    from fastworkflow.review_proposals import model_edit_reader

    if proposal is None:
        return []
    reader = model_edit_reader(
        planning_names=tuple(str(name) for name in planning_names)
    )
    return list(reader(reply, _proposal_with_rendering(proposal, rendering)))


class ResponseGenerator:
    def __call__(
        self,
        workflow: Workflow,
        command: str,
        input: None,
    ) -> fastworkflow.CommandResponse:
        if isinstance(input, Signature.SlotInput):
            proposal = workflow.context.get("consultation_proposal")
            from fastworkflow.review_proposals import _dedupe_proposed_edits

            raw_edits = _dedupe_proposed_edits(
                _expand_slot_edits(input.item, input.edits, proposal)
            )
            engine_edits = _proposed_edits_to_engine(
                proposal, raw_edits, reply_text=command
            )
            output = Signature.Output(reply=command, edits=engine_edits)
            return fastworkflow.CommandResponse(
                response="",
                artifacts=output.model_dump(),
                output=output,
            )

        reply = command or ""
        try:
            edits = _read_edits(workflow, reply)
        except EditParseError:
            output = Signature.Output(reply=reply, edits=[])
        except PendingItemSlot as pending:
            output = Signature.Output(
                reply=reply,
                pending_proposed=[item.model_dump() for item in pending.proposed],
            )
        except EditRefused as refused:
            output = Signature.Output(reply=reply, refusal=str(refused))
        else:
            output = Signature.Output(reply=reply, edits=edits)
        return fastworkflow.CommandResponse(
            response="",
            artifacts=output.model_dump(),
            output=output,
        )
