from typing import List

import fastworkflow
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.session import Session
from fastworkflow import CommandOutput, CommandResponse

from ..retail_data import load_data
from ..tools.modify_pending_order_payment import ModifyPendingOrderPayment


class Signature:
    """Modify pending order payment"""
    class Input(BaseModel):
        order_id: str = Field(
            default="NOT_FOUND",
            description="The order ID to modify (must start with #)",
            pattern=r"^(#W\d+|NOT_FOUND)$",
            examples=["#W0000000"],
        )
        payment_method_id: str = Field(
            default="NOT_FOUND",
            description="Payment method ID to switch to",
            pattern=r"^((gift_card|credit_card)_\d+|NOT_FOUND)$",
            examples=["gift_card_0000000", "credit_card_0000000"],
        )

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        status: str = Field(description="Whether payment modification succeeded.")

    plain_utterances: List[str] = [
        "I want to change the payment method for my pending order.",
        "Can you update my order to use a different payment method?",
        "Please switch the payment method for my order to a new credit card.",
        "I'd like to pay with a different gift card for my pending order.",
        "How can I modify the payment method on my order before it ships?",
    ]
    template_utterances: List[str] = []

    @staticmethod
    def generate_utterances(session: fastworkflow.Session, command_name: str) -> List[str]:
        workflow = session.workflow_snapshot.workflow
        utterance_definition = fastworkflow.UtteranceRegistry.get_definition(workflow.workflow_folderpath)
        utterances_obj = utterance_definition.get_command_utterances(workflow.path, command_name)
        from fastworkflow.train.generate_synthetic import generate_diverse_utterances
        return generate_diverse_utterances(utterances_obj.plain_utterances, command_name)


class ResponseGenerator:
    def __call__(self, session: Session, command: str, command_parameters: Signature.Input) -> CommandOutput:
        output = self._process_command(session, command_parameters)
        return CommandOutput(
            session_id=session.id,
            command_responses=[CommandResponse(response=f"Modified details: {output.status}")],
        )

    def _process_command(self, session: Session, input: Signature.Input) -> Signature.Output:
        data = load_data()
        result = ModifyPendingOrderPayment.invoke(
            data=data,
            order_id=input.order_id,
            payment_method_id=input.payment_method_id,
        )
        return Signature.Output(status=result) 