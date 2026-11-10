from typing import List

import fastworkflow
from pydantic import BaseModel, Field, ConfigDict, model_validator
from fastworkflow.workflow import Workflow

from ..retail_data import load_data
from ..tools.modify_pending_order_payment import ModifyPendingOrderPayment


class Signature:
    """Modify pending order payment method"""
    class Input(BaseModel):
        order_id: str = Field(
            description="The order ID to modify",
            json_schema_extra={"from_command_output": "get_user_details"},
            pattern=r"^(#?[\w\d]+)$",
            examples=["#W0000000"],
        )
        payment_method_id: str = Field(
            description="Payment method ID to switch to",
            json_schema_extra={"from_command_output": "get_user_details"},
            pattern=r"^((gift_card|credit_card|paypal)_\d+)$",
            examples=["gift_card_0000000", "credit_card_0000000", "paypal_0000000"],
        )

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

        @model_validator(mode="after")
        def order_id_has_hash(self, info):
            if not self.order_id.startswith('#'):
                self.order_id = f'#{self.order_id}'
            return self

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
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> List[str]:
        utterance_definition = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)
        utterances_obj = utterance_definition.get_command_utterances(command_name)
        from fastworkflow.train.generate_synthetic import generate_diverse_utterances
        return generate_diverse_utterances(utterances_obj.plain_utterances, command_name)

class ResponseGenerator:
    def __call__(self, workflow: Workflow, command: str, command_parameters: Signature.Input) -> Signature.Output:
        output = self._process_command(workflow, command_parameters)
        return output

    def _process_command(self, workflow: Workflow, input: Signature.Input) -> Signature.Output:
        data = load_data()
        result = ModifyPendingOrderPayment.invoke(
            data=data,
            order_id=input.order_id,
            payment_method_id=input.payment_method_id,
        )
        return Signature.Output(status=result) 