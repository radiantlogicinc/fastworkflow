from typing import List

import fastworkflow
from pydantic import BaseModel, Field, ConfigDict, model_validator
from fastworkflow.workflow import Workflow

from ..retail_data import load_data
from ..tools.exchange_delivered_order_items import ExchangeDeliveredOrderItems


class Signature:
    """Exchange delivered order items"""
    class Input(BaseModel):
        order_id: str = Field(
            description="The order ID to exchange",
            json_schema_extra={"from_command_output": "get_user_details"},
            pattern=r"^(#?[\w\d]+)$",
            examples=["#W00000000"],
        )
        item_ids: List[str] = Field(
            description="The item IDs to be exchanged",
            json_schema_extra={"from_command_output": "get_order_details"},
            default_factory=list,
            examples=["'34742388', '59475928'"],
        )
        new_item_ids: List[str] = Field(
            description="The new item IDs to exchange for",
            json_schema_extra={"from_command_output": "get_product_details"},
            default_factory=list,
            examples=["'45854949', '68789960'"],
        )
        payment_method_id: str = Field(
            description="Payment method ID for price difference.",
            json_schema_extra={"from_command_output": ["get_order_details", "get_user_details"]},
            pattern=r"^((gift_card|credit_card|paypal)_\d+)$",
            examples=["gift_card_0000000", "credit_card_0000000", "paypal_0000000"],
        )

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

        @model_validator(mode="after")
        def order_id_and_exchange_items(self, info):
            if not self.order_id.startswith('#'):
                self.order_id = f'#{self.order_id}'

            error_message = ''
            if len(self.item_ids) == 0:
                error_message = 'item ids must be specified\n'
            for item_id in self.item_ids:
                try:
                    int(item_id)
                except ValueError:
                    error_message += 'item ids must be integers\n'
                    break

            if len(self.new_item_ids) == 0:
                error_message += 'new item ids must be specified\n'
            for item_id in self.new_item_ids:
                try:
                    int(item_id)
                except ValueError:
                    error_message += 'new item ids must be integers\n'
                    break

            if len(self.new_item_ids) != len(self.item_ids):
                error_message += 'the number of item ids and new item ids must match for a valid exchange\n'

            if common_items := set(self.item_ids).intersection(
                set(self.new_item_ids)
            ):
                common_items_str = ', '.join(common_items)
                error_message += f'cannot exchange items for themselves: {common_items_str}. new item ids must differ from item ids\n'

            if error_message:
                raise fastworkflow.Refused(error_message)
            return self

    class Output(BaseModel):
        status: str = Field(description="Whether exchange succeeded.")

    plain_utterances: List[str] = [
        "I received the wrong size in my order. Can I get it exchanged for a different one?",
        "The item I got doesn't fit well. I'd like to swap it with another of the same kind.",
        "Can you help me replace one of the items I received with a different variant?",
        "I'd like to exchange some things from my last delivery — they weren't quite right.",
        "One of the products I got isn't what I expected. Can I trade it in for a different one?",
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
        result = ExchangeDeliveredOrderItems.invoke(
            data=data,
            order_id=input.order_id,
            item_ids=input.item_ids,
            new_item_ids=input.new_item_ids,
            payment_method_id=input.payment_method_id,
        )
        return Signature.Output(status=result) 