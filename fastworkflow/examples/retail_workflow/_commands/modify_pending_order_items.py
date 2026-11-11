from typing import List

import fastworkflow
from pydantic import BaseModel, Field, ConfigDict, model_validator
from fastworkflow.workflow import Workflow

from ..retail_data import load_data
from ..tools.modify_pending_order_items import ModifyPendingOrderItems


class Signature:
    """Modify pending order items"""
    class Input(BaseModel):
        order_id: str = Field(
            description="The order ID to modify",
            json_schema_extra={"from_command_output": "get_user_details"},
            pattern=r"^(#?[\w\d]+)$",
            examples=["#W0000000"],
        )
        item_ids: List[str] = Field(
            description="List of item IDs to be modified",
            json_schema_extra={"from_command_output": "get_order_details"},
            default_factory=list,
            examples=["'62727828', '76938178'"],
        )
        new_item_ids: List[str] = Field(
            description="List of new item IDs to replace with",
            json_schema_extra={"from_command_output": "get_product_details"},
            default_factory=list,
            examples=["'2352352', '42674747'"],
        )
        payment_method_id: str = Field(
            description="Payment method ID for price differences",
            json_schema_extra={"from_command_output": ["get_order_details", "get_user_details"]},
            pattern=r"^((gift_card|credit_card|paypal)_\d+)$",
            examples=["gift_card_0000000", "credit_card_0000000", "paypal_0000000"],
        )

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

        @model_validator(mode="after")
        def order_id_and_modify_items(self, info):
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
                error_message += f'cannot modify items for themselves: {common_items_str}. new item ids must differ from item ids\n'

            if error_message:
                raise fastworkflow.Refused(error_message)
            return self

    class Output(BaseModel):
        status: str = Field(description="Whether item modification succeeded.")

    plain_utterances: List[str] = [
        "I want to exchange some items in my pending order for different ones.",
        "Can I change the products in my order before it ships?",
        "Please update my order items to these new ones and charge the payment method accordingly.",
        "I need to swap out certain items in my order with other variants of the same product.",
        "How can I modify the items in my pending order and handle any price difference with my payment method?",
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
        result = ModifyPendingOrderItems.invoke(
            data=data,
            order_id=input.order_id,
            item_ids=input.item_ids,
            new_item_ids=input.new_item_ids,
            payment_method_id=input.payment_method_id,
        )
        return Signature.Output(status=result)