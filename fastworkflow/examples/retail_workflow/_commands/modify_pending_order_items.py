from typing import List

import fastworkflow
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.workflow import Workflow
from fastworkflow import CommandOutput, CommandResponse

from ..retail_data import load_data
from ..tools.modify_pending_order_items import ModifyPendingOrderItems


class Signature:
    """Modify pending order items"""
    class Input(BaseModel):
        order_id: str = Field(
            default="NOT_FOUND",
            description="The order ID to modify",
            pattern=r"^(#?[\w\d]+|NOT_FOUND)$",
            examples=["#W0000000"],
            json_schema_extra={
                "available_from": ["get_user_details"]
            }
        )
        item_ids: List[str] = Field(
            default_factory=list,
            description="List of item IDs to be modified",
            examples=["1008292230", "2468135790"],
            json_schema_extra={
                "available_from": ["get_order_details"]
            }
        )
        new_item_ids: List[str] = Field(
            default_factory=list,
            description="List of new item IDs to replace with",
            examples=["1008292231", "2468135791"],
            json_schema_extra={
                "available_from": ["get_product_details"]
            }
        )
        payment_method_id: str = Field(
            default="NOT_FOUND",
            description="Payment method ID for price differences",
            pattern=r"^((gift_card|credit_card|paypal)_\d+|NOT_FOUND)$",
            examples=["gift_card_0000000", "credit_card_0000000", "paypal_0000000"],
            json_schema_extra={
                "available_from": ["get_order_details", "get_user_details"]
            }
        )

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

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

    @staticmethod
    def validate_extracted_parameters(
        workflow: fastworkflow.Workflow, 
        command: str, cmd_parameters: "Signature.Input"
    ) -> tuple[bool, str]:
        if not cmd_parameters.order_id.startswith('#'):
            cmd_parameters.order_id = f'#{cmd_parameters.order_id}'
        return (True, '')


class ResponseGenerator:
    def __call__(self, workflow: Workflow, command: str, command_parameters: Signature.Input) -> CommandOutput:
        output = self._process_command(workflow, command_parameters)
        response = (
            f'Response: Modified details: {output.status}'
        )
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[CommandResponse(response=response)],
        )

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