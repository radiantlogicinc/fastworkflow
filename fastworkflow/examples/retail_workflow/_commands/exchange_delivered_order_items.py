from typing import List

import fastworkflow
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.workflow import Workflow
from fastworkflow import CommandOutput, CommandResponse

from ..retail_data import load_data
from ..tools.exchange_delivered_order_items import ExchangeDeliveredOrderItems


class Signature:
    """Exchange delivered order items"""
    class Input(BaseModel):
        order_id: str = Field(
            default="NOT_FOUND",
            description="The order ID to exchange",
            pattern=r"^(#?[\w\d]+|NOT_FOUND)$",
            examples=["#W00000000"],
            json_schema_extra={
                "available_from": ["get_user_details"]
            }
        )
        item_ids: List[str] = Field(
            default_factory=list,
            description="The item IDs to be exchanged",
            examples=["1008292230"],
            json_schema_extra={
                "available_from": ["get_order_details"]
            }
        )
        new_item_ids: List[str] = Field(
            default_factory=list,
            description="The new item IDs to exchange for",
            examples=["1008292230"],
            json_schema_extra={
                "available_from": ["get_product_details"]
            }
        )
        payment_method_id: str = Field(
            default="NOT_FOUND",
            description="Payment method ID for price difference.",
            pattern=r"^((gift_card|credit_card|paypal)_\d+|NOT_FOUND)$",
            examples=["gift_card_0000000", "credit_card_0000000", "paypal_0000000"],
            json_schema_extra={
                "available_from": ["get_order_details", "get_user_details"]
            }
        )

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

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
            f'Response: {output.status}'
        )
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[CommandResponse(response=response)],
        )

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