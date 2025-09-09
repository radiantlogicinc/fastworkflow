from typing import List

import fastworkflow
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.workflow import Workflow
from fastworkflow import CommandOutput, CommandResponse

# Domain helpers
from ..retail_data import load_data
from ..tools.get_order_details import GetOrderDetails


class Signature:
    """Get order details"""
    class Input(BaseModel):
        order_id: str = Field(
            default="NOT_FOUND",
            description=(
                "The order ID to get details for"
            ),
            pattern=r"^(#?[\w\d]+|NOT_FOUND)$",
            examples=["#W0000000"],
            json_schema_extra={
                "available_from": ["get_user_details"]
            }
        )

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        order_details: str = Field(
            description=(
                "Detailed information about the order such as "
                "shipping address, "
                "items ordered with details including name, product and item id, price and options, "
                "fulfillments with details including tracking id and item ids, "
                "order status, and"
                "payment history with details including transaction type, amount and payment method id"
            ),
            json_schema_extra={
                "used_by": [
                    "exchange_delivered_order_items",
                    "get_product_details",
                    "modify_pending_order_items",
                    "return_delivered_order_items",
                ]
            },
        )

    plain_utterances: List[str] = [
        "Can you tell me what's going on with my order #W1234567?",
        "I want to check the status of the order I placed yesterday.",
        "What's the current update on my recent purchase?",
        "Can you pull up the details of my last order?",
        "Is there any info on when my package will be delivered?",
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
    def __call__(
        self,
        workflow: Workflow,
        command: str,
        command_parameters: Signature.Input,
    ) -> CommandOutput:
        output = self._process_command(workflow, command_parameters)
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                CommandResponse(response=f"Order details: {output.order_details}")
            ],
        )

    def _process_command(self, workflow: Workflow, input: Signature.Input) -> Signature.Output:
        data = load_data()
        order_details = GetOrderDetails.invoke(data=data, order_id=input.order_id)
        return Signature.Output(order_details=order_details) 