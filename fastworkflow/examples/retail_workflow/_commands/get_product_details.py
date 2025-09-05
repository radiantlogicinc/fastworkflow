from typing import List

import fastworkflow
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.workflow import Workflow
from fastworkflow import CommandOutput, CommandResponse

from ..retail_data import load_data
from ..tools.get_product_details import GetProductDetails


class Signature:
    """Get product details"""
    class Input(BaseModel):
        product_id: str = Field(
            default="NOT_FOUND",
            description="The product ID (numeric string)",
            pattern=r"^(\d{10}|NOT_FOUND)$",
            examples=["6086499569"],
            json_schema_extra={
                "available_from": ["get_order_details", "list_all_product_types"]
            }
        )

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        product_details: str = Field(
            description=(
                "Detailed product information of all product variants of a given product type"
                "including item id, options, availability and price"
            ),
            json_schema_extra={
                "used_by": ["exchange_delivered_order_items", "modify_pending_order_items"]
            }
        )

    plain_utterances: List[str] = [
        "Can you give me more information about this product?",
        "I need the specs and availability for an item I saw earlier.",
        "What are the details of that product with the long ID?",
        "Is this product still in stock and what are its features?",
        "Can you check if this product is available and tell me more about it?",
    ]

    template_utterances: List[str] = []

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> List[str]:
        utterance_definition = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)
        utterances_obj = utterance_definition.get_command_utterances(command_name)

        from fastworkflow.train.generate_synthetic import generate_diverse_utterances
        return generate_diverse_utterances(utterances_obj.plain_utterances, command_name)


class ResponseGenerator:
    def __call__(self, workflow: Workflow, command: str, command_parameters: Signature.Input) -> CommandOutput:
        output = self._process_command(workflow, command_parameters)
        response = (
            f'Response: {output.product_details}'
        )
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[CommandResponse(response=response)],
        )

    def _process_command(self, workflow: Workflow, input: Signature.Input) -> Signature.Output:
        data = load_data()
        product_details = GetProductDetails.invoke(data=data, product_id=input.product_id)
        return Signature.Output(product_details=product_details) 