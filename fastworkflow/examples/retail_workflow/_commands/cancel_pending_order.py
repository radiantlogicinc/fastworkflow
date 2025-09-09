# common imports
from pydantic import BaseModel
from fastworkflow.workflow import Workflow

# For command metadata extraction
import os
from typing import Annotated
from pydantic import Field, ConfigDict
import fastworkflow
from fastworkflow.train.generate_synthetic import generate_diverse_utterances

# For response generation
from fastworkflow import CommandOutput, CommandResponse
from ..retail_data import load_data 
from ..tools.cancel_pending_order import CancelPendingOrder


class Signature:
    """Cancel pending orders"""
    class Input(BaseModel):
        order_id: Annotated[
            str,
            Field(
                default="NOT_FOUND",
                description="The order ID to cancel",
                pattern=r"^(#?[\w\d]+|NOT_FOUND)$",
                examples=["123", "#abc123", "order456"],
                json_schema_extra={
                    "available_from": ["get_user_details"]
                }
            )
        ]

        reason: Annotated[
            str,
            Field(
                default="ordered by mistake",
                description="Reason for cancellation. If reason is invalid, use the default reason value",
                json_schema_extra={
                    "enum": ["no longer needed", "ordered by mistake"]
                },
                examples=["no longer needed", "ordered by mistake"]
            )
        ]

        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            validate_assignment=True
        )

    class Output(BaseModel):
        status: str = Field(
            description="whether cancellation succeeded)",
        )

    plain_utterances = [
        "I want to cancel my order because I no longer need it.",
        "Please cancel order #W1234567 — I ordered it by mistake.",
        "Can you cancel my pending order?",
        "I made a mistake and need to cancel an order I just placed.",
        "Cancel my order, I don't need it anymore.",
        "I accidentally placed an order — can you help me cancel it?",
        "Please stop processing order #W0000001, I no longer need the items.",
        "I'd like to cancel my order before it's shipped.",
        "I want to cancel a pending order — reason: ordered by mistake.",
        "Can I cancel my order? I changed my mind and don't need it."
    ]

    template_utterances = []

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> list[str]:
        utterance_definition = fastworkflow.RoutingRegistry.get_definition(workflow.folderpath)
        utterances_obj = utterance_definition.get_command_utterances(command_name)

        command_name = os.path.splitext(os.path.basename(__file__))[0]
        return generate_diverse_utterances(
            utterances_obj.plain_utterances, command_name
        )

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
        command_parameters: Signature.Input
    ) -> CommandOutput:
        output = self._process_command(workflow, command_parameters)
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                CommandResponse(response=f"current status is: {output.status}")
            ]
        )

    def _process_command(self,
        workflow: Workflow, input: Signature.Input
    ) -> Signature.Output:
        """
        get the review status of the entitlements in this workitem.

        :param input: The input parameters for the function.

        return the review status of the entitlements for the current workitem if workitem_path and workitem_id are not provided.
            if workitem_id is specified, the workitem_path must be specified.
        """
        data=load_data()

        # Call CancelPendingOrder's invoke method
        result = CancelPendingOrder.invoke(
            data=data,
            order_id=input.order_id,  # Assuming Signature.Input has order_id
            reason=input.reason      # Assuming Signature.Input has reason
        )
        
        return Signature.Output(status=result)