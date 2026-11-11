from typing import List

import fastworkflow
from pydantic import BaseModel, Field, ConfigDict, model_validator
from fastworkflow.workflow import Workflow

from ..retail_data import load_data
from ..tools.modify_pending_order_address import ModifyPendingOrderAddress


class Signature:
    """Modify pending order address"""
    class Input(BaseModel):
        order_id: str = Field(
            description="The order ID to modify",
            json_schema_extra={"from_command_output": "get_user_details"},
            pattern=r"^(#?[\w\d]+)$",
            examples=["#W0000000"],
        )
        address1: str = Field(description="First line of address", examples=["123 Main St"])
        address2: str = Field(description="Second line of address", examples=["Apt 1"])
        city: str = Field(description="City name", examples=["San Francisco"])
        state: str = Field(
            description="State code",
            pattern=r"^([A-Z]{2})$",
            examples=["CA"],
        )
        country: str = Field(description="Country name", examples=["USA"])
        zip: str = Field(
            description="ZIP code",
            pattern=r"^(\d{5})$",
            examples=["12345"],
        )

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

        @model_validator(mode="after")
        def order_id_has_hash(self, info):
            if not self.order_id.startswith('#'):
                self.order_id = f'#{self.order_id}'
            return self

    class Output(BaseModel):
        status: str = Field(description="Whether modification succeeded.")

    plain_utterances: List[str] = [
        "Can I update the shipping address for my order?",
        "I need to change the delivery address on an order I just placed.",
        "Please modify the address for my pending order.",
        "I made a mistake in the shipping details — can I fix it?",
        "I'd like to change where my order is being delivered.",
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
        result = ModifyPendingOrderAddress.invoke(
            data=data,
            order_id=input.order_id,
            address1=input.address1,
            address2=input.address2,
            city=input.city,
            state=input.state,
            country=input.country,
            zip=input.zip,
        )
        return Signature.Output(status=result) 