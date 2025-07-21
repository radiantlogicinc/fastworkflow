from typing import Any
from pydantic import BaseModel, Field

import fastworkflow
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from fastworkflow.utils.signatures import DatabaseValidator

from ...application.chatroom import ChatRoom

# the signature class defines our intent
class Signature:
    class Input(BaseModel):
        user_name: str = Field(
            description="Name of a person",
            examples=['John', 'Jane Doe'],
            json_schema_extra={'db_lookup': True}
        )

    class Output(BaseModel):
        user_found: bool = Field(
            description="Whether we found the user",
        )

    plain_utterances = [
        "Set Fred as the current user",
    ]

    @staticmethod
    def db_lookup(workflow: fastworkflow.Workflow, 
                  field_name: str, 
                  field_value: str
                  ) -> tuple[bool, str | None, list[str]]:
        if field_name == 'user_name':
            chatroom: ChatRoom = workflow.command_context_for_response_generation
            key_values = chatroom.list_users()
            matched, corrected_value, field_value_suggestions = DatabaseValidator.fuzzy_match(field_value, key_values)
            return (matched, corrected_value, field_value_suggestions)
        return (False, '', [])

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> list[str]:
        """This function will be called by the framework to generate utterances for training"""
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)


class ResponseGenerator:
    """Call the User.send_message instance method."""

    def _process_command(self, workflow: fastworkflow.Workflow, input: Signature.Input) -> Signature.Output:
        chatroom: ChatRoom = workflow.command_context_for_response_generation

        if input.user_name not in chatroom.list_users():
            return Signature.Output(user_found=False)

        for user in chatroom.users:
            if user.name == input.user_name:
                break

        chatroom.current_user = user

        # lets change the current context to this user
        workflow.current_command_context = user

        return Signature.Output(user_found=True)

    def __call__(self, workflow: 
                 fastworkflow.Workflow, 
                 command: str, 
                 command_parameters: Signature.Input) -> fastworkflow.CommandOutput:
        """The framework will call this function to process the command"""
        output = self._process_command(workflow, command_parameters)
        
        response = (
            f'Response: {output.model_dump_json()}'
        )

        return fastworkflow.CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                fastworkflow.CommandResponse(response=response)
            ]
        )
