from typing import Any
from pydantic import BaseModel, Field, model_validator

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
        )

        @model_validator(mode="after")
        def user_name_is_in_the_room(self, info):
            workflow = info.context.get("workflow")
            chatroom: ChatRoom = workflow.record
            key_values = chatroom.list_users()
            matched, corrected_value, suggestions = DatabaseValidator.fuzzy_match(self.user_name, key_values)
            if matched:
                self.user_name = corrected_value
            elif suggestions:
                raise fastworkflow.Refused(
                    f"No user named '{self.user_name}' in this chat room. Did you mean {', '.join(suggestions)}?",
                    field="user_name")
            return self

    class Output(BaseModel):
        user_found: bool = Field(
            description="Whether we found the user",
        )

    plain_utterances = [
        "Set Fred as the current user",
    ]

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> list[str]:
        """This function will be called by the framework to generate utterances for training"""
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)


class ResponseGenerator:
    """Call the User.send_message instance method."""

    def _process_command(self, workflow: fastworkflow.Workflow, input: Signature.Input) -> Signature.Output:
        chatroom: ChatRoom = workflow.record

        if input.user_name not in chatroom.list_users():
            return Signature.Output(user_found=False)

        for user in chatroom.users:
            if user.name == input.user_name:
                break

        chatroom.current_user = user

        # lets change the current context to this user
        workflow.record = user

        return Signature.Output(user_found=True)

    def __call__(self, workflow: 
                 fastworkflow.Workflow, 
                 command: str, 
                 command_parameters: Signature.Input) -> Signature.Output:
        """The framework will call this function to process the command"""
        output = self._process_command(workflow, command_parameters)
        return output
