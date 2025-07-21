import fastworkflow
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from pydantic import BaseModel, Field

from ...application.chatroom import ChatRoom
from ...application.user import User, PremiumUser

# the signature class defines our intent
class Signature:
    class Input(BaseModel):
        user_name: str = Field(
            description="Name of a person",
            examples=['John', 'Jane Doe'],
            min_length=3,
            max_length=20
        )
        is_premium_user: bool = Field(
            description="Whether this is a premium user",
        )

    class Output(BaseModel):
        user_added: bool = Field(
            description="Whether the user was added",
        )

    plain_utterances = [
        "Add Fred to our list of users",
        "Update the chatroom with new premium user John Smith",
        "We have a new regular user Mary Jane",
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
        chatroom: ChatRoom = workflow.command_context_for_response_generation

        if input.user_name in chatroom.list_users():
            return Signature.Output(user_added=False)

        chatroom.add_user(
            PremiumUser(chatroom, input.user_name) 
            if input.is_premium_user else
            User(chatroom, input.user_name)
        )

        return Signature.Output(user_added=True)

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
