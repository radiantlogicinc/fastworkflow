import fastworkflow
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from pydantic import BaseModel, Field

from ...application.chatroom import ChatRoom

# the signature class defines our intent
class Signature:
    class Output(BaseModel):
        user_name: str = Field(
            description="Name of a person",
            examples=['John', 'Jane Doe']
        )

    plain_utterances = [
        "who is the current user?",
    ]

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> list[str]:
        """This function will be called by the framework to generate utterances for training"""
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)


class ResponseGenerator:
    """Call the User.send_message instance method."""

    def _process_command(self, workflow: fastworkflow.Workflow) -> Signature.Output:
        chatroom: ChatRoom = workflow.command_context_for_response_generation
        if chatroom.current_user:
            return Signature.Output(user_name=chatroom.current_user.name)
        else:
            return Signature.Output(user_name='Anonymous')

    def __call__(self, workflow: 
                 fastworkflow.Workflow, 
                 command: str) -> fastworkflow.CommandOutput:
        """The framework will call this function to process the command"""
        output = self._process_command(workflow)
        
        response = (
            f'Response: {output.model_dump_json()}'
        )

        return fastworkflow.CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                fastworkflow.CommandResponse(response=response)
            ]
        )
