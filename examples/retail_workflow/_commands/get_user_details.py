from typing import List

import fastworkflow
from pydantic import BaseModel, Field, ConfigDict
from fastworkflow.session import Session
from fastworkflow import CommandOutput, CommandResponse

from ..retail_data import load_data
from ..tools.get_user_details import GetUserDetails


class Signature:
    """Get user details"""
    class Input(BaseModel):
        user_id: str = Field(
            default="NOT_FOUND",
            description="The user ID to get details for",
            pattern=r"^([a-z]+_[a-z]+_\d+|NOT_FOUND)$",
            examples=["sara_doe_496"],
        )

        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        status: str = Field(description="User details payload.")

    plain_utterances: List[str] = [
        "Can you pull up my account info?",
        "I want to see all the orders linked to my profile.",
        "What details do you have on my user account?",
        "Can you show me everything tied to my profile?",
        "I'd like to review my account and recent activity.",
    ]
    template_utterances: List[str] = []

    @staticmethod
    def generate_utterances(session: fastworkflow.Session, command_name: str) -> List[str]:
        utterance_definition = fastworkflow.UtteranceRegistry.get_definition(session.workflow_snapshot.workflow_folderpath)
        utterances_obj = utterance_definition.get_command_utterances(command_name)
        from fastworkflow.train.generate_synthetic import generate_diverse_utterances
        return generate_diverse_utterances(utterances_obj.plain_utterances, command_name)


class ResponseGenerator:
    def __call__(self, session: Session, command: str, command_parameters: Signature.Input) -> CommandOutput:
        output = self._process_command(session, command_parameters)
        return CommandOutput(
            session_id=session.id,
            command_responses=[CommandResponse(response=f"User details: {output.status}")],
        )

    def _process_command(self, session: Session, input: Signature.Input) -> Signature.Output:
        data = load_data()
        result = GetUserDetails.invoke(data=data, user_id=input.user_id)
        return Signature.Output(status=result) 