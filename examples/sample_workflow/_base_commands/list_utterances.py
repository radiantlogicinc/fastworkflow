from pydantic import BaseModel, ConfigDict

import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.session import Session
from fastworkflow.train.generate_synthetic import generate_diverse_utterances


class Signature:
    class Input(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        workitem_path: str
        utterances: list[str]

    # Constants from plain_utterances.json
    plain_utterances = [
        "what can you do?",
        "what are my options?",
        "what are my choices?",
        "what are my capabilities?",
        "what can i do?",
        "what can i use?",
        "what are my tools?",
        "now what?",
        "list commands",
        "list utterances"
    ]

    # Constants from template_utterances.json
    template_utterances = []

    @staticmethod
    def generate_utterances(session: Session, command_name: str) -> list[str]:
        workflow = session.workflow_snapshot.workflow
        utterance_definition = fastworkflow.UtteranceRegistry.get_definition(workflow.workflow_folderpath)
        utterances_obj = utterance_definition.get_command_utterances(
            workflow.path, command_name
        )
        result = generate_diverse_utterances(utterances_obj.plain_utterances, command_name)
        
        utterance_list: list[str] = [command_name] + result

        return utterance_list


class ResponseGenerator:
    def _process_command(
        self, session: Session, input: Signature.Input
    ) -> Signature.Output:
        """
        Provides helpful information about this type of work-item.
        If the workitem_path is not provided, it provides information about the current work-item.

        :param input: The input parameters for the function.
        """
        # Get the current workitem type
        current_workitem = session.workflow_snapshot.active_workitem

        workflow_folderpath = session.workflow_snapshot.workflow.workflow_folderpath
        utterance_definition = fastworkflow.UtteranceRegistry.get_definition(workflow_folderpath)
        utterances = utterance_definition.get_sample_utterances(current_workitem.path)

        return Signature.Output(
            workitem_path=current_workitem.path, utterances=utterances
        )

    def __call__(
        self,
        session: Session,
        command: str,
        command_parameters: Signature.Input
    ) -> CommandOutput:
        output = self._process_command(session, command_parameters)

        # Format the list of commands
        utterance_list = "\n".join([f"- {cmd}" for cmd in output.utterances])

        # Create the response
        response = (
            f"Here are some example commands available in this task ({output.workitem_path}):\n"
            f"{utterance_list}\n"
            f"Your chat message must fall within the scope of these utterances."
        )

        return CommandOutput(
            session_id=session.id,
            command_responses=[
                CommandResponse(response=response)
            ]
        ) 