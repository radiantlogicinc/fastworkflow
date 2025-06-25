import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.session import Session
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from pydantic import BaseModel, ConfigDict, Field


class Signature:
    plain_utterances = [
        "That is not what I meant",
        "Not what I asked",
        "You misunderstood",
        "None of these commands",
        "Incorrect command",
        "Wrong command",
        "Change command",
        "Different command",
    ]

    class Output(BaseModel):
        valid_command_names: list[str]

    @staticmethod
    def generate_utterances(session: Session, command_name: str) -> list[str]:
        return generate_diverse_utterances(Signature.plain_utterances, command_name)


class ResponseGenerator:
    def _process_command(self, session: Session) -> Signature.Output:
        sub_sess = session.workflow_context["subject_session"]  #type: fastworkflow.Session
        subject_crd = fastworkflow.RoutingRegistry.get_definition(
            sub_sess.workflow_folderpath)
        
        crd = fastworkflow.RoutingRegistry.get_definition(
            session.workflow_folderpath)
        cme_command_names = crd.get_command_names('IntentDetection')

        fully_qualified_command_names = (
            set(cme_command_names) | 
            set(subject_crd.get_command_names(sub_sess.current_command_context_name))
        ) - {'wildcard'}

        valid_command_names = [
            fully_qualified_command_name.split('/')[-1] 
            for fully_qualified_command_name in fully_qualified_command_names
        ]

        return Signature.Output(valid_command_names=sorted(valid_command_names))

    def __call__(self, session: Session, command: str) -> CommandOutput:
        output = self._process_command(session)

        response = (
            "\n".join([
                f"{command_name}"
                for command_name in output.valid_command_names
            ])
        )
        response = (
            "Please enter the correct command from the list below:\n"
            f"{response}\n\nor type 'abort' to cancel"
        )

        return CommandOutput(
            session_id=session.id,
            command_responses=[
                CommandResponse(
                    response=response,
                    artifacts=output.model_dump(),
                )
            ],
        ) 