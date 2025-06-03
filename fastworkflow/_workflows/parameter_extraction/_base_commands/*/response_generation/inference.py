from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.session import Session

from ..response_generation.command_implementation import process_command


class ResponseGenerator:
    def __call__(
        self, session: Session, command: str
    ) -> CommandOutput:
        output_of_process_command = process_command(session, command)

        if output_of_process_command.parameter_is_valid:
            session.workflow_snapshot.workflow.is_complete = True

            return CommandOutput(
                command_responses=[
                    CommandResponse(
                        response="",
                        artifacts={
                            "command_name": output_of_process_command.command_name,
                            "cmd_parameters": output_of_process_command.cmd_parameters,
                        },
                    )
                ]
            )

        return CommandOutput(
            command_responses=[
                CommandResponse(
                    response=(
                        f"PARAMETER EXTRACTION ERROR: "
                        f"{output_of_process_command.error_msg}"
                    ),
                    success=False
                )
            ]
        )
