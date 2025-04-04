import fastworkflow
from fastworkflow import CommandOutput, CommandResponse

from ..response_generation.command_implementation import process_command

class ResponseGenerator:
    def __call__(
        self, 
        session: fastworkflow.Session, 
        command: str,
    ) -> CommandOutput:
        output_of_process_command = process_command(session, command)

        if output_of_process_command.command_name:
            session.workflow_snapshot.workflow.is_complete = True

            return CommandOutput(
                command_responses=[
                    CommandResponse(
                        response="command aborted" if output_of_process_command.command_name == "abort" else "",
                        artifacts={
                            "command": output_of_process_command.command,
                            "command_name": output_of_process_command.command_name,
                        },
                    )
                ]
            )

        return CommandOutput(
            command_responses=[
                CommandResponse(
                    response=output_of_process_command.error_msg,
                    success=False
                )
            ]
        )
