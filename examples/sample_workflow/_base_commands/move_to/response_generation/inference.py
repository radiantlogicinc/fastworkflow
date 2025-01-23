from fastworkflow import CommandOutput, CommandResponse, Action
from fastworkflow.session import Session

from ..parameter_extraction.signatures import CommandParameters
from .command_implementation import process_command


class ResponseGenerator:
    def __call__(
        self,
        session: Session,
        command: str,
        command_parameters: CommandParameters
    ) -> CommandOutput:
        output = process_command(session, command_parameters)

        return CommandOutput(
            session_id=session.id,
            command_responses=[
                CommandResponse(
                    response=(
                        f"move to workitem succeeded: {output.target_workitem_found}\n"
                        f"active workitem status: {output.status_of_target_workitem}"
                    ),
                    next_actions=[
                        Action(
                            session_id=session.id,
                            workitem_path="/sample_workflow",
                            command_name="move_to",
                            command="Move to mytask",
                            parameters={"workitem_path": "mytask", "workitem_id": None},
                        )
                    ],
                )
            ]
        )


# if __name__ == "__main__":
