import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
from ..application.todo_manager import TodoListManager

class ResponseGenerator:
    def __call__(self, session: fastworkflow.Session, command: str) -> CommandOutput:
        filepath = (
            f'{session.workflow_snapshot.workflow_folderpath}/'
            'application/'
            'todo_list.json'
        )
        session.root_command_context = TodoListManager(filepath)

        return CommandOutput(
            session_id=session.id,
            command_responses=[
                CommandResponse(
                    response = (
                        "Application initialized. "
                        f"Current command context={session.current_command_context_name}"
                    )
                )
            ]
        )
