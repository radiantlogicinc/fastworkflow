import json
from pydantic import BaseModel, ConfigDict

from fastworkflow import CommandOutput, CommandResponse
import fastworkflow
from fastworkflow.session import Session
from fastworkflow.train.generate_synthetic import generate_diverse_utterances


class Signature:
    class Input(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        success: bool
        num_of_todos: int

    # Constants from plain_utterances.json
    plain_utterances = [
        "Save to do list",
        "Save this to do list",
        "save this todo list",
        "save todos"
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
        return [command_name] + result


class ResponseGenerator:
    def _process_command(
        self, session: Session, input: Signature.Input
    ) -> Signature.Output:
        """Command implementation for save_todo_list"""
        workflow = session.workflow_snapshot.workflow
        
        # Get all todos from the workflow
        todo_list = []
        workitem = workflow.next_workitem(skip_completed=False)
        while workitem is not None:
            if workitem.path == "/todo_list/todo":
                todo_list.append({
                    "id": workitem.id,
                    "description": workitem.description,
                    "assign_to": workitem.assign_to,
                    "status": "COMPLETE" if workitem.is_complete else "INCOMPLETE"
                })
            workitem = workitem.next_workitem(skip_completed=False)

        # Save to JSON file
        workflow_folderpath = workflow.workflow_folderpath
        with open(f"{workflow_folderpath}/todo_list.json", 'w') as file:
            json.dump({"todo_list": todo_list}, file, indent=4)

        return Signature.Output(
            success=True, num_of_todos=len(todo_list)
        )

    def __call__(
        self,
        session: Session,
        command: str,
        command_parameters: Signature.Input
    ) -> CommandOutput:
        output = self._process_command(session, command_parameters)

        # Create the response
        response = f"Successfully saved {output.num_of_todos} todos"

        return CommandOutput(
            session_id=session.id,
            command_responses=[
                CommandResponse(response=response)
            ]
        ) 