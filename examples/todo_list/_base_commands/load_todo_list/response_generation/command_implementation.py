import json

from pydantic import BaseModel

from fastworkflow.session import Session


class CommandProcessorOutput(BaseModel):
    success:bool
    num_of_todos:int

def process_command(
    session: Session
) -> CommandProcessorOutput:
    """Command implementation for load_todo_list"""
    workflow = session.workflow_snapshot.workflow
    
    # Reading from a JSON file
    workflow_folderpath = workflow.workflow_folderpath
    with open(f"{workflow_folderpath}/todo_list.json", 'r') as file:
        data = json.load(file)
    todo_dictlist=data["todo_list"]

    for todo_dict in todo_dictlist:
        todo=workflow.add_workitem("/todo_list/todo",todo_dict["id"])
        todo.is_complete = todo_dict["status"] == "COMPLETE"

    session.workflow_snapshot.workflow = workflow

    return CommandProcessorOutput(
        success=True,num_of_todos=len(todo_dictlist)
    )