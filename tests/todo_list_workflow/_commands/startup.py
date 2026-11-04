import fastworkflow
from ..application.todo_manager import TodoListManager

class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str, input: None) -> str:
        filepath = (
            f'{workflow.folderpath}/'
            'application/'
            'todo_list.json'
        )
        workflow.root_command_context = TodoListManager(filepath)
        
        response = {
            "message": "Application initialized.",
            "context": workflow.current_command_context_name
        }

        return str(response)
