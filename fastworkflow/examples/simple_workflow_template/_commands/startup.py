import fastworkflow
from ..application.workitem import WorkItem

class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str, input: None) -> str:
        filepath = (
            f'{workflow.folderpath}/'
            'simple_workflow_template.json'
        )
        workflow_schema = WorkItem.WorkflowSchema.from_json_file(filepath)
        workflow.root_command_context = workflow_schema.create_workitem("Epic")

        response = {
            "message": "Application initialized.",
        }

        return str(response)
