import fastworkflow
from fastworkflow.examples.simple_workflow_template._commands.startup import (
    ResponseGenerator as BaseResponseGenerator
)

class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str, input: None) -> str:
        # Call the base startup logic first
        base_generator = BaseResponseGenerator()
        base_output = base_generator(workflow, command)
        
        # Add custom initialization for this extended workflow
        custom_response = {
            "message": "Extended workflow initialized with custom features!",
            "base_message": str(base_output.command_response.response),
            "extended_features": ["Custom reporting", "Enhanced analytics", "Advanced notifications"]
        }
        
        return str(custom_response)
