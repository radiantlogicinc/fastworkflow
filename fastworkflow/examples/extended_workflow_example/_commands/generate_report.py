import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from pydantic import BaseModel

class Signature:
    class Input(BaseModel):
        report_type: str = "summary"
        include_details: bool = True

    plain_utterances = [
        "create a report"
    ]

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> list[str]:
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)


class ResponseGenerator:
    def __call__(self, workflow: fastworkflow.Workflow, command: str, cmd_parameters: Signature.Input) -> CommandOutput:
        # This is a new command specific to the extended workflow
        report_data = {
            "report_type": cmd_parameters.report_type,
            "current_context": str(type(workflow.root_command_context).__name__) if workflow.root_command_context else "None",
            "include_details": cmd_parameters.include_details,
            "generated_at": "2024-01-15 10:30:00",
            "total_workitems": self._count_workitems(workflow.root_command_context) if workflow.root_command_context else 0
        }
        
        if cmd_parameters.include_details and workflow.root_command_context:
            report_data["workitem_details"] = self._get_workitem_details(workflow.root_command_context)
        
        response = {
            "message": f"Generated {cmd_parameters.report_type} report",
            "report": report_data
        }
        
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                CommandResponse(response=str(response))
            ]
        )
    
    def _count_workitems(self, workitem) -> int:
        """Count total number of workitems in the hierarchy"""
        count = 1
        if hasattr(workitem, 'child_workitems') and workitem.child_workitems:
            for child in workitem.child_workitems:
                count += self._count_workitems(child)
        return count
    
    def _get_workitem_details(self, workitem) -> dict:
        """Get details about the workitem hierarchy"""
        details = {
            "type": workitem.type,
            "title": getattr(workitem, 'title', 'Untitled'),
            "status": getattr(workitem, 'status', 'Unknown'),
        }
        
        if hasattr(workitem, 'child_workitems') and workitem.child_workitems:
            details["children"] = [
                self._get_workitem_details(child) 
                for child in workitem.child_workitems
            ]
            
        return details
