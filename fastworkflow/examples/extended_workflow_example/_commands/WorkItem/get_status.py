from pydantic import BaseModel, Field
import datetime

import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.examples.simple_workflow_template._commands.WorkItem.get_status import (
    Signature as BaseSig,
    ResponseGenerator as BaseRG
)
from fastworkflow.examples.simple_workflow_template.application.workitem import WorkItem


class Signature(BaseSig):
    """Enhanced version of get_status with additional analytics and formatting."""
    
    # Extend the plain utterances with new ways to ask for status
    plain_utterances = BaseSig.plain_utterances + [
        "show detailed status",
        "what's the current progress",
        "give me an enhanced status report",
        "show analytics for this workitem"
    ]
    
    class Output(BaseSig.Output):
        # Extend the output with additional fields
        enhanced_analytics: dict = Field(description="Additional analytics and insights")
        timestamp: str = Field(description="When the status was generated")


class ResponseGenerator:
    """Enhanced response generator that adds analytics to the base status."""
    
    def _process_command(self, workflow: fastworkflow.Workflow) -> Signature.Output:
        # Get base status using the parent functionality
        base_generator = BaseRG()
        base_output = base_generator._process_command(workflow)
        
        # Add enhanced analytics
        workitem: WorkItem = workflow.command_context_for_response_generation
        analytics = self._generate_analytics(workitem)
        
        return Signature.Output(
            status_dict=base_output.status_dict,
            enhanced_analytics=analytics,
            timestamp=datetime.datetime.now().isoformat()
        )
    
    def _generate_analytics(self, workitem: WorkItem) -> dict:
        """Generate additional analytics about the workitem."""
        analytics = {
            "completion_percentage": self._calculate_completion_percentage(workitem),
            "child_count": workitem.get_child_count(),
            "depth_level": self._calculate_depth(workitem),
            "workitem_type": workitem.type,
        }
        
        # Add child status breakdown if there are children
        if hasattr(workitem, 'child_workitems') and workitem.child_workitems:
            child_statuses = {}
            for child in workitem.child_workitems:
                status = child.status
                child_statuses[status] = child_statuses.get(status, 0) + 1
            analytics["child_status_breakdown"] = child_statuses
        
        return analytics
    
    def _calculate_completion_percentage(self, workitem: WorkItem) -> float:
        """Calculate completion percentage based on status and children."""
        if workitem.is_complete:
            return 100.0
        
        if child_count := workitem.get_child_count():
            completed_children = workitem.get_child_count(is_complete = True)
            return (completed_children / child_count) * 100.0
        else:
            return 50.0  # Assume 50% for in-progress items without children
    
    def _calculate_depth(self, workitem: WorkItem) -> int:
        """Calculate the depth of the workitem in the hierarchy."""
        depth = 0
        current = workitem
        while current.parent:
            depth += 1
            current = current.parent
        return depth
    
    def __call__(self, workflow: fastworkflow.Workflow, command: str) -> CommandOutput:
        output = self._process_command(workflow)
        
        # Format the enhanced status information
        response_data = {
            "base_status": output.status_dict,
            "analytics": output.enhanced_analytics,
            "generated_at": output.timestamp,
            "message": "Enhanced status report with analytics"
        }
        
        response = f'Enhanced Response: {response_data}'
        
        return CommandOutput(
            workflow_id=workflow.id,
            command_responses=[
                CommandResponse(response=response)
            ]
        )
