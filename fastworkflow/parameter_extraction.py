import os

import fastworkflow
from fastworkflow.command_executor import CommandExecutor

def extract_command_parameters(
    workflow_session: fastworkflow.WorkflowSession,
    command_name: str,
    command: str,
) -> fastworkflow.CommandOutput:
    """
    This function is called when the command parameters are invalid.
    It extracts the command parameters from the command using DSPy.
    If the extraction fails, it starts the extraction failure workflow.
    Note: this function is called from the regular workflow as well as the extraction failure workflow.
    If it is called from the extraction failure workflow, the session is the source workflow session which is basically what we want.
    """
    startup_action = fastworkflow.Action(
        workitem_type="parameter_extraction",
        command_name="*",
        command=command,
    )

    # if we are already in the parameter extraction workflow, we can just perform the action
    if workflow_session.session.workflow_snapshot.workflow.type == "parameter_extraction":
        command_executor = CommandExecutor()
        command_output = command_executor.perform_action(workflow_session.session, startup_action)
        if len(command_output.command_responses) > 1:
            raise ValueError("Multiple command responses returned from parameter extraction workflow")    
        return (workflow_session.session.id, command_output)    

    fastworkflow_folder = os.path.dirname(os.path.abspath(__file__))
    parameter_extraction_workflow_folderpath = os.path.join(
        fastworkflow_folder, "_workflows", "parameter_extraction"
    )

    context = {
        "subject_command_name": command_name,
        "subject_workflow_snapshot": workflow_session.session.workflow_snapshot
    }

    pe_workflow_session = fastworkflow.WorkflowSession(
        workflow_session.command_router,
        workflow_session.command_executor,
        parameter_extraction_workflow_folderpath, 
        parent_session_id=workflow_session.session.id, 
        context=context,
        startup_action=startup_action, 
        user_message_queue=workflow_session.user_message_queue,
        command_output_queue=workflow_session.command_output_queue,
    )

    return pe_workflow_session.start()