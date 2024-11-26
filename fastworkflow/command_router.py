import os

import fastworkflow
from fastworkflow.command_name_prediction import guess_command_name
from fastworkflow.command_interfaces import CommandRouterInterface
from fastworkflow.command_executor import CommandExecutor

class CommandRouter(CommandRouterInterface):
    def route_command(
        self,
        workflow_session: 'fastworkflow.WorkflowSession',
        command: str,
    ) -> fastworkflow.CommandOutput:
        # command_output = _was_command_name_misunderstood(session, command)
        # if command_output.command_responses[0].artifacts["command_name_misunderstood"]:
            # misunderstood_command = 
            # command = ""

        cnp_workflow_session_id, command_output = guess_command_name(
            workflow_session=workflow_session,
            command=command,
        )
        current_session_id = fastworkflow.WorkflowSession.get_active_session_id()
        if current_session_id == cnp_workflow_session_id:
            return command_output
        if command_output.command_aborted:
            # return command output right away only if abort is not a valid command in the current session
            # otherwise let the current session processing continue
            # (e.g. abort command in parameter extraction session)
            workflow_folderpath = workflow_session.session.workflow_snapshot.workflow.workflow_folderpath
            command_routing_definition = fastworkflow.CommandRoutingRegistry.get_definition(workflow_folderpath)
            workitem_type = workflow_session.session.workflow_snapshot.active_workitem.type
            valid_command_names = command_routing_definition.get_command_names(workitem_type)
            if "abort" not in valid_command_names:
                return command_output

        # if misunderstood_command:
        #     command = misunderstood_command

        command_name = command_output.command_responses[0].artifacts["command_name"]
        command = command_output.command_responses[0].artifacts["command"]
        command_executor = CommandExecutor()
        return command_executor.invoke_command(
            workflow_session,
            command_name, 
            command
        )

def _was_command_name_misunderstood(
    session: fastworkflow.Session,
    command: str,
) -> fastworkflow.CommandOutput:
    fastworkflow_folder = os.path.dirname(os.path.abspath(__file__))
    misunderstanding_detection_workflow_folderpath = os.path.join(
        fastworkflow_folder, "_workflows", "misunderstood_command_detection"
    )

    session = fastworkflow.Session.create(
        misunderstanding_detection_workflow_folderpath,
        parent_session_id=session.id
    )

    startup_action = fastworkflow.Action(
        workitem_type="misunderstood_command_detection",
        command_name="was_command_name_misunderstood",
        command=command,
        session_id=session.id,
    )

    command_executor = CommandExecutor()
    return command_executor.perform_action(session, startup_action)
