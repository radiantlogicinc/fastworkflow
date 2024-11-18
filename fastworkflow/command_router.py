import os
import random

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
        command_output = _is_this_an_abort_command(workflow_session.session, command)
        if command_output.command_responses[0].artifacts["abort"]:
            return command_output

        # command_output = _was_command_name_misunderstood(session, command)
        # if command_output.command_responses[0].artifacts["command_name_misunderstood"]:
            # misunderstood_command = 
            # command = ""

        # if we are already in the command name prediction workflow, we can skip this step
        if workflow_session.session.workflow_snapshot.workflow.type == "command_name_prediction":
            command_output = fastworkflow.CommandOutput(
                command_responses=[fastworkflow.CommandResponse(
                    response="",
                    artifacts={"command_name": "guess_command_name"},
                )],
            )
        elif workflow_session.session.workflow_snapshot.workflow.type == "parameter_extraction":
            command_output = fastworkflow.CommandOutput(
                command_responses=[fastworkflow.CommandResponse(
                    response="",
                    artifacts={"command_name": "extract_parameters"},
                )],
            )
        else:
            command_output = guess_command_name(
                workflow_session=workflow_session,
                command=command,
            )

        # if misunderstood_command:
        #     command = misunderstood_command

        if command_output.command_responses[0].success:
            command_name = command_output.command_responses[0].artifacts["command_name"]
            command_executor = CommandExecutor()
            command_output = command_executor.invoke_command(
                workflow_session,
                command_name, 
                command
            )

        return command_output

def _is_this_an_abort_command(
    session: fastworkflow.Session,
    command: str,
) -> fastworkflow.CommandOutput:
    fastworkflow_folder = os.path.dirname(os.path.abspath(__file__))
    abort_command_detection_workflow_folderpath = os.path.join(
        fastworkflow_folder, "_workflows", "abort_command_detection"
    )

    session = fastworkflow.Session.create(
                            -random.randint(1, 100000000), 
                            abort_command_detection_workflow_folderpath, 
                        )

    startup_action = fastworkflow.Action(
        workitem_type="abort_command_detection",
        command_name="is_this_an_abort_command",
        command=command,
        session_id=session.id,
    )

    command_executor = CommandExecutor()
    return command_executor.perform_action(session, startup_action)

def _was_command_name_misunderstood(
    session: fastworkflow.Session,
    command: str,
) -> fastworkflow.CommandOutput:
    fastworkflow_folder = os.path.dirname(os.path.abspath(__file__))
    misunderstanding_detection_workflow_folderpath = os.path.join(
        fastworkflow_folder, "_workflows", "misunderstood_command_detection"
    )

    session = fastworkflow.Session.create(
                            -random.randint(1, 100000000), 
                            misunderstanding_detection_workflow_folderpath, 
                        )

    startup_action = fastworkflow.Action(
        workitem_type="misunderstood_command_detection",
        command_name="was_command_name_misunderstood",
        command=command,
        session_id=session.id,
    )

    command_executor = CommandExecutor()
    return command_executor.perform_action(session, startup_action)
