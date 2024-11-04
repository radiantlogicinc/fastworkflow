import os
from typing import Optional

from semantic_router import RouteLayer

from fastworkflow.command_executor import CommandExecutor, CommandOutput
from fastworkflow.command_name_extraction import extract_command_name
from fastworkflow.session import Session


class CommandRouter:
    def __init__(self, session: Session):
        self._session = session
        self._payload = {}

    def route_command(
        self,
        map_workitem_type_2_route_layer: dict[str, RouteLayer],
        command: str,
        payload: Optional[dict] = None,
    ) -> CommandOutput:
        if payload:
            self._payload = payload
        else:
            self._payload["error_msg"] = None if "error_msg" in self._payload else ...

        extraction_failure_workflow = (
            None
            if "parameter_extraction" in self._session.workflow_folderpath
            else "parameter_extraction"
        )

        abort_command, command_name = extract_command_name(
            session=self._session,
            map_workitem_type_2_route_layer=map_workitem_type_2_route_layer,
            command=command,
            extraction_failure_workflow=extraction_failure_workflow,
        )

        if abort_command:
            return CommandOutput(
                success=False,
                response="Command aborted",
                payload={"abort_command": True},
            )

        active_workitem_type = self._session.get_active_workitem().type
        command_base = CommandExecutor(self._session)
        command_output = command_base.invoke_command(
            active_workitem_type, command_name, command, self._payload
        )

        return command_output
