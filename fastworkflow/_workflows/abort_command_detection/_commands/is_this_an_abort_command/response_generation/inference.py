import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.session import Session


class ResponseGenerator:
    def __call__(
        self, session: Session, command: str
    ) -> CommandOutput:
        workitem_type = session.workflow_snapshot.get_active_workitem().type
        workflow_folderpath = session.workflow_snapshot.workflow.workflow_folderpath
        route_layer = fastworkflow.RouteLayerRegistry.get_route_layer(workflow_folderpath, workitem_type)

        # Use semantic router to decipher the command name
        command_name = route_layer(command).name
        return CommandOutput(
            session_id=session.id,
            command_responses=[
                CommandResponse(
                    response="Command aborted",
                    success=False,  # necessary! commandexecutor checks success to stop further processing
                    artifacts={"abort": command_name is not None},
                )
            ]
        )


# if __name__ == "__main__":
