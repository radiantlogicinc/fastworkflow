from typing import Optional, Union
from pydantic import BaseModel, Field, ConfigDict

import fastworkflow
from fastworkflow import CommandOutput, CommandResponse, Action
from fastworkflow.session import Session, WorkflowSnapshot
from fastworkflow.utils.signatures import InputForParamExtraction
from fastworkflow.train.generate_synthetic import generate_diverse_utterances

from .get_status import Signature as GetStatusSignature, ResponseGenerator as GetStatusResponseGenerator


class Signature:
    class Input(BaseModel):
        workitem_path: str = Field(
            default="NOT_FOUND", 
            description="The workitem type",
            pattern=r"^(//[^/]+|/[^/]+(?:/[^/]+)*|[^/]+(?:/[^/]+)*)$",
            examples=[
                "/<workflow_name>/<workitem_name>",
                "<another_workitem_name>",
                "//<another_path_name>",
            ], 
            json_schema_extra={
                "db_lookup": True
            }
        )

        workitem_id: Optional[Union[str, int]] = Field(
            default=None,
            description="The ID of the workitem", 
            examples=["John Doe", "24"]
        )
        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        target_workitem_found: bool
        status_of_target_workitem: str

    # Constants from plain_utterances.json
    plain_utterances = [
        "Go to",
        "Move to",
        "Lets switch to",
        "Switch to",
        "jump to",
        "Move to the work item at",
        "Can you switch to the work item at",
        "Go to the work item located at",
        "Switch to the work item identified by",
        "move to work item",
        "move to task",
        "move to step",
        "move to stage"
    ]

    # Constants from template_utterances.json
    template_utterances = [
        "Go to {workitem_path}",
        "Go to the work item at {workitem_path} with ID {workitem_id}",
        "Move to {workitem_path}",
        "Lets switch to {workitem_path}",
        "Switch to {workitem_path}",
        "jump to {workitem_id} under {workitem_path}",
        "Move to the work item at {workitem_path}",
        "Can you switch to the work item at {workitem_path} with ID {workitem_id}?",
        "Go to the work item located at {workitem_path}",
        "Switch to the work item identified by {workitem_id} at {workitem_path}"
    ]

    @staticmethod
    def generate_utterances(session: Session, command_name: str) -> list[str]:
        workflow = session.workflow_snapshot.workflow
        utterance_definition = fastworkflow.UtteranceRegistry.get_definition(workflow.workflow_folderpath)
        utterances_obj = utterance_definition.get_command_utterances(
            workflow.path, command_name
        )
        result = generate_diverse_utterances(utterances_obj.plain_utterances, command_name)
        
        utterance_list: list[str] = [command_name] + result

        # Generate inputs for all possible workitem paths
        workflow_definition = fastworkflow.WorkflowRegistry.get_definition(workflow.workflow_folderpath)
        workitem_paths = []
        workitem_paths.extend(
            f"{workitem_path}"
            for workitem_path in workflow_definition.paths_2_typemetadata
        )
        # add full paths
        workitem = workflow.next_workitem(skip_completed=False)
        while workitem is not None:
            workitem_paths.append(workitem.path)
            workitem = workitem.next_workitem(skip_completed=False)

        inputs = [
            Signature.Input(workitem_path=workitem_path, workitem_id=None)
            for workitem_path in workitem_paths
        ]

        for input in inputs:
            kwargs = {field: getattr(input, field) for field in input.model_fields}
            for template in utterances_obj.template_utterances:
                utterance = template.format(**kwargs)
                utterance_list.append(utterance)

        return utterance_list

    def db_lookup(self, workflow_snapshot: WorkflowSnapshot, command: str) -> list[str]:
        workflow_folderpath = workflow_snapshot.workflow.workflow_folderpath
        workflow_definition = fastworkflow.WorkflowRegistry.get_definition(workflow_folderpath)
        return list(workflow_definition.paths_2_typemetadata.keys())

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ResponseGenerator:
    def _process_command(
        self, session: Session, input: Signature.Input
    ) -> Signature.Output:
        """
        Move to the work-item specified by the given path and optional id.

        :param input: The input parameters for the function.
        """
        workitem = session.workflow_snapshot.workflow.find_workitem(
            input.workitem_path, input.workitem_id
        )

        target_workitem_found = workitem is not None
        if target_workitem_found:
            session.workflow_snapshot.active_workitem = workitem

        active_workitem = session.workflow_snapshot.active_workitem

        get_status_tool_output = GetStatusResponseGenerator()._process_command(
            session,
            GetStatusSignature.Input(
                workitem_path=active_workitem.path, workitem_id=active_workitem.id
            ),
        )
        return Signature.Output(
            target_workitem_found=target_workitem_found,
            status_of_target_workitem=get_status_tool_output.status,
        )

    def __call__(
        self,
        session: Session,
        command: str,
        command_parameters: Signature.Input
    ) -> CommandOutput:
        output = self._process_command(session, command_parameters)

        return CommandOutput(
            session_id=session.id,
            command_responses=[
                CommandResponse(
                    response=(
                        f"move to workitem succeeded: {output.target_workitem_found}\n"
                        f"active workitem status: {output.status_of_target_workitem}"
                    ),
                    next_actions=[
                        Action(
                            session_id=session.id,
                            workitem_path="/sample_workflow",
                            command_name="move_to",
                            command="Move to mytask",
                            parameters={"workitem_path": "mytask", "workitem_id": None},
                        )
                    ],
                )
            ]
        ) 