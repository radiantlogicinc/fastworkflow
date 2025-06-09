from typing import Optional, Union
from pydantic import BaseModel, Field, ConfigDict

import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.session import Session, WorkflowSnapshot
from fastworkflow.workflow_definition import NodeType
from fastworkflow.utils.signatures import InputForParamExtraction
from fastworkflow.train.generate_synthetic import generate_diverse_utterances


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
        status: str

    # Constants from plain_utterances.json
    plain_utterances = [
        "What is the current status of this work item?",
        "get status",
        "What is the status of this work flow?",
        "What is the status of this task?",
        "What is the status of this stage?",
        "What is the current status of this task?",
        "What is the current status of this step?",
        "get status"
    ]

    # Constants from template_utterances.json
    template_utterances = [
        "What is the status of the work item at {workitem_path}",
        "Can you tell me the status of the work item at {workitem_path} with ID {workitem_id}",
        "What is the current status of the work item at {workitem_path}",
        "Can you tell me the current status of the work item at {workitem_path} with ID {workitem_id}"
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
        workflow_folderpath = self.workflow_snapshot.workflow.workflow_folderpath
        workflow_definition = fastworkflow.WorkflowRegistry.get_definition(workflow_folderpath)

        return workflow_definition.paths_2_typemetadata.keys()

    def process_extracted_parameters(
        self, 
        workflow_snapshot: WorkflowSnapshot, 
        command: str, cmd_parameters: "Signature.Input"
    ) -> None:
        """
        This function gives you the chance to further process extracted parameters
        """
        if cmd_parameters.workitem_path == "NOT_FOUND":
            if active_workitem := workflow_snapshot.active_workitem:
                cmd_parameters.workitem_path = active_workitem.path
                cmd_parameters.workitem_id = active_workitem.id
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ResponseGenerator:
    def _process_command(
        self, session: Session, input: Signature.Input
    ) -> Signature.Output:
        """
        get the review status of the entitlements in this workitem.

        :param input: The input parameters for the function.

        return the review status of the entitlements for the current workitem if workitem_path and workitem_id are not provided.
            if workitem_id is specified, the workitem_path must be specified.
        """
        workitem = session.workflow_snapshot.active_workitem
        if workitem.node_type == NodeType.Workflow:
            status = (
                f"workitem_path: {workitem.path}, workitem_id: {workitem.id}\n"
                f"started: {workitem.has_started}, Complete: {workitem.is_complete}"
            )
        else:
            status = (
                f"workitem_path: {workitem.path}, workitem_id: {workitem.id}\n"
                f"Complete: {workitem.is_complete}"
            )

        return Signature.Output(status=status)

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
                CommandResponse(response=f"current status is: {output.status}")
            ]
        ) 