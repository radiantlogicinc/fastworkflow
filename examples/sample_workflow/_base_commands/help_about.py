import dspy
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict

import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.session import Session, WorkflowSnapshot
from fastworkflow.utils.signatures import InputForParamExtraction
from fastworkflow.utils.dspy_logger import DSPyRotatingFileLogger, DSPyForward
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
        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        help_info: dict

    # Constants from plain_utterances.json
    plain_utterances = [
        "I need help",
        "I need help with this task type",
        "Can you help me with this stage type",
        "Describe this step type",
        "What is this workitem type about?",
        "Help me understand this workflow type",
        "help about this workitem type"
    ]

    # Constants from template_utterances.json
    template_utterances = [
        "I need help with {workitem_path} type",
        "Can you help me with {workitem_path} type",
        "I need help with {workitem_path} type work item",
        "Can you help me with {workitem_path} type work item"
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
        inputs = [
            Signature.Input(workitem_path=workitem_path)
            for workitem_path in workflow_definition.paths_2_typemetadata
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

    def process_extracted_parameters(
        self,
        workflow_snapshot: WorkflowSnapshot,
        command: str,
        cmd_parameters: "Signature.Input"
    ) -> None:
        """
        This function gives you the chance to further process extracted parameters
        """
        if cmd_parameters.workitem_path == "NOT_FOUND":
            if active_workitem := workflow_snapshot.active_workitem:
                cmd_parameters.workitem_path = active_workitem.path
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ResponseGenerator:
    def _process_command(
        self, session: Session, input: Signature.Input
    ) -> Signature.Output:
        """
        Provides helpful information about this type of work-item.
        If the workitem_path is not provided, it provides information about the current work-item.

        :param input: The input parameters for the function.
        """
        workitem_path = input.workitem_path
        if not workitem_path:
            workitem = session.workflow_snapshot.active_workitem
            workitem_path = workitem.path

        help_info = {"workitem_path": workitem_path, "allowable_child_types": {}}

        workflow_folderpath = session.workflow_snapshot.workflow.workflow_folderpath
        workflow_definition = fastworkflow.WorkflowRegistry.get_definition(workflow_folderpath)

        for (
            child_path,
            child_size_metadata,
        ) in workflow_definition.paths_2_allowable_child_paths_2_sizemetadata[workitem_path].items():
            help_info["allowable_child_types"][child_path] = {
                "min_size": child_size_metadata.min,
                "max_size": child_size_metadata.max,
            }

        return Signature.Output(help_info=help_info)

    def __call__(
        self,
        session: Session,
        command: str,
        command_parameters: Signature.Input
    ) -> CommandOutput:
        output = self._process_command(session, command_parameters)

        LLM_RESPONSE_GEN = fastworkflow.get_env_var("LLM_RESPONSE_GEN")
        LITELLM_API_KEY_RESPONSE_GEN = fastworkflow.get_env_var("LITELLM_API_KEY_RESPONSE_GEN")
        lm = dspy.LM(LLM_RESPONSE_GEN, api_key=LITELLM_API_KEY_RESPONSE_GEN)
        answer_generator = ResponseGenerator.BasicQA(lm)

        with DSPyRotatingFileLogger("inference_log.jsonl"):
            prediction = answer_generator(
                context=str(output.help_info), question=command
            )

        return CommandOutput(
            session_id=session.id,
            command_responses=[
                CommandResponse(
                    response=prediction.answer,
                )
            ]
        )

    class BasicQA(dspy.Module):
        """DSPy Module for answering help questions"""

        def __init__(self, lm: dspy.LM):
            super().__init__()

            self.lm = lm
            self.generate_answer = dspy.Predict("context, question -> answer")

        @DSPyForward.intercept
        def forward(self, context, question):
            """forward pass"""
            with dspy.context(lm=self.lm):
                return self.generate_answer(context=context, question=question) 