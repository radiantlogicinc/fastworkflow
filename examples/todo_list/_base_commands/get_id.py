from typing import Optional
from pydantic import BaseModel, ConfigDict

from fastworkflow import CommandOutput, CommandResponse
import fastworkflow
from fastworkflow.session import Session
from fastworkflow.train.generate_synthetic import generate_diverse_utterances


class Signature:
    class Input(BaseModel):
        for_next_workitem: bool = False
        skip_completed: bool = True
        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        workitem_path: Optional[str]
        workitem_id: Optional[str]

    # Constants from plain_utterances.json
    plain_utterances = [
        "what are we working on?",
        "where am I",
        "where are we",
        "what is our position",
        "what is our current location",
        "what is our current step",
        "what is our current stage",
        "what is our current workflow",
        "What is the current work item?",
        "What is the next work item?",
        "Can you provide the name of the next work item?",
        "get the path and id of the work item",
        "Get the location of the next work item, skipping completed ones.",
        "Get the location of the next work item, including the ones already completed.",
        "get path and id"
    ]

    # Constants from template_utterances.json
    template_utterances = [
        "get the path and id of {next_or_current} workitem {with_or_without_skipping_completed}",
        "{with_or_without_skipping_completed}, get the path and id of the {next_or_current} workitem"
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

        inputs = [
            Signature.Input(for_next_workitem=for_next, skip_completed=skip)
            for for_next in [True, False]
            for skip in [True, False]
        ]

        for input in inputs:
            kwargs = {
                "next_or_current": "next" if input.for_next_workitem else "current",
                "with_or_without_skipping_completed": (
                    "skipping completed ones"
                    if input.skip_completed
                    else "including completed ones"
                ),
            }

            for template in utterances_obj.template_utterances:
                utterance = template.format(**kwargs)
                utterance_list.append(utterance)

        return utterance_list


class ResponseGenerator:
    def _process_command(
        self, session: Session, input: Signature.Input
    ) -> Signature.Output:
        """
        Get the path and id of the current or next work-item.

        :param input: The input parameters for the function.
        """
        active_workitem = session.workflow_snapshot.active_workitem
        if not input.for_next_workitem:
            return Signature.Output(
                workitem_path=active_workitem.path, workitem_id=active_workitem.id
            )

        next_workitem = active_workitem.next_workitem()
        if input.skip_completed:
            while next_workitem and next_workitem.is_complete:
                next_workitem = next_workitem.next_workitem()

        if not next_workitem:
            return Signature.Output(workitem_path=None, workitem_id=None)

        return Signature.Output(
            workitem_path=next_workitem.path, workitem_id=next_workitem.id
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
                    response=f"path: {output.workitem_path}, id: {output.workitem_id}"
                )
            ]
        ) 