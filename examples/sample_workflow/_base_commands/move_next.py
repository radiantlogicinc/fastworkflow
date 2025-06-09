from pydantic import BaseModel, ConfigDict

import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.session import Session
from fastworkflow.train.generate_synthetic import generate_diverse_utterances

from .get_status import Signature as GetStatusSignature, ResponseGenerator as GetStatusResponseGenerator


class Signature:
    class Input(BaseModel):
        skip_completed: bool = True
        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    class Output(BaseModel):
        next_workitem_found: bool
        status_of_next_workitem: str

    # Constants from plain_utterances.json
    plain_utterances = [
        "let's move to the next work item",
        "move next",
        "go to next",
        "proceed to next",
        "advance to next",
        "lets continue",
        "continue",
        "Move to the next work item, skipping completed ones.",
        "Can you move to the next work item?",
        "Skip completed work items and move to the next one.",
        "Go to the next work item, including the ones already completed.",
        "move to next",
        "move to next work item",
        "move to next task",
        "move to next step",
        "move to next stage"
    ]

    # Constants from template_utterances.json
    template_utterances = [
        "move to the next workitem {with_or_without_skipping_completed}",
        "{with_or_without_skipping_completed}, move to the next workitem"
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
            Signature.Input(skip_completed=skip)
            for skip in [True, False]
        ]

        for input in inputs:
            kwargs = {
                "with_or_without_skipping_completed": (
                    "skipping completed ones"
                    if input.skip_completed
                    else "including completed ones"
                )
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
        Move to the next work-item. Skip next work-items that are completed if the skip_completed flag is set to True.

        :param input: The input parameters for the function.
        """
        workitem = session.workflow_snapshot.active_workitem
        next_workitem = workitem.next_workitem(input.skip_completed)
        
        next_workitem_found = next_workitem is not None
        if next_workitem_found:
            session.workflow_snapshot.active_workitem = next_workitem

        active_workitem = session.workflow_snapshot.active_workitem

        get_status_tool_output = GetStatusResponseGenerator()._process_command(
            session,
            GetStatusSignature.Input(
                workitem_path=active_workitem.path, workitem_id=active_workitem.id
            ),
        )
        return Signature.Output(
            next_workitem_found=next_workitem_found,
            status_of_next_workitem=get_status_tool_output.status,
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
                        f"was the next workitem found: {output.next_workitem_found}\n"
                        f"status of the new workitem: {output.status_of_next_workitem}"
                    )
                )
            ]
        ) 