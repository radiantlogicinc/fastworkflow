import fastworkflow
from fastworkflow.workflow import Workflow
from fastworkflow.utils.parameterize_func_decorator import parameterize

from ..parameter_extraction.signatures import CommandParameters


@parameterize(for_next_workitem=[True, False], skip_completed=[True, False])
def generate_command_parameters(
    workflow: Workflow, for_next_workitem: bool, skip_completed: bool
) -> list[CommandParameters]:
    return [
        CommandParameters(
            for_next_workitem=for_next_workitem, skip_completed=skip_completed
        )
    ]


@parameterize(command_name=["get_path_and_id"])
def generate_utterances(workflow: Workflow, command_name: str) -> list[str]:
    utterance_definition = fastworkflow.UtteranceRegistry.get_definition(workflow.workflow_folderpath)
    utterances_obj = utterance_definition.get_command_utterances(
        workflow.type, command_name
    )

    utterance_list: list[str] = utterances_obj.plain_utterances.copy()

    inputs: list[CommandParameters] = generate_command_parameters(workflow)
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
