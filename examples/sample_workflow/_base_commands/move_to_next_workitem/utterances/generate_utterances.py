import fastworkflow
from fastworkflow.workflow import Workflow
from fastworkflow.utils.parameterize_func_decorator import parameterize

from ..parameter_extraction.signatures import CommandParameters


@parameterize(skip_completed=[True, False])
def generate_command_inputs(
    workflow: Workflow, skip_completed: bool
) -> list[CommandParameters]:
    return [CommandParameters(skip_completed=skip_completed)]


@parameterize(command_name=["move_to_next_workitem"])
def generate_utterances(workflow: Workflow, command_name: str) -> list[str]:
    utterance_definition = fastworkflow.UtteranceRegistry.get_definition(workflow.workflow_folderpath)
    utterances_obj = utterance_definition.get_command_utterances(
        workflow.type, command_name
    )

    utterance_list: list[str] = utterances_obj.plain_utterances.copy()

    inputs: list[CommandParameters] = generate_command_inputs(workflow)
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
