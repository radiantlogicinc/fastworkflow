import fastworkflow
from fastworkflow.workflow import Workflow
from fastworkflow.utils.parameterize_func_decorator import parameterize

from ..parameter_extraction.signatures import CommandParameters


@parameterize(skip_completed=[True, False])
def generate_command_inputs(
    workflow: Workflow, skip_completed: bool
) -> list[CommandParameters]:
    return [CommandParameters(skip_completed=skip_completed)]


@parameterize(command_name=["get_product_details"])
def generate_utterances(session: fastworkflow.Session, command_name: str) -> list[str]:
    workflow = session.workflow_snapshot.workflow
    utterance_definition = fastworkflow.UtteranceRegistry.get_definition(workflow.workflow_folderpath)
    utterances_obj = utterance_definition.get_command_utterances(
        workflow.path, command_name
    )

    from fastworkflow.train.generate_synthetic import generate_diverse_utterances
    utterance_list=generate_diverse_utterances(utterances_obj.plain_utterances,command_name)
    return utterance_list
