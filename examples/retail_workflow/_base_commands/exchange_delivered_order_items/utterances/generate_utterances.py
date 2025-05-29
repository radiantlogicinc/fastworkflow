import fastworkflow
from fastworkflow.workflow import Workflow
from fastworkflow.utils.parameterize_func_decorator import parameterize

from ..parameter_extraction.signatures import CommandParameters


def generate_command_inputs(workflow: Workflow) -> list[CommandParameters]:
    workflow_definition = fastworkflow.WorkflowRegistry.get_definition(workflow.workflow_folderpath)
    return [
        CommandParameters(workitem_path=workitem_path)
        for workitem_path in workflow_definition.paths_2_typemetadata
    ]


@parameterize(command_name=["exchange_delivered_order_items"])
def generate_utterances(session: fastworkflow.Session, command_name: str) -> list[str]:
    workflow = session.workflow_snapshot.workflow
    utterance_definition = fastworkflow.UtteranceRegistry.get_definition(workflow.workflow_folderpath)
    utterances_obj = utterance_definition.get_command_utterances(
        workflow.path, command_name
    )

    from fastworkflow.train.generate_synthetic import generate_diverse_utterances
    utterance_list=generate_diverse_utterances(utterances_obj.plain_utterances,command_name)
    return utterance_list
