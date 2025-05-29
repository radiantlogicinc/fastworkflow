import fastworkflow
from fastworkflow.workflow import Workflow
from fastworkflow.utils.parameterize_func_decorator import parameterize
from ..parameter_extraction.signatures import CommandParameters


def generate_command_inputs(workflow: Workflow) -> list[CommandParameters]:
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

    return [
        CommandParameters(workitem_path=workitem_path, workitem_id=None)
        for workitem_path in workitem_paths
    ]


@parameterize(command_name=["modify_pending_order_payment"])
def generate_utterances(session: fastworkflow.Session, command_name: str) -> list[str]:
    workflow = session.workflow_snapshot.workflow
    utterance_definition = fastworkflow.UtteranceRegistry.get_definition(workflow.workflow_folderpath)
    utterances_obj = utterance_definition.get_command_utterances(
        workflow.path, command_name
    )

    from fastworkflow.train.generate_synthetic import generate_diverse_utterances
    utterance_list=generate_diverse_utterances(utterances_obj.plain_utterances,command_name)
    return utterance_list
