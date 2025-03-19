import fastworkflow
from fastworkflow.workflow import Workflow
from fastworkflow.utils.parameterize_func_decorator import parameterize

from ..parameter_extraction.signatures import CommandParameters
from fastworkflow.train.generate_synthetic import generate_diverse_utterances

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


@parameterize(command_name=["get_status"])
def generate_utterances(session: fastworkflow.Session, command_name: str) -> list[str]:
    workflow = session.workflow_snapshot.workflow
    utterance_definition = fastworkflow.UtteranceRegistry.get_definition(workflow.workflow_folderpath)
    utterances_obj = utterance_definition.get_command_utterances(
        workflow.path, command_name
    )

    result=generate_diverse_utterances(utterances_obj.plain_utterances,command_name,10,10,5)
    all_utterances = [utt["utterance"] for utt in result["generated_utterances"]]
    utterance_list: list[str] = [command_name] + utterances_obj.plain_utterances+all_utterances

    inputs: list[CommandParameters] = generate_command_inputs(workflow)
    for input in inputs:
        kwargs = {field: getattr(input, field) for field in input.model_fields}
        for template in utterances_obj.template_utterances:
            utterance = template.format(**kwargs)
            utterance_list.append(utterance)

    return utterance_list
