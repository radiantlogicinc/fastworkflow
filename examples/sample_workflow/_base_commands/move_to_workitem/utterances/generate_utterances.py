import fastworkflow
from fastworkflow.workflow import Workflow
from fastworkflow.utils.parameterize_func_decorator import parameterize
from ..parameter_extraction.signatures import CommandParameters


def generate_command_inputs(workflow: Workflow) -> list[CommandParameters]:
    workflow_definition = fastworkflow.WorkflowRegistry.get_definition(workflow.workflow_folderpath)
    workitem_paths = []
    for workitem_type in workflow_definition.types:
        workitem_paths.append(f"{workitem_type}")

    # add full paths
    workitem = workflow.next_workitem(skip_completed=False)
    while workitem is not None:
        workitem_paths.append(workitem.path)
        workitem = workitem.next_workitem(skip_completed=False)

    return [
        CommandParameters(workitem_path=workitem_path, workitem_id=None)
        for workitem_path in workitem_paths
    ]


@parameterize(command_name=["move_to_workitem"])
def generate_utterances(workflow: Workflow, command_name: str) -> list[str]:
    utterance_definition = fastworkflow.UtteranceRegistry.get_definition(workflow.workflow_folderpath)
    utterances_obj = utterance_definition.get_command_utterances(
        workflow.type, command_name
    )

    utterance_list: list[str] = [command_name] + utterances_obj.plain_utterances

    inputs: list[CommandParameters] = generate_command_inputs(workflow)
    for input in inputs:
        kwargs = {}
        for field in input.model_fields:
            kwargs[field] = getattr(input, field)

        for template in utterances_obj.template_utterances:
            utterance = template.format(**kwargs)
            utterance_list.append(utterance)

    return utterance_list
