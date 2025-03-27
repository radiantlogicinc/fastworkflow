import fastworkflow
from fastworkflow.workflow import Workflow
from fastworkflow.utils.parameterize_func_decorator import parameterize

from ..parameter_extraction.signatures import CommandParameters
from fastworkflow.train.generate_synthetic import generate_diverse_utterances

def generate_command_inputs(workflow: Workflow) -> list[CommandParameters]:
    workflow_definition = fastworkflow.WorkflowRegistry.get_definition(workflow.workflow_folderpath)
    return [
        CommandParameters(workitem_path=workitem_path)
        for workitem_path in workflow_definition.paths_2_typemetadata
    ]


@parameterize(command_name=["help_about"])
def generate_utterances(session: fastworkflow.Session, command_name: str) -> list[str]:
    workflow = session.workflow_snapshot.workflow
    utterance_definition = fastworkflow.UtteranceRegistry.get_definition(workflow.workflow_folderpath)
    utterances_obj = utterance_definition.get_command_utterances(
        workflow.path, command_name
    )

    result=generate_diverse_utterances(utterances_obj.plain_utterances,command_name)
    
    utterance_list: list[str] = [command_name] + result

    # inputs: list[CommandParameters] = generate_command_inputs(workflow)
    # for input in inputs:
    #     kwargs = {}
    #     for field in input.model_fields:
    #         kwargs[field] = getattr(input, field)

    #     for template in utterances_obj.template_utterances:
    #         utterance = template.format(**kwargs)
    #         utterance_list.append(utterance)

    return utterance_list
