import fastworkflow
from fastworkflow.workflow import Workflow
from fastworkflow.utils.parameterize_func_decorator import parameterize


@parameterize(command_name=["list_utterances"])
def generate_utterances(workflow: Workflow, command_name: str) -> list[str]:
    utterance_definition = fastworkflow.UtteranceRegistry.get_definition(workflow.workflow_folderpath)
    utterances_obj = utterance_definition.get_command_utterances(
        workflow.type, command_name
    )

    return utterances_obj.plain_utterances
