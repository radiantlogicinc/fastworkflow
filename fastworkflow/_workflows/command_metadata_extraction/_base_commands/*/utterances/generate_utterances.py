import fastworkflow
from fastworkflow.workflow import Workflow
from fastworkflow.utils.parameterize_func_decorator import parameterize
from fastworkflow.train.generate_synthetic import generate_diverse_utterances

@parameterize(command_name=["*"])
def generate_utterances(session: fastworkflow.Session, command_name: str) -> list[str]:
    workflow = session.workflow_snapshot.workflow
    utterance_definition = fastworkflow.UtteranceRegistry.get_definition(workflow.workflow_folderpath)
    utterances_obj = utterance_definition.get_command_utterances(
        workflow.path, command_name
    )

    utterance_list: list[str] = [command_name] + utterances_obj.plain_utterances
    return generate_diverse_utterances(utterances_obj.plain_utterances, command_name)
    
