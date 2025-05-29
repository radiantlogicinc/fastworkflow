import fastworkflow
from fastworkflow.workflow import Workflow
from fastworkflow.utils.parameterize_func_decorator import parameterize


@parameterize(command_name=["find_user_id_by_name_zip"])
def generate_utterances(session: fastworkflow.Session, command_name: str) -> list[str]:
    workflow = session.workflow_snapshot.workflow
    utterance_definition = fastworkflow.UtteranceRegistry.get_definition(workflow.workflow_folderpath)
    utterances_obj = utterance_definition.get_command_utterances(
        workflow.path, command_name
    )
    from fastworkflow.train.generate_synthetic import generate_diverse_utterances
    utterance_list=generate_diverse_utterances(utterances_obj.plain_utterances,command_name)

    return utterance_list
