import fastworkflow
from fastworkflow.workflow import Workflow
from fastworkflow.utils.parameterize_func_decorator import parameterize
from fastworkflow.train.generate_synthetic import generate_diverse_utterances

@parameterize(command_name=["None_of_these"])
def generate_utterances(session: fastworkflow.Session, command_name: str) -> list[str]:
    workflow = session.workflow_snapshot.workflow
    utterance_definition = fastworkflow.UtteranceRegistry.get_definition(workflow.workflow_folderpath)
    utterances_obj = utterance_definition.get_command_utterances(
        workflow.path, command_name
    )
    #result=generate_diverse_utterances(utterances_obj.plain_utterances,command_name,5,5)
    result=generate_diverse_utterances(utterances_obj.plain_utterances,command_name,10,10,5)
    all_utterances = [utt["utterance"] for utt in result["generated_utterances"]]
    utterance_list: list[str] = [command_name] + utterances_obj.plain_utterances+all_utterances

    return utterance_list
