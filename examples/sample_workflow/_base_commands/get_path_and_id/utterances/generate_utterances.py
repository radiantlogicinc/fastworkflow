from fastworkflow.session import Session
from fastworkflow.utils.parameterize_func_decorator import parameterize
from ..parameter_extraction.signatures import CommandParameters


@parameterize(for_next_workitem=[True, False], skip_completed=[True, False])
def generate_command_parameters(session: Session, for_next_workitem: bool, skip_completed: bool) -> list[CommandParameters]:
    return [
        CommandParameters(for_next_workitem=for_next_workitem, skip_completed=skip_completed)
    ]

@parameterize(command_name=["get_path_and_id"])
def generate_utterances(session: Session, command_name: str) -> list[str]:
    utterances_obj = session.utterance_definition.get_command_utterances(
                    session.root_workitem_type, command_name)    
    
    utterance_list: list[str] = utterances_obj.plain_utterances.copy()

    inputs: list[CommandParameters] = generate_command_parameters(session)
    for input in inputs:
        kwargs = {
            "next_or_current": "next" if input.for_next_workitem else "current",
            "with_or_without_skipping_completed": "skipping completed ones" 
            if input.skip_completed else "including completed ones"
        }   

        for template in utterances_obj.template_utterances:
            utterance = template.format(**kwargs)
            utterance_list.append(utterance)

    return utterance_list

if __name__ == "__main__":
    import os

    # create a session
    session_id = 1234

    workflow_folderpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
    if not os.path.isdir(workflow_folderpath):
        raise ValueError(f"The provided folderpath '{workflow_folderpath}' is not valid. Please provide a valid directory.")

    session = Session(session_id, workflow_folderpath)

    generated_utterances = generate_utterances(session)
    
    for utterance in generated_utterances:
        print(utterance)

