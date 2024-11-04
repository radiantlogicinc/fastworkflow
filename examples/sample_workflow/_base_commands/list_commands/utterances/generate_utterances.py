from fastworkflow.session import Session
from fastworkflow.utils.parameterize_func_decorator import parameterize


@parameterize(command_name=["list_commands"])
def generate_utterances(session: Session, command_name: str) -> list[str]:
    utterances_obj = session.utterance_definition.get_command_utterances(
        session.root_workitem_type, command_name
    )

    return utterances_obj.plain_utterances.copy()


if __name__ == "__main__":
    import os

    # create a session
    session_id = 1234

    workflow_folderpath = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../")
    )
    if not os.path.isdir(workflow_folderpath):
        raise ValueError(
            f"The provided folderpath '{workflow_folderpath}' is not valid. Please provide a valid directory."
        )

    session = Session(session_id, workflow_folderpath)

    generated_utterances = generate_utterances(session)

    for utterance in generated_utterances:
        print(utterance)
