from pydantic import BaseModel

import fastworkflow
from fastworkflow.session import Session


class CommandProcessorOutput(BaseModel):
    workitem_path: str
    utterances: list[str]


def process_command(
    session: Session
) -> CommandProcessorOutput:
    """
    Provides helpful information about this type of work-item.
    If the workitem_path is not provided, it provides information about the current work-item.

    :param input: The input parameters for the function.
    """
    # Get the current workitem type
    current_workitem = session.workflow_snapshot.active_workitem

    workflow_folderpath = session.workflow_snapshot.workflow.workflow_folderpath
    utterance_definition = fastworkflow.UtteranceRegistry.get_definition(workflow_folderpath)
    utterances = utterance_definition.get_sample_utterances(current_workitem.path)

    return CommandProcessorOutput(
        workitem_path=current_workitem.path, utterances=utterances
    )


if __name__ == "__main__":
    # create a session id
    session_id = 1234
    session = Session(session_id, "shared/tests/lighthouse/workflows/accessreview")

    tool_output = process_command(session)
    print(tool_output)
