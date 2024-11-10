from typing import Optional

from pydantic import BaseModel

from fastworkflow.session import Session


class CommandProcessorOutput(BaseModel):
    workitem_type: str
    utterances: list[str]


def process_command(
    session: Session
) -> CommandProcessorOutput:
    """
    Provides helpful information about this type of work-item.
    If the workitem_type is not provided, it provides information about the current work-item.

    :param input: The input parameters for the function.
    """
    # Get the current workitem type
    current_workitem = session.get_active_workitem()

    utterances = session.utterance_definition.get_sample_utterances(current_workitem.type)

    return CommandProcessorOutput(
        workitem_type=current_workitem.type, utterances=utterances
    )


if __name__ == "__main__":
    # create a session id
    session_id = 1234
    session = Session(session_id, "shared/tests/lighthouse/workflows/accessreview")

    tool_output = process_command(session)
    print(tool_output)
