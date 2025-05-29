from pydantic import BaseModel

from fastworkflow.session import Session
from fastworkflow.workflow_definition import NodeType

from ..parameter_extraction.signatures import CommandParameters
from ....retail_data import load_data 
from ....tools.cancel_pending_order import CancelPendingOrder

class CommandProcessorOutput(BaseModel):
    status: str


def process_command(
    session: Session, input: CommandParameters
) -> CommandProcessorOutput:
    """
    get the review status of the entitlements in this workitem.

    :param input: The input parameters for the function.

    return the review status of the entitlements for the current workitem if workitem_path and workitem_id are not provided.
        if workitem_id is specified, the workitem_path must be specified.
    """
    data=load_data()

    # Call CancelPendingOrder's invoke method
    result = CancelPendingOrder.invoke(
        data=data,
        order_id=input.order_id,  # Assuming CommandParameters has order_id
        reason=input.reason      # Assuming CommandParameters has reason
    )
    
    return CommandProcessorOutput(status=result)




if __name__ == "__main__":
    # create a session id
    session_id = 1234
    session = Session(session_id, "shared/tests/lighthouse/workflows/accessreview")

    tool_input = CommandParameters(workitem_path=None, workitem_id=None)
    tool_output = process_command(session, tool_input)
    print(tool_output)
