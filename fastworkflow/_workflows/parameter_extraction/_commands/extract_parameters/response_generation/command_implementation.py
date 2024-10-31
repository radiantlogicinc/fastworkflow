from typing import Optional
from pydantic import BaseModel
from fastworkflow.session import Session


class OutputOfProcessCommand(BaseModel):
    parameter_is_valid: bool
    cmd_parameters: Optional[BaseModel] = None
    error_msg: Optional[str] = None

def process_command(session: Session, command: str, payload: Optional[dict] = None) -> OutputOfProcessCommand:
    """Extracts the parameter value from the command."""
    input_for_param_extraction_class = payload["input_for_param_extraction_class"]
    input_for_param_extraction = input_for_param_extraction_class.create(
        session = session,
        command = command,
        payload = payload)

    command_parameters_class = payload["command_parameters_class"]
    parameter_extraction_func = payload["parameter_extraction_func"]
    _, input_obj = parameter_extraction_func(
        session,
        input_for_param_extraction,
        command_parameters_class
    )

    parameter_validation_func = payload["parameter_validation_func"]
    is_valid, error_msg = parameter_validation_func(session, input_obj)
    if not is_valid:
        return OutputOfProcessCommand(parameter_is_valid=False, error_msg=error_msg)

    return OutputOfProcessCommand(parameter_is_valid=True, cmd_parameters=input_obj)


if __name__ == "__main__":
    # create a session id
    session_id = 1234
    session = Session(session_id, "shared/tests/lighthouse/workflows/_parameter_extraction")

    tool_output = process_command(session, "extract_parameters", payload=None)
    print(tool_output)
