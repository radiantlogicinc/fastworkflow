from typing import Optional

from pydantic import BaseModel

from fastworkflow.session import Session


class OutputOfProcessCommand(BaseModel):
    parameter_is_valid: bool
    cmd_parameters: Optional[BaseModel] = None
    error_msg: Optional[str] = None


def process_command(
    caller_session: Session, command: str
) -> OutputOfProcessCommand:
    """Extracts the parameter value from the command."""
    param_extraction_info = caller_session.parameter_extraction_info

    input_for_param_extraction_class = param_extraction_info["input_for_param_extraction_class"]
    input_for_param_extraction = input_for_param_extraction_class.create(
        session=caller_session, command=command
    )

    command_parameters_class = param_extraction_info["command_parameters_class"]
    parameter_extraction_func = param_extraction_info["parameter_extraction_func"]
    _, input_obj = parameter_extraction_func(
        caller_session, input_for_param_extraction, command_parameters_class
    )

    parameter_validation_func = param_extraction_info["parameter_validation_func"]
    is_valid, error_msg = parameter_validation_func(caller_session, input_obj)
    if not is_valid:
        return OutputOfProcessCommand(parameter_is_valid=False, error_msg=error_msg)

    return OutputOfProcessCommand(parameter_is_valid=True, cmd_parameters=input_obj)


if __name__ == "__main__":
    # create a session id
    session_id = 1234
    session = Session(
        session_id, "shared/tests/lighthouse/workflows/_parameter_extraction"
    )

    tool_output = process_command(session, "extract_parameters", payload=None)
    print(tool_output)
