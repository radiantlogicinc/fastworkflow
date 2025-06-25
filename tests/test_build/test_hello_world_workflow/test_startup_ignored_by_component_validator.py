import os
import pytest
from pathlib import Path
from fastworkflow.build.command_file_generator import validate_command_file_components_in_dir

def test_startup_ignored_by_component_validator(tmp_path):
    # sourcery skip: extract-duplicate-method
    """Test that startup.py is ignored by the component validator."""
    # Create a directory structure
    commands_dir = tmp_path / "_commands"
    commands_dir.mkdir()
    
    # Create a minimal startup.py without Signature class
    startup_path = commands_dir / "startup.py"
    startup_content = """
import fastworkflow
from fastworkflow import CommandOutput, CommandResponse

class ResponseGenerator:
    def __call__(self, session: fastworkflow.Session, command: str) -> CommandOutput:
        return CommandOutput(
            session_id=session.id,
            command_responses=[
                CommandResponse(response="Startup executed")
            ]
        )
"""
    with open(startup_path, "w") as f:
        f.write(startup_content)
    
    # Create a valid command file
    valid_command_path = commands_dir / "valid_command.py"
    valid_command_content = """
import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.session import Session
from typing import Any, Dict
from pydantic import BaseModel, Field

class Signature:
    class Input(BaseModel):
        pass
    
    class Output(BaseModel):
        result: str = Field(description="Result of the command")
    
    plain_utterances = [
        "valid command"
    ]
    
    template_utterances = []
    
    @staticmethod
    def generate_utterances(session: Session, command_name: str) -> list[str]:
        return ["valid command"]
    
    def process_extracted_parameters(self, session: fastworkflow.Session, command: str, cmd_parameters: None) -> None:
        pass

class ResponseGenerator:
    def _process_command(self, session: Session) -> Signature.Output:
        return Signature.Output(result="Valid command executed")
    
    def __call__(self, session: Session, command: str) -> CommandOutput:
        output = self._process_command(session)
        return CommandOutput(
            session_id=session.id,
            command_responses=[
                CommandResponse(response=output.model_dump_json())
            ]
        )
"""
    with open(valid_command_path, "w") as f:
        f.write(valid_command_content)
    
    # Run validation
    errors = validate_command_file_components_in_dir(str(commands_dir))
    
    # Check that there are no errors
    assert len(errors) == 0, f"Unexpected errors: {errors}" 