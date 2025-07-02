import os
import sys
import subprocess
import pytest
from pathlib import Path

def test_build_generates_global_function_command(hello_world_app_dir, hello_world_build_dir):
    """Test that the build tool generates a global function command."""
    # Run the build command
    cmd = [
        sys.executable,
        "-m",
        "fastworkflow.build",
        "--app-dir",
        hello_world_app_dir,
        "--workflow-folderpath",
        str(hello_world_build_dir),
        "--overwrite"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Check that the command succeeded
    assert result.returncode == 0, f"Build failed with output: {result.stderr}"
    
    # Check that the add_two_numbers.py file was generated
    add_two_numbers_path = os.path.join(hello_world_build_dir, '_commands', 'add_two_numbers.py')   
    # Check the content of the file
    with open(add_two_numbers_path, "r") as f:
        content = f.read()
    
    # Verify it has the required components
    assert "class Signature:" in content
    assert "class Input(BaseModel):" in content
    assert "class Output(BaseModel):" in content
    assert "class ResponseGenerator:" in content
    assert "def __call__(self, workflow: Workflow, command: str" in content
    
    # Check for function parameters
    assert "a: float" in content
    assert "b: float" in content
    
    # Check for import of the function
    assert "from ..application.add_two_numbers import add_two_numbers" in content
    
    # Check for function call
    assert "result_val = add_two_numbers(" in content 