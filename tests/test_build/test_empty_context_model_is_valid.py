import os
import json
import pytest
from pathlib import Path
from fastworkflow.command_context_model import CommandContextModel

def test_empty_context_model_is_valid(tmp_path):
    """Test that an empty context model is valid."""
    # Create an empty context model file
    model_path = tmp_path / "_commands" / "context_inheritance_model.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "w") as f:
        json.dump({}, f)
    
    # Load the model
    model = CommandContextModel.load(tmp_path)
    
    # Verify it loads without errors and is an empty dict
    assert isinstance(model._command_contexts, dict)
    assert len(model._command_contexts) == 0 