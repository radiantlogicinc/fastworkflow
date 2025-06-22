import os
import json
import pytest
from pathlib import Path
from fastworkflow.context_model_loader import ContextModelLoader, ContextModelLoaderError

def test_empty_context_model_is_valid(tmp_path):
    """Test that an empty context model is valid."""
    # Create an empty context model file
    model_path = tmp_path / "context_inheritance_model.json"
    with open(model_path, "w") as f:
        json.dump({}, f)
    
    # Load the model
    loader = ContextModelLoader(model_path)
    model = loader.load()
    
    # Verify it loads without errors and is an empty dict
    assert isinstance(model, dict)
    assert len(model) == 0
    
    # Check that contexts property returns empty dict
    assert loader.contexts == {}
    
    # Check that bases returns empty list for any context
    assert loader.bases("NonExistentContext") == [] 