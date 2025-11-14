import os
import sys
import pytest
from pathlib import Path

import fastworkflow
from fastworkflow.utils import python_utils


def test_get_module():
    """Test the get_module function for basic functionality."""
    # Get the path to this test file
    test_file_path = os.path.abspath(__file__)
    
    # Get the module using the function
    module = python_utils.get_module(test_file_path)
    
    # Verify the module was loaded
    assert module is not None
    
    # Verify it's the correct module
    assert hasattr(module, "test_get_module")


def test_get_module_with_search_root():
    """Test the get_module function with a search root."""
    # Get the path to this test file
    test_file_path = os.path.abspath(__file__)
    
    # Get the parent directory as the search root
    search_root = os.path.dirname(test_file_path)
    
    # Get the module using the function
    module = python_utils.get_module(test_file_path, search_root)
    
    # Verify the module was loaded
    assert module is not None
    
    # Verify it's the correct module
    assert hasattr(module, "test_get_module")


def test_get_module_internal_workflow():
    """Test the get_module function with an internal workflow path."""
    # Get the path to an internal workflow module
    internal_wf_path = fastworkflow.get_internal_workflow_path("command_metadata_extraction")
    module_path = os.path.join(internal_wf_path, "_commands", "ErrorCorrection", "you_misunderstood.py")
    
    # Ensure the file exists
    assert os.path.exists(module_path), f"Test file not found: {module_path}"
    
    # Get the module using the function
    module = python_utils.get_module(module_path)
    
    # Verify the module was loaded
    assert module is not None
    
    # Verify it's the correct module by checking for expected attributes
    assert hasattr(module, "Signature")
    assert hasattr(module, "ResponseGenerator")
    
    # Verify the module has the expected structure
    assert hasattr(module.Signature, "plain_utterances")
    assert hasattr(module.Signature, "generate_utterances")
    assert hasattr(module.ResponseGenerator, "__call__")


def test_get_module_internal_workflow_with_external_search_root():
    """Ensure internal workflows load even when a different search_root is provided."""
    internal_wf_path = fastworkflow.get_internal_workflow_path("command_metadata_extraction")
    module_path = os.path.join(internal_wf_path, "_commands", "wildcard.py")
    assert os.path.exists(module_path), f"Test file not found: {module_path}"

    example_workflow = os.path.join(fastworkflow.get_fastworkflow_package_path(), "examples", "hello_world")
    assert os.path.isdir(example_workflow), f"Example workflow not found: {example_workflow}"

    module = python_utils.get_module(module_path, example_workflow)

    assert module is not None
    assert hasattr(module, "Signature")
    assert hasattr(module, "ResponseGenerator")


def test_get_module_import_path():
    """Test the get_module_import_path function."""
    # Test with a simple case
    file_path = "/path/to/project/module/file.py"
    source_dir = "/path/to/project"
    expected = "module.file"
    
    # Mock os.path.abspath to return the input unchanged
    original_abspath = os.path.abspath
    os.path.abspath = lambda x: x
    
    try:
        result = python_utils.get_module_import_path(file_path, source_dir)
        assert result == expected
    finally:
        # Restore original function
        os.path.abspath = original_abspath 