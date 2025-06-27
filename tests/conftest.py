"""
Pytest configuration and shared fixtures for FastWorkflow tests.
"""

import pytest
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set up environment for tests
os.environ.setdefault("PYTEST_RUNNING", "1")


@pytest.fixture(scope="session")
def setup_test_environment():
    """Set up the test environment."""
    # Ensure we're using the local fastworkflow module
    import fastworkflow
    
    # Initialize with minimal configuration for all tests
    fastworkflow.init({})
    
    yield
    
    # Cleanup after all tests


@pytest.fixture(scope="session", autouse=True)
def add_workflow_paths_to_syspath():
    """
    Add workflow paths to sys.path to enable relative imports in workflow modules.
    This simulates what Workflow class does in production code.
    """
    # Store original sys.path to restore later
    original_sys_path = list(sys.path)
    
    # Add common workflow paths used in tests
    workflow_paths = [
        Path(project_root) / "examples" / "retail_workflow",
        Path(project_root) / "examples" / "todo_list",
        Path(project_root) / "examples" / "hello_world",
        # Add any other workflow paths used in tests
    ]
    
    for path in workflow_paths:
        path_str = str(path.resolve())
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    
    yield
    
    # Restore original sys.path after tests complete
    sys.path = original_sys_path


@pytest.fixture
def add_temp_workflow_path():
    """
    Fixture for tests that create temporary workflow paths.
    Usage: Call this fixture with the temporary path to add it to sys.path.
    
    Example:
        def test_something(tmp_path, add_temp_workflow_path):
            add_temp_workflow_path(tmp_path)
            # Now tmp_path is in sys.path for this test
    """
    original_sys_path = list(sys.path)
    added_paths = []
    
    def _add_path(path):
        path_str = str(Path(path).resolve())
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
            added_paths.append(path_str)
    
    yield _add_path
    
    # Restore original sys.path after test completes
    for path in added_paths:
        if path in sys.path:
            sys.path.remove(path)


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add integration marker to integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Add slow marker to tests that might take longer
        if "mcp_server" in item.nodeid:
            item.add_marker(pytest.mark.slow)


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    ) 