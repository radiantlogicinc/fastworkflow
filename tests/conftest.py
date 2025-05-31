"""
Pytest configuration and shared fixtures for FastWorkflow tests.
"""

import pytest
import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set up environment for tests
os.environ.setdefault("PYTEST_RUNNING", "1")


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up the test environment."""
    # Ensure we're using the local fastworkflow module
    import fastworkflow
    
    # Initialize with minimal configuration for all tests
    fastworkflow.init({})
    
    yield
    
    # Cleanup after all tests


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