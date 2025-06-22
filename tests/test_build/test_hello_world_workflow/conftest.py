import os
import pytest
from pathlib import Path

@pytest.fixture
def hello_world_root_dir():
    """Path to the hello_world_workflow directory."""
    return Path("./tests/hello_world_workflow").resolve()

@pytest.fixture
def hello_world_app_dir(hello_world_root_dir):
    """Return the path to the hello_world_workflow application directory."""
    return str(hello_world_root_dir / "application")

@pytest.fixture
def hello_world_build_dir(tmp_path):
    """Return a temporary build directory inside hello_world_workflow for isolation."""
    build_dir = tmp_path / "hello_world_build"
    build_dir.mkdir()
    return str(build_dir) 