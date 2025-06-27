import os
import tempfile
import shutil
from pathlib import Path
import json

import pytest

from fastworkflow.build.__main__ import validate_directories, generate_startup_command
from fastworkflow.build.context_folder_generator import ContextFolderGenerator

class TestDirectoryLayout:
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary source and output directories for testing."""
        source_dir = tempfile.mkdtemp()
        output_dir = tempfile.mkdtemp()
        
        # Create a simple application structure in source_dir
        app_dir = os.path.join(source_dir, "application")
        os.makedirs(app_dir)
        
        # Create a simple manager class
        with open(os.path.join(app_dir, "manager.py"), "w") as f:
            f.write("""
class TestManager:
    def __init__(self, filepath):
        self.filepath = filepath
""")
        
        # Create __init__.py
        with open(os.path.join(app_dir, "__init__.py"), "w") as f:
            f.write("")
        
        yield {"source": source_dir, "output": output_dir}
        
        # Clean up
        shutil.rmtree(source_dir)
        shutil.rmtree(output_dir)
    
    def test_validate_directories_creates_commands_dir(self, temp_dirs):
        """Test that validate_directories creates _commands directory and __init__.py."""
        class Args:
            def __init__(self, source_dir, output_dir):
                self.source_dir = source_dir
                self.output_dir = output_dir
        
        args = Args(temp_dirs["source"], temp_dirs["output"])
        validate_directories(args)
        
        # Check that _commands directory was created
        commands_dir = os.path.join(temp_dirs["output"], "_commands")
        assert os.path.isdir(commands_dir)
        
        # Check that __init__.py was created
        init_py = os.path.join(commands_dir, "__init__.py")
        assert os.path.isfile(init_py)
    
    def test_generate_startup_command(self, temp_dirs):
        # sourcery skip: class-extract-method, extract-duplicate-method
        """Test that generate_startup_command creates a startup.py file."""
        # Create _commands directory
        commands_dir = os.path.join(temp_dirs["output"], "_commands")
        os.makedirs(commands_dir, exist_ok=True)
        
        # Generate startup command
        result = generate_startup_command(temp_dirs["output"], temp_dirs["source"])
        
        # Check that startup.py was created
        startup_py = os.path.join(commands_dir, "startup.py")
        assert os.path.isfile(startup_py)
        assert result == True
        
        # Check content
        with open(startup_py, "r") as f:
            content = f.read()
        
        # Verify imports
        assert "import fastworkflow" in content
        assert "from fastworkflow import CommandOutput, CommandResponse" in content
        
        # Verify the ResponseGenerator class exists
        assert "class ResponseGenerator:" in content
        assert "__call__" in content
        assert "workflow.root_command_context =" in content
        
        # Verify it doesn't overwrite existing file
        with open(startup_py, "w") as f:
            f.write("# Custom content")
        
        # Try to generate again without overwrite
        result = generate_startup_command(temp_dirs["output"], temp_dirs["source"])
        assert result == True
        
        # Content should not have changed
        with open(startup_py, "r") as f:
            content = f.read()
        assert content == "# Custom content"
        
        # Try to generate again with overwrite
        result = generate_startup_command(temp_dirs["output"], temp_dirs["source"], overwrite=True)
        assert result == True
        
        # Content should have changed
        with open(startup_py, "r") as f:
            content = f.read()
        assert "import fastworkflow" in content
    
    def test_context_folder_generator_creates_handler_files(self, temp_dirs):
        # sourcery skip: extract-duplicate-method
        """Test that ContextFolderGenerator creates _<ContextName>.py files."""
        # Create _commands directory
        commands_dir = os.path.join(temp_dirs["output"], "_commands")
        os.makedirs(commands_dir, exist_ok=True)
        
        # Create a simple context model
        model_path = os.path.join(commands_dir, "context_inheritance_model.json")
        model_data = {
            "TestContext": {"base": ["BaseContext"]},
            "BaseContext": {"base": []},
            "*": {"base": []}
        }
        
        with open(model_path, "w") as f:
            json.dump(model_data, f)
        
        # Create application files to match context model
        app_dir = os.path.join(temp_dirs["source"], "application")
        with open(os.path.join(app_dir, "testcontext.py"), "w") as f:
            f.write("class TestContext: pass")
        
        with open(os.path.join(app_dir, "basecontext.py"), "w") as f:
            f.write("class BaseContext: pass")
        
        # Generate context folders
        generator = ContextFolderGenerator(
            commands_root=commands_dir,
            model_path=model_path
        )
        created_folders = generator.generate_folders()
        
        # Check that folders were created
        assert "TestContext" in created_folders
        assert "BaseContext" in created_folders
        
        # Check that handler files were created
        test_handler = os.path.join(commands_dir, "TestContext", "_TestContext.py")
        base_handler = os.path.join(commands_dir, "BaseContext", "_BaseContext.py")
        
        assert os.path.isfile(test_handler)
        assert os.path.isfile(base_handler)
        
        # Check content of TestContext handler
        with open(test_handler, "r") as f:
            content = f.read()
        
        # Should import from application and have parent reference
        assert "from ...application.testcontext import TestContext" in content
        assert "from ...application.basecontext import BaseContext" in content
        assert "def get_parent(cls, command_context_object: TestContext) -> Optional[BaseContext]:" in content
        
        # Check content of BaseContext handler
        with open(base_handler, "r") as f:
            content = f.read()
        
        # Should import from application but have no parent
        assert "from ...application.basecontext import BaseContext" in content
        assert "def get_parent(cls, command_context_object: BaseContext) -> Optional[None]:" in content 