"""Tests for workflow inheritance functionality."""

import pytest
import tempfile
import os
import json
from pathlib import Path
from fastworkflow.command_directory import CommandDirectory
from fastworkflow.workflow_inheritance_model import WorkflowInheritanceModel, WorkflowInheritanceModelValidationError


class TestWorkflowInheritanceModel:
    """Test the WorkflowInheritanceModel class."""
    
    def test_empty_model(self):
        """Test creating an empty inheritance model."""
        model = WorkflowInheritanceModel()
        assert model.base == []
    
    def test_load_nonexistent_file(self):
        """Test loading from a directory without inheritance file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = WorkflowInheritanceModel.load(tmpdir)
            assert model.base == []
    
    def test_load_valid_file(self):
        """Test loading a valid inheritance model file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            inheritance_file = Path(tmpdir) / "workflow_inheritance_model.json"
            data = {
                "base": [
                    "fastworkflow.examples.simple_workflow_template",
                    "./local_template"
                ]
            }
            with open(inheritance_file, 'w') as f:
                json.dump(data, f)
            
            model = WorkflowInheritanceModel.load(tmpdir)
            assert len(model.base) == 2
            assert model.base[0] == "fastworkflow.examples.simple_workflow_template"
            assert model.base[1] == "./local_template"
    
    def test_load_invalid_json(self):
        """Test loading an invalid JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            inheritance_file = Path(tmpdir) / "workflow_inheritance_model.json"
            with open(inheritance_file, 'w') as f:
                f.write("{ invalid json }")
            
            with pytest.raises(WorkflowInheritanceModelValidationError, match="Invalid JSON"):
                WorkflowInheritanceModel.load(tmpdir)
    
    def test_validation_empty_base_entry(self):
        """Test validation fails for empty base entries."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            WorkflowInheritanceModel(base=["valid", "", "another_valid"])
    
    def test_resolve_package_path(self):
        """Test resolving a valid package path."""
        model = WorkflowInheritanceModel(base=["fastworkflow.examples.simple_workflow_template"])
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved_paths = model.resolve_base_paths(tmpdir)
            assert len(resolved_paths) == 1
            # Should resolve to the actual package directory
            assert os.path.isdir(resolved_paths[0])
            assert os.path.isdir(os.path.join(resolved_paths[0], "_commands"))
    
    def test_resolve_filesystem_path(self):
        """Test resolving a filesystem path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock base workflow directory
            base_dir = Path(tmpdir) / "base_workflow"
            commands_dir = base_dir / "_commands"
            commands_dir.mkdir(parents=True)
            (commands_dir / "test_command.py").touch()
            
            model = WorkflowInheritanceModel(base=[str(base_dir)])
            resolved_paths = model.resolve_base_paths(tmpdir)
            assert len(resolved_paths) == 1
            assert resolved_paths[0] == str(base_dir)
    
    def test_resolve_relative_filesystem_path(self):
        """Test resolving a relative filesystem path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock base workflow directory
            base_dir = Path(tmpdir) / "base_workflow"
            commands_dir = base_dir / "_commands"
            commands_dir.mkdir(parents=True)
            (commands_dir / "test_command.py").touch()
            
            # Create a current workflow directory
            current_dir = Path(tmpdir) / "current_workflow"
            current_dir.mkdir()
            
            model = WorkflowInheritanceModel(base=["../base_workflow"])
            resolved_paths = model.resolve_base_paths(str(current_dir))
            assert len(resolved_paths) == 1
            assert resolved_paths[0] == str(base_dir)
    
    def test_resolve_invalid_package(self):
        """Test that resolving an invalid package raises an error."""
        model = WorkflowInheritanceModel(base=["nonexistent.invalid.package"])
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(WorkflowInheritanceModelValidationError, match="Cannot import package"):
                model.resolve_base_paths(tmpdir)
    
    def test_resolve_invalid_filesystem_path(self):
        """Test that resolving an invalid filesystem path raises an error."""
        model = WorkflowInheritanceModel(base=["/nonexistent/path"])
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(WorkflowInheritanceModelValidationError, match="does not exist"):
                model.resolve_base_paths(tmpdir)


class TestCommandDirectoryInheritance:
    """Test CommandDirectory with inheritance functionality."""
    
    def test_load_with_no_inheritance(self):
        """Test loading a workflow with no inheritance model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple workflow with one command
            commands_dir = Path(tmpdir) / "_commands"
            commands_dir.mkdir()
            
            command_file = commands_dir / "test_command.py"
            command_file.write_text("""
class ResponseGenerator:
    pass
""")
            
            directory = CommandDirectory.load(tmpdir)
            assert "test_command" in directory.get_commands()
    
    def test_load_with_inheritance_precedence(self):
        """Test that inheritance follows correct precedence rules."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create base workflow
            base_dir = Path(tmpdir) / "base_workflow"
            base_commands_dir = base_dir / "_commands"
            base_commands_dir.mkdir(parents=True)
            
            base_command = base_commands_dir / "shared_command.py"
            base_command.write_text("""
class ResponseGenerator:
    def __call__(self, workflow, command):
        return "base_version"
""")
            
            base_only_command = base_commands_dir / "base_only_command.py"
            base_only_command.write_text("""
class ResponseGenerator:
    pass
""")
            
            # Create extending workflow
            extend_dir = Path(tmpdir) / "extending_workflow"
            extend_commands_dir = extend_dir / "_commands"
            extend_commands_dir.mkdir(parents=True)
            
            # Override the shared command
            extend_command = extend_commands_dir / "shared_command.py"
            extend_command.write_text("""
class ResponseGenerator:
    def __call__(self, workflow, command):
        return "extended_version"
""")
            
            # Add a new command
            new_command = extend_commands_dir / "new_command.py"
            new_command.write_text("""
class ResponseGenerator:
    pass
""")
            
            # Create inheritance model
            inheritance_file = extend_dir / "workflow_inheritance_model.json"
            inheritance_data = {"base": [str(base_dir)]}
            with open(inheritance_file, 'w') as f:
                json.dump(inheritance_data, f)
            
            # Load the extending workflow
            directory = CommandDirectory.load(str(extend_dir))
            commands = directory.get_commands()
            
            # Should have all commands: base_only, shared (overridden), and new
            assert "base_only_command" in commands
            assert "shared_command" in commands
            assert "new_command" in commands
            
            # The shared command should point to the extended version's file
            shared_metadata = directory.get_command_metadata("shared_command")
            assert "extending_workflow" in shared_metadata.response_generation_module_path
    
    def test_load_commands_from_path_skips_missing_directories(self):
        """Test that _load_commands_from_path gracefully handles missing _commands directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = CommandDirectory(workflow_folderpath=tmpdir)
            
            # Should not raise an error even if _commands doesn't exist
            CommandDirectory._load_commands_from_path(directory, tmpdir)
            assert len(directory.get_commands()) == 0


class TestIntegrationWithExistingExample:
    """Test integration with the actual extended_workflow_example."""
    
    def test_extended_workflow_example_loads_correctly(self):
        """Test that the extended_workflow_example loads and has expected commands."""
        example_path = "fastworkflow/examples/extended_workflow_example"
        
        # Skip if the example doesn't exist (in case it's not created yet)
        if not os.path.exists(example_path):
            pytest.skip("extended_workflow_example not found")
        
        directory = CommandDirectory.load(example_path)
        commands = directory.get_commands()
        
        # Should have commands from both base template and extended workflow
        expected_base_commands = ["startup"]  # At minimum, should have startup
        expected_extended_commands = ["generate_report"]  # New command in extended workflow
        
        for cmd in expected_base_commands + expected_extended_commands:
            assert cmd in commands, f"Command '{cmd}' not found in {commands}"
        
        # Should have WorkItem commands from base template
        workitem_commands = [cmd for cmd in commands if cmd.startswith("WorkItem/")]
        assert len(workitem_commands) > 0, "Should inherit WorkItem commands from base template"
        
        # The startup command should be overridden by the extended version
        startup_metadata = directory.get_command_metadata("startup")
        assert "extended_workflow_example" in startup_metadata.response_generation_module_path
