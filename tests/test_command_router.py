from pathlib import Path
import os
import pprint
import uuid
import pytest
import shutil
import tempfile

import fastworkflow
from fastworkflow.command_routing import RoutingDefinition
from fastworkflow.command_directory import CommandDirectory


# Helper to create a temporary _commands directory *inside* project root so that
# python_utils.get_module does not reject module import paths during hydration.
def _create_commands(root_dir: Path):
    """Create a sample _commands tree:

    _commands/
        a.py
        ctx1/
            x.py
            y.py
        ctx2/
            _ctx2.py
    """
    root = root_dir / "_commands"
    root.mkdir()

    # global command
    (root / "a.py").write_text("# dummy\nclass ResponseGenerator:\n    pass")
    # ignored file
    (root / "ignore.txt").write_text("txt")

    # ctx1 with two commands
    ctx1 = root / "ctx1"
    ctx1.mkdir()
    (ctx1 / "x.py").write_text("# x\nclass ResponseGenerator:\n    pass")
    (ctx1 / "y.py").write_text("# y\nclass ResponseGenerator:\n    pass")

    # ctx2 with context implementation
    ctx2 = root / "ctx2"
    ctx2.mkdir()
    (ctx2 / "_ctx2.py").write_text("# ctx2 impl\nclass Context:\n    pass")
    (ctx2 / "dummy.py").write_text("# dummy\nclass ResponseGenerator:\n    pass")

    return root


def test_scan_and_lookup(add_temp_workflow_path, tmp_path):
    # sourcery skip: extract-method
    # Build workspace for this test under the repository root so that
    # fastworkflow.utils.python_utils.get_module allows dynamic imports.
    project_root = Path(__file__).resolve().parents[1]
    test_workspace = project_root / "__tmp_router_tests" / f"ws_{uuid.uuid4().hex}"
    test_workspace.mkdir(parents=True, exist_ok=True)

    _create_commands(test_workspace)
    # Add the temporary path to sys.path so modules can be imported
    add_temp_workflow_path(test_workspace)
    
    # Debug: Check that the files were created correctly
    print(f"\nTest directory structure:")
    for root, dirs, files in os.walk(test_workspace):
        level = root.replace(str(test_workspace), '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{sub_indent}{f}")

    # First load the command directory directly to debug
    cmd_dir = CommandDirectory.load(str(test_workspace))
    print("\nCommand Directory Metadata:")
    print(f"Commands: {cmd_dir.get_commands()}")
    
    # Now use the router
    router = RoutingDefinition.build(test_workspace)
    try:
        router.scan(use_cache=False)  # Bypass cache for tests
        
        print("\nRouter Command Directory:")
        pprint.pprint(router.command_directory_map)
        print("\nRouter Routing Definition:")
        pprint.pprint(router.routing_definition_map)

        # Command directory checks - need to account for core commands
        assert "a" in router.command_directory_map["*"]
        assert "x" in router.command_directory_map["ctx1"]
        assert "y" in router.command_directory_map["ctx1"]
        assert "dummy" in router.command_directory_map["ctx2"]

        # Routing definition checks
        assert "*" in router.routing_definition_map["a"]
        assert "ctx1" in router.routing_definition_map["x"]
        assert "ctx1" in router.routing_definition_map["y"]
        assert "ctx2" in router.routing_definition_map["dummy"]
        
        # Check nonexistent command - use get() to avoid KeyError
        assert router.routing_definition_map.get("unknown", set()) == set()
    finally:
        # Clean up the temporary workspace to avoid leaving files in repo root
        shutil.rmtree(test_workspace, ignore_errors=True)


def test_missing_root_dir(tmp_path, add_temp_workflow_path):
    # Expect router.scan to raise RuntimeError when the workflow path lacks a _commands folder.
    add_temp_workflow_path(Path.cwd())

    router = RoutingDefinition.build(tmp_path / "nonexistent")

    with pytest.raises(RuntimeError):
        router.scan(use_cache=False)  # Bypass cache for tests


def test_scan_with_commands_directory():
    """Test that the router can scan a directory with commands."""
    # Create a test workspace
    test_workspace = os.path.join(os.path.dirname(__file__), "example_workflow")
    
    # Create a router and scan the directory
    router = RoutingDefinition.build(test_workspace)
    
    # Check that the command directory is populated
    assert len(router.command_directory_map) > 0
    
    # Check that the routing definition is populated
    assert len(router.routing_definition_map) > 0
    
    # Check that the global context exists
    assert "*" in router.command_directory_map
    
    # Check that commands in the global context are mapped correctly
    for command in router.command_directory_map["*"]:
        assert "*" in router.routing_definition_map[command]


def test_get_commands_for_context():
    """Test that get_commands_for_context returns the correct commands."""
    # Create a test workspace
    test_workspace = os.path.join(os.path.dirname(__file__), "example_workflow")
    
    # Create a router and scan the directory
    router = RoutingDefinition.build(test_workspace)
    
    # Get commands for the global context
    commands = router.get_commands_for_context("*")
    
    # Check that the commands are returned as a set
    assert isinstance(commands, set)
    
    # Check that the commands are the same as in the command directory
    assert commands == router.command_directory_map["*"]


def test_get_contexts_for_command():
    """Test that get_contexts_for_command returns the correct contexts."""
    # Create a test workspace
    test_workspace = os.path.join(os.path.dirname(__file__), "example_workflow")
    
    # Create a router and scan the directory
    router = RoutingDefinition.build(test_workspace)
    
    # Ensure we have at least one command in the global context
    if not router.command_directory_map["*"]:
        router.command_directory_map["*"] = {"dummy_command"}
        router.routing_definition_map["dummy_command"] = {"*"}
    
    # Get a command from the global context
    command = next(iter(router.command_directory_map["*"]))
    
    # Get the contexts for the command
    contexts = router.get_contexts_for_command(command)
    
    # Check that the contexts include the global context
    assert "*" in contexts
    
    # Test with a nonexistent command
    contexts = router.get_contexts_for_command("nonexistent_command")
    assert contexts == set()


def test_scan_with_nonexistent_directory():
    """Test that the router handles a nonexistent directory gracefully."""
    # Create a temporary directory that doesn't exist
    with tempfile.TemporaryDirectory() as tmpdir:
        nonexistent_dir = Path(tmpdir) / "nonexistent"
        
        # Create a router and scan the directory
        # This should not raise an exception, but the command directory and routing definition should be empty
        try:
            router = RoutingDefinition.build(str(nonexistent_dir))
            
            # Check that the command directory is empty
            assert len(router.command_directory_map) == 1  # Just the global context
            assert router.command_directory_map["*"] == set()
            
            # Check that the routing definition is empty
            assert len(router.routing_definition_map) == 0
        except Exception as e:
            # If an exception is raised, fail the test with the exception message
            pytest.fail(f"RoutingDefinition.build raised an exception for a nonexistent directory: {str(e)}") 