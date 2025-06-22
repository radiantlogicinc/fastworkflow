from pathlib import Path
import os
import pprint

import fastworkflow
from fastworkflow.command_router import CommandRouter
from fastworkflow.command_directory import CommandDirectory


def _create_commands(tmp_path: Path):
    """Create a sample _commands tree:

    _commands/
        a.py
        ctx1/
            x.py
            y.py
        ctx2/
            _ctx2.py
    """
    root = tmp_path / "_commands"
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


def test_scan_and_lookup(tmp_path, add_temp_workflow_path):
    fastworkflow.init({"SPEEDDICT_FOLDERNAME": "___workflow_contexts"})

    cmd_root = _create_commands(tmp_path)
    # Add the temporary path to sys.path
    add_temp_workflow_path(tmp_path)
    
    # Debug: Check that the files were created correctly
    print(f"\nTest directory structure:")
    for root, dirs, files in os.walk(tmp_path):
        level = root.replace(str(tmp_path), '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{sub_indent}{f}")

    # First load the command directory directly to debug
    cmd_dir = CommandDirectory.load(str(tmp_path))
    print("\nCommand Directory Metadata:")
    print(f"Commands: {cmd_dir.get_commands()}")
    
    # Now use the router
    router = CommandRouter(tmp_path)
    router.scan(use_cache=False)  # Bypass cache for tests
    
    print("\nRouter Command Directory:")
    pprint.pprint(router.command_directory)
    print("\nRouter Routing Definition:")
    pprint.pprint(router.routing_definition)

    # Command directory checks - need to account for core commands
    assert "a" in router.get_commands_for_context("*")
    assert "x" in router.get_commands_for_context("ctx1")
    assert "y" in router.get_commands_for_context("ctx1")
    assert "dummy" in router.get_commands_for_context("ctx2")

    # Routing definition checks
    assert "*" in router.get_contexts_for_command("a")
    assert "ctx1" in router.get_contexts_for_command("x")
    assert "ctx1" in router.get_contexts_for_command("y")
    assert "ctx2" in router.get_contexts_for_command("dummy")
    assert router.get_contexts_for_command("unknown") == set()


def test_missing_root_dir(tmp_path, add_temp_workflow_path):
    add_temp_workflow_path(tmp_path)
    router = CommandRouter(tmp_path / "nonexistent")
    router.scan(use_cache=False)  # Bypass cache for tests
    
    # Check that we have the global context, but don't check for specific core commands
    # as they might change in the future
    assert "*" in router.command_directory 