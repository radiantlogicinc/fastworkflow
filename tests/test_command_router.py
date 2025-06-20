from pathlib import Path

import fastworkflow
from fastworkflow.command_router import CommandRouter


def _create_commands(tmp_path: Path):
    """Create a sample _commands tree:

    _commands/
        a.py
        ctx1/
            x.py
            y.py
        ctx2/
    """
    root = tmp_path / "_commands"
    root.mkdir()

    # global command
    (root / "a.py").write_text("# dummy")
    # ignored file
    (root / "ignore.txt").write_text("txt")

    # ctx1 with two commands
    ctx1 = root / "ctx1"
    ctx1.mkdir()
    (ctx1 / "x.py").write_text("# x")
    (ctx1 / "y.py").write_text("# y")

    # ctx2 empty directory
    (root / "ctx2").mkdir()

    return root


def test_scan_and_lookup(tmp_path):
    fastworkflow.init({"SPEEDDICT_FOLDERNAME": "___workflow_contexts"})

    cmd_root = _create_commands(tmp_path)

    router = CommandRouter(cmd_root)
    router.scan()

    # Command directory checks
    assert router.get_commands_for_context("*") == {"a"}
    assert router.get_commands_for_context("ctx1") == {"x", "y"}
    assert router.get_commands_for_context("ctx2") == set()

    # Routing definition checks
    assert router.get_contexts_for_command("a") == {"*"}
    assert router.get_contexts_for_command("x") == {"ctx1"}
    assert router.get_contexts_for_command("y") == {"ctx1"}
    assert router.get_contexts_for_command("unknown") == set()


def test_missing_root_dir(tmp_path):
    router = CommandRouter(tmp_path / "nonexistent")
    router.scan()
    # Only global context with empty commands
    assert router.command_directory == {"*": set()}
    assert router.routing_definition == {} 