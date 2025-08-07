import fastworkflow
from fastworkflow.command_directory import CommandDirectory
import pytest


def test_startup_command_has_no_input_and_no_utterances():
    """Ensure that the global `startup` command is recognised as an action-only
    command (no `Signature.Input`) and, consequently, that its metadata does
    not advertise a `command_parameters_class`.  Attempting to fetch
    utterances should raise `KeyError`.
    """
    # Use the real test workflow under the repo's tests directory
    workflow_path = "./tests/todo_list_workflow"
    cmd_dir = CommandDirectory.load(workflow_path)

    # Hydrate so metadata is fully populated before assertions
    cmd_dir.ensure_command_hydrated("startup")
    metadata = cmd_dir.get_command_metadata("startup")

    # The command must *not* define Signature.Input
    assert metadata.command_parameters_class is None

    # And there should be no utterance metadata registered
    assert cmd_dir.get_utterance_metadata("startup") is None 