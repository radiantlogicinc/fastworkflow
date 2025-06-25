import os
import pytest
import fastworkflow
from fastworkflow.command_directory import CommandDirectory, UtteranceMetadata


@pytest.fixture(scope="module")
def cme_workflow_path() -> str:
    """Return the absolute path to the internal command_metadata_extraction workflow."""
    return fastworkflow.get_internal_workflow_path("command_metadata_extraction")


def test_cme_command_directory_populates_metadata(cme_workflow_path):
    """Ensure CommandDirectory.load populates command, utterance and context metadata for CME workflow."""
    cmd_dir = CommandDirectory.load(cme_workflow_path)

    # Expected contexts and their commands (hard-coded – these should remain stable)
    expected_mapping: dict[str, list[str]] = {
        "ErrorCorrection": [
            "abort",
            "you_misunderstood",
        ],
        "IntentDetection": [
            "reset_context",
            "what_can_i_do",
            "what_is_current_context",
        ],
    }

    # Verify command & utterance metadata is present for every expected command
    for context_name, command_names in expected_mapping.items():
        for command_name in command_names:
            qualified = f"{context_name}/{command_name}"

            # ---- Command metadata ----
            assert qualified in cmd_dir.map_command_2_metadata, (
                f"Missing CommandMetadata for '{qualified}'"
            )
            cmd_meta = cmd_dir.map_command_2_metadata[qualified]
            # Critical fields should not be empty
            assert cmd_meta.response_generation_module_path, (
                f"response_generation_module_path empty for '{qualified}'"
            )

            # ---- Utterance metadata ----
            utter_meta = cmd_dir.get_utterance_metadata(qualified)
            assert isinstance(utter_meta, UtteranceMetadata), (
                f"Missing or invalid UtteranceMetadata for '{qualified}'"
            )
            # At least one utterance (plain or template) must exist
            assert (
                utter_meta.plain_utterances or utter_meta.template_utterances
            ), f"No utterances found for '{qualified}'"

    # ---- Context metadata ----
    # map_context_2_metadata can be empty if no context callback classes exist.
    # We simply assert that the attribute exists and is a dictionary.
    assert isinstance(cmd_dir.map_context_2_metadata, dict)

    # Verify that 'wildcard' exists as a global command
    assert "wildcard" in cmd_dir.map_command_2_metadata, "Missing CommandMetadata for 'wildcard'"
    assert "wildcard" in cmd_dir.map_command_2_utterance_metadata, "Missing UtteranceMetadata for 'wildcard'" 