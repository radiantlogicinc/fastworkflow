import os
from pathlib import Path

import fastworkflow
from fastworkflow.command_metadata_api import CommandMetadataAPI


def _hello_world_path() -> str:
    return str((Path(__file__).parent / "hello_world_workflow").resolve())


def _cme_workflow_path() -> str:
    return fastworkflow.get_internal_workflow_path("command_metadata_extraction")


def test_get_enhanced_command_info_hello_world(setup_test_environment):
    subject_path = _hello_world_path()
    cme_path = _cme_workflow_path()

    result = CommandMetadataAPI.get_enhanced_command_info(
        subject_workflow_path=subject_path,
        cme_workflow_path=cme_path,
        active_context_name="*",
    )

    assert isinstance(result, dict)
    assert "commands" in result

    commands = result["commands"]
    add_cmd = next((c for c in commands if c.get("name") == "add_two_numbers"), None)
    assert add_cmd is not None

    # Validate utterances and signature inputs/outputs
    expected_plain = [
        "add two numbers",
        "add two numbers {a} {b}",
        "call add_two_numbers with {a} {b}",
    ]

    # plain_utterances come from Signature on the Signature class
    assert set(add_cmd.get("plain_utterances", [])) == set(expected_plain)

    inputs = add_cmd.get("inputs", [])
    input_names = {i.get("name") for i in inputs}
    assert {"first_num", "second_num"} <= input_names

    # Outputs should include the computed field
    outputs = add_cmd.get("outputs", [])
    output_names = {o.get("name") for o in outputs}
    assert "sum_of_two_numbers" in output_names


def test_get_params_for_all_commands_hello_world(setup_test_environment):
    subject_path = _hello_world_path()

    params = CommandMetadataAPI.get_params_for_all_commands(subject_path)
    assert isinstance(params, dict)
    assert "add_two_numbers" in params

    add_meta = params["add_two_numbers"]
    assert "inputs" in add_meta and "outputs" in add_meta

    input_names = {i.get("name") for i in add_meta["inputs"]}
    assert {"first_num", "second_num"} <= input_names

    output_names = {o.get("name") for o in add_meta["outputs"]}
    assert "sum_of_two_numbers" in output_names


def test_get_all_commands_metadata_hello_world(setup_test_environment):
    subject_path = _hello_world_path()

    metadata_list = CommandMetadataAPI.get_all_commands_metadata(subject_path)
    assert isinstance(metadata_list, list)

    add_meta = next((m for m in metadata_list if m.get("command_name") == "add_two_numbers"), None)
    assert add_meta is not None

    # File path should point to the command module
    assert isinstance(add_meta.get("file_path"), (str, Path))
    assert str(add_meta["file_path"]).endswith("/_commands/add_two_numbers.py")

    # Input/Output models should be detected
    assert add_meta.get("input_model") == "Input"
    assert add_meta.get("output_model") == "Output"

    # Utterances should include the known set
    expected_plain = [
        "add two numbers",
        "add two numbers {a} {b}",
        "call add_two_numbers with {a} {b}",
    ]
    assert set(add_meta.get("plain_utterances", [])) == set(expected_plain)


