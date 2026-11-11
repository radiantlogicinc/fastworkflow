"""--startup_action accepts a JSON file path or an inline JSON object (shared helper)."""

import json

import pytest

import fastworkflow


def test_file_path_loads_the_action(tmp_path):
    path = tmp_path / "startup_action.json"
    path.write_text(json.dumps({
        "command_name": "User/get_details",
        "parameters": {"channel_id": "u-42"},
    }), encoding="utf-8")

    action = fastworkflow.load_startup_action(str(path))

    assert action.command_name == "User/get_details"
    assert action.parameters == {"channel_id": "u-42"}


def test_inline_json_object_loads_the_action():
    action = fastworkflow.load_startup_action(
        '{"command_name": "User/get_details", "parameters": {"channel_id": "u-42"}}'
    )

    assert action.command_name == "User/get_details"
    assert action.parameters == {"channel_id": "u-42"}


def test_inline_json_tolerates_leading_whitespace():
    action = fastworkflow.load_startup_action('  {"command_name": "A/b"}')

    assert action.command_name == "A/b"


def test_missing_file_gives_a_clear_error(tmp_path):
    missing = tmp_path / "no_such_action.json"

    with pytest.raises(ValueError) as excinfo:
        fastworkflow.load_startup_action(str(missing))

    message = str(excinfo.value)
    assert "--startup_action file not found" in message
    assert str(missing) in message
    assert "JSON file" in message
