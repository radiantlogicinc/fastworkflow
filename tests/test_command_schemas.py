"""Schemas for clients (fix-3rsq.12.2, design 3.14).

- ``GET /command_schemas`` (operation_id ``get_command_schemas``) lists every listed command of the
  channel's workflow with its kind and its input and output JSON Schemas; framework commands are hidden.
- A per-command MCP tool declares ``outputSchema`` when its command declares a ``Signature.Output``, and
  its result carries ``structuredContent`` with the Output as JSON, beside the text content. A command
  that declares an Output and returns only text is a call error (fix-3rsq.declared-output-required).
- A turn's ``command_response`` still carries ``next_actions`` and ``recommendations``.

No mocks and no model calls: the tests copy ``tests/hello_world_workflow`` into a temp folder and edit the
copy's command. The copy's ``add_two_numbers`` either returns its ``Signature.Output`` (the shipped form of
3.14) or declares no Output at all, so both cases are exercised on real code paths.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import fastworkflow
from fastworkflow.mcp_server import FastWorkflowMCPServer, command_schema_listing

HELLO = Path(__file__).resolve().parent / "hello_world_workflow"
_COPY_IGNORE = shutil.ignore_patterns("___workflow_contexts", "___convo_info", "__pycache__")
COMMAND_FILE = "_commands/add_two_numbers.py"
FRAMEWORK_COMMANDS = ("review_plan", "review_proposal", "review_subtasks")

_HEADER = """import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.workflow import Workflow
from fastworkflow.utils.signatures import InputForParamExtraction
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from fastworkflow.utils.context_utils import list_context_names
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from ..application.add_two_numbers import add_two_numbers


class Signature:
    class Input(BaseModel):
        first_num: float = Field(description="First number")
        second_num: float = Field(description="Second number")

"""

_OUTPUT_CLASS = """    class Output(BaseModel):
        sum_of_two_numbers: float = Field(description="The sum of the two provided numbers")

"""

_BODY = """    plain_utterances = [
        "add two numbers",
        "add two numbers {a} {b}",
        "call add_two_numbers with {a} {b}"
    ]

    template_utterances = []

    @staticmethod
    def generate_utterances(workflow: Workflow, command_name: str) -> list[str]:
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)

class ResponseGenerator:
    def _process_command(self, workflow: Workflow, input: Signature.Input) -> float:
        '''Execute add_two_numbers function'''
        return add_two_numbers(a=input.first_num, b=input.second_num)

"""

_CALL_WITH_OUTPUT = """    def __call__(self, workflow: Workflow, command: str, command_parameters: Signature.Input) -> Signature.Output:
        # Design 3.14: the command returns its Signature.Output; the framework wraps it.
        return Signature.Output(sum_of_two_numbers=self._process_command(workflow, command_parameters))
"""

_CALL_WITHOUT_OUTPUT = """    def __call__(self, workflow: Workflow, command: str, command_parameters: Signature.Input) -> str:
        total = self._process_command(workflow, command_parameters)
        return f"Response: {total}"
"""


_CALL_WITH_CR_OUTPUT = """    def __call__(self, workflow: Workflow, command: str, command_parameters: Signature.Input) -> fastworkflow.CommandResponse:
        total = self._process_command(workflow, command_parameters)
        return fastworkflow.CommandResponse(response=f"Response: {total}",
                                            output=Signature.Output(sum_of_two_numbers=total))
"""


def _copy_hello_declaring_output_with_call(folder: Path, call: str) -> str:
    """A copy of the hello workflow whose add_two_numbers declares an Output and returns ``call``'s form."""
    shutil.copytree(HELLO, folder, ignore=_COPY_IGNORE, symlinks=True)
    (folder / COMMAND_FILE).write_text(_HEADER + _OUTPUT_CLASS + _BODY + call)
    return str(folder)


def _copy_hello(folder: Path, *, declare_output: bool) -> str:
    """A copy of the hello workflow whose add_two_numbers declares and returns an Output, or declares none."""
    shutil.copytree(HELLO, folder, ignore=_COPY_IGNORE, symlinks=True)
    source = _HEADER + (_OUTPUT_CLASS if declare_output else "") + _BODY
    source += _CALL_WITH_OUTPUT if declare_output else _CALL_WITHOUT_OUTPUT
    (folder / COMMAND_FILE).write_text(source)
    return str(folder)


@pytest.fixture
def env_files():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_file = os.path.join(project_root, "env", ".env")
    passwords_file = os.path.join(project_root, "passwords", ".env")
    if not os.path.isfile(env_file) or not os.path.isfile(passwords_file):
        pytest.skip("env files missing for FastAPI tests")
    return env_file, passwords_file


@pytest.fixture
def output_workflow(tmp_path):
    return _copy_hello(tmp_path / "hello_output", declare_output=True)


@pytest.fixture
def no_output_workflow(tmp_path):
    return _copy_hello(tmp_path / "hello_no_output", declare_output=False)


@pytest.fixture
def app_module(output_workflow, env_files, tmp_path):
    env_file, passwords_file = env_files
    sys.argv = ["pytest", "--workflow_path", output_workflow, "--env_file_path", env_file,
                "--passwords_file_path", passwords_file]
    import fastworkflow.run_fastapi_mcp.__main__ as main

    importlib.reload(main)
    from tests.fastapi_hermetic import init_fastapi_hermetic_env, restore_fastapi_env

    previous_env = init_fastapi_hermetic_env(env_file, passwords_file, tmp_path / "workflow_contexts")
    try:
        yield main
    finally:
        restore_fastapi_env(previous_env)


@pytest.fixture
def mcp_env(env_files, tmp_path):
    """Initialize fastworkflow for the per-command MCP server; restored after the test."""
    from tests.fastapi_hermetic import init_fastapi_hermetic_env, restore_fastapi_env

    env_file, passwords_file = env_files
    previous_env = init_fastapi_hermetic_env(env_file, passwords_file, tmp_path / "workflow_contexts")
    try:
        yield
    finally:
        restore_fastapi_env(previous_env)


def _mcp_server(workflow_path: str) -> FastWorkflowMCPServer:
    chat_session = fastworkflow.ChatSession()
    chat_session.start_workflow(workflow_path, workflow_id_str=str(uuid.uuid4()), keep_alive=True)
    return FastWorkflowMCPServer(chat_session)


def _auth(client: TestClient, channel_id: str) -> dict:
    init = client.post("/initialize", json={"channel_id": channel_id, "user_id": "schemas"})
    assert init.status_code == 200, init.text
    return {"Authorization": f"Bearer {init.json()['access_token']}"}


# ---------------------------------------------------------------------------
# GET /command_schemas
# ---------------------------------------------------------------------------

def test_command_schemas_lists_every_listed_command_with_kind_and_schemas(app_module, output_workflow):
    channel_id = f"schemas_{uuid.uuid4().hex[:8]}"
    client = TestClient(app_module.app)
    headers = _auth(client, channel_id)

    resp = client.get("/command_schemas", headers=headers)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Every listed command, and nothing else: framework commands and the internal CME workflow's
    # commands (IntentDetection/*, wildcard) are in the directory but hidden.
    from fastworkflow.command_directory import CommandDirectory
    cme_root = os.path.realpath(fastworkflow.get_internal_workflow_path("command_metadata_extraction"))
    directory = CommandDirectory.load(output_workflow).map_command_2_metadata
    expected = {
        name for name, meta in directory.items()
        if not meta.framework
        and not os.path.realpath(meta.workflow_folderpath or output_workflow).startswith(cme_root)
    }
    assert "wildcard" in directory and "IntentDetection/go_up" in directory  # present, then hidden
    assert set(body) == expected
    assert "add_two_numbers" in body
    assert not set(body) & set(FRAMEWORK_COMMANDS)
    assert not any(name.startswith("IntentDetection/") or name == "wildcard" for name in body)

    entry = body["add_two_numbers"]
    assert set(entry) == {"kind", "input", "output"}
    assert entry["kind"] == "execution"
    assert set(entry["input"]["properties"]) == {"first_num", "second_num"}
    assert entry["output"]["properties"]["sum_of_two_numbers"]["type"] == "number"


def test_command_schemas_output_is_null_for_a_command_with_no_output(app_module_no_output, no_output_workflow):
    channel_id = f"schemas_{uuid.uuid4().hex[:8]}"
    client = TestClient(app_module_no_output.app)
    headers = _auth(client, channel_id)

    body = client.get("/command_schemas", headers=headers).json()

    assert body["add_two_numbers"]["kind"] == "execution"
    assert body["add_two_numbers"]["input"] is not None
    assert body["add_two_numbers"]["output"] is None


@pytest.fixture
def app_module_no_output(no_output_workflow, env_files, tmp_path):
    env_file, passwords_file = env_files
    sys.argv = ["pytest", "--workflow_path", no_output_workflow, "--env_file_path", env_file,
                "--passwords_file_path", passwords_file]
    import fastworkflow.run_fastapi_mcp.__main__ as main

    importlib.reload(main)
    from tests.fastapi_hermetic import init_fastapi_hermetic_env, restore_fastapi_env

    previous_env = init_fastapi_hermetic_env(env_file, passwords_file, tmp_path / "workflow_contexts_no_output")
    try:
        yield main
    finally:
        restore_fastapi_env(previous_env)


def test_command_schemas_requires_a_token(app_module):
    client = TestClient(app_module.app)
    assert client.get("/command_schemas").status_code == 401


def test_command_schema_listing_hides_framework_commands(output_workflow, mcp_env):
    listing = command_schema_listing(output_workflow)
    assert "review_plan" not in listing and "review_subtasks" not in listing
    assert "review_proposal" not in listing


# ---------------------------------------------------------------------------
# Per-command MCP tools: outputSchema and structuredContent
# ---------------------------------------------------------------------------

def test_mcp_tool_declares_output_schema_and_returns_structured_content(output_workflow, mcp_env):
    server = _mcp_server(output_workflow)

    tools = {tool["name"]: tool for tool in server.list_tools()}
    tool = tools["add_two_numbers"]
    assert "outputSchema" in tool
    assert tool["outputSchema"]["properties"]["sum_of_two_numbers"]["type"] == "number"

    result = server.call_tool("add_two_numbers", {"first_num": 2.0, "second_num": 3.0})
    assert result.isError is False
    assert result.structuredContent == {"sum_of_two_numbers": 5.0}
    assert "5" in result.content[0].text

    # Over JSON-RPC, the result carries structuredContent beside content, with the MCP field names.
    rpc = server.handle_json_rpc_request({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "add_two_numbers", "arguments": {"first_num": 2.0, "second_num": 3.0}},
    })
    assert rpc["result"]["structuredContent"] == {"sum_of_two_numbers": 5.0}
    assert rpc["result"]["content"][0]["type"] == "text"
    assert rpc["result"]["isError"] is False


def test_mcp_tool_without_an_output_declares_no_output_schema(no_output_workflow, mcp_env):
    server = _mcp_server(no_output_workflow)

    tools = {tool["name"]: tool for tool in server.list_tools()}
    assert "outputSchema" not in tools["add_two_numbers"]

    rpc = server.handle_json_rpc_request({
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": "add_two_numbers", "arguments": {"first_num": 2.0, "second_num": 3.0}},
    })
    assert "structuredContent" not in rpc["result"]
    assert rpc["result"]["content"][0]["text"]


def test_declared_output_returned_as_text_is_a_call_error_over_mcp(tmp_path, mcp_env):
    server = _mcp_server(_copy_hello_declaring_output_with_call(tmp_path / "text_output", _CALL_WITHOUT_OUTPUT))

    result = server.call_tool("add_two_numbers", {"first_num": 2.0, "second_num": 3.0})
    assert result.isError is True
    assert result.content[0].text == (
        "Error: add_two_numbers declares an Output but returned str; return the Output, "
        "or CommandResponse(response=..., output=<Output>)")
    assert getattr(result, "structuredContent", None) is None


def test_command_response_carrying_the_output_gives_structured_content_over_mcp(tmp_path, mcp_env):
    server = _mcp_server(_copy_hello_declaring_output_with_call(tmp_path / "cr_output", _CALL_WITH_CR_OUTPUT))

    result = server.call_tool("add_two_numbers", {"first_num": 2.0, "second_num": 3.0})
    assert result.isError is False
    assert result.structuredContent == {"sum_of_two_numbers": 5.0}
    assert result.content[0].text == "Response: 5.0"


# ---------------------------------------------------------------------------
# The REST turn body keeps next_actions and recommendations
# ---------------------------------------------------------------------------

def test_turn_body_command_response_keeps_next_actions_and_recommendations(app_module):
    channel_id = f"schemas_{uuid.uuid4().hex[:8]}"
    client = TestClient(app_module.app)
    headers = _auth(client, channel_id)

    resp = client.post(
        "/perform_action",
        headers=headers,
        json={
            "action": {
                "command_name": "add_two_numbers",
                "parameters": {"first_num": 2.0, "second_num": 3.0},
            },
            "timeout_seconds": 30,
        },
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["command_outputs"], "the action produced no command output"
    for command_output in body["command_outputs"]:
        command_response = command_output["command_response"]
        assert "next_actions" in command_response
        assert "recommendations" in command_response
        assert command_response["next_actions"] == []
        assert command_response["recommendations"] == []
    # The Output rides in command_response.output, as the schema describes.
    assert body["command_outputs"][-1]["command_response"]["output"] == {"sum_of_two_numbers": 5.0}
