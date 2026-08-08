"""Integration tests for the TurnOutput response contract (fix-qtq).

Every turn surface of run_fastapi_mcp — /invoke_agent, /invoke_agent_stream,
/invoke_assistant, /perform_action, and /initialize's startup_output — now
answers with the public ``TurnOutput`` projection. Before this migration some of
them answered with the older ``CommandOutput`` shape instead, so an integrator
had to handle both. These tests pin the single shape, and pin the distinction
that makes it safe: HTTP status reports whether the CALL worked, while
status/failure_reason/success report whether the TURN worked. A failed turn is a
200.

No mocks (testing_rules.mdc): the deterministic tests drive real direct actions
against tests/hello_world_workflow, the response-rendering tests build real
``TurnExecution``/``TurnOutput`` values, and the streaming tests run a real agent
turn against fastworkflow/examples/hello_world.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
import uuid

import pytest
from fastapi.testclient import TestClient

import fastworkflow
from fastworkflow.run_fastapi_mcp.turns import (
    ExecState,
    TurnExecution,
    render_turn_response,
)
from fastworkflow.run_fastapi_mcp.utils import persist_pending_after_turn

# Keys of the public TurnOutput projection (fastworkflow/turn.py). ``success`` is
# a computed field, so it is part of the serialized shape, not something the
# server adds on the way out.
TURN_OUTPUT_KEYS = frozenset(
    {"turn_key", "status", "failure_reason", "answer", "command_outputs", "success"}
)

# Top-level keys the pre-migration CommandOutput shape had. They must NOT be at
# the top level of a turn response any more; they live inside each entry of
# ``command_outputs``.
COMMAND_OUTPUT_ONLY_KEYS = frozenset({"workflow_name", "command_name", "command_parameters"})


@pytest.fixture
def env_files():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_file = os.path.join(project_root, "env", ".env")
    passwords_file = os.path.join(project_root, "passwords", ".env")
    if not os.path.isfile(env_file) or not os.path.isfile(passwords_file):
        pytest.skip("env files missing for FastAPI tests")
    return env_file, passwords_file


def _load_app_module(workflow_path: str, env_files):
    env_file, passwords_file = env_files
    sys.argv = [
        "pytest",
        "--workflow_path",
        workflow_path,
        "--env_file_path",
        env_file,
        "--passwords_file_path",
        passwords_file,
    ]
    import fastworkflow.run_fastapi_mcp.__main__ as main

    importlib.reload(main)
    from dotenv import dotenv_values

    fastworkflow.init({**dotenv_values(env_file), **dotenv_values(passwords_file)})
    if fastworkflow.RoutingRegistry:
        fastworkflow.RoutingRegistry.clear_registry()
    return main


@pytest.fixture
def action_app_module(env_files):
    """Server over tests/hello_world_workflow: direct actions need no NLU or LLM."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    workflow_path = os.path.join(project_root, "tests", "hello_world_workflow")
    if not os.path.isdir(workflow_path):
        pytest.skip(f"hello_world_workflow not found at {workflow_path}")
    return _load_app_module(workflow_path, env_files)


@pytest.fixture
def agent_app_module(env_files):
    """Server over the trained example workflow, for real agent (streaming) turns."""
    package_path = fastworkflow.get_fastworkflow_package_path()
    workflow_path = os.path.join(package_path, "examples", "hello_world")
    if not os.path.isdir(workflow_path):
        pytest.skip(f"hello_world workflow not found at {workflow_path}")
    return _load_app_module(workflow_path, env_files)


def _channel(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _add_action() -> dict:
    return {
        "command_name": "add_two_numbers",
        "parameters": {"first_num": 2.0, "second_num": 3.0},
    }


def assert_turn_output_projection(payload: dict, *, status: str | None = None) -> None:
    """Assert *payload* carries the TurnOutput projection, with the right types."""
    missing = TURN_OUTPUT_KEYS - set(payload)
    assert not missing, f"TurnOutput keys missing from response: {sorted(missing)}"
    leaked = COMMAND_OUTPUT_ONLY_KEYS & set(payload)
    assert not leaked, f"pre-migration CommandOutput keys at top level: {sorted(leaked)}"
    assert isinstance(payload["turn_key"], str) and payload["turn_key"]
    assert isinstance(payload["success"], bool)
    assert isinstance(payload["answer"], str)
    assert isinstance(payload["command_outputs"], list)
    assert payload["failure_reason"] is None or isinstance(payload["failure_reason"], str)
    if status is not None:
        assert payload["status"] == status


def _done_execution(result, *, kind: str = "invoke_agent") -> TurnExecution:
    """A real, finished TurnExecution carrying *result* — no mocks involved."""
    execn = TurnExecution(
        turn_key=fastworkflow.mint_turn_key(),
        channel_id=_channel("render"),
        kind=kind,
        idempotency_key="contract",
    )
    execn.exec_state = ExecState.DONE
    execn.result = result
    return execn


# ---------------------------------------------------------------------------
# The wire shape, over real HTTP
# ---------------------------------------------------------------------------

def test_perform_action_answers_with_the_turn_output_projection(action_app_module):
    """A direct action is its own logical turn, reported as a TurnOutput."""
    channel_id = _channel("action")
    with TestClient(action_app_module.app) as client:
        init = client.post("/initialize", json={"channel_id": channel_id})
        assert init.status_code == 200
        headers = {"Authorization": f"Bearer {init.json()['access_token']}"}

        resp = client.post(
            "/perform_action",
            headers=headers,
            json={"action": _add_action(), "timeout_seconds": 30},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert_turn_output_projection(body, status="completed")
    assert body["success"] is True
    # The transport's own lifecycle field, distinct from the turn's status.
    assert body["exec_state"] == "done"
    # Each command's CommandOutput is preserved one level down, artifacts and all.
    assert body["command_outputs"], "the action produced no command output"
    assert "command_responses" in body["command_outputs"][-1]
    assert "5" in body["answer"], f"expected the sum in the answer, got {body['answer']!r}"


def test_startup_output_is_the_same_projection_as_a_turn_endpoint(action_app_module):
    """The payoff: /initialize and /perform_action no longer disagree on shape.

    startup_output used to be a bare CommandOutput while the turn endpoints
    already returned a TurnOutput-based body, which is exactly the inconsistency
    an integrator had to write two code paths for.
    """
    channel_id = _channel("shape")
    with TestClient(action_app_module.app) as client:
        init = client.post(
            "/initialize",
            json={
                "channel_id": channel_id,
                "user_id": "u_shape",
                "startup_action": _add_action(),
            },
        )
        assert init.status_code == 200
        startup_output = init.json()["startup_output"]

        headers = {"Authorization": f"Bearer {init.json()['access_token']}"}
        action = client.post(
            "/perform_action",
            headers=headers,
            json={"action": _add_action(), "timeout_seconds": 30},
        )

    assert startup_output is not None, "startup finished in-window but reported nothing"
    assert_turn_output_projection(startup_output, status="completed")
    assert action.status_code == 200
    assert TURN_OUTPUT_KEYS <= set(action.json())
    assert TURN_OUTPUT_KEYS <= set(startup_output)


# ---------------------------------------------------------------------------
# Transport status vs. turn outcome
# ---------------------------------------------------------------------------

def test_a_failed_turn_is_still_a_successful_response():
    """A turn that failed is not a call that failed: 200, with the reason in body.

    Mapping this onto an HTTP error would collapse two independent questions and
    leave a client unable to distinguish "the workflow could not finish" from
    "the server is broken".
    """
    result = fastworkflow.TurnOutput(
        turn_key=fastworkflow.mint_turn_key(),
        status=fastworkflow.TurnStatus.FAILED,
        failure_reason="max_iters_exhausted",
        answer="I ran out of steps.",
        command_outputs=[
            fastworkflow.CommandOutput(
                command_responses=[
                    fastworkflow.CommandResponse(response="nope", success=False)
                ]
            )
        ],
    )

    code, body = render_turn_response(_done_execution(result))

    assert code == 200
    assert_turn_output_projection(body, status="failed")
    assert body["failure_reason"] == "max_iters_exhausted"
    assert body["success"] is False
    assert "error" not in body, "a failed turn must not be reported as a transport error"


def test_an_awaiting_user_turn_reports_its_question_in_the_answer():
    """A suspended turn is a normal 200 whose status says it is waiting."""
    result = fastworkflow.TurnOutput(
        turn_key=fastworkflow.mint_turn_key(),
        status=fastworkflow.TurnStatus.AWAITING_USER,
        answer="Which list did you mean?",
        command_outputs=[
            fastworkflow.CommandOutput(
                command_name="ask_user",
                command_responses=[
                    fastworkflow.CommandResponse(
                        response="Which list did you mean?", success=False
                    )
                ],
            )
        ],
    )

    code, body = render_turn_response(_done_execution(result))

    assert code == 200
    assert_turn_output_projection(body, status="awaiting_user")
    assert body["answer"] == "Which list did you mean?"
    # Unanswered ask_user counts as not-yet-successful (TurnOutput.success docs).
    assert body["success"] is False


def test_a_deferred_turn_reports_only_its_polling_handle():
    """Nothing to project yet, so the 202 body stays the handle plus exec_state."""
    execn = TurnExecution(
        turn_key=fastworkflow.mint_turn_key(),
        channel_id=_channel("defer"),
        kind="invoke_agent",
        idempotency_key="contract",
    )
    execn.exec_state = ExecState.RUNNING

    code, body = render_turn_response(execn)

    assert code == 202
    assert body == {"turn_key": execn.turn_key, "exec_state": "running"}


# ---------------------------------------------------------------------------
# Durable suspended state now keys off TurnStatus
# ---------------------------------------------------------------------------

def test_suspended_state_is_persisted_from_the_turn_status(action_app_module):
    """persist_pending_after_turn reads TurnStatus, not a command's artifacts dict.

    It used to infer suspension from ``command_responses[0].artifacts
    ['awaiting_user']`` — a per-command detail standing in for a turn-level fact.
    Here the context does NOT claim to be awaiting_user, so only the status can
    save the blob.
    """
    channel_id = _channel("suspend")
    app_module = action_app_module
    store = app_module.session_manager.session_state_store

    async def body():
        await app_module.ensure_user_runtime_exists(
            channel_id=channel_id,
            session_manager=app_module.session_manager,
            workflow_path=app_module.ARGS.workflow_path,
            run_startup=False,
        )
        runtime = await app_module.session_manager.get_session(channel_id)
        store.clear(channel_id)
        assert not runtime.execution_context.awaiting_user

        persist_pending_after_turn(
            app_module.session_manager,
            runtime,
            fastworkflow.TurnOutput(
                turn_key=fastworkflow.mint_turn_key(),
                status=fastworkflow.TurnStatus.AWAITING_USER,
                answer="Which one?",
            ),
        )
        saved = store.exists(channel_id)

        persist_pending_after_turn(
            app_module.session_manager,
            runtime,
            fastworkflow.TurnOutput(
                turn_key=fastworkflow.mint_turn_key(),
                status=fastworkflow.TurnStatus.COMPLETED,
                answer="7",
            ),
        )
        return saved, store.exists(channel_id)

    saved, cleared = asyncio.run(body())

    assert saved, "an AWAITING_USER turn did not persist its suspended state"
    assert not cleared, "a COMPLETED turn did not clear the suspended state"


# ---------------------------------------------------------------------------
# Published schema and MCP tool generation
# ---------------------------------------------------------------------------

def test_the_published_schema_describes_the_turn_output(action_app_module):
    """TurnOutput has to survive OpenAPI generation, not just JSON serialization.

    /initialize declares response_model=InitializeResponse, so changing
    startup_output's type changes the published schema. If TurnOutput could not
    be rendered as a component, /openapi.json — and with it Swagger and every
    generated client — would break.
    """
    schema = action_app_module.app.openapi()
    components = schema["components"]["schemas"]

    assert "TurnOutput" in components, "TurnOutput is missing from the published schema"
    # `success` is computed, so its presence here proves computed fields are published.
    assert TURN_OUTPUT_KEYS == set(components["TurnOutput"]["properties"])

    startup_output = components["InitializeResponse"]["properties"]["startup_output"]
    refs = [
        option.get("$ref")
        for option in startup_output.get("anyOf", [startup_output])
    ]
    assert "#/components/schemas/TurnOutput" in refs, (
        f"startup_output does not reference TurnOutput: {startup_output}"
    )


def test_the_mcp_tools_still_generate(action_app_module):
    """The MCP surface is derived from the OpenAPI schema, so it can break with it.

    isError is deliberately NOT mapped to `not success` — see the TODO in
    mcp_specific.py. What this asserts is that the turn-bearing tools are still
    generated at all.
    """
    from fastapi_mcp import FastApiMCP

    mcp = FastApiMCP(action_app_module.app)
    tool_names = {tool.name for tool in mcp.tools}

    # invoke_agent is the operation_id of /invoke_agent_stream (see mcp_specific).
    assert {"invoke_agent", "invoke_assistant"} <= tool_names, (
        f"turn tools missing from the MCP surface: {sorted(tool_names)}"
    )


# ---------------------------------------------------------------------------
# Conversation persistence across the turn path
# ---------------------------------------------------------------------------

def test_a_turn_still_feeds_the_conversation_endpoints(action_app_module):
    """Returning a TurnOutput must not starve conversation history.

    /new_conversation, /post_feedback and /activate_conversation all work off
    ``ctx.conversation_history``, which the turn path populates as a side effect
    rather than through the return value. A cutover that changed only the return
    type could still have broken them by routing through a different dispatch,
    so this asserts the side effect survived.
    """
    channel_id = _channel("convo")
    app_module = action_app_module

    with TestClient(app_module.app) as client:
        init = client.post("/initialize", json={"channel_id": channel_id})
        assert init.status_code == 200
        headers = {"Authorization": f"Bearer {init.json()['access_token']}"}

        action = client.post(
            "/perform_action",
            headers=headers,
            json={"action": _add_action(), "timeout_seconds": 30},
        )
        assert action.status_code == 200

        runtime = asyncio.run(app_module.session_manager.get_session(channel_id))
        assert runtime is not None
        messages = runtime.execution_context.conversation_history.messages
        assert messages, "the turn recorded no conversation history"

        feedback = client.post(
            "/post_feedback", headers=headers, json={"binary_or_numeric_score": True}
        )
        assert feedback.status_code == 200
        assert messages[-1]["feedback"]["binary_or_numeric_score"] == 1.0

        # The turn completed, so there is nothing suspended to abandon — and
        # cancel_pending must say so rather than claim it cleared something.
        cancelled = client.post("/cancel_pending", headers=headers)
        assert cancelled.status_code == 200
        assert cancelled.json() == {"status": "ok", "cleared": False}

        # Durably recorded too, not just in memory.
        conversations = client.get("/conversations", headers=headers, params={"limit": 5})
        assert conversations.status_code == 200


# ---------------------------------------------------------------------------
# Streaming: the terminal 'output' event
# ---------------------------------------------------------------------------

def _parse_ndjson(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _parse_sse(text: str) -> list[dict]:
    events: list[dict] = []
    current: dict = {}
    for line in text.split("\n"):
        if line.startswith("event: "):
            current["type"] = line[7:]
        elif line.startswith("data: "):
            current["data"] = json.loads(line[6:])
        elif line == "" and current:
            events.append(current)
            current = {}
    if current:
        events.append(current)
    return events


@pytest.mark.parametrize("stream_format", ["ndjson", "sse"])
def test_the_stream_terminal_output_event_is_a_turn_output(agent_app_module, stream_format):
    """The final 'output' event carries a TurnOutput, not a CommandOutput.

    Runs a real agent turn, so it makes real LLM calls (per testing_rules.mdc,
    these are integration tests against real components).
    """
    channel_id = _channel(f"stream_{stream_format}")
    with TestClient(agent_app_module.app) as client:
        init = client.post(
            "/initialize",
            json={"channel_id": channel_id, "stream_format": stream_format},
        )
        assert init.status_code == 200
        headers = {"Authorization": f"Bearer {init.json()['access_token']}"}

        resp = client.post(
            "/invoke_agent_stream",
            headers=headers,
            json={"user_query": "add 2 and 3", "timeout_seconds": 90},
        )
        assert resp.status_code == 200
        events = (
            _parse_ndjson(resp.text)
            if stream_format == "ndjson"
            else _parse_sse(resp.text)
        )

    assert events, "stream produced no events"
    outputs = [e for e in events if e["type"] == "output"]
    errors = [e for e in events if e["type"] == "error"]
    assert outputs, f"stream produced no output event (errors: {errors})"
    assert events[-1]["type"] == "output", "the output event must be terminal"

    # The turn may or may not have succeeded — that is the LLM's business. What
    # this test owns is the SHAPE of the report, whatever the outcome was.
    assert_turn_output_projection(outputs[-1]["data"])
