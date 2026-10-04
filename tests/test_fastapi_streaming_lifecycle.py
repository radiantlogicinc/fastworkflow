"""Integration tests for streaming as a registered turn (Release B, step 7).

`/invoke_agent_stream` used to run an entire turn outside the turn lifecycle: no
`TurnExecution`, a `runtime.lock.locked()` busy check of its own, and a 504 that
abandoned the executor future without awaiting it. Three consequences the design
enumerates — an unrelated query answered as somebody else's clarification, a
context snapshotted and closed while a detached thread still mutates it, and a
runtime evicted between response construction and first body iteration.

It is now admitted through the registry like every other turn, which is what
makes it visible to the 409 guard, to eviction, and to the shutdown drain.

COST (fix-14ac): every turn here used to run the planner, so the whole file
cost real model calls and could not be run in a no-paid-calls environment —
which is how its documented event allowlist went stale without anyone
noticing. The turns that exercise the TRANSPORT now run the session's runtime
in deterministic mode: the same route, registry, CME pipeline, trace queue and
streaming body, answered locally instead of by a planner. Nothing about what
is asserted changed. A case that genuinely needs the planner would be marked
paid; none of the remaining ones do.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import shutil
import sys
import uuid

import pytest
from fastapi.testclient import TestClient

import fastworkflow


@pytest.fixture
def hello_world_workflow_path():
    """The repo's trained test workflow, not the shipped example.

    ``fastworkflow/examples/hello_world`` ships without intent-classifier
    artifacts (no ``___command_info/global/threshold.json``), so a turn that
    actually reaches the NLU pipeline dies there. That never showed while
    every turn went to the planner; running these locally surfaced it
    immediately. ``tests/hello_world_workflow`` is the trained one the rest of
    the suite drives (testing_rules.mdc), and the transport under test does
    not care which workflow answers.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    workflow_path = os.path.join(project_root, "tests", "hello_world_workflow")
    if not os.path.isdir(workflow_path):
        pytest.skip(f"hello_world workflow not found at {workflow_path}")
    return workflow_path


@pytest.fixture
def env_files(tmp_path):
    """Workflow env files written from the shipped template.

    Previously the repo's own ``env/.env`` and ``passwords/.env``, which meant
    the whole file skipped on a machine that has no passwords file — so the
    tests were unrunnable for two independent reasons at once. The keys here
    are placeholders and are never used: no test in this file reaches a model.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    template = os.path.join(
        project_root, "fastworkflow", "examples", "fastworkflow.env"
    )
    if not os.path.isfile(template):
        pytest.skip(f"env template missing at {template}")

    env_file = tmp_path / "fastworkflow.env"
    shutil.copy(template, env_file)
    passwords_file = tmp_path / "fastworkflow.passwords.env"
    passwords_file.write_text(
        "\n".join(
            f"LITELLM_API_KEY_{role}=placeholder-not-used"
            for role in (
                "SYNDATA_GEN",
                "PARAM_EXTRACTION",
                "RESPONSE_GEN",
                "PLANNER",
                "AGENT",
                "CONVERSATION_STORE",
            )
        )
        + "\n"
    )
    return str(env_file), str(passwords_file)


@pytest.fixture
def app_module(hello_world_workflow_path, env_files, tmp_path):
    env_file, passwords_file = env_files
    sys.argv = [
        "pytest",
        "--workflow_path",
        hello_world_workflow_path,
        "--env_file_path",
        env_file,
        "--passwords_file_path",
        passwords_file,
    ]
    import fastworkflow.run_fastapi_mcp.__main__ as main

    importlib.reload(main)
    from tests.fastapi_hermetic import init_fastapi_hermetic_env, restore_fastapi_env

    previous_env = init_fastapi_hermetic_env(
        env_file, passwords_file, tmp_path / "workflow_contexts"
    )
    try:
        yield main
    finally:
        restore_fastapi_env(previous_env)


def _channel(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _initialize(client: TestClient, channel_id: str, stream_format: str = "ndjson"):
    resp = client.post(
        "/initialize",
        json={"channel_id": channel_id, "stream_format": stream_format},
    )
    assert resp.status_code == 200
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _answer_locally(app_module, channel_id: str) -> None:
    """Let this session's runtime answer without the planner.

    It is the real WorkflowExecutionContext either way — intent detection,
    command execution, the trace queue and the turn record all still run. Only
    the routing decision inside it changes, and that decision is the paid one.
    """
    async def configure():
        runtime = await app_module.session_manager.get_session(channel_id)
        runtime.execution_context._run_as_agent = False

    asyncio.run(configure())


def test_a_streaming_turn_completes_and_retires_itself(app_module):
    """The happy path still streams, and the execution ends up terminal."""
    channel_id = _channel("stream")

    with TestClient(app_module.app) as client:
        headers = _initialize(client, channel_id)
        _answer_locally(app_module, channel_id)
        resp = client.post(
            "/invoke_agent_stream",
            headers=headers,
            # A command that needs no parameters: "add 2 and 3" would route to
            # add_two_numbers and hand its arguments to DSPy extraction, which
            # is another model call. The transport does not care which command
            # answers, only that a real one does.
            json={"user_query": "what can i do", "timeout_seconds": 60},
        )
        assert resp.status_code == 200
        events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]

    assert events, "stream produced no events"
    # The interactions arrive before the answer, which is the whole point of
    # streaming them — and the turn really answered. Without this the file
    # passes just as happily on a stream whose only content is a failure,
    # which is how a workflow with no trained classifier went unnoticed.
    assert events[0]["type"] == "trace"
    assert events[-1]["type"] == "output"
    assert events[-1]["data"]["success"] is True
    assert events[-1]["data"]["answer"]
    # 'timeout' joined the documented set when the delivery deadline stopped
    # being reported as a (terminal) 'error': it says the turn passed the
    # deadline and is STILL running, so it can appear here and cannot be the
    # last event. The terminal set is unchanged.
    assert {e["type"] for e in events} <= {"trace", "timeout", "output", "error"}
    assert events[-1]["type"] in ("output", "error")
    assert sum(e["type"] in ("output", "error") for e in events) == 1

    # Registered, ran, and cleared its own active pointer.
    assert not app_module.turn_registry.has_active(channel_id)


def test_a_streaming_turn_is_visible_to_the_registry_while_it_runs(app_module):
    """Ownership, not delivery: the 409 guard and eviction both read this pointer."""
    channel_id = _channel("visible")
    seen = {}

    async def body():
        await app_module.ensure_user_runtime_exists(
            channel_id=channel_id,
            session_manager=app_module.session_manager,
            workflow_path=app_module.ARGS.workflow_path,
            run_startup=False,
        )
        registry = app_module.turn_registry
        started = asyncio.Event()
        release = asyncio.Event()

        async def work():
            started.set()
            await release.wait()
            return fastworkflow.TurnOutput(
                turn_key=fastworkflow.mint_turn_key(),
                status=fastworkflow.TurnStatus.COMPLETED,
            )

        runtime = await app_module.session_manager.get_session(channel_id)
        execn = await registry.start_or_get_active(
            channel_id,
            kind="invoke_agent_stream",
            idempotency_key="stream-visible",
            run_turn=lambda e: asyncio.create_task(
                app_module.run_owned_turn(
                    runtime, registry, e, work, app_module.session_manager
                )
            ),
        )

        await started.wait()
        seen["active_during"] = registry.has_active(channel_id)
        seen["busy_during"] = channel_id in app_module.session_manager.busy_channel_ids()

        release.set()
        await execn.done_event.wait()
        seen["active_after"] = registry.has_active(channel_id)
        seen["error"] = execn.error

    asyncio.run(body())

    assert seen["active_during"], "a running stream was invisible to the registry"
    assert seen["busy_during"], "a running stream looked idle to the shutdown drain"
    assert not seen["active_after"]
    assert seen["error"] is None


def test_a_second_stream_on_a_busy_channel_is_rejected_with_409(app_module):
    """Streaming now shares admission control with every other endpoint."""
    channel_id = _channel("busy")

    async def occupy():
        await app_module.ensure_user_runtime_exists(
            channel_id=channel_id,
            session_manager=app_module.session_manager,
            workflow_path=app_module.ARGS.workflow_path,
            run_startup=False,
        )
        # A live execution nothing will finish, so the channel stays busy.
        await app_module.turn_registry.start_or_get_active(
            channel_id,
            kind="invoke_agent",
            idempotency_key="occupier",
            run_turn=lambda e: asyncio.create_task(asyncio.sleep(30)),
        )

    client = TestClient(app_module.app)
    headers = _initialize(client, channel_id)
    asyncio.run(occupy())

    resp = client.post(
        "/invoke_agent_stream",
        headers=headers,
        json={"user_query": "add 2 and 3", "timeout_seconds": 5},
    )

    assert resp.status_code == 409
    assert "already in progress" in resp.json()["detail"]


def test_closed_admission_refuses_new_turns(app_module):
    """Shutdown closes admission atomically, so nothing registers behind the drain."""
    channel_id = _channel("closed")

    async def body():
        registry = app_module.turn_registry.__class__()
        await registry.close_admission()
        assert registry.admission_closed
        with pytest.raises(app_module.AdmissionClosedError):
            await registry.start_or_get_active(
                channel_id,
                kind="invoke_agent",
                idempotency_key="late",
                run_turn=lambda e: asyncio.create_task(asyncio.sleep(0)),
            )

    asyncio.run(body())


def test_the_delivery_deadline_does_not_abandon_the_executor(app_module):
    """A 504 used to release the lock while the executor thread kept mutating state.

    The deadline now governs delivery only: the client is told, and the turn keeps
    the lock and the registry pointer until the work actually exits.
    """
    channel_id = _channel("deadline")
    observed = {}

    async def body():
        await app_module.ensure_user_runtime_exists(
            channel_id=channel_id,
            session_manager=app_module.session_manager,
            workflow_path=app_module.ARGS.workflow_path,
            run_startup=False,
        )
        runtime = await app_module.session_manager.get_session(channel_id)
        # Local answer, real turn: the deadline behaviour under test belongs to
        # the streaming helper, not to whatever produced the answer.
        runtime.execution_context._run_as_agent = False
        timeouts = []

        async def on_timeout(detail):
            timeouts.append(detail)
            observed["still_locked_at_timeout"] = runtime.lock.locked()

        async def slow_work():
            async with runtime.lock:
                return await app_module.run_process_message_with_trace_stream(
                    runtime,
                    "what can i do",   # parameterless: no extraction call
                    0,  # deadline already passed on the first poll
                    app_module.session_manager,
                    lambda _t: None,
                    on_timeout=on_timeout,
                )

        output = await slow_work()
        observed["timeouts"] = timeouts
        observed["output_returned"] = output is not None
        observed["lock_free_after"] = not runtime.lock.locked()

    asyncio.run(body())

    assert observed["timeouts"], "the client was never told about the deadline"
    assert observed["still_locked_at_timeout"], "ownership was dropped at the deadline"
    assert observed["output_returned"], "the executor result was abandoned"
    assert observed["lock_free_after"]
